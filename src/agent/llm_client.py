# -*- coding: utf-8 -*-
"""
LLM 客户端封装（Phase 3）
================================================================================
【这个模块解决什么问题？】

  裸着调大模型 API 会带来一堆琐碎的、跟业务无关的代码：读 Key、建客户端、
  处理超时、重试、换模型、统计 token……如果这些都写在 Agent 主循环里，
  主循环会被淹没，读代码时根本看不出来「Agent 到底在干什么」。

  所以把这层「基础设施」单独抽出来，对外只暴露一个方法：

      response = client.chat(messages, tools=...)

  它内部替我们做完这些事：

    1. 选供应商   —— 按 LLM_PROVIDER_ORDER 顺序，挑第一个配了 Key 的
    2. 重试       —— 同一供应商内指数退避重试（仅对「可重试」的错误）
    3. 降级       —— 该供应商彻底失败就换下一个（Failover）
    4. 归一化     —— 把各家 SDK 的返回统一成我们自己的 LLMResponse
    5. 记账       —— 累计 token 用量，便于控制成本

【为什么 DeepSeek 和 GLM 能用同一套代码？】

  因为两家的 API 都「兼容 OpenAI 格式」—— 请求体、返回体结构完全一致，
  只是 base_url 和 model 名字不同。所以只需要一个 OpenAI SDK + 两份配置。
  这是 2024 年以后国内大模型 API 的事实标准，可以这样说：
  「我按 OpenAI 兼容协议做抽象，新增一个供应商只需要往配置里加一条，
   不改一行代码」——这就是「面向接口编程」带来的扩展性。

【为什么要做「归一化」而不是直接把 SDK 对象丢给上层？】

  上层（Agent 主循环、Phase 4 前端）不应该知道用的是哪家 SDK。
  如果直接把 openai 的 ChatCompletion 对象一路传下去，
  将来换 SDK、或者要 Mock 测试，都得跟着改一大片代码。
  在边界处转成自己的数据结构，是分层架构的基本纪律。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src import config as cfg
from src.exceptions import LLMError, LLMNotConfiguredError

# 这些 HTTP 状态码代表「请求本身有问题」，重试一万次也不会成功，直接放弃该供应商。
#   400 参数错误 / 401 Key 无效 / 403 无权限 / 404 模型名不存在 / 422 语义错误
# 反过来，429（限流）和 5xx（服务端抖动）是**值得重试**的 —— 这正是重试机制的价值所在。
_NON_RETRYABLE_STATUS: frozenset[int] = frozenset({400, 401, 403, 404, 422})


# ===========================================================================
# 一、数据结构：把 SDK 的返回翻译成我们自己的类型
# ===========================================================================

@dataclass
class ToolCall:
    """一次「工具调用请求」。

    注意 arguments 有两个字段，不是冗余：
      · arguments      —— 解析好的 dict，给工具执行器用
      · arguments_raw  —— 模型原始返回的 JSON 字符串
      为什么要留原始串？因为调下一轮 LLM 时，必须把 assistant 消息里
      的 tool_calls **原样回传**（包括参数串的每一个字符）。
      如果我们自己重新 json.dumps 一遍，某些 API 会因为「格式不一致」而报错。
      所以：解析版用于执行，原始串用于回传，各司其职。
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_raw: str = ""
    parse_error: str | None = None    # 参数 JSON 解析失败时的原因

    @classmethod
    def from_openai(cls, raw: Any) -> "ToolCall":
        """把 OpenAI 格式的 tool_call 对象转成 ToolCall。

        大模型给出的参数是**字符串形式的 JSON**（不是对象），
        而且它偶尔会写坏（少个引号、多个逗号、甚至返回空串）。
        所以这里必须做「防御性解析」：解析失败不抛异常，
        而是把错误记在 parse_error 里，让工具层把它作为「观察结果」喂回模型，
        模型看到错误后通常能自己改对 —— 这是 Agent 自我纠错能力的一部分。
        """
        function = getattr(raw, "function", None)
        name = getattr(function, "name", "") or ""
        arguments_raw = getattr(function, "arguments", "") or ""

        call = cls(
            id=getattr(raw, "id", "") or "",
            name=name,
            arguments_raw=arguments_raw,
        )

        if not arguments_raw.strip():
            # 完全不带参数是合法的（比如 list_metrics 不需要参数）
            return call

        try:
            parsed = json.loads(arguments_raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"参数必须是一个 JSON 对象，实际是 {type(parsed).__name__}")
            call.arguments = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            call.parse_error = str(exc)

        return call


@dataclass
class LLMResponse:
    """一次 LLM 调用的归一化结果。"""

    content: str = ""                              # 模型输出的正文（可能是空的）
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""                        # stop / tool_calls / length
    provider: str = ""                             # 实际是哪个供应商响应的
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # 【Phase 6 · 提示词缓存计量】
    # prompt_tokens 里「命中供应商上下文缓存」的那部分。为什么要单独拎出来？
    # 因为 ReAct 循环每多走一轮，就会把「系统提示词 + 历史消息」整包重发一次，
    # 其中系统提示词（含指标目录）是**每一轮都完全相同**的前缀。
    # DeepSeek / OpenAI 这类供应商会对相同前缀做缓存，命中部分按 1/10 左右计价。
    # 所以「缓存命中率」直接决定了这个 Agent 的真实成本 —— 它是提示词优化
    # 唯一能被量化的收益指标：优化前命中率越高，说明固定前缀越大。
    # 注意它**不是**独立于 prompt_tokens 的新增消耗，而是 prompt_tokens 的子集。
    cached_tokens: int = 0
    elapsed_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_assistant_message(self) -> dict[str, Any]:
        """还原成 OpenAI 格式的 assistant 消息，用于追加进对话历史。

        这是 ReAct 循环的关键一步：把模型「要调用工具」的这一轮意图
        完整记录到 messages 里，下一轮请求才能让模型看到自己上一轮做了什么。
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # 回传原始字符串，而不是重新序列化（见 ToolCall 的注释）
                        "arguments": call.arguments_raw or "{}",
                    },
                }
                for call in self.tool_calls
            ]
        return message


# ===========================================================================
# 二、客户端
# ===========================================================================

class LLMClient:
    """多供应商 LLM 客户端（DeepSeek 主 / 智谱 GLM 备）。

    设计要点是「依赖注入」：provider_order / providers / client_factory
    全部可以从外部传入。这样单元测试就能塞一个假客户端进来，
    在不联网、不花 token 的前提下验证重试与降级逻辑。
    （跟 Phase 2 里 SQLGenerator 允许注入 registry 是同一个思路。）
    """

    def __init__(
        self,
        provider_order: tuple[str, ...] | list[str] | None = None,
        providers: dict[str, dict[str, str]] | None = None,
        client_factory: Callable[[str, dict[str, str], str], Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
    ) -> None:
        self.provider_order: tuple[str, ...] = tuple(provider_order or cfg.LLM_PROVIDER_ORDER)
        self.providers: dict[str, dict[str, str]] = dict(providers or cfg.LLM_PROVIDERS)

        self.temperature = cfg.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = cfg.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        self.timeout = cfg.LLM_TIMEOUT_SECONDS if timeout is None else timeout

        # 默认构造器需要拿到实例上的 timeout，所以在这里包一层闭包；
        # 注入的自定义构造器保持「三参数」签名不变，测试写起来更简单。
        if client_factory is None:
            def factory(key: str, provider: dict[str, str], api_key: str) -> Any:
                return _default_client_factory(key, provider, api_key, self.timeout)

            self.client_factory = factory
        else:
            self.client_factory = client_factory
        self.max_retries = cfg.LLM_MAX_RETRIES if max_retries is None else max_retries
        self.backoff_seconds = (
            cfg.LLM_RETRY_BACKOFF_SECONDS if backoff_seconds is None else backoff_seconds
        )

        # SDK 客户端懒加载并缓存：同一个供应商只建一次客户端（内部有连接池）
        self._clients: dict[str, Any] = {}

        # 累计用量，便于在前端展示「这次对话花了多少 token」
        self.usage: dict[str, int] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }

    # ------------------------------------------------------------------
    # 供应商探测
    # ------------------------------------------------------------------
    def api_key(self, provider_key: str) -> str:
        """读某个供应商的 API Key。

        刻意「每次调用都读环境变量」而不是在 __init__ 里读一次缓存起来：
        环境变量可能在运行期被改（测试 monkeypatch、容器注入），
        缓存会导致读到旧值，这类 bug 很隐蔽。
        """
        provider = self.providers.get(provider_key, {})
        env_name = provider.get("api_key_env", "")
        return os.getenv(env_name, "").strip() if env_name else ""

    def available_providers(self) -> list[str]:
        """按优先级返回「配了 Key」的供应商列表。"""
        return [key for key in self.provider_order if self.api_key(key)]

    @property
    def is_configured(self) -> bool:
        """是否至少有一个可用供应商。前端可以用它决定要不要显示「未配置」提示。"""
        return bool(self.available_providers())

    def _get_client(self, provider_key: str) -> Any:
        """取（或创建）某个供应商的 SDK 客户端。"""
        if provider_key not in self._clients:
            provider = self.providers[provider_key]
            self._clients[provider_key] = self.client_factory(
                provider_key, provider, self.api_key(provider_key)
            )
        return self._clients[provider_key]

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """发一次对话请求，自动重试 + 主备降级。

        参数：
            messages    —— OpenAI 格式的对话历史
            tools       —— Function Calling 的工具清单（JSON Schema）
            tool_choice —— "auto"（默认）/ "none" / "required"

        异常：
            LLMNotConfiguredError —— 一个 Key 都没配
            LLMError              —— 所有可用供应商都失败了
        """
        available = self.available_providers()
        if not available:
            env_names = "、".join(
                self.providers[k].get("api_key_env", k) for k in self.provider_order
            )
            raise LLMNotConfiguredError(
                f"未检测到任何 LLM API Key（{env_names}），无法调用大模型。"
                f"请把 .env.example 复制为 .env 并填入 Key。"
            )

        failures: list[str] = []
        for provider_key in available:
            name = self.providers[provider_key].get("display_name", provider_key)
            try:
                return self._chat_with_provider(provider_key, messages, tools, tool_choice)
            except Exception as exc:  # noqa: BLE001 - 任何异常都应触发降级，不能中断整个对话
                failures.append(f"{name}：{exc}")

        raise LLMError(
            "所有 LLM 供应商调用均失败（已按顺序尝试 "
            + " → ".join(self.providers[k].get("display_name", k) for k in available)
            + "）：\n- "
            + "\n- ".join(failures)
        )

    # ------------------------------------------------------------------
    # 单个供应商：重试逻辑
    # ------------------------------------------------------------------
    def _chat_with_provider(
        self,
        provider_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
    ) -> LLMResponse:
        provider = self.providers[provider_key]
        client = self._get_client(provider_key)

        kwargs: dict[str, Any] = {
            "model": provider["model"],
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        last_error: Exception | None = None

        # range(max_retries + 1)：首次 + N 次重试
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                completion = client.chat.completions.create(**kwargs)
                return self._to_response(
                    completion,
                    provider_key=provider_key,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as exc:  # noqa: BLE001 - 交给下面判断是否值得重试
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable(exc):
                    break
                # 指数退避：1.5s → 3s → ...
                time.sleep(self.backoff_seconds * (2**attempt))

        # 循环至少执行过一次（max_retries >= 0），所以 last_error 一定有值
        if last_error is None:  # pragma: no cover - 纯防御分支
            raise LLMError(f"调用 {provider_key} 失败，但没有捕获到具体异常")
        raise last_error

    # ------------------------------------------------------------------
    # 归一化
    # ------------------------------------------------------------------
    def _to_response(
        self,
        completion: Any,
        provider_key: str,
        elapsed_ms: float,
    ) -> LLMResponse:
        """把 OpenAI 格式的返回转成 LLMResponse，并累计 token 用量。"""
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMError("大模型返回了空的 choices，无法解析")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMError("大模型返回的 choice 中没有 message 字段")

        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (
            prompt_tokens + completion_tokens
        )
        cached_tokens = _read_cached_tokens(usage)

        response = LLMResponse(
            # 有些模型在调用工具时 content 为 None，统一成空串，避免上层到处判 None
            content=getattr(message, "content", None) or "",
            tool_calls=[
                ToolCall.from_openai(raw)
                for raw in (getattr(message, "tool_calls", None) or [])
            ],
            finish_reason=getattr(choices[0], "finish_reason", "") or "",
            provider=provider_key,
            model=self.providers[provider_key]["model"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            elapsed_ms=elapsed_ms,
        )

        self.usage["llm_calls"] += 1
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += total_tokens
        self.usage["cached_tokens"] += cached_tokens

        return response


# ===========================================================================
# 三、工具函数
# ===========================================================================

def _read_cached_tokens(usage: Any) -> int:
    """从 usage 里读「命中提示词缓存」的 token 数。

    【为什么需要兼容两种字段名？】

      各家虽然都说「兼容 OpenAI 协议」，但缓存字段是各家自己加的扩展，
      命名并不统一，实测下来至少有两种形态：

        · DeepSeek  ：usage.prompt_cache_hit_tokens        （扁平字段）
        · OpenAI 系 ：usage.prompt_tokens_details.cached_tokens （嵌套对象）
          （智谱 GLM 走的就是 OpenAI 这一套结构）

      所以这里按「先扁平、后嵌套」的顺序探测，两个都没有就返回 0。
      返回 0 的语义是「未命中 / 该供应商未上报」，而不是「解析失败」——
      对计量来说两者处理方式一致，没必要区分，也就没必要抛异常。

    可以这样说明：**这就是「兼容协议」的真实边界** ——
    主干字段一致，扩展字段各家为政。做多供应商适配时，
    真正的工作量往往不在主流程，而在这些边角字段的兼容上。
    """
    if usage is None:
        return 0

    flat = getattr(usage, "prompt_cache_hit_tokens", None)
    if flat is not None:
        return int(flat or 0)

    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        nested = getattr(details, "cached_tokens", None)
        if nested is not None:
            return int(nested or 0)

    return 0


def _is_retryable(exc: Exception) -> bool:
    """判断一个异常值不值得重试。

    核心判断依据是 HTTP 状态码（OpenAI SDK 的异常对象带 status_code 属性）：
      · 4xx 里「请求本身有问题」的（Key 错、参数错、模型名错）→ 重试无意义，直接换供应商
      · 429 限流、5xx 服务端故障、网络超时/连接错误 → 值得重试

    这个区分很重要：如果无脑重试 401，会白白浪费 3 次等待时间，
    最后还是得降级，用户体验被拖慢好几秒。
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status not in _NON_RETRYABLE_STATUS
    # 没有状态码的（连接超时、DNS 失败等）属于网络类问题，重试通常有效
    return True


def _default_client_factory(
    provider_key: str, provider: dict[str, str], api_key: str, timeout: float
) -> Any:
    """默认的客户端构造器：用 OpenAI SDK 连各家兼容端点。

    openai 的 import 放在函数内部（懒加载），而不是模块顶层，原因有两个：
      1. 不配 Key、只用离线工具的场景根本不需要装 openai；
      2. 单元测试注入假客户端时，也完全不依赖这个库。
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - 只在缺依赖时触发
        raise LLMError(
            "未安装 openai 库，请执行：pip install openai python-dotenv"
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url=provider["base_url"],
        timeout=timeout,
    )