# -*- coding: utf-8 -*-
"""
LLM 客户端单元测试
================================================================================
【这个文件为什么完全不联网？】

  因为它用「依赖注入」塞了一个假客户端进来：
  LLMClient 允许传入 client_factory，我们传进去一个返回假对象的函数。
  这样就能在毫秒级、零成本的前提下验证最难手动复现的逻辑：

      · 主供应商失败后有没有真的降级到备用？
      · 401 这种「重试也不会好」的错误，有没有被聪明地跳过重试？
      · 模型返回的坏 JSON 参数，有没有被安全地兜住而不是崩掉？

  这三件事如果真去连 API 测，要花真金白银，而且没法稳定复现（得先想办法把
  主供应商搞挂）。**可测性本身就是一种设计质量** —— 一个模块好不好测，
  往往和它的依赖是否可注入直接相关。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.llm_client import LLMClient, LLMResponse, ToolCall
from src.exceptions import LLMError, LLMNotConfiguredError

# ---------------------------------------------------------------------------
# 测试用的供应商配置（把 api_key_env 指向测试专用的环境变量名，
# 避免误读开发者本机 .env 里的真实 Key —— 否则测试结果会随本机环境变化）
# ---------------------------------------------------------------------------
TEST_PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "display_name": "DeepSeek",
        "api_key_env": "TEST_DEEPSEEK_KEY",
        "base_url": "https://example.invalid/v1",
        "model": "deepseek-chat",
    },
    "glm": {
        "display_name": "智谱 GLM",
        "api_key_env": "TEST_GLM_KEY",
        "base_url": "https://example.invalid/v4",
        "model": "glm-4-flash",
    },
}


# ===========================================================================
# 一、测试替身（Fake）：一个「按剧本演出」的假 SDK
# ===========================================================================

class _ApiError(Exception):
    """模拟 SDK 抛出的带 HTTP 状态码的异常。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeCompletions:
    """假的 chat.completions：按剧本依次吐出响应，剧本用完就报错。

    「剧本用完就报错」这一点很关键：如果代码发生了**非预期的重试**，
    它会立刻炸出来，而不是悄悄蒙混过关。
    """

    def __init__(self, script: list) -> None:
        self._script = script
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("假客户端剧本已用完 —— 说明发生了非预期的额外调用（比如多余的重试）")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, script: list) -> None:
        self.completions = _FakeCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeFactory:
    """假的 client_factory：记录每个供应商被调用了几次、各自收到了什么请求。"""

    def __init__(self, scripts: dict[str, list]) -> None:
        self._scripts = scripts
        self.clients: dict[str, _FakeClient] = {}

    def __call__(self, provider_key: str, provider_cfg: dict, api_key: str):
        client = _FakeClient(list(self._scripts.get(provider_key, [])))
        self.clients[provider_key] = client
        return client

    def call_count(self, provider_key: str) -> int:
        client = self.clients.get(provider_key)
        return len(client.completions.calls) if client else 0


def make_completion(
    content: str = "",
    tool_calls: list[tuple[str, str, str]] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cached_tokens: int | None = None,
    cached_style: str = "flat",
):
    """构造一个「长得像 OpenAI SDK 返回值」的对象。

    刻意用 SimpleNamespace 而不是真的 ChatCompletion：
    我们只依赖属性访问（.choices / .message / .usage），
    假对象只要属性对得上就够了，这也反过来说明我们的解析代码没有过度依赖 SDK 细节。

    cached_tokens / cached_style 用于覆盖「各家缓存字段命名不一致」的兼容逻辑：
      · "flat"   —— DeepSeek 风格：usage.prompt_cache_hit_tokens
      · "nested" —— OpenAI/GLM 风格：usage.prompt_tokens_details.cached_tokens
      · None     —— 完全不带缓存字段（模拟未上报的供应商）
    """
    calls = None
    if tool_calls:
        calls = [
            SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=arguments),
            )
            for call_id, name, arguments in tool_calls
        ]
    message = SimpleNamespace(content=content, tool_calls=calls)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")
    usage_kwargs: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        if cached_style == "flat":
            usage_kwargs["prompt_cache_hit_tokens"] = cached_tokens
        else:
            usage_kwargs["prompt_tokens_details"] = SimpleNamespace(
                cached_tokens=cached_tokens
            )
    usage = SimpleNamespace(**usage_kwargs)
    return SimpleNamespace(choices=[choice], usage=usage)


# ===========================================================================
# 二、夹具
# ===========================================================================

@pytest.fixture
def env_keys(monkeypatch):
    """设置/清空测试用的 API Key。

    注意每次都显式 setenv 成空串（而不是 delenv）：
    getenv 读不到和读到空串在我们的实现里都算「没配」，但显式置空更直观。
    """

    def _set(deepseek: str = "", glm: str = "") -> None:
        monkeypatch.setenv("TEST_DEEPSEEK_KEY", deepseek)
        monkeypatch.setenv("TEST_GLM_KEY", glm)

    return _set


def build_client(scripts: dict[str, list] | None = None, **kwargs) -> tuple[LLMClient, _FakeFactory]:
    """造一个注入了假 SDK 的 LLMClient；backoff 设为 0 让测试不用真的等。"""
    factory = _FakeFactory(scripts or {})
    kwargs.setdefault("backoff_seconds", 0.0)
    client = LLMClient(
        provider_order=("deepseek", "glm"),
        providers=TEST_PROVIDERS,
        client_factory=factory,
        **kwargs,
    )
    return client, factory


# ===========================================================================
# 三、供应商探测
# ===========================================================================

def test_raises_when_no_api_key(env_keys):
    """两个 Key 都没配时，必须抛 LLMNotConfiguredError（而不是模糊的 LLMError）。

    区分这两种异常的实际价值：前端的提示语完全不同 ——
    「请去配 Key」vs「服务暂时不可用，请稍后重试」。
    """
    env_keys()
    client, _ = build_client()

    assert client.is_configured is False
    assert client.available_providers() == []
    with pytest.raises(LLMNotConfiguredError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])
    assert ".env" in str(excinfo.value)


def test_available_providers_follows_priority_order(env_keys):
    """两个 Key 都有时，探测结果必须按配置的优先级排序（deepseek 在前）。"""
    env_keys(deepseek="sk-primary", glm="glm-backup")
    client, _ = build_client()
    assert client.available_providers() == ["deepseek", "glm"]


def test_only_backup_key_still_works(env_keys):
    """只配了备用 Key 时，应该直接用备用，而不是报错。"""
    env_keys(glm="glm-backup")
    client, _ = build_client({"glm": [make_completion(content="glm 回答")]})

    response = client.chat([{"role": "user", "content": "hi"}])
    assert response.content == "glm 回答"
    assert response.provider == "glm"


# ===========================================================================
# 四、正常响应解析
# ===========================================================================

def test_content_and_usage_are_parsed(env_keys):
    env_keys(deepseek="sk-primary")
    client, _ = build_client(
        {"deepseek": [make_completion(content="最近 7 天日活 512", prompt_tokens=120, completion_tokens=30)]}
    )

    response = client.chat([{"role": "user", "content": "日活多少"}])

    assert response.content == "最近 7 天日活 512"
    assert response.has_tool_calls is False
    assert response.provider == "deepseek"
    assert response.model == "deepseek-chat"
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (120, 30, 150)
    assert response.elapsed_ms >= 0


def test_usage_accumulates_in_client(env_keys):
    """client.usage 是进程级累计口径，多次调用要累加。"""
    env_keys(deepseek="sk-primary")
    client, _ = build_client(
        {
            "deepseek": [
                make_completion(content="a", prompt_tokens=10, completion_tokens=5),
                make_completion(content="b", prompt_tokens=20, completion_tokens=5),
            ]
        }
    )

    client.chat([{"role": "user", "content": "1"}])
    client.chat([{"role": "user", "content": "2"}])

    assert client.usage == {
        "llm_calls": 2,
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
        "cached_tokens": 0,
    }


# ---------------------------------------------------------------------------
# 提示词缓存计量（Phase 6）
# ---------------------------------------------------------------------------

def test_cached_tokens_flat_style(env_keys):
    """DeepSeek 风格：usage.prompt_cache_hit_tokens（扁平字段）。"""
    env_keys(deepseek="sk-primary")
    client, _ = build_client(
        {"deepseek": [make_completion(content="ok", prompt_tokens=1000, cached_tokens=800)]}
    )

    response = client.chat([{"role": "user", "content": "hi"}])

    assert response.cached_tokens == 800
    assert client.usage["cached_tokens"] == 800
    # 关键约束：cached_tokens 是 prompt_tokens 的**子集**，不能叠加进总量
    assert response.prompt_tokens == 1000
    assert response.total_tokens == 1020


def test_cached_tokens_nested_style(env_keys):
    """OpenAI / 智谱 GLM 风格：usage.prompt_tokens_details.cached_tokens（嵌套字段）。"""
    env_keys(glm="sk-backup")
    client, _ = build_client(
        {
            "glm": [
                make_completion(
                    content="ok",
                    prompt_tokens=500,
                    cached_tokens=320,
                    cached_style="nested",
                )
            ]
        }
    )

    response = client.chat([{"role": "user", "content": "hi"}])

    assert response.cached_tokens == 320
    assert client.usage["cached_tokens"] == 320


def test_cached_tokens_absent_defaults_to_zero(env_keys):
    """供应商完全不上报缓存字段时，应当安全地落到 0，而不是抛异常。"""
    env_keys(deepseek="sk-primary")
    client, _ = build_client({"deepseek": [make_completion(content="ok")]})

    response = client.chat([{"role": "user", "content": "hi"}])

    assert response.cached_tokens == 0
    assert client.usage["cached_tokens"] == 0


def test_tools_are_forwarded_with_auto_choice(env_keys):
    """传了 tools 就必须一起把 tool_choice 传下去，否则部分供应商会忽略 tools。"""
    env_keys(deepseek="sk-primary")
    client, factory = build_client({"deepseek": [make_completion(content="ok")]})

    tools = [{"type": "function", "function": {"name": "list_metrics"}}]
    client.chat([{"role": "user", "content": "hi"}], tools=tools)

    sent = factory.clients["deepseek"].completions.calls[0]
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"


def test_no_tools_key_when_tools_is_none(env_keys):
    """没传工具时不应该硬塞 tools 字段（有些供应商会因此报参数错误）。"""
    env_keys(deepseek="sk-primary")
    client, factory = build_client({"deepseek": [make_completion(content="ok")]})

    client.chat([{"role": "user", "content": "hi"}])
    assert "tools" not in factory.clients["deepseek"].completions.calls[0]


# ===========================================================================
# 五、工具调用解析（最容易踩坑的地方）
# ===========================================================================

def test_tool_call_arguments_are_parsed_into_dict(env_keys):
    env_keys(deepseek="sk-primary")
    client, _ = build_client(
        {
            "deepseek": [
                make_completion(
                    tool_calls=[("call_1", "query_metric", '{"metric_id": "dau", "params": {"start_date": "2026-09-11"}}')]
                )
            ]
        }
    )

    response = client.chat([{"role": "user", "content": "hi"}])
    assert response.has_tool_calls

    call = response.tool_calls[0]
    assert call.name == "query_metric"
    assert call.arguments == {
        "metric_id": "dau",
        "params": {"start_date": "2026-09-11"},
    }
    assert call.parse_error is None


def test_empty_arguments_is_treated_as_no_params(env_keys):
    """list_metrics 这类无参工具，模型可能返回空串，必须当成「没有参数」而不是错误。"""
    env_keys(deepseek="sk-primary")
    client, _ = build_client({"deepseek": [make_completion(tool_calls=[("call_1", "list_metrics", "")])]})

    call = client.chat([{"role": "user", "content": "hi"}]).tool_calls[0]
    assert call.arguments == {}
    assert call.parse_error is None


def test_malformed_arguments_do_not_crash(env_keys):
    """模型把参数 JSON 写坏是常态。

    正确的处理是「记下错误但不抛异常」—— 让 Agent 把错误当成观察结果喂回模型，
    模型下一轮通常能自己改对。如果这里直接抛异常，整轮对话就废了。
    """
    env_keys(deepseek="sk-primary")
    client, _ = build_client({"deepseek": [make_completion(tool_calls=[("call_1", "query_metric", "{metric_id: dau}")])]})

    call = client.chat([{"role": "user", "content": "hi"}]).tool_calls[0]
    assert call.arguments == {}
    assert call.parse_error is not None


def test_arguments_that_are_not_an_object_are_rejected(env_keys):
    """参数必须是 JSON 对象；如果模型返回了数组或字符串，同样按解析失败处理。"""
    env_keys(deepseek="sk-primary")
    client, _ = build_client({"deepseek": [make_completion(tool_calls=[("call_1", "query_metric", "[1, 2]")])]})

    call = client.chat([{"role": "user", "content": "hi"}]).tool_calls[0]
    assert call.arguments == {}
    assert "JSON 对象" in (call.parse_error or "")


def test_assistant_message_preserves_raw_arguments(env_keys):
    """回传给模型的历史消息里，arguments 必须是**原始字符串**。

    为什么不能重新 json.dumps：某些 API 对「工具调用参数必须与上一轮完全一致」
    有校验，重新序列化（键顺序、空格不同）会被判定为不一致而报错。
    """
    env_keys(deepseek="sk-primary")
    raw = '{ "metric_id" : "dau" }'      # 故意带多余空格，验证是否原样保留
    client, _ = build_client({"deepseek": [make_completion(tool_calls=[("call_9", "query_metric", raw)])]})

    message = client.chat([{"role": "user", "content": "hi"}]).to_assistant_message()

    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == "call_9"
    assert message["tool_calls"][0]["function"]["arguments"] == raw


def test_assistant_message_without_tool_calls_has_no_tool_calls_key(env_keys):
    env_keys(deepseek="sk-primary")
    client, _ = build_client({"deepseek": [make_completion(content="plain")]})

    message = client.chat([{"role": "user", "content": "hi"}]).to_assistant_message()
    assert "tool_calls" not in message
    assert message["content"] == "plain"


def test_none_content_becomes_empty_string(env_keys):
    """模型调用工具时 content 常常是 None，必须归一成空串，否则上层到处要判 None。"""
    env_keys(deepseek="sk-primary")
    client, _ = build_client({"deepseek": [make_completion(content=None, tool_calls=[("c1", "list_metrics", "")])]})

    response = client.chat([{"role": "user", "content": "hi"}])
    assert response.content == ""


# ===========================================================================
# 六、降级（Failover）与重试（Retry）
# ===========================================================================

def test_falls_back_to_backup_provider(env_keys):
    """主供应商挂了，必须自动切到备用 —— 这是本项目可用性的关键一环。"""
    env_keys(deepseek="sk-primary", glm="glm-backup")
    client, factory = build_client(
        {
            "deepseek": [_ApiError("503 Service Unavailable", status_code=503)] * 3,
            "glm": [make_completion(content="来自 GLM 的回答")],
        },
        max_retries=2,
    )

    response = client.chat([{"role": "user", "content": "hi"}])

    assert response.content == "来自 GLM 的回答"
    assert response.provider == "glm"
    assert factory.call_count("deepseek") == 3   # 首次 + 2 次重试
    assert factory.call_count("glm") == 1


def test_all_providers_failing_raises_llm_error(env_keys):
    """全都挂了才抛 LLMError，并且错误信息里要能看出「试过哪些、各自为什么失败」。"""
    env_keys(deepseek="sk-primary", glm="glm-backup")
    client, _ = build_client(
        {
            "deepseek": [_ApiError("boom-1")] * 3,
            "glm": [_ApiError("boom-2")] * 3,
        },
        max_retries=2,
    )

    with pytest.raises(LLMError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])

    message = str(excinfo.value)
    assert "DeepSeek" in message and "智谱 GLM" in message
    assert "boom-1" in message and "boom-2" in message


def test_retry_then_success(env_keys):
    """瞬时抖动（网络超时）应该被重试救回来，不该直接降级。"""
    env_keys(deepseek="sk-primary")
    client, factory = build_client(
        {
            "deepseek": [
                _ApiError("Connection timed out"),      # 第一次失败
                make_completion(content="重试成功"),     # 第二次成功
            ]
        },
        max_retries=2,
    )

    response = client.chat([{"role": "user", "content": "hi"}])

    assert response.content == "重试成功"
    assert factory.call_count("deepseek") == 2


def test_auth_error_is_not_retried(env_keys):
    """401（Key 无效）重试再多次也没用。

    这里断言 deepseek 只被调用了 1 次 —— 如果实现里无脑重试，
    假客户端剧本里只有 1 条，第 2 次调用就会抛 AssertionError 把问题暴露出来。
    这个区分能实实在在地省下用户几秒的等待时间。
    """
    env_keys(deepseek="bad-key", glm="glm-backup")
    client, factory = build_client(
        {
            "deepseek": [_ApiError("Invalid API key", status_code=401)],
            "glm": [make_completion(content="备用顶上")],
        },
        max_retries=3,
    )

    response = client.chat([{"role": "user", "content": "hi"}])

    assert response.provider == "glm"
    assert factory.call_count("deepseek") == 1


def test_rate_limit_is_retried(env_keys):
    """429 限流属于「等一会儿再来就好」，必须重试。"""
    env_keys(deepseek="sk-primary")
    client, factory = build_client(
        {
            "deepseek": [
                _ApiError("Rate limit exceeded", status_code=429),
                make_completion(content="限流后成功"),
            ]
        },
        max_retries=1,
    )

    response = client.chat([{"role": "user", "content": "hi"}])
    assert response.content == "限流后成功"
    assert factory.call_count("deepseek") == 2


def test_empty_choices_triggers_fallback(env_keys):
    """供应商返回了结构不对的响应（没有 choices）时，也应该降级，而不是让上层炸掉。"""
    env_keys(deepseek="sk-primary", glm="glm-backup")
    client, _ = build_client(
        {
            "deepseek": [SimpleNamespace(choices=[], usage=None)],
            "glm": [make_completion(content="备用回答")],
        },
        max_retries=0,
    )

    assert client.chat([{"role": "user", "content": "hi"}]).content == "备用回答"


# ===========================================================================
# 七、数据结构本身的小行为
# ===========================================================================

def test_tool_call_defaults():
    call = ToolCall(id="x", name="list_metrics")
    assert call.arguments == {}
    assert call.arguments_raw == ""
    assert call.parse_error is None


def test_llm_response_has_tool_calls_property():
    assert LLMResponse(content="a").has_tool_calls is False
    assert LLMResponse(content="a", tool_calls=[ToolCall(id="1", name="t")]).has_tool_calls