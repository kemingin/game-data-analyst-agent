# -*- coding: utf-8 -*-
"""
Agent 主循环（Phase 3）—— 手搓 ReAct
================================================================================
【ReAct 是什么？】

  ReAct = Reasoning + Acting。核心思想一句话：
  **让模型「想一步、做一步、看结果、再想下一步」，而不是一次性把答案憋出来。**

  我们的循环长这样：

      messages = [系统提示词, 用户问题]
      for 轮次 in range(最大轮数):
          response = LLM.chat(messages, tools)          # ① 想（Reasoning）
          if response 没有工具调用:
              return response.content                   # ④ 收敛：给出最终回答
          for call in response.tool_calls:              # ② 做（Acting）
              result = tools.execute(call)              #    执行工具
              messages.append(工具结果)                  # ③ 看（Observation）

  就这么一个 for 循环，一共约 40 行。这就是 ReAct 的全部。

【为什么手搓而不用 LangChain / LlamaIndex？】（常被追问，三个理由）

  1. 架构约束必须握在自己手里。
     本项目的硬约束是「LLM 绝不能写 SQL」。而 LangChain 的 SQLDatabaseChain
     恰恰是让模型直接产 SQL 的 —— 用它等于把最核心的安全设计交给框架，
     还得再拆掉。我们的工具层是自己写的，模型能做的事只有「点菜」（选 metric_id），
     做菜（生成/校验/执行 SQL）全在程序里，安全边界清清楚楚。

  2. 复杂度不划算。
     ReAct 本体只有 40 行。引入框架要多掌握 Executor / Agent / Tool / Memory /
     Callback 一整套抽象概念，出问题时得先看懂框架源码才能定位，
     对个人项目是明显的负收益。

  3. 可控性。
     迭代上限、工具异常如何转成观察结果、每一步的 trace 怎么给前端展示 ——
     这些在自研循环里都是显式代码，一眼可见；在框架里要靠回调挂钩子，
     反而更难讲清楚，也更难调试。

  （当然，如果团队已经有 LangChain 技术栈、需要大量现成集成，
   用框架是更务实的选择。技术选型从来是看场景的，能说出「什么时候该用」
   比单纯说「我不用框架」更有说服力。）

【这一层最容易被忽略但最重要的两个设计】

  · 迭代上限（AGENT_MAX_ITERATIONS）
    模型有可能陷入「反复调同一个工具」的死循环。没有上限 = 无限的 API 账单。
    凡是循环 + 外部 API，都必须有熔断。

  · 异常转观察结果
    工具失败时不是抛异常中断，而是把错误文本当作「观察结果」喂回模型。
    模型看到「参数 day_n 不在允许范围内」后，通常能自己改对并重试 ——
    这就是 Agent 的自我纠错能力。关键不在于不出错，而在于错误是否
    以模型能理解的形式反馈回去。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from src import config as cfg
from src.agent.llm_client import LLMClient
from src.agent.prompts import build_system_prompt
from src.agent.tools import ToolExecutor
from src.exceptions import GameAgentError, LLMNotConfiguredError
from src.metrics.registry import MetricRegistry, get_registry

# 出问题时给用户的兜底文案。要「说清楚发生了什么 + 下一步怎么办」，
# 而不是一句冷冰冰的「系统错误」——这是产品体验的一部分。
_NOT_CONFIGURED_HINT = (
    "抱歉，我目前无法工作：还没有配置大模型的 API Key。\n"
    "请把项目根目录的 `.env.example` 复制为 `.env`，并填入 DEEPSEEK_API_KEY "
    "（或 GLM_API_KEY），然后重新启动。"
)

# ===========================================================================
# 【Phase 6 · 第五层防线】答案来源校验
# ===========================================================================
#
# 【为什么需要这一层？】
#
#   项目原本有四层防线（模板受控 → 语法校验 → 只读连接 → 参数绑定），
#   但四层全部作用在 **SQL 侧**：它们保证「执行的 SQL 是安全的」。
#   它们管不了另一件事：**模型压根没去查，直接凭记忆编了个数字。**
#
#   这不是假设。Phase 6 做真机降级验证时实测到：把主供应商摘掉、切到
#   GLM-4-Flash 后，问「最近 7 天的日活是多少」，模型一次工具都不调，
#   直接答「最近 7 天的日活是 100,000 人」—— 而且同一问题两次复现。
#   「禁止编造」当时只是提示词里的一句软约束，程序层没有任何机制能拦住它。
#
# 【这一层怎么工作？】
#
#   触发条件必须同时满足两条（缺一不可）：
#     ① 本轮对话中**没有任何一次成功的 query_metric**；
#     ② 最终回答里出现了「数据型数字」。
#   命中则判定为「疑似编造」，向上下文追加一条提醒、强制模型重新取数；
#   补救仍失败就不把那些数字给用户看（宁可说查不到，也不能编）。
#
# 【为什么要绕开「日期」和「版本号」？】
#
#   这是这一层最容易写错的地方。拒答场景的回答天然含数字：
#   「我的数据只覆盖 2026-06-20 ~ 2026-09-17（共 90 天）」——
#   如果按「含数字」一刀切，这条正确的拒答会被误判成编造，
#   反而把 100% 的超范围拒答率打坏。
#   所以先把日期、版本号从文本里剔掉，只看剩下的「数据值」形态：
#   百分比、千分位数字、3 位以上连续数字。
#   （「7 天」「30 天」这类窗口表述只有 1~2 位，天然不会误伤。）
#
# 【这一点怎么讲清楚？】
#   「软约束 + 硬校验」是 LLM 应用里反复出现的一对。提示词负责让模型
#   大概率做对，程序校验负责在它做错时兜住。只靠提示词，换一个弱一点的
#   模型就会失守；只靠校验，会频繁打断正常的推理。两者缺一不可 ——
#   这也是我做多供应商降级时才发现的问题：降级链路通了，
#   但备用模型的指令遵循能力下降，把提示词的软约束击穿了。」
_DATE_LIKE = re.compile(
    r"\d{4}\s*-\s*\d{1,2}(\s*-\s*\d{1,2})?"            # 2026-06-20 / 2026-06
    r"|\d{4}\s*年\s*(\d{1,2}\s*月\s*)?(\d{1,2}\s*日)?"  # 2026 年 6 月 20 日
)
_VERSION_LIKE = re.compile(r"[vV]\d+(\.\d+)+|\d+(\.\d+){2,}")  # v2.0.0 / 2.0.0
# ↑ 这里必须要求「带 v 前缀」或「至少两个点」，不能写成 [vV]?\d+(\.\d+)+。
#   后者会把 45.00 这种**小数**也当成版本号剔掉，而小数正是最需要抓的数据值形态
#   （留存率 42.66%、付费率 3.85% 全是小数）—— 那等于把防线的主要目标放走了。
# 剔除日期与版本号之后，这些形态才算「数据值」
_DATA_NUMBER = re.compile(r"\d+(\.\d+)?\s*%|\d{1,3}(,\d{3})+|\d{3,}")

_GUARD_REMINDER = (
    "【系统校验未通过】你刚才的回答里出现了具体数值，但本轮对话中"
    "没有任何一次成功的 query_metric 取数，这些数字无法溯源到数据库。\n"
    "请立即调用 get_metric_detail 确认口径，再调用 query_metric 取真实数据，"
    "然后基于工具返回的数字重新作答。\n"
    "如果你确实查不到（比如问题超出数据窗口），就如实说明查不到，"
    "不要给出任何未经工具返回的具体数值。"
)

_GUARD_BLOCKED_ANSWER = (
    "抱歉，我没能取到能支撑这个结论的真实数据，所以不能给你具体数字。\n"
    "请把问题问得更具体一些（例如「最近 7 天的日活是多少」），或稍后重试。"
)


def _has_ungrounded_number(text: str) -> bool:
    """判断一段回答里是否存在「疑似编造的数据值」。

    实现就是上面说的三步：剔日期 → 剔版本号 → 找数据值形态。
    故意写得保守（宁可漏判也不错判）：漏判只是少拦一次，错判却会
    把「正确的拒答」变成「拦截」，破坏产品的诚实性。
    """
    if not text:
        return False
    cleaned = _VERSION_LIKE.sub("", _DATE_LIKE.sub("", text))
    return bool(_DATA_NUMBER.search(cleaned))


@dataclass
class AgentStep:
    """Agent 执行过程中的一步，用于「过程透明」（前端展示 + 调试 + 评测）。

    为什么要把过程记录下来？
      1. 用户想知道「这个数字是怎么来的」——可解释性是数据产品的基本要求；
      2. 出问题时能一眼看出是哪一步走偏了，不用靠猜；
      3. Phase 5 的评测体系需要统计「平均调用了几次工具」「工具失败率」这类指标。

    result 存的是工具的**结构化返回**（columns / rows / rendered_sql）。
    为什么不能只留 observation 文本？
      因为前端要画图，而图要吃的是结构化的 rows；
      让前端去反解析 Markdown 表格是极脆的做法（列名带空格、NULL 写法一变就崩）。
      一句话：文本给人（和模型）看，结构化数据给程序用，两条路各走各的。
    """

    iteration: int                      # 第几轮循环（从 1 开始）
    kind: str                           # tool_call / final_answer
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: str = ""               # 工具返回给模型的文本
    result: dict[str, Any] = field(default_factory=dict)  # 工具的结构化返回（前端画图用）
    ok: bool = True
    content: str = ""                   # 模型这一轮的正文
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "kind": self.kind,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "observation": self.observation,
            "result": self.result,
            "ok": self.ok,
            "content": self.content,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class AgentAnswer:
    """一次问答的完整结果。

    answer 是给用户看的，steps 是给「过程透明」用的，两者分开，
    前端既能只显示答案，也能展开看完整推理链路。
    """

    question: str
    answer: str
    ok: bool = True
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    provider: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "ok": self.ok,
            "steps": [s.to_dict() for s in self.steps],
            "iterations": self.iterations,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
        }


class GameDataAgent:
    """游戏数据智能分析师 Agent。

    典型用法：

        agent = GameDataAgent()
        result = agent.ask("最近的次日留存怎么样？")
        print(result.answer)
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        tools: ToolExecutor | None = None,
        registry: MetricRegistry | None = None,
        max_iterations: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.registry: MetricRegistry = registry or get_registry()
        # llm_client 声明成 Any 而不是 LLMClient：只要实现了 chat() 就能替换进来，
        # 测试时塞一个脚本化的假客户端即可，不必真的联网。
        self.llm_client: Any = llm_client if llm_client is not None else LLMClient()
        self.tools: ToolExecutor = tools or ToolExecutor(registry=self.registry)
        self.max_iterations: int = (
            cfg.AGENT_MAX_ITERATIONS if max_iterations is None else max_iterations
        )
        # 第五层防线（答案来源校验）的补救次数上限，同样允许注入以便测试
        self.guard_retries: int = getattr(cfg, "AGENT_GUARD_RETRIES", 1)
        self.system_prompt: str = system_prompt or build_system_prompt(self.registry)

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentAnswer:
        """回答一个业务问题。

        history 是之前轮次的对话（OpenAI 格式的 user/assistant 消息），
        传入它可以支持多轮追问，比如用户先问「最近留存怎么样」，
        再追问「那版本上线后呢」——「那」字的指代需要上下文才能理解。

        这个方法**永远不抛异常**：所有失败都转成 AgentAnswer(ok=False)。
        原因：它是前端的唯一入口，一次失败不应该让整个页面崩掉。
        """
        question = (question or "").strip()
        if not question:
            return AgentAnswer(
                question="", answer="请输入你想了解的问题。", ok=False, error="问题为空"
            )

        start = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        steps: list[AgentStep] = []
        usage: dict[str, int] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            # Phase 6：缓存命中量（prompt_tokens 的子集），用于量化提示词固定开销
            "cached_tokens": 0,
        }
        provider = ""
        model = ""
        # 第五层防线用的两个计数器：
        #   grounded_calls —— 本次提问中「真的成功取到数」的次数（判据是工具返回 ok）
        #   guard_used     —— 已用掉的补救次数，防止反复拦截形成死循环
        grounded_calls = 0
        guard_used = 0

        try:
            for iteration in range(1, self.max_iterations + 1):
                # ---------- ① 想：让模型决定下一步 ----------
                response = self.llm_client.chat(messages, tools=self.tools.definitions)
                provider = response.provider or provider
                model = response.model or model
                self._accumulate_usage(usage, response)

                # 把 assistant 这一轮的动作记进对话历史。
                # 必须原样追加（包括 tool_calls），否则下一轮模型看不到自己做过什么。
                messages.append(response.to_assistant_message())

                # ---------- ④ 收敛：没有工具调用 = 模型给出最终答案 ----------
                if not response.has_tool_calls:
                    # ---------- ⑤ 第五层防线：答案来源校验 ----------
                    # 只有「一次都没成功取到数」+「答案里有数据型数字」同时成立，
                    # 才判定为疑似编造。单独任一条都不够：
                    #   · 只看到「没取数」就拦 → 会把正常的拒答、寒暄也拦下来；
                    #   · 只看到「有数字」就拦 → 拒答文案里天然带日期（2026-06-20）。
                    ungrounded = grounded_calls == 0 and _has_ungrounded_number(
                        response.content
                    )
                    if ungrounded and guard_used < self.guard_retries:
                        guard_used += 1
                        # 把这次拦截记进 trace：前端能展示「模型编了 → 系统拦了 →
                        # 模型重查了」的完整链路，而不是悄悄重试。
                        steps.append(
                            AgentStep(
                                iteration=iteration,
                                kind="guard",
                                content=response.content,
                                observation=_GUARD_REMINDER,
                            )
                        )
                        # 用 user 角色追加提醒（而不是 system）：
                        # 多数供应商对「对话中段的 system 消息」支持不一，
                        # 用 user 角色兼容性最好，语义上也说得通——这相当于
                        # 「用户（其实是系统）对它刚才的回答提出了质疑」。
                        messages.append({"role": "user", "content": _GUARD_REMINDER})
                        continue

                    steps.append(
                        AgentStep(
                            iteration=iteration,
                            kind="final_answer",
                            content=response.content,
                        )
                    )
                    if ungrounded:
                        # 补救次数已用尽仍在编造 → 不把那些数字给用户看。
                        # 这是「宁可说查不到，也不能编」这条产品原则的最后一道闸门。
                        return AgentAnswer(
                            question=question,
                            answer=_GUARD_BLOCKED_ANSWER,
                            ok=False,
                            steps=steps,
                            iterations=iteration,
                            provider=provider,
                            model=model,
                            usage=usage,
                            elapsed_ms=(time.perf_counter() - start) * 1000,
                            error="回答中出现无法溯源的数值（疑似编造），已被来源校验拦截",
                        )
                    return AgentAnswer(
                        question=question,
                        answer=response.content or "（模型没有返回内容）",
                        ok=True,
                        steps=steps,
                        iterations=iteration,
                        provider=provider,
                        model=model,
                        usage=usage,
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                    )

                # ---------- ② 做 + ③ 看：逐个执行工具 ----------
                # 注意是 for 而不是只取第一个：模型完全可能在一轮里同时发多个工具调用
                # （比如同时查 DAU 和留存）。全部执行完再进入下一轮，
                # 而且每个 tool_call 都必须有对应的 tool 消息，
                # 否则下一轮请求会因「缺失 tool 响应」被 API 拒绝。
                for call in response.tool_calls:
                    tool_start = time.perf_counter()
                    result = self.tools.execute(
                        call.name,
                        call.arguments_raw or call.arguments,
                    )
                    steps.append(
                        AgentStep(
                            iteration=iteration,
                            kind="tool_call",
                            tool_name=call.name,
                            arguments=call.arguments,
                            observation=result.content,
                            # 结构化数据一并存下来：Phase 4 前端靠它画图、
                            # Phase 5 评测靠它核对「模型说的数字是否等于库里返回的数字」。
                            result=result.data,
                            ok=result.ok,
                            content=response.content,
                            elapsed_ms=(time.perf_counter() - tool_start) * 1000,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        }
                    )
                    # 只有「真的取到数」才计数。注意判据是 result.ok 而不是
                    # 「调用过」——模型可能调了工具但参数非法被拦下，
                    # 那不算有数据支撑，这种情况第五层防线依然应该生效。
                    if call.name == "query_metric" and result.ok:
                        grounded_calls += 1

            # ---------- 熔断：轮次用尽仍未收敛 ----------
            return AgentAnswer(
                question=question,
                answer=(
                    f"抱歉，我连续尝试了 {self.max_iterations} 轮仍未得出可靠结论。"
                    f"建议把问题拆得更具体一些，例如「最近 7 天的次日留存率是多少」，"
                    f"这样我能更快定位到对应的指标。"
                ),
                ok=False,
                steps=steps,
                iterations=self.max_iterations,
                provider=provider,
                model=model,
                usage=usage,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=f"超过最大迭代轮数 {self.max_iterations}",
            )

        except LLMNotConfiguredError:
            return self._failure_answer(question, steps, usage, provider, model, start,
                                        _NOT_CONFIGURED_HINT, "LLM 未配置")
        except GameAgentError as exc:
            # 走到这里说明主备供应商全挂了，或出现了其他业务异常
            return self._failure_answer(
                question, steps, usage, provider, model, start,
                f"抱歉，本次分析失败了：{exc.message}\n请稍后重试，或换个问法。",
                exc.message,
            )
        except Exception as exc:  # noqa: BLE001 - 入口方法必须兜住一切，不能让前端崩
            return self._failure_answer(
                question, steps, usage, provider, model, start,
                f"抱歉，出现了未预期的错误：{exc}\n请稍后重试。",
                str(exc),
            )

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _accumulate_usage(usage: dict[str, int], response: Any) -> None:
        """累计 token 用量。

        为什么要在 Agent 层再统计一遍，而不是直接用 llm_client.usage？
          因为 client 的 usage 是「进程级累计」（一直往上加），
          而用户想知道的是「**这一次提问**花了多少 token」。
          两个口径都合理，但用途不同，所以各留一份。
        """
        usage["llm_calls"] += 1
        usage["prompt_tokens"] += int(getattr(response, "prompt_tokens", 0) or 0)
        usage["completion_tokens"] += int(getattr(response, "completion_tokens", 0) or 0)
        usage["total_tokens"] += int(getattr(response, "total_tokens", 0) or 0)
        usage["cached_tokens"] += int(getattr(response, "cached_tokens", 0) or 0)

    def _failure_answer(
        self,
        question: str,
        steps: list[AgentStep],
        usage: dict[str, int],
        provider: str,
        model: str,
        start: float,
        answer: str,
        error: str,
    ) -> AgentAnswer:
        """构造一个「失败但优雅」的回答。

        失败时也把已经走过的步骤带回去 —— 这样用户/开发能看到
        「是在哪一步失败的」，而不是只拿到一句「出错了」。
        """
        return AgentAnswer(
            question=question,
            answer=answer,
            ok=False,
            steps=steps,
            iterations=len({s.iteration for s in steps}),
            provider=provider,
            model=model,
            usage=usage,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            error=error,
        )