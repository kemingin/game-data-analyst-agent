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

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src import config as cfg
from src.agent.llm_client import LLMClient
from src.agent.prompts import build_system_prompt
from src.agent.tools import ToolExecutor
from src.exceptions import GameAgentError, LLMNotConfiguredError
from src.grounding import check_answer_numbers, collect_ground_truth_params, collect_ground_truth_values
from src.grounding import grounding_decision, unexplained_ratio
from src.metrics.registry import MetricRegistry, get_registry

if TYPE_CHECKING:
    # 只为类型标注导入，运行期不真正加载 —— 避免「Agent 层 import 组装根」
    # 这种反向依赖，也让 import 顺序不敏感。
    from src.context import DatasetContext

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
# 【这一层怎么工作？（Phase 9 起升级为「逐数字溯源」）】
#
#   旧判据（Phase 6）只问两个粗糙的问题：
#     ① 本轮有没有**成功取过数**？ ② 回答里有没有「数据型数字」？
#   只要有一次成功取数就整条放行 —— 于是「查了 1 次 DAU、又在答案里编了
#   5 个留存率」这种情况能大摇大摆过去。防线存在，但拦不住「混着编造」。
#
#   新判据把回答里**每一个数字**反向映射到本次查询的结果集：
#       pass     —— 没有数字，或每个数字都能在结果集里找到出处
#       annotate —— 少量数字找不到出处（占比 ≤ 1/3）：回答照给，末尾加提示
#       block    —— 找不到出处的占比 > 1/3：判定疑似编造，不把数字给用户看
#
#   判定用的原语（抽数字 / 比数字 / 收参考值）来自 src/grounding.py ——
#   与评测体系共用同一套口径。共用而不是另写一套，是为了避免出现
#   「运行时拦了、评测说没问题」这种自相矛盾的结论。
#
# 【为什么分级，而不是「一个数字对不上就拦」？】
#   因为「对不上」不完全等于编造。口语化的「稳定在 530 左右」（实际 533）、
#   以及推导池没覆盖到的表达（加权、中位数）都会被归到「找不到出处」。
#   一律拦截会把一整段正确推理因为一个近似值废掉 —— 过度拦截的代价是
#   用户再也不信这个提示。而一条回答里超过三分之一的数字都无出处，
#   就不是「表达方式的差异」能解释的了。
#
# 【向后兼容】一次成功取数都没有时，参考集为空 → 所有数字都无出处 →
#   占比 1.0 → 走 block，与旧判据「无取数 + 有数字 → 拦截」完全一致。
#   新判据是旧判据的**严格超集**：旧逻辑能拦的它都拦，
#   旧逻辑放过的「有查询但混着编造数字」它也能拦。
#
# 【为什么要绕开「日期」「版本号」「窗口量词」？】
#   拒答场景的回答天然含数字：
#   「我的数据只覆盖 2026-06-20 ~ 2026-09-17（共 90 天）」——
#   如果按「含数字」一刀切，这条正确的拒答会被误判成编造，
#   反而把 100% 的超范围拒答率打坏。这些剔除规则集中在
#   src/grounding.py 的 extract_numbers 里（每一条都对应一次真实误报）。
#
# 【这一点怎么讲清楚？】
#   「软约束 + 硬校验」是 LLM 应用里反复出现的一对。提示词负责让模型
#   大概率做对，程序校验负责在它做错时兜住。只靠提示词，换一个弱一点的
#   模型就会失守；只靠校验，会频繁打断正常的推理。两者缺一不可 ——
#   这也是我做多供应商降级时才发现的问题：降级链路通了，
#   但备用模型的指令遵循能力下降，把提示词的软约束击穿了。

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

# 分级处置的中间档：存疑数字不多，回答照给，但在末尾如实标注。
# 为什么是「加一行提示」而不是「把那些数字从正文里抹掉」？
#   ① 抹数字会破坏 Markdown 表格和句子结构，读起来更糟；
#   ② 用户有权知道「哪几个数没出处」，而不是被悄悄改过的答案。
# 文案里列出具体数字，是为了让用户能自己去核对 —— 可解释性要落到可验证上。
_GROUNDING_ANNOTATION = (
    "\n\n> 注：以上回答中的 {count} 个数字（{numbers}）未能在本次查询结果中"
    "找到出处，请谨慎采信；如需确认，我可以重新取数并给出明细。"
)


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
    # 第五层防线的判定结果（Phase 9）。为 None 表示这次回答没有经过溯源判定
    # （比如 LLM 未配置、熔断等提前返回的路径）。
    # 为什么要把判定结果随答案一起返回，而不是只写进 steps？
    #   因为前端要能一眼看出「这条答案是原样给的、还是被标注过」，评测台也要
    #   能直接统计「有多少条走了 annotate」—— 藏在自然语言里没法程序化读取。
    grounding: dict[str, Any] | None = None

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
            "grounding": self.grounding,
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
        context: "DatasetContext | None" = None,
    ) -> None:
        """依赖优先级：显式传入的参数 > context > 全局默认。

        【context 一次解决三件事，少一件都会出错】
          1. registry     —— 用哪套指标（上传数据集的方案 ≠ 内置方案）
          2. tools        —— 查哪个库（含 generator 的日期窗口、executor 的白名单）
          3. system_prompt —— 数据窗口写进提示词，否则模型会被告知错误的边界，
             出现「数据只到 9-17，却按 9-30 的窗口去问」这类错位。

          只传 context 而不单独传这三个参数，是最不容易接错的用法。
        """
        self.context = context
        self.registry: MetricRegistry = (
            registry or (context.registry if context is not None else get_registry())
        )
        # llm_client 声明成 Any 而不是 LLMClient：只要实现了 chat() 就能替换进来，
        # 测试时塞一个脚本化的假客户端即可，不必真的联网。
        self.llm_client: Any = llm_client if llm_client is not None else LLMClient()
        # 把 context 透传给工具层：让它自己去装配自洽的 generator/executor
        self.tools: ToolExecutor = tools or ToolExecutor(
            registry=self.registry, context=context
        )
        self.max_iterations: int = (
            cfg.AGENT_MAX_ITERATIONS if max_iterations is None else max_iterations
        )
        # 第五层防线（答案来源校验）的补救次数上限，同样允许注入以便测试
        self.guard_retries: int = getattr(cfg, "AGENT_GUARD_RETRIES", 1)
        if system_prompt is not None:
            self.system_prompt: str = system_prompt
        elif context is not None:
            self.system_prompt = context.build_system_prompt()
        else:
            # 未传 context → 与改造前逐行相同的旧行为
            self.system_prompt = build_system_prompt(self.registry)

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
                    # ---------- ⑤ 第五层防线：逐数字溯源 ----------
                    # 把回答里的每个数字拿去「本次会话真正取到过的数值」里找出处。
                    # 参考集有两部分，缺一不可：
                    #   · 结果集里的数值 —— 模型引用的原始数据；
                    #   · 查询用到的数值型参数 —— 「最近 7 天」的 7 来自参数而非结果集，
                    #     不把它算作有出处，会制造大量假警报。
                    # 注意 grounded_calls 不再直接决定拦截与否（旧判据的漏洞就在这），
                    # 它只作为诊断信息留在 trace 里，用来区分「压根没查」和「查了但混编」。
                    number_check = check_answer_numbers(
                        response.content,
                        collect_ground_truth_values(steps),
                        extra_allowed=collect_ground_truth_params(steps),
                    )
                    decision = grounding_decision(number_check)
                    ungrounded = decision == "block"
                    grounding = {
                        "decision": decision,
                        "total": number_check.total,
                        "traceable": number_check.matched_count,
                        "unexplained": [item.raw for item in number_check.unexplained],
                        "unexplained_ratio": round(unexplained_ratio(number_check), 4),
                        "grounded_calls": grounded_calls,
                    }
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
                                result=grounding,
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
                            grounding=grounding,
                        )

                    answer_text = response.content or "（模型没有返回内容）"
                    if decision == "annotate":
                        # 少量存疑：回答照给，但在末尾如实标注哪几个数字没出处。
                        # 只标注、不改正文 —— 抹掉数字会破坏 Markdown 表格，
                        # 而且用户有权知道被质疑的是哪几个数。
                        answer_text += _GROUNDING_ANNOTATION.format(
                            count=len(number_check.unexplained),
                            numbers=number_check.describe_unexplained(),
                        )
                        steps.append(
                            AgentStep(
                                iteration=iteration,
                                kind="grounding",
                                observation=answer_text,
                                result=grounding,
                            )
                        )
                    return AgentAnswer(
                        question=question,
                        answer=answer_text,
                        ok=True,
                        steps=steps,
                        iterations=iteration,
                        provider=provider,
                        model=model,
                        usage=usage,
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                        grounding=grounding,
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