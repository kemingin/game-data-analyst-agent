# -*- coding: utf-8 -*-
"""
Agent 主循环单元测试
================================================================================
【怎么测一个「会调大模型」的东西？】

  关键在于 LLMClient 是可注入的。测试里塞一个「按剧本演出」的假客户端：
  你告诉它第一轮返回什么、第二轮返回什么，它就照着演。

  这样我们就能精确地验证 ReAct 循环的每一条分支：
    · 模型直接回答       → 一轮结束
    · 模型要调工具       → 执行工具并把结果塞回上下文
    · 一轮要调多个工具   → 每个都要执行、每个都要有对应的 tool 消息
    · 工具失败           → 失败信息要变成观察结果，而不是异常
    · 模型一直调工具     → 必须在 max_iterations 处熔断
    · LLM 挂了           → 要优雅降级，不能把异常抛给前端

  其中「每个 tool_call 都必须有对应的 tool 消息」这条最容易被忽略：
  OpenAI 协议要求它们一一对应，少一条下一轮请求就会被 API 直接拒绝（400）。
  这条 bug 在手动测试时表现为「Agent 偶尔抽风」，很难定位，所以必须有测试兜住。
"""

from __future__ import annotations

import json

import pytest

from src.agent.llm_client import LLMResponse, ToolCall
from src.agent.react_agent import AgentAnswer, GameDataAgent
from src.exceptions import LLMError, LLMNotConfiguredError

# ===========================================================================
# 一、测试替身：按剧本演出的假 LLM 客户端
# ===========================================================================

class FakeLLMClient:
    """假客户端：只实现 chat()，因为主循环只用这一个方法。

    「脚本耗尽就抛异常」是刻意的：如果主循环发生了非预期的额外调用
    （比如多了几轮循环），测试会立刻失败，而不是悄悄通过。
    """

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, tool_choice=None):
        # 深拷贝一层 messages，因为主循环会继续往里 append，
        # 不复制的话我们事后看到的永远是「最终状态」，没法断言中间过程
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})

        if not self.responses:
            raise AssertionError("假客户端剧本已用完 —— 主循环可能多跑了轮次")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_response(
    content: str = "",
    tool_calls: list[tuple[str, str, dict]] | None = None,
    provider: str = "deepseek",
    model: str = "deepseek-chat",
    prompt_tokens: int = 100,
    completion_tokens: int = 10,
) -> LLMResponse:
    """构造一个 LLMResponse。tool_calls 用 (id, 工具名, 参数字典) 三元组描述。"""
    calls = [
        ToolCall(
            id=call_id,
            name=name,
            arguments=arguments,
            arguments_raw=json.dumps(arguments, ensure_ascii=False),
        )
        for call_id, name, arguments in (tool_calls or [])
    ]
    return LLMResponse(
        content=content,
        tool_calls=calls,
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def build_agent(responses: list, **kwargs) -> tuple[GameDataAgent, FakeLLMClient]:
    llm = FakeLLMClient(responses)
    kwargs.setdefault("max_iterations", 6)
    return GameDataAgent(llm_client=llm, **kwargs), llm


# ===========================================================================
# 二、正常路径
# ===========================================================================

def test_direct_answer_without_tools():
    """模型不需要工具就能回答时，一轮结束，不再多余调用。"""
    agent, llm = build_agent([make_response(content="你好，我是游戏数据智能分析师。")])

    result = agent.ask("你好")

    assert result.ok
    assert result.answer == "你好，我是游戏数据智能分析师。"
    assert result.iterations == 1
    assert len(result.steps) == 1
    assert result.steps[0].kind == "final_answer"
    assert len(llm.calls) == 1


def test_first_request_carries_system_prompt_and_tools():
    """第一次请求必须同时带上系统提示词、历史消息和工具清单。"""
    agent, llm = build_agent([make_response(content="ok")])

    agent.ask("最近日活多少")

    first = llm.calls[0]
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][-1] == {"role": "user", "content": "最近日活多少"}
    assert first["tools"]  # 工具清单非空


def test_tool_call_then_final_answer():
    """标准的两轮：先调工具拿数据，再基于数据作答。"""
    agent, llm = build_agent(
        [
            make_response(
                content="我先看看有哪些指标。",
                tool_calls=[("call_1", "list_metrics", {})],
            ),
            make_response(content="系统支持 12 个指标。"),
        ]
    )

    result = agent.ask("你们能查什么？")

    assert result.ok
    assert result.answer == "系统支持 12 个指标。"
    assert result.iterations == 2
    assert [s.kind for s in result.steps] == ["tool_call", "final_answer"]
    assert result.steps[0].tool_name == "list_metrics"
    assert result.steps[0].ok
    # 模型这一轮的正文也要被记下来（前端展示「思考过程」用）
    assert result.steps[0].content == "我先看看有哪些指标。"
    assert len(llm.calls) == 2


def test_tool_result_is_fed_back_into_context():
    """上一轮的工具结果必须出现在下一轮的 messages 里。

    这是 ReAct 的「Observation」环节。如果漏了这一步，
    模型会在第二轮里完全不知道工具返回了什么，只能凭空编 —— 这正是最危险的情况。
    """
    agent, llm = build_agent(
        [
            make_response(tool_calls=[("call_1", "list_metrics", {})]),
            make_response(content="好的"),
        ]
    )

    agent.ask("你们能查什么？")

    second_roles = [m["role"] for m in llm.calls[1]["messages"]]
    # system → user 提问 → assistant 要求调工具 → tool 返回结果
    assert second_roles == ["system", "user", "assistant", "tool"]

    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["tool_call_id"] == "call_1"
    # 工具返回的原文要一字不改地传下去（含真实指标 ID）
    assert "dau" in tool_message["content"]

    assistant_message = llm.calls[1]["messages"][2]
    assert assistant_message["tool_calls"][0]["function"]["name"] == "list_metrics"


def test_multiple_tool_calls_in_one_turn():
    """模型一轮里同时要求调多个工具时，每个都要执行、且都有对应的 tool 消息。"""
    agent, llm = build_agent(
        [
            make_response(
                tool_calls=[
                    ("call_a", "list_metrics", {}),
                    ("call_b", "get_metric_detail", {"metric_id": "dau"}),
                ]
            ),
            make_response(content="都查好了"),
        ]
    )

    result = agent.ask("能查什么？日活怎么算？")

    assert result.ok
    assert [s.tool_name for s in result.steps[:2]] == ["list_metrics", "get_metric_detail"]
    assert len(result.steps) == 3

    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_a", "call_b"]


def test_tool_structured_data_is_carried_into_step():
    """工具的结构化返回必须一路传到 AgentStep.result（Phase 4 前端画图依赖它）。

    为什么专门测这一条？
      因为图表处在数据流的末端：中间任何一环把 data 丢掉，
      界面上的表现都只是「图没出来」，不会报任何错 —— 这种静默失败最难排查。
      所以用真实数据库跑一次，把 columns / rows / rendered_sql 三样都断言到。
    """
    agent, _ = build_agent(
        [
            make_response(
                tool_calls=[
                    ("call_1", "query_metric", {"metric_id": "retention_rate", "params": {"day_n": 1}})
                ]
            ),
            make_response(content="次日留存已查到。"),
        ]
    )

    result = agent.ask("次留是多少？")

    step = result.steps[0]
    assert step.ok
    data = step.result
    assert data["metric_id"] == "retention_rate"
    assert data["columns"], "缺少列名，前端无法建表"
    assert data["rows"], "缺少数据行，前端无法画图"
    assert data["rendered_sql"], "缺少展示版 SQL，前端无法展示「这条 SQL 是怎么来的」"
    # 序列化时也不能丢（前端拿到的是 to_dict() 的产物）
    assert step.to_dict()["result"]["metric_id"] == "retention_rate"


# ===========================================================================
# 三、异常与失败路径
# ===========================================================================

def test_tool_failure_becomes_observation_and_agent_recovers():
    """工具失败时不该中断对话，而要把错误当成观察结果喂回去。

    这里模拟「模型把参数名写成了 days」：工具的报错里带着「可用参数」清单，
    模型看到后就能在下一轮改对 —— 这就是 Agent 的自我纠错能力。
    """
    agent, llm = build_agent(
        [
            make_response(
                tool_calls=[("call_1", "query_metric", {"metric_id": "dau", "params": {"days": 7}})]
            ),
            make_response(content="参数名写错了，我改用默认的最近 7 天。"),
        ]
    )

    result = agent.ask("最近 7 天日活多少？")

    assert result.ok
    assert result.steps[0].ok is False
    assert "可用参数" in result.steps[0].observation
    # 失败信息确实进了模型上下文
    assert "可用参数" in llm.calls[1]["messages"][-1]["content"]
    # 对话没有中断，照常走到了最终回答
    assert result.answer == "参数名写错了，我改用默认的最近 7 天。"


def test_loop_stops_at_max_iterations():
    """模型陷入「反复调工具」的死循环时必须熔断。

    这是「循环 + 外部 API」的必备保护：没有它，一次异常请求就能烧掉大量 token。
    """
    endless = [make_response(tool_calls=[("call_1", "list_metrics", {})]) for _ in range(10)]
    agent, llm = build_agent(endless, max_iterations=3)

    result = agent.ask("一直查下去")

    assert result.ok is False
    assert result.iterations == 3
    assert len(llm.calls) == 3              # 只调了 3 次，没有第 4 次
    assert len(result.steps) == 3
    assert "3 轮" in result.answer           # 文案里说清尝试了几轮
    assert "超过最大迭代轮数" in result.error


def test_not_configured_returns_actionable_message():
    """没配 Key 时，必须给出「怎么修」的具体指引，而不是一句「出错了」。"""
    agent, _ = build_agent([LLMNotConfiguredError("没有可用的 API Key")])

    result = agent.ask("日活多少")

    assert result.ok is False
    assert ".env" in result.answer
    assert "DEEPSEEK_API_KEY" in result.answer


def test_all_providers_failed_returns_friendly_message():
    agent, _ = build_agent([LLMError("所有 LLM 供应商调用均失败：\n- DeepSeek：503")])

    result = agent.ask("日活多少")

    assert result.ok is False
    assert "本次分析失败" in result.answer
    assert result.error is not None


def test_unexpected_exception_is_contained():
    """未预期的异常也必须被兜住 —— ask() 是前端唯一入口，不能让页面崩掉。"""
    agent, _ = build_agent([RuntimeError("意料之外的错误")])

    result = agent.ask("日活多少")

    assert result.ok is False
    assert "意料之外的错误" in result.answer


def test_failure_keeps_steps_for_debugging():
    """失败时也要把走过的步骤带回去，方便定位「是在哪一步挂的」。"""
    agent, _ = build_agent(
        [
            make_response(tool_calls=[("call_1", "list_metrics", {})]),
            LLMError("第二轮挂了"),
        ]
    )

    result = agent.ask("能查什么？")

    assert result.ok is False
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "list_metrics"


def test_empty_question_is_rejected_without_calling_llm():
    """空问题直接在本地拦掉，不浪费一次 API 调用。"""
    agent, llm = build_agent([make_response(content="不该被调用")])

    result = agent.ask("   ")

    assert result.ok is False
    assert llm.calls == []
    assert "请输入" in result.answer


# ===========================================================================
# 四、上下文、用量与序列化
# ===========================================================================

def test_history_is_prepended_in_order():
    """多轮追问：历史消息要插在系统提示词之后、当前问题之前。"""
    agent, llm = build_agent([make_response(content="ok")])
    history = [
        {"role": "user", "content": "最近留存怎么样"},
        {"role": "assistant", "content": "D1 留存 42.66%"},
    ]

    agent.ask("那版本上线后呢？", history=history)

    roles = [m["role"] for m in llm.calls[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.calls[0]["messages"][3]["content"] == "那版本上线后呢？"


def test_usage_is_accumulated_per_question():
    """usage 是「这一次提问」的口径，要把这次所有轮次的 token 加起来。"""
    agent, _ = build_agent(
        [
            make_response(tool_calls=[("c1", "list_metrics", {})], prompt_tokens=100, completion_tokens=20),
            make_response(content="done", prompt_tokens=200, completion_tokens=30),
        ]
    )

    result = agent.ask("能查什么？")

    assert result.usage["llm_calls"] == 2
    assert result.usage["prompt_tokens"] == 300
    assert result.usage["completion_tokens"] == 50
    assert result.usage["total_tokens"] == 350


def test_provider_and_model_are_reported():
    """前端要显示「本次由哪个模型回答」，便于用户判断可信度与成本。"""
    agent, _ = build_agent([make_response(content="ok", provider="glm", model="glm-4-flash")])

    result = agent.ask("你好")

    assert result.provider == "glm"
    assert result.model == "glm-4-flash"


def test_custom_system_prompt_is_used():
    agent, llm = build_agent([make_response(content="ok")], system_prompt="自定义提示词")

    agent.ask("你好")

    assert llm.calls[0]["messages"][0]["content"] == "自定义提示词"


def test_answer_to_dict_is_json_serializable():
    """to_dict 的结果要能直接 json.dumps（Phase 4 的 Streamlit 会这么用）。"""
    agent, _ = build_agent([make_response(content="ok")])

    payload = json.dumps(agent.ask("你好").to_dict(), ensure_ascii=False)

    assert "question" in payload
    assert "steps" in payload


def test_default_agent_constructs_without_api_key():
    """没有任何 Key 时，构造 Agent 本身不应该失败 —— 只有真正提问才会走到报错分支。

    这一点对 Phase 4 很重要：前端启动时要先渲染页面，
    不能因为「没配 Key」就整个起不来（那样用户连配 Key 的界面都看不到）。
    """
    agent = GameDataAgent()
    assert isinstance(agent.ask(""), AgentAnswer)


# ===========================================================================
# 五、第五层防线：答案来源校验（Phase 6 新增）
# ===========================================================================
#
# 【这组测试要守住什么？】
#
#   真机验证时发现的缺陷：把主供应商摘掉、切到能力较弱的 GLM-4-Flash 后，
#   模型会跳过工具直接凭记忆编数字（实测「日活 100,000 人」）。
#   「禁止编造」原本只是提示词里的软约束，这组测试守住的是**程序层的硬校验**。
#
#   两条边界必须同时守住，缺一不可：
#     · 该拦的必须拦住（编数字不给用户看）
#     · 不该拦的绝不能拦（正确拒答、寒暄、已取数后的作答）

def test_ungrounded_number_detection_ignores_dates_and_versions():
    """检测函数必须放过日期与版本号 —— 否则会把正确的拒答误判成编造。

    这条测试的样本全部取自 Phase 5/6 真机跑出来的**真实回答**，
    不是凭空构造的，所以它测的是「线上真的会遇到的文本形态」。
    """
    from src.agent.react_agent import _has_ungrounded_number

    # 真实拒答文案：含日期、窗口天数，但没有数据值 → 不该被判为编造
    refusal = (
        "我的数据只覆盖 2026-06-20 ~ 2026-09-17（共 90 天），"
        "去年 12 月（2025-12）超出了这个范围，暂时查不了。"
        "想看活跃趋势的话，我可以给你最近 7 天或最近 30 天的日活走势。"
    )
    assert _has_ungrounded_number(refusal) is False

    # 版本对比类回答里出现 v2.0.0 也不能算数据值
    assert _has_ungrounded_number("v2.0.0 上线后留存有所提升。") is False

    # 真实编造文案：千分位数字 → 必须被抓出来
    assert _has_ungrounded_number("最近 7 天的日活是 100,000 人。") is True
    # 百分比形态
    assert _has_ungrounded_number("次日留存率大约是 45.00%。") is True
    # 无千分位的大数字
    assert _has_ungrounded_number("日活大约 120000 左右。") is True

    # 空文本、纯寒暄不应误报
    assert _has_ungrounded_number("") is False
    assert _has_ungrounded_number("你好，我是游戏数据智能分析师。") is False


def test_guard_intercepts_fabrication_and_forces_requery():
    """核心场景：模型编数字 → 系统拦截 → 模型重新取数 → 基于真实数据作答。

    剧本三幕：
      ① 模型跳过工具直接编「100,000 人」        → 应被拦截
      ② 被提醒后老老实实调用 query_metric        → 工具真的执行（连真库）
      ③ 基于工具返回作答                          → 正常返回
    """
    agent, llm = build_agent(
        [
            make_response(content="最近 7 天的日活是 100,000 人。"),
            make_response(tool_calls=[("c1", "query_metric", {"metric_id": "dau"})]),
            make_response(content="最近 7 天的日活已从数据库取到，详见上表。"),
        ]
    )

    result = agent.ask("最近 7 天的日活是多少？")

    assert result.ok is True
    # 拦截这件事必须留在 trace 里，而不是悄悄重试 —— 可解释性是数据产品的基本要求
    assert [s.kind for s in result.steps] == ["guard", "tool_call", "final_answer"]
    assert result.steps[0].content == "最近 7 天的日活是 100,000 人。"   # 被拦下的原文
    assert "系统校验未通过" in result.steps[0].observation               # 喂回模型的提醒
    assert result.steps[1].tool_name == "query_metric"
    assert result.steps[1].ok is True     # 补救这一次是「真的取到数」，否则防线仍会拦
    assert len(llm.calls) == 3

    # 第二次请求里必须带上那条提醒，否则模型不知道自己为什么被拦
    second_roles = [m["role"] for m in llm.calls[1]["messages"]]
    assert second_roles == ["system", "user", "assistant", "user"]
    assert "系统校验未通过" in llm.calls[1]["messages"][-1]["content"]


def test_guard_does_not_fire_when_data_came_from_tools():
    """已经取过数了就不该拦 —— 基于工具返回的数字作答是正常行为。

    这是防「优化过头」的护栏：校验太激进会把正常的分析流程打断，
    每打断一次就多花一轮 token，还拖慢响应。
    """
    agent, llm = build_agent(
        [
            make_response(tool_calls=[("c1", "query_metric", {"metric_id": "dau"})]),
            make_response(content="最近 7 天的日活平均是 96,000 人。"),
        ]
    )

    result = agent.ask("最近 7 天的日活是多少？")

    assert result.ok is True
    assert [s.kind for s in result.steps] == ["tool_call", "final_answer"]
    assert result.steps[0].ok is True
    assert result.answer == "最近 7 天的日活平均是 96,000 人。"
    assert len(llm.calls) == 2          # 没有多余的重试


def test_guard_does_not_fire_on_legitimate_refusal():
    """正确拒答不能被拦 —— 这是超范围拒答率 100% 的守门测试。

    拒答文案天然含日期（2026-06-20）和窗口天数（90 天），
    如果校验规则写成「含数字就拦」，这条用例会失败，拒答率也会从 100% 掉下来。
    """
    agent, llm = build_agent(
        [
            make_response(
                content=(
                    "我的数据只覆盖 2026-06-20 ~ 2026-09-17（共 90 天），"
                    "去年 12 月超出了这个范围，暂时查不了。"
                )
            )
        ]
    )

    result = agent.ask("去年 12 月的日活是多少？")

    assert result.ok is True
    assert [s.kind for s in result.steps] == ["final_answer"]
    assert "2026-06-20" in result.answer
    assert len(llm.calls) == 1


def test_guard_gives_up_and_hides_numbers_after_retries():
    """补救用尽后仍在编造 → 不把那些数字给用户看（宁可说查不到，也不能编）。

    这是整条防线的最后一道闸门。返回 ok=False 而不是「带个警告照常显示」，
    是因为编造数字对运营决策的危害远大于「这次没答上来」——
    用户看到一个看起来合理的假数字，是会真的照着做决策的。
    """
    agent, llm = build_agent(
        [
            make_response(content="最近 7 天的日活是 100,000 人。"),
            make_response(content="最近 7 天的日活就是 100,000 人。"),
        ]
    )
    assert agent.guard_retries == 1

    result = agent.ask("最近 7 天的日活是多少？")

    assert result.ok is False
    assert "100,000" not in result.answer
    assert "不能给你具体数字" in result.answer
    assert "疑似编造" in (result.error or "")
    assert [s.kind for s in result.steps] == ["guard", "final_answer"]
    assert len(llm.calls) == 2          # 补救 1 次，不多不少


def test_guard_never_loops_forever():
    """即使模型一直编，也必须在上限处停下 —— 不能变成死循环烧 token。

    这里把补救次数设成 0（相当于关掉补救），验证的是「防线本身不会失控」：
    一次拦截机会都不给时，应当直接走「隐藏数字」分支，而不是继续循环。
    """
    agent, llm = build_agent(
        [make_response(content="日活是 100,000 人。")],
        max_iterations=6,
    )
    agent.guard_retries = 0

    result = agent.ask("日活多少？")

    assert result.ok is False
    assert len(llm.calls) == 1
    assert [s.kind for s in result.steps] == ["final_answer"]