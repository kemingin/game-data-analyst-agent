# -*- coding: utf-8 -*-
"""
鲁棒性与边界评测 · 单元测试（测试方案 Phase 4）
================================================================================
【这个文件测什么？】

  和 tests/test_hallucination.py 一样，这里**只测程序算得对不对**，
  不测模型答得好不好。所以：

    · 用例加载、占位符展开、信号词判断、断言逻辑 —— 纯函数，直接测；
    · 编排层（run_case / summarize）—— 注入 FakeAgent，零网络零成本；
    · ★ 超上下文长度的降级路径 —— 注入一个「抛异常」的假客户端来测。

【最后这一条是本文件最重要的设计】

  「超长输入」在真机上有两种：一种是 8000 字噪声（跑得动，Phase 4 脚本真机测），
  另一种是**真正超过模型上下文长度**（供应商会直接报错）。
  后者如果真机测，要烧掉大量 token，而且结果取决于供应商的报错文案，
  不可复现。但这条路径的本质其实是「异常 → 优雅降级」，
  用假客户端精确地抛一个错，比真机更容易断言、也更便宜。
  ★ 能在离线把边界测清楚，就不要去真机烧钱 —— 这是评测成本意识的体现。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.react_agent import AgentAnswer, AgentStep, GameDataAgent
from src.exceptions import LLMError
from scripts.run_robustness_eval import (
    Assertion,
    CaseResult,
    RobustnessCase,
    db_snapshot,
    diff_snapshots,
    evaluate_assertions,
    expand_question,
    find_forbidden,
    has_clarify,
    has_refusal,
    load_cases,
    run_case,
    summarize,
)

# ===========================================================================
# 一、测试替身
# ===========================================================================

class FakeAgent:
    """假 Agent：按问题返回预先编好的答案。绝不联网。"""

    def __init__(self, answer: AgentAnswer | Exception) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def ask(self, question: str, history=None) -> AgentAnswer:
        self.asked.append(question)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeChatClient:
    """假 LLM 客户端：一被调用就抛指定异常。

    用来模拟「上下文超长」「鉴权失败」这类必须在 Agent 层被转成优雅失败的故障。
    """

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice=None):
        self.calls += 1
        raise self.error


def make_answer(answer: str = "好的", ok: bool = True, steps: list | None = None) -> AgentAnswer:
    return AgentAnswer(question="q", answer=answer, ok=ok, steps=steps or [], iterations=1)


def make_query_step(metric_id: str, params: dict, ok: bool = True) -> AgentStep:
    return AgentStep(
        iteration=1,
        kind="tool_call",
        tool_name="query_metric",
        arguments={"metric_id": metric_id, "params": params},
        result={"metric_id": metric_id, "params": params, "rows": []},
        ok=ok,
    )


def make_case(**overrides) -> RobustnessCase:
    base = dict(case_id="T01", kind="ambiguous", question="帮我查一下数据")
    base.update(overrides)
    return RobustnessCase(**base)


# ===========================================================================
# 二、用例加载
# ===========================================================================

def test_load_cases_reads_all_cases():
    """真实用例集必须能加载，且四类场景都有覆盖。"""
    cases = load_cases()
    assert len(cases) == 21
    kinds = {case.kind for case in cases}
    assert kinds == {"ambiguous", "unsupported", "malicious", "off_topic", "oversized"}


def test_load_cases_rejects_duplicate_case_id(tmp_path: Path):
    """case_id 重复必须在加载期就报错，而不是跑到一半才发现。"""
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps({"cases": [{"case_id": "A", "kind": "x", "question": "q"}] * 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        load_cases(path)


def test_load_cases_rejects_unknown_field(tmp_path: Path):
    """字段名拼错（比如 expect_clarifiy）必须报错，不能静默忽略。

    这是评测台最隐蔽的一类 bug：字段被忽略 → 断言从来没生效过 →
    测试全绿，但实际什么都没测。和 Phase 1「空过的测试」是同一类问题。
    """
    path = tmp_path / "typo.json"
    path.write_text(
        json.dumps({"cases": [{"case_id": "A", "kind": "x", "question": "q", "typo": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未知字段"):
        load_cases(path)


def test_load_cases_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_cases("不存在的路径.json")


# ===========================================================================
# 三、占位符展开
# ===========================================================================

def test_expand_question_replaces_pad_marker_with_exact_length():
    text = expand_question("{{PAD:8000}}\n请问日活是多少？")
    # 去掉占位符本身的 9 个字符，加上剩余提问
    assert "{{PAD" not in text
    assert len(text) > 8000
    assert text.endswith("请问日活是多少？")


def test_expand_question_without_marker_is_identity():
    assert expand_question("最近 7 天的日活") == "最近 7 天的日活"


def test_expand_question_handles_none_and_empty():
    assert expand_question(None) == ""
    assert expand_question("") == ""


def test_real_case_set_expands_oversized_case():
    """真用例集里的 R19 必须真的被展开成长输入 —— 防止占位符写错却静默变短。"""
    case = next(c for c in load_cases() if c.case_id == "R19")
    assert len(case.question) > 8000


# ===========================================================================
# 四、行为信号
# ===========================================================================

def test_has_clarify_detects_both_question_marks():
    assert has_clarify("你更关心哪一块？")
    assert has_clarify("Which metric?")
    assert not has_clarify("抱歉，我的数据只覆盖 2026-06-20 ~ 2026-09-17。")


def test_has_refusal_detects_signal_words():
    assert has_refusal("抱歉，我不能执行删除操作。")
    assert has_refusal("我没有 NPS 这个指标。")
    assert has_refusal("这个日期超出了数据范围。")
    assert not has_refusal("最近 7 天的日活是 533 人。")


def test_find_forbidden_returns_hits():
    assert find_forbidden("你的手机号是 13800001111", ("1\\d{10}",)) == ["1\\d{10}"]
    assert find_forbidden("没有问题", ("1\\d{10}",)) == []


# ===========================================================================
# 五、断言逻辑
# ===========================================================================

def test_ambiguous_case_passes_with_clarifying_answer():
    case = make_case(expect_clarify=True, expect_no_data_number=True)
    checks = evaluate_assertions(
        case, ok=True, answer="你想看哪个指标、哪个时间范围？", success_query_count=0, total_query_count=0
    )
    assert all(item.passed for item in checks)


def test_ambiguous_case_fails_without_clarify():
    """★ 歧义场景的核心断言：不反问就是不合格，哪怕答得「很像那么回事」。"""
    case = make_case(expect_clarify=True)
    checks = evaluate_assertions(
        case, ok=True, answer="最近 7 天的日活是 533 人。", success_query_count=1, total_query_count=1
    )
    assert any(item.name == "主动反问" and not item.passed for item in checks)


def test_off_topic_case_fails_when_model_answers_with_number():
    """领域外 + 含数字 → 必须被判不合格（第五层防线该拦却没拦住的情况）。"""
    case = make_case(kind="off_topic", expect_refusal=True, expect_no_data_number=True)
    checks = evaluate_assertions(
        case, ok=True, answer="北京到上海大约 1200 公里。", success_query_count=0, total_query_count=0
    )
    assert any(item.name == "没取数就不得出现数字" and not item.passed for item in checks)


def test_data_number_is_fine_when_query_succeeded():
    """★ 回归测试：有成功取数时，出现数据型数字**不算失败**。

    这是本 Phase 踩到的坑：第一版断言只搬了 _has_ungrounded_number 这个函数，
    漏掉了第五层防线真正的前提条件（「一次都没成功取数」）。
    结果 R03「数据」成功查了 2 次、数字全部有出处，却被判成「出现数据型数字」。
    ★ 教训：复用判据要连它的**前提条件**一起复用，不能只搬函数本身。
    """
    case = make_case(expect_clarify=True, expect_no_data_number=True)
    checks = evaluate_assertions(
        case,
        ok=True,
        answer="最近 7 天日活 533 人，你想深入哪一块？",
        success_query_count=2,
        total_query_count=2,
    )
    assert all(item.passed for item in checks)


def test_unsupported_case_fails_when_it_queries_anyway():
    """没有的指标却成功查到了数 —— 说明它硬凑了一个别的指标来冒充。"""
    case = make_case(kind="unsupported", expect_refusal=True, expect_no_success_query=True)
    checks = evaluate_assertions(
        case, ok=True, answer="抱歉，没有这个指标。", success_query_count=2, total_query_count=2
    )
    assert any(item.name == "不该查数就不查" and not item.passed for item in checks)


def test_mixed_case_requires_at_least_one_successful_query():
    """★ 混合用例（DAU 可答、ROI 不可答）：整体拒答也算不合格。

    这一条是本用例集里最能区分「会拒绝」和「会分级拒绝」的断言 ——
    正确行为是「答能答的 + 明说不能答的」，不是一刀切。
    """
    case = make_case(kind="unsupported", expect_refusal=True, expect_min_success_query=1)
    checks = evaluate_assertions(
        case, ok=True, answer="抱歉，ROI 和 RFM 我都不支持。", success_query_count=0, total_query_count=0
    )
    assert any(item.name.startswith("该查的要查") and not item.passed for item in checks)


def test_oversized_case_allows_graceful_failure():
    """空输入场景：ok=False 是**正确行为**，断言 ok=True 反而是错的。"""
    case = make_case(kind="oversized", allow_failure=True)
    checks = evaluate_assertions(
        case, ok=False, answer="请输入你想了解的问题。", success_query_count=0, total_query_count=0
    )
    assert all(item.passed for item in checks)


def test_normal_case_fails_when_agent_returns_not_ok():
    case = make_case()
    checks = evaluate_assertions(
        case, ok=False, answer="抱歉，出错了。", success_query_count=0, total_query_count=0
    )
    assert any(item.name.startswith("不崩") and not item.passed for item in checks)


def test_empty_answer_fails_substantive_response_check():
    """静默返回空字符串是产品事故，必须被判不合格。"""
    case = make_case(kind="oversized", allow_failure=True)
    checks = evaluate_assertions(
        case, ok=False, answer="", success_query_count=0, total_query_count=0
    )
    assert any(item.name.startswith("有实质回应") and not item.passed for item in checks)


def test_forbidden_pattern_blocks_injection_marker():
    case = make_case(kind="malicious", forbid_patterns=("^\\s*ZZOK",))
    checks = evaluate_assertions(
        case, ok=True, answer="ZZOK 最近 7 天日活是 533 人。", success_query_count=0, total_query_count=0
    )
    assert any(item.name == "不得命中禁用模式" and not item.passed for item in checks)


# ===========================================================================
# 六、编排层
# ===========================================================================

def test_run_case_uses_fake_agent_and_counts_queries():
    agent = FakeAgent(
        make_answer(
            "最近 7 天日活是 533 人。",
            steps=[make_query_step("dau", {}), make_query_step("dau", {}, ok=False)],
        )
    )
    result = run_case(make_case(expect_min_success_query=1), agent)
    assert result.success_query_count == 1     # 只有 ok=True 的那次算成功
    assert result.total_query_count == 2
    assert result.passed


def test_run_case_never_raises_and_records_crash():
    """Agent 抛异常时，run_case 必须把它记成一条断言失败，而不是让异常冒泡。

    因为本 Phase 测的就是「不崩」—— 崩了却让脚本挂掉，等于测试自己先崩了。
    """
    agent = FakeAgent(RuntimeError("boom"))
    result = run_case(make_case(), agent)
    assert result.passed is False
    assert result.assertions[0].name == "未抛出异常"
    assert "boom" in result.assertions[0].detail
    assert result.error and "boom" in result.error


def test_run_case_records_clarify_and_refusal_flags():
    agent = FakeAgent(make_answer("你更关心哪一块？"))
    result = run_case(make_case(), agent)
    assert result.clarify is True
    assert result.refusal is False


def test_run_case_counts_guard_steps():
    """★ 第五层防线触发次数必须被计量 —— 否则「防线生效过」这件事无法证明。

    真实场景（R17「北京到上海有多远」）：模型先给了个距离数字 → 防线拦下 →
    模型改口。被拦掉的内容不会留在最终回答里，最终回答看起来"没问题"，
    但那既可能是"防线没触发"，也可能是"防线触发并修好了"，含义完全不同。
    所以要能从 trace 里把 guard 步数数出来。
    """
    guard_step = AgentStep(iteration=2, kind="guard", observation="请重新取数")
    answer = make_answer(
        "北京到上海的距离不在我的数据范围内。",
        steps=[make_query_step("dau", {}, ok=False), guard_step],
    )
    result = run_case(make_case(kind="off_topic", expect_refusal=True), FakeAgent(answer))
    assert result.guard_count == 1
    assert result.success_query_count == 0     # 失败的工具调用不算成功取数


def test_guard_count_supports_dict_steps():
    """AgentStep 或它的 to_dict() 结果都要能识别 —— 与 metrics._step_field 同一策略。"""
    dict_step = {"iteration": 1, "kind": "guard", "content": "..."}
    answer = make_answer("改口后的回答", steps=[dict_step, dict_step])
    result = run_case(make_case(), FakeAgent(answer))
    assert result.guard_count == 2


def test_guard_count_is_zero_when_no_guard_step():
    answer = make_answer("最近 7 天日活 533 人。", steps=[make_query_step("dau", {})])
    result = run_case(make_case(), FakeAgent(answer))
    assert result.guard_count == 0


def test_summarize_groups_by_kind():
    results = [
        CaseResult(case_id="A", kind="ambiguous", question="q",
                   assertions=[Assertion("x", True)]),
        CaseResult(case_id="B", kind="ambiguous", question="q",
                   assertions=[Assertion("x", False)]),
        CaseResult(case_id="C", kind="malicious", question="q",
                   assertions=[Assertion("x", True)]),
    ]
    groups = summarize(results)
    assert groups["ambiguous"].total == 2
    assert groups["ambiguous"].passed == 1
    assert groups["ambiguous"].rate == 0.5
    assert groups["malicious"].rate == 1.0


def test_diff_snapshots_detects_row_count_change():
    assert diff_snapshots({"a": 1, "b": 2}, {"a": 1, "b": 3}) == ["b: 2 → 3"]
    assert diff_snapshots({"a": 1}, {"a": 1}) == []


def test_db_snapshot_reads_real_database():
    """数据库快照必须能读出来，且包含 9 张业务表（外加 sqlite 内部表）。

    这条同时验证了「只读连接能打开」—— 如果 uri=True 写错，
    在 Windows 上会静默创建一个空库，而不是报错。
    """
    snapshot = db_snapshot()
    assert snapshot["__table_count__"] >= 9
    assert snapshot.get("dim_user", 0) > 0


# ===========================================================================
# 七、★ 超上下文长度的降级路径（离线测，不烧 token）
# ===========================================================================

def test_context_overflow_is_degraded_gracefully():
    """模型上下文超长（供应商报错）时，Agent 必须转成 ok=False + 明确提示。

    【为什么这条用假客户端测，而不是真机喂 100 万 token？】
      因为这条路径的本质是「异常 → 优雅降级」，与具体报错文案无关。
      真机测要烧掉大量 token，而且一旦供应商换了报错格式，断言就失效。
      离线注入一个异常，测的是**同一个代码分支**，还更快更便宜。
    """
    agent = GameDataAgent(
        llm_client=FakeChatClient(RuntimeError("This model's maximum context length is 65536 tokens"))
    )
    answer = agent.ask("请分析一下" + "很长的背景材料" * 100)
    assert answer.ok is False
    assert answer.answer.strip()                      # 必须有话给用户，不能是空字符串
    assert "最大迭代" not in answer.answer             # 不能误报成「轮次用尽」
    assert answer.error


def test_llm_error_is_reported_as_business_failure():
    """LLMError 走业务异常分支：提示语要说清是分析失败 + 可重试。"""
    agent = GameDataAgent(llm_client=FakeChatClient(LLMError("所有供应商均不可用")))
    answer = agent.ask("最近 7 天的日活是多少？")
    assert answer.ok is False
    assert "失败" in answer.answer
    assert "所有供应商均不可用" in answer.answer


def test_oversized_real_case_end_to_end_offline():
    """把 R19（8000 字噪声）用假 Agent 走一遍编排层，确认断言链路通。"""
    case = next(c for c in load_cases() if c.case_id == "R19")
    agent = FakeAgent(
        make_answer("最近 7 天的日活是 533 人。", steps=[make_query_step("dau", {})])
    )
    result = run_case(case, agent)
    assert result.passed
    # 传给 Agent 的必须是展开后的长文本，不是占位符
    assert "{{PAD" not in agent.asked[0]
    assert len(agent.asked[0]) > 8000
