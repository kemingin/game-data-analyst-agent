# -*- coding: utf-8 -*-
"""
答案数字溯源原语单元测试（Phase 9）
================================================================================
【这个测试文件测什么、不测什么？】

  抽数字、比数字这些**原语**的细致行为，已经在 tests/test_eval_metrics.py 里
  被真实数据反复打磨过了（四档分类、容差、推导池……），这里不重复。

  本文件只测两件在 Phase 9 才出现的东西：

    1. 从执行轨迹里收集参考值的两个函数（新写的）；
    2. 分级处置（pass / annotate / block）的判定与边界。

  另外补一条**同源校验**：运行时防线和评测台必须用的是同一份实现。
  这条断言看起来像废话，但它守的正是这次改造的全部意义 ——
  两边各写一套，就会出现「运行时拦了、评测说没问题」这种自相矛盾的结论。

【测试纪律：正反两面都要有】
  「该判成有出处」和「该判成无出处」都要断言，
  否则一个「永远返回 block」的实现也能让正面用例全绿。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.grounding import (
    UNGROUNDED_RATIO_LIMIT,
    check_answer_numbers,
    collect_ground_truth_params,
    collect_ground_truth_values,
    extract_numbers,
    grounding_decision,
    unexplained_ratio,
)

# ===========================================================================
# 一、测试替身：两种形态的执行轨迹
# ===========================================================================
#
# 为什么要造两份？因为真实世界里轨迹有两种形态：
#   · 刚跑完的会话里是 AgentStep **对象**；
#   · 存盘后重新读出来是它的 to_dict() **字典**（离线复算历史会话）。
# 两处都必须能工作 —— 这是 src/grounding.py 里 _step_field 存在的原因。


@dataclass
class FakeStep:
    """AgentStep 的最小替身：只保留溯源要读的四个字段。"""

    kind: str
    tool_name: str = ""
    ok: bool = True
    result: dict = field(default_factory=dict)


def as_dict(step: FakeStep) -> dict:
    """把替身转成 to_dict() 之后的形态。"""
    return {
        "kind": step.kind,
        "tool_name": step.tool_name,
        "ok": step.ok,
        "result": step.result,
    }


def query_step(rows: list[dict], params: dict | None = None, ok: bool = True) -> FakeStep:
    """构造一个 query_metric 步骤（默认成功）。"""
    return FakeStep(
        kind="tool_call",
        tool_name="query_metric",
        ok=ok,
        result={"rows": rows, "params": params or {}},
    )


# ===========================================================================
# 二、从执行轨迹里收集参考值
# ===========================================================================

def test_collect_ground_truth_values_reads_result_rows():
    """成功查询返回的行里，每个数值都要进参考集。"""
    steps = [query_step([{"dau": 533}, {"dau": 540}])]

    assert collect_ground_truth_values(steps) == [533.0, 540.0]


def test_collect_ground_truth_values_skips_failed_queries():
    """★ 失败的工具调用不算「取到过数」。

    判据必须是 ok 而不是「调用过」：模型可能调了工具但参数非法被拦下，
    那种情况下它手上根本没有数据，防线依然应该生效。
    """
    steps = [query_step([{"dau": 999999}], ok=False)]

    assert collect_ground_truth_values(steps) == []


def test_collect_ground_truth_values_ignores_other_tools():
    """list_metrics / get_metric_detail 的返回不算数据来源。"""
    steps = [
        FakeStep(kind="tool_call", tool_name="list_metrics", result={"rows": [{"x": 123}]}),
        FakeStep(kind="final_answer", result={"rows": [{"x": 456}]}),
    ]

    assert collect_ground_truth_values(steps) == []


def test_collect_ground_truth_values_supports_dict_steps():
    """存盘后读出来的字典轨迹也要能算 —— 否则离线复算会全军覆没。"""
    steps = [as_dict(query_step([{"dau": 533}]))]

    assert collect_ground_truth_values(steps) == [533.0]


def test_collect_ground_truth_values_handles_none_and_garbage():
    """没有轨迹、result 不是字典、rows 缺失 —— 都不能抛异常。"""
    assert collect_ground_truth_values(None) == []
    assert collect_ground_truth_values([]) == []
    assert collect_ground_truth_values([FakeStep(kind="tool_call", tool_name="query_metric", result={})]) == []
    # result 是字符串（历史数据格式坏掉）时也要安全跳过
    assert collect_ground_truth_values(
        [FakeStep(kind="tool_call", tool_name="query_metric", result="坏数据")]
    ) == []


def test_collect_ground_truth_params_takes_numeric_params_only():
    """数值型参数（day_n=7）要收，「最近 7 天」的 7 才有出处。

    字符串参数（日期）不能混进来 —— 它们不是数字，float() 也转不动。
    """
    steps = [query_step([{"dau": 533}], params={"day_n": 7, "start_date": "2026-09-11"})]

    assert collect_ground_truth_params(steps) == [7.0]


def test_collect_ground_truth_params_skips_booleans():
    """bool 是 int 的子类，必须显式排除 —— 否则 True 会变成 1.0 混进参考集。"""
    steps = [query_step([{"dau": 533}], params={"include_inactive": True, "day_n": 7})]

    assert collect_ground_truth_params(steps) == [7.0]


# ===========================================================================
# 三、分级处置
# ===========================================================================

def test_unexplained_ratio_is_zero_without_numbers():
    """没有数字 = 没有风险，返回 0.0 而不是 1.0（也不能是除零错误）。"""
    check = check_answer_numbers("这次查询没有返回数据。", [])

    assert check.total == 0
    assert unexplained_ratio(check) == 0.0


@pytest.mark.parametrize(
    "answer, reference, expected",
    [
        # 全部有出处 → 通过
        ("日活 533 人。", [533.0], "pass"),
        # 没有数字 → 通过（拒答、寒暄都属于这一类）
        ("抱歉，这个范围我查不到。", [533.0], "pass"),
        # 3 个数字里 1 个无出处（1/3）→ 标注（边界值，取「不超过」）
        ("日活 533 人、540 人，另有 3 项缺失。", [533.0, 540.0], "annotate"),
        # 2 个数字里 1 个无出处（1/2 > 1/3）→ 拦截
        ("日活 533 人，留存 42.66%。", [533.0], "block"),
    ],
)
def test_grounding_decision_three_branches(answer, reference, expected):
    """三档判定 + 边界值。1/3 这一档取「不超过」，所以刚好 1/3 走标注。"""
    check = check_answer_numbers(answer, reference)

    assert grounding_decision(check) == expected


def test_grounding_decision_boundary_is_inclusive():
    """把边界单独钉一次：占比恰好等于阈值时是 annotate 而不是 block。

    为什么值得单测？因为「>」和「>=」写错一个符号，
    线上表现是「本来只是提示，变成了直接拒答」—— 用户能直接感知到。
    """
    # 4 个数字里 1 个无出处 = 0.25，2 个 = 0.5；构造恰好 1/3 需要 3 个数字 1 个无出处
    check = check_answer_numbers("日活 533 人、540 人，另有 3 项缺失。", [533.0, 540.0])

    assert check.total == 3
    assert len(check.unexplained) == 1
    assert unexplained_ratio(check) == pytest.approx(UNGROUNDED_RATIO_LIMIT)
    assert grounding_decision(check) == "annotate"


def test_no_successful_query_still_blocks():
    """★ 向后兼容：一次成功取数都没有时，仍然走拦截（与旧判据一致）。

    这是「宁可说查不到，也不能编」这条产品原则的底线，
    改造判据时最不能碰坏的就是它。
    """
    steps = [query_step([{"dau": 533}], ok=False)]     # 调了，但没取到
    check = check_answer_numbers(
        "最近 7 天的日活是 100,000 人。",
        collect_ground_truth_values(steps),
        extra_allowed=collect_ground_truth_params(steps),
    )

    assert check.total == 1
    assert grounding_decision(check) == "block"


# ===========================================================================
# 四、端到端：新判据相对旧判据的增量
# ===========================================================================

def test_mixed_answer_with_successful_query_is_blocked():
    """★ 本次改造要解决的核心场景。

    旧判据是「一次都没成功取数 且 有数字」—— 只要有过一次成功取数就整条放行。
    于是「查了 DAU、又顺手编了留存率和付费率」能大摇大摆通过。

    这里显式构造那条路径：参考集**非空**（确实取到过数），
    但答案里混着两个编造值 → 必须判 block。
    """
    steps = [query_step([{"dau": 533}])]
    answer = "日活 533 人，次日留存 42.66%，付费率 3.85%。"

    check = check_answer_numbers(
        answer,
        collect_ground_truth_values(steps),
        extra_allowed=collect_ground_truth_params(steps),
    )

    assert check.total == 3
    assert check.matched_count == 1                    # 只有 533 有出处
    assert sorted(item.raw for item in check.unexplained) == ["3.85", "42.66"]
    assert grounding_decision(check) == "block"


def test_derived_analysis_is_not_punished():
    """反面证据：会分析的回答不能被判成编造。

    均值、涨幅、极差这些数字在结果集里根本不存在，但完全正确。
    如果不把它们算作「有出处」，模型越会分析、被判编造的数字越多 ——
    这个指标就变成了「惩罚分析能力」。
    """
    steps = [query_step([{"dau": 465}, {"dau": 533}])]
    # 均值 499、极差 68、涨幅 (533-465)/465*100 = 14.62%
    answer = "两天日活为 465 和 533 人，均值 499 人，极差 68 人，环比涨了 14.62%。"

    check = check_answer_numbers(
        answer,
        collect_ground_truth_values(steps),
        extra_allowed=collect_ground_truth_params(steps),
    )

    assert check.unexplained == []
    assert grounding_decision(check) == "pass"


def test_window_days_come_from_params_not_rows():
    """「最近 7 天」的 7 来自参数而不是结果集 —— 不纳入就会制造假警报。"""
    steps = [query_step([{"dau": 533}], params={"day_n": 7})]
    answer = "最近 7 天的日活是 533 人。"

    check = check_answer_numbers(
        answer,
        collect_ground_truth_values(steps),
        extra_allowed=collect_ground_truth_params(steps),
    )

    # 7 被后缀规则（「7 天」）过滤掉了，所以这里只有 533 一个数字
    assert check.total == 1
    assert grounding_decision(check) == "pass"


# ===========================================================================
# 五、回归：日期识别漏网会把正确拒答拦掉
# ===========================================================================

def test_year_month_dates_are_not_treated_as_numbers():
    """★ 回归测试：只写年月的日期（2025-12）必须整体抠掉。

    踩坑经过：日期规则里原本只有「年-月-日」和「月-日」两种短横线写法，
    于是「2025-12」被后者从中间咬掉一段，匹配成「25-12」，
    剩下一个孤零零的「20」被当成数据值。

    后果不是小瑕疵：超范围拒答文案里天然会写「去年 12 月（2025-12）」，
    而拒答场景下参考集是空的 → 占比 1.0 → **一条完全正确的拒答被拦成编造**，
    超范围拒答率会直接掉下来。这是把 extract_numbers 接进运行时防线时
    被拒答用例当场抓出来的。
    """
    refusal = "我的数据只覆盖 2026-06-20 ~ 2026-09-17，去年 12 月（2025-12）超出了范围。"

    assert extract_numbers(refusal) == []


def test_year_month_slash_and_full_dates_still_work():
    """正反两面：抠掉年月之后，真正的数据值不能被连带误伤。"""
    text = "2025-12 上线，2026/09 复测，日活 533 人，留存 42.66%。"

    assert [n.value for n in extract_numbers(text)] == [533.0, 42.66]


# ===========================================================================
# 六、同源校验：两边必须共用一份实现
# ===========================================================================

def test_runtime_and_eval_share_the_same_primitives():
    """★ 运行时防线与评测台用的必须是**同一个函数对象**。

    这不是形式主义：如果两边各维护一套抽数字规则，
    就会出现「运行时把这条答案拦了，评测台却说数字全部有出处」——
    同一份输出得出两个互相矛盾的结论，整个评测体系的可信度就没了。
    同源是这次把原语下沉到 src/grounding.py 的**唯一理由**。
    """
    import src.eval.metrics as eval_metrics
    from src.grounding import NumberCheck

    assert eval_metrics.extract_numbers is extract_numbers
    assert eval_metrics.check_answer_numbers is check_answer_numbers
    assert eval_metrics.NumberCheck is NumberCheck


def test_grounding_module_does_not_depend_on_agent_or_eval():
    """★ 中立性校验：这个模块不许反向依赖 agent / eval 层。

    为什么必须钉住？因为 src/eval/__init__.py 会 import runner，
    而 runner 又 import agent —— 一旦 grounding 反过来 import 它们，
    就会形成循环导入，表现为「单独跑测试没事、app.py 一启动就 ImportError」。
    这条断言让违规在单元测试阶段就暴露，而不是等到手工点开界面。
    """
    import src.grounding as grounding

    # 只看真正的 import 语句，避免注释里提到这些模块名就误报
    import_lines = [
        line
        for line in Path(grounding.__file__).read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
        and ("src.agent" in line or "src.eval" in line)
    ]

    assert import_lines == []
