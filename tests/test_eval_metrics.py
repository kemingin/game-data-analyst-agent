# -*- coding: utf-8 -*-
"""
评测体系单元测试（Phase 5）
================================================================================
【评测系统为什么必须被测试？】

  这是最容易出笑话的地方：**一个算分算错的评测系统，比没有评测更糟**——
  它会让你以为模型 95% 准确，实际上可能只有 60%，而且你还拿着这个数字去
  对外讲。所以评测算法本身必须像业务代码一样被严格覆盖。

  好在 Phase 5 的算法全是纯函数（输入 dict / 数字，输出判定结果），
  喂构造数据就能验证，不需要真的调大模型，跑一次不到一秒。

  另外，本文件还顺带承担了一个职责：**校验评测集本身**。
  比如 E07 写的 `@data_end-13` 到底能不能被解析、解析出来对不对，
  在这里会被验证 —— 保证「尺子」本身是准的。
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from src import config as cfg
from src.agent.react_agent import AgentAnswer, AgentStep
from src.eval.eval_set import EvalCase, EvalSet, load_eval_set
from src.eval.metrics import (
    EvalSummary,
    check_answer_numbers,
    check_metric_mapping,
    check_params,
    collect_reference_values,
    extract_numbers,
    format_rate,
    is_close,
    metric_calls,
    summarize,
)
from src.eval.runner import CaseResult, EvalRunner, build_report

# ===========================================================================
# 一、测试替身
# ===========================================================================

def make_step(
    tool_name: str,
    arguments: dict,
    result: dict | None = None,
    ok: bool = True,
) -> AgentStep:
    """造一个工具调用步骤。"""
    return AgentStep(
        iteration=1,
        kind="tool_call",
        tool_name=tool_name,
        arguments=arguments,
        result=result or {},
        ok=ok,
    )


def make_query_step(metric_id: str, params: dict, rows: list[dict] | None = None) -> AgentStep:
    """造一个 query_metric 步骤（result 里带已解析的参数，这是真实结构）。"""
    return make_step(
        "query_metric",
        {"metric_id": metric_id, "params": params},
        result={
            "metric_id": metric_id,
            "params": params,
            "rows": rows or [],
            "columns": list((rows or [{}])[0].keys()),
            "rendered_sql": "SELECT ...",
        },
    )


class FakeAgent:
    """假 Agent：按问题返回预先编好的答案，用于验证评测流程本身。"""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.asked: list[str] = []
        self.histories: list[list | None] = []   # 记录每次调用传进来的前情，供多轮用例断言

    def ask(self, question: str, history: list | None = None):
        self.asked.append(question)
        self.histories.append(history)
        item = self.answers.get(question)
        if isinstance(item, Exception):
            raise item
        return item


# ===========================================================================
# 二、数字抽取
# ===========================================================================

def test_extract_numbers_basic():
    numbers = extract_numbers("最近 7 天日活 465 人，次日留存 42.66%，ARPU 8.46 元。")
    values = [n.value for n in numbers]
    # 「7 天」是时间量词要被剔除，其余三个是数据值
    assert 7 not in values
    assert 465 in values
    assert 42.66 in values
    assert 8.46 in values


def test_extract_numbers_skips_dates():
    """日期必须被整体剔除，否则每条回答都会产生一堆假的对不上。

    这是数字对账最容易踩的坑：2026-09-11 会被拆成 2026 / 09 / 11。
    """
    values = [n.value for n in extract_numbers("统计区间为 2026-09-11 ~ 2026-09-17，共 465 人。")]
    assert 2026 not in values
    assert 9 not in values
    assert 11 not in values
    assert 465 in values


def test_extract_numbers_skips_chinese_dates_and_versions():
    values = [n.value for n in extract_numbers("2026年9月11日上线 v2.0.0 版本，当天日活 512 人。")]
    assert 2026 not in values
    assert 2.0 not in values
    assert 512 in values


def test_extract_numbers_skips_day_index_notations():
    """D7 / 第 30 天 这类口径名不是数据值，要剔除。"""
    values = [n.value for n in extract_numbers("D7 留存为 20.55%，第 30 天留存 10.71%。")]
    assert 7 not in values
    assert 30 not in values
    assert 20.55 in values
    assert 10.71 in values


def test_extract_numbers_handles_thousands_separator_and_negative():
    values = [n.value for n in extract_numbers("总计 1,181,298 行，环比 -3.2%。")]
    assert 1181298 in values
    assert -3.2 in values


def test_extract_numbers_skips_short_dash_dates():
    """09-05 这种省略年份的简写日期也要整体剔除。

    真机评测暴露的问题：模型用 Markdown 表格列日期时写成「09-05」，
    数字正则把它拆成 09 和 -05，-05 被判成"存疑"，纯属误报。
    注意「-3.2%」这种真正的负数不能被误伤。
    """
    values = [n.value for n in extract_numbers("| 09-05 | 99 人 | 09-17 | 54 人 | 环比 -3.2% |")]
    assert 99 in values
    assert 54 in values
    assert -3.2 in values
    assert 5 not in values
    assert -5 not in values


def test_extract_numbers_skips_step_notations_with_space():
    """「第 3 步」「步骤 3」里的序号不是数据值，即使中间带空格也要剔除。

    真机评测暴露的问题：原来的判断只看数字前一个字符，而「卡在第 3 步」
    里数字前面是空格，于是 3 被当成数据值，在 E13 里连报 5 次。
    """
    values = [n.value for n in extract_numbers("主要卡在第 3 步，步骤 3 的流失最严重，整体完成率 44.24%。")]
    assert values == [44.24]


def test_extract_numbers_skips_month_counter():
    """「3 个月」里的 3 是时间量词，不是数据值。"""
    values = [n.value for n in extract_numbers("建议拉长到 3 个月再看，目前完成率 44.24%。")]
    assert values == [44.24]


def test_extract_numbers_empty_input():
    assert extract_numbers(None) == []
    assert extract_numbers("") == []
    assert extract_numbers("这段文字里没有任何数字。") == []


# ===========================================================================
# 三、数字比对
# ===========================================================================

def test_is_close_allows_rounding():
    assert is_close(42.66, 42.66)
    assert is_close(42.7, 42.66)        # 模型四舍五入到一位小数
    assert is_close(465, 465)
    assert not is_close(465, 488)       # 不同天的日活，必须判为不一致
    assert not is_close(42.66, 40.30)   # 版本前 vs 版本后，必须判为不一致


def test_collect_reference_values_skips_non_numeric():
    rows = [
        {"channel_name": "自然量", "cohort_size": 620, "retention_rate_pct": 46.61},
        {"channel_name": "抖音买量", "cohort_size": "410", "retention_rate_pct": None},
    ]
    values = collect_reference_values(rows)
    assert 620 in values
    assert 46.61 in values
    assert 410 in values          # 数字型字符串也要收进来
    assert len(values) == 3


def test_check_answer_numbers_reports_both_sides():
    rows = [{"dau": 465}, {"dau": 488}, {"dau": 512}]
    check = check_answer_numbers("9/11 日活 465 人，9/12 是 488 人，另有 9999 人。", collect_reference_values(rows))
    assert check.total == 3
    assert check.matched_count == 2
    assert [n.value for n in check.unexplained] == [9999]
    assert "9999" in check.describe_unexplained()


def test_check_answer_numbers_classifies_spoken_approximation():
    """口语化近似要归入「合理近似」，不能算存疑。

    真实场景：「稳定在 530 左右」，而真实值是 533。这是人话不是编造。
    第一版规则用 0.5% 的严格容差去卡它，直接判成"对不上"——
    结果 E01 那条最普通的用例只报出 71% 的可追溯率，全是假警报。
    """
    check = check_answer_numbers("最近几天日活稳定在 530 左右。", [533.0])

    assert check.total == 1
    assert len(check.approximate) == 1
    assert check.unexplained == []
    assert check.rate == 1.0


def test_check_answer_numbers_classifies_derived_values():
    """模型自己算出来的衍生值（均值 / 极差 / 涨幅）要归入「可推导」。

    这条是整套规则的立身之本：一个合格的分析回答**必然**包含衍生指标，
    比如「7 天均值 517」「比低点回升约 15%」。
    如果这些都被算成"没出处"，那模型越会分析、得分越低 ——
    指标就变成了在惩罚分析能力，方向完全错了。
    """
    values = [120.0, 260.0, 380.0, 500.0]
    mean = sum(values) / len(values)                             # 315
    swing = (max(values) - min(values)) / min(values) * 100      # 316.7

    check = check_answer_numbers(
        f"均值 {mean:.0f} 人，比最低点回升约 {swing:.0f}%。",
        values,
    )

    assert check.total == 2
    assert len(check.derived) == 2
    assert check.unexplained == []
    assert check.rate == 1.0


def test_check_answer_numbers_prefers_strict_over_derived():
    """判定顺序必须是「先硬后软」：严格 → 近似 → 推导。

    如果反过来先判推导，一个恰好等于某对参考值之差的编造数字
    就会被归进「可推导」桶里 —— 真问题被藏起来，这比误报更危险。
    """
    check = check_answer_numbers("日活 500 人。", [500.0, 1000.0])

    assert len(check.matched) == 1        # 精确命中就该进最硬的那一档
    assert check.derived == []


def test_check_answer_numbers_accepts_single_step_churn_rate():
    """单步流失率 (前-后)/前 也要算「可推导」。

    真机评测 E13：模型写「第 3 步单步就掉了 41.32%」，这是
    (334-196)/334*100。原来推导池只算 (a-b)/b，两个方向里缺了这一个，
    于是一条完全正确的漏斗归因被报成存疑。
    """
    # 漏斗：334 人进入步骤 2，其中 196 人走完步骤 3
    values = [334.0, 196.0]
    churn = (334 - 196) / 334 * 100        # 41.32

    check = check_answer_numbers(f"第 3 步单步就掉了 {churn:.2f}%。", values)

    assert check.total == 1
    assert len(check.derived) == 1
    assert check.unexplained == []


def test_definition_numbers_are_allowed_as_source():
    """指标定义里的数字（如「行业参考区间 10%~25%」）应被放行。

    模型引用口径说明是正常行为，不该被反复报成「存疑」——
    误报多了，复核的人就不再认真看这个指标了。
    """
    from src.eval.runner import _definition_numbers

    numbers = _definition_numbers("stickiness")
    assert 25 in numbers or 10 in numbers
    # 不存在的指标不能把整条评测带崩
    assert _definition_numbers("not_a_metric") == []
    assert _definition_numbers(None) == []


def test_check_answer_numbers_rate_is_none_without_numbers():
    """回答里没有数字时，一致率应为「不适用」而不是 0%。

    「没有可测样本」和「测了全错」是完全不同的结论，不能混为一谈。
    """
    check = check_answer_numbers("这次查询没有返回数据。", [])
    assert check.total == 0
    assert check.rate is None


def test_check_answer_numbers_extra_allowed_reduces_false_alarm():
    """期望参数值和行数应被允许 —— 否则「最近 14 天」「共 7 行」都会被算成对不上。"""
    rows = [{"dau": 465}]
    check = check_answer_numbers(
        "最近 14 天共 1 天有数据，日活 465 人。",
        collect_reference_values(rows),
        extra_allowed=[14, 1],
    )
    assert check.unexplained == []


# ===========================================================================
# 四、轨迹解析
# ===========================================================================

def test_metric_calls_extracts_only_query_metric():
    steps = [
        make_step("list_metrics", {}),
        make_query_step("dau", {"start_date": "2026-09-11", "end_date": "2026-09-17"}),
        make_step("get_metric_detail", {"metric_id": "dau"}),
        AgentStep(iteration=2, kind="final_answer", content="回答"),
    ]
    calls = metric_calls(steps)
    assert len(calls) == 1
    assert calls[0]["metric_id"] == "dau"
    assert calls[0]["params"]["start_date"] == "2026-09-11"


def test_metric_calls_prefers_resolved_params():
    """参数要优先取 result 里已解析的真实取值，而不是模型传的表达式。

    因为要拿真实日期去和期望值比对，'@data_end-13' 这种表达式没法直接比。
    """
    step = make_step(
        "query_metric",
        {"metric_id": "dau", "params": {"start_date": "@data_end-13"}},
        result={"metric_id": "dau", "params": {"start_date": "2026-09-04", "end_date": "2026-09-17"}},
    )
    assert metric_calls([step])[0]["params"]["start_date"] == "2026-09-04"


def test_metric_calls_works_on_serialized_dicts():
    """也要能吃 to_dict() 之后的结果 —— 支持对存盘的历史评测离线复算。"""
    step = make_query_step("dau", {"start_date": "2026-09-11"})
    calls = metric_calls([step.to_dict()])
    assert calls[0]["metric_id"] == "dau"


def test_metric_calls_handles_empty():
    assert metric_calls(None) == []
    assert metric_calls([]) == []


# ===========================================================================
# 五、三项判定
# ===========================================================================

def test_check_metric_mapping():
    steps = [make_query_step("retention_rate", {"day_n": 1})]
    assert check_metric_mapping(steps, "retention_rate") is True
    assert check_metric_mapping(steps, "dau") is False
    # 不声明期望指标 → 不适用（None），不计入分母
    assert check_metric_mapping(steps, None) is None


def test_check_metric_mapping_allows_extra_queries():
    """多查了几个指标不该被判错 —— 交叉验证是合理行为。"""
    steps = [
        make_query_step("dau", {}),
        make_query_step("retention_rate", {"day_n": 1}),
    ]
    assert check_metric_mapping(steps, "retention_rate") is True


def test_check_params_pass_and_normalization():
    """int 1 和字符串 "1" 要视为相同，否则会假失败。"""
    steps = [make_query_step("retention_rate", {"day_n": 1})]
    ok, detail = check_params(steps, "retention_rate", {"day_n": "1"})
    assert ok is True
    assert detail["day_n"]["ok"] is True


def test_check_params_fail_and_report_detail():
    steps = [make_query_step("new_user_count", {"start_date": "2026-09-11", "end_date": "2026-09-17"})]
    ok, detail = check_params(
        steps, "new_user_count", {"start_date": "2026-09-04", "end_date": "2026-09-17"}
    )
    assert ok is False
    assert detail["start_date"]["ok"] is False
    assert detail["start_date"]["expected"] == "2026-09-04"
    assert detail["start_date"]["actual"] == "2026-09-11"
    assert detail["end_date"]["ok"] is True


def test_check_params_skips_undeclared_keys():
    """只校验显式声明的键 —— 默认值本来就该由系统兜底，模型不传才是对的。"""
    steps = [make_query_step("dau", {"start_date": "2026-09-11", "end_date": "2026-09-17"})]
    ok, detail = check_params(steps, "dau", {"start_date": "2026-09-11"})
    assert ok is True
    assert set(detail) == {"start_date"}


def test_check_params_not_applicable_without_expectation():
    steps = [make_query_step("dau", {})]
    assert check_params(steps, "dau", {}) == (None, {})
    assert check_params(steps, None, {"day_n": 1}) == (None, {})


def test_check_params_fails_when_metric_never_queried():
    """指标都没查对时，参数判定应为 False 而不是 None —— 这是"确实抽错了"，
    不该被排除在分母之外（否则选错指标会同时"豁免"参数评分）。"""
    ok, detail = check_params([make_query_step("dau", {})], "retention_rate", {"day_n": 1})
    assert ok is False
    assert detail["day_n"]["actual"] is None


def test_check_params_passes_if_any_call_matches():
    """多查一个口径做对比，不该被判错 —— 评测要惩罚的是漏查，不是多查。

    真机评测 E15：问「v2.0.0 上线后留存变好了吗」，模型先查 D1 又查 D7 对比，
    最后一次是 D7。按"只看最后一次"的口径，它会被判成参数抽错，
    可它明明查对了 D1 还多做了功课。
    """
    steps = [
        make_query_step("version_retention", {"version_id": "v2.0.0", "day_n": 1}),
        make_query_step("version_retention", {"version_id": "v2.0.0", "day_n": 7}),
    ]
    ok, detail = check_params(steps, "version_retention", {"version_id": "v2.0.0", "day_n": 1})
    assert ok is True
    assert detail["day_n"]["actual"] == 1


def test_check_params_reports_last_call_when_none_match():
    """全都没命中时，明细报最后一次调用的值 —— 人工复核时最近一次最有用。"""
    steps = [
        make_query_step("retention_rate", {"day_n": 3}),
        make_query_step("retention_rate", {"day_n": 14}),
    ]
    ok, detail = check_params(steps, "retention_rate", {"day_n": 7})
    assert ok is False
    assert detail["day_n"]["actual"] == 14


# ===========================================================================
# 六、汇总
# ===========================================================================

def make_result(
    case_id: str,
    mapping: bool | None = None,
    params: bool | None = None,
    numbers: tuple[int, int] = (0, 0),
    tokens: int = 1000,
    expect_no_query: bool = False,
    refuse_ok: bool = False,
) -> CaseResult:
    from src.eval.metrics import NumberCheck

    check = NumberCheck(total=numbers[1])
    check.matched = [(0.0, 0.0)] * numbers[0]
    return CaseResult(
        case_id=case_id,
        question=f"问题 {case_id}",
        metric_mapping_ok=mapping,
        params_ok=params,
        number_check=check,
        expect_no_query=expect_no_query,
        refuse_ok=refuse_ok,
        usage={"total_tokens": tokens},
        iterations=2,
        elapsed_ms=1000.0,
    )


def test_summarize_computes_rates():
    results = [
        make_result("A", mapping=True, params=True, numbers=(3, 3), tokens=5000),
        make_result("B", mapping=True, params=False, numbers=(2, 3), tokens=7000),
        make_result("C", mapping=False, params=None, numbers=(1, 2), tokens=3000),
    ]
    summary = summarize(results)

    assert summary.total == 3
    assert (summary.mapping_correct, summary.mapping_checked) == (2, 3)
    assert summary.metric_mapping_rate == pytest.approx(0.6667, abs=1e-4)
    # 参数只有两条可测（C 没声明期望参数）
    assert (summary.param_correct, summary.param_checked) == (1, 2)
    assert summary.param_rate == 0.5
    # 数字对账：对上 6 个 / 共 8 个
    assert (summary.number_matched, summary.number_total) == (6, 8)
    assert summary.number_rate == 0.75
    assert summary.total_tokens == 15000
    assert summary.avg_tokens == 5000.0


def test_summarize_cache_hit_rate():
    """Phase 6：缓存命中率 = 命中 token / 输入 token（分母是输入侧，不是总 token）。

    这里刻意构造「命中 800 / 输入 2000」的用例，如果实现误用 total_tokens
    当分母（比如写成 800/2500），断言会失败。
    """
    results = [
        CaseResult(case_id="A", question="q1", usage={
            "total_tokens": 1300, "prompt_tokens": 1000, "cached_tokens": 500,
        }),
        CaseResult(case_id="B", question="q2", usage={
            "total_tokens": 1300, "prompt_tokens": 1000, "cached_tokens": 300,
        }),
    ]
    summary = summarize(results)

    assert summary.total_prompt_tokens == 2000
    assert summary.total_cached_tokens == 800
    assert summary.cache_hit_rate == 0.4
    assert summary.avg_cached_tokens == 400.0


def test_cache_hit_rate_is_none_when_not_reported():
    """供应商不上报缓存字段时，命中率是「不适用」而不是 0% —— 数据缺失 ≠ 真的没命中。"""
    results = [CaseResult(case_id="A", question="q", usage={"total_tokens": 100})]
    summary = summarize(results)

    assert summary.total_cached_tokens == 0
    assert summary.cache_hit_rate is None


def test_summarize_rate_is_none_not_zero_when_no_samples():
    """没有可测样本时必须返回 None。

    如果返回 0%，报告会显示「指标映射准确率 0.00%」—— 看起来像模型完全失效，
    实际上只是评测集里没有这类用例。这个区别会误导所有看报告的人。
    """
    summary = summarize([make_result("A", mapping=None, params=None, numbers=(0, 0))])
    assert summary.metric_mapping_rate is None
    assert summary.param_rate is None
    assert summary.number_rate is None


def test_summarize_refuse_cases():
    results = [
        make_result("E17", expect_no_query=True, refuse_ok=True),
        make_result("E18", expect_no_query=True, refuse_ok=False),
    ]
    summary = summarize(results)
    assert (summary.refused_correctly, summary.refuse_checked) == (1, 2)
    assert summary.refuse_rate == 0.5


def test_summarize_empty():
    summary = summarize([])
    assert summary.total == 0
    assert summary.avg_tokens == 0.0
    assert summary.metric_mapping_rate is None


def test_format_rate():
    assert format_rate(0.8567) == "85.67%"
    assert format_rate(0.0) == "0.00%"
    assert format_rate(None) == "不适用"


# ===========================================================================
# 七、评测集本身（校验"尺子"是准的）
# ===========================================================================

def test_eval_set_loads_all_cases():
    eval_set = load_eval_set()
    assert len(eval_set) == 31
    assert eval_set.meta["version"] == "1.1"
    # 每条用例都必须有唯一 ID 和非空问题
    assert len({c.case_id for c in eval_set.all()}) == 31
    assert all(c.question.strip() for c in eval_set.all())


def test_eval_set_v11_expansion_keeps_categories_balanced():
    """v1.1 扩充后，四类用例（长尾问法 / 多轮追问 / 边界 / 常规）都得有。

    为什么要专门守这条？因为扩充评测集时最容易犯的错是**只往一个方向加**：
    加一堆长尾问法，看着用例数涨了，但多轮和边界能力其实没被覆盖。
    用一条断言把「覆盖结构」钉住，比人肉 review 靠谱。
    """
    counts = load_eval_set().by_category()
    for category in ("活跃", "留存", "增长", "商业化", "渠道", "版本", "长尾问法", "多轮追问", "边界"):
        assert counts.get(category, 0) > 0, f"分类「{category}」没有用例"


def test_eval_set_history_field_is_loaded():
    """多轮追问用例的前情对话必须被正确加载成 OpenAI 消息格式。

    前情是这一批用例的**唯一新增变量**：如果 history 没被加载，
    追问句（「那 v1.9.0 呢？」）会变成一个没有指代对象的孤立问题，
    模型只能瞎猜 —— 那样测出来的分数是假的，不是模型不行，是尺子没装好。
    """
    case = load_eval_set().get("E29")
    assert len(case.history) == 2
    assert case.history[0]["role"] == "user"
    assert case.history[0]["content"] == "v2.0.0 上线后新用户留存变好了吗？"
    assert case.history[1]["role"] == "assistant"

    # 常规单轮用例的 history 必须是空元组（而不是 None），
    # 这样 runner 里 list(case.history) 永远安全，不用写 if 判空
    assert load_eval_set().get("E01").history == ()


def test_eval_set_multiturn_cases_still_declare_expected_metric():
    """多轮用例同样要声明期望指标 —— 否则「指代消解」这个能力无法被评分。

    追问句里刻意不含指标名（这正是考点），所以期望值只能靠用例声明；
    如果漏写，评测会把它当成「不校验映射」的模糊提问，白白放跑一个考点。
    """
    eval_set = load_eval_set()
    multiturn = [c for c in eval_set.all() if c.history]
    assert len(multiturn) == 3
    for case in multiturn:
        assert case.expected_metric_id, f"{case.case_id} 是多轮用例但没声明期望指标"


def test_eval_set_dangerous_operation_case_blocks_query():
    """危险操作（删数据）必须归类为「不查数直接拒答」。

    这条守的是项目第二条硬约束（SQL 仅允许 SELECT）。
    如果这条用例被误配成普通用例，runner 会去给它算参考值，
    评测本身反而变成了"教模型去查数"。
    """
    case = load_eval_set().get("E30")
    assert case.expect_no_query is True
    assert case.resolved_reference_metric_id() is None


def test_runner_passes_history_to_agent_for_multiturn_cases():
    """runner 必须把多轮用例的前情透传给 Agent。

    这是最容易漏的一环：JSON 里写了 history、EvalCase 也加载了，
    但只要 runner 忘了传，多轮用例就会退化成「单轮孤立提问」——
    模型答不好，评测却会把责任算在模型头上。**尺子没装好，不能怪被测对象。**
    """
    eval_set = load_eval_set()
    case = eval_set.get("E27")
    agent = FakeAgent(
        {case.question: AgentAnswer(question=case.question, answer="已取到。", ok=True, steps=[])}
    )
    runner = EvalRunner(agent=agent, eval_set=eval_set)
    runner.run_case(case)

    assert agent.histories == [[
        {"role": "user", "content": "最近 7 天的日活是多少？"},
        {"role": "assistant", "content": "已取到最近 7 天的日活逐日数据，整体波动不大。"},
    ]]


def test_runner_passes_none_history_for_single_turn_cases():
    """单轮用例必须传 None（而不是空列表）—— 空列表会让 Agent 走进
    「这是多轮对话」的分支，多拼一段空的上下文，白白浪费 token。"""
    eval_set = load_eval_set()
    case = eval_set.get("E01")
    agent = FakeAgent(
        {case.question: AgentAnswer(question=case.question, answer="已取到。", ok=True, steps=[])}
    )
    runner = EvalRunner(agent=agent, eval_set=eval_set)
    runner.run_case(case)

    assert agent.histories == [None]


def test_eval_set_covers_every_metric():
    """评测集要覆盖全部 12 个指标 —— 否则准确率再高也只证明了部分能力。

    这条断言的价值：以后往注册表里加了新指标，如果忘了补评测用例，
    测试会直接失败提醒你。**让测试守住评测集的完整性**，
    比靠自觉维护靠谱得多。
    """
    from src.metrics.registry import get_registry

    covered = {c.expected_metric_id for c in load_eval_set().all() if c.expected_metric_id}
    all_metrics = set(get_registry().ids())
    assert all_metrics - covered == set(), f"以下指标没有评测用例：{all_metrics - covered}"


def test_eval_set_resolves_relative_date_expressions():
    """期望参数里的 @data_end-13 必须被解析成真实日期。

    验证方式：用 config 里的 DATA_END 反推期望值，而不是写死 '2026-09-04' ——
    这样即使将来数据窗口变了，测试依然成立（测的是"解析逻辑对不对"，
    不是"今天日期是多少"）。
    """
    expected_start = (cfg.DATA_END - timedelta(days=13)).isoformat()
    case = load_eval_set().get("E07")
    assert case.expected_params["start_date"] == expected_start
    assert case.expected_params["end_date"] == cfg.DATA_END.isoformat()
    # 参考参数应与期望参数一致（runner 靠它算真值）
    assert case.reference_params == case.expected_params


def test_eval_set_does_not_inflate_params_with_defaults():
    """没有声明期望参数的用例，解析后必须仍为空 —— 否则会把默认值也纳入校验，
    惩罚"模型不传默认值"这个正确行为。"""
    case = load_eval_set().get("E01")
    assert case.expected_params == {}


def test_eval_set_out_of_scope_cases_have_no_reference():
    eval_set = load_eval_set()
    for case_id in ("E17", "E18", "E30", "E31"):
        case = eval_set.get(case_id)
        assert case.expect_no_query is True
        assert case.resolved_reference_metric_id() is None


def test_eval_set_rejects_duplicate_ids(tmp_path):
    import json

    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            {
                "meta": {},
                "cases": [
                    {"case_id": "X1", "question": "问题一"},
                    {"case_id": "X1", "question": "问题二"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复的 case_id"):
        EvalSet(path)


def test_eval_set_rejects_invalid_params_at_load_time(tmp_path):
    """参数写错的用例要在**加载时**就报错，而不是等跑完 20 条真机评测才发现。

    早失败是评测系统该有的品质 —— 一条废用例会让整轮评测的分数失去意义。
    """
    import json

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "meta": {},
                "cases": [
                    {
                        "case_id": "X1",
                        "question": "问题",
                        "expected_metric_id": "retention_rate",
                        "expected_params": {"day_n": 999},   # 不在枚举范围内
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        EvalSet(path)


# ===========================================================================
# 八、执行器与报告
# ===========================================================================

def test_runner_computes_reference_independently():
    """真值必须来自独立路径（直接查库），而不是 Agent 自己查出来的数据。"""
    runner = EvalRunner(eval_set=load_eval_set())
    case = runner.eval_set.get("E01")
    rows, params = runner.build_reference(case)

    assert len(rows) == 7                      # 默认最近 7 天
    assert params["start_date"] == (cfg.DATA_END - timedelta(days=6)).isoformat()
    assert params["end_date"] == cfg.DATA_END.isoformat()


def test_runner_run_case_scores_a_correct_answer():
    """构造一个"完全答对"的 Agent，验证评测器给满分。

    这是评测器的自检：如果连一个明显正确的回答都判不及格，
    那说明尺子坏了，而不是模型不行。
    """
    from src.agent.react_agent import AgentAnswer
    from src.eval.metrics import NumberCheck

    case = load_eval_set().get("E04")
    reference_rows, _ = EvalRunner(eval_set=load_eval_set()).build_reference(case)
    rate = reference_rows[0]["retention_rate_pct"]
    cohort = reference_rows[0]["cohort_size"]

    answer = AgentAnswer(
        question=case.question,
        answer=f"次日留存率为 {rate}%，样本量 {cohort} 人。",
        ok=True,
        steps=[make_query_step("retention_rate", {"day_n": 1}, rows=reference_rows)],
        iterations=2,
        usage={"total_tokens": 5000},
    )
    runner = EvalRunner(agent=FakeAgent({case.question: answer}), eval_set=load_eval_set())
    result = runner.run_case(case)

    assert result.ok is True
    assert result.metric_mapping_ok is True
    assert result.params_ok is True
    assert isinstance(result.number_check, NumberCheck)
    assert result.number_check.rate == 1.0


def test_runner_traces_numbers_from_self_queried_metrics():
    """模型主动多查的指标，其数字也应算「可追溯」（Phase 6 修正的误报）。

    场景来自真机：被问「付费转化情况怎么样」时，模型会顺手多查 arppu / arpu
    来做交叉验证。那些数字全部来自真实工具返回，但原本的对账集只由
    期望指标（payment_rate）算出来，于是被报成「无出处」—— 典型的误报。

    修正后：对账集会走**同样的独立查询路径**，用模型自己声明的
    metric_id + params 再查一次库。注意这**没有放宽「数字必须来自数据库」**，
    只是把「允许哪些指标的数字参与对账」放宽到了模型实际查过的范围。
    """
    eval_set = load_eval_set()
    case = eval_set.get("E09")          # 期望指标 payment_rate
    runner = EvalRunner(eval_set=eval_set)
    payment_rows, _ = runner.build_reference(case)

    # 另查一个真实存在的指标，拿到它的真值（模拟模型"顺手多查"）
    arpu_rows, _ = runner.build_reference(
        EvalCase(case_id="tmp", question="", reference_metric_id="arpu")
    )
    assert arpu_rows, "arpu 参考数据应能查到，否则这条测试失去意义"
    arpu_value = arpu_rows[0]["arpu"]

    answer = AgentAnswer(
        question=case.question,
        answer=f"付费率为 {payment_rows[0]['payment_rate_pct']}%，同期 ARPU 为 {arpu_value} 元。",
        ok=True,
        steps=[
            make_query_step("payment_rate", {}, rows=payment_rows),
            make_query_step("arpu", {}, rows=arpu_rows),      # 模型自己多查的
        ],
        iterations=3,
    )
    result = EvalRunner(agent=FakeAgent({case.question: answer}), eval_set=eval_set).run_case(case)

    assert result.number_check.rate == 1.0, (
        f"来自 arpu 真值的数字被判成无出处：{result.number_check.unexplained}"
    )


def test_runner_still_flags_numbers_not_in_any_queried_metric():
    """放宽口径不等于放行编造 —— 模型没查过的数字，依然必须被判为无出处。

    这是上一条修正的"反向守门测试"。如果只测"放宽后变好了"，
    很容易在改代码时把对账集直接写空，那指标就永远 100%，也就没意义了。
    """
    eval_set = load_eval_set()
    case = eval_set.get("E09")
    runner = EvalRunner(eval_set=eval_set)
    payment_rows, _ = runner.build_reference(case)

    answer = AgentAnswer(
        question=case.question,
        answer=f"付费率为 {payment_rows[0]['payment_rate_pct']}%，行业均值约 7.77%。",
        ok=True,
        steps=[make_query_step("payment_rate", {}, rows=payment_rows)],
        iterations=2,
    )
    result = EvalRunner(agent=FakeAgent({case.question: answer}), eval_set=eval_set).run_case(case)

    assert result.number_check.rate < 1.0
    assert any(abs(item.value - 7.77) < 1e-6 for item in result.number_check.unexplained)


def test_runner_flags_wrong_metric_and_wrong_params():
    """构造一个"选错指标 + 参数抽错"的 Agent，验证评测器能抓出来。

    注意一个容易误解的点：这里答案里写的 20.55% 其实是**正确的 D7 值**，
    所以数字对账会通过 —— 因为真值是按「期望指标」独立算出来的，
    和被测 Agent 查了什么指标无关。
    也就是说：**「数字说对了」不能证明「路径走对了」**，两者必须分开评。
    这正是要把映射、参数、数字拆成三个独立指标的原因。
    """
    from src.agent.react_agent import AgentAnswer

    case = load_eval_set().get("E05")     # 期望 retention_rate + day_n=7
    answer = AgentAnswer(
        question=case.question,
        answer="七日留存率为 20.55%。",
        ok=True,
        # 查的是 dau，而且 day_n 也错了
        steps=[make_query_step("dau", {"day_n": 1})],
        iterations=2,
    )
    runner = EvalRunner(agent=FakeAgent({case.question: answer}), eval_set=load_eval_set())
    result = runner.run_case(case)

    assert result.metric_mapping_ok is False
    assert result.params_ok is False
    # 数字碰巧说对了 —— 但路径是错的，所以不能被判为合格
    assert result.number_check.rate == 1.0
    # 报告要把它列进「需要人工关注」
    report = build_report([result])
    assert "[E05] 指标映射错误" in report
    assert "[E05] 参数抽取错误" in report


def test_runner_flags_fabricated_number():
    """答案里出现参考数据里完全没有的数字 → 必须被挑出来。"""
    from src.agent.react_agent import AgentAnswer

    case = load_eval_set().get("E04")     # 次日留存
    answer = AgentAnswer(
        question=case.question,
        answer="次日留存率约为 78.9%，样本量 9999 人。",   # 明显编造
        ok=True,
        steps=[make_query_step("retention_rate", {"day_n": 1})],
        iterations=2,
    )
    runner = EvalRunner(agent=FakeAgent({case.question: answer}), eval_set=load_eval_set())
    result = runner.run_case(case)

    assert result.metric_mapping_ok is True
    assert result.number_check.unexplained
    assert {"78.9", "9999"} & {item.raw for item in result.number_check.unexplained}
    assert "找不到出处" in build_report([result])


def test_runner_handles_agent_exception():
    """Agent 抛异常时评测器要如实记录，不能中断整轮评测。"""
    case = load_eval_set().get("E01")
    runner = EvalRunner(agent=FakeAgent({case.question: RuntimeError("网络断了")}),
                        eval_set=load_eval_set())
    result = runner.run_case(case)

    assert result.ok is False
    assert "网络断了" in result.error


def test_runner_scores_out_of_scope_refusal():
    """超范围用例：没有成功的查数 = 正确拒答。"""
    from src.agent.react_agent import AgentAnswer

    case = load_eval_set().get("E17")
    answer = AgentAnswer(
        question=case.question,
        answer=f"我的数据只覆盖 {cfg.DATA_START} ~ {cfg.DATA_END}，这个问题暂时答不了。",
        ok=True,
        steps=[],
        iterations=1,
    )
    runner = EvalRunner(agent=FakeAgent({case.question: answer}), eval_set=load_eval_set())
    result = runner.run_case(case)

    assert result.refuse_ok is True
    assert result.number_check is None      # 超范围用例不做数字对账


def test_runner_detects_query_on_out_of_scope_question():
    """超范围问题却执行了查数 → 判定为没有守住数据边界。"""
    from src.agent.react_agent import AgentAnswer

    case = load_eval_set().get("E17")
    answer = AgentAnswer(
        question=case.question,
        answer="去年 12 月的日活大约是 500 人。",
        ok=True,
        steps=[make_query_step("dau", {"start_date": "2025-12-01"}, rows=[{"dau": 500}])],
        iterations=2,
    )
    runner = EvalRunner(agent=FakeAgent({case.question: answer}), eval_set=load_eval_set())
    result = runner.run_case(case)
    assert result.refuse_ok is False


def test_runner_run_all_reports_progress():
    """progress 回调要按顺序收到每一条，前端/CLI 靠它显示实时进度。"""
    from src.agent.react_agent import AgentAnswer

    eval_set = load_eval_set()
    answers = {
        case.question: AgentAnswer(question=case.question, answer="测试回答", ok=True, steps=[], iterations=1)
        for case in eval_set.all()
    }
    runner = EvalRunner(agent=FakeAgent(answers), eval_set=eval_set)

    seen: list[str] = []
    results = runner.run_all(progress=lambda i, n, c, r: seen.append(c.case_id))

    assert len(results) == len(eval_set)
    assert seen == [c.case_id for c in eval_set.all()]


def test_build_report_contains_key_sections():
    results = [
        make_result("E01", mapping=True, params=True, numbers=(3, 3)),
        make_result("E02", mapping=False, params=False, numbers=(1, 2)),
    ]
    report = build_report(results, eval_set=load_eval_set())

    assert "总览" in report
    assert "逐例明细" in report
    assert "需要人工关注的问题" in report
    assert "指标映射准确率" in report
    assert "答案数字可追溯率" in report
    # 出问题的用例要出现在「需要人工关注」里
    assert "[E02]" in report
    # 报告要讲清楚算法边界，避免读者误以为"数字对上 = 结论正确"
    assert "无法判断" in report


def test_build_report_handles_all_clean():
    results = [make_result("E01", mapping=True, params=True, numbers=(3, 3))]
    report = build_report(results)
    assert "无。所有用例" in report


def test_case_result_to_dict_is_json_serializable():
    """逐例结果要能落盘成 JSON，供离线复算。"""
    import json

    result = make_result("E01", mapping=True, params=True, numbers=(2, 3))
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "E01" in payload
    assert json.loads(payload)["number_total"] == 3