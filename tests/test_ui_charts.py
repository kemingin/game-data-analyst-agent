# -*- coding: utf-8 -*-
"""
前端选图策略单元测试（Phase 4）
================================================================================
【为什么界面代码也要写测试？】

  因为在本项目里，「选图」不是审美问题，而是**产品正确性问题**：
    · 把留存趋势画成柱状图，用户就看不出拐点；
    · 把漏斗指标渲染成单个数字，最有价值的「逐层流失」就全丢了。

  pick_chart_spec() 是纯函数（输入 dict，输出一个描述），不碰 streamlit / plotly，
  所以能像 Phase 2/3 的代码一样被 pytest 完整覆盖。
  **这正是把选图逻辑从 app.py 里拆出来的理由**：
  写在 Streamlit 脚本里的逻辑是没法测的（它一 import 就连带整个运行时）。

  下面的样例数据不是编的，全部按指标模板真实输出的列结构构造，
  这样测试通过 = 真实数据下的选图是对的。
"""

from __future__ import annotations

from src.ui.charts import (
    build_figure,
    format_value,
    headline_values,
    numeric_columns,
    pick_chart_spec,
    to_dataframe,
)

# ===========================================================================
# 一、测试数据构造
# ===========================================================================

def make_data(
    metric_id: str,
    metric_name: str,
    unit: str,
    columns: list[str],
    rows: list[dict],
) -> dict:
    """按 ToolResult.data 的真实结构造数据。"""
    return {
        "metric_id": metric_id,
        "metric_name": metric_name,
        "unit": unit,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": False,
        "rendered_sql": "SELECT ... :start_date ...",
    }


DAU_TREND = make_data(
    "dau", "日活跃用户数（DAU）", "人",
    ["date", "dau"],
    [
        {"date": "2026-09-11", "dau": 465},
        {"date": "2026-09-12", "dau": 488},
        {"date": "2026-09-13", "dau": 512},
    ],
)

CHANNEL_COMPARISON = make_data(
    "channel_retention", "渠道留存对比（渠道分群）", "%",
    ["channel_name", "cohort_size", "retained_users", "retention_rate_pct"],
    [
        {"channel_name": "自然量", "cohort_size": 620, "retained_users": 289, "retention_rate_pct": 46.61},
        {"channel_name": "抖音买量", "cohort_size": 410, "retained_users": 172, "retention_rate_pct": 41.95},
        {"channel_name": "腾讯广告", "cohort_size": 260, "retained_users": 98, "retention_rate_pct": 37.69},
    ],
)

TUTORIAL_FUNNEL = make_data(
    "tutorial_funnel", "新手引导漏斗转化率", "%",
    [
        "step0_registered", "step1_started", "step2_mid", "step3_done",
        "step1_conv_pct", "step2_conv_pct", "step3_conv_pct", "tutorial_completion_pct",
    ],
    [
        {
            "step0_registered": 812, "step1_started": 779, "step2_mid": 638, "step3_done": 476,
            "step1_conv_pct": 95.94, "step2_conv_pct": 81.90, "step3_conv_pct": 74.61,
            "tutorial_completion_pct": 58.62,
        }
    ],
)

RETENTION_SINGLE_ROW = make_data(
    "retention_rate", "留存率（D1 / D7 / D30）", "%",
    ["cohort_size", "retained_users", "retention_rate_pct"],
    [{"cohort_size": 812, "retained_users": 346, "retention_rate_pct": 42.61}],
)

VERSION_COMPARISON = make_data(
    "version_retention", "版本前后新用户留存对比", "%",
    ["version_segment", "cohort_size", "retained_users", "retention_rate_pct"],
    [
        {"version_segment": "版本上线后", "cohort_size": 366, "retained_users": 170, "retention_rate_pct": 46.45},
        {"version_segment": "版本上线前两周", "cohort_size": 240, "retained_users": 97, "retention_rate_pct": 40.42},
        {"version_segment": "更早（前 15~28 天）", "cohort_size": 210, "retained_users": 79, "retention_rate_pct": 37.62},
    ],
)


# ===========================================================================
# 二、基础工具函数
# ===========================================================================

def test_to_dataframe_keeps_sql_column_order():
    """DataFrame 的列顺序必须和 SQL 的 SELECT 顺序一致。

    为什么要断言这个？因为表格是给人对照 SQL 看的。
    列顺序一旦乱掉，「channels 的第 4 列是留存率」这种认知就崩了。
    """
    df = to_dataframe(CHANNEL_COMPARISON)
    assert list(df.columns) == [
        "channel_name", "cohort_size", "retained_users", "retention_rate_pct",
    ]
    assert len(df) == 3


def test_to_dataframe_handles_empty_and_none():
    assert to_dataframe(None).empty
    assert to_dataframe({}).empty
    # 有列名但没有数据行：仍然是空表，但列结构要保留
    assert list(to_dataframe({"columns": ["date", "dau"], "rows": []}).columns) == ["date", "dau"]


def test_numeric_columns_recognizes_numeric_strings():
    """数字型的字符串也要被认成数值列。

    因为 SQLite 在某些拼接场景下会返回字符串形态的数字，
    如果只判断 dtype，这类列会被误判成文本而被整列丢弃。
    """
    data = make_data(
        "dau", "日活跃用户数（DAU）", "人",
        ["date", "dau"],
        [{"date": "2026-09-11", "dau": "465"}, {"date": "2026-09-12", "dau": "488"}],
    )
    assert numeric_columns(to_dataframe(data)) == ["dau"]


def test_numeric_columns_skips_text_and_all_null():
    data = make_data(
        "x", "x", "",
        ["name", "value", "empty"],
        [{"name": "自然量", "value": 1, "empty": None}, {"name": "抖音", "value": 2, "empty": None}],
    )
    assert numeric_columns(to_dataframe(data)) == ["value"]


def test_format_value():
    assert format_value(465, "人") == "465人"
    assert format_value(8.456, "元") == "8.46元"
    assert format_value(42.61, "%") == "42.61%"
    assert format_value(1181298, "行") == "1,181,298行"
    assert format_value(None, "%") == "-"
    assert format_value("自然量", "") == "自然量"


# ===========================================================================
# 三、选图策略（本文件的核心）
# ===========================================================================

def test_empty_result_produces_no_chart():
    spec = pick_chart_spec(make_data("dau", "日活跃用户数（DAU）", "人", ["date", "dau"], []))
    assert spec.kind == "none"
    assert "没有返回任何数据" in spec.reason


def test_none_data_does_not_crash():
    """data 为 None 时不能抛异常 —— 前端不能因为一个意外格式就整页崩掉。"""
    spec = pick_chart_spec(None)
    assert spec.kind == "none"


def test_date_series_becomes_line_chart():
    spec = pick_chart_spec(DAU_TREND)
    assert spec.kind == "line"
    assert spec.x == "date"
    assert spec.y == "dau"
    assert spec.unit == "人"


def test_channel_comparison_becomes_bar_chart():
    spec = pick_chart_spec(CHANNEL_COMPARISON)
    assert spec.kind == "bar"
    assert spec.x == "channel_name"


def test_bar_prefers_percentage_over_sample_size():
    """默认纵轴必须是留存率，而不是样本量。

    这是最容易被写错的一条：如果按「第一列数值」去挑，会挑到 cohort_size，
    运营打开页面看到一根根「样本量柱状图」，完全答非所问。
    """
    spec = pick_chart_spec(CHANNEL_COMPARISON)
    assert spec.y == "retention_rate_pct"
    # 但其他数值列仍要留给用户手动切换
    assert set(spec.y_options) == {"cohort_size", "retained_users", "retention_rate_pct"}


def test_single_row_becomes_metric_card():
    spec = pick_chart_spec(RETENTION_SINGLE_ROW)
    assert spec.kind == "metric"
    assert spec.y == "retention_rate_pct"


def test_tutorial_funnel_becomes_funnel_chart():
    """漏斗指标必须特判成漏斗图。

    它是嵌套在内层的：如果按「只有一行 → 指标卡」的通用规则走，
    界面上只会出现一个注册人数，把 8 列里最有价值的逐层流失全丢掉。
    """
    spec = pick_chart_spec(TUTORIAL_FUNNEL)
    assert spec.kind == "funnel"
    assert [col for col, _ in spec.funnel_steps] == [
        "step0_registered", "step1_started", "step2_mid", "step3_done",
    ]
    assert [label for _, label in spec.funnel_steps][0] == "注册"


def test_version_comparison_becomes_bar_chart():
    spec = pick_chart_spec(VERSION_COMPARISON)
    assert spec.kind == "bar"
    assert spec.x == "version_segment"
    assert spec.y == "retention_rate_pct"


def test_date_like_values_without_date_in_name_are_detected():
    """列名不含 date，但值长得像日期的列，也要被认成日期列。"""
    data = make_data(
        "dau", "日活跃用户数（DAU）", "人",
        ["stat_day", "dau"],
        [{"stat_day": "2026-09-11", "dau": 465}, {"stat_day": "2026-09-12", "dau": 488}],
    )
    spec = pick_chart_spec(data)
    assert spec.kind == "line"
    assert spec.x == "stat_day"


def test_high_cardinality_column_is_not_a_group_dimension():
    """唯一值太多的列（如 user_id）不是分组维度，不该被拿去画柱状图。

    否则会出现一张 60 根柱子的图，既看不清也没意义。
    """
    rows = [{"user_id": 10000 + i, "play_minutes": 30 + i} for i in range(60)]
    data = make_data("x", "用户游戏时长", "分钟", ["user_id", "play_minutes"], rows)
    assert pick_chart_spec(data).kind == "none"


def test_prefer_y_overrides_automatic_choice():
    """用户手动指定的纵轴优先级最高。"""
    spec = pick_chart_spec(CHANNEL_COMPARISON, prefer_y="cohort_size")
    assert spec.y == "cohort_size"
    # 非法列名要被忽略，回退到自动选择，不能报错
    assert pick_chart_spec(CHANNEL_COMPARISON, prefer_y="不存在的列").y == "retention_rate_pct"


def test_reason_is_always_explainable():
    """每一种图的选型理由都必须写出来（产品体验与可维护性都要求）。"""
    for data in (DAU_TREND, CHANNEL_COMPARISON, TUTORIAL_FUNNEL, RETENTION_SINGLE_ROW, VERSION_COMPARISON):
        spec = pick_chart_spec(data)
        assert spec.reason, f"{data['metric_id']} 缺少选图理由"


# ===========================================================================
# 四、指标卡取值
# ===========================================================================

def test_headline_values_puts_main_metric_first():
    """主指标排第一，样本量用中文标签跟上。

    这条规则对应项目的硬性要求「对比类结论必须同时报样本量」——
    提示词管住了模型怎么说，界面这一层再机械保证一次"样本量一定看得见"。
    """
    spec = pick_chart_spec(RETENTION_SINGLE_ROW)
    values = headline_values(RETENTION_SINGLE_ROW, spec)
    assert values[0] == ("retention_rate_pct", "42.61%")
    assert ("样本量（新增用户）", "812") in values
    assert ("留存人数", "346") in values


def test_headline_values_empty_on_no_rows():
    assert headline_values(None, pick_chart_spec(None)) == []


# ===========================================================================
# 五、Plotly 渲染
# ===========================================================================

def test_build_figure_returns_none_for_non_chart_kinds():
    """指标卡和"不出图"两种情况都不该走到 Plotly 渲染。"""
    assert build_figure(RETENTION_SINGLE_ROW, pick_chart_spec(RETENTION_SINGLE_ROW)) is None
    assert build_figure(None, pick_chart_spec(None)) is None


def test_build_figure_renders_every_chart_kind():
    """三种真正要画的图型都能渲染出 Figure，且标题不为空。"""
    line = build_figure(DAU_TREND, pick_chart_spec(DAU_TREND))
    bar = build_figure(CHANNEL_COMPARISON, pick_chart_spec(CHANNEL_COMPARISON))
    funnel = build_figure(TUTORIAL_FUNNEL, pick_chart_spec(TUTORIAL_FUNNEL))

    for figure in (line, bar, funnel):
        assert figure is not None
        assert figure.layout.title.text

    # 折线图与柱状图的数据点数量要和结果行数一致
    assert len(line.data[0].x) == 3
    assert len(line.data[0].y) == 3
    assert len(bar.data[0].x) == 3
    # 漏斗图有 4 层
    assert len(funnel.data[0].y) == 4
    assert len(funnel.data[0].x) == 4