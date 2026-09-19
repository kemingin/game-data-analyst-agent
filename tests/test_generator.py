# -*- coding: utf-8 -*-
"""
SQL 生成模块单元测试
================================================================================
覆盖三类场景：
  1. 正常路径 —— 每个指标的默认参数都能生成合法 SQL
  2. 参数解析 —— 相对日期表达式、字符串转整数、默认值合并
  3. 异常路径 —— 未知参数、类型错误、枚举越界、日期超范围、区间倒挂

【为什么测试要固定一个「假的数据窗口」？】
  SQLGenerator 允许注入 data_start / data_end。
  测试里写死 2026-06-20 ~ 2026-09-17，这样即使将来 config 里的数据窗口滚动，
  这些断言也不会突然失败。测试要稳定，才能作为回归保护网。
"""

from __future__ import annotations

from datetime import date

import pytest

from src.exceptions import MetricNotFoundError, MetricParamError
from src.metrics import get_registry
from src.sqlgen.generator import PLACEHOLDER_RE, SQLGenerator

FAKE_START = date(2026, 6, 20)
FAKE_END = date(2026, 9, 17)


@pytest.fixture(scope="module")
def generator() -> SQLGenerator:
    return SQLGenerator(data_start=FAKE_START, data_end=FAKE_END)


# ---------------------------------------------------------------------------
# 一、正常路径：全部指标都能用默认参数生成
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_generate_with_defaults_succeeds(generator, metric):
    """不传任何参数也能生成 —— 这是「用户没说时间」时的兜底路径，必须通。"""
    generated = generator.generate(metric.metric_id)
    assert generated.metric_id == metric.metric_id
    assert generated.sql == metric.sql_template
    assert generated.params


@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_rendered_sql_has_no_leftover_placeholder(generator, metric):
    """渲染后的展示版 SQL 不能残留任何 :占位符，否则用户看到的是天书。"""
    generated = generator.generate(metric.metric_id)
    leftover = PLACEHOLDER_RE.findall(generated.rendered_sql)
    assert leftover == [], f"{metric.metric_id} 渲染后残留占位符：{leftover}"


@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_sql_keeps_placeholders_for_binding(generator, metric):
    """执行版 SQL 必须保留占位符 —— 这是「参数绑定防注入」的前提。

    如果哪天有人把 generated.sql 改成渲染版，这个测试会立刻失败，
    起到「安全回归测试」的作用。
    """
    generated = generator.generate(metric.metric_id)
    assert PLACEHOLDER_RE.search(generated.sql), (
        f"{metric.metric_id} 的执行版 SQL 丢失了占位符，参数绑定防线会失效"
    )


# ---------------------------------------------------------------------------
# 二、参数解析
# ---------------------------------------------------------------------------

def test_default_relative_date_resolved(generator):
    """默认值 @data_end-6 / @data_end 应该被解析成真实日期。"""
    generated = generator.generate("dau")
    assert generated.params["end_date"] == "2026-09-17"
    assert generated.params["start_date"] == "2026-09-11"


def test_user_supplied_relative_expression(generator):
    """用户传进来的参数也支持相对表达式。"""
    generated = generator.generate("dau", {"start_date": "@data_end-29"})
    assert generated.params["start_date"] == "2026-08-19"


def test_int_accepts_string_input(generator):
    """LLM 经常把数字抽成字符串，生成器要能自动转换。"""
    generated = generator.generate("retention_rate", {"day_n": "7"})
    assert generated.params["day_n"] == 7


def test_slash_date_is_normalized(generator):
    """'2026/09/01' 这种写法要能识别并规范化成 ISO 格式。"""
    generated = generator.generate("dau", {"start_date": "2026/09/01"})
    assert generated.params["start_date"] == "2026-09-01"


def test_datetime_with_time_part_is_truncated(generator):
    """带时间部分的日期只取日期部分。

    这里必须同时传 start_date 和 end_date：
    只传 end_date 的话，start_date 会走默认值 @data_end-6 = 2026-09-11，
    反而比 2026-09-10 更晚，会先触发「区间不合法」而看不到截断效果。
    """
    generated = generator.generate(
        "dau", {"start_date": "2026-09-01", "end_date": "2026-09-10 23:59:59"}
    )
    assert generated.params["end_date"] == "2026-09-10"
    assert generated.params["start_date"] == "2026-09-01"


def test_rendered_sql_inlines_values(generator):
    """渲染版应该把日期用单引号包起来写进 SQL，方便用户阅读。"""
    generated = generator.generate("retention_rate", {"day_n": 7})
    # 默认批次整体前移 7 天：2026-09-17 - 6 - 7 = 2026-09-04
    assert "'2026-09-04'" in generated.rendered_sql
    assert "CAST(7 AS TEXT)" in generated.rendered_sql


def test_default_cohort_window_shifts_with_day_n(generator):
    """留存类指标的默认批次窗口必须随 N 一起前移，否则会返回空结果。

    这是验证阶段发现的真空缺陷：原默认窗口是「最近 7 天」，而 SQL 又会剔除
    「注册日 + N 天 > 数据截止日」的用户，两者交集在 D7/D30 下恒为空，
    查询能跑通但 cohort_size = 0，结果全是 NULL —— 是最难被发现的那类错误。
    """
    d1 = generator.generate("retention_rate", {"day_n": 1})
    assert d1.params["cohort_start"] == "2026-09-10"
    assert d1.params["cohort_end"] == "2026-09-16"

    d7 = generator.generate("retention_rate", {"day_n": 7})
    assert d7.params["cohort_start"] == "2026-09-04"
    assert d7.params["cohort_end"] == "2026-09-10"

    d30 = generator.generate("retention_rate", {"day_n": 30})
    assert d30.params["cohort_start"] == "2026-08-12"
    assert d30.params["cohort_end"] == "2026-08-18"
    # 批次窗口长度恒为 7 天，只有位置在动 —— 这样 D1/D7/D30 之间才可比
    for generated in (d1, d7, d30):
        span = date.fromisoformat(generated.params["cohort_end"]) - date.fromisoformat(
            generated.params["cohort_start"]
        )
        assert span.days == 6


def test_date_default_can_reference_int_param(generator):
    """日期表达式除了字面偏移，还能引用前面的整数参数。"""
    generated = generator.generate("channel_retention", {"day_n": 7})
    assert generated.params["cohort_start"] == "2026-08-12"   # 09-17 - 29 - 7
    assert generated.params["cohort_end"] == "2026-09-10"     # 09-17 - 7


def test_single_quote_in_string_value_is_escaped(generator):
    """字符串值里的单引号要翻倍转义，保证展示版 SQL 语法正确。"""
    rendered = SQLGenerator.render("SELECT :x AS v", {"x": "O'Brien"})
    assert rendered == "SELECT 'O''Brien' AS v"


def test_ltv_defaults_are_fully_observable(generator):
    """LTV 的默认批次刻意避开最近 30 天，保证观察窗口完整（否则会低估 LTV）。"""
    generated = generator.generate("ltv")
    assert generated.params["cohort_end"] == "2026-08-18"
    assert generated.params["cohort_start"] == "2026-07-20"
    assert generated.params["window_days"] == 30


# ---------------------------------------------------------------------------
# 三、异常路径
# ---------------------------------------------------------------------------

def test_unknown_metric_raises(generator):
    with pytest.raises(MetricNotFoundError):
        generator.generate("i_do_not_exist")


def test_unknown_param_raises(generator):
    """参数名写错必须报错，不能静默用默认值糊弄过去。"""
    with pytest.raises(MetricParamError) as exc:
        generator.generate("retention_rate", {"days": 7})
    assert "不支持参数" in str(exc.value)


def test_enum_violation_raises(generator):
    """留存天数只允许 1/7/30，传 5 要拦下来。"""
    with pytest.raises(MetricParamError) as exc:
        generator.generate("retention_rate", {"day_n": 5})
    assert "不在允许范围" in str(exc.value)


def test_bad_int_raises(generator):
    with pytest.raises(MetricParamError):
        generator.generate("retention_rate", {"day_n": "七天"})


def test_bool_is_rejected_for_int(generator):
    """True 是 int 的子类，若不单独拦截会被当成 1，属于典型的隐蔽错误。"""
    with pytest.raises(MetricParamError):
        generator.generate("retention_rate", {"day_n": True})


def test_bad_date_format_raises(generator):
    with pytest.raises(MetricParamError) as exc:
        generator.generate("dau", {"start_date": "上个月"})
    assert "日期格式" in str(exc.value)


def test_date_outside_data_window_raises(generator):
    """超出数据覆盖范围不是「查不到」，而是「知识缺失」，必须显式报错。

    对应 Phase 5 评测里的「知识缺失问题」场景。
    """
    with pytest.raises(MetricParamError) as exc:
        generator.generate("dau", {"start_date": "2026-01-01"})
    assert "超出" in str(exc.value)


def test_reversed_date_range_raises(generator):
    """日期倒挂会让 SQL 静默返回空结果，必须拦下来。"""
    with pytest.raises(MetricParamError) as exc:
        generator.generate(
            "dau", {"start_date": "2026-09-10", "end_date": "2026-09-01"}
        )
    assert "不合法" in str(exc.value)


def test_bad_relative_expression_raises(generator):
    with pytest.raises(MetricParamError):
        generator.generate("dau", {"start_date": "@tomorrow"})