# -*- coding: utf-8 -*-
"""
工具层单元测试
================================================================================
【这个文件在验证什么？】

  工具层是「LLM 与现实世界之间唯一的接口」，它的正确性直接决定三件事：
    1. 模型能不能看懂有哪些工具（schema 合规）
    2. 模型能不能拿到正确的口径信息（detail 内容完整）
    3. 取数时有没有真的走安全链路（走 Phase 2 的生成→校验→执行）

  最后一条最关键：本文件里所有 query_metric 的测试都是打**真实数据库**的，
  因为它们要验证的不是「逻辑对不对」，而是「LLM 点菜之后，菜到底出不出得来」。

  另外还有一类测试专门验证「失败时反馈得好不好」：
  参数写错时要给出「可用参数有哪些」，这是模型自我纠错的信息基础。
  反馈质量差 → 模型改不对 → 反复重试 → token 白烧，这是一条完整的因果链。
"""

from __future__ import annotations

import pytest

from src import config as cfg
from src.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from src.metrics import get_registry

requires_db = pytest.mark.skipif(
    not cfg.DB_PATH.exists(),
    reason="数据库不存在，请先执行 python scripts/build_database.py",
)


@pytest.fixture(scope="module")
def executor() -> ToolExecutor:
    return ToolExecutor()


# ===========================================================================
# 一、工具定义（schema 合规性）
# ===========================================================================

def test_definitions_follow_openai_schema():
    """工具清单必须严格符合 Function Calling 的 schema。

    为什么这条值得单独测？
      因为 schema 写错时**不会报错**，模型只会「默默不调用这个工具」，
      表现为「Agent 好像变笨了」，非常难排查。用测试把它钉死。
    """
    assert [d["function"]["name"] for d in TOOL_DEFINITIONS] == [
        "list_metrics",
        "get_metric_detail",
        "query_metric",
    ]

    for definition in TOOL_DEFINITIONS:
        assert definition["type"] == "function"
        function = definition["function"]

        # description 是模型决定「要不要调用」的唯一依据，不能为空
        assert function["description"].strip()

        params = function["parameters"]
        # 顶层必须是 object（参数最终要序列化成一个 JSON 对象）
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert isinstance(params["required"], list)
        # required 里声明的字段必须在 properties 里真实存在，否则模型会一脸懵
        for name in params["required"]:
            assert name in params["properties"]


def test_definitions_returns_a_fresh_copy(executor):
    """definitions 返回的必须是副本。

    如果直接返回模块级常量，任何调用方（或框架）原地改一下，
    就会污染全局，影响后续所有请求 —— 这类「共享可变状态」的 bug 很难查。
    """
    first = executor.definitions
    first.clear()
    assert len(executor.definitions) == 3


# ===========================================================================
# 二、list_metrics
# ===========================================================================

def test_list_metrics_covers_entire_registry(executor):
    result = executor.execute("list_metrics")

    assert result.ok
    expected = get_registry().ids()
    assert [m["metric_id"] for m in result.data["metrics"]] == expected
    assert result.data["count"] == len(expected)

    # 给模型看的文本里必须逐个出现 metric_id，否则它没法拿去调下一个工具
    for metric_id in expected:
        assert metric_id in result.content


def test_list_metrics_ignores_unexpected_arguments(executor):
    """模型偶尔会自作主张塞参数。

    对于无参工具，多传参数不影响结果，没必要报错 ——
    报错只会浪费一轮交互，收益为零。这是「宽进严出」的一个小例子。
    """
    result = executor.execute("list_metrics", {"foo": "bar"})
    assert result.ok


# ===========================================================================
# 三、get_metric_detail
# ===========================================================================

def test_detail_contains_definition_params_and_columns(executor):
    result = executor.execute("get_metric_detail", {"metric_id": "retention_rate"})

    assert result.ok
    metric = get_registry().get("retention_rate")

    # 三样东西缺一不可：口径（算的是什么）、参数（怎么调）、输出字段（返回什么）
    assert metric.business_definition in result.content
    assert "day_n" in result.content
    assert "retention_rate_pct" in result.content

    assert result.data["metric_id"] == "retention_rate"
    assert result.data["unit"] == "%"
    assert {p["name"] for p in result.data["params"]} == {"day_n", "cohort_start", "cohort_end"}


def test_detail_lists_enum_values(executor):
    """有枚举取值的参数必须把可选值告诉模型，否则它可能填 day_n=3（不被支持）。"""
    result = executor.execute("get_metric_detail", {"metric_id": "retention_rate"})
    assert "取值范围=1/7/30" in result.content


def test_detail_carries_business_warning(executor):
    """业务定义里的「易错点」提示必须传给模型，这是保证口径一致的关键。"""
    result = executor.execute("get_metric_detail", {"metric_id": "mau"})
    assert "滚动" in result.content


def test_detail_missing_metric_id(executor):
    result = executor.execute("get_metric_detail", {})
    assert result.ok is False
    assert "metric_id" in result.content
    assert result.error


def test_detail_unknown_metric_id_reports_available_ids(executor):
    """指标 ID 写错时要给出「可用指标清单」，模型据此就能自己纠正。

    这条测试守护的是「自我纠错的信息基础」——错误反馈里没有可用选项，
    模型就只能瞎猜，最后必然以轮次耗尽告终。
    """
    result = executor.execute("get_metric_detail", {"metric_id": "not_a_metric"})

    assert result.ok is False
    assert "not_a_metric" in result.content
    assert "dau" in result.content


# ===========================================================================
# 四、参数解析与未知工具
# ===========================================================================

def test_arguments_can_be_a_json_string(executor):
    """ToolExecutor 是独立可复用的组件，不能假设调用方一定传 dict。"""
    result = executor.execute("get_metric_detail", '{"metric_id": "dau"}')
    assert result.ok
    assert result.data["metric_id"] == "dau"


def test_malformed_arguments_string_is_reported(executor):
    result = executor.execute("get_metric_detail", "{这不是合法的 JSON")
    assert result.ok is False
    assert "解析失败" in result.content


def test_arguments_of_wrong_type_is_reported(executor):
    result = executor.execute("get_metric_detail", 12345)
    assert result.ok is False
    assert result.error


def test_unknown_tool_lists_available_tools(executor):
    result = executor.execute("drop_table", {})

    assert result.ok is False
    assert "drop_table" in result.content
    # 提示可用工具，模型才能改对
    assert "list_metrics" in result.content


# ===========================================================================
# 五、query_metric —— 真正取数（打真实数据库）
# ===========================================================================

@requires_db
def test_query_dau_returns_seven_rows(executor):
    result = executor.execute("query_metric", {"metric_id": "dau"})

    assert result.ok
    assert result.data["columns"] == ["date", "dau"]
    assert result.data["row_count"] == 7
    assert len(result.data["rows"]) == 7
    assert result.data["unit"] == "人"


@requires_db
def test_query_observation_contains_table_and_guardrail(executor):
    """喂给模型的观察结果必须包含：指标名、单位、参数、Markdown 表格、以及「别编造」的提醒。"""
    result = executor.execute("query_metric", {"metric_id": "dau"})

    assert "日活跃用户数" in result.content
    assert "单位：人" in result.content
    assert "start_date" in result.content          # 本次实际生效的参数
    assert "| date |" in result.content.replace(" | ", " | ")   # 表格表头
    assert "严禁" in result.content or "不要推测" in result.content


@requires_db
def test_query_retention_with_explicit_day_n(executor):
    result = executor.execute(
        "query_metric", {"metric_id": "retention_rate", "params": {"day_n": 7}}
    )

    assert result.ok
    assert result.data["params"]["day_n"] == 7
    row = result.data["rows"][0]
    assert row["cohort_size"] > 0
    assert 0 <= row["retention_rate_pct"] <= 100


@requires_db
def test_query_channel_retention_returns_one_row_per_channel(executor):
    """渠道对比要一行一个渠道，且按留存率降序（模板里写了 ORDER BY）。"""
    result = executor.execute("query_metric", {"metric_id": "channel_retention"})

    assert result.ok
    channels = [row["channel_name"] for row in result.data["rows"]]
    assert len(channels) == len(set(channels))     # 每个渠道只出现一次
    rates = [row["retention_rate_pct"] for row in result.data["rows"]]
    assert rates == sorted(rates, reverse=True)


@requires_db
def test_query_unknown_param_gives_usable_hint(executor):
    """参数名写错时必须告诉模型「可用参数有哪些」。

    这是本文件里最有价值的一条错误反馈测试：
    模型看到「可用参数：start_date, end_date」，下一轮就能改对；
    只回一句「参数错误」，它只能原地打转。
    """
    result = executor.execute(
        "query_metric", {"metric_id": "dau", "params": {"days": 7}}
    )

    assert result.ok is False
    assert "days" in result.content
    assert "start_date" in result.content


@requires_db
def test_query_out_of_range_date_is_explained(executor):
    """超出数据窗口不是「查询失败」，而是「知识缺失」，提示必须说清数据范围。"""
    result = executor.execute(
        "query_metric", {"metric_id": "dau", "params": {"start_date": "2020-01-01"}}
    )

    assert result.ok is False
    assert "2020-01-01" in result.content
    assert cfg.DATA_START.isoformat() in result.content


@requires_db
def test_query_unknown_metric_id(executor):
    result = executor.execute("query_metric", {"metric_id": "no_such_metric"})
    assert result.ok is False
    assert "no_such_metric" in result.content


@requires_db
def test_query_with_missing_metric_id(executor):
    result = executor.execute("query_metric", {})
    assert result.ok is False
    assert "metric_id" in result.content


@requires_db
def test_query_with_non_object_params(executor):
    """params 传成数组时要给出「应该传键值对」的说明，并附上正确写法示例。"""
    result = executor.execute("query_metric", {"metric_id": "dau", "params": [1, 2]})

    assert result.ok is False
    assert "键值对" in result.content or "JSON 对象" in result.content


@requires_db
def test_query_result_includes_rendered_sql(executor):
    """前端要能展示「Agent 到底执行了什么 SQL」，所以结果里必须带上展示版 SQL。"""
    result = executor.execute("query_metric", {"metric_id": "dau"})

    assert result.ok
    sql = result.data["rendered_sql"]
    assert sql.upper().startswith("SELECT")
    assert ":start_date" not in sql          # 展示版必须把占位符替换成真实值
    assert "2026-" in sql


@requires_db
def test_query_observation_truncates_long_tables(executor):
    """90 天的 DAU 有 90 行，喂给模型的文本要按 max_rows_to_llm 截断。

    这道闸门是「token 预算控制」：数据库层允许返 500 行（给人看表格），
    但全塞进上下文既贵又挤占推理空间。数据本身一行不少地留在 data 里给前端用。
    """
    limited = ToolExecutor(max_rows_to_llm=3)
    result = limited.execute(
        "query_metric",
        {
            "metric_id": "dau",
            "params": {"start_date": cfg.DATA_START.isoformat(), "end_date": cfg.DATA_END.isoformat()},
        },
    )

    assert result.ok
    assert result.data["row_count"] > 3            # 结构化数据保留全量
    assert "只显示前 3 行" in result.content        # 文本被截断


@requires_db
def test_query_is_read_only_end_to_end(executor):
    """工具层不可能被诱导写库：metric_id 只能来自注册表，SQL 由模板生成。

    这里用一个带 SQL 片段的「指标名」做提示词注入尝试，
    期望结果是「指标不存在」，而不是任何写入行为。
    """
    result = executor.execute(
        "query_metric", {"metric_id": "dau; DROP TABLE dim_user"}
    )
    assert result.ok is False

    # 数据库仍然完好
    check = executor.execute("query_metric", {"metric_id": "dau"})
    assert check.ok
    assert check.data["row_count"] == 7


# ===========================================================================
# 六、ToolResult 的序列化
# ===========================================================================

def test_tool_result_to_dict_is_json_serializable(executor):
    import json

    result = executor.execute("list_metrics")
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "list_metrics" in payload