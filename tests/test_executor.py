# -*- coding: utf-8 -*-
"""
SQL 执行器单元测试
================================================================================
这个文件里最重要的一条是 test_all_metrics_end_to_end：
它把 Phase 2 的三段流水线（生成 → 校验 → 执行）串起来，
对注册表里的每一个指标都真刀真枪跑一遍真实数据库。

为什么这条测试价值最高？
  因为前面所有测试都是用「假 SQL」验证「逻辑对不对」，
  只有端到端测试能回答真正的问题：**这个指标到底能不能查出数？**
  12 个指标的 SQL 模板只要有一个写错表名、写错字段、写错日期函数，
  这里就会立刻暴露。
"""

from __future__ import annotations

import sqlite3

import pytest

from src import config as cfg
from src.metrics import get_registry
from src.sqlgen.generator import SQLGenerator
from src.sqlgen.executor import QueryExecutor

pytestmark = pytest.mark.skipif(
    not cfg.DB_PATH.exists(),
    reason="数据库不存在，请先执行 python scripts/build_database.py",
)


@pytest.fixture(scope="module")
def executor() -> QueryExecutor:
    return QueryExecutor()


@pytest.fixture(scope="module")
def generator() -> SQLGenerator:
    return SQLGenerator()


# ---------------------------------------------------------------------------
# 一、端到端：12 个指标全部真实查通（Phase 2 的核心验收）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_all_metrics_end_to_end(generator, executor, metric):
    """生成 → 校验 → 执行，每个指标都要能跑通并返回至少一行结果。"""
    generated = generator.generate(metric.metric_id)
    result = executor.execute(generated.sql, generated.params, allowed_tables=metric.source_tables)

    assert result.success, f"{metric.metric_id} 执行失败：{result.error}"
    assert result.row_count >= 1, f"{metric.metric_id} 没有返回任何数据"
    assert result.columns, f"{metric.metric_id} 没有返回列信息"
    assert result.elapsed_ms < 10_000


def test_dau_returns_one_row_per_day(executor, generator):
    """DAU 用默认的「最近 7 天」参数，应该正好返回 7 行。"""
    generated = generator.generate("dau")
    result = executor.execute(generated.sql, generated.params, allowed_tables=generated.source_tables)

    assert result.success
    assert result.columns == ["date", "dau"]
    assert result.row_count == 7
    # 日均活跃在几千这个量级（5000 名用户），量级断言能挡住「查错表」这类低级错误
    assert all(0 < row[1] <= 5000 for row in result.rows)


def test_retention_rate_is_a_sane_percentage(executor, generator):
    """D1 留存率必须落在 0~100 之间，且量级符合游戏行业常识（10%~60%）。"""
    generated = generator.generate("retention_rate", {"day_n": 1})
    result = executor.execute(generated.sql, generated.params, allowed_tables=generated.source_tables)

    assert result.success
    cohort_size, retained, rate = result.rows[0]
    assert cohort_size > 0
    assert 0 <= retained <= cohort_size
    assert 10 <= rate <= 60


def test_tutorial_funnel_is_monotonic(executor, generator):
    """漏斗必须单调递减：完成步骤 3 的人不可能多于完成步骤 1 的人。"""
    generated = generator.generate("tutorial_funnel")
    result = executor.execute(generated.sql, generated.params, allowed_tables=generated.source_tables)

    assert result.success
    step0, step1, step2, step3 = result.rows[0][:4]
    assert step0 >= step1 >= step2 >= step3


# ---------------------------------------------------------------------------
# 二、只读保护
# ---------------------------------------------------------------------------

def test_connection_is_read_only(executor):
    """即使绕过校验器直接拿连接，数据库层也必须拒绝写入。

    这是第 3 层防线（executor）的独立验证：连接串是 mode=ro，
    不是靠「代码里记得不要写」来保证安全，而是数据库物理上不允许写。
    """
    conn = executor.connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO dim_channel VALUES (99, 'x', 'y', 0)")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM dim_channel")
    finally:
        conn.close()


def test_read_only_uri_encodes_spaces(executor):
    """本项目路径含空格，URI 必须做百分号编码，否则连不上库。"""
    assert " " not in executor.read_only_uri
    assert executor.read_only_uri.endswith("?mode=ro")
    # 能真正连上才说明 URI 是对的
    executor.connect().close()


def test_write_sql_is_blocked_by_validator(executor):
    """通过 execute() 走一遍，写入语句应由校验器拦下并转成结构化错误。"""
    result = executor.execute("DROP TABLE dim_user")
    assert result.success is False
    assert "禁止" in result.error or "只允许 SELECT" in result.error


# ---------------------------------------------------------------------------
# 三、错误处理与保护机制
# ---------------------------------------------------------------------------

def test_error_is_returned_not_raised(executor):
    """查询失败要以 QueryResult 的形式返回，不能把异常抛给 Agent。"""
    result = executor.execute("SELECT COUNT(*) FROM no_such_table")
    assert result.success is False
    assert result.error
    assert result.rows == []


def test_missing_binding_param_returns_error(executor):
    """占位符没给值时应返回失败结果（而不是崩掉）。"""
    result = executor.execute(
        "SELECT COUNT(*) FROM dim_user WHERE register_date >= :start_date"
    )
    assert result.success is False
    assert result.error


def test_row_limit_is_enforced(executor):
    """执行器会把返回行数压在上限内，并标记 truncated。"""
    limited = QueryExecutor(max_rows=5)
    result = limited.execute("SELECT user_id FROM dim_user")
    assert result.success
    assert result.row_count == 5
    assert result.truncated is True


def test_no_truncation_when_result_is_small(executor):
    result = QueryExecutor(max_rows=500).execute("SELECT channel_id FROM dim_channel")
    assert result.success
    assert result.row_count == 6
    assert result.truncated is False


def test_timeout_interrupts_heavy_query():
    """超时必须由数据库主动中断，而不是「假装放弃等待」。

    这里构造一个三表笛卡尔积（21 万 × 21 万 × 21 万），
    没有超时保护的话这条 SQL 会跑到天荒地老。
    """
    fast = QueryExecutor(timeout_seconds=0.5)
    result = fast.execute(
        "SELECT COUNT(*) FROM user_daily_snapshot a, "
        "user_daily_snapshot b, user_daily_snapshot c"
    )
    assert result.success is False
    assert "超时" in result.error


# ---------------------------------------------------------------------------
# 四、结果格式化
# ---------------------------------------------------------------------------

def test_to_dicts_returns_row_dicts(executor):
    result = executor.execute("SELECT channel_id, channel_name FROM dim_channel ORDER BY channel_id")
    rows = result.to_dicts()
    assert rows[0]["channel_id"] == 1
    assert rows[0]["channel_name"] == "自然量"
    assert len(rows) == result.row_count


def test_to_markdown_renders_table(executor):
    result = executor.execute("SELECT COUNT(*) AS n FROM dim_channel")
    markdown = result.to_markdown()
    assert markdown.startswith("| n |")
    assert "---" in markdown


def test_to_markdown_on_failure_returns_error_text(executor):
    result = executor.execute("SELECT COUNT(*) FROM no_such_table")
    assert "查询失败" in result.to_markdown()