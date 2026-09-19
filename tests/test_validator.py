# -*- coding: utf-8 -*-
"""
SQL 安全校验模块单元测试
================================================================================
测试分两类，缺一不可：

  【拦截测试】危险的 SQL 必须被拒 —— 这是「不能漏」，
              漏一个就是一个真实的安全漏洞。

  【放行测试】合法的 SQL 必须通过 —— 这是「不能误伤」，
              误伤会让正常分析全部失败，而且报错信息会很误导人。
              （真实的教训：表名提取器把 `FROM` 关键字当成了表名，
                导致所有多行 SQL 被判定为「引用了未授权的表」。）

  安全模块的测试必须两边都覆盖：只测拦截，会漏掉误伤；
  只测放行，会漏掉漏洞。
"""

from __future__ import annotations

from datetime import date

import pytest

from src.exceptions import SQLSecurityError
from src.metrics import get_registry
from src.sqlgen.generator import SQLGenerator
from src.sqlgen.validator import ALLOWED_TABLES, SQLValidator


@pytest.fixture(scope="module")
def validator() -> SQLValidator:
    return SQLValidator(max_rows=100)


@pytest.fixture(scope="module")
def generator() -> SQLGenerator:
    return SQLGenerator(data_start=date(2026, 6, 20), data_end=date(2026, 9, 17))


# ---------------------------------------------------------------------------
# 一、放行：注册表里每个指标的模板都必须通过校验且表名解析正确
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_all_registry_templates_pass(validator, generator, metric):
    generated = generator.generate(metric.metric_id)
    safe_sql = validator.validate(generated.sql, allowed_tables=metric.source_tables)
    assert safe_sql.strip()


@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_extracted_tables_match_declared_source_tables(validator, metric):
    """从 SQL 里解析出的表，必须恰好等于指标声明的来源表。

    这条断言同时保护了两件事：
      1. 解析器足够准（不能把关键字/CTE 名当成表）；
      2. JSON 里的 source_tables 没写漏、也没多写。
    """
    report = validator.inspect(metric.sql_template)
    assert set(report.tables) == set(metric.source_tables), (
        f"{metric.metric_id} 解析出 {report.tables}，声明的是 {metric.source_tables}"
    )


def test_cte_names_are_not_treated_as_tables(validator):
    """WITH 定义的临时结果集不是真实表，不能因为它在 FROM 后面就当成表。"""
    sql = "WITH t AS (SELECT user_id FROM dim_user) SELECT COUNT(*) FROM t"
    assert validator.extract_tables(sql) == {"dim_user"}


def test_comma_join_tables_are_detected(validator):
    """逗号连接的隐式表也必须被识别到，否则就是一个绕过白名单的口子。"""
    sql = "SELECT a.user_id FROM dim_user a, dim_channel b"
    assert validator.extract_tables(sql) == {"dim_user", "dim_channel"}


def test_from_keyword_at_line_start_is_not_a_table(validator):
    """回归测试：换行后紧跟的 FROM 曾被误判成表名，导致合法 SQL 被拒。"""
    sql = "WITH x AS (\n  SELECT release_date\n  FROM dim_version\n)\nSELECT * FROM x"
    assert validator.extract_tables(sql) == {"dim_version"}


# ---------------------------------------------------------------------------
# 二、拦截：写入、DDL、多语句、越权表、危险函数
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO dim_channel VALUES (9, 'x', 'y', 0)",
        "UPDATE dim_user SET user_group = 'S'",
        "DELETE FROM user_daily_snapshot",
        "DROP TABLE dim_user",
        "ALTER TABLE dim_user ADD COLUMN x TEXT",
        "CREATE TABLE t (a INT)",
        "ATTACH DATABASE 'e:/tmp/evil.db' AS evil",
        "PRAGMA table_info(dim_user)",
        "VACUUM",
        "SELECT 1; DROP TABLE dim_user",
        "WITH t AS (SELECT 1) INSERT INTO dim_channel SELECT * FROM t",
    ],
    ids=[
        "insert", "update", "delete", "drop", "alter", "create",
        "attach", "pragma", "vacuum", "multi_statement", "cte_then_insert",
    ],
)
def test_write_and_ddl_statements_are_rejected(validator, sql):
    with pytest.raises(SQLSecurityError):
        validator.validate(sql)


def test_non_select_leading_keyword_is_rejected(validator):
    """不以 SELECT / WITH 开头的语句一律拒绝（含 EXPLAIN 这类管理语句）。"""
    with pytest.raises(SQLSecurityError) as exc:
        validator.validate("EXPLAIN QUERY PLAN SELECT * FROM dim_user")
    assert "只允许 SELECT" in str(exc.value)


def test_unknown_table_is_rejected(validator):
    with pytest.raises(SQLSecurityError) as exc:
        validator.validate("SELECT * FROM secret_salary_table")
    assert "未授权的表" in str(exc.value)


def test_unknown_table_in_comma_join_is_rejected(validator):
    """逗号连接里的越权表同样要被抓到。"""
    with pytest.raises(SQLSecurityError):
        validator.validate("SELECT a.user_id FROM dim_user a, secret_salary_table b")


def test_forbidden_function_is_rejected(validator):
    with pytest.raises(SQLSecurityError) as exc:
        validator.validate("SELECT readfile('e:/tmp/a.txt')")
    assert "禁止的函数" in str(exc.value)


def test_system_table_is_rejected(validator):
    """sqlite_master 不在白名单里，读它同样越权（会泄露库结构）。"""
    with pytest.raises(SQLSecurityError):
        validator.validate("SELECT name FROM sqlite_master")


def test_empty_sql_is_rejected(validator):
    for bad in ("", "   ", "\n"):
        with pytest.raises(SQLSecurityError):
            validator.validate(bad)


def test_allowed_tables_can_be_tightened(validator):
    """指标级白名单可以比全局白名单更严：只允许碰自己声明的表。"""
    with pytest.raises(SQLSecurityError):
        validator.validate(
            "SELECT COUNT(*) FROM user_daily_snapshot",
            allowed_tables=("dim_user",),
        )


def test_allowed_tables_outside_dataset_whitelist_is_rejected(validator):
    """如果指标声明了本数据集白名单之外的表，说明配置本身就有问题，必须报错。

    注：多数据集改造后，比对基准从模块常量 ALLOWED_TABLES 换成实例的
    self.allowed_tables（默认值仍是 ALLOWED_TABLES），错误文案随之更新。
    """
    with pytest.raises(SQLSecurityError) as exc:
        validator.validate(
            "SELECT COUNT(*) FROM dim_user",
            allowed_tables=("dim_user", "not_exist_table"),
        )
    assert "本数据集的白名单" in str(exc.value)


# ---------------------------------------------------------------------------
# 三、注释归一化
# ---------------------------------------------------------------------------

def test_comments_are_stripped_before_scanning(validator):
    """注释里的关键字不算数 —— 否则写个注释就会被误拦。"""
    safe_sql = validator.validate("SELECT COUNT(*) FROM dim_user /* DROP TABLE x */ LIMIT 5")
    assert "DROP" not in safe_sql.upper()


def test_keyword_split_by_comment_is_still_rejected(validator):
    """用注释把关键字拆开（SEL/**/ECT）会被归一化成非法语句，同样被拦。"""
    with pytest.raises(SQLSecurityError):
        validator.validate("SEL/**/ECT * FROM dim_user")


def test_line_comment_cannot_smuggle_write_keyword(validator):
    """行注释拼接写入语句会在「多语句」这一关被拦下。"""
    with pytest.raises(SQLSecurityError):
        validator.validate("SELECT 1 -- comment\n; DELETE FROM dim_user")


# ---------------------------------------------------------------------------
# 四、行数上限
# ---------------------------------------------------------------------------

def test_limit_is_appended_when_missing(validator):
    sql = "SELECT * FROM dim_user"
    report = validator.inspect(sql)
    assert report.limit_added is True
    assert report.normalized_sql.rstrip().endswith("LIMIT 100")
    assert report.row_limit == 100


def test_existing_small_limit_is_kept(validator):
    sql = "SELECT * FROM dim_user LIMIT 10"
    report = validator.inspect(sql)
    assert report.limit_added is False
    assert report.normalized_sql.rstrip().endswith("LIMIT 10")
    assert report.row_limit == 10


def test_existing_large_limit_is_shrunk_to_cap(validator):
    """已写的 LIMIT 超过上限时要收紧，不能让调用方绕过行数保护。"""
    sql = "SELECT * FROM dim_user LIMIT 99999"
    report = validator.inspect(sql)
    assert report.normalized_sql.rstrip().endswith("LIMIT 100")
    assert report.row_limit == 100


def test_trailing_semicolon_does_not_break_limit(validator):
    """结尾分号要先去掉再补 LIMIT，否则会拼出 `...; LIMIT 100` 的语法错误。"""
    safe_sql = validator.validate("SELECT * FROM dim_user;")
    assert ";" not in safe_sql
    assert safe_sql.rstrip().endswith("LIMIT 100")


def test_whitelist_contains_all_analysis_tables():
    """白名单必须覆盖数据字典里 9 张表，缺一张就会有指标跑不通。"""
    assert ALLOWED_TABLES == {
        "dim_game", "user_game", "dim_channel", "dim_version", "dim_user",
        "dim_position", "game_event_log", "user_daily_snapshot", "hr_recruitment_data",
    }