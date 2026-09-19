# -*- coding: utf-8 -*-
"""
列语义角色推断测试（Phase 8 · b1）
==================================
覆盖三类断言：
  ① 正面 —— 规范列名的表能识别出预期角色；
  ② 反面 —— 无意义列名（c1/c2/c3）识别不出任何角色，且不抛错；
  ③ 冲突 —— 同一列命中多角色时按优先级正确归属（register_date 压过 date），
            同一角色命中多列时取最精确的那一列。

【为什么这类纯函数也要测？】
  它是「半自动草稿闭环」的入口判断。规则写错不会有任何报错，
  只会让草稿少一个指标、或者把 register_date 当成普通 date 用 ——
  后者的后果是留存率算错，而 SQL 看起来完全正常。
  这种「静默算错」正是单测最该守的地方。
"""

from __future__ import annotations

import pytest

from src.metrics.schema_roles import infer_roles


def _col(name: str, sql_type: str = "INTEGER", is_date: bool = False) -> dict:
    """构造一列元信息，字段与 DatasetTable.columns 同构。"""
    return {"name": name, "sql_name": name, "sql_type": sql_type, "is_date": is_date}


def _date_col(name: str) -> dict:
    return _col(name, sql_type="DATE", is_date=True)


# ===========================================================================
# 一、正面：规范列名
# ===========================================================================

def test_snapshot_table_roles():
    """快照表（user_daily_snapshot 风格）应识别出 日期/用户/活跃标志。"""
    roles = infer_roles([
        _date_col("date"),
        _col("user_id"),
        _col("is_active"),
        _col("playtime_minutes"),
    ])
    assert roles["date"] == "date"
    assert roles["user_id"] == "user_id"
    assert roles["active_flag"] == "is_active"


def test_event_log_table_roles():
    """事件流水表应识别出 日期/用户/事件名。"""
    roles = infer_roles([
        _date_col("event_date"),
        _col("user_id"),
        _col("event_name"),
    ])
    assert roles["date"] == "event_date"
    assert roles["user_id"] == "user_id"
    assert roles["event_name"] == "event_name"


def test_chinese_column_names():
    """中文列名也要能识别（上传的 CSV 很可能是中文表头）。"""
    roles = infer_roles([
        _date_col("日期"),
        _col("用户id"),
        _col("是否活跃"),
        _col("渠道"),
    ])
    assert roles["date"] == "日期"
    assert roles["active_flag"] == "是否活跃"
    assert roles["channel"] == "渠道"


def test_alias_column_names():
    """常见别名（uid / dt / reg_date）应被识别。"""
    roles = infer_roles([
        _date_col("dt"),
        _col("uid"),
        _date_col("reg_date"),
    ])
    assert roles["date"] == "dt"
    assert roles["user_id"] == "uid"
    assert roles["register_date"] == "reg_date"


# ===========================================================================
# 二、反面：识别不出
# ===========================================================================

def test_meaningless_columns_yield_empty():
    """无意义列名识别不出任何角色，且**不抛错**（诚实失败而非崩溃）。"""
    roles = infer_roles([_col("c1"), _col("c2"), _col("c3")])
    assert roles == {}


def test_empty_columns():
    """空列表直接返回空 dict。"""
    assert infer_roles([]) == {}


def test_date_role_requires_date_column():
    """列名叫 date 但类型不是日期时，不应被当作日期角色。

    【为什么这条重要？】
      若把 TEXT 类型的 "date" 列当日期列，指标窗口的 BETWEEN 比较会变成
      字符串比较，取到的数据可能是错的且不报错。
    """
    roles = infer_roles([_col("date", sql_type="TEXT", is_date=False)])
    assert "date" not in roles


def test_date_role_accepts_sql_type_only():
    """refresh_tables() 重建的快照 is_date 恒为 False，只靠 sql_type 也要能认。"""
    roles = infer_roles([_col("date", sql_type="DATE", is_date=False)])
    assert roles["date"] == "date"


# ===========================================================================
# 三、冲突：优先级与唯一性
# ===========================================================================

def test_register_date_beats_plain_date():
    """同一列同时像「注册日期」和「日期」时，必须归 register_date。

    【为什么？】留存率的分母是「按注册日圈定的用户批次」，
      若把 register_date 当普通 date 用，留存率会退化成「某天的活跃占比」，
      数字看着合理但语义完全错了。
    """
    roles = infer_roles([_date_col("register_date")])
    assert roles.get("register_date") == "register_date"
    assert "date" not in roles


def test_one_column_serves_only_one_role():
    """一列不饰多角：分配出去的列不能再被别的角色占用。"""
    roles = infer_roles([_date_col("register_date"), _date_col("date")])
    assert roles["register_date"] == "register_date"
    assert roles["date"] == "date"
    # 两个角色落在不同的列上
    assert roles["register_date"] != roles["date"]


def test_most_precise_pattern_wins_within_role():
    """同一角色命中多列时，取匹配模式更精确（更长）的那一列。"""
    # event_date 命中 "event_date"（10 字符），date 只命中 "\bdate\b"（8 字符）
    roles = infer_roles([_date_col("date"), _date_col("event_date")])
    assert roles["date"] == "event_date"


@pytest.mark.parametrize(
    "column_name,expected_role",
    [
        ("channel", "channel"),
        ("app_version", "version"),
        ("revenue", "revenue"),
        ("amount", "revenue"),
        ("user_level", "level"),
    ],
)
def test_individual_roles(column_name, expected_role):
    """逐角色冒烟：每个角色至少有一个能命中的典型列名。"""
    roles = infer_roles([_col(column_name)])
    assert roles.get(expected_role) == column_name
