# -*- coding: utf-8 -*-
"""
列语义角色推断（纯函数，零依赖）
===============================
把一张表的「列元信息」映射成「语义角色」，供指标模板匹配使用。

【为什么需要这一步？】
    上传的数据集表名、列名千变万化（可能是 user_daily_snapshot / user_log /
    daily_snapshot …，列可能是 date / dt / day）。而指标模板的 SQL 是按
    「语义角色」写的（用 {date} / {user_id} / {active_flag} 占位），不是按
    具体列名写。
    草稿生成的第一步，就是把「这张表里哪一列扮演哪个角色」推断出来，
    模板才能把占位符替换成真实列名。

【为什么放在 src/metrics/ 而不是 src/data/?】
    它产出的是「语义角色」（指标语义），不是「表结构」（数据层）。
    后续要同时被草稿生成（draft.py）与将来可能的「角色校验」复用，
    属于指标语义层。

【设计取舍】
    纯规则、确定性、可单测 —— 不调用 LLM。
    代价是「列名不规范就识别不到」，但这正是设计意图：
    让「系统能自动识别什么」是可预测的，剩下的交给手工搭建兜底。
"""

from __future__ import annotations

import re
from typing import Any


# ===========================================================================
# 一、角色规则表
# ===========================================================================
# 每项：(角色, 分配优先级, 是否必须为日期列, 匹配模式列表)
#   · 优先级：数值越大越先分配。用于解决「同一列命中多个角色」的冲突，
#     例如 register_date 同时匹配 date，必须优先归 register_date。
#   · is_date：为 True 时，只有推断为日期列（is_date=True 或 sql_type 含
#     DATE/DT）的列才进入候选。命中这个角色可能改变建库时列的判定，留作约束。
_ROLE_SPECS: list[tuple[str, int, bool, tuple[str, ...]]] = [
    #          角色            优先级  须日期  匹配模式（正则，转小写后匹配）
    ("user_id",        100, False, (r"user_id", r"userid", r"\buid\b", r"player_id",
                                    r"account_id", r"member_id")),
    ("register_date",   90, True,  (r"register_date", r"reg_date", r"注册日期",
                                    r"created_date", r"create_date", r"signup_date")),
    ("active_flag",     80, False, (r"is_active", r"active_flag", r"是否活跃",
                                    r"active\b", r"actived")),
    ("event_name",      70, False, (r"event_name", r"\bevent\b", r"事件名", r"action_name")),
    ("date",            60, True,  (r"\bdate\b", r"_date$", r"\bday\b", r"\bdt\b",
                                    r"_dt$", r"event_date", r"log_date", r"日期", r"时间")),
    ("channel",         50, False, (r"channel", r"渠道", r"\bsource\b", r"media")),
    ("version",         40, False, (r"version", r"版本", r"app_version", r"game_version")),
    ("revenue",         30, False, (r"revenue", r"\bamount\b", r"\bmoney\b", r"金额",
                                    r"\bprice\b", r"\bpay\b", r"付费")),
    ("level",           20, False, (r"level", r"lv$", r"等级", r"关卡")),
]


def _normalize(name: str) -> str:
    """统一成小写、去掉常见分隔符，便于正则匹配。"""
    return name.lower().replace("-", "_").replace(" ", "_").replace("/", "_")


_DATE_TYPES = ("date", "datetime", "timestamp", "time")


def _looks_like_date(col: dict[str, Any]) -> bool:
    """判断一列是否被推断为日期列。

    【为什么看 is_date 的 bool，也要看字符串类型？】
      DatasetTable.columns 的 is_date 来自上传时的两道门推断（列名 + 值采样），
      但 refresh_tables() 重建的快照里 is_date 恒为 False、只留 sql_type。
      两处口径应统一：既认 is_date 标志，也认 sql_type 关键字 ——
      规则层只关心「这列是不是日期」，不关心数据从哪来。
    """
    if col.get("is_date"):
        return True
    # 【为什么用 lower 而不是 upper？】_DATE_TYPES 全是小写字面量，
    # 统一降到小写再比较，避免 "DATE" 与 "date" 比不出结果这类静默失败。
    sql_type = str(col.get("sql_type", "")).lower()
    return any(dt in sql_type for dt in _DATE_TYPES)


def _column_name(col: dict[str, Any]) -> str:
    """取最接近原始列名的名称（SQL 名可能带了转义引号）。"""
    raw = col.get("name")
    sql = col.get("sql_name")
    return str(sql if raw is None else raw)


def infer_roles(columns: list[dict[str, Any]]) -> dict[str, str]:
    """从一列元信息列表推断「角色 → SQL 列名」映射。

    Args:
        columns: 与 DatasetTable.columns 同构，每项含 name / sql_name /
                 sql_type / is_date。

    Returns:
        形如 {"date": "date", "user_id": "user_id", "active_flag": "is_active"}
        的映射；识别不到任何角色时返回空 dict（不抛错）。

    【冲突解决——两步走】
      1) 列太多 / 角色太多不可能全分。先按「全局角色优先级」从高到低遍历，
         每个角色取它命中的候选列里「最合适的一个」：
            · 日期类角色只接受日期列；
            · 同角色多列时，越“长/精确”的正则优先级越高（先排它），
              再取该候选里类型更贴合（日期列更贴日期角色）的。
      2) 已分配给某角色的列从池中移除，后续角色不再能占用它 ——
         保证「一列不饰多角」。
    """
    if not columns:
        return {}

    # 预编译每条规则的正则，并记录「模式最长优先」的分量
    compiled: list[tuple[str, int, bool, list[tuple[int, re.Pattern[str]]]]] = []
    for role, prio, is_date, patterns in _ROLE_SPECS:
        cands = []
        for pat in patterns:
            rx = re.compile(pat)
            cands.append((len(pat), rx))  # 长度越长越精确
        compiled.append((role, prio, is_date, cands))

    # 未分配列池：存 (索引, 列dict)
    pool: dict[int, dict[str, Any]] = {i: c for i, c in enumerate(columns)}

    result: dict[str, str] = {}

    # 按角色优先级（高→低）分配
    for role, prio, is_date, cands in sorted(
        compiled, key=lambda spec: spec[1], reverse=True
    ):
        if role in result:
            continue

        best: tuple[int, str] | None = None   # (正则长度分量, 列索引)
        for idx, col in pool.items():
            name = _normalize(_column_name(col))
            # 日期类角色强制要求是日期列
            if is_date and not _looks_like_date(col):
                continue
            # 找该列命中的最精确模式
            hit_len = -1
            for plen, rx in cands:
                if rx.search(name):
                    hit_len = max(hit_len, plen)
            if hit_len >= 0:
                if best is None or hit_len > best[0]:
                    best = (hit_len, idx)

        if best is not None:
            idx = best[1]
            result[role] = _column_name(pool[idx])
            del pool[idx]

    return result