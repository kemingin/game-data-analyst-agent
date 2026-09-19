# -*- coding: utf-8 -*-
"""
侧边栏数据概览（Phase 4）
================================================================================
【为什么要做「数据概览」这一块？】

  一个数据分析 Agent 最容易让人不安的地方是：**我不知道你说的话有没有依据。**
  运营同学看到「D1 留存 46.36%」，第一反应往往是「你这是从哪来的数？」

  侧边栏的数据概览回答的就是这个信任问题：告诉他
    · 数据库里到底有哪几张表、各有多少行
    · 数据覆盖到哪一天（T+1 的边界在哪）
  这些信息让「数据有边界」这件事变得可见，而不是藏在代码里。

【为什么这里的查询绕过 QueryExecutor，自己开一个只读连接？】

  因为 Phase 2 的 QueryExecutor 是给「业务指标查询」用的，它有一整套护栏：
  表白名单、强制 LIMIT、超时、行数上限……这些对「统计全表行数」并不适用
  （COUNT(*) 会被强制 LIMIT 截断，白名单也不该为元信息再开一次）。
  更重要的是：**护栏层的职责应该保持单一**，不该为了前端展示往里塞特例。

  所以这里用最朴素的 sqlite3 只读连接自己查。唯一必须坚持的原则是
  `mode=ro` 这个只读打开方式 —— 前端展示代码永远不该有写库的能力。
"""

from __future__ import annotations

import sqlite3
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

from src import config as cfg

# 每张表的中文说明。为什么要硬编码一份？
# 因为 SQLite 里没有「表注释」这种元数据，而给运营看的界面必须说人话。
# 表名对不上时兜底显示 "-"，不会因为新增了表就让界面出错。
_TABLE_NOTES: dict[str, str] = {
    "dim_user": "用户维表（注册日 / 渠道 / 分群）",
    "dim_game": "游戏维表（真实 Steam 元数据）",
    "dim_channel": "渠道维表（含单个注册用户买量成本）",
    "dim_version": "版本维表（上线日期，版本对比的地基）",
    "dim_position": "招聘岗位维表",
    "user_game": "真实 Steam 用户-游戏行为（时长 / 拥有关系）",
    "user_daily_snapshot": "用户日快照（活跃 + 付费，指标层的主力表）",
    "game_event_log": "事件日志（登录 / 对战 / 付费 / 引导步骤）",
    "hr_recruitment_data": "招聘漏斗数据（简历 → 面试 → 录用）",
    # 下面是 SQL 视图或中间表可能出现的名字，留空不影响展示
}

# 【为什么要一个「未传入」哨兵，而不是直接用 None？】
#   多数据集下「日期窗口」有三种状态，None 只能表达两种：
#     · 没传          → 用全局 config 的值（改造前的行为，必须保持）
#     · 传了具体日期  → 用这个数据集自己的窗口
#     · 明确传 None   → 这个数据集**没有可识别的日期字段**，界面该显示「未知」
#   第三种状态如果也走 config，界面就会出现「可用数据范围：2026-06-20 ~ 2026-09-17」
#   而数据里一个日期都没有 —— 这是「配置的承诺冒充数据的真相」，
#   比不显示更糟：用户会据此以为数据覆盖了那个区间。
_UNSET: Any = object()

# 只读 URI 前缀：mode=ro 让 SQLite 在打开文件时就以只读挂载，
# 任何写操作在驱动层直接报错，比「连上之后再检查」可靠得多。
_READ_ONLY_PREFIX = "file:"


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    """以只读方式打开数据库。

    注意路径必须做 URL 编码：本项目路径里有空格（`Game Data Analyst Agent`），
    不编码的话 `file:E:/Game Data/x.db` 会被 SQLite 当成非法 URI 直接报错。
    （这一条和 Phase 2 的 QueryExecutor 踩的是同一个坑，属于可复用的经验。）
    """
    uri = f"{_READ_ONLY_PREFIX}{urllib.parse.quote(str(db_path))}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _iso(value: date | str) -> str | None:
    """把 date / ISO 字符串统一成 "YYYY-MM-DD"；解析不出返回 None。

    【为什么不直接用 str(value)？】
      因为上传数据集记录的窗口来自 CSV，格式可能是 "2026/9/7"。
      直接透传会让界面出现两种日期写法，也让「天数」算不出来。
      解析失败时返回 None（显示「未知」）比显示一个错日期更诚实。
    """
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        return None


def _span_days(start_iso: str, end_iso: str) -> int:
    """闭区间天数：2026-09-10 ~ 2026-09-16 是 7 天（不是 6 天）。

    与 SQL 里 `julianday(MAX) - julianday(MIN) + 1` 的口径保持一致 ——
    同一件事在两处用不同的算法，是「界面数字和报告数字对不上」的常见根因。
    """
    try:
        return (date.fromisoformat(end_iso) - date.fromisoformat(start_iso)).days + 1
    except ValueError:
        return 0


def load_table_preview(
    db_path: str | Path,
    table: str,
    limit: int = 50,
) -> dict:
    """读取一张表的前若干行，用于上传后的「导入结果核对」。

    【为什么表名可以直接拼进 SQL？】
      因为它来自 sqlite_master / DatasetStore 的登记信息（由 upload.py 建表时写入），
      不是用户输入 —— 与 load_table_stats 里 COUNT(*) 的说明同理。
      但仍然用双引号包裹：用户上传 order.csv 时表名就是 order，不加引号会语法错误。

    返回 {"columns": [...], "rows": [[...]], "error": None | str}
    与 load_table_stats 一致，失败不抛异常而是塞进 error —— 这是侧边栏的展示信息，
    单块显示不出来不该让整页白屏。
    """
    result: dict = {"columns": [], "rows": [], "error": None}
    path = Path(db_path)
    if not path.exists():
        result["error"] = f"数据库文件不存在：{path}"
        return result

    quoted = '"' + table.replace('"', '""') + '"'
    try:
        conn = _connect_read_only(path)
    except sqlite3.Error as exc:  # pragma: no cover - 只在库损坏/被锁时触发
        result["error"] = f"数据库打开失败：{exc}"
        return result

    try:
        cursor = conn.cursor()
        # limit 走参数绑定（它是值，不是标识符，SQL 允许绑定）
        cursor.execute(f"SELECT * FROM {quoted} LIMIT ?", (int(limit),))
        result["columns"] = [desc[0] for desc in cursor.description or ()]
        result["rows"] = [list(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        result["error"] = f"读取预览失败：{exc}"
    finally:
        conn.close()

    return result


def load_table_stats(
    db_path: str | Path | None = None,
    data_start: Any = _UNSET,   # date | str | None；默认哨兵 = 「没传」
    data_end: Any = _UNSET,     # date | str | None；默认哨兵 = 「没传」
    table_notes: dict[str, str] | None = None,
) -> dict:
    """读取数据库概览信息。

    参数：
      db_path     — 数据库文件；None → 默认内置库（改造前行为）
      data_end    — 若传 None 表示「本数据集无日期窗口」，界面显示「未知」，
                    且不再尝试用快照表的真实日期覆盖 config 值
      table_notes — 表名 → 中文说明的映射；None → 用内置这套默认说明

    返回结构：
        {
          "db_size_mb": 106.9,
          "db_path": "...",
          "table_count": 9,
          "total_rows": 1181298,
          "tables": [{"name":..., "rows":..., "note":...}, ...],
          "window": {"start": "2026-06-20", "end": "2026-09-17", "days": 90},
        }

    任何一步失败都不会抛异常，而是把错误塞进返回值的 "error" 字段。
    原因：这是侧边栏的展示信息，数据库万一被占用/损坏，
    也应该只是「这块显示不出来」，而不是整个页面白屏。
    """
    path = Path(db_path) if db_path else cfg.DB_PATH
    notes_map = table_notes if table_notes is not None else _TABLE_NOTES

    # ---- 日期窗口默认值的解析 ----
    # 用一个 object() 哨兵区分「没传」与「明确传 None」：
    #   _UNSET（没传）   → 回落 config 的全局窗口（改造前的行为，必须保留）
    #   传了 date / str  → 用数据集自己的窗口
    #   传了 None        → 数据集无日期，显示「未知」
    if data_start is _UNSET and data_end is _UNSET:
        # 逐字节保留改造前的行为：窗口与天数都直接取 config
        start_default: str | None = cfg.DATA_START.isoformat()
        end_default: str | None = cfg.DATA_END.isoformat()
        days_default = cfg.DATA_DAYS
    else:
        start_default = None if data_start is None else _iso(data_start)
        end_default = None if data_end is None else _iso(data_end)
        if start_default and end_default:
            days_default = _span_days(start_default, end_default)
        else:
            # 只知道一端（或两端都不知道）→ 无法算跨度，用 0 表示「未知」，
            # 而不是硬凑一个数字。前端据此显示「未知」。
            days_default = 0

    result: dict = {
        "db_path": str(path),
        "db_size_mb": 0.0,
        "table_count": 0,
        "total_rows": 0,
        "tables": [],
        "window": {
            "start": start_default,
            "end": end_default,
            "days": days_default,
        },
        "error": None,
    }

    if not path.exists():
        result["error"] = f"数据库文件不存在：{path}"
        return result

    result["db_size_mb"] = round(path.stat().st_size / 1024 / 1024, 1)

    try:
        conn = _connect_read_only(path)
    except sqlite3.Error as exc:  # pragma: no cover - 只在库损坏/被锁时触发
        result["error"] = f"数据库打开失败：{exc}"
        return result

    try:
        cursor = conn.cursor()
        # sqlite_master 是 SQLite 的系统表，记录所有表/索引/视图的定义。
        # 过滤掉 sqlite_ 开头的内部表和 _ 前缀的临时表，只看业务表。
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '\\_%' ESCAPE '\\' "
            "ORDER BY name"
        )
        names = [row[0] for row in cursor.fetchall()]

        tables: list[dict] = []
        total = 0
        for name in names:
            # 表名来自 sqlite_master，不是用户输入，不存在注入面。
            # 但仍然不用参数绑定 —— 因为 SQLite 的参数占位符不能用于表名/列名，
            # 这是 SQL 的通用限制（DDL 层不支持绑定）。这里用双引号包裹以防万一。
            cursor.execute(f'SELECT COUNT(*) FROM "{name}"')
            rows = int(cursor.fetchone()[0])
            total += rows
            tables.append(
                {"name": name, "rows": rows, "note": notes_map.get(name, "-")}
            )

        # 用真实数据把「数据窗口」再校准一次：config.py 里写的是生成参数，
        # 而这里读到的是**库里实际存在的日期范围**，两者不一致时以库为准。
        # 这个细节很关键 —— 界面上的边界必须来自数据本身，而不是配置文件的承诺。
        try:
            cursor.execute(
                "SELECT MIN(date), MAX(date), "
                "CAST(julianday(MAX(date)) - julianday(MIN(date)) + 1 AS INTEGER) "
                "FROM user_daily_snapshot"
            )
            min_date, max_date, span_days = cursor.fetchone()
            if min_date and max_date:
                result["window"] = {
                    "start": str(min_date),
                    "end": str(max_date),
                    "days": int(span_days or 0),
                }
        except sqlite3.Error:
            # 快照表不存在或没有 date 字段：保留 config 里的窗口，不影响其他展示
            pass

        tables.sort(key=lambda item: item["rows"], reverse=True)
        result["tables"] = tables
        result["table_count"] = len(tables)
        result["total_rows"] = total
    except sqlite3.Error as exc:  # pragma: no cover
        result["error"] = f"读取数据概览失败：{exc}"
    finally:
        conn.close()

    return result