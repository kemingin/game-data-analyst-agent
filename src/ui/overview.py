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
from pathlib import Path

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


def load_table_stats(db_path: str | Path | None = None) -> dict:
    """读取数据库概览信息。

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
    result: dict = {
        "db_path": str(path),
        "db_size_mb": 0.0,
        "table_count": 0,
        "total_rows": 0,
        "tables": [],
        "window": {
            "start": cfg.DATA_START.isoformat(),
            "end": cfg.DATA_END.isoformat(),
            "days": cfg.DATA_DAYS,
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
            tables.append({"name": name, "rows": rows, "note": _TABLE_NOTES.get(name, "-")})

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