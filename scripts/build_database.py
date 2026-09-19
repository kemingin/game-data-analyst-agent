# -*- coding: utf-8 -*-
"""
Phase 1 · 一键建库脚本
================================================================================
流程严格遵循数仓工程的正确顺序：

    删旧库 → 建表(schema.sql) → 批量导数 → 建索引(indexes.sql) → ANALYZE → 数据验收

【为什么要按这个顺序？】
  1. 先建表再导数：这是显而易见的。
  2. 导数完成后再建索引：如果表上先有索引，每插入一行 SQLite 都要维护 B+ 树，
     84 万行数据导入会慢 3~5 倍。先导数、后建索引是标准的批量导入优化手段。
  3. 最后 ANALYZE：让查询优化器拿到列的数据分布统计，才知道该不该走索引。

【运行方式】
    python scripts/build_database.py

【前置条件】
    先执行 python scripts/generate_mock_data.py 生成模拟数据 CSV
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src.data.steam_data import load_game_meta, load_user_game

SCHEMA_FILE = Path(__file__).resolve().parents[1] / "src" / "data" / "schema.sql"
INDEX_FILE = Path(__file__).resolve().parents[1] / "src" / "data" / "indexes.sql"


# ===========================================================================
# 一、建库
# ===========================================================================

def create_connection() -> sqlite3.Connection:
    """创建（或重建）SQLite 数据库并执行建表语句。"""
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 每次重建：删掉旧库，保证结果和脚本一致（可复现）
    if cfg.DB_PATH.exists():
        cfg.DB_PATH.unlink()
        print(f"  · 已删除旧数据库 {cfg.DB_PATH.name}")

    conn = sqlite3.connect(str(cfg.DB_PATH))

    # 导入期的性能优化（这些 PRAGMA 只影响当前连接）：
    #   journal_mode=WAL      : 写前日志，读写并发更好
    #   synchronous=OFF       : 关闭每次写入的落盘等待，批量导入快很多
    #                           （代价是断电可能丢数据，开发环境完全可接受）
    #   temp_store=MEMORY     : 临时表放内存，排序/聚合更快
    #   cache_size=-200000    : 约 200MB 页缓存（负数单位是 KB）
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;
        """
    )

    # executescript 可以一次执行包含多条 SQL 的文件（DDL 就是一堆 CREATE TABLE）
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    conn.commit()
    print("  · 建表完成（9 张表）")
    return conn


def load_table(conn: sqlite3.Connection, df: pd.DataFrame, table: str, note: str = "") -> None:
    """把 DataFrame 批量写入 SQLite。

    参数说明：
        if_exists="append"  : 追加写入（表已经由 schema.sql 建好了，不要覆盖）
        index=False         : 不要把 pandas 的行号当成一列写进去
        method="multi"      : 拼成 INSERT INTO t VALUES (...),(...),(...) 的多值语句，
                              比逐行 INSERT 快 5~10 倍

    为什么 chunksize 要动态算？
        SQLite 对「一条 SQL 里最多能有多少个绑定变量（? 占位符）」有硬上限
        （老版本 999，新版本 32766）。method="multi" 会把 chunksize 行拼成一条语句，
        所以单条语句的变量数 = 行数 × 列数，很容易超限报 "too many SQL variables"。
        这里按 900 个变量反推安全行数，保证永远不超限。
    """
    n_cols = max(1, len(df.columns))
    step = max(1, 900 // n_cols)      # 900 < 999，留一点安全边界

    t0 = time.perf_counter()
    df.to_sql(
        table,
        conn,
        if_exists="append",
        index=False,
        chunksize=step,
        method="multi",
    )
    elapsed = time.perf_counter() - t0
    print(f"  · {table:<22s} {len(df):>9,} 行  ({elapsed:5.1f}s)  {note}")


# ===========================================================================
# 二、数据验收（这一步非常重要：入库后必须自证数据是可信的）
# ===========================================================================

def run(conn: sqlite3.Connection, sql: str) -> tuple[list[tuple], float]:
    """执行一条查询，返回 (结果行, 耗时秒数)。

    耗时用 time.perf_counter() 测量 —— 它比 time.time() 精度更高，
    是做性能对比时的标准选择。
    """
    t0 = time.perf_counter()
    rows = conn.execute(sql).fetchall()
    return rows, time.perf_counter() - t0


def verify(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 78)
    print("数据验收报告")
    print("=" * 78)

    # ---------- 1. 各表行数 ----------
    print("\n【1】各表行数")
    tables = [
        "dim_game", "user_game", "dim_channel", "dim_version", "dim_user",
        "dim_position", "game_event_log", "user_daily_snapshot", "hr_recruitment_data",
    ]
    total = 0
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        total += n
        print(f"    {t:<24s} {n:>10,}")
    print(f"    {'合计':<24s} {total:>10,}")

    # ---------- 2. 时间范围 ----------
    print("\n【2】数据时间范围（有没有把日期写串？）")
    row = conn.execute(
        "SELECT MIN(event_date), MAX(event_date), MIN(event_time), MAX(event_time) "
        "FROM game_event_log"
    ).fetchone()
    print(f"    事件日期：{row[0]} ~ {row[1]}")
    print(f"    事件时刻：{row[2]} ~ {row[3]}")
    row = conn.execute("SELECT MIN(date), MAX(date) FROM user_daily_snapshot").fetchone()
    print(f"    快照日期：{row[0]} ~ {row[1]}")

    # ---------- 3. 一致性校验：活跃标记 ⇔ 登录事件 ----------
    # 逻辑：把「快照里标记为活跃」和「当天有登录事件」做全外连接，找出对不上的记录。
    # 用 LEFT JOIN + UNION + LEFT JOIN 实现 FULL OUTER JOIN（SQLite 3.39 之前不支持 FULL JOIN）。
    print("\n【3】一致性校验：快照 is_active=1 与「当天有登录事件」是否完全一致")
    mismatch_sql = """
        WITH active_snap AS (
            SELECT user_id, date FROM user_daily_snapshot WHERE is_active = 1
        ),
        active_evt AS (
            SELECT DISTINCT user_id, event_date AS date
            FROM game_event_log WHERE event_name = '登录'
        )
        SELECT
            (SELECT COUNT(*) FROM active_snap s
              LEFT JOIN active_evt e ON s.user_id = e.user_id AND s.date = e.date
              WHERE e.user_id IS NULL) AS snap_only,
            (SELECT COUNT(*) FROM active_evt e
              LEFT JOIN active_snap s ON s.user_id = e.user_id AND s.date = e.date
              WHERE s.user_id IS NULL) AS evt_only
    """
    snap_only, evt_only = conn.execute(mismatch_sql).fetchone()
    status = "通过" if (snap_only == 0 and evt_only == 0) else "不通过"
    print(f"    仅快照标记活跃、但无登录事件：{snap_only} 条")
    print(f"    仅有登录事件、但快照未标记活跃：{evt_only} 条")
    print(f"    校验结果：{status}")

    # ---------- 4. 一致性校验：快照收入 ⇔ 充值事件金额 ----------
    print("\n【4】一致性校验：快照 revenue 合计 与 充值事件金额合计")
    snap_sum = conn.execute("SELECT ROUND(SUM(revenue), 2) FROM user_daily_snapshot").fetchone()[0]
    evt_sum = conn.execute(
        "SELECT ROUND(SUM(event_value), 2) FROM game_event_log WHERE event_name = '充值'"
    ).fetchone()[0]
    print(f"    快照收入合计   : {snap_sum:,.2f} 元")
    print(f"    充值事件合计   : {evt_sum:,.2f} 元")
    print(f"    校验结果：{'通过' if snap_sum == evt_sum else '不通过'}")

    # ---------- 5. 业务可用性抽检：能不能回答「版本上线后新用户留存变化」 ----------
    # 这一段 SQL 提前演示了 Phase 3 要用的关键技术，注释写得很细。
    print("\n【5】业务抽检：新用户次日留存率（D1）—— 版本上线前后对比")
    print("    说明：只统计注册日 <= 2026-09-16 的用户，因为 09-17 注册的用户还没有次日数据。")
    d1_sql = """
        WITH new_user AS (
            -- 分母：新增用户（这里 1 个用户 1 条注册记录）
            SELECT user_id, register_date
            FROM dim_user
            WHERE register_date <= '2026-09-16'
        ),
        with_d1 AS (
            -- LEFT JOIN 到「注册日期 +1 天」的快照行，取那天是否活跃
            SELECT
                n.register_date,
                n.user_id,
                COALESCE(s.is_active, 0) AS d1_active
            FROM new_user n
            LEFT JOIN user_daily_snapshot s
                   ON s.user_id = n.user_id
                  AND s.date = date(n.register_date, '+1 day')
        )
        SELECT
            CASE
                WHEN register_date >= '2026-09-05' THEN 'B_版本后(09-05起)'
                WHEN register_date >= '2026-08-22' THEN 'A_版本前两周(08-22~09-04)'
                ELSE 'C_更早'
            END AS period,
            COUNT(*) AS new_users,
            SUM(d1_active) AS d1_users,
            ROUND(SUM(d1_active) * 100.0 / COUNT(*), 1) AS d1_rate
        FROM with_d1
        GROUP BY period
        ORDER BY period
    """
    rows, _ = run(conn, d1_sql)
    print(f"    {'区间':<24s}{'新增用户':>10s}{'次日留存用户':>14s}{'D1留存率':>10s}")
    for period, new_users, d1_users, rate in rows:
        print(f"    {period:<24s}{new_users:>10,}{d1_users:>14,}{rate:>9.1f}%")

    # ---------- 6. 索引是否真的生效 ----------
    print("\n【6】执行计划验证（EXPLAIN QUERY PLAN）")
    demo_sql = (
        "SELECT COUNT(*) FROM game_event_log "
        "WHERE event_name = '注册' AND event_date = '2026-09-05'"
    )
    plan, elapsed = run(conn, "EXPLAIN QUERY PLAN " + demo_sql)
    for line in plan:
        print(f"    {line[-1]}")
    rows, elapsed = run(conn, demo_sql)
    print(f"    实际查询耗时：{elapsed * 1000:.1f} ms（结果 {rows[0][0]} 条）")
    print("    ↑ 出现 SEARCH ... USING INDEX 就说明走索引了；SCAN 是全表扫描，要么缺索引，要么优化器认为不值得走索引。")

    # ---------- 7. 数据库体积 ----------
    size_mb = cfg.DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n【7】数据库文件：{cfg.DB_PATH}")
    print(f"    体积：{size_mb:.1f} MB")


# ===========================================================================
# 三、主流程
# ===========================================================================

def main() -> None:
    t_start = time.perf_counter()
    print("=" * 78)
    print("SQLite 建库：建表 → 导数 → 建索引 → ANALYZE → 验收")
    print("=" * 78)

    # --- 检查模拟数据是否已生成 ---
    required = [
        "dim_channel.csv", "dim_version.csv", "dim_position.csv", "dim_user.csv",
        "game_event_log.csv", "user_daily_snapshot.csv", "hr_recruitment_data.csv",
    ]
    missing = [f for f in required if not (cfg.GENERATED_DIR / f).exists()]
    if missing:
        raise SystemExit(
            "缺少模拟数据文件：" + ", ".join(missing)
            + "\n请先执行： python scripts/generate_mock_data.py"
        )

    print("\n[1/5] 建库与建表")
    conn = create_connection()

    print("\n[2/5] 导入数据")
    # --- 真实数据 ---
    meta = load_game_meta()
    load_table(conn, meta, "dim_game", "<- 真实 Steam 游戏元数据")

    ug = load_user_game(dedup=True)
    load_table(conn, ug, "user_game", "<- 真实用户游玩明细（已按 user_id+appid 去重）")

    # --- 模拟运营数据 ---
    for csv_name, table in [
        ("dim_channel.csv", "dim_channel"),
        ("dim_version.csv", "dim_version"),
        ("dim_position.csv", "dim_position"),
        ("dim_user.csv", "dim_user"),
        ("game_event_log.csv", "game_event_log"),
        ("user_daily_snapshot.csv", "user_daily_snapshot"),
        ("hr_recruitment_data.csv", "hr_recruitment_data"),
    ]:
        df = pd.read_csv(cfg.GENERATED_DIR / csv_name)
        load_table(conn, df, table)

    print("\n[3/5] 创建索引并更新统计信息")
    t0 = time.perf_counter()
    conn.executescript(INDEX_FILE.read_text(encoding="utf-8"))
    conn.commit()
    print(f"  · 11 个索引创建完成 + ANALYZE 完成（{time.perf_counter() - t0:.1f}s）")

    print("\n[4/5] 数据验收")
    verify(conn)

    print("\n[5/5] 收尾")
    # WAL 模式会产生一个 .db-wal 伴生文件，把数据真正合并回主库文件
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()
    conn.close()
    print("  · 数据库连接已关闭，WAL 已合并")

    print(f"\n✓ 建库完成，总耗时 {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()