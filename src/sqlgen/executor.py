# -*- coding: utf-8 -*-
"""
SQL 执行器（Phase 2）
================================================================================
【职责边界】

  本模块只负责「把一条已经校验过的 SQL 安全地跑出去」，它做四件事：

      1. 只读连接   —— mode=ro + PRAGMA query_only，物理上写不进去
      2. 参数绑定   —— cursor.execute(sql, params)，结构/数据彻底分离
      3. 超时保护   —— SQLite 的 progress handler，超时主动中断
      4. 行数上限   —— 校验器已补 LIMIT，这里再兜一层 fetchmany

  它**不负责**判断 SQL 是否安全 —— 那是 validator 的职责。
  但 execute() 默认会先调用一次校验器，保证「任何执行路径都经过校验」，
  避免调用方忘记校验（这是很容易犯的错）。

【为什么要用 SQLite 的 progress handler 做超时？】

  常见的错误做法是「开一个线程跑查询，主线程 sleep 后放弃等待」——
  这只是「假装超时了」，那条 SQL 还在后台疯狂占用 CPU，连接也回收不了。

  正确做法是让数据库自己中断：SQLite 每执行约 N 条虚拟机指令就会回调一次
  progress handler，我们在这个回调里检查是否超过截止时间，
  返回非 0 就让它立刻中止并抛 OperationalError("interrupted")。
  这是真正的「服务端中断」，代价是回调本身要足够轻量（只做一次时间比较）。

  在真实生产里对应的概念是 MySQL 的 max_execution_time、PostgreSQL 的
  statement_timeout，思路完全一致：**把超时控制下沉到数据库层**。
"""

from __future__ import annotations

import sqlite3
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config as cfg
from src.sqlgen.validator import SQLValidator

# progress handler 的调用频率：每执行这么多条 SQLite 虚拟机指令回调一次。
# 太小会拖慢查询（回调开销大），太大则超时不够及时，5000 是实践中比较均衡的值。
_PROGRESS_OPCODES: int = 5000


@dataclass
class QueryResult:
    """一次查询的完整结果，包含数据、耗时与错误信息。

    Phase 3 的 Agent 会把 columns + rows 交给大模型生成洞察，
    Phase 4 的前端会把同一份结果渲染成表格和图表。
    """

    sql: str
    params: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False          # 行数是否已达上限（可能还有更多数据没返回）
    elapsed_ms: float = 0.0
    error: str | None = None

    def to_dicts(self) -> list[dict[str, Any]]:
        """把结果转成「字典列表」，方便序列化成 JSON 或喂给大模型。"""
        return [dict(zip(self.columns, row)) for row in self.rows]

    def to_markdown(self, max_preview: int = 30) -> str:
        """把结果渲染成 Markdown 表格，用于日志、CLI 展示、以及喂给 LLM。"""
        if not self.success:
            return f"查询失败：{self.error}"
        if not self.columns:
            return "查询成功，但没有返回任何列。"

        preview = self.rows[:max_preview]
        header = "| " + " | ".join(self.columns) + " |"
        divider = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join("NULL" if v is None else str(v) for v in row) + " |"
            for row in preview
        ]
        lines = [header, divider, *body]
        if self.row_count > len(preview):
            lines.append(f"…（共 {self.row_count} 行，此处只显示前 {len(preview)} 行）")
        return "\n".join(lines)


class QueryExecutor:
    """SQL 执行器。默认只读、带行数上限与超时保护。"""

    def __init__(
        self,
        db_path: str | Path | None = None,
        max_rows: int | None = None,
        timeout_seconds: float | None = None,
        validator: SQLValidator | None = None,
    ) -> None:
        self.db_path: Path = Path(db_path) if db_path else cfg.DB_PATH
        self.max_rows: int = max_rows if max_rows is not None else cfg.SQL_MAX_ROWS
        self.timeout_seconds: float = (
            timeout_seconds if timeout_seconds is not None else cfg.SQL_TIMEOUT_SECONDS
        )
        self.validator: SQLValidator = validator or SQLValidator(max_rows=self.max_rows)

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    @property
    def read_only_uri(self) -> str:
        """只读连接串。

        两个细节：
          1. `mode=ro` 让 SQLite 在打开文件时就以只读方式挂载，
             任何写操作都会直接报错，比「连上之后再检查」可靠得多。
          2. 路径必须做 URL 编码：本项目路径里带空格，
             不编码的话 `file:E:/Game Data/x.db` 会被解析成非法 URI。
        """
        posix_path = self.db_path.resolve().as_posix()
        quoted = urllib.parse.quote(posix_path, safe="/:")
        return f"file:{quoted}?mode=ro"

    def connect(self) -> sqlite3.Connection:
        """建立只读连接（也对外暴露，供前端侧边栏做「数据概览」用）。"""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"数据库不存在：{self.db_path}，请先执行 python scripts/build_database.py"
            )
        conn = sqlite3.connect(self.read_only_uri, uri=True)
        # 双保险：即便连接以读写方式打开，query_only 也会把本连接变成只读。
        # 另外它也阻止了「用 SELECT 创建临时表」这类擦边写法。
        conn.execute("PRAGMA query_only = ON")
        return conn

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        allowed_tables: tuple[str, ...] | set[str] | None = None,
        validate: bool = True,
    ) -> QueryResult:
        """执行 SQL，任何失败都通过 QueryResult 返回，不向上抛异常。

        为什么把异常「吞掉」转成返回值？
          因为这是给 Agent 用的工具。Agent 需要拿到结构化的失败原因
          （比如「超时了，请缩小时间范围」）然后自己决定下一步，
          而不是让整个对话崩掉。异常转返回值是「工具层」的标准做法。
        """
        params = dict(params or {})
        start = time.perf_counter()

        # ---------- 第 1 步：安全校验 ----------
        if validate:
            try:
                sql = self.validator.validate(sql, allowed_tables=allowed_tables)
            except Exception as exc:                      # noqa: BLE001 - 安全类异常统一转结构化结果
                return QueryResult(
                    sql=sql,
                    params=params,
                    success=False,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    error=str(exc),
                )

        # ---------- 第 2 步：执行 ----------
        conn: sqlite3.Connection | None = None
        try:
            conn = self.connect()
            deadline = time.perf_counter() + self.timeout_seconds

            # 超时回调：返回非 0 表示「请中断当前查询」
            conn.set_progress_handler(
                lambda: 1 if time.perf_counter() > deadline else 0,
                _PROGRESS_OPCODES,
            )

            cursor = conn.execute(sql, params)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # 用 fetchmany 而不是 fetchall：fetchall 会把结果全部读进内存，
            # 万一上游漏了 LIMIT，一次几百万行足以把进程撑爆。
            # truncated 的语义是「已返回的行数达到上限，可能还有更多」——
            # 因为 LIMIT 已经写在 SQL 里了，我们无法再区分「恰好这么多」和「被截断」，
            # 所以这里给的是一个偏保守的判断，前端文案也应写成「结果可能被截断」。
            rows = [tuple(row) for row in cursor.fetchmany(self.max_rows)]
            truncated = len(rows) >= self.max_rows

            return QueryResult(
                sql=sql,
                params=params,
                success=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )

        except sqlite3.OperationalError as exc:
            message = str(exc)
            if "interrupted" in message.lower():
                message = (
                    f"查询超时（超过 {self.timeout_seconds:g} 秒被强制中断），"
                    f"请缩小时间范围或减少对比维度后重试"
                )
            elif "readonly" in message.lower():
                message = f"数据库为只读模式，拒绝执行写操作：{message}"
            return QueryResult(
                sql=sql,
                params=params,
                success=False,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=message,
            )
        except sqlite3.Error as exc:
            return QueryResult(
                sql=sql,
                params=params,
                success=False,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=f"SQL 执行失败：{exc}",
            )
        finally:
            if conn is not None:
                conn.close()