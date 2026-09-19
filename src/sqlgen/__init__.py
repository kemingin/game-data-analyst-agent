# -*- coding: utf-8 -*-
"""SQL 生成与执行层（Phase 2）。

三段式流水线：
    SQLGenerator（生成） → SQLValidator（校验） → QueryExecutor（执行）

分开的好处：每一段都能被单独单元测试，出问题时能立刻定位是哪一段。
"""

from src.sqlgen.executor import QueryExecutor, QueryResult
from src.sqlgen.generator import GeneratedSQL, SQLGenerator
from src.sqlgen.validator import ALLOWED_TABLES, FORBIDDEN_KEYWORDS, SQLValidator

__all__ = [
    "ALLOWED_TABLES",
    "FORBIDDEN_KEYWORDS",
    "GeneratedSQL",
    "QueryExecutor",
    "QueryResult",
    "SQLGenerator",
    "SQLValidator",
]