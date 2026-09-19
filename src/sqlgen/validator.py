# -*- coding: utf-8 -*-
"""
SQL 安全校验模块（Phase 2）
================================================================================
【为什么模板是受控的，还要再做一层校验？】

  这是「纵深防御（Defense in Depth）」的思路。现实工程里，单一防线一定会被绕过：

    第 1 层：模板受控（JSON 里的 SQL 是预审过的）
             ↳ 但如果有人往 JSON 里提交了一个带 DROP 的模板呢？评审会漏。
    第 2 层：本模块的语法校验（本文件）
             ↳ 拦截非 SELECT、写入关键字、越权表名、多语句；
               并强制补 LIMIT，防止一次性拉回百万行把内存打爆。
    第 3 层：执行层只读（executor.py）
             ↳ 连接串用 mode=ro + PRAGMA query_only，就算前两层全挂了，
               数据库物理上也拒绝写入。
    第 4 层：参数绑定（generator.py）
             ↳ 参数永远不作为 SQL 文本拼接，注入无从谈起。

  **只要任何一层没被绕过，系统就是安全的；四层同时失效才可能出事。**
  这就是回答「你怎么保证 LLM 不写危险 SQL」的标准结构。

【本模块检查哪些东西？】

  1. 必须是单条语句（禁止 `;` 拼接第二条）
  2. 必须以 SELECT 或 WITH 开头（也就是只读查询）
  3. 不能出现写入/DDL/管理类关键字（INSERT/UPDATE/DROP/ATTACH/PRAGMA...）
  4. 不能调用危险函数（readfile / writefile / load_extension...）
  5. 引用的表必须全部在白名单内（默认「全库 9 张表」，若指标声明了来源表则更严）
  6. 必须带 LIMIT，且不超过 max_rows

【关于「注释剥离」】

  攻击者常用 `/*...*/` 或 `--` 把关键字拆开，比如 `SEL/**/ECT` 或
  `DROP--\n TABLE`。所以校验前**先剥离注释再扫描**，
  这样 `SEL/**/ECT` 会被还原成 `SELECT`，反而更容易被规则识别。
  先归一化、再检查，是做输入校验的通用套路。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src import config as cfg
from src.exceptions import SQLSecurityError

# ---------------------------------------------------------------------------
# 一、白名单：全库允许被查询的表
# ---------------------------------------------------------------------------
# 只列「读」的表，且与 src/data/schema.sql 保持一致。
# 注意：这里用 frozenset（不可变集合），避免运行期被意外修改。
ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "dim_game",
        "user_game",
        "dim_channel",
        "dim_version",
        "dim_user",
        "dim_position",
        "game_event_log",
        "user_daily_snapshot",
        "hr_recruitment_data",
    }
)

# ---------------------------------------------------------------------------
# 二、黑名单：只读查询里绝不该出现的关键字
# ---------------------------------------------------------------------------
# 说明：SQLite 里能改数据/改结构/碰外部文件的关键字都收进来了。
#       刻意**没有**收录 END / RELEASE 这类词 —— 它们会误伤 CASE ... END
#       和 release_date 字段名，属于典型的「黑名单误伤」问题。
FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    {
        # --- 数据写入 ---
        "INSERT", "UPDATE", "DELETE", "REPLACE", "UPSERT", "MERGE", "TRUNCATE",
        # --- 结构变更 ---
        "CREATE", "DROP", "ALTER", "REINDEX", "RENAME",
        # --- 事务与权限 ---
        "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "GRANT", "REVOKE", "SET",
        # --- 数据库管理与外部资源 ---
        "ATTACH", "DETACH", "PRAGMA", "VACUUM", "ANALYZE", "EXPLAIN",
        "LOAD", "DECLARE", "EXEC", "EXECUTE", "CALL",
        # --- 结果外带 ---
        "INTO", "OUTFILE", "DUMPFILE",
    }
)

# 危险函数：SQLite 的文件读写扩展与自定义函数
FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {"readfile", "writefile", "load_extension", "fts3_tokenizer", "edit"}
)

# ---------------------------------------------------------------------------
# 三、预编译正则（模块级编译一次，避免每次调用都重新编译）
# ---------------------------------------------------------------------------
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")

# 第一条语句必须是 SELECT 或 WITH
_READ_ONLY_HEAD_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

# 括号内关键字黑名单正则：\b 保证是完整单词，不会被 release_date 这类名字误伤
_FORBIDDEN_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(sorted(FORBIDDEN_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_FORBIDDEN_FUNCTION_RE = re.compile(
    r"\b(" + "|".join(sorted(FORBIDDEN_FUNCTIONS, key=len, reverse=True)) + r")\s*\(",
    re.IGNORECASE,
)

# 抓取 FROM / JOIN 后面的表名
_FROM_JOIN_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", re.IGNORECASE)

# 抓取 CTE（WITH x AS (...), y AS (...)）的名字，用于把它们从「表」里排除
_CTE_RE = re.compile(
    r"(?:\bWITH\b|,)\s+(?:RECURSIVE\s+)?([A-Za-z_]\w*)\s+AS\s*\(",
    re.IGNORECASE,
)

# 抓取 FROM 子句正文（到 WHERE / GROUP BY ... 为止），用于识别逗号连接的隐式表。
#
# 【为什么终止符里必须有 SELECT / WITH？】
#   这是踩过坑才补上的。考虑 CTE 串联的写法：
#       WITH a AS (SELECT ... FROM t1),
#            b AS (SELECT ... FROM a)
#       SELECT col1, col2 FROM b
#   最后一个 CTE 的 `FROM a)` 之后再没有 WHERE/GROUP BY 之类的终止符，
#   于是 `.*?` 会一路吞到字符串结尾，把**外层 SELECT 的字段列表**也当成 FROM 正文，
#   按逗号切开后 `col1` / `col2` 会被误判成表名 → 合法 SQL 被判「越权」。
#   把 SELECT / WITH 加入终止符，FROM 子句就绝不会跨越到下一个查询块。
_FROM_CLAUSE_RE = re.compile(
    r"\bFROM\b(.*?)(?=\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b"
    r"|\bLIMIT\b|\bUNION\b|\bSELECT\b|\bWITH\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_LIMIT_NUMBER_RE = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)

# 严格匹配「表名 [AS] 别名」这一种形态，用于解析逗号连接的隐式表
_TABLE_REF_RE = re.compile(r"([A-Za-z_]\w*)(?:\s+(?:AS\s+)?[A-Za-z_]\w*)?", re.IGNORECASE)

# 解析逗号连接时，要排除这些「长得像标识符、其实是关键字」的词。
# 举例：某个逗号分段恰好从 FROM 开始（`..., FROM dim_version`），
# 若不排除，FROM 会被当成一个表名，导致合法 SQL 被误判成越权 —— 即「误伤」。
_NON_TABLE_WORDS: frozenset[str] = frozenset(
    {
        "from", "join", "inner", "left", "right", "full", "cross", "outer",
        "natural", "using", "where", "group", "order", "by", "having", "limit",
        "union", "except", "intersect", "select", "as", "on", "and", "or",
        "not", "case", "when", "then", "else", "end", "distinct", "all",
        "with", "recursive", "values", "offset", "window",
    }
)


@dataclass(frozen=True)
class ValidationReport:
    """校验结果，便于前端/日志展示「这次校验做了什么」。"""

    tables: tuple[str, ...]      # 识别到的真实表
    row_limit: int               # 最终生效的行数上限
    limit_added: bool            # 是否是校验器自动补上的 LIMIT
    normalized_sql: str          # 剥离注释并去掉结尾分号后的 SQL


class SQLValidator:
    """SQL 安全校验器。

    使用方式：
        validator = SQLValidator()
        safe_sql = validator.validate(sql, allowed_tables=metric.source_tables)
        # safe_sql 已补齐 LIMIT，可直接交给 executor 执行
    """

    def __init__(self, max_rows: int | None = None) -> None:
        self.max_rows: int = max_rows if max_rows is not None else cfg.SQL_MAX_ROWS

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def validate(
        self,
        sql: str,
        allowed_tables: tuple[str, ...] | set[str] | None = None,
    ) -> str:
        """校验 SQL 并返回「已补齐 LIMIT」的可执行版本。

        参数：
            allowed_tables —— 额外收紧的表范围（通常传指标的 source_tables）。

        异常：
            SQLSecurityError —— 命中任意一条安全规则
        """
        # inspect 已经把 LIMIT 处理完了，normalized_sql 就是可直接执行的版本
        return self.inspect(sql, allowed_tables).normalized_sql

    def inspect(
        self,
        sql: str,
        allowed_tables: tuple[str, ...] | set[str] | None = None,
    ) -> ValidationReport:
        """执行全部校验并返回报告（不修改 SQL，方便测试与审计）。"""
        if not sql or not sql.strip():
            raise SQLSecurityError("SQL 为空，拒绝执行")

        normalized = self._strip_comments(sql)
        normalized = self._strip_trailing_semicolon(normalized)

        if not normalized.strip():
            raise SQLSecurityError("SQL 剥离注释后为空，拒绝执行")

        self._check_single_statement(normalized)
        self._check_read_only(normalized)
        self._check_forbidden_tokens(normalized)

        tables = self.extract_tables(normalized)
        permitted = self._resolve_permitted_tables(allowed_tables)
        self._check_table_whitelist(tables, permitted)

        final_sql, limit_added = self._plan_row_limit(normalized)
        row_limit = self._effective_row_limit(final_sql)

        return ValidationReport(
            tables=tuple(sorted(tables)),
            row_limit=row_limit,
            limit_added=limit_added,
            normalized_sql=final_sql,
        )

    # ------------------------------------------------------------------
    # 各条规则
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_comments(sql: str) -> str:
        """剥离块注释与行注释，防止用注释把关键字拆开绕过检查。"""
        return _LINE_COMMENT_RE.sub(" ", _BLOCK_COMMENT_RE.sub(" ", sql))

    @staticmethod
    def _strip_trailing_semicolon(sql: str) -> str:
        """去掉结尾的分号。

        结尾分号是无害的（很多客户端会自动加），但如果不去掉，
        后面补 LIMIT 时会变成 `SELECT ...; LIMIT 500` 而语法错误。
        """
        return sql.rstrip().rstrip(";").rstrip()

    @staticmethod
    def _check_single_statement(sql: str) -> None:
        """禁止多语句。

        `SELECT 1; DROP TABLE dim_user` 是最经典的注入形态。
        剥掉结尾分号后如果还有分号，一定是多语句，直接拒绝。
        """
        if ";" in sql:
            index = sql.index(";")
            raise SQLSecurityError(
                f"检测到多条语句（第 {index} 个字符处的分号），只允许执行单条查询"
            )

    @staticmethod
    def _check_read_only(sql: str) -> None:
        """强制以 SELECT 或 WITH 开头。"""
        if not _READ_ONLY_HEAD_RE.match(sql):
            head = sql.strip()[:30].replace("\n", " ")
            raise SQLSecurityError(f"只允许 SELECT 查询，当前语句以「{head}」开头")

    @staticmethod
    def _check_forbidden_tokens(sql: str) -> None:
        """扫描写入/管理类关键字与危险函数。"""
        keyword = _FORBIDDEN_KEYWORD_RE.search(sql)
        if keyword:
            raise SQLSecurityError(
                f"SQL 中出现禁止的关键字：{keyword.group(1).upper()}"
            )

        function = _FORBIDDEN_FUNCTION_RE.search(sql)
        if function:
            raise SQLSecurityError(
                f"SQL 中调用了禁止的函数：{function.group(1)}"
            )

    @classmethod
    def extract_tables(cls, sql: str) -> set[str]:
        """从 SQL 中提取真实表名（小写）。

        要处理三种情况，缺一不可：
          1. `FROM t` / `JOIN t`        —— 最常见
          2. `FROM t AS x` / `FROM t x` —— 别名，只取表名
          3. `FROM t1, t2`              —— 逗号连接的隐式交叉连接

        第 3 种是很多人会漏的绕过手法：如果只扫 FROM/JOIN，
        `SELECT * FROM dim_user, hr_recruitment_data` 里的第二张表就漏检了。

        另外必须把 CTE 名排除掉 —— 它们出现在 FROM 后面，但并不是真表。
        """
        cte_names = {name.lower() for name in _CTE_RE.findall(sql)}

        tables: set[str] = set()

        # 情况 1 / 2
        for name in _FROM_JOIN_RE.findall(sql):
            lowered = name.lower()
            if lowered not in _NON_TABLE_WORDS:
                tables.add(lowered)

        # 情况 3：先取出 FROM 子句正文，再按逗号切分
        for clause in _FROM_CLAUSE_RE.findall(sql):
            for part in clause.split(","):
                # 只接受「表名」或「表名 别名」/「表名 AS 别名」这两种形态。
                # 用严格的全匹配而不是「取第一个词」，是为了避免把 CASE、WHEN
                # 这类关键字误判成表名（它们可能恰好出现在某个逗号分段的开头），
                # 造成「把合法 SQL 判成越权」的误伤。
                match = _TABLE_REF_RE.fullmatch(part.strip())
                if match:
                    name = match.group(1).lower()
                    if name not in _NON_TABLE_WORDS:
                        tables.add(name)

        return tables - cte_names

    @staticmethod
    def _resolve_permitted_tables(
        allowed_tables: tuple[str, ...] | set[str] | None,
    ) -> set[str]:
        """确定本次允许引用的表集合。"""
        if allowed_tables is None:
            return set(ALLOWED_TABLES)

        requested = {name.lower() for name in allowed_tables}
        outside = requested - ALLOWED_TABLES
        if outside:
            # 指标自己声明的来源表都不能超出全局白名单，这是配置层面的防线
            raise SQLSecurityError(
                f"指标声明的来源表不在全局白名单中：{', '.join(sorted(outside))}"
            )
        return requested

    @staticmethod
    def _check_table_whitelist(tables: set[str], permitted: set[str]) -> None:
        illegal = tables - permitted
        if illegal:
            raise SQLSecurityError(
                f"SQL 引用了未授权的表：{', '.join(sorted(illegal))}；"
                f"本次允许的表：{', '.join(sorted(permitted))}"
            )

    # ------------------------------------------------------------------
    # 行数上限
    # ------------------------------------------------------------------
    def _plan_row_limit(self, sql: str) -> tuple[str, bool]:
        """判断是否需要补 LIMIT，返回 (SQL, 是否补过)。

        - 没有 LIMIT：追加 `LIMIT max_rows`
        - 已有 LIMIT 但写的是**字面数字**且超过上限：收紧到上限
        - 已有 LIMIT 且用的是占位符或更小的数字：保持原样
        """
        if not _LIMIT_RE.search(sql):
            return f"{sql.rstrip()}\nLIMIT {self.max_rows}", True

        match = _LIMIT_NUMBER_RE.search(sql)
        if match and int(match.group(1)) > self.max_rows:
            start, end = match.span(1)
            return f"{sql[:start]}{self.max_rows}{sql[end:]}", False

        return sql, False

    def _effective_row_limit(self, sql: str) -> int:
        """算出最终生效的行数上限（用于报告展示）。"""
        match = _LIMIT_NUMBER_RE.search(sql)
        if match:
            return min(int(match.group(1)), self.max_rows)
        return self.max_rows