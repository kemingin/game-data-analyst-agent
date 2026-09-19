# -*- coding: utf-8 -*-
"""
CSV 上传建表
=============
把用户上传的 CSV 字节流变成一张可查询的 SQLite 表。

【为什么这个模块不 import streamlit？】
    它是纯函数库 —— 输入是 bytes，输出是结构化的结果对象。
    前端只负责「拿到 bytes 交给它、把结果画出来」。
    这样它可以脱离浏览器用 pytest 直接测，边界情况（GBK 编码、前导零、
    空文件、只有表头）都能自动化覆盖，不用手点页面。
    这与 src/ui/overview.py「查询逻辑与渲染分离」是同一个做法。

【这是全项目唯一一处「外部输入直达 SQL 文本」的路径】
    项目其它地方的 SQL 都是程序按预审模板填参生成的（参数走占位符绑定），
    但**表名和列名不能用参数绑定** —— 这是 SQL 的固有限制。
    所以这里的标识符处理必须双层防护：
      ① 清洗（白名单正则）—— 解决「名字合法」
      ② 双引号包裹      —— 解决「名字恰好是关键字」
    两层职责不同、互不替代，缺一层都会出问题。
"""

from __future__ import annotations

import io
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src import config as cfg


# ---------------------------------------------------------------------------
# 一、常量
# ---------------------------------------------------------------------------

# 【编码探测顺序为什么必须是这样？】
#   GB18030 是 GBK 的超集，且**几乎任何字节序列都能被它解码成功**（只是解出乱码）。
#   所以顺序绝不能反过来：先试 GB18030 会把 UTF-8 文件解成乱码且「不报错」，
#   用户要看到一堆乱码才发现问题。
#   必须先试严格的 UTF-8 —— 它解码失败是明确信号，能可靠地触发回退。
#
#   utf-8-sig 放最前是为了吃掉 BOM：否则第一列列名会变成 "\ufeffdate"，
#   用户看到的表头里藏着一个不可见字符，排查成本极高。
_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "big5")

# 标识符白名单：只放行 ASCII 字母数字下划线 + 中日韩统一表意文字。
# 【为什么用白名单而不是黑名单？】
#   黑名单永远列不全（引号、分号、注释符、换行、Unicode 同形字符……），
#   而表名来自用户上传的文件名 —— 全项目唯一的不可信输入。
#   白名单只放行「确定安全」的字符，漏掉某个危险字符的风险为零。
_IDENT_UNSAFE_RE = re.compile(r"[^0-9A-Za-z_\u4e00-\u9fff]")

# 前导零：形如 007 / 0912。这类值几乎都是编号（用户 ID、渠道码、工号），
# 而不是数值 —— 它们的前导零有业务含义，一旦被转成数字就再也找不回来。
_LEADING_ZERO_RE = re.compile(r"^[+-]?0\d")

# 列名里出现这些词，说明它「可能」是日期列
_DATE_NAME_HINT_RE = re.compile(r"(date|time|day|日期|时间|月份|年月)", re.IGNORECASE)

# 值里出现明确的日期分隔符（- 或 /），才算「像日期」。
# 【为什么要求分隔符？】因为 pd.to_datetime 极其宽容，会把纯数字
# "20260919" 当日期、甚至把 1,2,3 当成 1970 年后的时间戳 ——
# 这是 pandas 最著名的静默错误之一。先设一道门槛，宁可少识别，不可错识别。
_DATE_VALUE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")

# pandas 生成的占位列名，如 "Unnamed: 0"（Excel 导出的 CSV 里极常见）
_UNNAMED_RE = re.compile(r"^Unnamed: \d+$")


# ---------------------------------------------------------------------------
# 二、结果数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSpec:
    """一列的推断结果。"""

    name: str                    # 原始列名（保留给用户核对）
    sql_name: str                # 落库列名（已清洗、唯一）
    sql_type: str                # INTEGER / REAL / TEXT
    is_date: bool = False        # 是否识别为日期列（落库仍是 TEXT，但会被建索引）
    non_null_count: int = 0
    sample: tuple[str, ...] = ()  # 前 3 个非空样本值，给用户核对推断是否正确
    renamed: bool = False        # 原名是否被清洗过（前端据此提示）

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sql_name": self.sql_name,
            "sql_type": self.sql_type,
            "is_date": self.is_date,
            "non_null_count": self.non_null_count,
            "sample": list(self.sample),
            "renamed": self.renamed,
        }


@dataclass
class UploadResult:
    """一个上传文件的处理结果。

    【为什么失败不抛异常，而是塞进 error？】
      因为多文件上传时，用户需要看到「哪个文件成功、哪个失败、为什么」。
      抛异常只能报告第一个错误，后面的文件就没机会被检查了 ——
      用户得改一个、传一次、再改一个、再传一次。与项目里
      ToolExecutor / QueryExecutor「失败转成结果对象」是同一个原则。
    """

    ok: bool
    source_file: str
    table_name: str = ""
    row_count: int = 0
    columns: list[ColumnSpec] = field(default_factory=list)
    encoding: str = ""
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source_file": self.source_file,
            "table_name": self.table_name,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
            "encoding": self.encoding,
            "warnings": list(self.warnings),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# 三、解码
# ---------------------------------------------------------------------------

def decode_csv_bytes(raw: bytes) -> tuple[str, str]:
    """把上传的字节流解码成文本，返回 (文本, 实际使用的编码)。

    【为什么自己解码，而不是把 encoding 参数交给 pd.read_csv？】
      因为 pandas 在编码错误时可能抛错、也可能静默替换成 U+FFFD，
      行为取决于版本与 errors 参数。自己先解码，就能在解码阶段
      一次性确定「编码是否真的成立」，并把结果明确告诉用户。
      「我知道我用了什么编码」比「猜对了」更重要 —— 猜错了要能被发现。
    """
    for enc in _ENCODING_CANDIDATES:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    # 全部候选都失败：用替换字符兜底，但编码名会显示出来供用户判断。
    # 【为什么宁可带乱码也不拒绝？】因为一个字节的问题不该让整份数据被拒，
    # 而且前端会显示「编码：utf-8(replace)」，用户能看出这里有问题。
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


# ---------------------------------------------------------------------------
# 四、列名清洗
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """清洗列名：去空白、去重、补空名、删全空列。返回 (新 df, 警告列表)。

    要处理的现实情况（Excel 导出的 CSV 里全都常见）：
      · "Unnamed: 0"      —— 索引列被写进文件，全空，直接删掉
      · "金额 (元)"        —— 含空格、括号、中文
      · 两列都叫 "value"   —— pandas 会自行改名成 value.1，需要统一成 value_2
      · 空列名
    """
    warnings: list[str] = []
    rename: dict[Any, str] = {}
    drop: list[Any] = []
    seen: dict[str, int] = {}

    for idx, raw_col in enumerate(df.columns):
        raw = str(raw_col)
        name = raw.strip()

        # --- 全空列：pandas 的 Unnamed 占位列，或名字就是空的 ---
        if _UNNAMED_RE.match(name) or name == "":
            is_empty = df[raw_col].astype(str).str.strip().eq("").all()
            if is_empty:
                drop.append(raw_col)
                warnings.append(f"已删除全空列 {raw!r}")
                continue
            name = f"col_{idx + 1}"
            warnings.append(f"列名缺失，已命名为 {name!r}")

        # --- 白名单清洗 ---
        clean = _IDENT_UNSAFE_RE.sub("_", name)
        clean = re.sub(r"_+", "_", clean).strip("_")
        if not clean:
            clean = f"col_{idx + 1}"
        if clean[0].isdigit():
            # SQL 标识符可以以数字开头（加引号后），但可读性差且易与数字字面量混淆
            clean = "c_" + clean

        # --- 去重 ---
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 1

        rename[raw_col] = clean
        if clean != raw:
            warnings.append(f"列名已调整：{raw!r} → {clean!r}")

    if drop:
        df = df.drop(columns=drop)
    df = df.rename(columns=rename)
    return df, warnings


def quote_ident(name: str) -> str:
    """把标识符包成 SQLite 双引号形式（内部双引号翻倍转义）。

    【清洗过了为什么还要再包一层？】
      两层防护的职责完全不同：
        · 清洗解决「名字合法」—— 去掉引号、空格、分号这些破坏结构的东西；
        · 双引号解决「名字被当成关键字」—— 清洗后恰好叫 index / group / order
          的列，不加引号会让 SQL 直接语法错误。
      互相不能替代。
    """
    return '"' + name.replace('"', '""') + '"'


def sanitize_table_name(file_name: str, existing: set[str]) -> str:
    """从上传文件名推导安全、唯一、可读的落库表名。

    规则：取文件名主干 → 转小写 → 白名单过滤 → 首字符不能是数字（否则加 t_ 前缀）
          → 空则用 t_<短哈希> → 与已有表重名则追加 _2 / _3

    【为什么要强制转小写？】
      两个理由：
        ① 项目现有 9 张表（src/data/schema.sql）全部是小写 snake_case，
           上传表跟随同一约定，整个库的命名风格才一致；
        ② SQL 校验的白名单比对是大小写敏感的（validator 用集合做差集），
           如果上传的表叫 "Sales" 而指标模板里写 "sales"，会被误判成
           「引用了未授权的表」。统一小写就把这个坑彻底消掉了。
    """
    stem = Path(file_name).stem
    clean = _IDENT_UNSAFE_RE.sub("_", stem)
    clean = re.sub(r"_+", "_", clean).strip("_").lower()

    if not clean:
        import hashlib

        clean = "t_" + hashlib.md5(file_name.encode("utf-8")).hexdigest()[:8]
    if clean[0].isdigit():
        clean = "t_" + clean
    clean = clean[:50]

    if clean not in existing:
        return clean
    n = 2
    while f"{clean}_{n}" in existing:
        n += 1
    return f"{clean}_{n}"


# ---------------------------------------------------------------------------
# 五、类型推断
# ---------------------------------------------------------------------------

def _infer_scalar_type(values: list[str]) -> str:
    """对一组「已确认非空」的字符串值推断 SQLite 类型。"""
    # --- 第一道：前导零列直接判 TEXT ---
    # 【为什么放在最前面？】因为它是唯一一条「即使能转成数字也不该转」的规则。
    #   "007" 用 int() 能成功（得 7），但它几乎肯定是编号而不是数值。
    #   用「无损往返」判据（str(int(x)) == x）也能拦下 "007"，
    #   但会把 "0" 也卷进来 —— "0" 是合法数值，不该被判成文本。
    #   显式的前导零规则比往返判据更准确地表达了这个意图。
    if any(_LEADING_ZERO_RE.match(v) for v in values):
        return "TEXT"

    # --- 第二道：无损整数 ---
    # 【为什么用「无损往返」而不是 pd.to_numeric？】
    #   to_numeric 会静默改数据："007" → 7、1.0 → 1。
    #   用户 ID、渠道编码、手机号前缀这类列的前导零是有业务含义的，
    #   吃掉就再也找不回来，而且查询不会报错 —— 是最难发现的一类数据损坏。
    is_int = True
    for v in values:
        try:
            if str(int(v)) != v:
                is_int = False
                break
        except ValueError:
            is_int = False
            break
    if is_int:
        return "INTEGER"

    # --- 第三道：浮点 ---
    try:
        for v in values:
            float(v)
        return "REAL"
    except ValueError:
        return "TEXT"


def infer_column_types(df: pd.DataFrame, sample_rows: int = 1000) -> dict[str, str]:
    """推断每列的 SQLite 类型（INTEGER / REAL / TEXT）。

    只抽样前 sample_rows 行 —— 大文件全量扫描会明显变慢。
    【抽样是有代价的】第 N+1 行如果是脏数据，落库时仍可能出错，
    所以真正的写入路径还要用容错方式（见 _coerce_value）。
    """
    types: dict[str, str] = {}
    head = df.head(sample_rows)

    for col in df.columns:
        # 【为什么用 .tolist() 转成 Python 原生字符串再判断？】
        #   两个原因：① 绕开 pandas 3.0 的 dtype 变更（默认字符串 dtype
        #   从 object 变成了 str/StringDtype，写 df.dtypes == object 会恒为 False）；
        #   ② 我们读 CSV 时统一用 dtype=str，所以这里拿到的就是原始文本，
        #   正好满足「无损往返」判据需要的「原始文本」前提。
        values = [str(v).strip() for v in head[col].tolist() if str(v).strip() != ""]
        types[col] = _infer_scalar_type(values) if values else "TEXT"
    return types


def infer_date_columns(
    df: pd.DataFrame,
    types: dict[str, str],
    sample_rows: int = 200,
) -> set[str]:
    """识别日期列。

    【为什么要两道门，而不直接对每列跑 pd.to_datetime？】
      因为它太宽容了：会把 1、2、3 解析成 1970 年后的时间戳，
      把 "20260919" 当成日期，把 "NA" 当成 NaT。
      所以设两道门 —— **列名像日期** 或 **样本里 80% 以上出现明确日期分隔符**
      —— 两者过其一才进入解析尝试。宁可少识别，不可错识别。
    """
    date_cols: set[str] = set()
    head = df.head(sample_rows)

    for col in df.columns:
        # 数字列不可能是日期（日期在 CSV 里是文本）
        if types.get(col) != "TEXT":
            continue

        values = [str(v).strip() for v in head[col].tolist() if str(v).strip() != ""]
        if not values:
            continue

        name_hint = bool(_DATE_NAME_HINT_RE.search(str(col)))
        hits = sum(1 for v in values if _DATE_VALUE_RE.match(v))
        value_hint = hits >= max(1, int(len(values) * 0.8))

        if not (name_hint or value_hint):
            continue

        try:
            # format="mixed" 允许同一列里混用 - 和 / 分隔符（如 2026-09-10 和 2026/9/10）。
            # 【为什么必须加？】pandas 2.x 起会按第一个值推断**单一格式**，
            #   后续不同格式的值被 coerce 成 NaT —— 于是混了 - 和 / 的整列都落选，
            #   data_start/data_end 变成 None，提示词告诉模型一个错误的窗口。
            #   两道门（列名/值门槛 + 全部可解析）仍然保留，只是放宽了「格式必须统一」这条
            #   不合理的隐含约束。
            parsed = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed")
        except (ValueError, TypeError):
            continue
        # 要求全部可解析：只要有一个解析不出来，就说明这列不是纯日期
        if parsed.notna().all():
            date_cols.add(col)

    return date_cols


def _to_iso_date(value: str) -> str:
    """把日期文本规范成 ISO（YYYY-MM-DD）；解析不出来就原样返回。

    【为什么必须规范化？】因为日期在库里存的是 TEXT，指标的 SQL 模板
    靠字符串比较做 BETWEEN（如 `date BETWEEN '2026-09-10' AND '2026-09-16'`）。
    如果源文件里混着 "2026/9/7" 这种写法，字符串比较会得出错误结果 ——
    "2026/9/7" > "2026-09-16"（因为 '/' 的 ASCII 码大于 '-'）。
    这是「格式不统一导致指标静默算错」的典型案例。
    """
    try:
        ts = pd.to_datetime(value, errors="raise")
    except (ValueError, TypeError):
        return value
    return ts.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 六、建表与写入
# ---------------------------------------------------------------------------

def build_create_table_sql(table: str, columns: list[ColumnSpec]) -> str:
    """生成 CREATE TABLE 语句。

    【为什么自己写 DDL，而不是让 df.to_sql 自动建表？】
      因为 to_sql 用 pandas 的 dtype 映射来定列类型，而 dtype 是 read_csv 猜的 ——
      前导零、"NA" 被当成 NaN、日期被当字符串，这些问题会原样落进表结构，
      之后再想改就要重建库。我们的类型决策（无损整数、日期存 TEXT）
      必须由 DDL 明确表达出来。
    """
    cols = ",\n  ".join(f"{quote_ident(c.sql_name)} {c.sql_type}" for c in columns)
    return f"CREATE TABLE {quote_ident(table)} (\n  {cols}\n)"


def _coerce_value(value: Any, spec: ColumnSpec) -> Any:
    """把单元格值转成可安全写入的类型。

    空串统一写成 NULL（SQLite 里空串和 NULL 语义不同，
    空串会参与 COUNT(col)、影响聚合结果）。
    """
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    if spec.is_date:
        return _to_iso_date(text)
    if spec.sql_type == "INTEGER":
        try:
            return int(text)
        except ValueError:
            return text  # 抽样没覆盖到的脏数据：降级成文本而不是丢弃整行
    if spec.sql_type == "REAL":
        try:
            return float(text)
        except ValueError:
            return text
    return text


def create_table_from_dataframe(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table: str,
    columns: list[ColumnSpec],
) -> None:
    """建表 + 批量写入。

    【为什么日期存 TEXT 而不是声明 DATE？】
      SQLite 没有真正的 DATE 类型。声明成 DATE 会得到 NUMERIC 亲和性，
      写入 '2026-09-17' 时会被尝试转成数字，语义反而更糟。
      项目现有 9 张表（src/data/schema.sql）里 date / event_date 也都是 TEXT，
      保持同一约定，BETWEEN 的字符串比较才能正确工作。

    【分块公式的来源】照搬 scripts/build_database.py 的做法：
        step = max(1, 900 // n_cols)
      method="multi" 会把 step 行拼成一条 INSERT，单条语句的绑定变量数
      = step × n_cols ≤ 900 < 999（SQLite 老版本的硬上限），保证永不超限。
    """
    conn.execute(build_create_table_sql(table, columns))

    if len(df) == 0:
        return  # 只有表头：建表成功，无数据可写

    col_names = [c.sql_name for c in columns]
    placeholders = ", ".join("?" for _ in col_names)
    insert_sql = (
        f"INSERT INTO {quote_ident(table)} "
        f"({', '.join(quote_ident(n) for n in col_names)}) "
        f"VALUES ({placeholders})"
    )

    n_cols = max(1, len(columns))
    step = max(1, 900 // n_cols)

    rows: list[tuple[Any, ...]] = []
    for record in df.itertuples(index=False, name=None):
        rows.append(tuple(_coerce_value(v, spec) for v, spec in zip(record, columns)))

    for start in range(0, len(rows), step):
        conn.executemany(insert_sql, rows[start : start + step])


# ---------------------------------------------------------------------------
# 七、单文件处理
# ---------------------------------------------------------------------------

def _read_csv_text(text: str) -> tuple[pd.DataFrame, list[str]]:
    """解析 CSV 文本。分隔符不是逗号时会自动重试。"""
    warnings: list[str] = []

    # 统一按字符串读、不做 NA 推断：这样「原始文本」得以保留，
    # 「无损往返」判据才有依据（见 infer_column_types 的说明）。
    df = pd.read_csv(
        io.StringIO(text),
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )

    # 只解析出 1 列 → 大概率分隔符不是逗号
    if df.shape[1] == 1:
        first_line = text.split("\n", 1)[0]
        for sep in ("\t", ";", "|"):
            if sep in first_line:
                retry = pd.read_csv(
                    io.StringIO(text),
                    sep=sep,
                    dtype=str,
                    keep_default_na=False,
                    na_values=[],
                )
                if retry.shape[1] > 1:
                    warnings.append(f"检测到分隔符为 {sep!r}，已按此解析")
                    return retry, warnings

    return df, warnings


def _process_one_file(
    file_name: str,
    raw: bytes,
    existing_tables: set[str],
    max_rows: int,
    max_columns: int,
    max_bytes: int,
) -> tuple[UploadResult, pd.DataFrame | None, list[ColumnSpec] | None]:
    """处理单个文件：校验 → 解码 → 解析 → 清洗 → 推断。

    返回 (结果, 清洗后的 df, 列规格)；失败时后两项为 None。
    """
    start = time.perf_counter()

    def fail(msg: str, encoding: str = "") -> tuple[UploadResult, None, None]:
        return (
            UploadResult(
                ok=False,
                source_file=file_name,
                encoding=encoding,
                error=msg,
                elapsed_ms=(time.perf_counter() - start) * 1000,
            ),
            None,
            None,
        )

    # --- 大小校验放在最前：不浪费内存去解析一个注定要被拒的文件 ---
    if len(raw) == 0:
        return fail("文件为空（0 字节）")
    if len(raw) > max_bytes:
        return fail(
            f"文件超过大小上限：{len(raw) / 1024 / 1024:.1f} MB > "
            f"{max_bytes / 1024 / 1024:.0f} MB"
        )

    text, encoding = decode_csv_bytes(raw)
    if not text.strip():
        return fail("文件没有可解析的内容", encoding)

    try:
        df, parse_warnings = _read_csv_text(text)
    except (pd.errors.ParserError, ValueError) as exc:
        return fail(f"CSV 解析失败：{exc}", encoding)

    if df.shape[1] == 0:
        return fail("未解析出任何列", encoding)
    if df.shape[1] > max_columns:
        return fail(
            f"列数超过上限：{df.shape[1]} > {max_columns}", encoding
        )
    if len(df) > max_rows:
        return fail(
            f"行数超过上限：{len(df)} > {max_rows}，建议拆分后再上传", encoding
        )

    df, col_warnings = normalize_columns(df)
    if df.shape[1] == 0:
        return fail("清洗后没有可用的列（所有列都是空的）", encoding)

    warnings = parse_warnings + col_warnings
    if len(df) == 0:
        # 【为什么允许「只有表头」？】因为用户可能先传个模板确认结构。
        # 此时无法推断类型，全部按 TEXT 建表 —— 这是唯一诚实的选择。
        warnings.append("文件只有表头、没有数据行，已按 TEXT 建表（无法推断类型）")

    types = infer_column_types(df)
    date_cols = infer_date_columns(df, types)

    columns: list[ColumnSpec] = []
    for original, sql_name in zip(df.columns, df.columns):
        non_null = int((df[sql_name].astype(str).str.strip() != "").sum())
        samples = tuple(
            str(v).strip()
            for v in df[sql_name].tolist()
            if str(v).strip() != ""
        )[:3]
        columns.append(
            ColumnSpec(
                name=sql_name,
                sql_name=sql_name,
                sql_type=types.get(sql_name, "TEXT"),
                is_date=sql_name in date_cols,
                non_null_count=non_null,
                sample=samples,
                renamed=False,
            )
        )

    table = sanitize_table_name(file_name, existing_tables)
    if table != Path(file_name).stem:
        warnings.append(f"表名已调整：{Path(file_name).stem!r} → {table!r}")

    result = UploadResult(
        ok=True,
        source_file=file_name,
        table_name=table,
        row_count=len(df),
        columns=columns,
        encoding=encoding,
        warnings=warnings,
        elapsed_ms=(time.perf_counter() - start) * 1000,
    )
    return result, df, columns


# ---------------------------------------------------------------------------
# 八、主入口
# ---------------------------------------------------------------------------

def import_csv_files(
    files: list[tuple[str, bytes]],
    db_path: str | Path,
    raw_dir: str | Path,
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_mb: int | None = None,
) -> tuple[list[UploadResult], dict[str, Any]]:
    """把一批上传文件写成一个全新的 SQLite 库。

    返回 (每个文件的结果, 汇总信息)。

    【为什么是「全新库」而不是「往已有库里加表」？】
      与 scripts/build_database.py 同一条纪律：删旧建新，保证结果与输入一致、
      可复现。往已有库里加表会让「同一个数据集目录下混着两次导入的数据」，
      换指标方案重导时也说不清哪些表是哪次的。

    【为什么原始 CSV 要另存一份？】
      因为「可重建」是本项目一贯的要求。上传的字节流只在这次请求里存在，
      不落盘的话，库一旦损坏或需要换方案重导，数据就彻底没了。
      文件名冲突时追加 _2 后缀，不覆盖 —— 保留历史版本比省磁盘更重要。
    """
    db_path = Path(db_path)
    raw_dir = Path(raw_dir)
    max_rows = cfg.UPLOAD_MAX_ROWS if max_rows is None else max_rows
    max_columns = cfg.UPLOAD_MAX_COLUMNS if max_columns is None else max_columns
    max_mb = cfg.UPLOAD_MAX_MB if max_mb is None else max_mb
    max_bytes = max_mb * 1024 * 1024

    results: list[UploadResult] = []
    prepared: list[tuple[UploadResult, pd.DataFrame, list[ColumnSpec]]] = []
    existing: set[str] = set()

    for file_name, raw in files:
        result, df, columns = _process_one_file(
            file_name, raw, existing, max_rows, max_columns, max_bytes
        )
        results.append(result)
        if result.ok and df is not None and columns is not None:
            existing.add(result.table_name)
            prepared.append((result, df, columns))

    summary: dict[str, Any] = {
        "ok": False,
        "db_path": str(db_path),
        "raw_dir": str(raw_dir),
        "tables": [],
        "data_start": None,
        "data_end": None,
        "warnings": [],
    }

    # 【任何一个文件失败 → 整体不落库、不登记】
    # 理由见 dataset.py::add_from_upload 的说明：部分成功的多表数据集
    # 对下游是灾难（指标模板静默查不到表）。
    if not prepared or len(prepared) != len(files):
        summary["message"] = "存在导入失败的文件，未创建数据集"
        return results, summary

    # --- 建库 ---
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # 删旧建新（可复现）

    conn = sqlite3.connect(db_path)
    try:
        # 导入期放宽持久化：这是「一次性灌数据」，崩了重来即可，
        # 不值得为每条 INSERT 付 fsync 的代价。导入完再改回来。
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("BEGIN")

        for result, df, columns in prepared:
            create_table_from_dataframe(conn, df, result.table_name, columns)

        conn.commit()

        # --- 索引在数据导入之后创建 ---
        # 【为什么顺序不能反？】先建索引再导数据，每插一行都要维护 B+ 树，
        # 实测慢 3~5 倍。批量导入的标准做法是「先灌数据、后建索引」。
        date_indexes: list[tuple[str, str]] = []
        for result, _df, columns in prepared:
            for spec in columns:
                if spec.is_date:
                    idx = f"idx_{result.table_name}_{spec.sql_name}"
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {quote_ident(idx)} "
                        f"ON {quote_ident(result.table_name)} ({quote_ident(spec.sql_name)})"
                    )
                    date_indexes.append((result.table_name, spec.sql_name))

        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()

    # --- 落原始 CSV ---
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    for file_name, raw in files:
        target = raw_dir / Path(file_name).name
        n = 2
        while target.exists():
            target = raw_dir / f"{Path(file_name).stem}_{n}{Path(file_name).suffix}"
            n += 1
        target.write_bytes(raw)
        saved_files.append(target.name)

    # --- 汇总 ---
    # 数据窗口：取所有日期列的最小 / 最大值。
    # 【为什么取不到就留 None？】None 表示「未识别」，而不是「没有数据」——
    # 上层会回落到 config 的窗口。把「不知道」和「没有」区分开，
    # 才不会让界面显示一个和数据无关的窗口（那正是改造前 overview.py 的问题）。
    dmin: str | None = None
    dmax: str | None = None
    for _result, df, columns in prepared:
        for spec in columns:
            if not spec.is_date:
                continue
            vals = [v for v in df[spec.sql_name].tolist() if str(v).strip()]
            iso = sorted(_to_iso_date(str(v)) for v in vals)
            if not iso:
                continue
            if dmin is None or iso[0] < dmin:
                dmin = iso[0]
            if dmax is None or iso[-1] > dmax:
                dmax = iso[-1]

    summary.update(
        {
            "ok": True,
            # 【为什么这里手工拼，而不是 [r.to_dict() for r in results]？】
            #   UploadResult.to_dict() 是给**前端**看的 DTO（字段名 table_name），
            #   而 DatasetStore.add_from_upload 要的是**数据层**的表快照（字段名 name）。
            #   两者复用同一个 dict 会埋下「键名对不上」的暗雷 ——
            #   而且只在真正走上传路径时才炸，单测不覆盖就发现不了。
            #   所以这里显式构造数据层需要的形状，两个 DTO 各自独立演进。
            "tables": [
                {
                    "name": result.table_name,
                    "row_count": result.row_count,
                    "source_file": result.source_file,
                    "columns": [spec.to_dict() for spec in columns],
                }
                for result, _df, columns in prepared
            ],
            "raw_files": saved_files,
            "data_start": dmin,
            "data_end": dmax,
            "total_rows": sum(r.row_count for r in results),
            "table_count": len(prepared),
        }
    )
    return results, summary
