# -*- coding: utf-8 -*-
"""
CSV 上传建表单元测试（Phase 7）
================================================================================
被测对象：src/data/upload.py 的解码 / 列名清洗 / 类型推断 / 建表导数。

【为什么这个模块能脱离浏览器直接测？】
  因为它是纯函数库 —— 输入是 bytes，输出是结构化结果对象，不 import streamlit。
  这正是设计时的目标：边界情况（GBK 编码、前导零、空文件、只有表头、
  分隔符不是逗号）都能用 pytest 自动化覆盖，而不用一次一次手点页面。

【测试分四组】
  一、解码 —— 编码探测正确且顺序严格
  二、标识符处理 —— 列名 / 表名的清洗与去重（含关键字、中文、数字开头）
  三、类型与日期推断 —— 无损往返、前导零、日期识别两道门
  四、集成路径 —— import_csv_files 建真库 + summary 键名契约

【这份文件里有一条「回归锁」】
  test_import_summary_tables_have_name_key 专门锁住「summary['tables'] 里
  必须是 name 而不是 table_name」这个键名契约。它曾是一个只在真正走上传
  路径时才炸的暗雷（DatasetStore.add_from_upload 读 t['name']，而初版
  用的是前端 DTO 的 table_name）—— 单测不覆盖就永远发现不了。
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from src.data.upload import (
    ColumnSpec,
    build_create_table_sql,
    create_table_from_dataframe,
    decode_csv_bytes,
    import_csv_files,
    infer_column_types,
    infer_date_columns,
    normalize_columns,
    quote_ident,
    sanitize_table_name,
)

# 造一个最简但有代表性的数据集：日期、前导零编号、纯数字、浮点、文本全都有
DEMO_CSV = """\
date,channel_id,user_id,revenue,note
2026-09-10,007,u_001,12.5,hello
2026-09-11,007,u_002,0,world
2026-09-12,0912,u_003,99.9,中文备注
2026-09-13,0912,u_004,3.14,
"""

DEMO_BYTES = DEMO_CSV.encode("utf-8")


def _import(tmp_path: Path, files=None, **kwargs) -> tuple[list, dict]:
    """import_csv_files 的薄封装：默认塞这份合理的 demo，并指向沙箱目录。"""
    files = files if files is not None else [("demo_data.csv", DEMO_BYTES)]
    db = tmp_path / "ds" / "data.db"
    raw = tmp_path / "ds" / "raw"
    return import_csv_files(files, db_path=db, raw_dir=raw, **kwargs)


# ---------------------------------------------------------------------------
# 一、解码
# ---------------------------------------------------------------------------

def test_utf8_without_bom_decodes_as_utf8():
    text, enc = decode_csv_bytes(DEMO_BYTES)
    assert "中文备注" in text
    assert enc in ("utf-8", "utf-8-sig")


def test_utf8_bom_is_stripped():
    """BOM 必须被吃掉，否则第一列列名会变成 \\ufeffdate。

    utf-8-sig 排在最前就是为了这个：否则表头里藏一个不可见字符，
    用户看到 date 但程序里其实是 \ufeffdate，排查成本极高。
    """
    raw = b"\xef\xbb\xbf" + DEMO_BYTES
    text, enc = decode_csv_bytes(raw)
    assert enc == "utf-8-sig"
    assert not text.startswith("\ufeff")


def test_gbk_decodes_to_gb18030():
    """GBK 文件（本项目常见的国内运营数据）解码成功且编码名正确。"""
    utf8_text = "名称,数值\n苹果,3\n香蕉,5\n"
    gbk_bytes = utf8_text.encode("gbk")
    text, enc = decode_csv_bytes(gbk_bytes)
    assert enc == "gb18030"  # gb18030 是 gbk 的超集
    assert "苹果" in text


def test_encoding_order_utf8_before_gb18030():
    """编码探测顺序不能变：必须先试严格的 UTF-8。

    因为 GB18030 几乎能解任何字节（只是解成乱码），顺序一反过来，
    UTF-8 文件就会被解成乱码且不报错。断言「解出来和原文一致」
    就是在锁这个不变式 —— 编码名本身是 utf-8 还是 utf-8-sig 无所谓。
    """
    original = "café,1\n"
    text, enc = decode_csv_bytes(original.encode("utf-8"))
    assert text == original
    assert enc in ("utf-8-sig", "utf-8")


def test_undecodable_bytes_falls_back_with_replace():
    """全部候选都失败时用替换字符兜底，但编码名里带 (replace) 供用户判断。

    【为什么宁可带乱码也不拒绝？】一个字节的问题不该让整份数据被拒。
    """
    raw = bytes([0xFF, 0xFE, 0x00, 0x41])  # 无效 UTF-8，gb18030 也未必成立
    text, enc = decode_csv_bytes(raw)
    assert enc == "utf-8(replace)"


# ---------------------------------------------------------------------------
# 二、标识符处理
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("金额 (元)", "金额_元"),        # 中文+空格+括号
        ("  Sales  ", "Sales"),          # 头尾空白
        ("index", "index"),              # 关键字：清洗放行，双引号处理
        ("group", "group"),
        ("order", "order"),
        ("007", "c_007"),                # 数字开头加前缀
        ("a:b:c", "a_b_c"),
    ],
)
def test_normalize_columns_cleans_names(raw, expected):
    import pandas as pd

    df = pd.DataFrame({raw: [1]})
    out, _warnings = normalize_columns(df)
    assert list(out.columns) == [expected]


def test_normalize_columns_dedupes():
    """清洗后重名的列追加 _2，不能出现两个同名 sql_name。

    【为什么用 "a b" + "a_b" 这组？】因为它们**清洗后**才撞名 ——
    这才是真正的重名来源。而 pandas 读 CSV 时遇到重名列会自行改成
    value.1，那类名字清洗后反而不撞（. 变成 _）。
    """
    import pandas as pd

    df = pd.DataFrame({"a b": [1], "a_b": [2]})
    out, _w = normalize_columns(df)
    assert list(out.columns) == ["a_b", "a_b_2"]


def test_normalize_columns_drops_empty_unnamed():
    """Excel 导出的 Unnamed: 0 全空列要删掉，不能留在表里。

    【注意是全空才算】只要该列有任何一个值，它就会被改名保留（col_N）——
    删掉有数据的列是不可接受的。
    """
    import pandas as pd

    df = pd.DataFrame({"Unnamed: 0": ["", ""], "ok": ["1", "2"]})
    out, warnings = normalize_columns(df)
    assert list(out.columns) == ["ok"]
    assert any("删除" in w for w in warnings)


def test_quote_ident_doubles_internal_quotes():
    """双引号包裹 + 内部引号翻倍：清洗解决「合法」，这里解决「不破坏结构」。"""
    assert quote_ident("plain") == '"plain"'
    assert quote_ident('a"b') == '"a""b"'


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Sales Report.csv", "sales_report"),      # 主干 + 小写
        ("007 数据.txt", "t_007_数据"),             # 数字开头加 t_ 前缀
        ("Order.Csv", "order"),                    # 大小写与扩展名
    ],
)
def test_sanitize_table_name(name, expected):
    assert sanitize_table_name(name, set()) == expected


def test_sanitize_table_name_hashes_when_stem_is_empty():
    """文件名主干全是非法字符时，用短哈希兜底（保证合法且稳定）。"""
    got = sanitize_table_name("..  ..", set())
    assert re.fullmatch(r"t_[0-9a-f]{8}", got)
    assert got == sanitize_table_name("..  ..", set())  # 稳定


def test_sanitize_table_name_dedupes():
    assert sanitize_table_name("a.csv", {"a"}) == "a_2"
    assert sanitize_table_name("a.csv", {"a", "a_2"}) == "a_3"


# ---------------------------------------------------------------------------
# 三、类型 / 日期推断
# ---------------------------------------------------------------------------

def test_leading_zero_column_is_text():
    """前导零列（007 / 0912）判 TEXT，且必须先于「能转数字」判据。

    「007」用 int() 能成功（得 7），但它几乎肯定是编号而不是数值；
    前导零有业务含义，一旦被转成数字就再也找不回来。
    """
    import pandas as pd

    df = pd.DataFrame({"code": ["007", "0912", "123"]})
    types = infer_column_types(df)
    assert types["code"] == "TEXT"


def test_lossless_int_is_integer():
    """能无损往返成整数的列才判 INTEGER（0 是合法数值，不能判文本）。"""
    import pandas as pd

    df = pd.DataFrame({"n": ["7", "0", "99"]})
    assert infer_column_types(df)["n"] == "INTEGER"


def test_lossy_number_falls_back_to_real():
    import pandas as pd

    df = pd.DataFrame({"f": ["1.5", "2", "9.9"]})
    assert infer_column_types(df)["f"] == "REAL"


def test_text_with_letters_stays_text():
    import pandas as pd

    df = pd.DataFrame({"t": ["abc", "x", "hello world"]})
    assert infer_column_types(df)["t"] == "TEXT"


def test_date_column_detection_requires_hint():
    """日期识别要过两道门：列名像日期 或 值里带明确分隔符。

    【为什么设两道门】pd.to_datetime 太宽容：会把 1、2、3 解析成
    1970 年后的时间戳，把 "20260919" 当日期。宁可少识别不可错识别。
    """
    import pandas as pd

    # 列名带 date → 命中（即使分隔符 < 80%）
    df1 = pd.DataFrame({"purchase_date": ["2026-09-10", "2026-09-11"]})
    date_cols1 = infer_date_columns(df1, infer_column_types(df1))
    assert "purchase_date" in date_cols1

    # 值里 100% 是明确日期 → 命中
    df2 = pd.DataFrame({"when": ["2026/9/1", "2026/9/2", "2026/9/3"]})
    date_cols2 = infer_date_columns(df2, infer_column_types(df2))
    assert "when" in date_cols2

    # 纯数字 + 列名不像日期 → 拒绝（否则 007 会被当成日期吃掉）
    df3 = pd.DataFrame({"code": ["007", "008", "009"]})
    date_cols3 = infer_date_columns(df3, infer_column_types(df3))
    assert "code" not in date_cols3


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
def test_column_named_like_date_but_unparseable_is_not_date():
    """列名像日期、值却解析不出来 → 不判日期。

    【为什么这条重要】列名只是「线索」，最终必须靠值能解析出来才算数。
    否则一个叫 update_time 但存着「刚刚 / 昨天」的列会被建上日期索引，
    指标按它做 BETWEEN 时会得到一堆空结果 —— 而且不报错。
    """
    import pandas as pd

    df = pd.DataFrame({"update_time": ["刚刚", "昨天", "前天"]})
    assert infer_date_columns(df, infer_column_types(df)) == set()


def test_mixed_date_formats_are_detected(tmp_path):
    """同一列里混用 - 和 / 分隔符时，仍然被识别为日期列。

    【为什么这条曾经是已知限制】pandas 2.x 按第一个值推断**单一格式**，
    混合格式的后续值被 coerce 成 NaT，整列落选。后果是安全的但丢信息：
    不建日期索引、data_start/data_end 为 None，前端显示「数据窗口：未知」。

    【修复方式】给 pd.to_datetime 加 format="mixed"，两道门与
    「全部可解析」判据都保留，只是放宽了「格式必须统一」这条隐含约束。
    """
    import pandas as pd

    df = pd.DataFrame({"date": ["2026-09-10", "2026-09-11", "2026/9/12"]})
    assert "date" in infer_date_columns(df, infer_column_types(df))


def test_build_create_table_sql_quotes_keywords():
    """DDL 里关键字列名也要被双引号包住，否则 CREATE TABLE 直接语法错误。"""
    specs = [ColumnSpec(name="a", sql_name="index", sql_type="INTEGER")]
    sql = build_create_table_sql("t", specs)
    assert '"index" INTEGER' in sql
    assert sql.startswith("CREATE TABLE \"t\"")


# ---------------------------------------------------------------------------
# 四、集成路径：建真库 + 汇总契约
# ---------------------------------------------------------------------------

def test_import_creates_real_db_and_writes_data(tmp_path):
    """真正的端到端：bytes → 一个能查的表，列类型与窗口都正确。"""
    results, summary = _import(tmp_path)
    assert summary["ok"] is True
    assert len(results) == 1 and results[0].ok
    assert results[0].table_name == "demo_data"

    conn = sqlite3.connect(str(tmp_path / "ds" / "data.db"))
    try:
        rows = conn.execute("SELECT COUNT(*) FROM demo_data").fetchone()
        assert rows[0] == 4
        assert conn.execute(
            "SELECT revenue FROM demo_data WHERE user_id='u_004'"
        ).fetchone()[0] == pytest.approx(3.14)
        # 前导零保住了：007 不能变成 7
        assert conn.execute(
            "SELECT channel_id FROM demo_data ORDER BY rowid LIMIT 1"
        ).fetchone()[0] == "007"
    finally:
        conn.close()

    assert summary["data_start"] == "2026-09-10"
    assert summary["data_end"] == "2026-09-13"


def test_import_summary_tables_have_name_key(tmp_path):
    """回归锁：summary['tables'] 的元素必须用数据层的 'name' 键。

    【为什么必须锁定？】DatasetStore.add_from_upload 读的是 t['name']，
    而初版误用了上传结果的前端 DTO 键 'table_name'。两个 DTO 各自演进，
    键名一旦错位，只有在真正走上传路径时才炸，且单测不覆盖就发现不了。
    这条用例就是那枚暗雷的回归锁。
    """
    _results, summary = _import(tmp_path)
    assert summary["tables"], "应有至少一张表"
    for t in summary["tables"]:
        assert "name" in t
        assert "table_name" not in t  # 前端 DTO 的键不该漏进数据层契约


def test_import_slash_dates_are_normalized_to_iso(tmp_path):
    """源文件里的 2026/9/1 写法要规范成 2026-09-01。

    【为什么必须规范化】日期在库里存 TEXT，指标模板靠字符串比较做 BETWEEN。
    而 "2026/9/7" > "2026-09-16"（'/' 的 ASCII 码大于 '-'），
    格式不统一会让指标静默算错 —— 这是最典型的一类数据事故。
    """
    csv = "date,value\n2026/9/1,1\n2026/9/2,2\n2026/9/3,3\n"
    _results, summary = _import(tmp_path, files=[("slash.csv", csv.encode("utf-8"))])
    assert summary["ok"] is True
    assert summary["data_start"] == "2026-09-01"
    assert summary["data_end"] == "2026-09-03"

    conn = sqlite3.connect(str(tmp_path / "ds" / "data.db"))
    try:
        assert conn.execute("SELECT date FROM slash LIMIT 1").fetchone()[0] == "2026-09-01"
    finally:
        conn.close()


def test_date_column_gets_an_index(tmp_path):
    """日期列要建索引 —— 而且必须建在数据导入**之后**。

    【为什么顺序不能反】先建索引再导数据，每插一行都要维护 B+ 树，
    实测慢 3~5 倍。批量导入的标准做法是「先灌数据、后建索引」。
    """
    _import(tmp_path)
    conn = sqlite3.connect(str(tmp_path / "ds" / "data.db"))
    try:
        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    finally:
        conn.close()
    assert "idx_demo_data_date" in names


def test_raw_csv_is_archived(tmp_path):
    """原始 CSV 必须另存一份：库坏了 / 要换方案重导时，字节流已经不在请求里了。"""
    _import(tmp_path)
    saved = tmp_path / "ds" / "raw" / "demo_data.csv"
    assert saved.exists()
    assert saved.read_bytes() == DEMO_BYTES


def test_partial_failure_does_not_write_db(tmp_path):
    """任何一个文件失败 → 整体不落库、不登记，且失败原因被描述清楚。"""
    files = [
        ("ok.csv", DEMO_BYTES),
        ("too_big.csv", b"a,b\n" + b"1,2\n" * 5),
    ]
    results, summary = _import(tmp_path, files=files, max_rows=3)
    assert summary["ok"] is False
    assert not (tmp_path / "ds" / "data.db").exists()
    assert results[1].ok is False
    assert "超过上限" in (results[1].error or "")
    # 失败的文件也要落原始 CSV 吗？不落 —— 整批都没入库，留着会让人误以为成功过
    assert not (tmp_path / "ds" / "raw").exists()


def test_empty_file_is_rejected(tmp_path):
    """0 字节文件被拒（大小校验放在解析之前，不浪费内存）。"""
    results, summary = import_csv_files(
        [("empty.csv", b"")],
        db_path=tmp_path / "ds" / "data.db",
        raw_dir=tmp_path / "ds" / "raw",
    )
    assert results[0].ok is False
    assert "空" in (results[0].error or "")
    assert summary["ok"] is False


def test_tab_separated_values_auto_detected(tmp_path):
    """分隔符不是逗号时自动重试（TSV 也能导入）。"""
    tsv = "date\tvalue\n2026-09-10\t42\n2026-09-11\t7\n"
    results, summary = _import(
        tmp_path, files=[("data.tsv", tsv.encode("utf-8"))]
    )
    assert summary["ok"] is True
    conn = sqlite3.connect(str(tmp_path / "ds" / "data.db"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM data").fetchone()[0] == 2
        assert conn.execute("SELECT value FROM data LIMIT 1").fetchone()[0] == 42
    finally:
        conn.close()


def test_header_only_creates_text_table(tmp_path):
    """只有表头 → 建表成功但无法推断类型，全部 TEXT（诚实选择）。"""
    results, summary = _import(
        tmp_path, files=[("schema.csv", "a,b,c\n".encode("utf-8"))]
    )
    assert summary["ok"] is True
    assert results[0].row_count == 0
    conn = sqlite3.connect(str(tmp_path / "ds" / "data.db"))
    try:
        cols = {
            c[1]: c[2] for c in conn.execute("PRAGMA table_info(schema)").fetchall()
        }
        assert all(v == "TEXT" for v in cols.values())
    finally:
        conn.close()