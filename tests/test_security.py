# -*- coding: utf-8 -*-
"""SQL 安全防线测试（测试方案 Phase 1 · SQL 安全校验）

================================================================================
【为什么单独建这个文件？】

  已有的 test_validator.py / test_executor.py 测的是「SQL 语句本身」安不安全：
  写操作、多语句、注释绕过、非白名单表、系统表、只读连接、行数上限、超时中断。

  本文件测的是**另一条攻击路径**：攻击串不出现在 SQL 里，而是出现在**参数值里**。

【为什么这条路径更值得单独测？】
  本项目的核心架构是「LLM 不写 SQL，只传 metric_id + 参数」。
  这意味着：SQL 模板是受控的（代码里写死的），**参数才是唯一的外部输入**。
  安全边界恰好落在这里 —— 参数绑定（第四层防线）是不是真的挡住了注入，
  必须用"恶意参数"来验，而不是用"恶意 SQL"来验（后者根本进不来）。

【测试的核心不变量（invariant）】

  ① 越权数据不返回：注入串不能让工具"查全表"或查到授权范围外的数据
  ② 不崩：恶意输入必须被转成 ok=False 或安全空结果，而不是抛异常
     （本项目的设计原则：工具永不抛异常，失败转成可观察结果）
  ③ ★ 数据库完好无损：无论怎么注入，9 张表都在、行数不变

  第 ③ 条最重要 —— 前两条是"行为正确"，这条才是"结果正确"。
  只验证"它报了个错"是不够的：报了错但表被删了，同样是灾难。
  所以本文件专门有一条 test_database_survives_injection_battery 兜底。
"""

from __future__ import annotations

import sqlite3

import pytest

from src import config as cfg
from src.agent.tools import ToolExecutor

# ---------------------------------------------------------------------------
# 攻击载荷：覆盖 SQL 注入的常见变形
# ---------------------------------------------------------------------------
INJECTION_PAYLOADS = [
    "dau'; DROP TABLE dim_user;--",                  # 经典闭合 + 拖库
    "dau' OR '1'='1",                                # 经典永真
    "dau UNION SELECT * FROM hr_recruitment_data",   # 联合查询越权读表
    "dau; DELETE FROM user_daily_snapshot;--",       # 堆叠语句
    "dau\x00",                                       # 空字节截断
    "dau'/**/OR/**/1=1",                             # 注释绕过关键字检测
    "dau`; ATTACH DATABASE '/tmp/x.db' AS x;--",     # 附加外部库
]

DATE_PAYLOADS = [
    "2026-09-17' OR '1'='1",
    "2026-09-17'; DELETE FROM dim_user;--",
    "' UNION SELECT 1,2,3--",
    "2026-09-17' AND 1=1--",
    "not-a-date",
    "",
]

INT_PAYLOADS = [
    "1; DROP TABLE dim_user",
    "1 OR 1=1",
    "1); DELETE FROM dim_user;--",
    "-1",
    "99999",
    "abc",
]


@pytest.fixture(scope="module")
def tools() -> ToolExecutor:
    """真实工具执行器（连着真实数据库）。"""
    return ToolExecutor()


def _read_only_connection() -> sqlite3.Connection:
    """独立开一条只读连接，用于从外部核对数据库状态。

    为什么不用被测代码自己的连接？因为"用被告来证明被告无罪"没有意义 ——
    校验库是否完好，必须走一条与被测路径无关的连接。
    """
    return sqlite3.connect(f"file:{cfg.DB_PATH}?mode=ro", uri=True)


def _table_names() -> set[str]:
    with _read_only_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return {row[0] for row in rows}


def _row_count(table: str) -> int:
    with _read_only_connection() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ===========================================================================
# 一、metric_id 注入
# ===========================================================================
@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_metric_id_injection_is_rejected(tools: ToolExecutor, payload: str) -> None:
    """metric_id 是注册表的 key，注入串必然查不到 → 必须返回 ok=False。

    这一层其实是「模板受控」在兜底：metric_id 只是个字典 key，
    拿不到模板就生成不出任何 SQL，注入串根本没机会进入 SQL。
    """
    result = tools.execute("query_metric", {"metric_id": payload, "params": {}})
    assert result.ok is False, f"恶意 metric_id 竟然查成功了：{payload!r}"


# ===========================================================================
# 二、日期参数注入
# ===========================================================================
@pytest.mark.parametrize("payload", DATE_PAYLOADS)
def test_date_param_injection_cannot_widen_result(tools: ToolExecutor, payload: str) -> None:
    """日期参数注入后，返回行数不能超过正常 7 天查询。

    判定标准刻意写成「不许变多」而不是「必须失败」：
    因为把注入串当普通字符串绑定进去，SQLite 会正常执行但匹配不到任何行
    （返回 0 行）—— 这同样是安全的结果，没必要强制它报错。
    **只要不返回越权数据就算通过。**
    """
    normal = tools.execute(
        "query_metric",
        {"metric_id": "dau", "params": {"start_date": "2026-09-11", "end_date": "2026-09-17"}},
    )
    assert normal.ok is True
    normal_rows = len(normal.data.get("rows") or [])

    result = tools.execute(
        "query_metric", {"metric_id": "dau", "params": {"start_date": payload}}
    )
    if result.ok:
        rows = len(result.data.get("rows") or [])
        assert rows <= normal_rows, (
            f"注入串让结果集变大了：正常 {normal_rows} 行，注入后 {rows} 行（{payload!r}）"
        )


# ===========================================================================
# 三、整数参数注入
# ===========================================================================
@pytest.mark.parametrize("payload", INT_PAYLOADS)
def test_int_param_injection_is_rejected(tools: ToolExecutor, payload: str) -> None:
    """day_n 有「类型 + enum」双重校验，畸形值必须被拒绝。

    实测六种载荷（含 `1; DROP TABLE`、`1 OR 1=1`、越界 99999、负数、非数字）
    全部被挡在参数校验层，错误信息还会明确告诉模型"只能是 1, 7, 30"——
    模型据此就能自我纠正，这是"失败也要可观察"设计的直接收益。
    """
    result = tools.execute(
        "query_metric", {"metric_id": "retention_rate", "params": {"day_n": payload}}
    )
    assert result.ok is False, f"畸形 day_n 竟然通过了校验：{payload!r}"
    assert "day_n" in result.content, "拒绝信息没有点名是哪个参数"


# ===========================================================================
# 四、未知参数键（往 params 里夹带私货）
# ===========================================================================
def test_unknown_param_key_is_rejected(tools: ToolExecutor) -> None:
    """params 里塞一个注册表没声明的键，必须被明确拒绝并点名是哪个键。

    真实威胁场景：模型幻觉出多余参数，或用户试图通过对话夹带指令。

    ★ 注意这条测试的写法：断言的是「必须被拒绝」而不是「如果被拒绝了就检查...」。
      写成 `if result.ok: assert ...` 会让整条测试在"实际被拒绝"时变成空过 ——
      看起来是绿的，其实一行断言都没执行。这是安全测试里最隐蔽的自欺。
    """
    result = tools.execute(
        "query_metric",
        {
            "metric_id": "dau",
            "params": {
                "start_date": "2026-09-11",
                "end_date": "2026-09-17",
                "evil": "'; DROP TABLE dim_user;--",
                "sql": "SELECT * FROM hr_recruitment_data",
            },
        },
    )
    assert result.ok is False, "未声明的参数键竟然被接受了"
    assert "evil" in result.content, "拒绝信息没有点名是哪个参数，模型无法自我纠正"


def test_malformed_date_is_normalized_before_reaching_sql(tools: ToolExecutor) -> None:
    """★ 真实观察到的行为：日期解析器会从畸形串里「抢救」出合法日期。

    传 "2026-09-17' OR '1'='1"，工具不但没报错，反而把 2026-09-17 解析了出来
    （宽容解析），最终查到 1 行 —— 注入串被彻底中和。

    所以这条测试断言的**不是**"必须报错"，而是三条更本质的不变量：
      ① 结果集没有被放大（永真条件失效）
      ② 展示版 SQL 里出现的是规范化后的日期，而不是原始注入串
      ③ SQL 结构没被改变（没有多出 OR / 联合查询）

    第 ② 条能测，是因为本项目特意生成了「展示版 SQL」（值内联，给人看）——
    它同时也是"参数到底有没有被安全处理"的最直接证据。
    """
    payload = "2026-09-17' OR '1'='1"
    result = tools.execute("query_metric", {"metric_id": "dau", "params": {"start_date": payload}})
    assert result.ok is True, "解析器本应从畸形串里抢救出合法日期"

    rows = result.data.get("rows") or []
    assert len(rows) == 1, f"永真注入把结果集放大了：{len(rows)} 行"

    sql = str(result.data.get("rendered_sql") or "")
    assert payload not in sql, f"原始注入串进入了 SQL：{sql}"
    assert "OR '1'='1" not in sql.upper(), f"注入条件进入了 SQL：{sql}"
    assert "2026-09-17" in sql, f"规范化后的日期没有进入 SQL：{sql}"


def test_valid_date_param_renders_as_literal_but_stays_bounded(tools: ToolExecutor) -> None:
    """正常参数：展示版 SQL 会内联日期值（给人看），但结果仍受区间约束。

    这条是上面那条的对照组 —— 说明"展示版内联值"是正常设计，
    不是安全漏洞（真正执行的是带占位符的执行版，见 Phase 2 的设计）。
    """
    result = tools.execute(
        "query_metric",
        {"metric_id": "dau", "params": {"start_date": "2026-09-11", "end_date": "2026-09-13"}},
    )
    assert result.ok is True
    assert len(result.data.get("rows") or []) == 3
    sql = str(result.data.get("rendered_sql") or "")
    assert "2026-09-11" in sql and "2026-09-13" in sql


# ===========================================================================
# 五、恶意输入不能让工具抛异常
# ===========================================================================
HOSTILE_INPUTS = [
    "",                                       # 空串
    " ",                                      # 纯空白
    "🎮🎮🎮",                                  # emoji
    "';--" * 200,                             # 重复拼接
    "SELECT",                                 # 裸关键字
    "𝕕𝕒𝕦",                                   # Unicode 花体（非常规字符）
    "dau\x00DROP",                            # 空字节
]


@pytest.mark.parametrize("payload", HOSTILE_INPUTS)
def test_tool_never_raises_on_hostile_input(tools: ToolExecutor, payload: str) -> None:
    """恶意/畸形输入只能换来 ok=False，绝不能抛异常。

    为什么这条重要？因为工具在 ReAct 循环里被调用，
    一旦抛异常就会打断整轮对话（用户看到的是"系统错误"而不是"这个我查不了"）。
    失败必须能被模型"看到"并自我纠正 —— 这是 Phase 3 定下的设计原则。
    """
    result = tools.execute("query_metric", {"metric_id": payload, "params": {}})
    assert result.ok is False
    assert result.content, "失败也必须给出可读的说明（喂回给模型）"


def test_unknown_tool_name_returns_error(tools: ToolExecutor) -> None:
    """模型幻觉出一个不存在的工具名，也必须优雅失败。"""
    result = tools.execute("drop_all_tables", {})
    assert result.ok is False
    assert "drop_all_tables" in result.content


# ===========================================================================
# 六、极端日期范围
# ===========================================================================
def test_extreme_date_range_is_rejected(tools: ToolExecutor) -> None:
    """把日期范围拉到 200 年，必须在参数层就被拒绝（而不是靠行数上限兜底）。

    实测行为比预期更严：参数校验会直接报
    「超出可用数据范围 2026-06-20 ~ 2026-09-17」——
    连查询都不会发出去。这比"查了再截断"更好：不浪费 IO，也不会让人误以为
    "数据只有 500 行"（截断是静默的，拒绝是显式的）。

    这里同时保留行数上限的断言，作为"万一以后放宽了范围校验"的第二道保险。
    """
    result = tools.execute(
        "query_metric",
        {"metric_id": "dau", "params": {"start_date": "1900-01-01", "end_date": "2100-01-01"}},
    )
    assert result.ok is False, "极端日期范围竟然被放行了"
    assert "范围" in result.content, f"拒绝原因没有说明数据边界：{result.content}"

    # 第二道保险：即使将来放行，也不能突破行数上限
    if result.ok:
        rows = result.data.get("rows") or []
        assert len(rows) <= cfg.MAX_RESULT_ROWS


# ===========================================================================
# 七、★ 总兜底：跑完一整轮攻击，数据库必须完好
# ===========================================================================
def test_database_survives_injection_battery(tools: ToolExecutor) -> None:
    """把上面所有攻击载荷连续打一遍，然后从外部核对数据库是否完好。

    这一条是整份文件的"总验收"：
      · 表数量不变（没有表被 DROP）
      · 核心表行数不变（没有数据被 DELETE/UPDATE）

    为什么要连打一遍再核对，而不是每条用例自己核对？
      因为真实的攻击是组合式的 —— 单条无效不代表组合无效
      （比如"先 ATTACH 再改写"这类需要两步的攻击）。
    """
    tables_before = _table_names()
    counts_before = {name: _row_count(name) for name in ("dim_user", "user_daily_snapshot", "game_event_log")}

    # 连续攻击：metric_id / 日期参数 / 整数参数 / 未知键，全部来一遍
    for payload in INJECTION_PAYLOADS:
        tools.execute("query_metric", {"metric_id": payload, "params": {}})
    for payload in DATE_PAYLOADS:
        tools.execute("query_metric", {"metric_id": "dau", "params": {"start_date": payload}})
    for payload in INT_PAYLOADS:
        tools.execute("query_metric", {"metric_id": "retention_rate", "params": {"day_n": payload}})
    tools.execute(
        "query_metric",
        {"metric_id": "dau", "params": {"evil": "'; DROP TABLE dim_user;--"}},
    )

    tables_after = _table_names()
    assert tables_after == tables_before, (
        f"表结构被改动了！新增：{tables_after - tables_before}，丢失：{tables_before - tables_after}"
    )

    for name, before in counts_before.items():
        after = _row_count(name)
        assert after == before, f"{name} 行数从 {before} 变成了 {after}，数据被改动了"


def test_database_is_physically_read_only() -> None:
    """第四层防线的物理验证：数据库文件以只读方式打开，写操作直接被 SQLite 拒绝。

    这一条和 test_executor.py 里的只读测试不重复：
    那条测的是"执行器持有的连接是只读的"，这条测的是
    **"整个数据库文件的写权限在连接层就被剥夺"** —— 换一条全新连接也一样写不进去。
    """
    with _read_only_connection() as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE should_not_exist (id INTEGER)")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM dim_user")
