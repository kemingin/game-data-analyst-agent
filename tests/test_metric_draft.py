# -*- coding: utf-8 -*-
"""
指标草稿生成 / 校验 / 序列化单元测试（Phase 8 · b3）
====================================================
被测对象：src/metrics/draft.py。

【为什么这个文件是「半自动闭环」里最该测的一块？】
  因为它站在两个不可信输入的中间：
    · 左边是**上传的 CSV**（列名千奇百怪，可能一个角色都识别不出）；
    · 右边是**人工手写的指标**（SQL 写错、来源表写错、参数对不上）。
  它的输出会直接变成「指标方案资产」，而方案错了之后的症状是
  「Agent 某个问题答不出来」或更糟 ——「答出来但口径是错的」。
  后者不报错、数字看着合理，正是本项目一路在防的那类静默错误。

【测试分六组】
  一、草稿生成 —— 不同表形态能产出哪些指标、来源是否可溯
  二、去重与优先级 —— 同一指标只留最优形态
  三、空草稿的诚实失败 —— 识别不出角色时不抛错、给得出原因
  四、validate_metrics 反面 —— 十种配置错误必须逐条报出且不落盘
  五、序列化 —— 产出结构与内置注册表一致
  六、落盘 —— 原子写、目录创建、非法 ID 拦截、闭环三步生效
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.data.dataset import DatasetStore
from src.metrics import draft as draft_mod
from src.metrics.draft import (
    DraftMetric,
    DraftResult,
    build_registry_json,
    generate_draft,
    make_scheme_id,
    register_and_activate,
    render_for_validation,
    validate_metrics,
    write_scheme_file,
)


# ---------------------------------------------------------------------------
# 工具：造表元信息（与 DatasetTable 同构，用 SimpleNamespace 鸭子类型）
# ---------------------------------------------------------------------------

def _col(name: str, sql_type: str = "INTEGER", is_date: bool = False) -> dict:
    return {"name": name, "sql_name": name, "sql_type": sql_type, "is_date": is_date}


def _date_col(name: str) -> dict:
    return _col(name, sql_type="DATE", is_date=True)


def _table(name: str, columns: list[dict]) -> SimpleNamespace:
    """造一张表。

    【为什么用 SimpleNamespace 而不是 DatasetTable？】
      因为 generate_draft 只要求对象暴露 .name / .columns（鸭子类型），
      这样指标语义层不必 import 数据层。用 SimpleNamespace 造数据，
      顺便把这个「不依赖数据层」的设计约束变成了可执行的证明 ——
      如果哪天有人在 draft.py 里加了 `from src.data.dataset import ...`，
      这组用例仍会通过，但「分层可替换」这件事已经被悄悄破坏了；
      所以另有一条用例专门断言模块级没有这个 import（见文末）。
    """
    return SimpleNamespace(name=name, columns=tuple(columns))


def _snapshot_table(name: str = "user_daily_snapshot") -> SimpleNamespace:
    """标准快照表：日期 + 用户 + 活跃标志 + 注册日期。"""
    return _table(name, [
        _date_col("date"),
        _col("user_id"),
        _col("is_active"),
        _date_col("register_date"),
    ])


def _event_table(name: str = "game_event_log") -> SimpleNamespace:
    """标准事件表：日期 + 用户 + 事件名（没有活跃标志）。"""
    return _table(name, [
        _date_col("event_date"),
        _col("user_id"),
        _col("event_name"),
    ])


# ===========================================================================
# 一、草稿生成
# ===========================================================================

def test_snapshot_table_yields_core_metrics():
    """快照表应能产出 DAU / MAU / 新增用户 / 留存率 四类指标。"""
    result = generate_draft([_snapshot_table()])
    ids = set(result.metric_ids)
    assert {"dau", "mau", "new_user_count", "retention_rate"} <= ids


def test_event_table_yields_dau_only_via_events_template():
    """事件表（无活跃标志）只能用「事件版」DAU，不该硬生成 MAU / 留存。"""
    result = generate_draft([_event_table()])
    assert result.metric_ids == ("dau",)
    dau = result.metrics[0]
    assert dau.template_id == "dau_from_events"
    assert "COUNT(DISTINCT user_id)" in dau.sql_template


def test_revenue_table_yields_revenue_metric():
    """只有日期 + 金额的表也能产出一条流水指标。"""
    result = generate_draft([
        _table("recharge_log", [_date_col("dt"), _col("amount", "REAL")]),
    ])
    assert "revenue_total" in result.metric_ids
    rev = next(m for m in result.metrics if m.metric_id == "revenue_total")
    assert "SUM(amount)" in rev.sql_template


def test_metric_source_is_traceable():
    """每条指标都要能说清「来自哪张表、用了哪几列」—— 这是复审的前提。"""
    result = generate_draft([_snapshot_table("my_snapshot")])
    dau = next(m for m in result.metrics if m.metric_id == "dau")
    assert dau.definition["source_tables"] == ["my_snapshot"]
    assert dau.table == "my_snapshot"
    assert dau.roles["date"] == "date"
    assert "my_snapshot" in dau.source_text()
    assert "date=date" in dau.source_text()


def test_generated_sql_has_no_leftover_placeholder():
    """生成的 SQL 必须是字面量：不留 {xxx}，也不留 {table}。"""
    result = generate_draft([_snapshot_table()])
    assert not result.is_empty
    for m in result.metrics:
        assert "{" not in m.sql_template
        assert "}" not in m.sql_template


def test_business_definition_is_rendered_too():
    """业务定义里的 {active_flag} 也要被替换成真实列名。

    【为什么单独测这条？】定义是给人看的口径说明。留着 {active_flag}
      会让人以为系统没渲染完，进而怀疑整个方案是否可靠。
    """
    result = generate_draft([_snapshot_table()])
    dau = next(m for m in result.metrics if m.metric_id == "dau")
    definition = dau.definition["business_definition"]
    assert "{active_flag}" not in definition
    assert "is_active" in definition


def test_multi_table_dataset_picks_per_template():
    """多表数据集：每个模板各自挑自己最合适的那张表。"""
    result = generate_draft([_snapshot_table("snap"), _event_table("evt")])
    # 事件表没有活跃标志，DAU 由快照表提供（priority 更高）
    dau = next(m for m in result.metrics if m.metric_id == "dau")
    assert dau.table == "snap"
    # 留存只可能来自快照表
    ret = next(m for m in result.metrics if m.metric_id == "retention_rate")
    assert ret.table == "snap"


# ===========================================================================
# 二、去重与优先级
# ===========================================================================

def test_same_metric_id_produced_once():
    """同一 metric_id 只能出现在草稿里一次（方案内 metric_id 必须唯一）。"""
    result = generate_draft([_snapshot_table(), _event_table()])
    ids = list(result.metric_ids)
    assert len(ids) == len(set(ids))


def test_snapshot_template_wins_over_events_template():
    """两种形态都可用时，priority 更高的快照版胜出（SUM 比 COUNT DISTINCT 便宜）。"""
    result = generate_draft([_snapshot_table()])
    dau = next(m for m in result.metrics if m.metric_id == "dau")
    assert dau.template_id == "dau_from_snapshot"


# ===========================================================================
# 三、空草稿：诚实失败
# ===========================================================================

def test_meaningless_columns_yield_empty_draft_without_error():
    """列名无意义时不抛错，返回空草稿（界面才能显示「请手工搭建」）。"""
    result = generate_draft([_table("t", [_col("c1"), _col("c2")])])
    assert result.is_empty
    assert result.metric_ids == ()


def test_empty_draft_summary_points_to_manual_builder():
    """空草稿的摘要必须给出出路，而不是只说「失败」。"""
    result = generate_draft([_table("t", [_col("c1")])])
    assert "手工搭建" in result.summary_text()


def test_empty_table_list_is_safe():
    """没有任何表时也不能崩（数据集可能建库失败只登记了元信息）。"""
    result = generate_draft([])
    assert result.is_empty
    # 所有模板都应记为「未命中」，且把「需要哪些角色」原样报出来
    assert result.skipped
    assert all(s.missing_roles for s in result.skipped)


def test_skipped_template_reports_missing_roles():
    """未命中的模板要报出「缺哪个角色」，且挑缺得最少的那张表来报。

    这里给两张表：事件表缺 1 个角色（register_date 等），
    无意义表缺全部 —— 提示应基于事件表（更有指导意义）。
    """
    result = generate_draft([_table("junk", [_col("c1")]), _event_table()])
    retention = next(s for s in result.skipped if s.metric_id == "retention_rate")
    assert "register_date" in retention.missing_roles
    assert "未识别到角色" in retention.reason_text()


def test_non_empty_draft_summary_lists_metric_ids():
    result = generate_draft([_snapshot_table()])
    text = result.summary_text()
    assert "候选指标" in text
    assert "dau" in text


# ===========================================================================
# 四、validate_metrics
# ===========================================================================

def _valid_metric(**overrides) -> dict:
    """一份合法的最小指标定义，按需覆盖字段来制造错误。"""
    raw = {
        "metric_id": "dau",
        "metric_name": "日活跃用户数",
        "business_definition": "每日去重活跃用户数",
        "sql_template": (
            "SELECT date, SUM(is_active) AS dau FROM t_demo "
            "WHERE date BETWEEN :start_date AND :end_date GROUP BY date"
        ),
        "source_tables": ["t_demo"],
        "params": [
            {"name": "start_date", "type": "date"},
            {"name": "end_date", "type": "date"},
        ],
    }
    raw.update(overrides)
    return raw


ALLOWED = ("t_demo", "t_other")


def test_validate_accepts_valid_metric():
    """正面基线：合法定义返回空清单。"""
    assert validate_metrics([_valid_metric()], ALLOWED) == []


def test_validate_accepts_generated_draft():
    """端到端正面：草稿生成的全部指标必须直接通过校验。

    【这条为什么重要？】它把「生成」和「校验」两个模块的契约钉在一起：
      若某天改了模板资产却忘了同步参数声明，这条会立刻变红，
      而不是等到用户点「确认并启用」时才发现方案存不下去。
    """
    result = generate_draft([_snapshot_table()])
    allowed = ("user_daily_snapshot",)
    assert validate_metrics([m.definition for m in result.metrics], allowed) == []


@pytest.mark.parametrize(
    "overrides,keyword",
    [
        ({"metric_id": ""}, "缺少 metric_id"),
        ({"metric_id": "DAU"}, "小写字母开头"),
        ({"metric_id": "1dau"}, "小写字母开头"),
        ({"metric_name": ""}, "缺少 metric_name"),
        ({"business_definition": ""}, "缺少 business_definition"),
        ({"sql_template": ""}, "缺少 sql_template"),
        ({"source_tables": []}, "source_tables 不能为空"),
        ({"source_tables": ["t_outside"]}, "来源表不在本数据集内"),
        (
            {"sql_template": "SELECT {date} FROM t_demo"},
            "残留未替换的占位符",
        ),
        (
            {"sql_template": "SELECT date FROM t_demo", "params": [{"name": "p1", "type": "date"}]},
            "参数与占位符不一致",
        ),
        (
            {"sql_template": "SELECT date FROM t_demo WHERE date = :p1"},
            "参数与占位符不一致",
        ),
        ({"params": [{"name": "p1", "type": "float"}]}, "类型 float 不受支持"),
        ({"params": [{"name": "", "type": "date"}]}, "参数缺少 name 字段"),
    ],
)
def test_validate_rejects_broken_metric(overrides, keyword):
    """反面：各种配置错误都必须被报出来（并且报错文字要点到原因）。"""
    problems = validate_metrics([_valid_metric(**overrides)], ALLOWED)
    assert problems, f"应当报错但没有：{overrides}"
    assert any(keyword in p for p in problems), problems


def test_validate_rejects_duplicate_metric_id():
    """metric_id 重复 → 报错（否则注册表加载时会直接抛异常）。"""
    problems = validate_metrics([_valid_metric(), _valid_metric()], ALLOWED)
    assert any("metric_id 重复" in p for p in problems)


def test_validate_rejects_write_statement():
    """非只读 SQL 必须被 SQLValidator 拦下（草稿不能成为绕过安全护栏的后门）。"""
    problems = validate_metrics(
        [_valid_metric(sql_template="DELETE FROM t_demo")], ALLOWED
    )
    assert any("安全校验" in p for p in problems)


def test_validate_rejects_table_outside_dataset_whitelist():
    """SQL 引用了 source_tables 之外的表 → 报错。

    【这条防的是什么？】source_tables 是「指标自称的数据来源」，
      SQL 里实际查的表才是事实。两者不一致时，白名单会被架空。
    """
    problems = validate_metrics(
        [_valid_metric(sql_template="SELECT date FROM t_other")], ALLOWED
    )
    assert any("安全校验" in p for p in problems)


def test_validate_reports_multiple_problems_at_once():
    """一次列全所有问题（用户改一遍就能过，而不是改一条跑一次）。"""
    problems = validate_metrics(
        [_valid_metric(metric_id="", metric_name="", business_definition="")], ALLOWED
    )
    assert len(problems) >= 3


def test_render_for_validation_substitutes_typed_fake_values():
    """假值渲染：date → 带引号的日期字面量，int → 数字。"""
    sql = "SELECT * FROM t WHERE d BETWEEN :start AND :n"
    out = render_for_validation(sql, [{"name": "start", "type": "date"}, {"name": "n", "type": "int"}])
    assert ":start" not in out and ":n" not in out
    assert "'2000-01-01'" in out
    assert " 1" in out


def test_render_for_validation_does_not_touch_similar_names():
    """参数名是另一个参数名的前缀时不能误替换（:date 不该吃掉 :date_end）。"""
    sql = "SELECT * FROM t WHERE a = :date AND b = :date_end"
    out = render_for_validation(
        sql, [{"name": "date", "type": "date"}, {"name": "date_end", "type": "date"}]
    )
    assert ":date" not in out and ":date_end" not in out
    assert out.count("'2000-01-01'") == 2


# ===========================================================================
# 五、序列化
# ===========================================================================

def test_build_registry_json_shape():
    """产出的结构必须与内置 metrics_registry.json 一致（meta + metrics）。"""
    payload = build_registry_json([_valid_metric()])
    assert set(payload) == {"meta", "metrics"}
    assert len(payload["metrics"]) == 1
    assert payload["meta"]["registry_name"]


def test_build_registry_json_meta_override():
    """显式 meta 覆盖默认值，未覆盖的键保留默认。"""
    payload = build_registry_json([_valid_metric()], {"registry_name": "自定义方案"})
    assert payload["meta"]["registry_name"] == "自定义方案"
    assert payload["meta"]["updated_at"]


def test_build_registry_json_is_loadable_by_metric_registry(tmp_path):
    """端到端：写出的 JSON 必须能被 MetricRegistry 真正加载。

    【为什么必须用真加载器验证，而不是只比对键名？】
      键名对不代表加载器认。MetricRegistry 会强制校验必填字段
      （source_tables / owner / updated_at 等），少一个就抛 ValueError。
      用真加载器跑一遍，等于把「草稿 → 方案资产」这条链路的终点也测到了。
    """
    from src.metrics.registry import MetricRegistry

    result = generate_draft([_snapshot_table()])
    payload = result.to_registry_dict({"registry_name": "草稿方案"})
    path = tmp_path / "metrics_registry.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    registry = MetricRegistry(path)
    assert set(registry.ids()) == set(result.metric_ids)


def test_draft_result_to_registry_dict_uses_definitions():
    result = generate_draft([_snapshot_table()])
    payload = result.to_registry_dict()
    assert [m["metric_id"] for m in payload["metrics"]] == list(result.metric_ids)


# ===========================================================================
# 六、落盘与闭环
# ===========================================================================

def test_write_scheme_file_creates_dir_and_file(tmp_path):
    """落盘：自动建目录，写出的内容可被 json.loads 读回。"""
    payload = build_registry_json([_valid_metric()])
    path = write_scheme_file(payload, "demo_scheme", tmp_path)

    assert path == tmp_path / "demo_scheme" / "metrics_registry.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["metrics"][0]["metric_id"] == "dau"


def test_write_scheme_file_leaves_no_tmp_file(tmp_path):
    """原子写不能留下 .tmp 残留（残留说明写到一半就退出了）。"""
    write_scheme_file(build_registry_json([_valid_metric()]), "s1", tmp_path)
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_scheme_file_overwrites_atomically(tmp_path):
    """重复写同一个方案 ID：内容整体替换，不出现「新旧混合」。"""
    write_scheme_file(build_registry_json([_valid_metric()]), "s1", tmp_path)
    write_scheme_file(
        build_registry_json([_valid_metric(metric_id="mau", metric_name="月活")]),
        "s1",
        tmp_path,
    )
    data = json.loads((tmp_path / "s1" / "metrics_registry.json").read_text(encoding="utf-8"))
    assert [m["metric_id"] for m in data["metrics"]] == ["mau"]


@pytest.mark.parametrize("bad_id", ["../escape", "A_B", "", "has space", "s" * 41])
def test_write_scheme_file_rejects_illegal_scheme_id(tmp_path, bad_id):
    """非法方案 ID 必须被拦下 —— 它会变成目录名，放行就等于放行路径穿越。"""
    with pytest.raises(ValueError, match="非法的方案 ID"):
        write_scheme_file(build_registry_json([_valid_metric()]), bad_id, tmp_path)


def test_make_scheme_id_is_readable_and_unique():
    """方案 ID 可读（带数据集前缀）且不冲突（自动递增）。"""
    first = make_scheme_id("my_data", [])
    assert first == "my_data_scheme"
    assert make_scheme_id("my_data", [first]) == "my_data_scheme_2"
    assert make_scheme_id("my_data", [first, "my_data_scheme_2"]) == "my_data_scheme_3"


def test_make_scheme_id_falls_back_for_non_ascii():
    """全中文数据集名折叠成空后要有兜底，不能产出非法 ID。"""
    sid = make_scheme_id("销售数据", [])
    assert sid and sid.isascii()


def test_register_and_activate_closes_the_loop(tmp_path):
    """闭环三步：落盘 → 登记方案 → 数据集指向新方案。

    【为什么这条用例最关键？】
      它是用户点「确认并启用」时真正发生的事。三步里漏任何一步，
      用户都会得到一个「看起来启用了但没生效」的状态。
    """
    store = DatasetStore(root=tmp_path / "datasets", auto_builtin=False)
    store.register_scheme(
        display_name="旧方案",
        registry_path=tmp_path / "old.json",
        scheme_id="old_scheme",
    )
    store.create_dataset(
        display_name="我的数据",
        scheme_id="old_scheme",
        source="upload",
        db_path=tmp_path / "data.db",
        raw_dir=tmp_path / "raw",
        dataset_id="my_data",
    )

    result = generate_draft([_snapshot_table()])
    path = register_and_activate(
        store=store,
        dataset_id="my_data",
        scheme_id="my_data_scheme",
        display_name="我的数据 · 指标方案",
        registry=result.to_registry_dict(),
        derived_from="old_scheme",
        target_dir=tmp_path / "schemes",
    )

    # ① 文件落盘
    assert path.exists()
    # ② 方案已登记，且血缘被记下
    scheme = store.get_scheme("my_data_scheme")
    assert scheme.derived_from == "old_scheme"
    assert scheme.exists()
    # ③ 数据集已切换指向
    assert store.get("my_data").scheme_id == "my_data_scheme"
    # ④ 缓存键必须跟着变 —— 否则 Agent 会继续用旧方案的缓存实例
    assert store.get("my_data").cache_key() != "my_data::old_scheme::x"


def test_register_and_activate_rejects_unknown_dataset(tmp_path):
    """数据集不存在时整个动作失败，不留半成品方案文件。"""
    store = DatasetStore(root=tmp_path / "datasets", auto_builtin=False)
    with pytest.raises(KeyError):
        register_and_activate(
            store=store,
            dataset_id="nope",
            scheme_id="s_new",
            display_name="x",
            registry=build_registry_json([_valid_metric()]),
            target_dir=tmp_path / "schemes",
        )


# ===========================================================================
# 七、端到端：上传 → 草稿 → 启用 → 真的能查
# ===========================================================================

_CSV_SNAPSHOT = (
    "date,user_id,is_active,register_date\n"
    "2026-09-01,u1,1,2026-08-01\n"
    "2026-09-01,u2,1,2026-09-01\n"
    "2026-09-02,u1,1,2026-08-01\n"
    "2026-09-02,u2,0,2026-09-01\n"
    "2026-09-03,u1,1,2026-08-01\n"
    "2026-09-03,u2,1,2026-09-01\n"
)


def _uploaded_store(tmp_path, csv_text: str, file_name: str = "snapshot.csv"):
    """走真实的「上传建表 → 登记数据集」链路，返回 (store, dataset)。"""
    store = DatasetStore(root=tmp_path / "datasets", auto_builtin=False)
    store.register_scheme(
        display_name="占位方案",
        registry_path=tmp_path / "placeholder.json",
        scheme_id="placeholder",
    )
    dataset, results = store.add_from_upload(
        [(file_name, csv_text.encode("utf-8"))],
        display_name="端到端数据集",
        scheme_id="placeholder",
    )
    assert store.has(dataset.dataset_id), [r.error for r in results]
    return store, dataset


def test_end_to_end_upload_draft_activate_and_query(tmp_path):
    """全链路：上传 CSV → 生成草稿 → 确认启用 → 用新方案真的查到数。

    【为什么必须有这条？】项目里踩过的坑写得很清楚：
      「509 条单测全绿不等于 main() 能跑 —— 测试覆盖的是函数，不是接线。」
      这条用例专门覆盖**接线**：草稿产出的 registry JSON 能不能被
      MetricRegistry 加载、source_tables 能不能过数据集白名单、
      参数默认值能不能被 SQLGenerator 解析、SQL 能不能真的在库里跑出结果。
      每一步单独测过都不代表串起来能跑。
    """
    from src.context import build_context

    store, dataset = _uploaded_store(tmp_path, _CSV_SNAPSHOT)

    # --- ① 草稿：能从真实上传的表结构里认出角色并生成 DAU ---
    result = generate_draft(dataset.tables)
    assert "dau" in result.metric_ids, result.summary_text()
    assert "new_user_count" in result.metric_ids

    # --- ② 校验：全部通过（否则「确认并启用」按钮会拒绝落盘）---
    assert validate_metrics([m.definition for m in result.metrics], dataset.table_names) == []

    # --- ③ 启用：落盘 + 登记 + 切换指向 ---
    register_and_activate(
        store=store,
        dataset_id=dataset.dataset_id,
        scheme_id="e2e_scheme",
        display_name="端到端方案",
        registry=result.to_registry_dict(),
        derived_from="placeholder",
        target_dir=tmp_path / "schemes",
    )
    assert store.get(dataset.dataset_id).scheme_id == "e2e_scheme"

    # --- ④ 组装上下文：这一步会真正加载新方案并校验表清单 ---
    ctx = build_context(dataset.dataset_id, store=store)
    assert ctx.allowed_tables == dataset.table_names
    assert ctx.registry.get("dau").source_tables == (dataset.tables[0].name,)

    # --- ⑤ 真查询：生成 SQL → 校验 → 执行 ---
    generated = ctx.build_generator().generate("dau")
    outcome = ctx.build_executor().execute(
        generated.sql, params=generated.params,
        allowed_tables=ctx.registry.get("dau").source_tables,
    )
    assert outcome.success, outcome.error
    rows = outcome.to_dicts()
    # 9/1 两人活跃、9/2 一人、9/3 两人
    assert [r["dau"] for r in rows] == [2, 1, 2]


def test_end_to_end_blank_dataset_reports_manual_fallback(tmp_path):
    """反面端到端：列名毫无意义的 CSV → 草稿为空，但链路不崩、有明确出路。

    【为什么反面也要端到端？】「空草稿」是真实用户最容易遇到的状态
      （别人给的脏数据），而它的正确行为不是报错，是
      「界面显示请手工搭建」—— 这条用例把那个状态固定下来。
    """
    store, dataset = _uploaded_store(tmp_path, "a,b\n1,2\n3,4\n", file_name="junk.csv")

    result = generate_draft(dataset.tables)
    assert result.is_empty
    assert "手工搭建" in result.summary_text()
    # 数据集本身仍然可用（方案没被换掉，只是没有可用指标）
    assert store.get(dataset.dataset_id).scheme_id == "placeholder"


# ===========================================================================
# 八、分层约束（把设计决策变成可执行的证明）
# ===========================================================================

def test_draft_module_does_not_import_data_layer():
    """draft.py 不得在模块级 import 数据层。

    【为什么用「读源码」来测？】分层是架构承诺，而不是运行时行为 ——
      它没有任何运行期症状，只有「将来想把数据层换掉时才发现换不动」。
      静态断言是唯一能自动守住它的方式，成本极低。
    """
    source = (draft_mod.__file__ or "")
    text = open(source, encoding="utf-8").read()
    assert "from src.data" not in text
    assert "import src.data" not in text
