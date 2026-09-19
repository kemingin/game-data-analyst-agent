# -*- coding: utf-8 -*-
"""
指标模板资产与加载自检测试（Phase 8 · b2）
==========================================
覆盖四类断言：
  ① 正面 —— 真实资产能加载、能自检、能渲染出可执行的 SQL 骨架；
  ② 反面 —— 六种「配置写错」都能在**加载期**被拦下（而不是等到 SQL 报错）；
  ③ 一致性 —— KNOWN_ROLES 与 schema_roles 的角色表不漂移；
  ④ 契约 —— 模板里 {角色} 与 :参数 两类占位符各自的规则。

【为什么「加载期自检」值得单独写一整套反面用例？】
  因为模板资产的错误后果是**错在很远的地方**：占位符拼错 → 草稿生成照常成功
  → 人工复审看不出问题 → 运行时 SQL 报「no such column: {date}」。
  这类「错误信息离原因很远」的问题，唯一便宜的防线就是加载期 Fail Fast。
  反面用例的作用是把「自检确实会拦」这件事固定下来 —— 否则将来有人为了
  「让资产加载得更宽容」而删掉某条检查，没有任何测试会变红。
"""

from __future__ import annotations

import json

import pytest

from src.metrics import schema_roles, templates
from src.metrics.templates import (
    KNOWN_ROLES,
    MetricTemplate,
    _validate_template,
    load_templates,
)


# ===========================================================================
# 一、正面：真实资产
# ===========================================================================

def test_asset_loads_and_is_sorted_by_priority():
    """真实资产能加载，且按 priority 升序返回（数值小的先被草稿生成挑中）。"""
    items = load_templates()
    assert len(items) >= 5

    keys = [(t.priority, t.template_id) for t in items]
    assert keys == sorted(keys)


def test_same_metric_id_may_have_multiple_templates():
    """同一个 metric_id 允许多份模板（快照版 / 事件版），靠 priority 择优。

    【为什么不在加载期去重 metric_id？】
      去重是「草稿生成」的职责：一份数据集只有一种表形态，去重要看实际命中；
      而模板资产本身必须保留全部形态，否则事件型数据集永远拿不到 DAU 模板。
    """
    items = load_templates()
    dau = [t for t in items if t.metric_id == "dau"]
    assert len(dau) >= 2
    # 快照版（priority=10）必须排在事件版（priority=20）之前
    assert dau[0].template_id == "dau_from_snapshot"
    assert dau[0].priority < dau[1].priority


@pytest.mark.parametrize("template", load_templates(), ids=lambda t: t.template_id)
def test_every_template_passes_selfcheck(template: MetricTemplate):
    """逐模板验证两类占位符与声明完全对应（自检的「已通过」样本）。

    这些断言与 _validate_template 的检查项重复，是刻意的：
    自检代码本身也可能写错（比如把 {table} 也算成角色），
    这里用**独立写法**再断言一遍，等于给自检结果做二次证明。
    """
    assert template.role_placeholders == set(template.required_roles)
    assert template.param_placeholders == set(template.param_names)
    assert "{table}" in template.sql_template
    assert template.metric_name and template.business_definition


@pytest.mark.parametrize("template", load_templates(), ids=lambda t: t.template_id)
def test_render_sql_leaves_no_placeholder(template: MetricTemplate):
    """给齐所需角色后，渲染结果里不应再残留任何 {xxx} 占位符。

    【为什么这条重要？】占位符残留 = 生成的 SQL 语法错误。
      如果渲染是「尽力而为」而不是「要么全替换要么报错」，
      错误就会推迟到查询执行时才暴露，而那时人工已经确认过方案了。
    """
    fake_roles = {r: f"col_{r}" for r in template.required_roles}
    sql = template.render_sql("t_demo", fake_roles)
    assert "{" not in sql and "}" not in sql
    assert "t_demo" in sql
    # 运行时参数必须原样保留（由 SQLGenerator 后续填充）
    assert template.param_placeholders == set(
        templates._PARAM_PLACEHOLDER_RE.findall(sql)
    )


def test_render_sql_replaces_table_and_roles():
    """具名验证：留存模板的四个角色占位符都被替换成真实列名。"""
    retention = next(t for t in load_templates() if t.template_id == "retention_from_snapshot")
    sql = retention.render_sql(
        "user_daily_snapshot",
        {"register_date": "reg_date", "date": "dt", "user_id": "uid", "active_flag": "is_active"},
    )
    assert "user_daily_snapshot" in sql
    assert "reg_date" in sql and "uid" in sql and "is_active" in sql
    assert ":day_n" in sql  # 参数不能被误伤
    assert "{date}" not in sql and "{table}" not in sql


def test_missing_roles_reports_absent_roles():
    """missing_roles 精确报出缺哪个角色（草稿生成据此跳过不匹配的模板）。"""
    t = next(x for x in load_templates() if x.template_id == "retention_from_snapshot")
    assert t.missing_roles({"date": "dt", "user_id": "uid"}) == ["register_date", "active_flag"]
    assert t.missing_roles({r: "c" for r in t.required_roles}) == []


def test_templates_without_roles_are_unreachable_by_dau_event_template():
    """反面：只有日期列的表，事件版 DAU 模板应当「不匹配」而不是硬生成。"""
    t = next(x for x in load_templates() if x.template_id == "dau_from_events")
    assert t.missing_roles({"date": "dt"}) == ["user_id"]


# ===========================================================================
# 二、一致性：角色清单不漂移
# ===========================================================================

def test_known_roles_matches_schema_roles():
    """templates.KNOWN_ROLES 必须与 schema_roles._ROLE_SPECS 完全一致。

    【为什么要断言？】两份清单是刻意分开写的（契约 vs 实现），
      分开就一定会漂移：某天在 schema_roles 里加了新角色（比如 device_type），
      模板里想用却因为 KNOWN_ROLES 没同步而加载失败 —— 报错指向模板，
      真正原因却在另一个文件。这条测试把漂移变成一次明确的失败。
    """
    declared = {spec[0] for spec in schema_roles._ROLE_SPECS}
    assert set(KNOWN_ROLES) == declared


# ===========================================================================
# 三、反面：加载期自检必须拦下的六种错
# ===========================================================================

def _base_template(**overrides) -> dict:
    """构造一份合法的最小模板，再按需覆盖某个字段制造错误。"""
    raw = {
        "template_id": "t_demo",
        "metric_id": "dau",
        "metric_name": "演示指标",
        "business_definition": "演示用",
        "required_roles": ["date"],
        "sql_template": "SELECT {date} FROM {table}",
        "params": [],
    }
    raw.update(overrides)
    return raw


def test_selfcheck_accepts_valid_template():
    """正面基线：合法模板不抛错（否则下面每条反面用例都可能是「误拦」）。"""
    _validate_template(_base_template(), set())


@pytest.mark.parametrize(
    "missing_field",
    ["template_id", "metric_id", "metric_name", "business_definition", "sql_template", "required_roles"],
)
def test_selfcheck_rejects_missing_required_field(missing_field):
    """必填字段缺失 → 报错并点名缺了哪个字段。"""
    raw = _base_template()
    raw[missing_field] = "" if missing_field != "required_roles" else []
    with pytest.raises(ValueError, match=missing_field):
        _validate_template(raw, set())


def test_selfcheck_rejects_duplicate_template_id():
    """template_id 重复 → 报错（否则草稿里会出现两份来源不明、内容不同的指标）。"""
    with pytest.raises(ValueError, match="模板 ID 重复"):
        _validate_template(_base_template(), {"t_demo"})


def test_selfcheck_rejects_unknown_role():
    """声明了角色表以外的角色 → 报错，并列出可用角色（便于直接改对）。"""
    raw = _base_template(required_roles=["date", "device_type"],
                         sql_template="SELECT {date}, {device_type} FROM {table}")
    with pytest.raises(ValueError, match="未知角色"):
        _validate_template(raw, set())


def test_selfcheck_rejects_undeclared_role_in_sql():
    """SQL 里用了角色但没在 required_roles 声明 → 报错。

    【为什么这比「多声明」更危险？】草稿生成只按 required_roles 取值，
      未声明的角色会渲染成原样的 {xxx}，SQL 语法错误。
    """
    raw = _base_template(sql_template="SELECT {date}, {user_id} FROM {table}")
    with pytest.raises(ValueError, match="未在 required_roles 中声明"):
        _validate_template(raw, set())


def test_selfcheck_rejects_declared_but_unused_role():
    """声明了却在 SQL 里没用 → 报错（会让「明明不需要这列」的模板匹配不上）。"""
    raw = _base_template(required_roles=["date", "user_id"])
    with pytest.raises(ValueError, match="未使用"):
        _validate_template(raw, set())


def test_selfcheck_rejects_param_not_in_sql():
    """声明了参数但 SQL 里没有 :占位符 → 报错（参数会被静默丢弃）。"""
    raw = _base_template(
        sql_template="SELECT {date} FROM {table}",
        params=[{"name": "start_date", "type": "date"}],
    )
    with pytest.raises(ValueError, match="参数与占位符不一致"):
        _validate_template(raw, set())


def test_selfcheck_rejects_sql_param_not_declared():
    """SQL 里用了 :参数 但没声明 → 报错（运行时会「无值可填」）。"""
    raw = _base_template(sql_template="SELECT {date} FROM {table} WHERE {date} >= :start_date")
    with pytest.raises(ValueError, match="参数与占位符不一致"):
        _validate_template(raw, set())


def test_selfcheck_rejects_missing_table_placeholder():
    """缺少 {table} → 报错（否则生成的 SQL 不知道该查哪张表）。"""
    raw = _base_template(sql_template="SELECT {date} FROM some_table")
    with pytest.raises(ValueError, match=r"缺少 \{table\} 占位符"):
        _validate_template(raw, set())


def test_table_placeholder_is_not_treated_as_role():
    """{table} 不能被当成角色 —— 这条防的是自检自身的实现错误。

    【背景】{table} 与 {date} 的写法完全一样（都是 {小写名}）。
      如果取角色占位符时不排除 table，那么每个模板都会因为
      「SQL 使用了未声明的角色：table」而被判非法，整套资产直接加载不了。
      这里用显式断言把「table 不是角色」钉死。
    """
    t = MetricTemplate(
        template_id="t", metric_id="m", metric_name="n", business_definition="d",
        sql_template="SELECT {date} FROM {table}", required_roles=("date",),
    )
    assert t.role_placeholders == {"date"}
    assert "table" not in t.role_placeholders


# ===========================================================================
# 四、反面：加载入口的失败路径
# ===========================================================================

def test_load_templates_missing_file(tmp_path):
    """资产文件不存在 → FileNotFoundError（路径写错时一眼可见）。"""
    with pytest.raises(FileNotFoundError):
        load_templates(str(tmp_path / "nope.json"))


def test_load_templates_invalid_json(tmp_path):
    """非法 JSON → ValueError，且错误信息带上文件路径。"""
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="不是合法 JSON"):
        load_templates(str(bad))


def test_load_templates_empty_list(tmp_path):
    """templates 为空 → ValueError（空资产会让草稿永远为空，属于配置事故）。"""
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"templates": []}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="资产为空"):
        load_templates(str(empty))


def test_load_templates_propagates_validation_error(tmp_path):
    """非法模板 → ValueError（证明加载入口确实调用了自检）。"""
    raw = {"templates": [_base_template(sql_template="SELECT {date} FROM t_no_table")]}
    bad = tmp_path / "bad_template.json"
    bad.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"缺少 \{table\} 占位符"):
        load_templates(str(bad))


def test_load_templates_reads_optional_fields(tmp_path):
    """可选字段缺省时的默认值（priority 默认 100、good_direction 默认 up）。"""
    raw = {"templates": [_base_template()]}
    f = tmp_path / "min.json"
    f.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    items = load_templates(str(f))
    assert len(items) == 1
    assert items[0].priority == 100
    assert items[0].good_direction == "up"
    assert items[0].aliases == ()
    assert items[0].param_names == ()
