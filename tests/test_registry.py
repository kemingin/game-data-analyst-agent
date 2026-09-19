# -*- coding: utf-8 -*-
"""
指标注册表单元测试
================================================================================
这个测试文件的价值不在「跑通代码」，而在于它是**数据治理的自动化守门人**。

  注册表是人工维护的 JSON，人手一定会写错：漏字段、参数名写错、
  别名重复、来源表拼错……这些错误在运行期表现为「查不出数」或「口径不一致」，
  非常难查。所以用测试把不变量（invariant）固化下来，
  每次改 JSON 跑一遍 pytest，就能立刻发现。

  这就是「配置即代码」的实践：配置文件也要有测试。
"""

from __future__ import annotations

import re

import pytest

from src.exceptions import MetricNotFoundError
from src.metrics import get_registry
from src.sqlgen.generator import PLACEHOLDER_RE
from src.sqlgen.validator import ALLOWED_TABLES

# Phase 2 要求必须覆盖的核心指标
REQUIRED_METRIC_IDS = {
    "dau",
    "mau",
    "retention_rate",
    "payment_rate",
    "arpu",
    "ltv",
    "tutorial_funnel",
    "channel_retention",
}


@pytest.fixture(scope="module")
def registry():
    """整个测试模块共用一个注册表实例（只读，可以安全共享）。"""
    return get_registry()


# ---------------------------------------------------------------------------
# 一、结构完整性
# ---------------------------------------------------------------------------

def test_registry_loads(registry):
    """注册表能正常加载，且至少有 12 个指标。"""
    assert len(registry) >= 12


def test_meta_contains_data_window(registry):
    """meta 里必须写明数据窗口 —— 它是 Agent 判断「知识缺失」的依据。"""
    window = registry.meta.get("data_window", {})
    assert window.get("start") and window.get("end")
    assert window["start"] < window["end"]


def test_required_metrics_present(registry):
    """Phase 2 要求的核心指标一个都不能少。"""
    missing = REQUIRED_METRIC_IDS - set(registry.ids())
    assert not missing, f"缺少必需指标：{missing}"


def test_no_duplicate_metric_ids(registry):
    """注册表内部按 dict 去重，这里再确认一次 ID 与对象一一对应。"""
    ids = registry.ids()
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_required_fields_not_empty(metric):
    """每个指标都填齐了「指标ID/中文名/业务定义/SQL模板/来源表/负责人/更新时间」。

    这是 Phase 2 对注册表的硬性要求（也常被问「你怎么保证口径可追溯」）。
    """
    assert metric.metric_id
    assert metric.metric_name
    assert len(metric.business_definition) >= 20, "业务定义太短，说不清口径"
    assert metric.sql_template.strip()
    assert metric.source_tables
    assert metric.owner
    assert metric.updated_at


@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_source_tables_within_whitelist(metric):
    """指标声明的来源表必须在全局白名单内。

    否则指标层的「来源表」和 SQL 安全层的「白名单」就会打架：
    指标说要用某张表，校验器却拒绝，运行期直接 500。
    """
    illegal = set(metric.source_tables) - ALLOWED_TABLES
    assert not illegal, f"{metric.metric_id} 声明了白名单外的表：{illegal}"


# ---------------------------------------------------------------------------
# 二、SQL 模板与参数的一致性（最容易写错的地方）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_template_placeholders_match_declared_params(metric):
    """模板里的 :占位符 必须与声明的 params 完全一致（双向）。

    多一个：执行时报「缺少绑定参数」；
    少一个：说明参数声明多余，SQL 里压根没用上（配置写错了）。
    两者都要在测试阶段被拦住。
    """
    in_template = set(PLACEHOLDER_RE.findall(metric.sql_template))
    declared = set(metric.param_names)

    assert in_template - declared == set(), (
        f"{metric.metric_id} 模板使用了未声明的占位符：{in_template - declared}"
    )
    assert declared - in_template == set(), (
        f"{metric.metric_id} 声明了未使用的参数：{declared - in_template}"
    )


@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_param_type_is_supported(metric):
    """参数类型只能是生成器支持的三类，避免 JSON 里写出没人实现的类型。"""
    for param in metric.params:
        assert param.type in {"date", "int", "string"}, (
            f"{metric.metric_id}.{param.name} 的类型 {param.type} 不被支持"
        )


@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_param_default_is_within_enum(metric):
    """默认值必须落在枚举范围内，否则「不传参」这个默认路径根本跑不通。"""
    for param in metric.params:
        if param.enum and param.has_default:
            assert param.default in param.enum, (
                f"{metric.metric_id}.{param.name} 默认值 {param.default} 不在枚举 {param.enum} 内"
            )


# ---------------------------------------------------------------------------
# 三、日期默认值的引用合法性
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", get_registry().all(), ids=lambda m: m.metric_id)
def test_date_default_reference_is_resolvable(metric):
    """日期默认值里用 `:参数名` 时，被引用的参数必须「声明在前」且是整数。

    生成器是按声明顺序逐个解析参数的，引用一个排在后面的参数会直接报错。
    这条约定只存在于 JSON 的书写顺序里，肉眼很难发现，所以必须用测试固化。
    """
    order = list(metric.param_names)

    for index, param in enumerate(metric.params):
        if param.type != "date" or not param.has_default:
            continue
        for ref in re.findall(r"[+-]:([A-Za-z_]\w*)", str(param.default)):
            assert ref in order, (
                f"{metric.metric_id}.{param.name} 引用了不存在的参数 :{ref}"
            )
            assert order.index(ref) < index, (
                f"{metric.metric_id}.{param.name} 引用了声明在其后的参数 :{ref}，"
                f"生成器按顺序解析会取不到值"
            )
            assert metric.get_param(ref).type == "int", (
                f"{metric.metric_id}.{param.name} 只能引用整数参数，:{ref} 不是"
            )


# ---------------------------------------------------------------------------
# 四、检索能力
# ---------------------------------------------------------------------------

def test_get_unknown_metric_raises(registry):
    """取不存在的指标必须抛 MetricNotFoundError，并且错误信息里带上可用指标。"""
    with pytest.raises(MetricNotFoundError) as exc:
        registry.get("not_a_real_metric")
    assert "可用指标" in str(exc.value)


def test_get_metric_returns_object(registry):
    metric = registry.get("retention_rate")
    assert metric.metric_name.startswith("留存率")
    assert metric.get_param("day_n") is not None
    assert metric.get_param("no_such_param") is None


def test_find_by_alias_prefers_specific_alias(registry):
    """关键词粗匹配时，「更具体的别名」应该排在前面。

    问题里同时出现 tutorial_funnel 的专属别名「新手引导漏斗」
    和 retention_rate 的泛化别名「留存」时，应该优先命中新手引导漏斗，
    因为长别名（更具体）比短别名（更泛化）信息量更大。
    """
    hits = registry.find_by_alias("新手引导漏斗的转化情况怎么样，另外留存如何")
    assert hits, "应该至少命中一个指标"
    assert hits[0].metric_id == "tutorial_funnel"


def test_find_by_alias_returns_empty_for_unrelated_text(registry):
    """跟指标完全无关的问题，粗匹配应该返回空 —— 这正是「知识缺失」的信号。"""
    assert registry.find_by_alias("今天食堂吃什么") == []


def test_catalog_text_covers_all_metrics(registry):
    """给 LLM 的指标目录必须包含全部指标 ID 和参数名。"""
    catalog = registry.to_catalog_text()
    for metric in registry.all():
        assert metric.metric_id in catalog
        assert metric.metric_name in catalog
    assert str(len(registry)) in catalog


# ---------------------------------------------------------------------------
# Phase 6 · 提示词目录精简（别名去重 + 紧凑渲染）
# ---------------------------------------------------------------------------

def test_catalog_line_is_more_compact_than_summary_line(registry):
    """catalog_line 必须比 summary_line 短：它俩是「提示词省着放 / 工具放全」的分工。

    这条断言的价值在于**锁住设计意图**：将来如果有人图省事把两个方法合并，
    或者往 catalog_line 里加回「分类」字段，测试会立刻失败并提醒他
    「提示词里的每一个字段都是按 token 付过账的」。
    """
    for metric in registry.all():
        assert len(metric.catalog_line()) < len(metric.summary_line())


def test_catalog_line_drops_category(registry):
    """紧凑版不带分类 —— 分类只服务前端分组，对 metric_id 映射没有区分度。"""
    for metric in registry.all():
        assert "分类:" not in metric.catalog_line()
        assert "分类:" in metric.summary_line()


def test_aliases_are_deduplicated_against_name_and_id(registry):
    """别名去重：与指标名主干 / 名称括号缩写 / metric_id 重复的别名要被剔除。

    这是 Phase 6 精简的核心动作，必须回归锁定 —— 否则哪天有人往 JSON 里
    补一个和名称一字不差的别名，token 就又悄悄涨回去了。
    """
    for metric in registry.all():
        kept = metric._dedup_aliases()
        normalized = {metric._normalize_alias(a) for a in kept}

        # ① 不能出现归一化后完全相同的重复别名
        assert len(normalized) == len(kept), f"{metric.metric_id} 的别名仍有重复"

        # ② 不能与 metric_id 重复
        assert metric._normalize_alias(metric.metric_id) not in normalized

        # ③ 不能与「去掉括号后的名称主干」重复
        core = re.sub(r"[（(][^）)]*[）)]", "", metric.metric_name)
        assert metric._normalize_alias(core) not in normalized

        # ④ 不能与名称括号内的缩写重复（如「日活跃用户数（DAU）」里的 DAU）
        for group in re.findall(r"[（(]([^）)]*)[）)]", metric.metric_name):
            for part in re.split(r"[/、,，]", group):
                if part.strip():
                    assert metric._normalize_alias(part) not in normalized


def test_dedup_keeps_colloquial_aliases(registry):
    """去重必须「保守」：只删重复项，不能把口语别名一起删掉。

    这是防「优化过头」的护栏 —— 别名是模型把口语映射到 metric_id 的线索，
    删过头会逼模型多调一次 list_metrics，反而让整包上下文重发一遍，得不偿失。
    """
    retention = registry.get("retention_rate")
    kept = retention._dedup_aliases()
    assert any("次留" in a or "留存" in a for a in kept), "留存类的口语别名被误删了"