# -*- coding: utf-8 -*-
"""
指标草稿模板加载器
==================
把 `metric_templates.json` 加载成有类型的对象，并在**加载时就自检**。

【为什么加载时就要自检？】
    模板资产是人工维护的 JSON，人手一定会写错：占位符拼错、参数名对不上、
    角色名写了个不存在的。这些错误的后果是「草稿生成出来但 SQL 跑不通」，
    而报错信息会是「SQL 语法错误」——离真正的原因（模板里少了个花括号）很远。
    把自检放在加载期，等于把「配置错误」变成「启动就报错」，排查成本从
    「翻 SQL 日志」降到「看一行异常」。

    这与 src/metrics/registry.py 对指标注册表的处理是同一套纪律：配置即代码。

【与 registry.py 的分工】
    registry.py 管「已确认的指标方案」（给 Agent 用）；
    templates.py 管「用来生成草稿的模板」（给草稿生成用）。
    两者结构相似但用途不同：模板多一层 {角色} 占位符与 required_roles。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src import config as cfg


# 默认资产路径：与 metrics_registry.json 并列放在指标语义层目录下
DEFAULT_TEMPLATE_PATH: Path = cfg.BASE_DIR / "src" / "metrics" / "metric_templates.json"

# 匹配 {xxx} 占位符。用 [a-z_]+ 限定，避免把 SQL 里的花括号（本项目不用）
# 或 Python 风格的格式化串误当成角色。
_BRACE_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")

# {table} 是**表名**占位符，不是语义角色。
# 【为什么必须显式排除？】它的写法与角色占位符完全一样（同为 {小写名}），
#   若混进角色集合，自检的「SQL 用了未声明的角色」会把每个模板都判为非法 ——
#   而报错信息会指向一个根本不存在的问题（模板没写 table 角色）。
#   把它单列成常量，比在正则里写负向断言更直白，也让「谁是角色」一目了然。
_TABLE_PLACEHOLDER = "table"

# 匹配 :参数 占位符。SQLite 的 :name 绑定参数语法。
# 【为什么用 \b 边界？】避免把 PostgreSQL 的 ::类型转换或时间字符串 "10:30"
# 里的冒号误判成参数。
_PARAM_PLACEHOLDER_RE = re.compile(r"(?<!:):([a-z_][a-z0-9_]*)\b")

# 角色名白名单 —— 必须与 schema_roles.py 的 _ROLE_SPECS 保持一致。
# 【为什么要在这里再列一份，不直接 import？】
#   import 会让「模板资产」反向依赖「角色推断模块」的实现细节；
#   而这份清单是**契约**（模板里能写哪些角色），不是实现。
#   单测里有一条专门断言两者一致，防止两边漂移。
KNOWN_ROLES: frozenset[str] = frozenset({
    "user_id", "date", "register_date", "active_flag", "event_name",
    "channel", "version", "revenue", "level",
})


def _role_placeholders_in(sql: str) -> frozenset[str]:
    """取出 SQL 里的全部**角色**占位符（已剔除 {table}）。"""
    return frozenset(
        name for name in _BRACE_PLACEHOLDER_RE.findall(sql)
        if name != _TABLE_PLACEHOLDER
    )


def substitute_placeholders(text: str, table: str, roles: dict[str, str]) -> str:
    """把文本里的 {table} 与 {角色} 替换成真实表名/列名。

    【为什么用 replace 而不是 str.format？】
      因为模板里同时有 {角色} 与 :参数 两种占位符，而 SQL 本身还可能含
      别的花括号。str.format 会把所有花括号都当占位符，遇到未提供的 key
      直接抛 KeyError；逐个 replace 更可控：只替换我们明确知道的键，
      其余原样保留。

    【为什么做成模块级函数而不是 MetricTemplate 的方法？】
      因为**业务定义文本**里也可能引用列名（例：dau 的定义写着
      「以 {active_flag}=1 作为活跃判据」）。草稿生成时需要把定义和 SQL
      按同一套规则替换 —— 两处各写一遍替换逻辑，迟早会漂移成两种行为。
    """
    out = text.replace("{" + _TABLE_PLACEHOLDER + "}", table)
    for role, column in roles.items():
        out = out.replace("{" + role + "}", column)
    return out


@dataclass(frozen=True)
class MetricTemplate:
    """一份指标模板（字段与 JSON 一一对应）。"""

    template_id: str
    metric_id: str
    metric_name: str
    business_definition: str
    sql_template: str
    required_roles: tuple[str, ...]
    priority: int = 100
    aliases: tuple[str, ...] = ()
    category: str = ""
    unit: str = ""
    good_direction: str = "up"
    dimensions: tuple[str, ...] = ()
    params: tuple[dict[str, Any], ...] = ()
    output_columns: tuple[dict[str, str], ...] = ()
    notes: str = ""
    owner: str = ""
    updated_at: str = ""

    @property
    def role_placeholders(self) -> frozenset[str]:
        """模板 SQL 里用到的全部 {角色} 占位符（不含 {table}）。"""
        return _role_placeholders_in(self.sql_template)

    @property
    def param_placeholders(self) -> frozenset[str]:
        """模板 SQL 里用到的全部 :参数 占位符。"""
        return frozenset(_PARAM_PLACEHOLDER_RE.findall(self.sql_template))

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p["name"] for p in self.params)

    def render_sql(self, table: str, roles: dict[str, str]) -> str:
        """把 {table} 与 {角色} 占位符替换成真实表名/列名。"""
        return substitute_placeholders(self.sql_template, table, roles)

    def missing_roles(self, roles: dict[str, str]) -> list[str]:
        """返回本模板需要、但给定角色表里没有的角色（空 = 可用）。"""
        return [r for r in self.required_roles if r not in roles]


def _validate_template(raw: dict[str, Any], seen_ids: set[str]) -> None:
    """单份模板的加载期自检。任何问题都直接抛 ValueError（Fail Fast）。"""
    tid = raw.get("template_id", "<未知>")

    required = ("template_id", "metric_id", "metric_name", "business_definition",
                "sql_template", "required_roles")
    missing = [f for f in required if not raw.get(f)]
    if missing:
        raise ValueError(f"模板 {tid} 缺少必填字段：{', '.join(missing)}")

    if tid in seen_ids:
        raise ValueError(f"模板 ID 重复：{tid}")
    seen_ids.add(tid)

    roles = tuple(raw["required_roles"])
    unknown = [r for r in roles if r not in KNOWN_ROLES]
    if unknown:
        raise ValueError(
            f"模板 {tid} 声明了未知角色：{', '.join(unknown)}；"
            f"可用角色：{', '.join(sorted(KNOWN_ROLES))}"
        )

    sql = raw["sql_template"]

    # ① SQL 里用到的角色占位符，必须全部在 required_roles 里声明 ——
    #    否则草稿生成时不知道该往哪个角色取值，替换会静默失败。
    used = _role_placeholders_in(sql)
    undeclared = sorted(used - set(roles))
    if undeclared:
        raise ValueError(
            f"模板 {tid} 的 SQL 使用了未在 required_roles 中声明的角色："
            f"{', '.join(undeclared)}"
        )

    # ② required_roles 里声明了、但 SQL 里没用到的角色，属冗余声明，
    #    会让模板「明明不需要这列却匹配不上」，也一并拦下。
    unused = sorted(set(roles) - used)
    if unused:
        raise ValueError(
            f"模板 {tid} 声明了 required_roles 却在 SQL 中未使用：{', '.join(unused)}"
        )

    # ③ :参数 占位符与 params 必须一一对应（双向）
    declared_params = {p["name"] for p in raw.get("params", [])}
    used_params = set(_PARAM_PLACEHOLDER_RE.findall(sql))
    if used_params != declared_params:
        only_sql = sorted(used_params - declared_params)
        only_json = sorted(declared_params - used_params)
        detail = []
        if only_sql:
            detail.append(f"SQL 用了但未声明的参数：{', '.join(only_sql)}")
        if only_json:
            detail.append(f"声明了但 SQL 未用的参数：{', '.join(only_json)}")
        raise ValueError(f"模板 {tid} 参数与占位符不一致 —— " + "；".join(detail))

    # ④ {table} 必须出现，否则生成的 SQL 不知道该查哪张表
    if "{table}" not in sql:
        raise ValueError(f"模板 {tid} 的 SQL 缺少 {{table}} 占位符")


@lru_cache(maxsize=4)
def load_templates(path: str | None = None) -> tuple[MetricTemplate, ...]:
    """加载并自检模板资产（带缓存，与 get_registry 同风格）。

    【为什么参数是 str 而不是 Path？】lru_cache 要求参数可哈希，
    Path 虽然可哈希，但同一条路径写成 "a/b" 与 Path("a/b") 会被当成两个键，
    统一用字符串最省心。
    """
    target = Path(path) if path else DEFAULT_TEMPLATE_PATH

    if not target.exists():
        raise FileNotFoundError(f"指标模板资产不存在：{target}")

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"指标模板资产不是合法 JSON：{target} —— {exc}") from exc

    items = raw.get("templates")
    if not items:
        raise ValueError(f"指标模板资产为空：{target}")

    seen_ids: set[str] = set()
    templates: list[MetricTemplate] = []
    for item in items:
        _validate_template(item, seen_ids)
        templates.append(
            MetricTemplate(
                template_id=item["template_id"],
                metric_id=item["metric_id"],
                metric_name=item["metric_name"],
                business_definition=item["business_definition"],
                sql_template=item["sql_template"],
                required_roles=tuple(item["required_roles"]),
                priority=int(item.get("priority", 100)),
                aliases=tuple(item.get("aliases", ())),
                category=item.get("category", ""),
                unit=item.get("unit", ""),
                good_direction=item.get("good_direction", "up"),
                dimensions=tuple(item.get("dimensions", ())),
                params=tuple(item.get("params", ())),
                output_columns=tuple(item.get("output_columns", ())),
                notes=item.get("notes", ""),
                owner=item.get("owner", ""),
                updated_at=item.get("updated_at", ""),
            )
        )

    # 按 priority 升序返回 —— 数值越小越优先（草稿生成按此顺序择优）。
    # 【为什么不在加载时就去重 metric_id？】因为「同一 metric_id 有多份模板」
    #   是刻意支持的（快照版/事件版），去重是**草稿生成**的职责，
    #   而模板资产本身要保留全部形态。
    templates.sort(key=lambda t: (t.priority, t.template_id))
    return tuple(templates)