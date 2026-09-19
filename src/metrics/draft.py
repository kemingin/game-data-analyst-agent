# -*- coding: utf-8 -*-
"""
指标草稿生成 · 校验 · 序列化（半自动闭环的中段）
================================================
本模块把「一张（或多张）表的列元信息」变成「一份可人工复审的指标草稿」。

【它在整个闭环里的位置】

    dataset.tables（列元信息）
        │  schema_roles.infer_roles()      ① 列 → 语义角色
        ▼
    {date: "dt", user_id: "uid", ...}
        │  metric_templates（模板资产）      ② 角色齐备的模板 → 渲染真实列名
        ▼
    DraftMetric（候选指标 + 来源表 + 命中模板）
        │  人工复审（勾选 / 手工新增 / 改 SQL）
        ▼
    validate_metrics()                      ③ 落盘前把配置错误挡在门外
        │
        ▼
    write_scheme_file()                     ④ 原子写 → 登记 → 切换数据集指向

【为什么草稿按「表结构」生成，而不是按「用户当前的问题」生成？】
    这是本模块最重要的一个取舍。见实施计划里的说明，核心是：
    指标方案是**长期资产**（一次建好反复用），而问题是一次性的。
    按问题生成的话，用户每问一个新指标都要重走一遍「生成 → 复审 → 确认」，
    反而比一次建全更累。问题 → 指标的映射由 Agent 现有的指标目录匹配负责。

【为什么校验要在这里再做一遍（而不是只靠 SQLValidator）？】
    因为两者管的是不同的错误：
      · SQLValidator 管「SQL 本身危不危险」（只读、表名白名单、行数上限）；
      · validate_metrics 管「这份指标配置是否自洽」——
        metric_id 重复、来源表写了个本数据集没有的表、
        SQL 里残留着 {date} 这种没替换掉的占位符、
        声明了 :start_date 参数却没在 params 里写。
    后三类错误不会触发任何安全异常，但会让 Agent 在运行时静默失败，
    所以必须在「落盘之前」一次性全部报出来（诚实失败，不写坏文件）。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

from src import config as cfg
from src.exceptions import SQLSecurityError
from src.metrics.schema_roles import infer_roles
from src.metrics.templates import (
    MetricTemplate,
    load_templates,
    substitute_placeholders,
)
from src.sqlgen.validator import SQLValidator


# ---------------------------------------------------------------------------
# 一、常量与正则
# ---------------------------------------------------------------------------
# 指标 ID 规范：小写字母开头，后接小写字母/数字/下划线。
# 【为什么用白名单正则？】metric_id 会进入系统提示词的指标目录、也会被
#   用作 JSON 的键，还是 LLM 输出里的映射目标。放行大写或空格会让
#   「模型输出的 metric_id」与「注册表里的 metric_id」比不相等 ——
#   表现为「明明有这个指标却报 MetricNotFound」，排查成本很高。
_METRIC_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

# 允许的参数类型（与 registry.MetricParam.type 的约定一致）
_PARAM_TYPES: frozenset[str] = frozenset({"date", "int", "string"})

# 残留的花括号占位符（渲染后不该再有）
_LEFTOVER_BRACE_RE = re.compile(r"\{([a-z_]+)\}")

# :参数 占位符（与 templates.py 保持同一套规则）
_PARAM_RE = re.compile(r"(?<!:):([a-z_][a-z0-9_]*)\b")

# 校验时用来替换 :参数的假值。
# 【为什么需要「假值渲染」？】
#   1. 让 SQL 变成可直接执行的字面量形态，便于把校验后的语句给人看；
#   2. 彻底消除「参数名恰好撞上关键字/表名解析」的干扰 ——
#      :start_date 这类绑定变量在词法上仍是标识符，
#      某些解析路径（如 FROM 子句提取）可能把它当表名。
#      换成 '2000-01-01' 之后，校验只看 SQL 的结构，不看参数长什么样。
_FAKE_PARAM_VALUE: dict[str, str] = {
    "date": "'2000-01-01'",
    "int": "1",
    "string": "'x'",
}


# ---------------------------------------------------------------------------
# 二、数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DraftMetric:
    """一条候选指标 = 注册表格式的定义 + 它的来源（可追溯到哪张表哪几列）。

    【为什么把「定义」和「来源」放在同一个对象里，而不是两个平行列表？】
      平行列表（metrics + matches）必须靠下标或 metric_id 对齐，
      而任何一次过滤/排序都可能让两边错位 —— 错位后界面会把 A 指标的
      来源表显示成 B 指标的，且**看起来完全正常**。放进同一个对象，
      错位在类型层面就不可能发生。
    """

    definition: dict[str, Any]          # 与 metrics_registry.json 的 metrics[] 同构
    template_id: str                    # 命中的模板 ID（手工搭建的指标为空串）
    table: str                          # 命中的表名（手工搭建的指标由用户填写）
    roles: dict[str, str] = field(default_factory=dict)  # 角色 → 真实列名

    # ---------------- 便捷访问 ----------------
    @property
    def metric_id(self) -> str:
        return str(self.definition.get("metric_id", ""))

    @property
    def metric_name(self) -> str:
        return str(self.definition.get("metric_name", ""))

    @property
    def sql_template(self) -> str:
        return str(self.definition.get("sql_template", ""))

    def source_text(self) -> str:
        """人类可读的来源说明，用于复审界面（例：`user_daily_snapshot`（date=dt, user_id=uid））。"""
        if not self.roles:
            return f"`{self.table}`"
        detail = "，".join(f"{role}={col}" for role, col in sorted(self.roles.items()))
        return f"`{self.table}`（{detail}）"

    @classmethod
    def from_definition(cls, definition: dict[str, Any]) -> "DraftMetric":
        """从一份手工搭建的定义造 DraftMetric（无模板、无角色推断结果）。

        【为什么手工搭建也要包成 DraftMetric？】因为落盘前的校验与序列化
          只认一种形态。若手工指标走另一条分支，那条分支就会缺少校验 ——
          而「手工输入」恰恰是最需要校验的入口。
        """
        tables = tuple(definition.get("source_tables") or ())
        return cls(
            definition=dict(definition),
            template_id="",
            table=tables[0] if tables else "",
            roles={},
        )


@dataclass(frozen=True)
class SkippedTemplate:
    """一条没能匹配上任何表的模板（附带原因，用于界面解释「为什么没生成 X」）。"""

    template_id: str
    metric_id: str
    metric_name: str
    missing_roles: tuple[str, ...]

    def reason_text(self) -> str:
        if not self.missing_roles:
            return "未找到合适的表"
        return "数据中未识别到角色：" + "、".join(self.missing_roles)


@dataclass(frozen=True)
class DraftResult:
    """草稿生成结果：候选指标 + 未命中的模板。"""

    metrics: tuple[DraftMetric, ...] = ()
    skipped: tuple[SkippedTemplate, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.metrics

    @property
    def metric_ids(self) -> tuple[str, ...]:
        return tuple(m.metric_id for m in self.metrics)

    def to_registry_dict(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """转成 metrics_registry.json 的结构。"""
        return build_registry_json([m.definition for m in self.metrics], meta)

    def summary_text(self) -> str:
        """一行式摘要（供界面提示与日志使用）。"""
        if self.is_empty:
            return (
                "未生成任何候选指标：数据中没有识别到可用的语义角色"
                "（需要至少一列能被识别为日期 / 用户 / 活跃标志等）。请使用手工搭建。"
            )
        ids = "、".join(self.metric_ids)
        return f"共生成 {len(self.metrics)} 个候选指标：{ids}"


# ---------------------------------------------------------------------------
# 三、草稿生成
# ---------------------------------------------------------------------------

def generate_draft(
    tables: Sequence[Any],
    template_path: str | Path | None = None,
) -> DraftResult:
    """按表结构生成候选指标草稿。

    Args:
        tables: 表清单，每项需暴露 ``.name`` 与 ``.columns``
                （与 ``DatasetTable`` 同构；``columns`` 为
                ``[{"name","sql_name","sql_type","is_date"}]``）。
                刻意**鸭子类型**而不 import ``src.data.dataset`` ——
                指标语义层不该反向依赖数据层，否则「数据层可被替换」
                这个分层承诺就破了。
        template_path: 模板资产路径（默认内置资产；测试可注入）。

    Returns:
        DraftResult。**识别不到角色时不抛错，而是返回空结果** ——
        这是刻意的「诚实失败」：草稿为空是一个可展示、可解释的状态，
        而抛异常会让整块界面报红，用户看不到任何出路。

    算法（三步，全部确定性、零 LLM 调用）：
        1) 逐表推断角色 → ``[{table: roles}, ...]``
        2) 按 ``priority`` 升序遍历模板（数值小的优先），
           对每个模板找**第一张**角色齐备的表，渲染 SQL 与业务定义
        3) 同一 ``metric_id`` 只保留第一次命中的结果
           （即 priority 最高的那种表形态胜出）
    """
    items = load_templates(str(template_path) if template_path else None)

    # ① 逐表推断角色
    table_roles: list[tuple[str, dict[str, str]]] = []
    for t in tables:
        roles = infer_roles(list(getattr(t, "columns", ()) or ()))
        table_roles.append((getattr(t, "name", ""), roles))

    metrics: list[DraftMetric] = []
    skipped: list[SkippedTemplate] = []
    produced: set[str] = set()

    for tpl in items:  # 已按 priority 升序
        hit = _first_match(tpl, table_roles)
        if hit is None:
            skipped.append(
                SkippedTemplate(
                    template_id=tpl.template_id,
                    metric_id=tpl.metric_id,
                    metric_name=tpl.metric_name,
                    missing_roles=tuple(_best_missing(tpl, table_roles)),
                )
            )
            continue

        if tpl.metric_id in produced:
            # 同一指标已有更优形态命中 —— 这是「多模板共存」的正常结果，
            # 不是失败，所以既不产出也不记为 skipped（记了会让界面噪声很大）。
            continue

        table_name, roles = hit
        produced.add(tpl.metric_id)
        metrics.append(
            DraftMetric(
                definition=_build_definition(tpl, table_name, roles),
                template_id=tpl.template_id,
                table=table_name,
                roles=dict(roles),
            )
        )

    return DraftResult(metrics=tuple(metrics), skipped=tuple(skipped))


def _first_match(
    tpl: MetricTemplate,
    table_roles: Sequence[tuple[str, dict[str, str]]],
) -> tuple[str, dict[str, str]] | None:
    """找第一张「角色齐备」的表；没有则返回 None。"""
    for table_name, roles in table_roles:
        if not tpl.missing_roles(roles):
            return table_name, roles
    return None


def _best_missing(
    tpl: MetricTemplate,
    table_roles: Sequence[tuple[str, dict[str, str]]],
) -> list[str]:
    """在所有表里挑「缺得最少」的那张，报出它缺哪些角色。

    【为什么要挑缺得最少的，而不是固定报第一张表的？】
      提示语的用途是「告诉用户差什么就能生成这个指标」。
      固定报第一张表的话，用户可能看到「缺 register_date」，
      而实际上另一张表只缺一个角色 —— 提示会把人引向错误的方向。
    """
    if not table_roles:
        return list(tpl.required_roles)
    return min(
        (tpl.missing_roles(roles) for _, roles in table_roles),
        key=len,
    )


def _build_definition(
    tpl: MetricTemplate,
    table: str,
    roles: dict[str, str],
) -> dict[str, Any]:
    """把模板渲染成一条注册表格式的指标定义。

    【注意 sql_template 里不再有 {table} / {角色}】
      这是刻意的：注册表里的指标 SQL 是**字面量 SQL**（与内置方案一致），
      表名写死、来源由 source_tables 声明。这样运行时链路（SQLGenerator →
      Validator → Executor）完全不需要知道「模板」这个概念，
      方案对 Agent 是透明的。
    """
    return {
        "metric_id": tpl.metric_id,
        "metric_name": tpl.metric_name,
        "aliases": list(tpl.aliases),
        "category": tpl.category,
        # 业务定义里也可能引用列名（例：以 {active_flag}=1 判定活跃），
        # 与 SQL 用同一套替换规则，避免定义里留着 {xxx} 让人看不懂
        "business_definition": substitute_placeholders(
            tpl.business_definition, table, roles
        ),
        "unit": tpl.unit,
        "good_direction": tpl.good_direction,
        "dimensions": list(tpl.dimensions),
        "params": [dict(p) for p in tpl.params],
        "sql_template": tpl.render_sql(table, roles),
        "source_tables": [table],
        "output_columns": [dict(c) for c in tpl.output_columns],
        "notes": tpl.notes,
        "owner": tpl.owner,
        "updated_at": tpl.updated_at,
    }


# ---------------------------------------------------------------------------
# 四、校验
# ---------------------------------------------------------------------------

def render_for_validation(sql: str, params: Iterable[dict[str, Any]]) -> str:
    """把 ``:参数`` 替换成同类型的假值，得到一个可执行形态的 SQL。

    仅用于**校验与预览**，绝不用于执行（执行时参数走真正的绑定）。
    """
    out = sql
    for p in params:
        name = str(p.get("name", "")).strip()
        if not name:
            continue
        fake = _FAKE_PARAM_VALUE.get(str(p.get("type", "string")), "'x'")
        out = re.sub(r"(?<!:):" + re.escape(name) + r"\b", fake, out)
    return out


def validate_metrics(
    metrics: Sequence[dict[str, Any]],
    allowed_tables: Iterable[str],
) -> list[str]:
    """逐条校验指标定义，返回问题清单（空列表 = 全部通过）。

    【为什么返回清单而不是抛异常？】
      因为界面的职责是「把问题一次列全」让人改。逐条抛异常的话，
      用户改一条、跑一次、再看到下一条 —— 而这类错误通常一次有好几个。
      返回清单让「一次修完」成为可能。
      落盘由调用方根据「清单为空」来决定，本函数不做任何副作用。

    检查项：
      1. metric_id 非空、符合命名规范、全局唯一
      2. metric_name / business_definition 非空（口径必须写清楚）
      3. sql_template 非空、且**不残留 {xxx} 占位符**（残留 = 渲染失败）
      4. source_tables 非空、且都在本数据集的表清单内
      5. params 类型合法；SQL 里的 :参数 与 params 声明**双向一致**
      6. SQL 通过 SQLValidator（只读 / 单语句 / 表名白名单 / 行数上限）
    """
    problems: list[str] = []
    allowed = {str(t).lower() for t in allowed_tables}
    seen_ids: set[str] = set()

    for index, metric in enumerate(metrics, start=1):
        raw_id = str(metric.get("metric_id", "")).strip()
        label = raw_id or f"第 {index} 条指标"

        # --- 1. metric_id ---
        if not raw_id:
            problems.append(f"{label}：缺少 metric_id")
        elif not _METRIC_ID_RE.match(raw_id):
            problems.append(
                f"{label}：metric_id 只能用小写字母开头、含小写字母/数字/下划线"
                f"（当前：{raw_id}）"
            )
        elif raw_id in seen_ids:
            problems.append(f"{label}：metric_id 重复，同一方案内必须唯一")
        else:
            seen_ids.add(raw_id)

        # --- 2. 名称与口径 ---
        if not str(metric.get("metric_name", "")).strip():
            problems.append(f"{label}：缺少 metric_name")
        if not str(metric.get("business_definition", "")).strip():
            problems.append(
                f"{label}：缺少 business_definition —— 指标口径是业务约定，"
                f"必须写清楚，否则模型与人都无从判断算法是否一致"
            )

        # --- 3. SQL 与残留占位符 ---
        sql = str(metric.get("sql_template", "") or "")
        leftover = sorted(set(_LEFTOVER_BRACE_RE.findall(sql)))
        if not sql.strip():
            problems.append(f"{label}：缺少 sql_template")
        elif leftover:
            problems.append(
                f"{label}：SQL 中残留未替换的占位符 "
                f"{'、'.join('{' + n + '}' for n in leftover)}；"
                f"草稿生成的 SQL 应是字面量，手工填写时请直接写真实表名/列名"
            )

        # --- 4. 来源表 ---
        source_tables = tuple(str(t) for t in (metric.get("source_tables") or ()))
        outside: list[str] = []
        if not source_tables:
            problems.append(f"{label}：source_tables 不能为空（必须声明数据来源）")
        else:
            outside = [t for t in source_tables if t.lower() not in allowed]
            if outside:
                problems.append(
                    f"{label}：来源表不在本数据集内：{'、'.join(outside)}；"
                    f"本数据集可用的表：{'、'.join(sorted(allowed)) or '（无）'}"
                )

        # --- 5. 参数一致性 ---
        params = tuple(metric.get("params") or ())
        declared: set[str] = set()
        for p in params:
            pname = str(p.get("name", "")).strip()
            ptype = str(p.get("type", ""))
            if not pname:
                problems.append(f"{label}：参数缺少 name 字段")
                continue
            if pname in declared:
                problems.append(f"{label}：参数名重复：{pname}")
            declared.add(pname)
            if ptype not in _PARAM_TYPES:
                problems.append(
                    f"{label}：参数 {pname} 的类型 {ptype or '（空）'} 不受支持；"
                    f"只支持 {'/'.join(sorted(_PARAM_TYPES))}"
                )

        used = set(_PARAM_RE.findall(sql)) if sql else set()
        if used != declared:
            only_sql = sorted(used - declared)
            only_json = sorted(declared - used)
            detail = []
            if only_sql:
                detail.append(f"SQL 用了但未声明的参数：{'、'.join(only_sql)}")
            if only_json:
                detail.append(f"声明了但 SQL 未用的参数：{'、'.join(only_json)}")
            problems.append(f"{label}：参数与占位符不一致 —— " + "；".join(detail))

        # --- 6. SQL 安全校验 ---
        # 只在前面的结构检查都过了才做：否则 SQL 本身可能还是残缺的，
        # 安全校验报出来的会是「表名解析失败」这类**次生**错误，
        # 把真正的原因（少了个花括号）埋掉。
        if sql.strip() and not leftover and source_tables and not outside:
            try:
                SQLValidator(allowed_tables=allowed).validate(
                    render_for_validation(sql, params),
                    allowed_tables=source_tables,
                )
            except SQLSecurityError as exc:
                problems.append(f"{label}：SQL 未通过安全校验 —— {exc}")

    return problems


# ---------------------------------------------------------------------------
# 五、序列化与落盘
# ---------------------------------------------------------------------------

def build_registry_json(
    metrics: Sequence[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """产出符合 ``metrics_registry.json`` 结构的字典。

    【meta 为什么要有默认值？】内置注册表的 meta 是人工维护的说明性内容，
      而用户确认的草稿方案没有「委员会评审」这一环 —— 给它一个
      说明「本方案由草稿生成并经人工确认」的默认 meta，
      比留空更利于日后追溯这份 JSON 是怎么来的。
    """
    default_meta = {
        "registry_name": "草稿生成的指标方案",
        "registry_version": "1.0",
        "updated_at": date.today().isoformat(),
        "owner": "由半自动指标草稿闭环生成（人工复审确认）",
        "description": (
            "本方案由「列语义角色推断 + 指标模板匹配」自动生成候选草稿，"
            "经人工复审勾选/修改后确认落盘。指标口径仍以本文件的定义为准，"
            "运行时由程序按 SQL 模板填充参数生成 SQL，LLM 不直接写 SQL。"
        ),
    }
    if meta:
        default_meta.update(meta)
    return {"meta": default_meta, "metrics": [dict(m) for m in metrics]}


def write_scheme_file(
    registry: dict[str, Any],
    scheme_id: str,
    target_dir: str | Path | None = None,
) -> Path:
    """把方案注册表原子写入 ``<target_dir>/<scheme_id>/metrics_registry.json``。

    Args:
        registry: ``build_registry_json`` 的产物。
        scheme_id: 方案 ID（同时是目录名，必须通过白名单校验）。
        target_dir: 方案根目录，默认 ``cfg.SCHEMES_DIR``。

    Returns:
        写入的文件路径。

    【为什么是「先写 .tmp 再 os.replace」？】
      直接 open(path, "w") 写的话，进程在写到一半时挂掉会留下**半截 JSON**。
      而这份 JSON 是方案的唯一事实来源 —— 半个文件等于方案彻底损坏，
      用户连「改回去」的机会都没有。
      os.replace 在同一文件系统内是原子的：要么看到旧文件、要么看到新文件，
      不存在中间态。与 DatasetStore._save 是同一套做法。

    【为什么每个方案一个子目录？】与 data/datasets/<id>/ 的布局对齐；
      将来方案若需要附带别的资产（如口径变更说明），有地方放。
    """
    if not _METRIC_ID_RE.match(scheme_id):
        # 方案 ID 会变成目录名，必须先过白名单 —— 否则 "../" 能写穿目录
        raise ValueError(
            f"非法的方案 ID：{scheme_id}；只允许小写字母开头、含小写字母/数字/下划线"
        )

    root = Path(target_dir) if target_dir else cfg.SCHEMES_DIR
    scheme_dir = root / scheme_id
    scheme_dir.mkdir(parents=True, exist_ok=True)
    target = scheme_dir / "metrics_registry.json"

    payload = json.dumps(registry, ensure_ascii=False, indent=2)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)
    return target


def register_and_activate(
    store: Any,
    dataset_id: str,
    scheme_id: str,
    display_name: str,
    registry: dict[str, Any],
    derived_from: str | None = None,
    target_dir: str | Path | None = None,
) -> Path:
    """落盘 → 登记方案 → 把数据集指向该方案（闭环的最后一步）。

    【为什么把这三步收在一个函数里？】
      它们是一个**原子业务动作**：「确认启用这套指标」。
      拆开写在界面层，任何一步漏掉都会得到一个坏状态：
        · 只落盘不登记 → 方案文件存在但索引里没有，数据集无法引用；
        · 只登记不切换 → 用户以为启用了，实际还在跑旧方案（且界面无提示）。
      收在一处后，「启用」只有一条路径，且可在单测里直接验证三步都生效。

    Args:
        store: ``DatasetStore``（鸭子类型，避免指标语义层 import 数据层）。
        derived_from: 血缘 —— 这套方案派生自哪个旧方案（复用需求留痕）。
    """
    path = write_scheme_file(registry, scheme_id, target_dir)
    store.register_scheme(
        display_name=display_name,
        registry_path=path,
        scheme_id=scheme_id,
        derived_from=derived_from,
    )
    store.set_scheme(dataset_id, scheme_id)
    return path


def make_scheme_id(dataset_id: str, existing: Iterable[str]) -> str:
    """基于数据集 ID 生成一个不冲突的方案 ID（例：`upload_demo_scheme_2`）。

    【为什么不用时间戳？】时间戳 ID（如 scheme_20260919_1530）不可读，
      而方案 ID 会出现在界面与索引里，人需要一眼看出「这是哪个数据集的方案」。
      用数据集 ID 做前缀 + 递增序号，既可读又稳定可预期（同样的输入得到同样的 ID）。
    """
    base = re.sub(r"[^a-z0-9_]", "_", dataset_id.lower()).strip("_") or "scheme"
    base = base[:30]
    taken = {str(s) for s in existing}
    if f"{base}_scheme" not in taken:
        return f"{base}_scheme"
    n = 2
    while f"{base}_scheme_{n}" in taken:
        n += 1
    return f"{base}_scheme_{n}"
