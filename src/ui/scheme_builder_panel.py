# -*- coding: utf-8 -*-
"""
侧边栏「指标方案搭建」面板（半自动指标草稿闭环的前端）
================================================================================
Phase 7 把架构升级成「多数据集 + 指标方案可复用」之后，闭环缺了最关键的一段：

    上传完数据集 → 只能从已有方案里挑一个 → 而内置方案的 12 个指标全部依赖
    内置的 9 张表，新上传的库一张都没有 → `check_scheme_compatibility()` 诚实报告
    「可用指标 0 个」→ **用户彻底卡住，没有任何自助出路。**

本面板补上这一段：

    ┌──────────────────────────────────────────────────────┐
    │ ① 兼容性提示（复用已有方案 / 当前匹配度）              │
    │ ② 🔍 生成候选指标草稿（按表结构推，不调 LLM）          │
    │ ③ 草稿复审：逐条勾选 + 展开看业务定义与 SQL            │
    │ ④ 🛠 手工新增指标（草稿识别不到时的兜底出路）          │
    │ ⑤ ✅ 确认并启用 → 校验全过才落盘 + 切换数据集指向       │
    └──────────────────────────────────────────────────────┘

【三个设计决策，以及为什么】

① 草稿按「表结构」生成，不按「用户当前的问题」生成
   方案是长期资产（一次建好反复用），问题是一次性的。按问题生成的话，
   用户每问一个新指标都要重走一遍「生成 → 复审 → 确认」，反而更累。
   问题 → 指标的映射由 Agent 现有的指标目录匹配负责（registry.to_catalog_text）。

② 纯规则，不调 LLM
   指标口径是业务约定，必须人工确认（项目既有约束）。规则匹配确定性、
   可单测、零 token，而且它的失败模式是**可预测**的（列名不规范就识别不到），
   这正是「半自动」里「自动」那半应该有的样子。

③ 手工搭建也生成「新方案」，而不是允许直接改 SQL 绕过护栏
   产物统一是一份 registry JSON，于是复用了现有的全部链路：
   加载校验 / 白名单 / 生成器 / 执行器 / 对账。若允许「只对这一条指标放行」，
   就等于在「LLM 不直接写 SQL」这条护栏上开了一个洞。

【Streamlit 特有的一处坑：草稿必须存在 session_state 里】
  每次交互都会重跑整份脚本。若把草稿算在渲染路径上，那么「勾选一条指标」
  这个动作会触发重算 —— 而重算会**重置勾选状态**（因为勾选值是按 metric_id
  存 widget key 的，重算本身不丢，但草稿对象若每次都新建，
  展开区里的内容会闪、顺序也可能变）。存进 session_state 后，
  草稿只在用户点「生成」时变一次，复审过程是稳定的。
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from src import config as cfg
from src.data.dataset import Dataset, DatasetStore
from src.metrics.draft import (
    DraftMetric,
    DraftResult,
    build_registry_json,
    generate_draft,
    make_scheme_id,
    register_and_activate,
    validate_metrics,
)

# --- session_state 键 ---
# 【为什么统一用 scheme_ 前缀？】dataset_panel._DYNAMIC_WIDGET_PREFIXES 里
#   登记了 "scheme_"，切换数据集时会按前缀整片清掉这些键 ——
#   否则上一份数据集的草稿会留在界面上，用户勾选的是**别的数据**的指标，
#   而界面看起来完全正常。这是本项目一路在防的那类「来源错位」。
_DRAFT_KEY = "scheme_draft"
_MANUAL_KEY = "scheme_manual"
_FLASH_KEY = "_scheme_flash"   # 故意用下划线开头：切换数据集时不该被清掉


# ===========================================================================
# 一、对外入口
# ===========================================================================

def render_scheme_builder(dataset: Dataset, store: DatasetStore) -> None:
    """渲染指标方案搭建面板（放进 dataset_panel 的折叠区里调用）。"""
    flash = st.session_state.pop(_FLASH_KEY, None)
    if flash:
        st.success(flash, icon="🎉")

    _render_compatibility(dataset, store)

    if not dataset.tables:
        st.warning(
            "本数据集没有登记任何表结构（可能是建库失败只留了元信息）。"
            "请重新上传，或先在「数据集」面板核对导入结果。",
            icon="⚠️",
        )
        return

    _render_draft_actions(dataset)

    result = st.session_state.get(_DRAFT_KEY)
    if result is None:
        st.caption(
            "点上面的按钮，系统会按表结构推断「哪一列扮演哪个语义角色」"
            "（日期 / 用户 / 活跃标志 / 金额…），再匹配内置的指标模板生成候选草稿。"
            "草稿是**按表结构生成的全部候选指标**，不是只针对某个问题 —— "
            "方案建一次可反复用。"
        )
        return

    _render_skipped(result)
    selected = _render_draft_list(dataset, result)
    manual_selected = _render_manual_section(dataset)
    _render_confirm(dataset, store, selected, manual_selected)


# ===========================================================================
# 二、兼容性提示
# ===========================================================================

def _render_compatibility(dataset: Dataset, store: DatasetStore) -> None:
    """当前方案与本数据集的匹配度（复用 DatasetStore 的检查逻辑）。"""
    info = store.check_scheme_compatibility(dataset.dataset_id)
    if info.get("ok"):
        st.success(
            f"当前方案（`{dataset.scheme_id}`）匹配："
            f"{len(info.get('usable_metrics', []))} / {info.get('total_metrics', 0)} 个指标可用。",
            icon="✅",
        )
    else:
        st.warning(info.get("message") or "当前方案与本数据集不匹配。", icon="⚠️")


# ===========================================================================
# 三、生成草稿
# ===========================================================================

def _render_draft_actions(dataset: Dataset) -> None:
    """「生成候选指标草稿」按钮。"""
    if st.button("🔍 生成候选指标草稿", key="scheme_generate", width="stretch"):
        # generate_draft 只读 dataset.tables（列元信息），不连库、不调 LLM ——
        # 所以不需要 spinner，也不会有超时风险。
        st.session_state[_DRAFT_KEY] = generate_draft(dataset.tables)
        # 手工新增的指标属于「上一轮草稿的补充」，重新生成时一并清掉，
        # 避免新旧两批混在一起后分不清哪条是哪次生成的。
        st.session_state.pop(_MANUAL_KEY, None)
        st.rerun()


def _render_skipped(result: DraftResult) -> None:
    """列出「没生成」的指标及原因。

    【为什么要把失败原因显式展示出来，而不是只展示成功的？】
      因为用户看得到「没有留存率」这件事，但不看原因就会以为是系统漏了。
      把「数据里没有 register_date 这一列」写出来，用户才知道
      要么改列名重传、要么手工搭建 —— 这是可行动的反馈。
    """
    if not result.skipped:
        return
    with st.expander(f"未生成的指标（{len(result.skipped)} 个）", expanded=False):
        st.caption(
            "以下指标需要的数据列在本数据集中没有被识别到。"
            "可以按提示调整列名后重新上传，或用下面的「手工新增指标」自行搭建。"
        )
        for item in result.skipped:
            st.markdown(f"- **{item.metric_name}**（`{item.metric_id}`）：{item.reason_text()}")


# ===========================================================================
# 四、草稿复审
# ===========================================================================

def _render_draft_list(dataset: Dataset, result: DraftResult) -> list[DraftMetric]:
    """逐条渲染候选指标，返回被勾选保留的那些。"""
    if result.is_empty:
        st.error(result.summary_text(), icon="🚫")
        return []

    st.markdown(f"**候选指标草稿**（{len(result.metrics)} 条）")
    allowed = dataset.table_names
    selected: list[DraftMetric] = []

    for metric in result.metrics:
        keep = st.checkbox(
            f"{metric.metric_name}（`{metric.metric_id}`）",
            value=True,
            key=f"scheme_keep_d_{metric.metric_id}",
            help="取消勾选表示这条指标不纳入方案。",
        )
        st.caption(f"　来源：{metric.source_text()} · 模板：`{metric.template_id}`")

        problems = validate_metrics([metric.definition], allowed)
        for problem in problems:
            # 草稿来自内置模板，正常情况下不该有问题；真出现了必须让人看见，
            # 而不是等点「确认」时才报 —— 那时用户已经逐条勾选过了。
            st.warning(f"　{problem}", icon="⚠️")

        with st.expander("查看业务定义与 SQL", expanded=False):
            st.markdown(f"**业务定义**：{metric.definition['business_definition']}")
            st.code(metric.sql_template, language="sql")
            params = metric.definition.get("params") or []
            if params:
                st.dataframe(
                    [
                        {
                            "参数": p.get("name"),
                            "类型": p.get("type"),
                            "默认值": p.get("default", "-"),
                            "说明": p.get("label") or p.get("description", ""),
                        }
                        for p in params
                    ],
                    hide_index=True,
                )

        if keep:
            selected.append(metric)

    return selected


# ===========================================================================
# 五、手工新增指标
# ===========================================================================

def _render_manual_section(dataset: Dataset) -> list[DraftMetric]:
    """手工搭建指标（一行输入框），返回被勾选保留的手工指标。

    【为什么这个兜底入口是必需的，而不是「锦上添花」？】
      纯规则推断的代价就是「列名不规范就识别不到」。没有手工入口的话，
      这类数据集在系统里就是死路一条 —— 而它恰恰是最需要分析的
      「别人给的一份脏数据」。
    """
    manual: list[DraftMetric] = list(st.session_state.get(_MANUAL_KEY) or [])

    with st.expander("🛠 手工新增指标（草稿识别不到时的兜底）", expanded=False):
        st.caption(
            "SQL 里请写**真实表名与列名**（不是 {占位符}）；"
            "日期参数用 `:参数名` 绑定，并在「参数定义」里声明，例如 "
            '`[{"name":"start_date","type":"date","default":"@data_end-6"}]`。'
            "只能写 SELECT 查询，表名必须来自本数据集。"
        )

        metric_id = st.text_input("指标 ID", key="scheme_manual_id",
                                  placeholder="例：weekly_active_users")
        metric_name = st.text_input("指标名称", key="scheme_manual_name",
                                    placeholder="例：周活跃用户数")
        definition = st.text_area("业务定义（口径说明）", key="scheme_manual_def",
                                  placeholder="例：统计区间内每周至少登录一次的去重用户数。",
                                  height=68)
        source_tables = st.multiselect(
            "来源表",
            sorted(dataset.table_names),
            key="scheme_manual_tables",
            help="必须与 SQL 里 FROM 的表一致，否则安全校验会拦下。",
        )
        params_text = st.text_area("参数定义（JSON 数组，可留空）", key="scheme_manual_params",
                                   placeholder='[{"name":"start_date","type":"date","default":"@data_end-6"}]',
                                   height=68)
        sql_text = st.text_area("SQL 模板", key="scheme_manual_sql", height=140,
                                placeholder="SELECT date, COUNT(DISTINCT user_id) AS wau\nFROM 你的表名\nWHERE date BETWEEN :start_date AND :end_date\nGROUP BY date")

        if st.button("加入待确认列表", key="scheme_manual_add"):
            new_metric, error = _build_manual_metric(
                metric_id=metric_id,
                metric_name=metric_name,
                definition=definition,
                source_tables=source_tables,
                params_text=params_text,
                sql_text=sql_text,
            )
            if error:
                st.error(error, icon="🚫")
            else:
                assert new_metric is not None
                manual.append(new_metric)
                st.session_state[_MANUAL_KEY] = manual
                st.rerun()

        if manual:
            st.divider()
            st.markdown(f"**已加入 {len(manual)} 条手工指标**")
            for index, item in enumerate(manual):
                st.checkbox(
                    f"{item.metric_name}（`{item.metric_id}`）",
                    value=True,
                    key=f"scheme_keep_m_{item.metric_id}",
                    help="取消勾选表示这条指标不纳入方案。",
                )
                st.caption(f"　来源：`{item.table or '-'}`")
                if st.button("移除", key=f"scheme_manual_del_{index}"):
                    manual.pop(index)
                    st.session_state[_MANUAL_KEY] = manual
                    st.rerun()

            # 按勾选状态过滤。此时上面所有 checkbox 都已实例化，
            # 读它们的 session_state 是安全的（Streamlit 只禁止「创建后再改」）。
            manual = [
                m for m in manual
                if st.session_state.get(f"scheme_keep_m_{m.metric_id}", True)
            ]

    return manual


def _build_manual_metric(
    metric_id: str,
    metric_name: str,
    definition: str,
    source_tables: list[str],
    params_text: str,
    sql_text: str,
) -> tuple[DraftMetric | None, str | None]:
    """把手工表单拼成一条指标定义；返回 (指标, 错误信息)。

    【为什么在「加入列表」时就做一次 JSON 解析与校验，而不是等确认？】
      因为参数的 JSON 语法错误是最常见的输入问题，而它离「确认」按钮
      可能还有好几步。当场报错，用户才容易把错误和输入框对应起来。
      注意这里只做**语法层**的解析，完整校验仍由 validate_metrics 在落盘前统一做
      （两处职责不同：这里管「能不能构成一条指标」，那里管「这份方案能不能用」）。
    """
    params: list[dict[str, Any]] = []
    if params_text.strip():
        try:
            parsed = json.loads(params_text)
        except json.JSONDecodeError as exc:
            return None, f"参数定义不是合法 JSON：{exc}"
        if not isinstance(parsed, list):
            return None, "参数定义必须是 JSON 数组（每个元素是一个参数对象）。"
        params = [p for p in parsed if isinstance(p, dict)]

    metric = {
        "metric_id": metric_id.strip(),
        "metric_name": metric_name.strip(),
        "business_definition": definition.strip(),
        "sql_template": sql_text,
        "source_tables": list(source_tables),
        "params": params,
        "aliases": [],
        "category": "自定义",
        "unit": "",
        "good_direction": "up",
        "dimensions": ["date"],
        "output_columns": [],
        "notes": "由人工在「指标方案搭建」面板手工新增。",
        "owner": "由用户手工搭建",
        "updated_at": "",
    }
    return DraftMetric.from_definition(metric), None


# ===========================================================================
# 六、确认并启用
# ===========================================================================

def _render_confirm(
    dataset: Dataset,
    store: DatasetStore,
    selected: list[DraftMetric],
    manual_selected: list[DraftMetric],
) -> None:
    """校验 → 落盘 → 登记方案 → 切换数据集指向（闭环最后一步）。"""
    st.divider()
    all_metrics = selected + manual_selected
    st.markdown(f"**待确认 {len(all_metrics)} 条指标**")

    if st.button("✅ 确认并启用这套指标方案", key="scheme_confirm",
                 type="primary", width="stretch"):
        _activate(dataset, store, all_metrics)


def _activate(
    dataset: Dataset,
    store: DatasetStore,
    metrics: list[DraftMetric],
) -> None:
    """执行「确认启用」：任何一条校验不过都不落盘。"""
    if not metrics:
        st.error("至少要保留一条指标才能启用方案。", icon="🚫")
        return

    definitions = [m.definition for m in metrics]
    problems = validate_metrics(definitions, dataset.table_names)
    if problems:
        # 【为什么不「尽力保存能存的部分」？】
        #   因为方案是一份**自洽资产**：半份方案会让「哪些指标能用」变成
        #   一个需要逐条试才知道的问题，而用户以为已经配置好了。
        #   诚实失败 + 一次列全所有问题，比留下一个残缺方案有用得多。
        st.error("方案未通过校验，**没有落盘**，请按下面的提示修正：", icon="🚫")
        for problem in problems:
            st.markdown(f"- {problem}")
        return

    scheme_id = make_scheme_id(
        dataset.dataset_id,
        [s.scheme_id for s in store.list_schemes()],
    )
    display_name = f"{dataset.display_name} · 指标方案"

    with st.spinner("正在写入指标方案并切换数据集指向…"):
        try:
            path = register_and_activate(
                store=store,
                dataset_id=dataset.dataset_id,
                scheme_id=scheme_id,
                display_name=display_name,
                registry=build_registry_json(
                    definitions,
                    {
                        "registry_name": display_name,
                        "description": (
                            f"由数据集「{dataset.display_name}」的草稿生成并经人工确认。"
                            f"共 {len(definitions)} 个指标。"
                        ),
                    },
                ),
                # 血缘：本方案派生自数据集原先指向的方案。
                # 先记下来再切换 —— 切换之后 dataset.scheme_id 就变了。
                derived_from=dataset.scheme_id,
                target_dir=cfg.SCHEMES_DIR,
            )
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"启用失败：{exc}", icon="🚫")
            return

    st.session_state[_FLASH_KEY] = (
        f"已启用新指标方案「{display_name}」（`{scheme_id}`，{len(definitions)} 个指标），"
        f"数据集已切换到该方案。方案文件：{path}"
    )
    # 【为什么必须清对话状态？】上一轮问答里的每个数字都来自**旧方案的口径**。
    #   不清的话，用户会在新方案下看到旧方案的结论 —— 数字看起来正常、
    #   来源却已经换了，正是本项目一直在防的那类错误。
    # 延迟 import：dataset_panel 在模块级 import 本模块（它负责组装侧边栏），
    # 这里若也模块级 import 回去就会形成循环。与 dataset.py 延迟 import
    # src.data.upload 是同一套做法。
    from src.ui.dataset_panel import reset_conversation_state

    reset_conversation_state()
    st.rerun()
