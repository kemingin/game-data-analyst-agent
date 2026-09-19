# -*- coding: utf-8 -*-
"""
侧边栏「数据集」面板（多数据集架构的前端入口）
================================================================================
上一轮侧边栏只有「数据概览 + 指标口径 + 审计日志」三块，它们是**单数据集**时代的
产物 —— 数据概览读的是全局库，指标字典读的是全局注册表。

引入「多数据集」后，侧边栏多出一块：

    ┌──────────────────────────────────────────────┐
    │ 🗂 数据集（下拉选择）                          │  ← 现在要分析哪份数据
    │    ✅/⚠️ 方案兼容性提示                        │  ← 不匹配时明确说清后果
    │    📤 上传 CSV → 自动建表 → 登记为新数据集      │  ← 新增数据的唯一入口
    │    👀 导入结果核对（仅上传数据集）              │  ← 列类型 / 日期识别 / 前 N 行
    └──────────────────────────────────────────────┘

【为什么做成独立模块，而不是塞进 app.py？】
  职责不同：app.py 管「整份脚本怎么编排」，这里管「分析谁的数据」。
  混在一起改一个必然碰到另一个。这与 Phase 4 把图表选型抽到 src/ui/charts.py
  是同一个取舍。

【两条依赖边界】
  1. 本模块不 import pandas —— 「选一个数据集」是轻量操作（读索引 + 查目录），
     不该为它付出加载 pandas 的代价。上传路径才需要 pandas，那条路径在
     src/data/upload.py 里（由 dataset.add_from_upload 延迟 import）。
  2. 本模块不 import Agent 层 —— 它只负责「拿到用户的意图并同步状态」，
     真正组装上下文在 src/context.py（组装根）。

【本面板的核心难点：Streamlit 的「无页面内存」模型】
  Streamlit 每次交互都把整份脚本从头重跑一遍，因此有两件事必须显式处理：

  ① 程序化切换选择器
     「上传成功后自动切到新数据集」要求我们**改一个已存在的 widget 的值**。
     Streamlit 禁止在 widget 实例化之后改它的 session_state，所以只能：
        本次运行记下意图 → st.rerun() → 下次运行在 widget 创建**之前**写入。
     这就是 _PENDING_SWITCH 存在的理由。

  ② 切换数据集时必须清状态
     上一份数据集的问答结果（turns）里，每个数字都来自**另一份数据**。
     不清的话，用户会在新数据集下看到旧数据集的结论 —— 数字看起来正常、
     来源却已经换了，正是本项目一直在防的那类错误。
     清理动作挂在 selectbox 的 on_change 回调上（回调在重跑**之前**执行，
     顺序才对；写成「if 变了就清」会在渲染历史对话时被旧值回填）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src import config as cfg
from src.data.dataset import Dataset, DatasetStore
from src.ui.overview import load_table_preview

# 「待切换的数据集」一次性标志（上传成功后置位，下一次运行消费掉）。
# 【为什么用一次性标志，而不是维护一个「当前数据集」的副本？】
#   副本方案要回答「副本和 widget 值哪个更新」，而这个判断没有可靠依据：
#   用户手动切换时 widget 更新（回调里同步副本），上传成功时副本更新 ——
#   两个方向都写一遍，任何一处漏写都会变成「切了又弹回去」。
#   改成一次性标志后，数据源只有一个（selectbox 的 widget 值），
#   程序化切换只在这一个明确的入口发生，不存在方向歧义。
_PENDING_SWITCH = "_pending_dataset_switch"

# 上传成功后置位：下一次运行先清空上传框。
# 【为什么要清？】Streamlit 会把已上传的文件留在 session_state 里，成功后
#   不清理的话文件仍挂在界面上，用户容易再点一次「导入并建库」，
#   结果是同一份数据被建了两遍（ID 会被自动加上 _2 后缀）。
_PENDING_CLEAR_UPLOADER = "_pending_clear_uploader"

# 上传成功的提示语（成功后要 st.rerun()，提示必须先存起来再在下一次运行显示）
_FLASH_KEY = "_dataset_flash"

# 切换数据集时要清掉的 widget key（固定名）。
# 这些是 app.py 里用到的：分类筛选、图表纵轴、复核人、勾选、确认按钮、导出按钮、
# 示例问题按钮。它们的值都只对「上一份数据集的那一轮问答」有意义。
# 其中 dataset_uploader / dataset_selector / upload_scheme / upload_name 属于本面板，
# 由 _PENDING_CLEAR_UPLOADER 单独处理（不能在这里清：selectbox 正在用它的值）。
_FIXED_WIDGET_KEYS: tuple[str, ...] = ("metric_category_filter",)
_DYNAMIC_WIDGET_PREFIXES: tuple[str, ...] = (
    "ycol_",      # 图表纵轴切换（key = ycol_{turn}_{block}）
    "chart_",     # Plotly 图表实例（key = chart_{turn}_{block}）
    "reviewer_",  # 复核人输入框（key = reviewer_{turn}）
    "agree_",     # 「我已复核」勾选框（key = agree_{turn}）
    "confirm_",   # 「记录人工确认」按钮（key = confirm_{turn}）
    "export_",    # 导出按钮（key = export_{turn}）
    "example_",   # 空态示例问题按钮（key = example_{index}）
)


# ===========================================================================
# 一、状态清理
# ===========================================================================

def _clear_stale_widget_keys() -> None:
    """清掉带轮次下标的 widget 状态。

    【为什么必须清？】这些 key 里嵌着 turn_index，而切换数据集后 turns 被清空、
    turn_index 会从 0 重新开始 —— 于是「上一份数据集第 1 轮的图表纵轴选择」
    会被当成「这一份数据集第 1 轮的纵轴选择」直接复用，选项对不上就报错、
    对得上就串台。两种结果都不能接受。

    【为什么不干脆清空整个 session_state？】
      因为里面还有「当前选中的数据集」（dataset_selector）与上传框状态，
      以及 Streamlit 自身的内部键。全清会把「我刚切到哪个数据集」一起抹掉，
      表现为「切换后又弹回内置数据集」。
    """
    for key in _FIXED_WIDGET_KEYS:
        st.session_state.pop(key, None)
    stale = [
        key
        for key in st.session_state.keys()
        if any(str(key).startswith(prefix) for prefix in _DYNAMIC_WIDGET_PREFIXES)
    ]
    for key in stale:
        del st.session_state[key]


def reset_conversation_state() -> None:
    """重置「与某一份数据集绑定」的会话状态。

    清三样东西：
      · turns            —— 历史问答。每个数字都来自上一份数据，必须清；
      · pending_question —— 待处理的示例问题（属于上一份数据集的提问意图）；
      · widget 状态      —— 见 _clear_stale_widget_keys。
    【审计日志（audit_log）为什么也清？】
      它记录的是「谁确认了哪条结论」，而那条结论属于上一份数据集。
      日志在界面上不带数据集标注，跨数据集混放会让人误以为结论来自当前数据 ——
      在数据产品里，「来源错位」比「记录少几条」严重得多。
    """
    st.session_state.turns = []
    st.session_state.pop("pending_question", None)
    _clear_stale_widget_keys()


def _on_dataset_change() -> None:
    """数据集下拉框的 on_change 回调：切换后清空旧数据集的痕迹。

    【为什么清理动作必须挂在这里？】
      Streamlit 的执行顺序是「回调 → 整个脚本重跑」。回调里清空，重跑时读到的
      就是干净状态。如果改写成「在选择器下面比较新旧值再清」，清理发生在重跑的
      中段 —— 而紧接着渲染历史对话时用的还是旧 turns，等于没清。
    """
    reset_conversation_state()


# ===========================================================================
# 二、缓存
# ===========================================================================

@st.cache_resource(show_spinner=False)
def get_store() -> DatasetStore:
    """全局唯一的数据集索引管理器。

    用 cache_resource 而不是每次重跑新建：索引文件读一次就够，
    而 Streamlit 每次交互都会重跑脚本，不缓存就是「每点一次按钮读一次 JSON」。
    另外 DatasetStore 的写方法（_save）是原地修改内部字典，
    缓存成单例才能保证「上传后立刻能列出新数据集」。
    """
    return DatasetStore()


@st.cache_data(show_spinner=False)
def _compatibility(dataset_id: str, cache_key: str) -> dict:
    """方案兼容性检查（缓存）。

    【为什么 cache_key 要参与缓存键？】
      这个检查会加载指标 JSON 并遍历全部指标，属于「数据集结构变了才需要重算」
      的结果。而结构变化一定伴随 dataset.cache_key()（含 created_at）变化，
      用它当缓存键既能命中、又不会漏掉更新 —— 与 Agent 缓存用的是同一个判据。
    """
    return get_store().check_scheme_compatibility(dataset_id)


@st.cache_data(show_spinner=False)
def _table_preview(db_path: str, table: str, limit: int, cache_key: str) -> dict:
    """表数据预览（缓存）。

    【为什么要缓存？】Streamlit 会渲染折叠面板里的内容（只是浏览器侧不展开），
      所以这段查询每次重跑都会执行。数据预览是只读且不变的（除非重新上传，
      而那会让 cache_key 变化），缓存掉最省事。
    """
    return load_table_preview(db_path, table, limit)


# ===========================================================================
# 三、选择器与兼容性提示
# ===========================================================================

def _dataset_label(dataset: Dataset) -> str:
    """下拉框里的显示文本：名称 + 来源 + 表数。"""
    tag = "内置" if dataset.source == "builtin" else "上传"
    return f"{dataset.display_name}（{tag} · {len(dataset.tables)} 张表）"


def _render_compatibility(store: DatasetStore, dataset: Dataset) -> None:
    """方案兼容性提示条。

    【为什么地基阶段就要做这个提示？】
      本轮不做「为新数据集生成指标方案」，所以上传的数据集只能沿用已有方案。
      若该方案的指标模板依赖的表在新库里不存在，每次提问都会以
      「引用了未授权的表」失败 —— 用户会以为系统坏了。
      与其让用户撞墙，不如在选择数据集时就把这件事说清楚。
      这是「诚实失败」，不是「假装能用」。
    """
    info = _compatibility(dataset.dataset_id, dataset.cache_key())

    if info.get("ok"):
        st.success(
            f"指标方案匹配：{len(info.get('usable_metrics', []))} / "
            f"{info.get('total_metrics', 0)} 个指标可用。",
            icon="✅",
        )
        return

    st.warning(info.get("message") or "指标方案与当前数据集不匹配。", icon="⚠️")
    if info.get("missing_tables"):
        with st.expander("缺少哪些表", expanded=False):
            st.markdown("\n".join(f"- `{name}`" for name in info["missing_tables"]))
            st.caption(
                "指标方案的 SQL 模板里写死了来源表名，表不存在时查询会被白名单拦下。"
            )


# ===========================================================================
# 四、上传
# ===========================================================================

def _render_upload_form(store: DatasetStore) -> None:
    """上传表单：选文件 → 起名 → 选指标方案 → 点按钮才真正导入。

    【为什么必须有个按钮，不能「选了文件就导入」？】
      Streamlit 会把上传的文件保存在 session_state 里跨重跑保留。
      如果「有文件就导入」，那么之后每一次重跑（切换图表纵轴、展开侧边栏……）
      都会重新导入一次 —— 一份数据被反复写库，且用户完全无感。
      按钮把「有文件」和「要导入」这两件事分开，是这里唯一正确的做法。
    """
    uploaded = st.file_uploader(
        "选择 CSV 文件（可多选，每个文件建成一张表）",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="dataset_uploader",
    )

    if not uploaded:
        st.caption(
            f"单文件 ≤ {cfg.UPLOAD_MAX_MB} MB、≤ {cfg.UPLOAD_MAX_ROWS:,} 行、"
            f"≤ {cfg.UPLOAD_MAX_COLUMNS} 列。系统会自动识别编码、清洗列名、"
            f"推断类型并把日期列建成索引。"
        )
        return

    default_name = Path(uploaded[0].name).stem
    name = st.text_input(
        "数据集名称",
        key="upload_name",
        placeholder=f"留空则用文件名：{default_name}",
    )

    # --- 指标方案选择：这就是「复用」的入口 ---
    schemes = store.list_schemes()
    scheme_ids = [s.scheme_id for s in schemes]
    scheme_labels = {s.scheme_id: s.display_name for s in schemes}
    if st.session_state.get("upload_scheme") not in scheme_ids:
        # 首次进入或方案列表变化时给出默认值（内置方案）。
        # 【为什么用 session_state 而不是 selectbox 的 index 参数？】
        #   因为 index 参数与「程序化设置 key」同时出现时，Streamlit 会打印
        #   「默认值与 Session State 冲突」的告警。统一走 session_state 最干净。
        st.session_state["upload_scheme"] = cfg.BUILTIN_SCHEME_ID
    scheme_id = st.selectbox(
        "指标方案",
        scheme_ids,
        format_func=lambda sid: scheme_labels.get(sid, sid),
        key="upload_scheme",
        help="选一个已有方案即可复用，不需要为新数据集重建指标定义。",
    )
    st.caption(
        "指标方案是**独立于数据集的资产**：多个数据集指向同一个方案 ID 即可复用，"
        "零复制、零同步成本 —— 这就是「我这套数据跟某某数据集指标一样」的实现方式。"
    )

    if st.button("导入并建库", key="upload_submit", type="primary", width="stretch"):
        _do_upload(
            store,
            uploaded,
            display_name=name.strip() or default_name,
            scheme_id=scheme_id,
        )


def _do_upload(
    store: DatasetStore,
    uploaded: list[Any],
    display_name: str,
    scheme_id: str,
) -> None:
    """执行导入，并逐条展示每个文件的结果。

    全部成功 → 切到新数据集 + 清空上传框 + 重跑；
    有任一文件失败 → 逐条列出原因，**不登记数据集**（诚实失败）。
    """
    files = [(item.name, item.getvalue()) for item in uploaded]

    with st.spinner("正在解析 CSV、建表并写入数据库…"):
        try:
            dataset, results = store.add_from_upload(
                files, display_name=display_name, scheme_id=scheme_id
            )
        except (ValueError, OSError) as exc:
            st.error(f"导入失败：{exc}", icon="🚫")
            return

    for result in results:
        if result.ok:
            st.success(
                f"`{result.source_file}` → 表 `{result.table_name}`"
                f"（{result.row_count:,} 行 · 编码 {result.encoding}）",
                icon="✅",
            )
            for warning in result.warnings[:3]:
                st.caption(f"　{warning}")
        else:
            st.error(f"`{result.source_file}` 导入失败：{result.error}", icon="❌")

    if not store.has(dataset.dataset_id):
        # add_from_upload 在失败时返回一个「未登记」的 Dataset，用 has() 判断。
        # 【为什么失败时整体拒绝？】部分成功的多表数据集对下游是灾难 ——
        # 指标模板会静默查不到表，报错信息却是「引用了未授权的表」，
        # 用户会以为自己上传成功了。
        st.warning(
            "本次没有创建数据集：存在导入失败的文件，请按上面的提示修正后重试。"
        )
        return

    total_rows = sum(table.row_count for table in dataset.tables)
    st.session_state[_FLASH_KEY] = (
        f"数据集「{dataset.display_name}」创建成功：{len(dataset.tables)} 张表 · "
        f"{total_rows:,} 行 · 数据窗口 "
        f"{dataset.data_start or '未知'} ~ {dataset.data_end or '未知'}"
    )
    # 切到新数据集：这里只置位，真正的赋值留到下一次运行的开头做 ——
    # selectbox 已在本轮实例化，此刻改它的 key 会直接抛 StreamlitAPIException。
    st.session_state[_PENDING_SWITCH] = dataset.dataset_id
    st.session_state[_PENDING_CLEAR_UPLOADER] = True
    reset_conversation_state()
    st.rerun()


# ===========================================================================
# 五、导入结果核对（仅上传数据集）
# ===========================================================================

def _render_preview(dataset: Dataset) -> None:
    """上传数据集的结构核对面板。

    【为什么只对上传数据集显示？】
      内置数据集的表结构是项目自带的、已经被 500+ 条单测覆盖，不需要每次核对；
      而上传数据集的列名与类型是**程序自动推断**的，用户必须能当场验证
      「编号列的前导零保住了吗」「日期列认出来了吗」——
      后者直接决定指标窗口能不能取到数。
    """
    if dataset.source != "upload" or not dataset.tables:
        return

    with st.expander("👀 导入结果核对", expanded=False):
        st.caption(
            "列名与类型由系统自动推断，请核对是否符合预期 —— "
            "尤其关注编号列（前导零必须保住）与日期列（决定指标窗口）。"
        )
        for index, table in enumerate(dataset.tables):
            if index:
                st.divider()
            st.markdown(
                f"**`{table.name}`** · {table.row_count:,} 行 · "
                f"来源 `{table.source_file or '-'}`"
            )
            if table.columns:
                st.dataframe(
                    [
                        {
                            "列名": col.get("sql_name") or col.get("name") or "-",
                            "类型": col.get("sql_type") or "-",
                            "日期列": "✓" if col.get("is_date") else "",
                            "样本值": " / ".join(
                                str(v) for v in (col.get("sample") or ())
                            )[:60],
                        }
                        for col in table.columns
                    ],
                    hide_index=True,
                )

            preview = _table_preview(
                str(dataset.db_path),
                table.name,
                cfg.UPLOAD_PREVIEW_ROWS,
                dataset.cache_key(),
            )
            if preview.get("error"):
                st.caption(f"预览不可用：{preview['error']}")
                continue
            columns = preview.get("columns") or []
            rows = preview.get("rows") or []
            if columns and rows:
                st.caption(f"前 {len(rows)} 行")
                st.dataframe(
                    [dict(zip(columns, row)) for row in rows],
                    hide_index=True,
                    height=200,
                )


# ===========================================================================
# 六、对外入口
# ===========================================================================

def render_dataset_picker(store: DatasetStore) -> str:
    """渲染整个数据集面板，返回**当前生效的 dataset_id**。

    调用方（app.py）拿到返回值后去 build_context，不需要自己判断「用户切没切」——
    「当前是谁」这件事只在本函数里决定，避免两处各判一次而不一致。

    【渲染顺序为什么不能变？】
        先消费「上一次运行遗留的两个待办」——清空上传框、切换到新数据集。
        这两个动作都必须发生在**对应 widget 实例化之前**：
        Streamlit 禁止在 widget 创建之后改它的 session_state 值。
        之后才是创建 selectbox，最后才是依赖选中值的提示与上传表单。
    """
    # --- 1. 处理上一次运行留下的待办（必须早于对应 widget 的创建）---
    if st.session_state.pop(_PENDING_CLEAR_UPLOADER, None):
        st.session_state.pop("dataset_uploader", None)

    # --- 2. 列数据集 ---
    datasets = store.list_datasets()
    if not datasets:
        st.error("没有任何数据集，且内置数据集自愈失败，请检查 data/datasets/index.json。")
        st.stop()

    options = [d.dataset_id for d in datasets]
    by_id = {d.dataset_id: d for d in datasets}

    # 程序化切换（上传成功）：下一次运行时在这里把新值写进 widget key，
    # 然后交给 selectbox 去渲染。这一步必须发生在 selectbox 实例化之前。
    pending = st.session_state.pop(_PENDING_SWITCH, None)
    if pending is not None and pending in by_id:
        st.session_state["dataset_selector"] = pending

    # 兜底：第一次运行或 widget 值已失效时，回落到内置数据集。
    # 【为什么不在这里再设一次 widget 值？】widget 已在下边的 selectbox 实例化
    # 时创建过了（前一次运行），取值的逻辑统一交给 Streamlit 的 widget key 机制，
    # 这里只在「值被清掉 / 从干净会话启动」时给一个初始值即可。
    if st.session_state.get("dataset_selector", None) not in by_id:
        st.session_state["dataset_selector"] = options[0]

    with st.expander("🗂 数据集", expanded=True):
        selected = st.selectbox(
            "当前数据集",
            options,
            format_func=lambda did: _dataset_label(by_id[did]),
            key="dataset_selector",
            on_change=_on_dataset_change,
            help="切换会清空当前对话与图表状态，并改用所选数据集的库与表白名单。",
        )

        flash = st.session_state.pop(_FLASH_KEY, None)
        if flash:
            st.success(flash, icon="🎉")

        dataset = by_id[selected]
        _render_compatibility(store, dataset)

        st.divider()
        _render_upload_form(store)

    _render_preview(by_id[selected])
    return selected
