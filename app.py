# -*- coding: utf-8 -*-
"""
Streamlit 前端（Phase 4）—— 项目主入口
================================================================================
启动方式：

    streamlit run app.py

【界面分成五块，每块只干一件事】

    侧边栏    数据集选择与上传 + 数据概览 + 指标口径卡片 + 人工确认审计日志
    对话区    用户提问 → Agent 作答（Streamlit 原生 chat 组件）
    数据区    指标卡 / 自动图表 / 原始结果表格
    过程区    默认折叠的「执行过程」：调了哪些工具、传了什么参数、跑了哪条 SQL

【执行顺序是硬约束，不能调换】

    main() 里的三步是刻意排的：

        ① 渲染数据集选择器        → 拿到「这一轮看哪个数据集」
        ② build_context(...)      → 组装上下文（库 / 指标 / 窗口 / 白名单）
        ③ 渲染其余界面与 Agent    → 全部只认 ctx

    为什么不能合成一步？因为「数据概览读哪个库」「指标字典读哪份 JSON」
    「Agent 连哪个库」这三件事的答案都由①决定。先渲染后判断，就会出现
    界面上写着数据集 A、查询却打在库 B 的错位 —— 而且不报错。

【这一层最重要的五个设计决策（设计要点）】

  1. Agent 实例必须用 st.cache_resource 缓存。
     Streamlit 的执行模型非常特殊：**每一次交互都会把整个脚本从头重跑一遍**。
     如果 GameDataAgent() 写在脚本主体里，用户每点一次按钮就要重建一次 LLM 客户端
     （连接池、重试状态全丢），还要重新加载一遍指标注册表。
     cache_resource 是官方为「全局单例资源」提供的解法。

  2. 问答结果要存进 st.session_state，而不是"重新问一遍"。
     LLM 调用是按 token 计费的。用户展开侧边栏、切换图表纵轴都会触发脚本重跑，
     如果每次重跑都重新调 Agent，一次 5 轮的对话能烧掉几十次 API 调用。
     存进 session_state 后，重跑只是「重新渲染」，不重新请求。

  3. 「执行过程」默认折叠。
     运营同学只看结论，开发/评审者才关心链路。
     默认展开会把结论淹没在技术细节里 —— 可解释性要「可获取」，而不是「强制展示」。

  4. 重大决策类结论必须过一道人工确认。
     这是项目硬约束第 4 条在前端的落地。做法不是"弹个提示"，而是：
     给出复核人输入框 + 勾选确认 + 写入审计日志。
     为什么这么重？因为「谁在什么时候确认了哪条结论」必须有迹可循 ——
     这正是数据产品里"人工兜底环节"应有的产品形态。

  5. 所有缓存的键里都必须带「数据集身份」。
     这是多数据集改造里最容易漏、也最难发现的一处：st.cache_resource /
     st.cache_data 的键由「函数名 + 参数」决定，改造前这几个函数**没有参数**，
     键恒定 —— 于是切换数据集后它们照旧返回上一个数据集的对象，
     表现为「界面上选了新数据集，数据概览和查询结果还是旧的」，而且不报错。
     现在它们的参数里都带着 dataset_id / cache_key，键随数据集身份变化。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import streamlit as st

from src import config as cfg
from src.agent.react_agent import AgentAnswer, AgentStep, GameDataAgent
from src.context import DatasetContext, build_context
from src.metrics.registry import Metric, MetricRegistry, get_registry
from src.ui.charts import (
    build_figure,
    headline_values,
    pick_chart_spec,
    to_dataframe,
)
from src.ui.dataset_panel import get_store, render_dataset_picker
from src.ui.overview import load_table_stats

# ---------------------------------------------------------------------------
# 页面级配置：必须是脚本里第一个 st.* 调用，否则 Streamlit 会报错
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="游戏数据智能分析师",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 空态时展示的示例问题。
# 为什么要给示例？因为「自然语言分析助手」最大的使用障碍不是不会用，
# 而是**不知道该问什么**。给几个真实运营会问的问题，转化率立刻不一样。
_EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "最近的次日留存率是多少？",
    "v2.0.0 上线后新用户留存变好了吗？",
    "各渠道的次日留存对比如何？",
    "最近 7 天的付费率和 ARPU 分别是多少？",
)

# 「重大决策类」的判定关键词（回答里的自我声明 + 问题里的决策意图）。
# 判定故意偏宽松：按项目规范，宁可多确认一次，也不要让一条"该人工拍板"的
# 建议悄悄溜进执行环节。这是「漏报比误报更贵」的典型场景。
_DECISION_ANSWER_MARKERS: tuple[str, ...] = ("人工复核", "需人工确认", "人工确认")
_DECISION_TOPIC_KEYWORDS: tuple[str, ...] = ("投放", "预算", "买量", "版本调优", "调优")

# 多轮追问时最多回带几轮历史
_MAX_HISTORY_EXCHANGES: int = 2


# ===========================================================================
# 一、缓存：把「贵的东西」只建一次
# ===========================================================================

@st.cache_resource(show_spinner=False)
def get_agent(dataset_id: str, cache_key: str) -> GameDataAgent:
    """按数据集缓存的 Agent 实例（含 LLM 客户端与工具执行器）。

    【为什么参数里必须有 cache_key？—— 这是多数据集改造里最容易漏的一处】
      st.cache_resource 的缓存键由「函数名 + 参数」决定。改造前这个函数**没有参数**，
      键恒定，所以切换数据集后它仍然返回上一个数据集的 Agent：
      指标注册表、库路径、数据窗口、表白名单全部还是旧的那套。
      而「A 方案的模板 + B 数据集的库」这种错配**不会报错**——
      若两个库恰好都有同名表，查询会成功并返回一个语义完全错误的数字，
      过程区显示的 SQL 看起来毫无异常。把 dataset.cache_key() 算进缓存键，
      键就跟着数据集身份走，这类错配在缓存层就被消解了。

    【为什么 dataset_id 也单独传？】
      只为了可读性与报错定位：cache_key 是拼接串，出问题时看不出是哪个数据集。
    """
    return GameDataAgent(context=build_context(dataset_id, store=get_store()))


@st.cache_resource(show_spinner=False)
def get_metric_registry(registry_path: str) -> MetricRegistry:
    """指标注册表（读 JSON 文件 + 建对象，没必要每次重跑都做）。

    路径进参数：每个数据集可能指向不同的指标方案文件，写死路径就等于
    「不管选哪个数据集，指标字典永远显示内置方案」。
    """
    return get_registry(registry_path)


@st.cache_data(ttl=600, show_spinner=False)
def get_table_stats(db_path: str, data_start: Any, data_end: Any) -> dict:
    """数据概览。加 10 分钟缓存：库结构不会秒变，而 COUNT(*) 全表扫是要花时间的。

    注意用的是 cache_data（缓存**数据**）而不是 cache_resource（缓存**资源**）：
    cache_data 会把返回值序列化一份副本，即使外部改了它也污染不到缓存本体。
    只读的展示数据适合 cache_data，带连接/句柄的对象才用 cache_resource。

    【日期参数为什么显式传 dataset 的值？】
      因为 cfg.DATA_START/END 是**内置数据集**的窗口。上传的数据集若照抄这个值，
      界面会出现「可用数据范围：2026-06-20 ~ 2026-09-17」而数据里根本没有这个区间
      —— 这是「配置的承诺冒充数据的真相」，比不显示更糟。
      dataset 的日期为 None 时传 None，overview 会显示「未知」（哨兵机制见 overview.py）。
    """
    return load_table_stats(db_path=db_path, data_start=data_start, data_end=data_end)


# ===========================================================================
# 二、业务判定与文本构造（纯逻辑，放在渲染函数之前）
# ===========================================================================

def is_decision_analysis(answer: AgentAnswer) -> bool:
    """判断这次问答是否属于「重大决策类分析」。

    两条命中路径，任一成立就算：
      1. 模型自己在回答里声明了「需人工复核」（提示词硬性规则第 5 条要求它必须写）；
      2. 用户问的是投放/预算/版本调优这类会花钱或动研发资源的问题。
    第 2 条是"机械兜底"——万一模型忘了声明，界面这一层仍然能拦住。
    """
    text = answer.answer or ""
    if any(marker in text for marker in _DECISION_ANSWER_MARKERS):
        return True
    question = answer.question or ""
    return any(keyword in question for keyword in _DECISION_TOPIC_KEYWORDS)


def fallback_notice(answer: AgentAnswer) -> str | None:
    """若本次回答来自备用模型，返回一句提示；走主模型时返回 None。

    为什么需要这个提示：降级保的是「可用性」，不是「体验」—— 真机实测备用模型
    （GLM-4-Flash）的响应耗时是主模型的 4~5 倍（12~18 秒 vs 平均 3.4 秒）。
    不提示的话，用户会以为页面卡死了，反而去点重试，白白多烧一次 token。

    判据取 cfg.LLM_PROVIDER_ORDER[0]（配置里的第一优先级）而不是写死 "deepseek"：
    以后调整主备顺序时，这里不用跟着改。
    """
    if not answer.provider:
        return None
    primary = cfg.LLM_PROVIDER_ORDER[0] if cfg.LLM_PROVIDER_ORDER else ""
    if not primary or answer.provider == primary:
        return None
    return (
        f"本次回答由**备用模型**（{answer.provider}/{answer.model}）生成："
        "主模型当前不可用，系统已自动降级，服务仍然可用。"
        "两点提醒：①备用模型响应明显更慢（实测为主模型的 4~5 倍），"
        "②其指令遵循能力较弱，出现过「不查库直接编数字」的情况（已被来源校验拦下），"
        "请对结论中的数字多留一分核对。"
    )


def build_history(turns: list[dict], max_exchanges: int = _MAX_HISTORY_EXCHANGES) -> list[dict]:
    """把历史问答整理成 OpenAI 消息格式，供多轮追问使用。

    两个刻意的限制：
      · 只取**成功**的轮次 —— 失败的回答没有 assistant 消息配它，
        直接拼进去会出现「连续两条 user 消息」，部分 API 会直接报 400。
      · 只取最近 2 轮 —— 多轮上下文能让「那版本上线后呢」这种追问成立，
        但历史越长 token 越贵，也越容易干扰模型判断当前问题。2 轮是够用的平衡点。
    """
    usable = [t for t in turns if t["answer"].ok]
    messages: list[dict[str, Any]] = []
    for turn in usable[-max_exchanges:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"].answer})
    return messages


def build_export_text(answer: AgentAnswer) -> str:
    """把一次问答导出成纯文本，便于复盘与留档。

    含「执行过程」而不只是结论：这份文件的价值在于事后能回答
    「当时这个数字是怎么算出来的」。只导出结论的话，三天后就没人敢用这个数了。
    """
    lines: list[str] = [
        "=" * 68,
        "游戏数据智能分析师 · 单次问答记录",
        f"导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 68,
        "",
        "【问题】",
        answer.question,
        "",
        "【回答】",
        answer.answer,
        "",
        "【执行过程】",
    ]

    for index, step in enumerate(answer.steps, start=1):
        if step.kind == "final_answer":
            lines.append(f"{index}. 第 {step.iteration} 轮：模型给出最终回答")
            continue
        if step.kind == "guard":
            lines.append(
                f"{index}. 第 {step.iteration} 轮：来源校验拦截"
                f"（模型给出了无出处的数字，已要求重新取数）"
            )
            lines.append(f"   被拦下的原文：{step.content or '（空）'}")
            continue
        status = "成功" if step.ok else "失败"
        args = json.dumps(step.arguments, ensure_ascii=False)
        lines.append(
            f"{index}. 第 {step.iteration} 轮：调用 {step.tool_name}"
            f"（{status}，{step.elapsed_ms:.0f} ms）参数={args}"
        )
        sql = (step.result or {}).get("rendered_sql")
        if sql:
            lines.append("   实际执行的 SQL（展示版）：")
            lines.extend("     " + row for row in str(sql).splitlines())

    usage = answer.usage or {}
    lines.extend(
        [
            "",
            "【运行信息】",
            f"模型：{answer.provider}/{answer.model}",
            f"循环轮数：{answer.iterations}",
            f"Token 用量：总 {usage.get('total_tokens', 0)}"
            f"（输入 {usage.get('prompt_tokens', 0)} / 输出 {usage.get('completion_tokens', 0)}）",
            f"总耗时：{answer.elapsed_ms:.0f} ms",
        ]
    )
    if answer.error:
        lines.append(f"错误信息：{answer.error}")
    # 降级这件事必须留在导出件里：导出的文本常被当成"分析记录"流转，
    # 事后追查「这条结论是谁算的、可不可信」时，模型来源是关键信息。
    if fallback_notice(answer):
        lines.append(
            f"降级提示：主模型不可用，本次回答由备用模型（{answer.provider}）生成，"
            "响应较慢且指令遵循能力较弱，结论中的数字建议二次核对。"
        )
    lines.append("")
    lines.append("提示：以上数字均来自数据库真实查询。涉及重大决策的建议需人工复核后执行。")
    return "\n".join(lines)


# ===========================================================================
# 三、侧边栏
# ===========================================================================

def render_sidebar_head(store: Any) -> str:
    """侧边栏上半部分：标题 + 数据集选择器。返回当前选中的 dataset_id。

    【为什么必须和下半部分拆成两个函数？】
      因为「选哪个数据集」决定了要组装哪个上下文，而数据概览、指标字典、Agent
      都依赖那个上下文。所以顺序只能是「先选 → 再装 → 后渲染」，
      写成一个大函数就会出现「先渲染数据概览、再知道要看哪个库」的倒置。
      这个顺序约束在下面的 main() 里体现得很直白。
    """
    with st.sidebar:
        st.markdown("## 🎮 游戏数据智能分析师")
        st.caption("自然语言 → 标准指标口径 → 受控 SQL → 真实数据")
        return render_dataset_picker(store)


def render_sidebar_tail(ctx: DatasetContext) -> None:
    """侧边栏下半部分：数据概览 + 指标字典 + 审计日志 + 页脚。

    全部依赖 ctx —— 它们展示的库、指标、窗口、白名单都必须来自同一个数据集，
    这正是「打包成上下文」的收益：这里只传一个参数，不可能传漏。
    """
    with st.sidebar:
        st.divider()
        _render_data_overview(ctx)
        st.divider()
        _render_metric_catalog(ctx)
        st.divider()
        _render_audit_log()
        st.divider()
        _render_footer_actions(ctx)


def _render_data_overview(ctx: DatasetContext) -> None:
    """数据概览：告诉用户「我的数据边界在哪」。

    日期显式取自 ctx.dataset（而不是让 overview 自己读 config），理由见
    get_table_stats 的注释：上传数据集的窗口必须来自数据本身。
    """
    stats = get_table_stats(
        str(ctx.db_path), ctx.dataset.data_start, ctx.dataset.data_end
    )

    with st.expander("📊 数据概览", expanded=True):
        if stats.get("error"):
            st.error(str(stats["error"]))
            return

        window = stats["window"]
        # 数据集的日期识别不出来时（上传的 CSV 里没有日期列），窗口是 None。
        # 此时显示「未知」而不是套用内置窗口 —— 宁可说不知道，也不能报一个假边界。
        has_window = bool(window["start"] and window["end"])

        col1, col2 = st.columns(2)
        col1.metric("业务表数量", f"{stats['table_count']} 张")
        col2.metric("总数据量", f"{stats['total_rows']:,} 行")
        col1.metric("数据窗口", f"{window['days']} 天" if has_window else "未知")
        col2.metric("库文件体积", f"{stats['db_size_mb']} MB")

        # 数据窗口单独用一行文字写清楚 —— 这是用户提问前最该知道的一件事，
        # 藏在两个 metric 里反而看不清。
        if has_window:
            st.caption(
                f"可用数据范围：**{window['start']}** ~ **{window['end']}**"
                f"（日粒度，T+1 更新）"
            )
        else:
            st.caption("可用数据范围：**未知**（本数据集中未识别出日期列）")

        rows = [
            {"表名": item["name"], "行数": item["rows"], "说明": item["note"]}
            for item in stats["tables"]
        ]
        st.dataframe(rows, height=240, hide_index=True)


def _render_metric_catalog(ctx: DatasetContext) -> None:
    """指标口径卡片：让「口径」这件事在界面上可见。

    运营和数据团队吵架最多的地方就是口径。把指标的定义、参数、来源表
    全部摊开放在这里，等于把「指标字典」做成了产品的一部分，
    而不是一份没人看的 PDF。

    【为什么注册表按 ctx.scheme 加载而不是用全局默认？】
      指标方案已经和数据集解耦 —— 不同数据集可以指向不同的方案文件。
      写死默认路径的话，界面上会出现「指标字典显示的是 A 方案，实际查询用的是 B 方案」，
      而这两个东西对不上时，用户会照着错的口径去理解数字。
    """
    registry = get_metric_registry(str(ctx.scheme.registry_path))

    with st.expander("📖 指标口径字典", expanded=False):
        categories = ["全部"] + sorted({m.category for m in registry.all() if m.category})
        choice = st.selectbox("按分类筛选", categories, key="metric_category_filter")

        metrics = [
            m for m in registry.all() if choice == "全部" or m.category == choice
        ]
        st.caption(
            f"共 {len(metrics)} 个指标 · 方案：{ctx.scheme.display_name}"
            f"（`{ctx.scheme_id}`）"
        )

        for metric in metrics:
            with st.expander(f"{metric.metric_name}", expanded=False):
                _render_metric_detail(metric)


def _render_metric_detail(metric: Metric) -> None:
    """单个指标的口径明细。"""
    direction = "越高越好" if metric.good_direction == "up" else "越低越好"
    st.markdown(
        f"`{metric.metric_id}` · 单位 **{metric.unit or '-'}** · {direction} · "
        f"分类 {metric.category or '-'}"
    )
    st.markdown(metric.business_definition)

    if metric.params:
        st.markdown("**可用参数**")
        for param in metric.params:
            default = param.default if param.has_default else "必填"
            enum_text = f" · 取值范围 {'/'.join(str(v) for v in param.enum)}" if param.enum else ""
            st.markdown(
                f"- `{param.name}`（{param.type}，默认 {default}）"
                f"：{param.label or param.description}{enum_text}"
            )
    else:
        st.caption("该指标无需参数")

    st.caption(f"来源表：{', '.join(metric.source_tables) or '-'}")
    if metric.notes:
        st.info(metric.notes, icon="⚠️")


def _render_audit_log() -> None:
    """人工确认审计日志。

    为什么一个 Demo 要有审计日志？
      因为「重大决策需要人工确认」如果只停留在提示文案上，就是一句空话。
      只有把「谁、什么时候、确认了哪条结论」记下来，这个环节才真正存在。
      这也回答了上线前必答的一题：你的 Agent 上线后，出错了谁负责？
    """
    log: list[dict] = st.session_state.get("audit_log", [])

    with st.expander(f"✅ 人工确认审计日志（{len(log)} 条）", expanded=False):
        if not log:
            st.caption("暂无记录。当 Agent 给出重大决策类建议时，会要求人工复核并记录在此。")
            return
        for record in log:
            st.markdown(
                f"**{record['time']}** · 复核人 {record['reviewer']}  \n"
                f"问题：{record['question']}"
            )
            st.divider()


def _render_footer_actions(ctx: DatasetContext) -> None:
    """底部：模型状态与操作按钮。"""
    agent = get_agent(ctx.dataset_id, ctx.cache_key())
    client = agent.llm_client

    if getattr(client, "is_configured", False):
        names = [
            cfg.LLM_PROVIDERS.get(key, {}).get("display_name", key)
            for key in client.available_providers()
        ]
        st.success(f"模型已就绪：{' / '.join(names)}", icon="🤖")
    else:
        st.error(
            "未检测到大模型 API Key。请把 `.env.example` 复制为 `.env`，"
            "填入 DEEPSEEK_API_KEY 或 GLM_API_KEY 后重启。",
            icon="🔑",
        )

    total_tokens = sum(
        turn["answer"].usage.get("total_tokens", 0) for turn in st.session_state.turns
    )
    st.caption(
        f"本次会话：{len(st.session_state.turns)} 次提问 · 累计消耗 "
        f"{total_tokens:,} tokens"
    )

    if st.button("清空当前对话", width="stretch"):
        # 只清对话，不清审计日志 —— 审计记录一旦产生就不该被"一键清空"抹掉，
        # 这是审计类数据的基本要求。
        st.session_state.turns = []
        st.rerun()


# ===========================================================================
# 四、对话区：一次问答的完整渲染
# ===========================================================================

def render_query_block(turn_index: int, block_index: int, step: AgentStep) -> None:
    """渲染「数据区」：指标卡 / 图表 / 原始表格。

    这里体现的是数据产品的核心主张 —— **把图直接摆好，不让人选**。
    选图逻辑在 src/ui/charts.py，本函数只负责把选好的结果画出来。
    """
    data = step.result or {}
    if not data:
        return

    base_spec = pick_chart_spec(data)
    options = list(base_spec.y_options)

    # 数值列多于一个时才给切换器。
    # 为什么要留这个口子？因为「留存率」和「样本量」都值得看，
    # 但默认只该显示一个 —— 自动选好最关心的那个，同时保留手动切换的自由。
    selected_y = base_spec.y
    if len(options) > 1:
        selected_y = st.selectbox(
            "图表纵轴（可切换）",
            options,
            index=options.index(base_spec.y) if base_spec.y in options else 0,
            key=f"ycol_{turn_index}_{block_index}",
        )

    spec = pick_chart_spec(data, prefer_y=selected_y)

    if spec.kind == "metric":
        values = headline_values(data, spec)
        if values:
            columns = st.columns(min(len(values), 4))
            for position, (label, value) in enumerate(values):
                columns[position % len(columns)].metric(label, value)
    else:
        figure = build_figure(data, spec)
        if figure is not None:
            st.plotly_chart(figure, key=f"chart_{turn_index}_{block_index}")

    if spec.reason:
        st.caption(f"图表说明：{spec.reason}")

    row_count = data.get("row_count", 0)
    with st.expander(f"📋 原始结果表格（{row_count} 行）", expanded=False):
        st.dataframe(to_dataframe(data), hide_index=True)
        if data.get("truncated"):
            st.warning(
                f"结果已达单次查询上限（{cfg.SQL_MAX_ROWS} 行），"
                f"表格只展示了一部分，请缩小查询范围后再看完整数据。"
            )


def render_process_block(turn_index: int, answer: AgentAnswer) -> None:
    """渲染「过程区」：工具调用链路 + 展示版 SQL。

    这是可解释性的落点，也是演示时最有说服力的一块 ——
    它把「模型猜的」和「程序算的」区分得一清二楚：
    你能看到模型只传了 metric_id 和参数，SQL 是系统生成的。
    """
    with st.expander("🔍 执行过程（工具调用链路 / SQL）", expanded=False):
        if not answer.steps:
            st.caption("本次没有产生工具调用，模型直接作答。")
            return

        for step_index, step in enumerate(answer.steps, start=1):
            if step.kind == "final_answer":
                st.markdown(f"**第 {step.iteration} 轮 · 模型给出最终回答**")
                continue
            if step.kind == "guard":
                st.markdown(
                    f"**第 {step.iteration} 轮 · 🛡 来源校验拦截** "
                    f"（模型给出了无出处的数字，已要求重新取数）"
                )
                st.caption(f"被拦下的原文：{step.content or '（空）'}")
                continue

            icon = "✅" if step.ok else "❌"
            st.markdown(
                f"**第 {step.iteration} 轮 · `{step.tool_name}`** {icon} "
                f"· {step.elapsed_ms:.0f} ms"
            )
            st.code(json.dumps(step.arguments, ensure_ascii=False, indent=2), language="json")

            result = step.result or {}
            rendered_sql = result.get("rendered_sql")
            if rendered_sql:
                st.caption("系统生成的 SQL（展示版：参数以 :name 占位，取值见下方参数）")
                st.code(str(rendered_sql), language="sql")
                st.caption(f"参数取值：{json.dumps(result.get('params', {}), ensure_ascii=False)}")

            with st.expander("工具返回给模型的观察结果", expanded=False):
                observation = step.observation or ""
                limit = 2000
                st.text(
                    observation[:limit] + ("\n…（已截断）" if len(observation) > limit else "")
                )

        st.caption(
            f"共 {answer.iterations} 轮 · 模型 {answer.provider}/{answer.model} · "
            f"工具调用 {sum(1 for s in answer.steps if s.kind == 'tool_call')} 次"
        )


def render_confirmation_block(turn_index: int, answer: AgentAnswer) -> None:
    """渲染「人工确认节点」—— 项目硬约束第 4 条的前端落地。"""
    with st.container(border=True):
        st.warning(
            "这条结论属于**重大决策类分析**。按项目规范，涉及投放预算、版本调优、"
            "留存策略调整的建议**必须先经人工复核**才能进入执行环节："
            "模型负责给出可能性，决策责任必须留在人这一侧。",
            icon="⚠️",
        )
        st.caption(f"待确认结论：{answer.answer[:180]}{'…' if len(answer.answer) > 180 else ''}")

        col_left, col_right = st.columns([2, 3])
        reviewer = col_left.text_input(
            "复核人", key=f"reviewer_{turn_index}", placeholder="姓名或工号"
        )
        agreed = col_right.checkbox(
            "我已复核数据口径、样本量与混淆因素", key=f"agree_{turn_index}"
        )

        if st.button(
            "记录人工确认",
            key=f"confirm_{turn_index}",
            type="primary",
            disabled=not (agreed and reviewer.strip()),
        ):
            st.session_state.audit_log.insert(
                0,
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "reviewer": reviewer.strip(),
                    "question": answer.question,
                    "answer": answer.answer,
                },
            )
            st.success("已记录到左侧「人工确认审计日志」，该结论视为已完成人工复核。")

        if not agreed or not reviewer.strip():
            st.caption("填写复核人并勾选确认项后，才能记录本次人工确认。")


def render_turn(turn_index: int, turn: dict) -> None:
    """渲染一轮完整的问答（用户消息 + 助手消息）。"""
    question: str = turn["question"]
    answer: AgentAnswer = turn["answer"]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🎮"):
        # ---- 降级提示：这条回答来自备用模型时，先打一个牌再给结论 ----
        notice = fallback_notice(answer)
        if notice:
            st.warning(notice, icon="🔄")

        # ---- 结论 ----
        if answer.ok:
            st.markdown(answer.answer)
        else:
            st.error(answer.answer)

        # ---- 数据区：每个查数成功的工具调用都渲染一块 ----
        block_index = 0
        for step in answer.steps:
            if step.kind == "tool_call" and step.tool_name == "query_metric" and step.ok:
                render_query_block(turn_index, block_index, step)
                block_index += 1

        # ---- 运行信息 + 导出 ----
        usage = answer.usage or {}
        info_col, export_col = st.columns([3, 1])
        info_col.caption(
            f"模型 {answer.provider}/{answer.model} · {answer.iterations} 轮 · "
            f"{usage.get('total_tokens', 0):,} tokens · {answer.elapsed_ms:.0f} ms"
        )
        export_col.download_button(
            "导出问答",
            data=build_export_text(answer),
            file_name=f"分析结论_{datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain",
            key=f"export_{turn_index}",
        )

        # ---- 过程区 ----
        render_process_block(turn_index, answer)

        # ---- 人工确认节点（仅重大决策类触发）----
        if answer.ok and is_decision_analysis(answer):
            render_confirmation_block(turn_index, answer)


# ===========================================================================
# 五、主流程
# ===========================================================================

def main() -> None:
    # --- 会话状态初始化 ---
    # 这两个 key 用 setdefault 而不是 if not in：写法更短，语义也更清楚
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []

    # --- 第一步：数据集选择器 ---
    # 必须先渲染它，才知道这一轮要给用户看哪个数据集的数据。
    store = get_store()
    dataset_id = render_sidebar_head(store)

    # --- 第二步：组装运行时上下文 ---
    # 一个对象打包了「查哪个库 + 用哪套指标 + 日期窗口 + 表白名单」。
    # 下面所有渲染与 Agent 调用都只认它，不再各自去读全局 config。
    try:
        ctx = build_context(dataset_id, store=store)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        st.error(f"无法加载数据集：{exc}")
        st.stop()

    # --- 第三步：侧边栏其余部分 ---
    render_sidebar_tail(ctx)

    st.title("🎮 游戏数据智能分析师")
    # 当前数据集要写在主区而不是只放侧边栏：侧边栏可以收起，
    # 而「这份结论是哪份数据的结论」必须始终可见。
    st.caption(
        f"当前数据集：**{ctx.dataset.display_name}**（`{ctx.dataset_id}`）· "
        f"指标方案：{ctx.scheme.display_name} · "
        f"数据窗口：{ctx.data_start} ~ {ctx.data_end}"
    )
    st.caption(
        "用中文提问 → 自动映射到标准指标口径 → 生成并校验 SQL → 查真实数据 → "
        "给出结论与图表。所有数字均来自数据库，模型不写 SQL、也不编数字。"
    )

    # --- 空态：给几个示例问题，降低"不知道问什么"的门槛 ---
    if not st.session_state.turns:
        st.markdown("#### 试着问一个运营问题，或者直接点下面的例子：")
        example_columns = st.columns(2)
        for index, example in enumerate(_EXAMPLE_QUESTIONS):
            if example_columns[index % 2].button(
                example, key=f"example_{index}", width="stretch"
            ):
                # 不能在这里直接调 Agent —— 按钮的返回值只在本次重跑有效，
                # 而 Agent 调用一旦开始就得独占这一轮。所以先记下问题，触发重跑，
                # 由下面统一处理。这是 Streamlit 里非常常见的模式。
                st.session_state.pending_question = example
                st.rerun()

    # --- 历史对话 ---
    for turn_index, turn in enumerate(st.session_state.turns):
        render_turn(turn_index, turn)

    # --- 输入 ---
    pending = st.session_state.pop("pending_question", None)
    typed = st.chat_input("用中文提问，例如：最近 7 天的次留和付费率怎么样？")
    question = typed or pending

    if question:
        turn_index = len(st.session_state.turns)

        # 先把用户消息渲染出来，再显示 spinner。
        # 为什么？Streamlit 是同步执行的，如果先调 Agent 再渲染，
        # 用户会盯着一个空白页面等十几秒 —— 先给出即时反馈是基本的交互要求。
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🎮"):
            with st.spinner("正在分析：理解问题 → 匹配指标口径 → 查询数据…"):
                # 用 ctx 取 Agent：库路径、指标方案、数据窗口、白名单都来自它，
                # 因此不会出现「切了数据集但查询还打在旧库上」。
                agent = get_agent(ctx.dataset_id, ctx.cache_key())
                answer = agent.ask(question, history=build_history(st.session_state.turns))

        st.session_state.turns.append({"question": question, "answer": answer})
        # 重跑一次让这一轮走完整的渲染流程（图表、过程区、确认节点都要渲染），
        # 而不是把渲染逻辑在"首次回答"和"历史回放"两处各写一遍 ——
        # 两份代码迟早会不一致，这是界面 bug 的常见来源。
        st.rerun()


if __name__ == "__main__":
    main()