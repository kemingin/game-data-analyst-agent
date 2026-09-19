# -*- coding: utf-8 -*-
"""
前端可视化层（Phase 4）—— 「自动选图」策略与 Plotly 渲染
================================================================================
【这一层解决什么问题？】

  Agent 的工具返回里已经带了一份结构化数据（data["columns"] / data["rows"]），
  前端要做的事只有一件：**根据数据的形状，自动挑一个最合适的图**。

  用户不需要选图表类型 —— 这正是「数据分析产品」和「BI 工具」的关键差别：
    · BI 工具（Tableau / Superset）把人当专家，让人自己拖字段、选图型；
    · 分析产品把人当运营，替他做好决定，只让他看结论。
  我们的用户是运营同学，所以必须自动。

【为什么把「选图」和「画图」拆成两个函数？】

      pick_chart_spec()   纯逻辑：输入 dict → 输出一个 ChartSpec 描述
      build_figure()      碰 plotly：照着描述把图渲染出来

  好处很实在：选图策略**不依赖 streamlit / plotly**，可以被 pytest 直接覆盖。
  选错图是产品事故（比如把留存趋势画成柱状图），必须有测试兜住；
  而渲染层换个绘图库只改一个函数。这是很通用的手法：把「决策」和「执行」分开，
  决策可测，执行可换。

【选图规则（也是容易被追问的地方）】

  数据形状                          出的图        为什么
  ----------------------------------------------------------------------
  没有数据行                        不出图        画空图不如直接说"没查到"
  新手引导漏斗（多层步骤列）        漏斗图        分层递进的过程，漏斗能直接看出哪层流失最多
  含日期列 + 数值列                 折线图        时间序列看趋势，折线是默认正确解
  含分组维度 + 数值列 + 多行        柱状图        组间对比用柱，长短一眼可比
  只有一行数据                      指标卡        一个点的折线图毫无信息量，数字卡片更清楚
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# 一、常量：把「经验规则」显式写出来，而不是散落在 if 里
# ---------------------------------------------------------------------------

# 漏斗指标的分层列：顺序不能乱，漏斗图天然依赖「从宽到窄」的顺序
_FUNNEL_STEPS: tuple[tuple[str, str], ...] = (
    ("step0_registered", "注册"),
    ("step1_started", "开始新手引导"),
    ("step2_mid", "引导进行中"),
    ("step3_done", "完成引导"),
)

# 列名里出现这些词，就认为它是日期列（大小写不敏感）
_DATE_COL_HINTS: tuple[str, ...] = ("date", "日期", "stat_date", "dt")

# 日期值的形状，用于「列名看不出来、但值长得像日期」的兜底判断
_DATE_VALUE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")

# 百分比列后缀：本项目所有比率类指标都以 _pct 结尾（口径约定，见指标注册表）
_PCT_SUFFIX: str = "_pct"

# 挑「主数值列」时的优先级名单。
# 为什么需要它？因为一个查询往往返回多个数值列，比如留存率会同时返回
# cohort_size（样本量）、retained_users（人数）、retention_rate_pct（留存率）。
# 默认要画的是**业务上最关心的那个数**，也就是留存率本身，而不是样本量。
_PREFERRED_Y: tuple[str, ...] = (
    "dau", "new_users", "mau", "active_users",
    "arpu", "arppu", "ltv", "total_revenue",
    "retained_users", "payers",
)


# ---------------------------------------------------------------------------
# 二、选图结论的数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChartSpec:
    """一张图的「施工图纸」——只描述画什么，不含任何渲染代码。

    reason 字段是刻意保留的：它是「为什么选这张图」的一句话解释，
    会直接显示在前端图表下方。数据分析产品里，图的选型本身也是一种结论，
    把它讲出来，用户下次自己看数时也能复用这个判断。
    """

    kind: str                 # line / bar / funnel / metric / none
    title: str                # 图表标题
    x: str | None = None      # 横轴列名
    y: str | None = None      # 纵轴列名（主数值列）
    unit: str = ""            # 单位（% / 人 / 元）
    reason: str = ""          # 选图理由（给用户看的）
    y_options: tuple[str, ...] = ()          # 所有可切换的数值列
    funnel_steps: tuple[tuple[str, str], ...] = ()  # 仅漏斗图使用


# ---------------------------------------------------------------------------
# 三、基础工具函数
# ---------------------------------------------------------------------------

def to_dataframe(data: dict[str, Any] | None) -> pd.DataFrame:
    """把工具返回的 data 转成 DataFrame。

    为什么不用 pd.DataFrame(rows) 直接构造？
      因为 dict 的键顺序虽然 Python 3.7+ 保证有序，但一旦某行的键缺失
      （NULL 列被漏掉），列顺序就会错位。
      显式传 columns=[...] 能锁定列顺序，和 SQL 的 SELECT 顺序保持一致 ——
      这对「表格要和人看到的 SQL 对得上」很重要。
    """
    if not data:
        return pd.DataFrame()
    rows = data.get("rows") or []
    columns = data.get("columns") or []
    if not rows:
        return pd.DataFrame(columns=columns)
    if columns:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """找出「所有非空值都是数字」的列。

    为什么用 pd.to_numeric(errors="raise") 试转，而不是判断 dtype？
      因为 SQLite 返回的数值在某些情况下会是字符串（比如做了字符串拼接），
      用试转的方式能把「看起来是数字的字符串」也认出来，更鲁棒。
      任何一行转不动，就说明这列是文本，直接判定为非数值列。
    """
    result: list[str] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        try:
            pd.to_numeric(series)
        except (ValueError, TypeError):
            continue
        result.append(col)
    return result


def _is_date_like(name: str, series: pd.Series) -> bool:
    """判断一列是不是日期列：先看列名，再看值的长相。"""
    if any(hint in name.lower() for hint in _DATE_COL_HINTS):
        return True
    values = [str(v).strip() for v in series.dropna()]
    return bool(values) and all(_DATE_VALUE_RE.match(v) for v in values)


def _find_date_column(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> str | None:
    """找出第一个日期列（跳过纵轴列本身）。"""
    for col in df.columns:
        if col in exclude:
            continue
        if _is_date_like(col, df[col]):
            return col
    return None


def _find_category_column(
    df: pd.DataFrame, exclude: tuple[str, ...] = (), max_unique: int = 50
) -> str | None:
    """找出第一个「分组维度」列，比如渠道名、版本批次、留存步骤。

    两个约束：
      · max_unique：唯一值太多说明它是标识类字段（如 user_id），不是分组维度；
      · 至少要有 2 个不同取值，否则分组对比没有意义。
    """
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        unique = series.dropna().unique()
        if 2 <= len(unique) <= max_unique:
            return col
    return None


def _pick_y_column(
    nums: list[str], metric_id: str, prefer_y: str | None = None
) -> str | None:
    """挑主数值列，按优先级逐级降级。

    优先级顺序（每一级都是「比上一级更差但可接受」的退路）：
      1. 用户手动指定的列 —— 人的意图永远优先
      2. _pct 结尾的列    —— 比率类指标是运营最关心的结论
      3. 与 metric_id 同名的列 —— 模板里最直接的结果列
      4. 常用指标名名单   —— dau / arpu / ltv 这类
      5. 最后一个数值列   —— SQL 模板的惯例是「结果列写在最后」
    """
    if not nums:
        return None
    if prefer_y and prefer_y in nums:
        return prefer_y
    for col in nums:
        if col.endswith(_PCT_SUFFIX):
            return col
    if metric_id in nums:
        return metric_id
    for col in nums:
        if col in _PREFERRED_Y:
            return col
    return nums[-1]


def format_value(value: Any, unit: str = "") -> str:
    """把数字格式化成给人看的样子。

    规则：整数不带小数（465 人，而不是 465.00 人），小数保留两位（8.46 元）。
    这是数据产品的基本体面 —— 「465.00 人」这种写法会让运营觉得系统很粗糙。
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return f"{value}{unit}"
    if isinstance(value, (int, float)):
        number = float(value)
        text = f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}"
        return f"{text}{unit}"
    return f"{value}{unit}"


# ---------------------------------------------------------------------------
# 四、选图策略（本模块的核心，纯逻辑、可单测）
# ---------------------------------------------------------------------------

def pick_chart_spec(
    data: dict[str, Any] | None,
    prefer_y: str | None = None,
) -> ChartSpec:
    """根据查询结果自动决定「画什么图」。

    这个函数**不抛异常、永远有返回值** —— 最差也返回 kind="none"。
    原因：前端不能因为一个意外格式的数据就整页崩掉，那是最糟糕的产品体验。
    """
    df = to_dataframe(data)
    metric_id = str((data or {}).get("metric_id") or "")
    metric_name = str((data or {}).get("metric_name") or metric_id or "查询结果")
    unit = str((data or {}).get("unit") or "")

    # --- 规则 0：没有数据行，不出图 -------------------------------------
    if df.empty:
        return ChartSpec(
            kind="none",
            title=metric_name,
            unit=unit,
            reason="本次查询没有返回任何数据行，无法绘制图表。",
        )

    # --- 规则 1：漏斗类指标优先走漏斗图 ---------------------------------
    # 为什么放在最前面？因为漏斗数据的行数只有 1 行、列却有 8 列，
    # 如果不特判，会掉进「只有一行 → 指标卡」的规则里，结果只显示一个注册数，
    # 把最有价值的「逐层流失」信息全丢掉。
    if metric_id == "tutorial_funnel":
        steps = tuple((col, label) for col, label in _FUNNEL_STEPS if col in df.columns)
        if len(steps) >= 3:
            return ChartSpec(
                kind="funnel",
                title=f"{metric_name} · 分层流失",
                unit=unit,
                funnel_steps=steps,
                reason="新手引导是「注册 → 引导 → 完成」的分层递进过程，"
                       "漏斗图能让每一层的流失一眼可见。",
            )

    nums = numeric_columns(df)
    if not nums:
        return ChartSpec(
            kind="none",
            title=metric_name,
            unit=unit,
            reason="结果里没有可绘制的数值列，请直接查看下方结果表格。",
        )

    y = _pick_y_column(nums, metric_id, prefer_y)
    date_col = _find_date_column(df, exclude=(y,) if y else ())

    # --- 规则 2：有日期列 → 折线图 --------------------------------------
    if date_col:
        return ChartSpec(
            kind="line",
            title=f"{metric_name}趋势",
            x=date_col,
            y=y,
            unit=unit,
            y_options=tuple(nums),
            reason=f"「{date_col}」是日期列，随时间变化的量用折线图看趋势最直观。",
        )

    # --- 规则 3：有分组维度且多行 → 柱状图 ------------------------------
    cat_col = _find_category_column(df, exclude=(y,) if y else ())
    if cat_col:
        return ChartSpec(
            kind="bar",
            title=f"{metric_name} · 分组对比",
            x=cat_col,
            y=y,
            unit=unit,
            y_options=tuple(nums),
            reason=f"「{cat_col}」是分组维度，共 {len(df)} 组，"
                   f"柱状图比折线图更适合做组间横向对比。",
        )

    # --- 规则 4：只有一行 → 指标卡 --------------------------------------
    if len(df) == 1:
        return ChartSpec(
            kind="metric",
            title=metric_name,
            y=y,
            unit=unit,
            y_options=tuple(nums),
            reason="结果只有一行数据，直接渲染成指标卡比画一个「只有一个点」的图更清楚。",
        )

    # --- 兜底：结构不适配任何图型，交给表格 -----------------------------
    return ChartSpec(
        kind="none",
        title=metric_name,
        unit=unit,
        reason="当前结果的列结构与内置图型都不匹配，请查看下方结果表格。",
    )


# ---------------------------------------------------------------------------
# 五、Plotly 渲染
# ---------------------------------------------------------------------------

# 统一的配色：主色取品牌蓝，对比色取橙色。
# 为什么要定成常量？因为前端所有图共用一套色，视觉上才像一个产品，
# 而不是「三个库拼出来的三张图」。
_COLOR_PRIMARY = "#2563eb"
_COLOR_ACCENT = "#f59e0b"


def build_figure(data: dict[str, Any] | None, spec: ChartSpec):
    """按 ChartSpec 渲染 Plotly 图。

    返回 None 表示「这类 spec 不出图」（指标卡由 Streamlit 的 st.metric 渲染）。

    注意 import 放在函数内部：plotly 体积不小（导入要 1 秒左右），
    而本模块的选图逻辑在单元测试里会被反复调用 ——
    让 plotly 只在真正要画图时才加载，测试跑起来能快好几倍。
    """
    if spec.kind in ("none", "metric"):
        return None
    # 折线/柱状图必须有纵轴列；漏斗图的数值来自 funnel_steps，不需要 y。
    if spec.kind in ("line", "bar") and not spec.y:
        return None

    import plotly.graph_objects as go

    df = to_dataframe(data)
    unit = spec.unit or ""
    axis_title = f"{spec.y}（{unit}）" if unit else str(spec.y)

    # ---------- 漏斗图 ----------
    if spec.kind == "funnel":
        labels = [label for _, label in spec.funnel_steps]
        values = [float(df.iloc[0][col]) for col, _ in spec.funnel_steps]
        figure = go.Figure(
            go.Funnel(
                y=labels,
                x=values,
                textinfo="value+percent initial",
                marker={"color": _COLOR_PRIMARY},
                connector={"line": {"color": "#cbd5e1"}},
            )
        )
        figure.update_layout(title=spec.title, height=360, template="plotly_white")
        return figure

    # ---------- 折线图 ----------
    if spec.kind == "line":
        figure = go.Figure(
            go.Scatter(
                x=[str(v) for v in df[spec.x]],
                y=pd.to_numeric(df[spec.y]),
                mode="lines+markers",
                line={"color": _COLOR_PRIMARY, "width": 2.5},
                marker={"size": 7},
                # hovertemplate 让鼠标悬停直接显示「日期 + 数值 + 单位」，
                # 比默认的 "x=..., y=..." 对运营友好得多
                hovertemplate="%{x}<br><b>%{y}</b>" + unit + "<extra></extra>",
            )
        )
        figure.update_layout(
            title=spec.title,
            height=380,
            template="plotly_white",
            xaxis_title=spec.x,
            yaxis_title=axis_title,
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        )
        return figure

    # ---------- 柱状图 ----------
    if spec.kind == "bar":
        figure = go.Figure(
            go.Bar(
                x=[str(v) for v in df[spec.x]],
                y=pd.to_numeric(df[spec.y]),
                # 柱子顶端直接标数值：对比类图表「一眼看出差距」比「精确读数」更重要，
                # 所以把数字写在柱子上，省掉用户来回扫坐标轴的力气。
                text=[format_value(v, unit) for v in df[spec.y]],
                textposition="outside",
                marker={"color": _COLOR_PRIMARY},
                hovertemplate="%{x}<br><b>%{y}</b>" + unit + "<extra></extra>",
            )
        )
        figure.update_layout(
            title=spec.title,
            height=380,
            template="plotly_white",
            xaxis_title=spec.x,
            yaxis_title=axis_title,
            bargap=0.35,
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        )
        return figure

    return None


def headline_values(data: dict[str, Any] | None, spec: ChartSpec) -> list[tuple[str, str]]:
    """给指标卡准备「(标题, 值)」列表。

    为什么单独抽一个函数？因为「一行数据」在业务上往往包含多个值得看的数，
    比如留存率这一行同时有 cohort_size（样本量）、retained_users（回流人数）
    和 retention_rate_pct（留存率）。
    产品上的正确做法是：**把样本量一起摆出来**。
    这正是项目硬性规则第 3 条要求的「对比类结论必须同时报样本量」，
    在界面层再做一次机械保证 —— 提示词管模型，界面管兜底。
    """
    df = to_dataframe(data)
    if df.empty:
        return []

    labels = {
        "cohort_size": "样本量（新增用户）",
        "retained_users": "留存人数",
        "active_users": "活跃用户数",
        "payers": "付费用户数",
        "total_revenue": "总流水",
    }
    # 主指标排第一位，其余数值列按 SQL 的输出顺序跟随。
    # 顺序固定很重要：卡片位置每刷新一次就变，会让人以为是新数据。
    nums = [col for col in df.columns if col in spec.y_options]
    ordered = ([spec.y] if spec.y in nums else []) + [c for c in nums if c != spec.y]
    # 指标单位只属于主指标。样本量、人数这些列的单位是「个/人」，
    # 套上指标单位会写出「812%」这种错误表述 —— 单位错了，数字就不可信了。
    return [
        (
            labels.get(col, col),
            format_value(df.iloc[0][col], spec.unit if col == spec.y else ""),
        )
        for col in ordered
    ]