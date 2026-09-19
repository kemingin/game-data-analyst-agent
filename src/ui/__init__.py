# -*- coding: utf-8 -*-
"""
前端层（Phase 4）
================================================================================
只放「与界面展示相关、但和 Streamlit 组件解耦」的纯逻辑：

    charts.py     选图策略 + Plotly 渲染
    overview.py   侧边栏数据概览（只读查库取元信息）

为什么要把这些从 app.py 里拆出来？
  app.py 是 Streamlit 的入口脚本，它有一个很特殊的性质：
  **每点一次按钮就从头到尾重跑一遍**。如果把所有逻辑都堆在里面，
  任何一次改动都要重新启动服务、点开页面、手动复现才能验证。
  拆出来之后，纯函数部分可以像 Phase 2/3 一样直接跑 pytest 验证。
"""

from src.ui.charts import (
    ChartSpec,
    build_figure,
    format_value,
    headline_values,
    pick_chart_spec,
    to_dataframe,
)
from src.ui.overview import load_table_stats

__all__ = [
    "ChartSpec",
    "build_figure",
    "format_value",
    "headline_values",
    "pick_chart_spec",
    "to_dataframe",
    "load_table_stats",
]