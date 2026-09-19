# -*- coding: utf-8 -*-
"""指标语义层（Phase 2）。

对外只暴露三样东西，保持包的最小接口：
    MetricRegistry / Metric / MetricParam  —— 数据结构
    get_registry()                         —— 取注册表（带缓存）
"""

from src.metrics.registry import (
    DEFAULT_REGISTRY_PATH,
    Metric,
    MetricParam,
    MetricRegistry,
    get_registry,
)

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "Metric",
    "MetricParam",
    "MetricRegistry",
    "get_registry",
]