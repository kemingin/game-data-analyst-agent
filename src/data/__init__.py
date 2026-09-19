# -*- coding: utf-8 -*-
"""
数据层：真实数据加载、模拟数据生成、数据库访问。

    dataset.py   数据集 / 指标方案 / 索引管理（多数据集架构的地基）
    upload.py    CSV 上传建表（解码 → 类型推断 → 建表导数）
    schema.sql   内置数据集的表结构定义
    steam_data.py / build_database.py  内置数据集的加载与建库

【为什么这里只导出 dataset 的符号，不导出 upload？】
  因为 upload.py 会 import pandas，而本包被大量脚本以 `import src.data` 形式引用。
  把 pandas 的加载推迟到真正调用上传功能时（dataset.add_from_upload 里是延迟 import），
  可以让「只是读一下数据集列表」这种轻量操作不必付出加载 pandas 的代价。
"""

from src.data.dataset import Dataset, DatasetStore, DatasetTable, MetricScheme

__all__ = [
    "Dataset",
    "DatasetStore",
    "DatasetTable",
    "MetricScheme",
]