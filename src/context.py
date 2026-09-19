# -*- coding: utf-8 -*-
"""
数据集运行时上下文（组装根）
=============================
把「查哪个库 + 用哪套指标 + 日期窗口 + 表白名单」打包成一个自洽的整体。

【为什么需要这个对象？—— 它解决的是「静默算错」】
    改造前，这四个值分散在四个全局位置：
        registry   → get_registry()          （全局单例）
        db_path    → cfg.DB_PATH             （全局常量）
        日期窗口   → cfg.DATA_START/END       （全局常量）
        表白名单   → validator.ALLOWED_TABLES（全局硬编码）

    多数据集下它们必须一起变。如果不打包，就要在 5 个注入点手工接线
    （generator / executor / validator / prompt / agent），共 5×4=20 处。
    而**接错不会报错，只会算错**：

        用 A 方案的 SQL 模板 + 连 B 数据集的库
        → 若两个库恰好都有 dim_user 表，查询会成功
        → 返回一个语义完全错误的数字
        → 过程区显示的 SQL 看起来完全正常

    这是最坏的一类 bug：不崩、不报错、数字是错的，而且可复现性极差。
    打包成上下文后，「配错」这件事在类型层面就被消解了 ——
    要么拿到一整套自洽的上下文，要么拿不到。

【为什么放在 src/ 根目录，而不是 src/data/？】
    因为它要 import SQLGenerator / QueryExecutor / SQLValidator / MetricRegistry，
    而项目声明的分层是「数据层 → 指标语义层 → SQL生成与校验 → Agent层」。
    放进 src/data/ 会倒置依赖。
    与 config.py 并列则是天然正确的：config.py 管静态配置，context.py 管运行时装配，
    它位于所有层之上，可以自由 import 下层，而没有任何层 import 它
    （tools.py / react_agent.py 只在 TYPE_CHECKING 下引用类型）。

【为什么是 frozen 的？】
    它是一次请求的配置快照。如果可变，下游某一层（比如某个工具）改了
    allowed_tables，会静默影响同一上下文里的其他调用者 ——
    而配置类的意外污染正是最难复现的 bug。
    冻结后，改配置只能走「换一个新上下文」，路径唯一、行为可预期。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src import config as cfg
from src.agent.prompts import build_system_prompt
from src.data.dataset import Dataset, DatasetStore, MetricScheme
from src.metrics.registry import MetricRegistry, get_registry
from src.sqlgen.executor import QueryExecutor
from src.sqlgen.generator import SQLGenerator
from src.sqlgen.validator import SQLValidator


@dataclass(frozen=True)
class DatasetContext:
    """一次查询所需的全部配置，打包成一个不可变整体。"""

    dataset: Dataset
    scheme: MetricScheme
    registry: MetricRegistry
    db_path: Path
    data_start: date
    data_end: date
    allowed_tables: frozenset[str]
    # ↑ 本数据集实际存在的表，取代改造前全局硬编码的 ALLOWED_TABLES

    # ---------------- 便捷属性 ----------------
    @property
    def dataset_id(self) -> str:
        return self.dataset.dataset_id

    @property
    def scheme_id(self) -> str:
        return self.scheme.scheme_id

    def cache_key(self) -> str:
        """缓存键，供 Streamlit 的 cache_resource 区分不同数据集的 Agent 实例。"""
        return self.dataset.cache_key()

    # ---------------- 工厂：把「接线」收敛到一处 ----------------
    def build_generator(self) -> SQLGenerator:
        """用本上下文的 registry + 数据窗口造 SQL 生成器。

        【为什么不直接用 SQLGenerator() 的默认值？】
          因为它的默认值是「全局 config 的数据窗口」，而上传数据集的数据窗口
          来自 CSV 里的真实日期。用错窗口的后果是：提示词告诉模型
          「你的数据到 2026-09-17」，而实际数据到 2025-03-01，
          模型于是自信地给出一个超出数据范围的结论。
        """
        return SQLGenerator(
            registry=self.registry,
            data_start=self.data_start,
            data_end=self.data_end,
        )

    def build_validator(self, max_rows: int | None = None) -> SQLValidator:
        """用本数据集的实际表清单造校验器。

        【这一行是「数据集隔离」真正生效的地方】
          改造前的语义是「指标声明的来源表不能超出全库白名单」；
          多数据集下「全库白名单」这个概念本身就不存在了，
          基准必须是本数据集实际有的表。
          新语义更严：指标声明的来源表必须是本数据集里真实存在的表。
        """
        return SQLValidator(max_rows=max_rows, allowed_tables=self.allowed_tables)

    def build_executor(
        self,
        max_rows: int | None = None,
        timeout_seconds: float | None = None,
    ) -> QueryExecutor:
        """造查询执行器（连本数据集的库）。

        【为什么 max_rows 要同时喂给 validator 和 executor？】
          validator 用它来补齐 / 收紧 SQL 里的 LIMIT，executor 用它做 fetchmany 的上限。
          两者不一致会出现「SQL 里写着 LIMIT 500，但只 fetch 了 100 行」这种
          静默丢数据 —— 图表画出来是断的，但没人会怀疑到这里。
        """
        rows = cfg.SQL_MAX_ROWS if max_rows is None else max_rows
        return QueryExecutor(
            db_path=self.db_path,
            max_rows=rows,
            timeout_seconds=timeout_seconds,
            validator=self.build_validator(max_rows=rows),
        )

    def build_system_prompt(self) -> str:
        """按本数据集的数据窗口生成提示词。"""
        return build_system_prompt(
            registry=self.registry,
            data_start=self.data_start,
            data_end=self.data_end,
        )

    # ---------------- 展示用 ----------------
    def describe(self) -> str:
        """一句话描述，用于日志与前端副标题。"""
        return (
            f"{self.dataset.display_name}（{self.dataset_id}）"
            f" · 方案：{self.scheme.display_name}"
            f" · 数据窗口：{self.data_start} ~ {self.data_end}"
            f" · 表 {len(self.allowed_tables)} 张"
        )


def build_context(
    dataset_id: str,
    store: DatasetStore | None = None,
    data_start: date | None = None,
    data_end: date | None = None,
) -> DatasetContext:
    """从 store 取出 dataset + scheme，组装出上下文。

    【日期窗口的优先级链】
        显式传入 > dataset.data_start/data_end（上传时从 CSV 识别）
                 > cfg.DATA_START/DATA_END
      内置数据集在索引里记的就是 cfg 的值，所以走哪一档结果都一样 ——
      这保证了改造前后行为逐字节一致。

    异常：
        KeyError —— 数据集或方案不存在
        FileNotFoundError / ValueError —— 指标注册表文件缺失或格式非法
    """
    store = store or DatasetStore()
    dataset = store.get(dataset_id)
    scheme = store.get_scheme(dataset.scheme_id)

    # get_registry 自带 lru_cache(maxsize=4) 且 path 参与缓存键，
    # 所以多个方案天然共存，不需要我们自己做缓存。
    registry = get_registry(str(scheme.registry_path))

    return DatasetContext(
        dataset=dataset,
        scheme=scheme,
        registry=registry,
        db_path=dataset.db_path,
        data_start=data_start or dataset.data_start or cfg.DATA_START,
        data_end=data_end or dataset.data_end or cfg.DATA_END,
        allowed_tables=dataset.table_names,
    )
