# -*- coding: utf-8 -*-
"""
评测集加载器（Phase 5）
================================================================================
【评测集为什么也要单独做成一个 JSON 文件？】

  和指标注册表是同一个理由：**评测标准应该是可评审的数据，而不是代码里的常量。**
  谁来定义「什么算答对」？这是业务问题，不该由工程师在 Python 里拍板。
  放在 JSON 里，产品/运营可以直接看、直接改、直接补用例，
  而"改评测集"和"改评测算法"是两件完全不同的事，混在一起会让评测结果没法解释
  （分数变了，是模型变好了还是尺子变了？）。

【评测集里刻意不写「期望数值」】

  因为数值会随数据更新而失效。如果用例里写死「D1 应该是 42.66%」，
  明天数据一更新，整套评测集体性失败，人就会开始不信任评测。
  所以用例只声明**期望的指标 + 期望的参数**，具体数值由 runner 独立查询得到。
  这样评测集是稳定的，而尺子是随数据自适应的 —— 这才是一个能长期跑的评测。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.sqlgen.generator import SQLGenerator

DEFAULT_EVAL_SET_PATH: Path = Path(__file__).resolve().with_name("eval_set.json")


@dataclass(frozen=True)
class EvalCase:
    """一条评测用例。"""

    case_id: str
    question: str
    category: str = ""
    notes: str = ""

    # 期望模型查哪个指标。为 None 表示「不校验映射」（比如模糊提问）
    expected_metric_id: str | None = None

    # 期望的参数（已解析成真实取值）。只校验这里出现的键
    expected_params: dict[str, Any] = field(default_factory=dict)

    # 参考数据怎么算：默认和期望指标/参数一致。
    # 单独留出这两个字段，是为了支持「模型该查 A，但要用 B 的数据来对账」这种用例。
    reference_metric_id: str | None = None
    reference_params: dict[str, Any] = field(default_factory=dict)

    # 超范围用例：期望模型不查数就拒答
    expect_no_query: bool = False

    # 多轮追问场景的「前情对话」（Phase 6 新增），OpenAI 格式的 user/assistant 消息。
    #
    # 【为什么前情要写死在 JSON 里，而不是「先真跑第一轮、把结果喂给第二轮」？】
    #   链式两轮看起来很真实，但它让评测变得**不可复现**：
    #   第二轮考什么，取决于第一轮模型当时答了什么。今天第一轮答得好，
    #   第二轮就简单；明天第一轮答岔了，第二轮跟着崩 —— 分数开始飘，
    #   人就再也不信这套评测了。
    #   写死前情的好处是：每一轮追问考察的能力被**精确锁定**，
    #   换模型、换供应商、改提示词，跑出来的差异都能归因到追问本身。
    #   代价是前情不像真机那么自然 —— 但评测要的是稳定和可解释，不是逼真。
    history: tuple[dict[str, str], ...] = ()

    def resolved_reference_metric_id(self) -> str | None:
        return self.reference_metric_id or self.expected_metric_id


class EvalSet:
    """评测集：负责加载 + 用 Phase 2 的日期解析器把期望参数解析成真实取值。"""

    def __init__(
        self,
        path: str | Path | None = None,
        generator: SQLGenerator | None = None,
    ) -> None:
        self.path: Path = Path(path) if path else DEFAULT_EVAL_SET_PATH
        # 复用 Phase 2 的生成器来解析期望参数，而不是自己再写一套日期逻辑。
        # 这一点很关键：如果评测自己实现一套解析规则，就会出现
        # 「系统按规则 A 解析、评测按规则 B 校验」的错位，分数变得没有意义。
        self.generator: SQLGenerator = generator or SQLGenerator()
        self.meta: dict[str, Any] = {}
        self._cases: list[EvalCase] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"评测集文件不存在：{self.path}")

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.meta = raw.get("meta", {})

        for item in raw.get("cases", []):
            self._cases.append(self._build_case(item))

        seen: set[str] = set()
        for case in self._cases:
            if case.case_id in seen:
                raise ValueError(f"评测集里存在重复的 case_id：{case.case_id}")
            seen.add(case.case_id)

    def _build_case(self, item: dict[str, Any]) -> EvalCase:
        """构造一条用例，顺便把期望参数解析成真实取值。

        解析放在加载期做，是为了**让评测集本身也接受校验**：
        如果某条用例的参数写错了（比如 start_date 超出数据窗口），
        加载时就会直接报错，而不是等跑完 20 条真机评测才发现有一条是废的。
        早失败、早发现，是评测系统该有的品质。
        """
        expected_metric_id = item.get("expected_metric_id") or None
        expected_params = self._resolve_params(expected_metric_id, item.get("expected_params") or {})

        reference_metric_id = item.get("reference_metric_id") or None
        declared_reference_params = item.get("reference_params")

        if declared_reference_params:
            # 用例显式声明了参考参数：按声明的来
            reference_params = self._resolve_params(
                reference_metric_id or expected_metric_id, declared_reference_params
            )
        elif reference_metric_id:
            # 参考指标和期望指标不同、又没声明参考参数：只能走该指标的默认参数
            reference_params = {}
        else:
            # 最常见的情况：参考就是期望本身。
            # 必须继承 expected_params —— 否则「最近两周」这类用例的真值会用
            # 默认的 7 天窗口去算，模型答对了反而被判"数字对不上"。
            # 这是评测里最隐蔽的一类错误：尺子量错了，还怪被测对象。
            reference_params = expected_params

        return EvalCase(
            case_id=str(item["case_id"]),
            question=str(item["question"]),
            category=str(item.get("category", "")),
            notes=str(item.get("notes", "")),
            expected_metric_id=expected_metric_id,
            expected_params=expected_params,
            reference_metric_id=reference_metric_id,
            reference_params=reference_params,
            expect_no_query=bool(item.get("expect_no_query", False)),
            history=tuple(
                {"role": str(m["role"]), "content": str(m["content"])}
                for m in item.get("history", [])
            ),
        )

    def _resolve_params(self, metric_id: str | None, params: dict[str, Any]) -> dict[str, Any]:
        """用指标模板把参数（含 @data_end-13 这类表达式）解析成真实取值。

        做法是直接调 generator.generate()：它会做类型转换、默认值填充、
        日期表达式解析、区间校验。我们只要最终结果（GeneratedSQL.params），
        SQL 本身不用。

        注意「期望参数为空」时不要走这条路径 —— 因为 generate 会把**所有**默认值
        都填进来，那就等于把默认值也纳入了校验，反而会惩罚"模型不传默认值"这个正确行为。
        所以空参数直接返回空字典，check_params 会因为 expected_params 为空而跳过校验。
        """
        if not metric_id or not params:
            return {}
        generated = self.generator.generate(metric_id, params)
        # 只保留用例显式声明的键，避免把默认值一起带进来
        return {key: generated.params[key] for key in params if key in generated.params}

    # ---------------- 访问 ----------------
    def all(self) -> list[EvalCase]:
        return list(self._cases)

    def get(self, case_id: str) -> EvalCase:
        for case in self._cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"评测集里没有这条用例：{case_id}")

    def by_category(self) -> dict[str, int]:
        """按分类统计用例数，方便报告里展示覆盖度。"""
        result: dict[str, int] = {}
        for case in self._cases:
            result[case.category] = result.get(case.category, 0) + 1
        return result

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self):
        return iter(self._cases)


def load_eval_set(path: str | Path | None = None) -> EvalSet:
    """便捷入口。"""
    return EvalSet(path)