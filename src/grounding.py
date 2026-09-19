# -*- coding: utf-8 -*-
"""
答案数字溯源（Phase 9）
================================================================================
【这个模块为什么存在？】

  第五层防线（答案来源校验）原本只在 `react_agent.py` 里做一件很粗的事：
  「本轮有没有成功取过数 + 答案里有没有数据型数字」。它管不住更细的问题 ——
  **答案里每一个数字，能不能在真实查询结果里找到出处？**

  要做到「逐数字溯源」，需要三样能力：抽数字、比数字、收参考值。
  这三样在评测体系（`src/eval/metrics.py`）里**早就有了**，而且是被真机数据
  反复打磨过的（见那个模块头部关于「四档分层」的说明）。所以正确做法是复用，
  而不是在运行时另写一套 —— 否则两边口径必然漂移，出现
  「运行时拦了、评测说没问题」这种自相矛盾的结论。

【为什么不直接从 src/agent 里 import src.eval.metrics？】

  因为会形成**真实的循环导入**，不是风格问题：

      app.py  import src.agent.react_agent
              → react_agent  from src.eval.metrics import ...
                → 触发包初始化 src/eval/__init__.py
                  → 它 import src.eval.runner
                    → runner  from src.agent.react_agent import AgentAnswer
                      → 该模块已在 sys.modules 但只执行到一半，
                        AgentAnswer 尚未定义 → ImportError

  依赖方向本来就是 **eval → agent**（评测依赖被测系统，天经地义）。
  要复用就得把共用部分**下沉到一个谁都不属于的中立模块** —— 就是这个文件。

【分层约定】

  本模块只做「数字的抽取与比对」和「从执行轨迹里收数值」，
  不 import `src.agent`、不 import `src.eval`、不依赖 LLM、不依赖真值。
  读执行轨迹用的是**鸭子类型**（只读 kind / tool_name / ok / result 四个属性），
  既能喂 `AgentStep` 对象，也能喂它 `to_dict()` 之后的字典 ——
  与「指标语义层不 import 数据层」是同一个思路：靠约定而不是靠 import 耦合。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# ===========================================================================
# 一、数字抽取
# ===========================================================================

# 日期要先从文本里抠掉。
# 为什么？因为「2026-09-11」里的 2026 / 09 / 11 会被数字正则当成三个数据值，
# 结果是每一条回答都产生一堆对不上的数字，对账率被严重拉低。
# 这是写数字对账最容易踩的第一个坑。
_DATE_PATTERNS: tuple[str, ...] = (
    r"\d{4}-\d{1,2}-\d{1,2}",      # 2026-09-11
    # 「只有年-月」的写法（2025-12）必须单独列一条。
    # 为什么？因为漏了它，后面的「09-05」简写规则会从中间咬掉一段：
    #   「2025-12」被 \d{1,2}-\d{1,2} 匹配成「25-12」，剩下一个孤零零的「20」，
    #   而 20 会被当成数据值 → 一条**正确的超范围拒答**被判成「编造」而拦截。
    #   这是把 extract_numbers 接进运行时防线时，被拒答用例当场抓出来的。
    r"\d{4}-\d{1,2}",              # 2025-12
    r"\d{4}/\d{1,2}/\d{1,2}",      # 2026/09/11
    r"\d{4}/\d{1,2}",              # 2026/09（同上，斜杠写法）
    r"\d{4}年\d{1,2}月\d{1,2}日",   # 2026年9月11日
    r"\d{4}年\d{1,2}月",            # 2026年9月
    r"\d{1,2}月\d{1,2}日",          # 9月11日
    r"\d{1,2}/\d{1,2}",            # 9/11
    # 「09-05」这种省略年份的简写日期。**必须排在 4 位年份的规则之后**，
    # 否则会把 2026-09-11 拆成「2026」和「-09」两段。
    # 真实教训：真机评测里 E07 的答案用表格列日期写成 09-05，
    # 结果 -05 被当成一个数据值判成"存疑"，纯属误报。
    r"\d{1,2}-\d{1,2}",
)

# 版本号要先抠掉：v2.0.0 会被拆成 2.0 和 0 两个"数字"，纯属噪声。
_VERSION_PATTERN = re.compile(r"[vV]\d+(?:\.\d+)+")

# 数字本体：允许千分位逗号、正负号、小数
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

# 「非数据值」的前后缀词表。
# 「最近 7 天」「第 30 天」「步骤 3」「3 个月」里的数字是**描述性编号或时间量词**，
# 不是从数据库里读出来的值。抽进来只会制造假的对不上。
#
# 为什么前缀要用 endswith 判断而不是只看前一个字符？
# 因为中文里数字前面经常有空格：「卡在第 3 步」的前一个字符是空格不是「第」，
# 只看一个字符就会漏判 —— 这是真机评测 E13 暴露出来的问题。
_NON_DATA_SUFFIXES: frozenset[str] = frozenset("天日月周年周步个")
_NON_DATA_PREFIXES: tuple[str, ...] = ("第", "步骤", "D", "d")


@dataclass(frozen=True)
class ExtractedNumber:
    """从回答里抽出来的一个数字，保留原文便于人工核对。"""

    value: float
    raw: str


def extract_numbers(text: str | None) -> list[ExtractedNumber]:
    """从回答文本里抽出所有「数据类数字」。

    过滤规则（每一条都是为了减少误报，理由都写在上面常量处）：
      · 日期（含 09-05 这种简写）、版本号先整体移除；
      · 前面以「第 / 步骤 / D / d」结尾的跳过（第 7 天、步骤 3、D7）；
      · 后面紧跟天/日/月/周/年/步/个 的跳过（最近 7 天、30 天留存、3 个月）。
    """
    if not text:
        return []

    cleaned = text
    for pattern in _DATE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = _VERSION_PATTERN.sub(" ", cleaned)

    results: list[ExtractedNumber] = []
    for match in _NUMBER_PATTERN.finditer(cleaned):
        raw = match.group(0)

        # 前缀：先 rstrip 掉空白再看结尾，才能处理「第 3 步」这种带空格的写法
        before = cleaned[: match.start()].rstrip()
        if before.endswith(_NON_DATA_PREFIXES):
            continue

        # 后缀：同样先 lstrip 掉空白
        after_text = cleaned[match.end(): match.end() + 3].lstrip()
        if after_text and after_text[0] in _NON_DATA_SUFFIXES:
            continue

        try:
            value = float(raw.replace(",", ""))
        except ValueError:  # pragma: no cover - 正则已保证可转，兜底而已
            continue
        results.append(ExtractedNumber(value=value, raw=raw))

    return results


# ===========================================================================
# 二、数字比对
# ===========================================================================

# 容差的设计依据：
#   · 相对容差 0.5% —— 模型常把 42.66% 写成 42.7%，这是正常四舍五入；
#   · 绝对容差 0.05 —— 防止数值很小时相对容差失去意义（比如 0.1 的 0.5% 是 0.0005）。
# 两者取大，兼顾"允许舍入"和"能识别真正的错误"。
_REL_TOLERANCE: float = 0.005
_ABS_TOLERANCE: float = 0.05

# 宽松容差：给"口语化近似"留的余地。
# 为什么必须有这一档？因为真实的分析回答里，「稳定在 530 左右」这种表达非常常见，
# 它指的是 533 —— 这是人话，不是编造。用严格容差去卡它，只会制造大量假警报。
# 3% 的取值依据：既覆盖了"说个大概数"的习惯（±3% 以内），
# 又不至于宽到能把"完全不同量级"的数字放进来。
_LOOSE_TOLERANCE: float = 0.03


def is_close(value: float, candidate: float, tolerance: float = _REL_TOLERANCE) -> bool:
    """判断两个数是否算「一致」（允许舍入误差）。"""
    return abs(value - candidate) <= max(abs(candidate) * tolerance, _ABS_TOLERANCE)


def derived_values(values: Sequence[float]) -> list[float]:
    """列出所有「能由参考值算出来」的候选值。

    为什么需要这个？因为一个合格的分析回答必然会包含**衍生指标**：
    「7 天均值 517」「比低点回升 15%」「峰值与谷值差 79 人」——
    这些数字在原始数据里根本不存在，但它们完全正确。
    如果不把它们算作"有出处"，模型越会分析，被判"编造"的数字反而越多，
    这个指标就变成了「惩罚分析能力」，彻底失去意义。

    覆盖的衍生方式（都是运营分析里最常见的）：
      合计 / 均值 / 最大值 / 最小值 / 极差
      两两差值 / 两两比值 / 两两涨跌幅（%）
    """
    if not values:
        return []

    candidates: list[float] = [
        sum(values),
        sum(values) / len(values),
        max(values),
        min(values),
        max(values) - min(values),
    ]

    # 两两组合：为什么连"比值"和"涨跌幅"都要算？
    # 因为「比上周涨了 15%」这种表述在分析结论里几乎必然出现，
    # 而它就是 (a-b)/b*100 —— 不把它列进来，几乎每条回答都会被判存疑。
    #
    # 为什么涨跌幅要算**两个方向**？因为分母的选取是行业惯例问题，两种都常见：
    #   · (a-b)/b —— 「比上周涨了 15%」，基准是旧值；
    #   · (a-b)/a —— 「这一步流失了 41%」，基准是前一步的量（漏斗单步流失率）。
    # 真机评测 E13 里模型写「第 3 步单步就掉了 41.32%」，用的就是后者，
    # 只算一个方向的话这条完全正确的分析反而会被判成存疑。
    for a in values:
        for b in values:
            if b == 0:
                continue
            candidates.append(a - b)
            candidates.append(a / b)
            candidates.append((a - b) / b * 100)
            if a != 0:
                candidates.append((a - b) / a * 100)

    return candidates


def collect_reference_values(
    rows: Sequence[dict[str, Any]] | None = None,
    extra: Iterable[Any] = (),
) -> list[float]:
    """把参考数据里所有数值收集成一个集合（用于对账）。

    为什么要连 extra 一起收？因为像「最近 7 天」的 7、样本量、返回行数这些
    也都是回答里会合法出现的数字。把它们纳入参考集，能显著减少假警报 ——
    对账这件事，宁可漏报也不要天天误报，误报多了就没人看了。
    """
    values: list[float] = []
    for row in rows or []:
        for cell in row.values():
            if isinstance(cell, bool) or cell is None:
                continue
            try:
                values.append(float(cell))
            except (TypeError, ValueError):
                continue
    for item in extra:
        if item is None or isinstance(item, bool):
            continue
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue
    return values


@dataclass
class NumberCheck:
    """一次「答案数字对账」的结果。

    四个分类桶按可信程度递减排列，只有 unexplained 需要人看。
    """

    total: int = 0
    matched: list[tuple[float, float]] = field(default_factory=list)       # 严格命中
    approximate: list[tuple[float, float]] = field(default_factory=list)   # 口语近似
    derived: list[tuple[float, float]] = field(default_factory=list)       # 可由参考值推导
    unexplained: list[ExtractedNumber] = field(default_factory=list)       # 需人工复核

    @property
    def matched_count(self) -> int:
        """「有出处」的数字总数（三种可信分类之和）。"""
        return len(self.matched) + len(self.approximate) + len(self.derived)

    @property
    def traceable_count(self) -> int:
        """同 matched_count。语义更清楚的别名，报告里用它。"""
        return self.matched_count

    @property
    def rate(self) -> float | None:
        """数字可追溯率。没有数字时返回 None（不适用，而不是 0%）。"""
        if self.total == 0:
            return None
        return round(self.matched_count / self.total, 4)

    def describe_unexplained(self) -> str:
        """把存疑数字拼成一行，供报告展示。"""
        return "、".join(item.raw for item in self.unexplained) or "-"


def check_answer_numbers(
    answer_text: str | None,
    reference_values: Sequence[float],
    extra_allowed: Iterable[Any] = (),
) -> NumberCheck:
    """把回答里的数字逐个拿去参考数据里找出处。

    三级判定顺序（严格 → 近似 → 推导），先命中先归类：
      1. 严格容差命中参考值        → matched
      2. 宽松容差命中参考值        → approximate（口语近似）
      3. 命中由参考值推导出的候选值 → derived（均值/差值/涨幅…）
      4. 都不命中                  → unexplained（**唯一需要人工看的**）

    为什么要按这个顺序？因为越靠前的判定越"硬"。如果先判推导，
    一个恰好等于某对数值之差的编造数字就会被归成"可推导"，把真问题藏起来。
    """
    allowed = list(reference_values) + collect_reference_values(extra=extra_allowed)
    numbers = extract_numbers(answer_text)

    check = NumberCheck(total=len(numbers))
    if not numbers:
        return check

    # 衍生候选值只在需要时算一次，避免每个数字都重算一遍
    derived_pool: list[float] | None = None

    for item in numbers:
        strict_hit = next((c for c in allowed if is_close(item.value, c)), None)
        if strict_hit is not None:
            check.matched.append((item.value, strict_hit))
            continue

        loose_hit = next(
            (c for c in allowed if is_close(item.value, c, tolerance=_LOOSE_TOLERANCE)), None
        )
        if loose_hit is not None:
            check.approximate.append((item.value, loose_hit))
            continue

        if derived_pool is None:
            derived_pool = derived_values(allowed)
        derived_hit = next(
            (c for c in derived_pool if is_close(item.value, c, tolerance=_LOOSE_TOLERANCE)), None
        )
        if derived_hit is not None:
            check.derived.append((item.value, derived_hit))
            continue

        check.unexplained.append(item)

    return check


# ===========================================================================
# 三、从执行轨迹里收集「本次会话真实取到过的数值」
# ===========================================================================

def _step_field(step: Any, name: str, default: Any = None) -> Any:
    """同时支持 AgentStep 对象和它的 to_dict() 结果。

    为什么要兼容两种形态？因为既要能对「刚跑完的活对象」判定，
    也要能对「存盘后重新读出来的 JSON」判定（离线复算历史会话）。
    """
    if isinstance(step, dict):
        return step.get(name, default)
    return getattr(step, name, default)


def _iter_grounded_query_steps(steps: Sequence[Any] | None):
    """筛出「真的取到数」的 query_metric 步骤。

    判据是 `ok` 而不是「调用过」—— 模型可能调了工具但参数非法被拦下，
    那不算有数据支撑，这种情况防线依然应该生效。
    """
    for step in steps or []:
        if _step_field(step, "kind") != "tool_call":
            continue
        if _step_field(step, "tool_name") != "query_metric":
            continue
        if not _step_field(step, "ok", False):
            continue
        yield step


def collect_ground_truth_values(steps: Sequence[Any] | None) -> list[float]:
    """收集本次会话「真实取到过的数值」，作为数字溯源的参考集。

    这就是「逐数字溯源」的参考基准：答案里的数字只要能在这个集合里
    找到出处（严格 / 近似 / 可推导），就算有据可依。
    """
    values: list[float] = []
    for step in _iter_grounded_query_steps(steps):
        result = _step_field(step, "result") or {}
        if not isinstance(result, dict):
            continue
        values.extend(collect_reference_values(result.get("rows")))
    return values


def collect_ground_truth_params(steps: Sequence[Any] | None) -> list[float]:
    """收集成功查询用过的**数值型参数**（如 day_n=7）。

    为什么这些也要算「有出处」？因为回答里说「最近 7 天」是正常的 ——
    这个 7 来自参数而非结果集。不把它放进允许集，会制造大量假警报。
    """
    values: list[float] = []
    for step in _iter_grounded_query_steps(steps):
        result = _step_field(step, "result") or {}
        if not isinstance(result, dict):
            continue
        params = result.get("params") or {}
        if not isinstance(params, dict):
            continue
        for value in params.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values.append(float(value))
    return values


# ===========================================================================
# 四、分级处置：存疑数字占比决定「放行 / 标注 / 拦截」
# ===========================================================================

# 存疑数字占比的上限。超过它，整条回答判定为「疑似编造」。
#
# 【为什么是 1/3 而不是「有任何一个存疑就拦」？】
#   因为 unexplained 不完全等于编造。它还包括模型合理但我们的推导池没覆盖到的
#   表达（比如对多个值做了加权、取了中位数）。一律拦截会把「一个四舍五入的数字
#   废掉一整段正确推理」，过度拦截的代价是用户再也不信这个提示。
#   而反过来，一条回答里超过三分之一的数字都找不到任何出处，
#   那就不是「表达方式的差异」能解释的了 —— 这是大面积编造。
#
# 【为什么写成常量而不是配置项？】
#   固定阈值、不引入自适应校准：可复现、可解释，也与全项目「阈值写死在常量里」
#   的心智一致。自适应阈值会让同一份输入在不同环境下得出不同结论，无法审计。
UNGROUNDED_RATIO_LIMIT: float = 1 / 3


def unexplained_ratio(check: NumberCheck) -> float:
    """存疑数字占比。没有数字时返回 0.0（无数字 = 无风险，不是 100% 存疑）。"""
    if check.total <= 0:
        return 0.0
    return len(check.unexplained) / check.total


def grounding_decision(check: NumberCheck) -> str:
    """把一次对账结果映射成三档处置：pass / annotate / block。

      pass     —— 没有数字，或全部有出处；
      annotate —— 有少量存疑（占比不超过阈值）：回答照常给，但在末尾提示；
      block    —— 存疑占比超过阈值：判定疑似编造，不给用户看这些数字。

    【向后兼容性】一次成功取数都没有时，参考集为空 → 所有数字都存疑 →
    ratio = 1.0 → 走 block，与改造前「无成功取数 + 有数字 → 拦截」完全一致。
    换句话说，新判据是旧判据的**严格超集**：旧逻辑能拦的它都拦，
    旧逻辑放过的「有查询但混着编造数字」它也能拦。
    """
    ratio = unexplained_ratio(check)
    if check.total == 0 or ratio == 0.0:
        return "pass"
    if ratio <= UNGROUNDED_RATIO_LIMIT:
        return "annotate"
    return "block"
