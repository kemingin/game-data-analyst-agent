# -*- coding: utf-8 -*-
"""
评测核心算法（Phase 5）
================================================================================
【为什么要单独做「评测体系」？】

  前面四个 Phase 证明了「这个系统能跑通」，但跑通 ≠ 可信。一定会被问：
  「你怎么知道它答得对？」如果只能回答「我试了几次都对」，这个项目就停在
  Demo 层面了。评测体系要回答的是**可量化的三个问题**：

      1. 模型有没有选对指标？        → 指标映射准确率
      2. 模型有没有把参数抽对？      → 参数抽取准确率
      3. 模型说的数字能不能对上数据？→ 答案数值一致率

  第三个最关键 —— 它是「零编造」这个核心主张的**机器化证明**。
  靠人眼看「它好像没瞎说」是不可复现的，必须用程序逐字对账。

【本模块为什么全是纯函数？】

  因为评测算法本身也需要被测试。把「怎么算分」和「怎么跑 Agent」拆开之后：
      · 算法部分可以喂构造数据，离线、快速、零成本地验证；
      · 跑 Agent 的部分（runner.py）真机执行时，只需要相信算法是对的。
  如果两者混在一起，评测代码就只能靠「跑一次真机看看结果」来验证 ——
  那等于没有测试，评测系统本身反而成了最不可信的一环。

【一条重要的设计立场：机器做初筛，人工做终审】

  「答案里的数字对不上参考值」有好几种可能：
      · 编造（严重问题）；
      · 口语化近似（把 533 说成「530 左右」）；
      · 合理衍生（自己算了 7 天均值 517、算了个涨幅 15%）。
  程序没法直接区分前两者和第三者。所以本模块不把"对不上"直接定性为编造，
  而是把每个数字分成四类，只把最后一类交给人：

      严格匹配 → 精确命中参考值（容差 0.5%，只允许四舍五入）
      合理近似 → 落在宽松容差内（3%），属于正常口语表达
      可推导   → 能由参考值算出来（合计 / 均值 / 极差 / 两两差值 / 比率 / 涨幅）
      存疑     → 以上都不是 ← **只有这一类需要人看**

  ★ 这条分层是**被真实数据逼出来的**，不是一开始就设计好的：
    第一版只有"对上/对不上"两档，结果拿最简单的「最近 7 天日活」去跑，
    14 个数字里 3 个判为对不上，一致率只有 78.6%。
    人工一看，那 3 个分别是「7 天均值 517」「回升约 15%」「稳定在 530 左右」——
    全都是正确的分析表达。如果放任这个指标这么报，看到的人会以为模型有 21% 在编，
    实际上它一个数都没编。
    评测指标一旦开始误报，就没人再信它了 —— 这比没有指标更糟。
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
    r"\d{4}/\d{1,2}/\d{1,2}",      # 2026/09/11
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
# 三、从执行轨迹里提取「模型做了什么」
# ===========================================================================

def _step_field(step: Any, name: str, default: Any = None) -> Any:
    """同时支持 AgentStep 对象和它的 to_dict() 结果。

    为什么要兼容两种形态？因为评测既要能对「刚跑完的活对象」评分，
    也要能对「存盘后重新读出来的 JSON」评分（离线复算历史评测结果）。
    """
    if isinstance(step, dict):
        return step.get(name, default)
    return getattr(step, name, default)


def metric_calls(steps: Sequence[Any] | None) -> list[dict[str, Any]]:
    """把执行轨迹里所有 query_metric 调用提取成 [{metric_id, params, ok}, ...]。

    参数优先取 result["params"]（**已解析**的真实取值，如 2026-09-11），
    取不到才退回 arguments（模型原始传参，可能还是 @data_end-13 这种表达式）。
    为什么？因为要拿真实取值去和期望值比对，表达式形式没法直接比。
    """
    calls: list[dict[str, Any]] = []
    for step in steps or []:
        if _step_field(step, "kind") != "tool_call":
            continue
        if _step_field(step, "tool_name") != "query_metric":
            continue

        result = _step_field(step, "result") or {}
        arguments = _step_field(step, "arguments") or {}
        calls.append(
            {
                "metric_id": str(result.get("metric_id") or arguments.get("metric_id") or ""),
                "params": dict(result.get("params") or arguments.get("params") or {}),
                "ok": bool(_step_field(step, "ok", False)),
            }
        )
    return calls


def check_metric_mapping(steps: Sequence[Any] | None, expected_metric_id: str | None) -> bool | None:
    """模型有没有查对指标。

    判定标准：**调用记录里出现过期望的 metric_id**（不要求只查它）。
    为什么允许它多查？因为「查了留存顺带查日活做交叉验证」是合理行为，
    不该被判错。评测要惩罚的是"选错指标"，不是"多做了功课"。
    """
    if not expected_metric_id:
        return None
    return any(call["metric_id"] == expected_metric_id for call in metric_calls(steps))


def _normalize(value: Any) -> Any:
    """参数值归一化后再比较。

    为什么需要？因为模型可能传 int 1，而注册表默认值是字符串 "1"，
    或者传 "2026-9-1" 而期望 "2026-09-01"。直接比会假失败。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number) if number.is_integer() else number
    text = str(value).strip()
    # 数字型字符串统一转成数字，让 1 和 "1" 视为相同
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text


def check_params(
    steps: Sequence[Any] | None,
    expected_metric_id: str | None,
    expected_params: dict[str, Any] | None,
) -> tuple[bool | None, dict[str, Any]]:
    """参数抽取是否准确。

    只校验 expected_params 里**显式声明**的键，未声明的键一律不管。
    为什么要这样设计？因为绝大多数用例的参数是「用默认值」，
    而默认值本来就该由系统兜底，模型不传才是对的。
    如果把默认参数也纳入校验，就会奖励「模型把默认值又抄了一遍」这种无意义行为。

    判定口径：**任意一次调用命中期望参数即算通过**，和 check_metric_mapping 一致。
    为什么不用"最后一次调用"？真机评测 E15 暴露了问题 —— 那条问「v2.0.0 上线后
    留存变好了吗」，模型先查 D1、又查 D7 做对比，最后一次是 D7，
    于是被判成"参数抽错"。可它明明查对了 D1，还多做了功课。
    **评测要惩罚的是漏查，不是多查。**

    返回 (是否通过, 明细)。明细里逐键列出期望值和实际值，方便报告直接展示。
    """
    if not expected_metric_id or not expected_params:
        return None, {}

    calls = [c for c in metric_calls(steps) if c["metric_id"] == expected_metric_id]
    if not calls:
        # 指标都没查对，参数无从谈起 —— 返回 False 而不是 None，
        # 因为这是"确实抽错了"，不该被排除在分母之外。
        return False, {key: {"expected": value, "actual": None} for key, value in expected_params.items()}

    def build_detail(actual_params: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {}
        passed = True
        for key, expected in expected_params.items():
            actual_value = actual_params.get(key)
            ok = _normalize(actual_value) == _normalize(expected)
            passed = passed and ok
            detail[key] = {"expected": expected, "actual": actual_value, "ok": ok}
        return passed, detail

    for call in calls:
        passed, detail = build_detail(call["params"])
        if passed:
            return True, detail

    # 全都没命中：把**最后一次**调用的明细报出来 —— 人工复核时最近一次最有参考价值
    _, detail = build_detail(calls[-1]["params"])
    return False, detail


# ===========================================================================
# 四、汇总
# ===========================================================================

@dataclass
class EvalSummary:
    """一次评测的汇总指标。所有比率都是「不适用时为 None」，而不是 0。"""

    total: int = 0
    ok_count: int = 0                       # Agent 正常返回的条数
    mapping_checked: int = 0
    mapping_correct: int = 0
    param_checked: int = 0
    param_correct: int = 0
    number_total: int = 0
    number_matched: int = 0
    refused_correctly: int = 0              # 超范围用例中正确拒答的条数
    refuse_checked: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0             # Phase 6：输入侧 token 总量
    total_cached_tokens: int = 0             # Phase 6：其中命中提示词缓存的部分
    total_elapsed_ms: float = 0.0
    total_iterations: int = 0

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float | None:
        """统一的口径：分母为 0 时返回 None，不返回 0。

        这个细节很重要 —— 「没有可测的样本」和「测了但全错」是完全不同的结论，
        用 0 表示会让报告看起来比实际糟，也会掩盖"评测集设计有缺口"这个事实。
        """
        if denominator <= 0:
            return None
        return round(numerator / denominator, 4)

    @property
    def metric_mapping_rate(self) -> float | None:
        return self._rate(self.mapping_correct, self.mapping_checked)

    @property
    def param_rate(self) -> float | None:
        return self._rate(self.param_correct, self.param_checked)

    @property
    def number_rate(self) -> float | None:
        return self._rate(self.number_matched, self.number_total)

    @property
    def refuse_rate(self) -> float | None:
        return self._rate(self.refused_correctly, self.refuse_checked)

    @property
    def avg_tokens(self) -> float:
        return round(self.total_tokens / self.total, 1) if self.total else 0.0

    @property
    def avg_elapsed_ms(self) -> float:
        return round(self.total_elapsed_ms / self.total, 1) if self.total else 0.0

    @property
    def cache_hit_rate(self) -> float | None:
        """提示词缓存命中率 = 命中 token / 输入 token。

        【为什么用「输入 token」当分母，而不是「总 token」？】

          因为缓存只作用于**输入侧**：供应商缓存的是「相同前缀」的 KV，
          模型吐出来的输出 token 一个子儿都省不了。
          用总 token 当分母会把输出也摊进来，命中率被稀释，
          看起来像「优化没效果」，实际是把两个不相干的量混在了一起。
          指标口径要跟它真正作用的范围对齐 —— 这是做指标时最容易犯的错。

        【这个数说明什么？】

          ReAct 每多走一轮，系统提示词（含指标目录）就要整包重发一次。
          命中率越高，说明「每轮都在重复付费的那部分固定开销」占比越大 ——
          也就是提示词精简的收益空间越大。
          Phase 5 基线是 0（当时没有计量），Phase 6 起开始记录并作为优化基准。

          注意：缓存有生效延迟（一般要求前缀稳定且被访问过一次以上），
          单条用例可能偏低；所以报告里看的是**整体命中率**，不是逐例。
        """
        return self._rate(self.total_cached_tokens, self.total_prompt_tokens)

    @property
    def avg_cached_tokens(self) -> float:
        return round(self.total_cached_tokens / self.total, 1) if self.total else 0.0

    @property
    def avg_iterations(self) -> float:
        return round(self.total_iterations / self.total, 2) if self.total else 0.0


def summarize(results: Sequence[Any]) -> EvalSummary:
    """把逐例结果汇总成总指标。

    results 里的元素只需要具备若干属性（CaseResult 或它的 dict 形态），
    用 _step_field 取值 —— 保持和 metric_calls 一样的兼容策略。
    """
    summary = EvalSummary(total=len(results))
    for item in results:
        if _step_field(item, "ok", False):
            summary.ok_count += 1

        mapping = _step_field(item, "metric_mapping_ok")
        if mapping is not None:
            summary.mapping_checked += 1
            if mapping:
                summary.mapping_correct += 1

        params_ok = _step_field(item, "params_ok")
        if params_ok is not None:
            summary.param_checked += 1
            if params_ok:
                summary.param_correct += 1

        numbers = _step_field(item, "number_check")
        if numbers is not None:
            summary.number_total += getattr(numbers, "total", 0)
            summary.number_matched += getattr(numbers, "matched_count", 0)

        if _step_field(item, "expect_no_query", False):
            summary.refuse_checked += 1
            if _step_field(item, "refuse_ok", False):
                summary.refused_correctly += 1

        usage = _step_field(item, "usage") or {}
        summary.total_tokens += int(usage.get("total_tokens", 0) or 0)
        summary.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        summary.total_cached_tokens += int(usage.get("cached_tokens", 0) or 0)
        summary.total_elapsed_ms += float(_step_field(item, "elapsed_ms", 0.0) or 0.0)
        summary.total_iterations += int(_step_field(item, "iterations", 0) or 0)

    return summary


def format_rate(rate: float | None) -> str:
    """比率的展示格式。None 显示成「不适用」，而不是「0.00%」。"""
    return "不适用" if rate is None else f"{rate * 100:.2f}%"