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

【Phase 9：数字分析原语已下沉到 src/grounding.py】

  「逐数字溯源」让运行时防线也需要「抽数字、比数字」。与其在 Agent 侧另写一套，
  不如把原语抽到中立模块，两边共用同一口径 —— 否则会出现
  「运行时拦了、评测说没问题」这种自相矛盾的结论。

  本模块保留的是**评测专属**部分：
      三、从执行轨迹里提取「模型做了什么」（指标映射 / 参数抽取）
      四、汇总（EvalSummary / summarize / format_rate）
  数字分析部分（原第一、二节）改为从 src.grounding 导入。这些名字依然存在于
  本模块的命名空间里，所以 `from src.eval.metrics import extract_numbers`
  这类既有引用无需改动。

  为什么是「下沉」而不是「让 Agent 直接 import 本模块」？
  因为那是**真实的循环导入**（eval → agent 是既定的依赖方向），
  完整链路图见 src/grounding.py 的模块头注释。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# 数字原语：来自评测台与运行时防线共用的中立模块。
# 这里再导出一次是为了兼容既有调用方（scripts / tests / eval/hallucination.py）。
from src.grounding import (  # noqa: F401
    ExtractedNumber,
    NumberCheck,
    check_answer_numbers,
    collect_reference_values,
    derived_values,
    extract_numbers,
    is_close,
)
from src.grounding import _step_field  # 同包内共用的「轨迹取值」原语


# ===========================================================================
# 三、从执行轨迹里提取「模型做了什么」
# ===========================================================================

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
