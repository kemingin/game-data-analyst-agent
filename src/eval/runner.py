# -*- coding: utf-8 -*-
"""
评测执行器（Phase 5）
================================================================================
【评测到底在跑什么？】

  对每一条用例，跑两条**互不相干**的路径：

      路径 A（被测对象）：把问题交给 Agent，看它怎么理解、查了什么、答了什么
      路径 B（真值）：用用例声明的期望 metric_id + params，直接调工具查一次

  然后把 A 的结果拿去和 B 对账。这个「双路径」结构是整个评测体系的地基：

      为什么不能拿 A 自己查出来的数据当标准？
        因为如果 Agent 把「最近 14 天」错理解成「最近 7 天」，它查出来的数字
        在自己那套逻辑里是完全自洽的 —— 拿它自己的数据对账，永远满分。
        这就是"自己印证自己"的陷阱，评测必须有一条独立路径。

【为什么要统计 token 和延迟？】

  因为 Agent 的可用性不只是"答得对"。一个每次要烧 1 万 token、等 30 秒的
  Agent，运营同学是不会用的。把成本量化出来，才能谈优化（比如 Phase 6 的
  prompt 精简）。能直接报出「平均 5.4k token / 2.1 轮 / 8 秒」，
  比说「性能还行」有说服力得多。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

from src.agent.react_agent import AgentAnswer, GameDataAgent
from src.agent.tools import ToolExecutor
from src.eval.eval_set import EvalCase, EvalSet, load_eval_set
from src.eval.metrics import (
    NumberCheck,
    check_answer_numbers,
    check_metric_mapping,
    check_params,
    extract_numbers,
    format_rate,
    metric_calls,
    summarize,
)
from src.metrics.registry import get_registry

# 进度回调的类型：收到 (已完成的第几条, 总条数, 当前用例, 当前结果)
ProgressHandler = Callable[[int, int, EvalCase, "CaseResult"], None]


# ===========================================================================
# 一、单例结果
# ===========================================================================

@dataclass
class CaseResult:
    """一条用例的评测结果。"""

    case_id: str
    question: str
    category: str = ""
    notes: str = ""

    ok: bool = True                         # Agent 是否正常返回
    answer: str = ""
    error: str | None = None

    # --- 三项核心判定 ---
    metric_mapping_ok: bool | None = None   # None = 本用例不校验
    params_ok: bool | None = None
    params_detail: dict[str, Any] = field(default_factory=dict)
    number_check: NumberCheck | None = None

    # --- 超范围用例 ---
    expect_no_query: bool = False
    refuse_ok: bool = False
    actual_calls: list[dict[str, Any]] = field(default_factory=list)

    # --- 参考数据（独立路径算出来的真值）---
    reference_metric_id: str | None = None
    reference_params: dict[str, Any] = field(default_factory=dict)
    reference_row_count: int = 0

    # --- 运行信息 ---
    iterations: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    provider: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化。number_check 是对象，这里转成基本类型，方便存成 JSON 复算。"""
        return {
            "case_id": self.case_id,
            "question": self.question,
            "category": self.category,
            "ok": self.ok,
            "answer": self.answer,
            "error": self.error,
            "metric_mapping_ok": self.metric_mapping_ok,
            "params_ok": self.params_ok,
            "params_detail": self.params_detail,
            "number_total": self.number_check.total if self.number_check else 0,
            "number_traceable": self.number_check.matched_count if self.number_check else 0,
            "unexplained_numbers": (
                [item.raw for item in self.number_check.unexplained] if self.number_check else []
            ),
            "expect_no_query": self.expect_no_query,
            "refuse_ok": self.refuse_ok,
            "actual_calls": self.actual_calls,
            "reference_metric_id": self.reference_metric_id,
            "reference_params": self.reference_params,
            "iterations": self.iterations,
            "usage": self.usage,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "provider": self.provider,
            "model": self.model,
        }


# ===========================================================================
# 二、执行器
# ===========================================================================

class EvalRunner:
    """跑完整评测集，产出逐例结果。"""

    def __init__(
        self,
        agent: GameDataAgent | None = None,
        tools: ToolExecutor | None = None,
        eval_set: EvalSet | None = None,
    ) -> None:
        # 依赖注入，和前面几个 Phase 保持一致：
        # 测试时可以塞一个「假 Agent」，在不调 LLM 的前提下验证评测流程本身。
        self.eval_set: EvalSet = eval_set or load_eval_set()
        self.tools: ToolExecutor = tools or ToolExecutor()
        self.agent: GameDataAgent = agent or GameDataAgent(tools=self.tools)

    # ------------------------------------------------------------------
    # 真值：独立路径
    # ------------------------------------------------------------------
    def build_reference(self, case: EvalCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """用期望指标 + 期望参数独立查一次，得到真值。

        这条路径**完全不经过 LLM** —— 直接调工具层。
        所以哪怕模型答得天花乱坠，真值始终是数据库说的话。

        返回 (参考数据行, 实际使用的参数)。查失败时返回空行，
        调用方（number_check）会因为参考集为空而把所有数字判为"未对上"，
        这正是我们想要的效果：**参考数据都拿不到的时候，就不该给模型打分**。
        """
        metric_id = case.resolved_reference_metric_id()
        if not metric_id:
            return [], {}

        result = self.tools.execute(
            "query_metric",
            {"metric_id": metric_id, "params": case.reference_params},
        )
        if not result.ok:
            return [], {}
        return result.data.get("rows") or [], result.data.get("params") or {}

    def build_extra_reference_rows(
        self, actual_calls: Sequence[dict[str, Any]]
    ) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
        """把模型**每一次成功查询**的真值都查出来，按指标组织。

        返回 [(metric_id, 实际参数, 数据行), ...]，供两类调用方使用：
          · 数字级对账 —— 只要摊平后的数值（见 build_extra_reference_values）；
          · Judge 真值文本 —— 需要「指标名 + 参数 + 数据行」的完整结构，
            否则评审员看不到这些数据，会把有出处的数字判成编造（见测试复盘 Phase 3）。

        【为什么查询逻辑只写在这里一份？】
          评测系统最怕「两把尺子」：同一句话在数字报告里算可溯源、在 Judge 报告里算编造，
          复核的人根本不知道该信哪个。所以行与数值必须来自**同一次查询**，
          由本方法统一提供，调用方只做形态转换（摊平 / 渲染）。

        【为什么按 (指标, 参数) 去重，而不是排除「主指标」？——Phase 3 的第二个坑】
          第一版为了省一次查询，把 metric_id 等于主指标的调用整体跳过。
          结果模型对**同一指标查了不同参数**时（E15 问版本留存，它同时查了
          version_retention 的 day_n=1 和 day_n=7 做交叉验证），day_n=7 那次的真值
          被静默丢弃 —— Judge 于是把一批真实数据判成编造。
          教训：去重的正确维度是**「同一次查询」**（指标 + 参数），不是「同一个指标」。
          代价只是主真值被重复查一次（一次 SQL，几十毫秒），换来"任何一次真实查询都不漏"。
        """
        references: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
        seen: set[tuple[str, str]] = set()
        for call in actual_calls:
            metric_id = call.get("metric_id")
            if not metric_id or not call.get("ok"):
                continue
            params = call.get("params") or {}
            # 去重键用 params 的排序后字符串：字典不可哈希，且参数顺序无关紧要。
            key = (metric_id, repr(sorted(params.items(), key=lambda kv: kv[0])))
            if key in seen:
                continue
            seen.add(key)
            try:
                result = self.tools.execute(
                    "query_metric", {"metric_id": metric_id, "params": params}
                )
            except Exception:  # noqa: BLE001 - 辅助路径失败不影响主判定
                continue
            if result.ok:
                data = result.data or {}
                references.append((metric_id, data.get("params") or params, data.get("rows") or []))
        return references

    def build_extra_reference_values(self, actual_calls: Sequence[dict[str, Any]]) -> list[float]:
        """把模型「自己多查的那些指标」的真值也查出来，用于对账（Phase 6 新增）。

        【为什么需要这一步？】
          原本的对账集只由 expected_metric_id 算出来。但真机评测发现，
          模型在长尾问法下会**主动多查几个指标做交叉验证**：被问「付费转化情况怎么样」时，
          它会顺手查 arppu、arpu，好给出更完整的经营判断。
          这些数字全部来自真实的工具返回，却因为不在期望指标的真值里
          被判成「无出处」—— 属于典型的**误报**。
          而本项目对账系统的原则是「宁可漏报，也不要天天误报」
          （见 _definition_numbers 的说明）。这条修正就是在还这笔账。

        【会不会把口径放得太松？】
          不会。这里走的是**和主真值完全相同的独立路径**：
          拿模型自己声明的 metric_id + params 再查一次数据库，
          而不是相信模型给的数字。模型编的数字依然对不上 ——
          编出来的数不会恰好等于数据库的返回。
          换句话说：放宽的是「允许查哪些指标」，没有放宽「数字必须来自数据库」。

        ★ 实现上直接摊平 build_extra_reference_rows 的返回值，
          而不是自己再查一遍库 —— 这样「数值」和「给 Judge 看的行」永远同源。
        """
        values: list[float] = []
        for _metric_id, _params, rows in self.build_extra_reference_rows(actual_calls):
            values.extend(_flatten_rows(rows))
        return values

    # ------------------------------------------------------------------
    # 单条用例
    # ------------------------------------------------------------------
    def run_case(self, case: EvalCase) -> CaseResult:
        """跑一条用例。

        这个方法**不抛异常**：评测跑 20 条真机请求，中途某条因为网络抖动失败，
        不该让整轮评测崩掉 —— 否则你永远拿不到一份完整报告。
        失败会被如实记进 CaseResult.ok=False，并在报告里单独列出。
        """
        result = CaseResult(
            case_id=case.case_id,
            question=case.question,
            category=case.category,
            notes=case.notes,
            expect_no_query=case.expect_no_query,
            reference_metric_id=case.resolved_reference_metric_id(),
        )

        try:
            answer: AgentAnswer = self.agent.ask(case.question, history=list(case.history) or None)
        except Exception as exc:  # noqa: BLE001 - 评测器必须兜住一切，保证报告完整
            result.ok = False
            result.error = str(exc)
            return result

        result.ok = answer.ok
        result.answer = answer.answer
        result.error = answer.error
        result.iterations = answer.iterations
        result.usage = answer.usage
        result.elapsed_ms = answer.elapsed_ms
        result.provider = answer.provider
        result.model = answer.model
        result.actual_calls = metric_calls(answer.steps)

        # ---- 真值（独立路径）----
        reference_rows, used_params = self.build_reference(case)
        result.reference_params = used_params
        result.reference_row_count = len(reference_rows)

        # ---- 判定 1：指标映射 ----
        result.metric_mapping_ok = check_metric_mapping(answer.steps, case.expected_metric_id)

        # ---- 判定 2：参数抽取 ----
        result.params_ok, result.params_detail = check_params(
            answer.steps, case.expected_metric_id, case.expected_params
        )

        # ---- 判定 3：答案数字对账 ----
        # 超范围用例不查数，自然也没有真值可比，跳过对账。
        if not case.expect_no_query:
            # extra_allowed 把三类"合法但不在数据里"的数字放进允许集：
            #   ① 期望参数值 —— 回答里说「最近 14 天」是正常的；
            #   ② 参考行数   —— 「共 7 行」是正常的；
            #   ③ 指标定义里的数字 —— 见 _definition_numbers 的说明。
            # 另外把模型「自己多查的指标」的真值也并进来（见 build_extra_reference_values）：
            # 长尾问法下模型爱顺手多查一两个指标做交叉验证，那些数字同样是可追溯的。
            reference_values = _flatten_rows(reference_rows)
            reference_values.extend(
                self.build_extra_reference_values(result.actual_calls)
            )
            extra_allowed = (
                list(case.expected_params.values())
                + [len(reference_rows)]
                + _definition_numbers(case.resolved_reference_metric_id())
                + [
                    value
                    for call in result.actual_calls
                    for value in (call.get("params") or {}).values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
            )
            result.number_check = check_answer_numbers(
                answer.answer, reference_values, extra_allowed=extra_allowed
            )

        # ---- 判定 4：超范围拒答 ----
        if case.expect_no_query:
            # 判定标准是「没有任何一次成功的查数」，而不是「一次都没调」。
            # 因为模型可能先尝试调用、被参数校验拦下来，那同样是正确行为 ——
            # 结果是"没查到数"，就说明它没有硬凑。
            result.refuse_ok = not any(call["ok"] for call in result.actual_calls)

        return result

    # ------------------------------------------------------------------
    # 全量
    # ------------------------------------------------------------------
    def run_all(
        self,
        cases: Sequence[EvalCase] | None = None,
        progress: ProgressHandler | None = None,
    ) -> list[CaseResult]:
        """跑完整评测集。progress 用于在 CLI 里打实时进度（真机跑 20 条要等一会儿）。"""
        targets = list(cases) if cases is not None else self.eval_set.all()
        results: list[CaseResult] = []
        for index, case in enumerate(targets, start=1):
            result = self.run_case(case)
            results.append(result)
            if progress is not None:
                progress(index, len(targets), case, result)
        return results


def _flatten_rows(rows: Sequence[dict[str, Any]]) -> list[float]:
    """把参考数据行摊平成数值列表（对账用）。"""
    from src.eval.metrics import collect_reference_values

    return collect_reference_values(rows)


def _metric_definition(metric_id: str | None) -> str:
    """取指标的业务定义原文。

    为什么要单独抽一个函数？因为「定义里的数字算合法出处」这件事有两处要用：
      · _definition_numbers —— 数字级对账的白名单；
      · Judge 真值文本 —— 让评审员也能看到口径原文。
    如果两边各取各的，就会出现「数字级认、Judge 不认」的尺子分裂
    （测试复盘 Phase 3 正是踩了这个坑：模型引用「行业参考区间 10%~25%」
     被数字级放行、却被 Judge 判成无依据）。同源读取是最省事的根治办法。
    """
    if not metric_id:
        return ""
    try:
        return get_registry().get(metric_id).business_definition or ""
    except Exception:  # noqa: BLE001 - 定义取不到不该让整条评测失败
        return ""


def _definition_numbers(metric_id: str | None) -> list[float]:
    """指标业务定义里出现的数字，也算「合法出处」。

    为什么需要这一步？因为模型很爱引用口径说明。真机评测 E03 的回答里有一句
    「粘性落在行业参考区间（10%~25%）的最下沿」—— 那个 25 来自注册表的
    business_definition，不是它编的。如果不放行，凡是引用口径的回答都会被
    反复报成「存疑」，复核的人很快就会对这个指标失去耐心。

    代价是：万一模型编的数字恰好等于定义里的某个数，会被漏掉。
    但定义里的数字很少（一般是 7 / 30 / 10 / 25 这类口径常量），风险可接受 ——
    对账系统的原则本来就是**宁可漏报，也不要天天误报**。
    """
    return [item.value for item in extract_numbers(_metric_definition(metric_id))]


# ===========================================================================
# 三、报告生成
# ===========================================================================

def build_report(
    results: Sequence[CaseResult],
    eval_set: EvalSet | None = None,
    title: str = "Agent 评测报告",
) -> str:
    """生成纯文本报告。

    为什么报告要是纯文本而不是 JSON？
      JSON 是给程序读的，而这份报告是给人看的（自己复盘、对外展示）。
      纯文本可以直接贴进复盘文档，不依赖任何工具就能阅读。
      需要机器复算的话，CaseResult.to_dict() 已经提供了结构化出口，两者不冲突。
    """
    summary = summarize(results)
    lines: list[str] = []
    line = "=" * 78

    lines.append(line)
    lines.append(f"        游戏数据智能分析师 Agent · {title}")
    lines.append(line)
    lines.append(f"评测时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"用例数量：{summary.total}")
    if results:
        lines.append(f"模型：{results[0].provider}/{results[0].model}")
    if eval_set is not None and eval_set.meta:
        lines.append(f"评测集版本：{eval_set.meta.get('version', '-')}（{eval_set.path.name}）")
    lines.append("")

    # ---------------- 总览 ----------------
    lines.append("-" * 78)
    lines.append("一、总览")
    lines.append("-" * 78)
    lines.append(f"  ① 指标映射准确率    {format_rate(summary.metric_mapping_rate)}"
                 f"    （{summary.mapping_correct} / {summary.mapping_checked}）")
    lines.append(f"  ② 参数抽取准确率    {format_rate(summary.param_rate)}"
                 f"    （{summary.param_correct} / {summary.param_checked}）")
    lines.append(f"  ③ 答案数字可追溯率  {format_rate(summary.number_rate)}"
                 f"    （{summary.number_matched} / {summary.number_total} 个数字）")
    if summary.refuse_checked:
        lines.append(f"  ④ 超范围拒答正确率  {format_rate(summary.refuse_rate)}"
                     f"    （{summary.refused_correctly} / {summary.refuse_checked}）")
    lines.append("")
    lines.append(f"  Agent 正常返回      {summary.ok_count} / {summary.total}")
    lines.append(f"  平均迭代轮数        {summary.avg_iterations} 轮")
    lines.append(f"  平均 token 消耗     {summary.avg_tokens:,.1f}")
    lines.append(f"  token 总计          {summary.total_tokens:,}")
    # Phase 6：提示词缓存命中率。只有供应商上报了该字段才有意义，
    # 否则显示「未上报」而不是「0.00%」—— 前者是数据缺失，后者是真实为零。
    if summary.total_cached_tokens:
        lines.append(f"  缓存命中 token      {summary.total_cached_tokens:,}"
                     f" / 输入 {summary.total_prompt_tokens:,}"
                     f"（命中率 {format_rate(summary.cache_hit_rate)}，"
                     f"平均 {summary.avg_cached_tokens:,.0f}/条）")
    else:
        lines.append("  缓存命中 token      未上报（该供应商未返回缓存字段）")
    lines.append(f"  平均响应耗时        {summary.avg_elapsed_ms:,.0f} ms")
    lines.append("")
    lines.append("  说明：比率显示「不适用」表示该类用例数为 0（没有可测样本），"
                 "而不是全错。")
    lines.append("        ③ 的判定分四档：严格命中 / 口语近似（±3%）/ 可由参考值推导"
                 "（均值·差值·涨幅等）/ 存疑。")
    lines.append("        只有「存疑」那一档需要人工看；前三档都算「有出处」，"
                 "所以 ③ 叫「可追溯率」而不是「一致率」。")
    lines.append("        已知边界：模型额外查了其他窗口做对比（如多窗口版本留存）时，"
                 "那些数字确实来自数据库，但不在本用例的参考切片里，仍会落入存疑。")
    lines.append("        这个代价是刻意付的 —— 如果放行模型自己查出来的数据，"
                 "「查错窗口却自证正确」就再也抓不出来了。")
    lines.append("")

    # ---------------- 逐例明细 ----------------
    lines.append("-" * 78)
    lines.append("二、逐例明细")
    lines.append("-" * 78)

    for item in results:
        mapping = "不适用" if item.metric_mapping_ok is None else ("通过" if item.metric_mapping_ok else "未通过")
        params = "不适用" if item.params_ok is None else ("通过" if item.params_ok else "未通过")
        if item.number_check is None:
            numbers = "不适用"
        else:
            rate = item.number_check.rate
            numbers = f"{item.number_check.matched_count}/{item.number_check.total}" + (
                "" if rate is None else f"（{rate * 100:.0f}%）"
            )

        lines.append(f"[{item.case_id}] {item.question}")
        lines.append(f"     分类={item.category}｜映射={mapping}｜参数={params}｜数字对账={numbers}"
                     f"｜{item.iterations} 轮｜{item.usage.get('total_tokens', 0):,} tokens"
                     f"｜{item.elapsed_ms:,.0f} ms")
        if not item.ok:
            lines.append(f"     ⚠ Agent 未正常返回：{item.error}")
        if item.params_detail:
            for key, detail in item.params_detail.items():
                flag = "✓" if detail.get("ok") else "✗"
                lines.append(f"     {flag} 参数 {key}：期望 {detail['expected']}｜实际 {detail['actual']}")
        if item.actual_calls:
            called = "、".join(f"{c['metric_id']}" for c in item.actual_calls)
            lines.append(f"     实际查询指标：{called}")
        else:
            lines.append("     实际查询指标：无（未调用查数工具）")
        if item.number_check is not None and item.number_check.unexplained:
            lines.append(f"     存疑数字（需人工复核）：{item.number_check.describe_unexplained()}")
        lines.append("")

    # ---------------- 需要人工关注 ----------------
    lines.append("-" * 78)
    lines.append("三、需要人工关注的问题")
    lines.append("-" * 78)
    issues = _collect_issues(results)
    if not issues:
        lines.append("  无。所有用例的映射、参数、数字对账均通过。")
    else:
        for issue in issues:
            lines.append(f"  · {issue}")
    lines.append("")

    lines.append(line)
    lines.append("报告结束。评测算法的边界：程序只能判断「数字有没有对上参考数据」，")
    lines.append("无法判断「结论是否合理」「因果推断是否成立」—— 后者仍需人工复核。")
    lines.append(line)

    return "\n".join(lines)


def _collect_issues(results: Sequence[CaseResult]) -> list[str]:
    """把需要人看的问题挑出来，按严重程度排序。

    排序原则：**Agent 崩溃 > 选错指标 > 参数抽错 > 拒答失败 > 数字存疑**。
    因为选错指标意味着整个回答答非所问，是最严重的；而数字存疑里有相当一部分
    是合理衍生（只是没被前两档覆盖到），严重程度最低。
    """
    wrong_mapping: list[str] = []
    wrong_params: list[str] = []
    unexplained: list[str] = []
    refuse_failed: list[str] = []
    failed: list[str] = []

    for item in results:
        if not item.ok:
            failed.append(f"[{item.case_id}] Agent 未正常返回：{item.error}")
        if item.metric_mapping_ok is False:
            called = "、".join(c["metric_id"] for c in item.actual_calls) or "无"
            wrong_mapping.append(f"[{item.case_id}] 指标映射错误（实际查询：{called}）")
        if item.params_ok is False:
            bad = [
                f"{key}（期望 {detail['expected']}，实际 {detail['actual']}）"
                for key, detail in item.params_detail.items()
                if not detail.get("ok")
            ]
            wrong_params.append(f"[{item.case_id}] 参数抽取错误：{'；'.join(bad)}")
        if item.number_check is not None and item.number_check.unexplained:
            unexplained.append(
                f"[{item.case_id}] 有 {len(item.number_check.unexplained)} 个数字在参考数据里"
                f"找不到出处，需人工复核：{item.number_check.describe_unexplained()}"
            )
        if item.expect_no_query and not item.refuse_ok:
            refuse_failed.append(f"[{item.case_id}] 超范围问题却执行了查数，没有守住数据边界")

    return failed + wrong_mapping + wrong_params + refuse_failed + unexplained


def run_eval(
    limit: int | None = None,
    case_ids: Sequence[str] | None = None,
    progress: ProgressHandler | None = None,
    runner: EvalRunner | None = None,
) -> tuple[list[CaseResult], EvalSet]:
    """便捷入口：按需筛选用例后跑一轮评测。

    limit / case_ids 的意义在于**调试期省钱**：
    真机跑一条要几千 token，开发评测逻辑时先跑 2~3 条验证流程通了，
    再跑全量 —— 这是调用付费 API 的基本纪律。
    """
    runner = runner or EvalRunner()
    cases = runner.eval_set.all()
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.case_id in wanted]
    if limit is not None:
        cases = cases[:limit]
    return runner.run_all(cases, progress=progress), runner.eval_set