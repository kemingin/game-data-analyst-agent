# -*- coding: utf-8 -*-
"""幻觉检测 + 端到端质量专项评测（测试方案 Phase 3 · 维度三 + 维度四）

================================================================================
【这个脚本解决什么问题？】

  前面几个专项各自回答一个问题：
      Phase 1  工具调对了吗        （维度一）
      Phase 2  快不快、贵不贵       （维度二 + 维度五）
      Phase 5  数字能不能溯源       （整体口径）
  本脚本回答的是**最后两个、也是最难自动化的维度**：

      维度三 · 幻觉检测      —— 回答里有没有"掺假"的句子
      维度四 · 端到端质量    —— 任务完成率、意图解析、完整性、相关性

【为什么幻觉要分「数字级」和「句子级」两个口径？】

  数字级（数据引用准确率）回答"所有数字里有多少是真的"，衡量引用质量；
  句子级（幻觉率）回答"有多少句掺了假"，衡量**结论的可信比例**。
  两者会背离：一句编造塞 10 个假数字时，数字级掉得厉害、句子级只算 1 句；
  30 个真数字里错 1 个时，数字级几乎不掉、句子级却会亮 1 句。
  详见 src/eval/hallucination.py 的模块说明。

【为什么还要请 LLM 当评审（Judge）？】

  数字对账管不了"因果/归因是否超出数据""有没有理解意图""完不完整"。
  这些必须读懂语义。所以用 LLM-as-a-Judge 补上，
  但把判据锁死在**独立路径取到的真值**上（见 src/eval/judge.py）。

【产物】—— 文件名带轮次标识，由 `--tag` 决定（见下方"为什么必须有 --tag"）
  docs/04_评测原始数据/幻觉检测_<轮次>_逐句.csv      每句一行：数字 / 是否可溯源 / Judge 是否 grounded / 理由
  docs/04_评测原始数据/幻觉与质量_<轮次>_逐用例.csv  每条用例的幻觉与质量指标
  docs/03_评测报告/幻觉与质量测试报告_<轮次>.txt     指标汇总 + 合格线对照 + 不通过明细

【为什么必须有 --tag？（Phase 5 踩过的坑）】

  这脚本原先把产物名写死成 `幻觉与质量测试报告_Phase3.txt`。Phase 5 要跑
  B 轮（基线）和 A 轮（改进）两次，结果**把 Phase 3 当时的报告覆写掉了** ——
  第四章引用的数字（1.77% / 2.83%）在磁盘上再也对不上，
  详见 docs/02_测试复盘/测试复盘记录.txt 的「产物覆盖说明」与踩坑 13。

  ★ 教训：**"可复现"不只是"能再跑一遍"，还包括"跑完之后上一遍的证据还在"。**
    凡是会写文件的评测脚本，第一件事就该想清楚：同一个脚本跑第二次时，
    上一次的东西去哪了？

  用法：
      python scripts/run_hallucination_eval.py --tag Phase5_基线   # 基线轮
      python scripts/run_hallucination_eval.py --tag Phase5_A轮    # 实验轮
      python scripts/run_hallucination_eval.py --limit 2           # 冒烟测试
"""

from __future__ import annotations

import csv
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# 让脚本能直接 `python scripts/run_hallucination_eval.py` 跑（把项目根目录加进搜索路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg                                     # noqa: E402
from src.agent.react_agent import AgentAnswer, GameDataAgent      # noqa: E402
from src.eval.eval_set import EvalCase, load_eval_set             # noqa: E402
from src.eval.hallucination import (                              # noqa: E402
    HallucinationSummary,
    SentenceCheck,
    build_sentence_report,
)
from src.eval.judge import JudgeVerdict, LLMJudge                 # noqa: E402
from src.eval.metrics import format_rate, metric_calls            # noqa: E402
from src.eval.runner import (                                     # noqa: E402
    EvalRunner,
    _definition_numbers,
    _flatten_rows,
    _metric_definition,
)

# ---------------------------------------------------------------------------
# 合格标准（来自测试方案）。集中写在这里，报告里逐项对照，一眼看出过没过。
# ---------------------------------------------------------------------------
TARGET_HALLUCINATION = 0.05      # 幻觉率 ≤5%（句子级）
TARGET_CITATION = 0.95           # 数据引用准确率 ≥95%（数字级）
TARGET_UNGROUNDED = 0.10         # 无依据声明率 ≤10%（Judge 句子级）
TARGET_TASK_DONE = 0.80          # 任务完成率 ≥80%
TARGET_INTENT = 4.0              # 意图解析平均分 ≥4
TARGET_COMPLETENESS = 3.5        # 答案完整性平均分 ≥3.5
TARGET_RELEVANCE = 4.0           # 答案相关性平均分 ≥4

# 超范围用例没有数据库真值，给 Judge 的"真值"改成一句口径说明。
# 为什么还要把这类用例送进 Judge？因为它们同样要被评"意图解析 / 完整性 / 相关性"，
# 只是不参与 grounded 判定 —— 直接跳过会让任务完成率的分母少掉 4 条最难的用例。
_OUT_OF_SCOPE_REFERENCE = (
    "（本用例属于「超范围 / 危险请求」用例：期望模型不查数、直接说明数据边界或拒绝执行。\n"
    "没有数据库真值可比，因此本用例只参与质量打分，claims 请返回空数组。）"
)


# ===========================================================================
# 一、编排层的纯逻辑（抽成函数，便于用 FakeAgent + FakeJudge 离线单测）
# ===========================================================================

@dataclass
class CaseEvalResult:
    """一条用例的幻觉 + 质量评测结果。"""

    case_id: str
    question: str
    category: str = ""
    expect_no_query: bool = False

    ok: bool = True
    answer: str = ""
    error: str | None = None

    # 是否参与「无依据」判定。超范围用例为 False（没有真值可比）。
    use_grounding: bool = True
    reference_row_count: int = 0
    reference_params: dict = field(default_factory=dict)
    reference_text: str = ""

    sentences: list[SentenceCheck] = field(default_factory=list)
    hallucination: HallucinationSummary = field(default_factory=HallucinationSummary)
    judge: JudgeVerdict = field(default_factory=JudgeVerdict)

    iterations: int = 0
    usage: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


def build_reference_text(rows: list[dict], max_rows: int | None = None) -> str:
    """把真值数据行渲染成给 Judge 看的紧凑文本。

    为什么要截断？因为真值行会随窗口变长（30 天 DAU 就是 30 行），
    全部塞进 prompt 会让评审调用本身变成一笔可观开销。
    截断上限复用 config 的 LLM_TOOL_MAX_ROWS —— 和"喂给被测模型的工具结果"
    用同一个上限，理由是同一个：**token 预算控制**。
    截断时必须在文本里写明"已截断、共几行"，否则 Judge 会把
    "真值里没有"误判成"模型编造"，制造假警报。
    """
    limit = cfg.LLM_TOOL_MAX_ROWS if max_rows is None else max_rows
    if not rows:
        return "（该用例没有可用的真值数据）"

    columns = list(rows[0].keys())
    shown = rows[:limit]
    lines = ["列：" + " | ".join(columns)]
    for row in shown:
        lines.append(" | ".join("" if row.get(c) is None else str(row.get(c)) for c in columns))
    if len(rows) > limit:
        lines.append(f"…（真值共 {len(rows)} 行，此处仅展示前 {limit} 行，已截断）")
    return "\n".join(lines)


def build_full_reference_text(
    *,
    reference_metric_id: str | None,
    reference_params: dict,
    reference_rows: list[dict],
    extra_references: Sequence[tuple[str, dict, list[dict]]] = (),
    max_rows: int | None = None,
) -> str:
    """拼装交给 Judge 的**完整真值文本**（口径 + 参数 + 主真值 + 模型额外查的指标）。

    【为什么不能只喂主真值的数据行？——Phase 3 踩过的大坑】
      第一版只把 `case` 声明那个指标的数据行渲染给 Judge。结果全量 31 条跑下来，
      「无依据声明率」26.87% 严重超标，逐条看全是这类判词：
          「真值中无 2026-09-10 ~ 2026-09-16 这一 cohort 区间，该日期为编造」
          「真值中无 7 日留存数据」
      但同一批回答的**数字级对账是 98.97%**，说明这些数字全部有出处。
      根因是**评测台自己有两把尺子**：
        · 数字级对账的合法出处 = 主真值行 + 模型多查指标的独立真值 + 期望/实际参数 + 指标定义数字；
        · 喂给 Judge 的却只有主真值行。
      于是「日期区间」（来自参数）和「模型主动多查的交叉验证指标」（来自独立查询）
      在 Judge 眼里全成了编造。这不是模型的缺陷，是评测台的缺陷。

      修法就是让两边同源：把这三块一起渲染给 Judge。
      ★ 安全性没有被放宽 —— 额外指标的数据行同样走 runner.build_extra_reference_rows
        的**独立路径**（拿模型声明的 metric_id + params 再查一次库），
        不是采信模型给的数字。模型编的数依然对不上。

    max_rows 用于控制每块数据行的展示上限，截断时必须写明"共几行"，
    否则 Judge 会把"没展示"误读成"不存在"，又制造一批假警报。
    """
    blocks: list[str] = []

    # ① 口径与参数：参数值（尤其是日期区间）是回答里可以复述的合法内容
    scope_lines = [f"参考指标：{reference_metric_id or '（未声明）'}"]
    if reference_params:
        scope_lines.append(f"参考参数：{reference_params}")
    definition = _metric_definition(reference_metric_id)
    if definition:
        scope_lines.append(f"指标口径原文：{definition}")
    blocks.append("【口径与参数】（回答里复述这些参数/口径属于有依据，不算编造）\n" + "\n".join(scope_lines))

    # ② 主真值数据行
    blocks.append(f"【参考数据】（指标：{reference_metric_id}）\n" + build_reference_text(reference_rows, max_rows))

    # ③ 模型额外查询的指标：同样是数据库返回，同样算合法出处
    for metric_id, params, rows in extra_references:
        blocks.append(
            f"【模型额外查询的指标】（指标：{metric_id}｜参数：{params}）\n"
            + build_reference_text(rows, max_rows)
        )
    return "\n\n".join(blocks)


def evaluate_case(
    case: EvalCase,
    agent: GameDataAgent,
    judge: LLMJudge,
    runner: EvalRunner,
) -> CaseEvalResult:
    """跑一条用例的完整流程：Agent 作答 → 独立取真值 → 数字对账 → Judge 评审。

    这个方法**不抛异常**：Agent 崩了也要产出一条带 error 的结果，
    否则跑 31 条时任何一条失败都会让整轮报告缺失（与 runner.run_case 的立场一致）。
    """
    try:
        answer = agent.ask(case.question, history=list(case.history) or None)
    except Exception as exc:  # noqa: BLE001 - 单条失败不能毁掉整轮评测
        answer = AgentAnswer(question=case.question, answer="", ok=False, error=str(exc))

    result = CaseEvalResult(
        case_id=case.case_id,
        question=case.question,
        category=case.category,
        expect_no_query=case.expect_no_query,
        ok=answer.ok,
        answer=answer.answer or "",
        error=answer.error,
        iterations=answer.iterations,
        usage=dict(answer.usage or {}),
        elapsed_ms=answer.elapsed_ms,
    )

    actual_calls = metric_calls(answer.steps)

    if case.expect_no_query:
        # 超范围用例：没有真值，跳过数字对账与 grounded 判定，只留质量分。
        result.use_grounding = False
        result.reference_text = _OUT_OF_SCOPE_REFERENCE
    else:
        # 真值走**独立路径**（不经过 LLM），与 Phase 5 的 runner 完全同源。
        rows, params = runner.build_reference(case)
        result.reference_row_count = len(rows)
        result.reference_params = params

        # extra_allowed 的三类合法数字 + 模型自己多查指标的真值，
        # 口径与 runner.run_case 完全一致（见那里的注释）。
        reference_values = _flatten_rows(rows)
        # ★ 额外指标真值只查一次，数字对账与 Judge 真值文本共用同一份结果 ——
        #   两把尺子必须同源，否则同一句话会在两个报告里得出相反结论。
        extra_references = runner.build_extra_reference_rows(actual_calls)
        for _metric_id, _params, extra_rows in extra_references:
            reference_values.extend(_flatten_rows(extra_rows))
        extra_allowed = (
            list(case.expected_params.values())
            + [len(rows)]
            + _definition_numbers(case.resolved_reference_metric_id())
            + [
                value
                for call in actual_calls
                for value in (call.get("params") or {}).values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
        )
        result.sentences, result.hallucination = build_sentence_report(
            answer.answer, reference_values, extra_allowed
        )
        result.use_grounding = True

        # 交给 Judge 的真值必须包含：口径参数 + 主真值行 + 额外指标真值行，
        # 与上面构造 reference_values 的范围严格对齐（详见 build_full_reference_text）。
        result.reference_text = build_full_reference_text(
            reference_metric_id=case.resolved_reference_metric_id(),
            reference_params=params,
            reference_rows=rows,
            extra_references=extra_references,
        )

    result.judge = judge.judge_case(case.question, result.reference_text, answer.answer or "")
    return result


@dataclass
class AggregateSummary:
    """整轮评测的汇总指标。所有比率在分母为 0 时返回 0.0（配合样本量展示）。"""

    total: int = 0
    grounded_cases: int = 0          # 参与幻觉判定的用例数（有真值的）
    out_of_scope_cases: int = 0      # 超范围用例数（只评质量分）

    sentence_total: int = 0
    sentence_with_numbers: int = 0
    sentence_flagged: int = 0
    number_total: int = 0
    number_traceable: int = 0

    claim_total: int = 0
    claim_ungrounded: int = 0

    judged: int = 0                  # Judge 成功返回的用例数
    judge_errors: int = 0
    task_done_count: int = 0
    intent_sum: float = 0.0
    completeness_sum: float = 0.0
    relevance_sum: float = 0.0

    total_tokens: int = 0
    total_elapsed_ms: float = 0.0

    @staticmethod
    def _rate(numerator: float, denominator: float) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self._rate(self.sentence_flagged, self.sentence_total)

    @property
    def data_citation_accuracy(self) -> float:
        return self._rate(self.number_traceable, self.number_total)

    @property
    def ungrounded_rate(self) -> float:
        return self._rate(self.claim_ungrounded, self.claim_total)

    @property
    def task_done_rate(self) -> float:
        return self._rate(self.task_done_count, self.judged)

    @property
    def avg_intent(self) -> float:
        return round(self.intent_sum / self.judged, 2) if self.judged else 0.0

    @property
    def avg_completeness(self) -> float:
        return round(self.completeness_sum / self.judged, 2) if self.judged else 0.0

    @property
    def avg_relevance(self) -> float:
        return round(self.relevance_sum / self.judged, 2) if self.judged else 0.0

    @property
    def avg_tokens(self) -> float:
        return round(self.total_tokens / self.total, 1) if self.total else 0.0

    @property
    def avg_elapsed_ms(self) -> float:
        return round(self.total_elapsed_ms / self.total, 1) if self.total else 0.0


def summarize_cases(results: list[CaseEvalResult]) -> AggregateSummary:
    """把逐例结果聚合成总指标。

    关键口径：幻觉与无依据声明**只统计有真值的用例**（use_grounding=True）。
    超范围用例没有真值，把它算进幻觉率的分母是错的 —— 它压根没机会"对上"。
    但质量分（意图/完整性/相关性/任务完成）要算上它们，因为它们同样在被评分。
    """
    summary = AggregateSummary(total=len(results))
    for item in results:
        if item.use_grounding:
            summary.grounded_cases += 1
            summary.sentence_total += item.hallucination.sentence_total
            summary.sentence_with_numbers += item.hallucination.sentence_with_numbers
            summary.sentence_flagged += item.hallucination.sentence_flagged
            summary.number_total += item.hallucination.number_total
            summary.number_traceable += item.hallucination.number_traceable
            summary.claim_total += len(item.judge.claims)
            summary.claim_ungrounded += item.judge.ungrounded_count
        else:
            summary.out_of_scope_cases += 1

        if item.judge.error:
            summary.judge_errors += 1
        else:
            summary.judged += 1
            if item.judge.task_done:
                summary.task_done_count += 1
            summary.intent_sum += item.judge.intent_score
            summary.completeness_sum += item.judge.completeness_score
            summary.relevance_sum += item.judge.relevance_score

        summary.total_tokens += int((item.usage or {}).get("total_tokens", 0) or 0)
        summary.total_elapsed_ms += float(item.elapsed_ms or 0.0)
    return summary


def collect_failures(results: list[CaseEvalResult]) -> list[tuple[str, CaseEvalResult, list[dict]]]:
    """挑出「不通过」的用例，返回 [(原因, 结果, 相关的 Judge claims), ...]。

    判定顺序按严重程度：Agent 崩溃 > Judge 解析失败 > 有编造数字 >
    Judge 判出无依据声明 > 任务未完成。
    先报最严重的，人工复核时才有优先级。
    """
    failures: list[tuple[str, CaseEvalResult, list[dict]]] = []
    for item in results:
        if not item.ok:
            failures.append(("Agent 未正常返回", item, []))
        if item.judge.error:
            failures.append((f"Judge 评审失败：{item.judge.error}", item, []))
        if item.use_grounding and item.hallucination.sentence_flagged > 0:
            failures.append(
                (f"有 {item.hallucination.sentence_flagged} 句含无法溯源的数字", item, [])
            )
        if item.use_grounding and not item.judge.error and item.judge.ungrounded_count > 0:
            ungrounded = [c for c in item.judge.claims if not c["grounded"]]
            failures.append(
                (f"Judge 判出 {item.judge.ungrounded_count} 条无依据声明", item, ungrounded)
            )
        if not item.judge.error and not item.judge.task_done:
            failures.append(("Judge 判定任务未完成（task_done=false）", item, []))
    return failures


# ===========================================================================
# 二、主流程
# ===========================================================================

# 不传 --tag 时用的默认轮次标识。刻意保留 Phase3 这个默认值，
# 是为了让"没有 tag"的老用法仍然指向同一组文件名 —— 不制造惊喜。
DEFAULT_TAG = "Phase3"


def artifact_paths(raw_dir: Path, report_dir: Path, tag: str) -> tuple[Path, Path, Path]:
    """按轮次标识拼出三份产物的路径，返回 (逐句 CSV, 逐用例 CSV, 文本报告)。

    为什么把命名收进一个函数？因为"覆写历史留档"这个坑的根因就是
    **文件名散落在代码里、各自写死**。收成一处之后，
    以后再加产物（比如逐轮对比表）也只会从这一个地方取名，
    不会再出现"漏改一个"的情况。

    ★ 为什么目录也作为参数传进来，而不是在函数里写死 "docs/03_评测报告"？
      因为「命名规则」和「目录归属」是两件会**各自独立变化**的事：
      命名规则是本项目的业务约束（轮次必须进文件名），
      目录归属只是文件摆放习惯，将来再挪位置不该逼着改命名逻辑和它的测试。
      参数化之后，测试也就不必依赖磁盘上真实存在这些目录。

    ★ 为什么 tag 直接拼进文件名、不做校验？
      因为这是内部评测脚本，tag 由使用者自己起名（如 Phase5_基线）。
      做白名单反而会在想记"临时复跑"时碍事。只做一件事：
      把非法路径字符换掉，避免有人手滑写出 `--tag ../x` 这种把文件写到别处的调用。
    """
    safe = "".join("_" if ch in '\\/:*?"<>|' else ch for ch in tag).strip() or DEFAULT_TAG
    return (
        raw_dir / f"幻觉检测_{safe}_逐句.csv",
        raw_dir / f"幻觉与质量_{safe}_逐用例.csv",
        report_dir / f"幻觉与质量测试报告_{safe}.txt",
    )


def _mark(value: float | None, target: float, higher_is_better: bool = True) -> str:
    """对照合格线打标。value 为 None 时表示"本批次没有可测样本"，不参与判定。

    为什么显式处理 None？因为「没有可测用例」和「测了但没通过」是两件不同的事 ——
    跑 --limit 2 这种小批量时，若把"不适用"也标成不合格，报告会满屏假警报。
    """
    if value is None:
        return "不适用"
    ok = value >= target if higher_is_better else value <= target
    return "合格" if ok else "★不合格"


def _show(rate: float) -> str:
    return format_rate(rate)


def _write_csv(path: Path, rows: list[dict]) -> None:
    """写 CSV。encoding 用 utf-8-sig：带 BOM 才能被 Excel 正确识别中文。

    空列表直接跳过：DictWriter 需要从第一行取列名，没有行就没有列名 ——
    这里选择"不生成文件"，而不是硬编一堆空列名（后者反而会误导读者）。
    """
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    # Windows 控制台默认 GBK，直接打印中文/emoji 会炸，这里强制 UTF-8。
    #
    # 【为什么包在 main() 里，而不像兄弟脚本那样放在模块顶层？】
    #   放在顶层的话，import 这个模块就会替换掉 pytest 的 stdout，
    #   导致测试里看不到输出（甚至报错）。为了让「编排层纯逻辑」
    #   能被 tests/test_hallucination.py 直接 import 复用，
    #   这里把副作用收进 main() —— 这是对兄弟脚本骨架的有意偏离。
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    limit: int | None = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    # 轮次标识：决定产物文件名（见模块顶部的「为什么必须有 --tag」）。
    # 沿用脚本里既有的手写 sys.argv 解析风格，不为了一个参数引入 argparse ——
    # 兄弟脚本都是这个风格，保持一致比"更规范"更重要。
    #
    # ★ 变量名叫 round_tag 而不是 tag：下面逐例打印状态时已经有一个局部变量叫
    #   `tag`（"拒答" / "句=… 数字=…"），同名会被循环覆盖掉 ——
    #   实测第一版就是这么写的，产物文件名变成了「幻觉与质量测试报告_拒答.txt」。
    round_tag = DEFAULT_TAG
    if "--tag" in sys.argv:
        round_tag = sys.argv[sys.argv.index("--tag") + 1]
    else:
        # 不传 tag 也能跑（保持老用法可用），但必须把风险喊出来 ——
        # 这个脚本已经因为"文件名写死"覆写过一次历史留档了（踩坑 13）。
        print(f"⚠ 未指定 --tag，产物将写入默认轮次「{DEFAULT_TAG}」的文件名；"
              f"如果这是一次重跑，请用 --tag <轮次标识> 区分，避免覆盖历史留档。")

    eval_set = load_eval_set()
    cases = eval_set.all()
    if limit:
        cases = cases[:limit]

    # runner 自带 agent（同一实例），避免重复构造 LLMClient。
    runner = EvalRunner(eval_set=eval_set)
    agent = runner.agent
    judge = LLMJudge()

    print(f"用例数：{len(cases)}    供应商顺序：{getattr(agent.llm_client, 'provider_order', '?')}")
    print("=" * 88)

    results: list[CaseEvalResult] = []
    for index, case in enumerate(cases, start=1):
        item = evaluate_case(case, agent, judge, runner)
        results.append(item)
        tag = "拒答" if case.expect_no_query else (
            f"句={item.hallucination.sentence_flagged}/{item.hallucination.sentence_total}"
            f" 数字={item.hallucination.number_traceable}/{item.hallucination.number_total}"
            f" 无依据={item.judge.ungrounded_count}"
        )
        print(
            f"[{index:>2}/{len(cases)}] {case.case_id:<4} "
            f"{'✓' if item.ok else '✗'} {tag:<34} "
            f"{item.elapsed_ms:>7.0f}ms  {case.question[:30]}"
        )

    if not results:
        print("没有可评测的用例。")
        return 0

    summary = summarize_cases(results)
    failures = collect_failures(results)
    # 一致性守卫本轮改判了几条（reason 说无依据、grounded 却填 true）。
    # 单独统计是因为它是**评测台自身的健康指标**：>0 说明提示词里那条硬要求
    # 仍被违反、只是被解析层兜住了 —— 兜住不等于消失，必须打印出来。
    consistency_fixed_total = sum(item.judge.consistency_fixed for item in results)

    # 产物按「谁读」分两处落盘：逐句/逐用例 CSV 是复核用的原始证据，
    # 汇总报告 txt 是人读的结论。目录常量统一取自 config。
    raw_dir = cfg.DOCS_RAWDATA_DIR
    report_dir = cfg.DOCS_REPORT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # CSV 1：逐句明细
    # -----------------------------------------------------------------
    sentence_rows: list[dict] = []
    for item in results:
        claim_by_index = {c["index"]: c for c in item.judge.claims}
        for sentence in item.sentences:
            claim = claim_by_index.get(sentence.index)
            if not item.use_grounding:
                grounded = "—"
            elif claim is None:
                grounded = ""          # Judge 没给这一句的结论
            else:
                grounded = "是" if claim["grounded"] else "否"
            sentence_rows.append(
                {
                    # 轮次放在第一列：这几份 CSV 最常见的用法是把多轮拼起来做透视，
                    # 有了这一列才不会把两轮的数据混成一轮（文件名只保护单文件，
                    # 保护不了"复制粘贴到一起"这一步）。
                    "轮次": round_tag,
                    "case_id": item.case_id,
                    "句子序号": sentence.index,
                    "句子文本": sentence.text,
                    "数字": "、".join(str(v) for v in sentence.numbers) or "-",
                    "数字个数": len(sentence.numbers),
                    "可溯源个数": sentence.traceable,
                    "是否含数字": "是" if sentence.has_data_number else "否",
                    "是否可溯源": "否" if sentence.unexplained else "是",
                    "无法溯源数字": "、".join(sentence.unexplained) or "-",
                    "Judge是否grounded": grounded,
                    "Judge理由": (claim or {}).get("reason", ""),
                }
            )

    # -----------------------------------------------------------------
    # CSV 2：逐用例明细
    # -----------------------------------------------------------------
    case_rows: list[dict] = []
    for item in results:
        case_rows.append(
            {
                "轮次": round_tag,
                "case_id": item.case_id,
                "category": item.category,
                "question": item.question,
                "expect_no_query": "是" if item.expect_no_query else "否",
                "ok": "是" if item.ok else "否",
                "reference_row_count": item.reference_row_count,
                "sentence_total": item.hallucination.sentence_total,
                "sentence_with_numbers": item.hallucination.sentence_with_numbers,
                "sentence_flagged": item.hallucination.sentence_flagged,
                "hallucination_rate": item.hallucination.hallucination_rate,
                "number_total": item.hallucination.number_total,
                "number_traceable": item.hallucination.number_traceable,
                "data_citation_accuracy": item.hallucination.data_citation_accuracy,
                "claims_total": len(item.judge.claims),
                "ungrounded_count": item.judge.ungrounded_count,
                "ungrounded_rate": item.judge.ungrounded_rate,
                "task_done": "是" if item.judge.task_done else "否",
                "intent_score": item.judge.intent_score,
                "completeness_score": item.judge.completeness_score,
                "relevance_score": item.judge.relevance_score,
                "judge_error": item.judge.error or "",
                "iterations": item.iterations,
                "total_tokens": (item.usage or {}).get("total_tokens", 0),
                "elapsed_ms": round(item.elapsed_ms, 1),
                "error": item.error or "",
            }
        )

    sentence_csv, case_csv, report_path = artifact_paths(raw_dir, report_dir, round_tag)
    _write_csv(sentence_csv, sentence_rows)
    _write_csv(case_csv, case_rows)

    # -----------------------------------------------------------------
    # 文本报告
    # -----------------------------------------------------------------
    lines: list[str] = []
    add = lines.append
    line = "=" * 88

    add(line)
    add("幻觉检测 + 端到端质量评测报告（测试方案 Phase 3 · 维度三 + 维度四）")
    add(line)
    # 轮次标识写进报告正文：文件名叫什么、报告自己也要说一遍。
    # 踩坑 13 的教训是"临时文件名 + 不在清单里 = 等于丢了" ——
    # 让产物**自我说明**是最便宜的一道保险：报告被单独发出去时也知道是哪一轮。
    add(f"轮次标识：{round_tag}")
    add(f"用例总数：{summary.total}"
        f"（有真值 {summary.grounded_cases} 条 / 超范围拒答 {summary.out_of_scope_cases} 条）")
    add(f"供应商：{getattr(agent.llm_client, 'provider_order', '?')}")
    add(f"平均耗时：{summary.avg_elapsed_ms:,.0f} ms    平均 token：{summary.avg_tokens:,.1f}"
        f"    总 token：{summary.total_tokens:,}")
    add("")
    add("  说明：以上 token 只统计 Agent 侧；Judge 的额外调用未计入（接口未回传用量）。")
    add("")

    # ---------------- 一、幻觉检测 ----------------
    add("-" * 88)
    add("一、幻觉检测（维度三）")
    add("-" * 88)
    add(f"  ① 幻觉率（句子级）        {_show(summary.hallucination_rate):>8}  "
        f"（{summary.sentence_flagged}/{summary.sentence_total} 句）  标准 ≤5%   "
        f"{_mark(summary.hallucination_rate, TARGET_HALLUCINATION, False)}")
    add(f"  ② 数据引用准确率（数字级）{_show(summary.data_citation_accuracy):>8}  "
        f"（{summary.number_traceable}/{summary.number_total} 个数字）  标准 ≥95%   "
        f"{_mark(summary.data_citation_accuracy, TARGET_CITATION)}")
    add(f"  ③ 无依据声明率（Judge）   {_show(summary.ungrounded_rate):>8}  "
        f"（{summary.claim_ungrounded}/{summary.claim_total} 条声明）  标准 ≤10%   "
        f"{_mark(summary.ungrounded_rate, TARGET_UNGROUNDED, False)}")
    add("")
    add(f"  含数字的句子 {summary.sentence_with_numbers} 句 / 总句数 {summary.sentence_total} 句"
        f"；不含数字的句子天然不可能产生幻觉，不计入幻觉率分子。")
    add("")

    # ---------------- 二、端到端质量 ----------------
    add("-" * 88)
    add("二、端到端质量（维度四）")
    add("-" * 88)
    add(f"  ① 任务完成率        {_show(summary.task_done_rate):>8}  "
        f"（{summary.task_done_count}/{summary.judged}）  标准 ≥80%   "
        f"{_mark(summary.task_done_rate, TARGET_TASK_DONE)}")
    add(f"  ② 意图解析（1-5）   {summary.avg_intent:>8.2f}  标准 ≥4.0   "
        f"{_mark(summary.avg_intent, TARGET_INTENT)}")
    add(f"  ③ 答案完整性（1-5） {summary.avg_completeness:>8.2f}  标准 ≥3.5   "
        f"{_mark(summary.avg_completeness, TARGET_COMPLETENESS)}")
    add(f"  ④ 答案相关性（1-5） {summary.avg_relevance:>8.2f}  标准 ≥4.0   "
        f"{_mark(summary.avg_relevance, TARGET_RELEVANCE)}")
    if summary.judge_errors:
        add(f"  ⚠ Judge 评审失败 {summary.judge_errors} 条，未计入质量分分母。")
    add("")

    # ---------------- 三、口径说明 ----------------
    add("-" * 88)
    add("三、口径说明")
    add("-" * 88)
    add("  1. 两个幻觉口径的区别")
    add("     · 幻觉率（句子级）= 含无法溯源数字的句子数 / 总句数。")
    add("       句子是「结论」的载体：一句里出现编造数字，整句结论就不可用。")
    add("       没有数字的句子（如「建议关注留存变化」）不进分子 —— 它无从编造数字。")
    add("     · 数据引用准确率（数字级）= 可溯源数字数 / 数字总数。")
    add("       衡量引用质量：所有数字里有多大比例是真的。")
    add("     两者会背离，必须同时看：一句编造塞 10 个假数字，数字级掉得厉害、")
    add("     句子级只算 1 句；30 个真数字错 1 个，数字级几乎不掉、句子级亮 1 句。")
    add("")
    add("  2. Judge 的判据")
    add("     · 评审员只能依据【真值数据】判断，不得引入任何外部知识；")
    add("       真值里没有的数字、对比、归因、因果，一律判为「无依据」。")
    add("     · 刻意区分「数字编造」与「合理的方法论 / 口径说明」——")
    add("       后者（如「留存率的分母剔除了观察期不足的用户」）不需要数据支撑，不算幻觉。")
    add("       不区分的话，评测会变成惩罚专业性。")
    add("     · 真值走**不经过 LLM 的独立路径**（runner.build_reference 直查数据库），")
    add("       所以不存在「Agent 查错了范围、再用自己的数据自证」的问题。")
    add("     · 解析层还有一道「一致性守卫」：若 Judge 的 reason 写了「与真值矛盾」")
    add("       「真值中无此数据」这类判断、而 grounded 却填了 true，会被自动改判为")
    add("       false 并在 reason 前打上 [一致性修正] 标记。这是从宽方向漏报的补丁 ——")
    add("       判词已经认定有问题，字段不该放过它。改判条数见下方「评测台健康度」。")
    add("")
    add("  3. 为什么没用 BERT / NLI 三信号方案？")
    add("     学术界的幻觉检测常用「NLI 蕴含 + 语义相似度 + 事实核查」三信号融合，")
    add("     但它要求 GPU 与数百 MB 预训练模型，中文还要额外选型。")
    add("     本项目是求职作品，目标是「一条命令跑起来」，引入这套依赖得不偿失。")
    add("     改用 LLM-as-a-Judge 做替代，并靠「真值独立路径」堵住自证风险 ——")
    add("     在这个规模下性价比明显更高。这是有意识的取舍，不是没考虑过。")
    add("")
    add("  4. 评测台健康度（这一节不评价 Agent，只评价评测台自己）")
    add(f"     · 一致性守卫改判条数：{consistency_fixed_total}")
    if consistency_fixed_total:
        add("       ⚠ 大于 0 说明 Judge 仍在违反「reason 与 grounded 必须自洽」这条硬要求，")
        add("         只是被解析层兜住了。**兜住不等于消失** —— 下一轮该盯的是这个数归零。")
    else:
        add("       0 表示本轮 Judge 的判词与字段完全自洽，没有需要兜住的条目。")
    add("")

    # ---------------- 四、不通过用例明细 ----------------
    add("-" * 88)
    add("四、不通过用例明细")
    add("-" * 88)
    if not failures:
        add("  无。全部用例在幻觉、无依据声明、任务完成三项上均通过。")
    for reason, item, claims in failures:
        add(f"  [{item.case_id}] {reason}")
        add(f"       问：{item.question}")
        add(f"       答：{item.answer[:120].replace(chr(10), ' ')}")
        if item.hallucination.sentence_flagged:
            flagged = [s for s in item.sentences if s.unexplained]
            for sentence in flagged:
                add(f"       ⚠ 第 {sentence.index} 句含无法溯源数字 "
                    f"{'、'.join(sentence.unexplained)}：{sentence.text[:60]}")
        for claim in claims:
            add(f"       ✗ Judge 第 {claim['index']} 句判无依据：{claim['reason'][:80]}")
        add("")

    # ---------------- 五、逐用例明细 ----------------
    add("-" * 88)
    add(f"五、逐用例明细（CSV 见 docs/{cfg.DOCS_RAWDATA_DIR.name}/{case_csv.name}）")
    add("-" * 88)
    add(f"  {'用例':<5}{'类型':<7}{'句数':<5}{'标红':<5}{'数字':<8}{'无依据':<7}"
        f"{'完成':<5}{'意图':<5}{'完整':<5}{'相关':<5}{'轮次':<5}提问")
    for item in results:
        kind = "拒答" if item.expect_no_query else item.category
        add(
            f"  {item.case_id:<5}{kind:<7}"
            f"{item.hallucination.sentence_total:<5}"
            f"{item.hallucination.sentence_flagged:<5}"
            f"{f'{item.hallucination.number_traceable}/{item.hallucination.number_total}':<8}"
            f"{item.judge.ungrounded_count:<7}"
            f"{'是' if item.judge.task_done else '否':<5}"
            f"{item.judge.intent_score:<5.0f}{item.judge.completeness_score:<5.0f}"
            f"{item.judge.relevance_score:<5.0f}{item.iterations:<5}{item.question[:24]}"
        )
    add("")
    add(line)
    add("报告结束。边界：数字级对账只能证明「数字有出处」，Judge 只能证明")
    add("「在给定真值下没有超出数据说话」—— 两者都无法替代人工对业务结论的最终判断。")
    add(line)

    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"\nCSV  → {sentence_csv}")
    print(f"CSV  → {case_csv}")
    print(f"报告 → {report_path}")

    # 有不合格项时返回 1，方便以后接 CI（"不适用"的指标不参与判定）
    checks = [
        _mark(summary.hallucination_rate, TARGET_HALLUCINATION, False),
        _mark(summary.data_citation_accuracy, TARGET_CITATION),
        _mark(summary.ungrounded_rate, TARGET_UNGROUNDED, False),
        _mark(summary.task_done_rate, TARGET_TASK_DONE),
        _mark(summary.avg_intent, TARGET_INTENT),
        _mark(summary.avg_completeness, TARGET_COMPLETENESS),
        _mark(summary.avg_relevance, TARGET_RELEVANCE),
    ]
    return 0 if all(item != "★不合格" for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
