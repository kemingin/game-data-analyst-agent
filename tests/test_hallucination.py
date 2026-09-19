# -*- coding: utf-8 -*-
"""
幻觉检测 + Judge + 编排层单元测试（测试方案 Phase 3）
================================================================================
【为什么这些测试必须"零网络、零成本"？】

  这是本项目踩过的坑：测试里忘了注入替身对象，结果真的跑起了真机 ——
  从 answer.usage["total_tokens"] 露馅（本该是 0，却是个几千的数）。
  评测代码一旦真的联网，测试就从"秒级、免费、可重复"变成了
  "分钟级、烧钱、还受网络波动影响"，CI 里根本没法用。

  所以本文件的两条纪律：
    1. 任何会调 LLM 的对象（LLMJudge）都必须注入 FakeClient；
    2. 任何会跑 Agent 的地方都必须注入 FakeAgent。
  下面每条断言都在验证"程序算得对不对"，而不是"模型答得好不好"。
  跑 Agent 真机是 scripts/run_hallucination_eval.py 的职责，不是测试的职责。
"""

from __future__ import annotations

import json

from src.agent.llm_client import LLMResponse
from src.agent.react_agent import AgentAnswer, AgentStep
from src.eval.eval_set import load_eval_set
from src.eval.hallucination import (
    SentenceCheck,
    build_sentence_report,
    check_sentence_numbers,
    split_sentences,
    summarize_hallucination,
)
from src.eval.judge import JudgeVerdict, LLMJudge, _extract_json
from src.eval.metrics import metric_calls
from src.eval.runner import EvalRunner, _flatten_rows
from scripts.run_hallucination_eval import (
    _OUT_OF_SCOPE_REFERENCE,
    CaseEvalResult,
    artifact_paths,
    build_full_reference_text,
    build_reference_text,
    collect_failures,
    evaluate_case,
    summarize_cases,
)

# ===========================================================================
# 一、测试替身（保证零网络：所有 LLM 交互都走这里）
# ===========================================================================

class FakeClient:
    """假 LLM 客户端：返回预设文本，或按需抛异常。绝不联网。"""

    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[list[dict]] = []      # 记录每次调用，供断言"确实只调了一次"

    def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content, total_tokens=123, provider="fake")


class FakeJudge:
    """假评审员：返回预设结论，记录调用参数。"""

    def __init__(self, verdict: JudgeVerdict | None = None) -> None:
        self.verdict = verdict or JudgeVerdict(
            task_done=True, intent_score=5, completeness_score=4, relevance_score=5
        )
        self.calls: list[tuple[str, str, str]] = []

    def judge_case(self, question: str, reference_text: str, answer: str) -> JudgeVerdict:
        self.calls.append((question, reference_text, answer))
        return self.verdict


class FakeAgent:
    """假 Agent：按问题返回预先编好的答案。"""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def ask(self, question: str, history: list | None = None):
        self.asked.append(question)
        item = self.answers.get(question)
        if isinstance(item, Exception):
            raise item
        return item


def make_query_step(metric_id: str, params: dict, rows: list[dict] | None = None) -> AgentStep:
    """造一个 query_metric 步骤（result 里带已解析参数，与真实结构一致）。"""
    return AgentStep(
        iteration=1,
        kind="tool_call",
        tool_name="query_metric",
        arguments={"metric_id": metric_id, "params": params},
        result={
            "metric_id": metric_id,
            "params": params,
            "rows": rows or [],
            "columns": list((rows or [{}])[0].keys()),
        },
        ok=True,
    )


# ===========================================================================
# 二、分句
# ===========================================================================

def test_split_sentences_splits_chinese_and_english_punctuation():
    """中英文句末标点都要切 —— 模型经常中英混排，只切中文会把两句粘一起。"""
    text = "日活是 465 人。留存率 42.66%？确实偏高！ARPU 8.46 元;最后一句"
    assert split_sentences(text) == [
        "日活是 465 人",
        "留存率 42.66%",
        "确实偏高",
        "ARPU 8.46 元",
        "最后一句",
    ]


def test_split_sentences_splits_on_newlines():
    assert split_sentences("第一行没有标点\n第二行也是") == ["第一行没有标点", "第二行也是"]


def test_split_sentences_keeps_markdown_table_row_intact():
    """Markdown 表格行必须整体成一句。

    表格是一行行数据，按标点/竖线切会把一行拆成碎片，
    既丢失单元格之间的对应关系，也会让"句子级"口径失真。
    """
    table = "| 09-05 | 99 人 |\n| 09-06 | 54 人 |"
    assert split_sentences(table) == ["| 09-05 | 99 人 |", "| 09-06 | 54 人 |"]


def test_split_sentences_mixes_table_and_prose():
    """正文按标点切、表格整行保留，两种规则要能共存。"""
    text = "逐日数据如下。\n| 日期 | 日活 |\n| 09-05 | 99 |\n整体波动不大。"
    assert split_sentences(text) == [
        "逐日数据如下",
        "| 日期 | 日活 |",
        "| 09-05 | 99 |",
        "整体波动不大",
    ]


def test_split_sentences_empty_input():
    assert split_sentences("") == []
    assert split_sentences(None) == []
    assert split_sentences("   \n  \n") == []


def test_split_sentences_filters_symbol_only_fragments():
    """纯符号/纯空白碎片要丢掉 —— 否则分母虚高，幻觉率被稀释。"""
    assert split_sentences("。。。！！！") == []
    assert split_sentences("---") == []
    assert split_sentences("正常一句话。！！！") == ["正常一句话"]


def test_split_sentences_strips_whitespace():
    assert split_sentences("  前面有空格 。  后面也有空格  ") == ["前面有空格", "后面也有空格"]


# ===========================================================================
# 三、逐句对账与聚合指标
# ===========================================================================

def test_check_sentence_numbers_flags_fabricated_number():
    """真值里是 465，回答说 999 → 该句必须被标红。"""
    check = check_sentence_numbers("日活是 999 人。", [465.0])

    assert check.flagged is True
    assert check.unexplained == ["999"]
    assert check.has_data_number is True
    assert check.traceable == 0


def test_check_sentence_numbers_marks_traceable_number():
    check = check_sentence_numbers("日活是 465 人。", [465.0])

    assert check.flagged is False
    assert check.traceable == 1
    assert check.numbers == [465.0]


def test_check_sentence_numbers_sentence_without_numbers_not_flagged():
    """没有数字的句子不该被判为幻觉。

    这条很关键：「建议关注留存变化」这种纯建议句天然没有出处问题，
    把它算进幻觉率的分子是荒谬的 —— 也是本模块区分 has_data_number 的原因。
    """
    check = check_sentence_numbers("建议关注留存变化。", [465.0])

    assert check.has_data_number is False
    assert check.flagged is False
    assert check.numbers == []


def test_check_sentence_numbers_honours_extra_allowed():
    """期望参数值、行数这类"合法但不在数据行里"的数字要能放行。

    放行靠的是 metrics.check_answer_numbers 的 extra_allowed 参数（复用同一套规则），
    不放行的话「样本量 7 人」会被误报成编造。
    """
    without = check_sentence_numbers("样本量 7 人，日活 465 人。", [465.0])
    assert without.unexplained == ["7"]

    with_extra = check_sentence_numbers("样本量 7 人，日活 465 人。", [465.0], extra_allowed=[7])
    assert with_extra.unexplained == []
    assert with_extra.traceable == 2


def test_summarize_hallucination_counts_sentences_and_numbers():
    checks, summary = build_sentence_report(
        "日活 465 人。另有 999 人。建议关注留存变化。", [465.0]
    )

    assert [c.index for c in checks] == [1, 2, 3]
    assert summary.sentence_total == 3
    assert summary.sentence_with_numbers == 2
    assert summary.sentence_flagged == 1
    assert summary.number_total == 2
    assert summary.number_traceable == 1
    assert summary.hallucination_rate == round(1 / 3, 4)
    assert summary.data_citation_accuracy == 0.5


def test_hallucination_rate_is_zero_without_sentences():
    """句子数为 0 时返回 0.0，而不是除零崩溃。"""
    summary = summarize_hallucination([])
    assert summary.hallucination_rate == 0.0
    assert summary.data_citation_accuracy == 0.0


def test_data_citation_accuracy_is_zero_without_numbers():
    """有句子但一个数字都没有时，数字级准确率同样是 0.0（不崩溃）。"""
    checks = [SentenceCheck(index=1, text="建议关注留存变化", has_data_number=False)]
    summary = summarize_hallucination(checks)

    assert summary.sentence_total == 1
    assert summary.number_total == 0
    assert summary.data_citation_accuracy == 0.0
    assert summary.hallucination_rate == 0.0


def test_hallucination_rate_counts_only_flagged_sentences():
    """幻觉率只统计"含编造数字的句子"，没有数字的句子不进分子。"""
    checks = [
        SentenceCheck(index=1, text="日活 465 人", numbers=[465.0], traceable=1, has_data_number=True),
        SentenceCheck(index=2, text="建议关注留存变化", has_data_number=False),
    ]
    summary = summarize_hallucination(checks)

    assert summary.sentence_total == 2
    assert summary.sentence_flagged == 0
    assert summary.hallucination_rate == 0.0


def test_build_sentence_report_numbers_sentences_from_one():
    """句子序号从 1 开始 —— 报告和 Judge 的 prompt 都给人/模型看，1 起始更自然。"""
    checks, _ = build_sentence_report("第一句。第二句。", [])
    assert [c.index for c in checks] == [1, 2]


def test_build_sentence_report_flags_fabrication_end_to_end():
    checks, summary = build_sentence_report("日活 465 人。另有 999 人。", [465.0])

    assert summary.sentence_flagged == 1
    assert [c.index for c in checks if c.flagged] == [2]
    assert checks[1].unexplained == ["999"]


# ===========================================================================
# 四、Judge 的容错 JSON 解析
# ===========================================================================

def test_extract_json_plain_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_fenced_block():
    """模型习惯性套 ```json 围栏，必须能剥掉。"""
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json("```\n{\"a\": 1}\n```") == {"a": 1}


def test_extract_json_with_surrounding_text():
    """前后带解释性文字时，退化成"第一个 { 到最后一个 }"。"""
    text = "好的，以下是评审结果：\n{\"a\": 1, \"b\": 2}\n希望有帮助。"
    assert _extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_top_level_array():
    """顶层是数组也要能解析（模型有时直接返回 claims 数组）。"""
    assert _extract_json('[{"index": 1}]') == [{"index": 1}]


def test_extract_json_repairs_trailing_comma():
    """尾逗号是 JSON 标准里的非法写法，但模型很爱写 —— 只修这一种瑕疵。"""
    assert _extract_json('{"claims": [{"index": 1},]}') == {"claims": [{"index": 1}]}


def test_extract_json_invalid_returns_none():
    assert _extract_json("这不是 JSON") is None
    assert _extract_json("") is None
    assert _extract_json(None) is None


# ===========================================================================
# 五、LLMJudge（全部注入 FakeClient，零网络）
# ===========================================================================

def test_judge_case_parses_verdict():
    payload = json.dumps(
        {
            "claims": [
                {"index": 1, "text": "日活 465 人", "grounded": True, "reason": "真值有 465"},
                {"index": 2, "text": "另有 999 人", "grounded": False, "reason": "真值中无此数据"},
            ],
            "task_done": True,
            "intent_score": 5,
            "completeness_score": 4,
            "relevance_score": 5,
        },
        ensure_ascii=False,
    )
    client = FakeClient(payload)
    verdict = LLMJudge(client=client).judge_case("最近日活", "列：dau\n465", "日活 465 人。另有 999 人。")

    assert verdict.error is None
    assert len(verdict.claims) == 2
    assert verdict.ungrounded_count == 1
    assert verdict.ungrounded_rate == 0.5
    assert verdict.task_done is True
    assert (verdict.intent_score, verdict.completeness_score, verdict.relevance_score) == (5.0, 4.0, 5.0)
    # 只调一次 LLM：句子核验与打分合并成一次调用（省 token 的关键设计）
    assert len(client.calls) == 1
    assert client.calls[0][0]["role"] == "system"
    assert "评审员" in client.calls[0][0]["content"]


def test_judge_case_all_grounded_rate_zero():
    payload = json.dumps(
        {"claims": [{"index": 1, "grounded": True, "reason": "口径说明"}], "task_done": True},
        ensure_ascii=False,
    )
    verdict = LLMJudge(client=FakeClient(payload)).judge_case("q", "真值", "答")

    assert verdict.ungrounded_count == 0
    assert verdict.ungrounded_rate == 0.0


def test_judge_case_garbage_returns_error():
    """解析失败不抛异常：返回 error 且分数为 0（工具失败返回可观察结果）。"""
    verdict = LLMJudge(client=FakeClient("抱歉，我无法完成这次评审。")).judge_case("q", "真值", "答")

    assert verdict.error is not None
    assert verdict.claims == []
    assert verdict.intent_score == 0.0
    assert verdict.task_done is False


def test_judge_case_llm_exception_returns_error():
    """LLM 调用本身报错也要兜住 —— 否则一条评审失败会毁掉整轮评测。"""
    verdict = LLMJudge(client=FakeClient(error=RuntimeError("网络断了"))).judge_case("q", "真值", "答")

    assert verdict.error is not None
    assert "网络断了" in verdict.error
    assert verdict.intent_score == 0.0


def test_judge_case_clamps_scores():
    """分数要夹到 1~5：模型给 7 分或 0 分都不该把平均分带偏。"""
    payload = json.dumps(
        {"claims": [], "task_done": True, "intent_score": 7, "completeness_score": 0,
         "relevance_score": "abc"},
        ensure_ascii=False,
    )
    verdict = LLMJudge(client=FakeClient(payload)).judge_case("q", "真值", "答")

    assert verdict.intent_score == 5.0
    assert verdict.completeness_score == 1.0
    assert verdict.relevance_score == 0.0        # 取不到数值 = 没给分


def test_judge_case_handles_array_payload_and_missing_grounded():
    """顶层是数组、且某条漏了 grounded 字段时，不主动指控为幻觉。"""
    payload = '[{"index": 1, "reason": "口径说明"}, {"index": 2, "grounded": false, "reason": "无数据"}]'
    verdict = LLMJudge(client=FakeClient(payload)).judge_case("q", "真值", "答")

    assert verdict.error is None
    assert len(verdict.claims) == 2
    assert verdict.claims[0]["grounded"] is True      # 缺字段 → 保守地判为有依据
    assert verdict.ungrounded_count == 1


def test_judge_case_fixes_reason_grounded_contradiction():
    """Phase 5 发现 5 的修复：reason 说「与真值矛盾」时，grounded 不能是 true。

    背景：Phase 5 的 A 轮里，E08 第 4 句的判词写「该描述与真值矛盾」，
    但 grounded 字段填了 true（按判据第 3 类本应判 false）。
    这是**从宽方向**的漏报 —— 判词已经认定有问题，字段却放过了它，
    会让"无依据声明率"偏低，属于评测台自身的缺陷。

    修法是解析层加一道一致性守卫：以 reason 为准翻成 false，并在 reason 上留痕。
    ★ 留痕这一步不能省：静默改判会让报告"看起来更准了，但没人知道它被改过"，
      那和篡改数据没有区别。所以测试里专门断言 [一致性修正] 标记存在。
    """
    payload = json.dumps(
        {
            "claims": [
                {"index": 1, "grounded": True, "reason": "该描述与真值矛盾"},
                {"index": 2, "grounded": True, "reason": "真值中无此数据"},
                {"index": 3, "grounded": True, "reason": "被真值中 465 支撑"},
                {"index": 4, "grounded": True, "reason": "方法论/口径说明，无需数据支撑"},
            ],
            "task_done": True,
        },
        ensure_ascii=False,
    )
    verdict = LLMJudge(client=FakeClient(payload)).judge_case("q", "真值", "答")

    assert verdict.consistency_fixed == 2            # 只有前两条自相矛盾
    assert verdict.claims[0]["grounded"] is False
    assert verdict.claims[1]["grounded"] is False
    assert "[一致性修正" in verdict.claims[0]["reason"]   # 改判必须留痕
    assert verdict.claims[2]["grounded"] is True     # 有真值支撑的句子不受影响
    assert verdict.claims[3]["grounded"] is True     # 口径说明类不受影响
    assert verdict.ungrounded_count == 2             # 计数跟着改判后的结果走


def test_artifact_paths_carry_round_tag():
    """产物文件名必须带轮次标识 —— 这是踩坑 13「重跑覆写历史留档」的回归测试。

    背景：产物名原先写死成 Phase3，导致 Phase 5 的 B/A 两轮把 Phase 3 的报告
    覆写掉了，第四章引用的数字在磁盘上再也对不上
    （详见 docs/02_测试复盘/测试复盘记录.txt）。
    所以这里断言三件事：① 不同轮次必然落在不同文件上；② tag 里的路径字符被替换掉，
    避免 `--tag ../x` 这种把产物写到目录外的调用；③ CSV 与 txt 分属两个目录
    （原始证据 / 人读结论），不会又混回同一层。

    ★ 为什么传的是拼出来的假目录、而不是真实的 docs 子目录？
      因为被测的是「命名 + 归属」这套规则，不是磁盘上有没有这些目录。
      传假目录，测试就与真实的目录命名解耦 —— 将来 docs 再改名，这个测试不用动。
    """
    from pathlib import Path

    raw_dir = Path("docs") / "04_评测原始数据"
    report_dir = Path("docs") / "03_评测报告"
    sentence_csv, case_csv, report = artifact_paths(raw_dir, report_dir, "Phase5_基线")
    assert sentence_csv.name == "幻觉检测_Phase5_基线_逐句.csv"
    assert case_csv.name == "幻觉与质量_Phase5_基线_逐用例.csv"
    assert report.name == "幻觉与质量测试报告_Phase5_基线.txt"

    # CSV 落在原始数据目录、报告落在报告目录 —— 两类读者的产物不许混层
    assert sentence_csv.parent == raw_dir
    assert case_csv.parent == raw_dir
    assert report.parent == report_dir

    # 换一个轮次标识，三个文件名必须全部跟着变 —— 否则又会出现"悄悄覆盖"
    other = artifact_paths(raw_dir, report_dir, "Phase5_A轮")
    assert {p.name for p in other}.isdisjoint({p.name for p in (sentence_csv, case_csv, report)})

    # 路径分隔符要被替换，防止产物被写到传入目录之外
    assert "/" not in artifact_paths(raw_dir, report_dir, "../x")[2].name
    assert "\\" not in artifact_paths(raw_dir, report_dir, "..\\x")[2].name


# ===========================================================================
# 六、编排层（FakeAgent + FakeJudge，不调真机）
# ===========================================================================

def test_build_reference_text_empty():
    assert "没有可用的真值数据" in build_reference_text([])


def test_build_reference_text_truncates():
    """行数超上限要截断，并**明确写出"已截断、共几行"**。

    不写清楚的话，Judge 会把"真值里没有"误判成"模型编造"，制造假警报。
    """
    rows = [{"day": f"09-{i:02d}", "dau": i} for i in range(1, 101)]
    text = build_reference_text(rows, max_rows=3)

    assert "列：day | dau" in text
    assert text.count("\n") == 4            # 表头 + 3 行 + 截断说明
    assert "真值共 100 行" in text
    assert "已截断" in text


def test_build_full_reference_text_includes_params_definition_and_extra_metrics():
    """给 Judge 的真值必须包含三块：口径参数 + 主真值行 + 模型额外查的指标真值。

    这是 Phase 3 全量跑出「无依据声明率 26.87% 不合格」后定位到的根因：
    只喂主真值行时，回答里的日期区间（来自参数）和模型多查的指标（如 7 日留存、
    ARPPU）在 Judge 眼里全是编造，而数字级对账其实是 98.97%。
    本用例把"三块都要在"钉死，防止以后又退回单块版本。
    """
    text = build_full_reference_text(
        reference_metric_id="stickiness",
        reference_params={"start_date": "2026-08-19", "end_date": "2026-09-17"},
        reference_rows=[{"dau_avg": 468.5, "mau": 4649, "stickiness_pct": 10.08}],
        extra_references=[("arppu", {"start_date": "2026-09-11"}, [{"arppu": 229.0}])],
    )

    assert "【口径与参数】" in text
    assert "2026-08-19" in text                 # 日期区间不出现 → Judge 会判「日期编造」
    assert "指标口径原文" in text                 # 定义里的 10%~25% 是合法出处，Judge 也要看到
    assert "行业参考区间" in text
    assert "stickiness_pct" in text
    assert "【模型额外查询的指标】" in text
    assert "arppu" in text and "229.0" in text


def test_build_full_reference_text_without_extras_still_works():
    """模型没多查指标时，不能出现空的「额外指标」块 —— 空块会诱导 Judge 乱判。"""
    text = build_full_reference_text(
        reference_metric_id="dau",
        reference_params={},
        reference_rows=[{"dau": 465}],
    )
    assert "【参考数据】" in text
    assert "【模型额外查询的指标】" not in text


def test_extra_reference_rows_and_values_share_one_source():
    """行与数值必须来自同一次查询 —— 这是「两把尺子」问题的根治办法。"""
    eval_set = load_eval_set()
    runner = EvalRunner(eval_set=eval_set)
    calls = metric_calls(
        [
            make_query_step("payment_rate", {"start_date": "2026-09-11", "end_date": "2026-09-17"}),
            make_query_step("arppu", {"start_date": "2026-09-11", "end_date": "2026-09-17"}),
        ]
    )

    references = runner.build_extra_reference_rows(calls)
    values = runner.build_extra_reference_values(calls)

    # 每一次成功查询都要留档（主指标也不再被排除，见 runner 里的说明）
    assert [metric_id for metric_id, _, _ in references] == ["payment_rate", "arppu"]
    assert all(rows for _, _, rows in references)          # 真值行确实查到了
    assert values == [v for _, _, rows in references for v in _flatten_rows(rows)]


def test_extra_reference_rows_keeps_same_metric_with_different_params():
    """★ 同指标不同参数必须都保留 —— 这是 Phase 3 修掉的第二个评测台缺陷。

    模型回答「版本留存变好了吗」时会同时查 version_retention 的 day_n=1 与 day_n=7
    做交叉验证。旧实现按 metric_id 整体去重，把 day_n=7 那次的真值静默丢弃，
    导致 Judge 把真实的 7 日留存数据判成编造。
    """
    eval_set = load_eval_set()
    runner = EvalRunner(eval_set=eval_set)
    calls = metric_calls(
        [
            make_query_step("version_retention", {"day_n": 1}),
            make_query_step("version_retention", {"day_n": 7}),
            make_query_step("version_retention", {"day_n": 7}),      # 完全重复的调用
        ]
    )

    references = runner.build_extra_reference_rows(calls)

    assert len(references) == 2                                    # 重复调用被去掉，不同参数被保留
    assert [params.get("day_n") for _, params, _ in references] == [1, 7]


def test_split_sentences_strips_ordered_list_marker():
    """列表序号是排版不是内容：不剥掉会被数字抽取当成数据，凭空造出幻觉。"""
    sentences = split_sentences("2. 对比 D1 / D7 / D30，看用户是在哪个阶段流失的")

    assert sentences == ["对比 D1 / D7 / D30，看用户是在哪个阶段流失的"]
    # 句子里的真实数字不受影响
    assert "D1" in sentences[0] and "D30" in sentences[0]


def test_split_sentences_strips_list_marker_wrapped_in_bold():
    """序号外面套了加粗标记时也要剥掉，且不能留下落单的 `**`。"""
    sentences = split_sentences("**2. 长期人均价值（LTV30）**：7/20 ~ 8/18 共 1,654 人")

    assert sentences == ["**长期人均价值（LTV30）**：7/20 ~ 8/18 共 1,654 人"]
    assert "2" not in sentences[0].split("**")[0]        # 序号确实没了


def test_split_sentences_keeps_decimal_number_at_sentence_start():
    """「3.7% 的付费率」不能被当成列表序号剥掉 —— 点后面没有空白。"""
    sentences = split_sentences("3.7% 的付费率属于偏低水平")

    assert sentences == ["3.7% 的付费率属于偏低水平"]


def test_evaluate_case_feeds_extra_metric_truth_to_judge():
    """端到端：模型多查的指标，其真值必须同时进入数字对账集与 Judge 真值文本。

    E09 问的是付费率，模型顺手多查了 arppu 做交叉验证（真实行为）。
    修复前：arppu 的数字被 Judge 判成编造；修复后：Judge 能看到 arppu 的真值。
    """
    eval_set = load_eval_set()
    case = eval_set.get("E09")
    runner = EvalRunner(eval_set=eval_set)
    answer = AgentAnswer(
        question=case.question,
        answer="最近 7 天付费转化率为 3.70%，付费用户人均付费 229.00 元。",
        ok=True,
        steps=[
            make_query_step("payment_rate", {}),
            make_query_step("arppu", {}),
        ],
        iterations=3,
    )
    judge = FakeJudge()
    result = evaluate_case(case, FakeAgent({case.question: answer}), judge, runner)

    reference_text = judge.calls[0][1]
    assert "【模型额外查询的指标】" in reference_text
    assert "arppu" in reference_text
    # 两个数字都应有出处（229.00 来自额外指标的真值，而不是被判成编造）
    assert result.hallucination.sentence_flagged == 0


def test_evaluate_case_scores_correct_answer():
    """构造"完全答对"的 Agent，验证编排层：真值走独立路径、Judge 拿到真值文本。"""
    eval_set = load_eval_set()
    case = eval_set.get("E04")             # 次日留存，day_n=1
    runner = EvalRunner(eval_set=eval_set)
    rows, _ = runner.build_reference(case)
    rate = rows[0]["retention_rate_pct"]
    cohort = rows[0]["cohort_size"]

    answer = AgentAnswer(
        question=case.question,
        answer=f"次日留存率为 {rate}%，样本量 {cohort} 人。",
        ok=True,
        steps=[make_query_step("retention_rate", {"day_n": 1}, rows=rows)],
        iterations=2,
        usage={"total_tokens": 5000},
        elapsed_ms=1000.0,
    )
    judge = FakeJudge()
    result = evaluate_case(case, FakeAgent({case.question: answer}), judge, runner)

    assert result.use_grounding is True
    assert result.hallucination.sentence_flagged == 0
    assert result.hallucination.number_traceable == result.hallucination.number_total
    assert result.reference_row_count == len(rows)
    # 真值文本必须传给 Judge —— 否则评审员只能靠常识判断，评审就失去意义
    assert len(judge.calls) == 1
    assert str(rate) in judge.calls[0][1]


def test_evaluate_case_flags_fabricated_number():
    eval_set = load_eval_set()
    case = eval_set.get("E04")
    runner = EvalRunner(eval_set=eval_set)

    answer = AgentAnswer(
        question=case.question,
        answer="次日留存率约为 78.9%，样本量 9999 人。",     # 明显编造
        ok=True,
        steps=[make_query_step("retention_rate", {"day_n": 1})],
        iterations=2,
    )
    result = evaluate_case(case, FakeAgent({case.question: answer}), FakeJudge(), runner)

    assert result.use_grounding is True
    assert result.hallucination.sentence_flagged == 1
    assert result.sentences[0].unexplained


def test_evaluate_case_skips_grounding_for_out_of_scope():
    """超范围用例没有真值：跳过 grounded 判定，只让 Judge 打质量分。"""
    eval_set = load_eval_set()
    case = eval_set.get("E17")
    runner = EvalRunner(eval_set=eval_set)

    answer = AgentAnswer(
        question=case.question,
        answer="我的数据只覆盖有限窗口，这个问题暂时答不了。",
        ok=True,
        steps=[],
        iterations=1,
    )
    judge = FakeJudge()
    result = evaluate_case(case, FakeAgent({case.question: answer}), judge, runner)

    assert result.use_grounding is False
    assert result.reference_row_count == 0
    assert result.hallucination.sentence_total == 0
    assert judge.calls[0][1] == _OUT_OF_SCOPE_REFERENCE


def test_evaluate_case_survives_agent_exception():
    """Agent 抛异常时也要产出一条带 error 的结果，不能中断整轮评测。"""
    eval_set = load_eval_set()
    case = eval_set.get("E01")
    runner = EvalRunner(eval_set=eval_set)

    result = evaluate_case(
        case, FakeAgent({case.question: RuntimeError("网络断了")}), FakeJudge(), runner
    )

    assert result.ok is False
    assert "网络断了" in result.error


def test_summarize_cases_aggregates():
    """幻觉指标只算有真值的用例；质量分算上全部用例（含超范围拒答）。"""
    grounded = CaseEvalResult(
        case_id="A",
        question="q1",
        use_grounding=True,
        hallucination=summarize_hallucination(
            [
                SentenceCheck(index=1, text="a", numbers=[465.0], traceable=1, has_data_number=True),
                SentenceCheck(index=2, text="b", numbers=[999.0], unexplained=["999"], has_data_number=True),
            ]
        ),
        judge=JudgeVerdict(
            claims=[{"index": 1, "grounded": True}, {"index": 2, "grounded": False}],
            ungrounded_count=1,
            ungrounded_rate=0.5,
            task_done=True,
            intent_score=5,
            completeness_score=4,
            relevance_score=5,
        ),
        usage={"total_tokens": 100},
        elapsed_ms=1000.0,
    )
    out_of_scope = CaseEvalResult(
        case_id="B",
        question="q2",
        use_grounding=False,
        judge=JudgeVerdict(task_done=False, intent_score=3, completeness_score=2, relevance_score=4),
        usage={"total_tokens": 200},
        elapsed_ms=2000.0,
    )
    summary = summarize_cases([grounded, out_of_scope])

    assert summary.total == 2
    assert summary.grounded_cases == 1
    assert summary.out_of_scope_cases == 1
    assert summary.sentence_total == 2
    assert summary.sentence_with_numbers == 2
    assert summary.hallucination_rate == 0.5
    assert summary.data_citation_accuracy == 0.5
    assert summary.claim_total == 2
    assert summary.ungrounded_rate == 0.5
    assert summary.task_done_rate == 0.5
    assert summary.avg_intent == 4.0
    assert summary.avg_completeness == 3.0
    assert summary.avg_relevance == 4.5
    assert summary.total_tokens == 300
    assert summary.avg_elapsed_ms == 1500.0


def test_collect_failures_lists_reasons():
    result = CaseEvalResult(
        case_id="A",
        question="q",
        use_grounding=True,
        hallucination=summarize_hallucination(
            [SentenceCheck(index=1, text="a", numbers=[999.0], unexplained=["999"], has_data_number=True)]
        ),
        judge=JudgeVerdict(
            claims=[{"index": 1, "grounded": False, "reason": "真值中无此数据"}],
            ungrounded_count=1,
            task_done=True,
        ),
    )
    failures = collect_failures([result])
    reasons = " ".join(reason for reason, _, _ in failures)

    assert "无法溯源" in reasons
    assert "无依据" in reasons
    # 无依据的 claim 明细要一并带出来，报告里展示 Judge 给的理由原文
    assert any(claims for _, _, claims in failures)
