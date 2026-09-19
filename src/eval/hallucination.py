# -*- coding: utf-8 -*-
"""
幻觉检测（测试方案 Phase 3 · 维度三）
================================================================================
【这个模块解决什么问题？】

  Phase 5 的 runner.py 已经能回答「整条回答里有多少个数字对不上」，
  但它给的是**一个整体比率**。整体比率有个致命弱点：一条回答里只要有一个
  编造的数字，比率可能只掉几个点，报告上看起来"还行"，可那句结论已经不可信了。

  所以本模块把粒度下沉一层：**把回答切成句子，逐句做数字对账**。
  这样能同时回答两个不同的问题：

      · 幻觉率（句子级）       —— 有多少句掺了假？
        句子是「结论」的载体，一句里有编造，整句结论就不能用。
      · 数据引用准确率（数字级）—— 所有数字里有多少是真的？
        衡量的是引用质量，反映"整体有多少数字是干净的"。

  ★ 这两个口径为什么必须分开看？
    一条长回答有 30 个数字、只有 1 个编造，数字级准确率是 96.7%（看着很好）；
    但如果那个编造的数字恰好出现在最核心的结论句里，句子级幻觉率能把它精确标出来。
    反过来，一句话里塞了 5 个数字、错了 3 个，数字级会报警，句子级只算 1 句。
    两个口径互为补充，单看任何一个都会漏掉另一类问题。

【为什么本模块不调 LLM？】

  因为「数字有没有出处」是**可判定**的：拿参考数据逐个比对就行，不需要模型判断。
  做成纯函数的另一个好处是可离线单测 —— 评测算法自己算错了，比模型答错更可怕。
  （需要"读懂语义"的判断交给 judge.py，两者职责严格分开。）

【为什么分句用标点切，而不是 nltk / spacy？】

  中文分句库（spacy 中文模型、哈工大 LTP 等）都要额外下载几十到几百 MB 的模型，
  而本项目要处理的只是**模型给出的短回答**（一般 5~15 句），
  用标点切分（。！？；!?; 和换行）已经足够，且零依赖、零加载时间。
  引入重型 NLP 库在这里是典型的"用大炮打蚊子"：收益为零，还增加部署成本。
  真到了要处理长文档的场景再换，那时才有必要。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Sequence

from src.eval.metrics import check_answer_numbers, extract_numbers

# ===========================================================================
# 一、分句
# ===========================================================================

# 中文 + 英文的句末标点。为什么连英文标点也要切？
# 因为模型经常中英混排（"DAU 是 465 人. 环比 +3%"），只切中文标点会把两句粘成一句，
# 于是"前半句干净、后半句编造"被合成一句，句级幻觉率就失真了。
_SENTENCE_DELIMITER = re.compile(r"[。！？；!?;]+")

# 「有实质内容」的判据：至少含一个单词字符（含中文、数字、字母）。
# 用它过滤掉纯标点/纯空白的碎片 —— 否则「。。。！」这种会被算成一句，
# 白白拉低幻觉率的分母（分母虚高 = 幻觉率被稀释，等于自欺欺人）。
_HAS_CONTENT = re.compile(r"\w")

# Markdown 有序列表的序号前缀，如「2. 」「1、」「3）」，允许外面套一层加粗/斜体标记
# （模型常写「**2. 长期人均价值**」）。序号后**必须跟空白**，这是关键：
# 靠这个条件把「2. 对比 D1/D7」和「3.7% 的付费率」区分开 ——
# 后者的点在数字中间、后面没有空格，不会被误剥。
# 第 1 组捕获缩进、第 2 组捕获强调标记，替换时原样放回，避免留下落单的 `**`。
_LIST_MARKER = re.compile(r"^(\s*)((?:\*\*|__|\*|_)?)\s*\d{1,2}\s*[.、)]\s+")


def _strip_list_marker(piece: str) -> str:
    """剥掉句子开头的有序列表序号（保留缩进与加粗/斜体标记）。

    【为什么必须剥？——Phase 3 实测到的假阳性】
      E05 / E21 的回答里有「2. 对比 D1 / D7 / D30，看用户是在哪个阶段流失的」
      这种列表项。序号 "2." 被数字抽取当成了一个数据型数字，
      于是整句被标成「含无法溯源数字」—— 幻觉率凭空多了一句。
      序号是排版，不是内容；不剥掉就是评测器自己制造幻觉。
    """
    return _LIST_MARKER.sub(lambda match: match.group(1) + match.group(2), piece, count=1)


def _is_table_row(line: str) -> bool:
    """判断一行是不是 Markdown 表格行。

    判据是「strip 后以 | 开头」。为什么表格行必须整体保留、不能按标点切？
      因为表格是**一行行数据**：「| 09-05 | 99 人 | 09-17 | 54 人 |」里的
      每个单元格都是独立事实，按标点/竖线切会把一行数据拆成碎片，
      既看不出原本的对应关系，也会让「数字级」和「句子级」两个口径同时失真。
      整行当作一句，正好对应"这一行数据是否可信"这个自然的问题。
    """
    return line.startswith("|")


def split_sentences(text: str | None) -> list[str]:
    """把一段回答切成句子列表。

    规则（按行处理，行内再按标点切）：
      1. Markdown 表格行（以 | 开头）**整体作为一句**，不按标点拆（见 _is_table_row）；
      2. 其余行按 。！？；!?; 切分；
      3. 每句 strip 掉首尾空白，并剥掉开头的有序列表序号（见 _strip_list_marker）；
      4. 丢弃空串和纯符号/纯空白的碎片。

    返回的是「有内容」的句子，顺序与原回答一致。
    空输入返回空列表（而不是 ['']）—— 调用方可以据此直接判断"没有可检测的句子"。
    """
    if not text:
        return []

    sentences: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_table_row(line):
            sentences.append(line)
            continue
        for fragment in _SENTENCE_DELIMITER.split(line):
            piece = _strip_list_marker(fragment.strip())
            if piece and _HAS_CONTENT.search(piece):
                sentences.append(piece)
    return sentences


# ===========================================================================
# 二、逐句数字对账
# ===========================================================================

@dataclass
class SentenceCheck:
    """一句话的对账结果。

    index 从 1 开始（第几句），不是 0 —— 因为报告和 Judge 的 prompt 都要给人/模型看，
    "第 1 句"比"第 0 句"自然，也避免把索引误当成数组下标。
    """

    index: int
    text: str
    numbers: list[float] = field(default_factory=list)   # 这一句里抽到的数据型数字
    traceable: int = 0                                    # 其中有出处的个数
    unexplained: list[str] = field(default_factory=list)  # 找不到出处的数字原文
    has_data_number: bool = False                         # 这一句是否含数据型数字

    @property
    def flagged(self) -> bool:
        """这一句是否被标记为「含无法溯源的数字」。"""
        return bool(self.unexplained)


def check_sentence_numbers(
    sentence: str,
    reference_values: Sequence[float],
    extra_allowed: Sequence[float] = (),
) -> SentenceCheck:
    """对一句话做数字对账。

    为什么直接复用 metrics.check_answer_numbers，而不是自己写一套比对？
      因为「严格命中 / 口语近似 / 可推导」这三档判定规则是这个项目的核心资产
      （见 metrics.py 的模块说明），两套规则并行迟早会打架 ——
      同一句话在两个报告里得出不同结论，是最难排查的一类 bug。
      ★ 复用还有一个隐含好处：数字抽取、日期剔除、衍生值推导全部一致，
        评测的"尺子"只有一把。

    extra_allowed 传的是「合法但不在数据行里」的数字（期望参数值、参考行数、
    指标定义里的口径常量），用法与 runner.py 完全一致。
    """
    # 先抽一次数字：既用于 numbers 字段，也用于判断 has_data_number。
    # （check_answer_numbers 内部也会抽，但那是它的实现细节，这里不依赖它。）
    numbers = [item.value for item in extract_numbers(sentence)]
    check = check_answer_numbers(sentence, reference_values, extra_allowed=extra_allowed)
    return SentenceCheck(
        index=0,                                  # 真实序号由 build_sentence_report 回填
        text=sentence,
        numbers=numbers,
        traceable=check.matched_count,
        unexplained=[item.raw for item in check.unexplained],
        has_data_number=bool(numbers),
    )


# ===========================================================================
# 三、聚合指标
# ===========================================================================

@dataclass
class HallucinationSummary:
    """一段回答的幻觉指标汇总。

    【两个比率的口径差异（本模块最容易被误读的地方）】

      幻觉率 = 含无法溯源数字的句子数 / 总句数        ← 句子级
        · 分子只统计"有编造数字的句子"，**没有数字的句子不计入分子**。
          像「建议关注留存变化」这种纯建议句，天然不可能有出处问题，
          把它算成幻觉是荒谬的 —— 这也是本模块必须区分 has_data_number 的原因。
        · 分母是总句数，所以它回答的是"整段回答里有多大比例的句子不可信"。

      数据引用准确率 = 可溯源数字数 / 总数字数          ← 数字级
        · 分母是**数字个数**，所以它回答的是"引用的数字有多大比例是真的"。
        · 两者可能同时偏高或偏低，但永远不会互相替代：
          一句编造里塞 10 个假数字，数字级掉得厉害、句子级只算 1 句；
          30 个真数字里错 1 个，数字级几乎不掉、句子级却会亮起 1 句。

    两个比率在分母为 0 时返回 0.0 而不是崩溃或 None：
      这是"整段回答没有可检测对象"的明确信号，报告里会配合样本量一起展示，
      单看 0.0 不会误导（因为旁边就写着 0/0）。
    """

    sentence_total: int = 0            # 总句数
    sentence_with_numbers: int = 0     # 含数据型数字的句数
    sentence_flagged: int = 0          # 含无法溯源数字的句数
    number_total: int = 0              # 数据型数字总数
    number_traceable: int = 0          # 有出处的数字数

    @property
    def hallucination_rate(self) -> float:
        """幻觉率（句子级）= 被标记句数 / 总句数。"""
        if self.sentence_total <= 0:
            return 0.0
        return round(self.sentence_flagged / self.sentence_total, 4)

    @property
    def data_citation_accuracy(self) -> float:
        """数据引用准确率（数字级）= 可溯源数字数 / 数字总数。"""
        if self.number_total <= 0:
            return 0.0
        return round(self.number_traceable / self.number_total, 4)


def summarize_hallucination(sentences: Sequence[SentenceCheck]) -> HallucinationSummary:
    """把逐句结果聚合成整段回答的指标。

    注意 sentence_flagged 只可能由「含无法溯源数字的句子」贡献 ——
    没有数字的句子（has_data_number=False）天然 unexplained 为空，
    因此不会进入幻觉率的分子。这一点由数据本身保证，不靠调用方自觉。
    """
    summary = HallucinationSummary(sentence_total=len(sentences))
    for item in sentences:
        if item.has_data_number:
            summary.sentence_with_numbers += 1
        if item.unexplained:
            summary.sentence_flagged += 1
        summary.number_total += len(item.numbers)
        summary.number_traceable += item.traceable
    return summary


def build_sentence_report(
    answer_text: str | None,
    reference_values: Sequence[float],
    extra_allowed: Sequence[float] = (),
) -> tuple[list[SentenceCheck], HallucinationSummary]:
    """便捷入口：一段回答 → （逐句明细, 汇总指标）。

    把「分句 → 逐句对账 → 聚合」串成一步，调用方（脚本/测试）只需要关心输入输出。
    句子序号在这里统一回填成 1 起始，保证明细、报告、Judge 三处的编号一致。
    """
    sentences = split_sentences(answer_text)
    checks = [
        replace(check_sentence_numbers(text, reference_values, extra_allowed), index=index)
        for index, text in enumerate(sentences, start=1)
    ]
    return checks, summarize_hallucination(checks)
