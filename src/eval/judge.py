# -*- coding: utf-8 -*-
"""
LLM-as-a-Judge 评审封装（测试方案 Phase 3 · 维度三 + 维度四）
================================================================================
【为什么数字对账之外还需要一个 LLM 评审？】

  hallucination.py 只能回答「数字有没有出处」。它管不了另外三类问题：

      · 句子里的**对比 / 归因 / 因果**是否超出数据（「留存下滑是版本造成的」）；
      · 回答有没有**真正理解用户的意图**（问的是留存，答的却是日活）；
      · 回答**完不完整、切不切题**。

  这三类都需要"读懂语义"才能判断，纯函数做不到，所以请一个模型来当评审员。
  这就是 LLM-as-a-Judge：用模型评模型，但**判据被程序锁死**。

【为什么一次调用同时拿回「句子核验」和「四项打分」？】

  因为这两件事读的是**同一份上下文**（用户问题 + 真值数据 + 待评审回答）。
  拆成两次调用，等于把这份上下文付两遍钱，而且多一个失败面（任何一次失败都要
  重试或降级）。合并之后：token 省一半、失败点少一半，模型也能在"核验句子"的
  同一遍推理里顺手给出整体打分，判断更一致。

【判据必须锁死在「真值数据」上（这是本模块的立身之本）】

  如果允许评审员用外部知识判断，它会退化成"常识裁判"：
  看到「次留 42%」觉得合理就判通过，哪怕真值里根本没有这个数。
  那这套评审就变成"模型觉得模型答得不错"，毫无意义。
  所以 prompt 里反复强调：真值里没有的，一律判无依据。

  这是本项目「防自证」的第二重保险：
      第一重 —— runner.build_reference 走**不经过 LLM** 的独立路径取真值；
      第二重 —— 评审员被约束为**只能依据这份真值**做判断。
  真值独立于被测 Agent，评审又独立于外部知识，两个"独立"叠加，
  才让"没有幻觉"这个结论有说服力。

【为什么不用 BERT / NLI 三信号方案？】

  学术界做幻觉检测常用「NLI 蕴含 + 语义相似度 + 事实核查」三信号融合，
  但那套方案要求：① 有 GPU（NLI 模型逐句推理在 CPU 上很慢）；
  ② 下载数百 MB 的预训练模型；③ 中文场景还要额外挑模型。
  本项目是求职作品，部署目标是"一条命令跑起来"，引入这套依赖得不偿失。
  用 LLM-as-a-Judge 做替代，并靠"真值独立路径"堵住自证风险 ——
  在**这个规模**下，性价比明显更高。这是有意识的取舍，不是没考虑过。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.agent.llm_client import LLMClient
from src.eval.hallucination import split_sentences

# ===========================================================================
# 一、评审提示词（模块级常量，便于评审、对比、迭代）
# ===========================================================================

JUDGE_SYSTEM_PROMPT = """\
你是「数据分析结果评审员」。你的唯一职责是：对照给定的【真值数据】，
逐句核验【待评审回答】里的事实性声明是否有依据，并给这份回答打分。

【第一原则：只能依据真值数据，不得引入任何外部知识】
真值数据是这份回答唯一允许的出处。凡是真值数据里没有的数字、对比、归因、因果结论，
一律判定为「无依据」（grounded = false）。
即使你凭常识认为某个说法是对的，只要真值里没有，也必须判为无依据 ——
因为这份评审要证明的是「回答有没有超出数据说话」，而不是「回答对不对」。

【真值数据由三块组成，任何一块里的内容都算「有依据」】
  · 【口径与参数】—— 参考指标名、查询参数（尤其是时间区间）、指标口径原文。
     回答里复述这些参数或口径（如「最近 7 天」「2026-09-04 ~ 09-10 注册的用户」
     「分母剔除了观察期不足的用户」）**不算编造**，判 grounded = true。
     注意：只有当复述的范围与这里给出的参数一致时才算；范围明显不同才算编造。
  · 【参考数据】—— 该用例声明指标的真实数据行。
  · 【模型额外查询的指标】—— 模型主动多查的交叉验证指标，其数据同样来自数据库，
     同样算「有依据」。真值里出现了多个指标时，用其中任何一个的数据支撑都算 grounded。

【逐句核验（必须区分下面六种情况，这是本评审的关键）】
  1. 数字编造 —— 回答给出了一个具体数值，但**三块真值里都找不到它、也无法由真值推导出来**
     → grounded = false，reason 写明"真值中无此数据"
  2. 复述口径 / 参数 / 方法论 —— 如「留存率的分母剔除了观察期不足的用户」
     「样本量过小的渠道仅供参考」「这个批次选在 8/18 截止是为了保证观察满 30 天」，
     或复述真值【口径与参数】里的时间区间
     → grounded = true，reason 写"方法论/口径说明，无需数据支撑"
     把口径说明误判成幻觉，会让评测变成惩罚专业性 —— 务必区分。
  3. 与真值数据一致的描述性判断 —— 如「整体平稳，在 455~655 人区间波动」
     「最高出现在 09-05」「A 渠道高于 B 渠道」这类**对真值数据的概括、排序、对比**
     → grounded = true。分析师的工作本来就是概括数据，不能要求它逐字复述。
     但若概括与真值矛盾（例如真值最低点是 09-04，回答却说 09-11 是低点）
     → grounded = false，reason 写明与真值矛盾之处。
  4. 无依据的因果归因 / 外部知识 —— 如「是周末效应」「因为版本更新和买量投放」
     「行业基准是 X」「LTV/CAC 大于 3 才算健康」「竞品也在涨」
     这类**以肯定语气断言**、且需要真值之外的信息才能成立的结论
     → grounded = false，reason 写明真值中缺少哪类支撑数据。
  5. 建议 / 排查方向 / 服务性表述 —— 如「建议优先排查步骤 3 的引导内容」
     「需要的话我可以按渠道拆开看」「以上建议需人工复核后执行」
     → grounded = true。给建议说的是"下一步做什么"，本来就不由数据支撑，不属于事实断言。
     注意：只有**建议本身**算 grounded；若建议句里夹带了独立的事实断言
     （如「行业通常以 LTV/CAC > 3 为健康线」），该断言仍要按第 4 类单独判定。
  6. 带不确定标记的推测 —— 如「可能与周末有关」「通常和新增用户有关，需交叉验证才能判断」
     「如果赶上节假日也会推高留存，不能直接说是版本改动带来的」
     → grounded = true。模型已经**自己声明了不确定**、并指出需要什么数据才能确认 ——
     这是诚实，不是幻觉。本评审的对象是「断言为真」的内容，不是「表达不确定性」的内容。
     判据是**有没有不确定标记**：有标记（可能/或许/通常/需进一步验证/不能直接断定）
     → 按第 6 类；没有标记、直接下结论（如「9/13 的峰值符合周末活跃规律」
     「说明老带新用户粘性明显更强」「继续抬客单价的空间有限」）→ 按第 4 类判无依据。

【一致性的硬要求：reason 与 grounded 必须自洽】
  grounded 是结论，reason 是理由，两者必须说同一件事：
    · 若 reason 里写了「与真值矛盾」「真值中无此数据」「真值里没有」这类判断，
      该句的 grounded **必须**填 false —— 不能一边认定它有问题、一边把字段填成 true。
      这是**从宽方向的漏报**，比误报更危险：判词已经认定有问题，字段却放过了它。
    · 反过来，若 grounded 填了 false，reason 里必须写清「缺的是哪类支撑数据」，
      不能只丢一个「无依据」就完事。
  ★ 解析层会对这一条做二次校验：凡是 reason 说了矛盾/真值中无、而 grounded 填了 true 的，
    会被自动改判为 false，并在 reason 前打上 [一致性修正] 标记。
    与其被程序改，不如一次判对。

【四项打分（1-5 的整数）】
  intent_score       意图解析：有没有理解用户真正想问的
  completeness_score 答案完整性：是否覆盖问题的所有方面
  relevance_score    答案相关性：是否直接回答、有没有跑题
  task_done          用户问题是否被正确回答（true / false）
     ★ 判定 task_done 时，只要核心结论有真值支撑、且正面回答了用户的问题，
       就给 true。若回答以「数据不足以判断」等合理说明代替结论，
       但该说法本身正确（真值确实不支持），同样算 true ——
       诚实地说"答不了"也是完成任务，不要因此判 false。

【输出要求】
只输出一个 JSON 对象，不要任何解释性文字、不要代码块围栏。严格按下面的 schema：
{
  "claims": [
    {"index": 1, "text": "该句原文（可截断到 40 字）", "grounded": true, "reason": "被真值中 XX 支撑"}
  ],
  "task_done": true,
  "intent_score": 5,
  "completeness_score": 4,
  "relevance_score": 5
}
claims 必须覆盖每一个编号句子；没有事实性声明的句子 grounded 填 true。
"""


def build_judge_prompt(question: str, reference_text: str, answer: str) -> str:
    """拼装评审用的 user 消息。

    为什么要在这里**重新分句并编号**？
      因为核验结果里的 index 必须能对回 hallucination.py 的句子编号。
      如果让模型自己分句，两套切分标准会产生错位，
      「第 3 句」在数字报告和 Judge 报告里指的是不同的东西 —— 报告就没法合起来看。
    """
    sentences = split_sentences(answer)
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(sentences, start=1))
    return (
        f"【用户问题】\n{question}\n\n"
        f"【真值数据】\n{reference_text}\n\n"
        f"【待评审回答（已按句编号）】\n{numbered or '（回答为空）'}\n\n"
        f"请逐句核验并打分，只输出 JSON。"
    )


# ===========================================================================
# 二、结果数据结构
# ===========================================================================

@dataclass
class JudgeVerdict:
    """一次评审的结果。

    claims 里每一项形如 {"index": int, "text": str, "grounded": bool, "reason": str}。
    解析失败时各分数保持 0、error 填原因 —— 与项目「工具失败返回可观察结果」
    的风格一致：失败是可观察的数据，不是异常。
    """

    claims: list[dict] = field(default_factory=list)
    ungrounded_count: int = 0
    ungrounded_rate: float = 0.0
    task_done: bool = False
    intent_score: float = 0.0
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    raw: str = ""
    error: str | None = None
    # 一致性守卫改判了几条（reason 说无依据、grounded 却填了 true）。
    # 为什么要把这个数单独记下来？因为它是**评测台自身的健康指标**：
    # 大于 0 说明提示词里那条"硬要求"没被遵守 —— 判据还在被违反，
    # 只是被解析层兜住了。报告里会打印它，避免问题被"兜住"就等于"消失"。
    consistency_fixed: int = 0


# ===========================================================================
# 三、容错 JSON 解析
# ===========================================================================

# 代码块围栏。模型即使被要求"只输出 JSON"，也常常习惯性套一层 ```json。
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# 轻微格式瑕疵修复：对象/数组结尾多出来的逗号（尾逗号在 JSON 标准里非法，
# 但模型很爱写）。只做这一种修复，不做更激进的改动 ——
# 过度修复会把"模型返回了错误内容"伪装成"解析成功"，反而掩盖问题。
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _loads_with_repair(candidate: str) -> Any | None:
    """先按标准 JSON 解析，失败再尝试去掉尾逗号。"""
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        pass
    repaired = _TRAILING_COMMA.sub(r"\1", candidate)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_json(text: str | None) -> Any | None:
    """从模型返回的文本里尽力抠出一个 JSON 值，抠不到返回 None。

    为什么需要容错解析？因为即便 prompt 里写了"只输出 JSON"，
    实测模型仍会：① 套一层 ```json 围栏；② 前后加一句"好的，以下是评审结果："；
    ③ 偶尔漏个尾逗号。解析失败就直接判整条用例失败，代价太大 ——
    这里多花几行做容错，能显著降低评测的"假失败"。

    尝试顺序（先精确、后退化）：
      1. 围栏内的内容
      2. 整段文本
      3. 第一个 { 到最后一个 }  ← 处理"前后有解释性文字"
      4. 第一个 [ 到最后一个 ]  ← 处理顶层是数组的情况
    每一步都允许"去尾逗号"修复。全都失败才返回 None。
    """
    if not text:
        return None

    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        parsed = _loads_with_repair(candidate)
        if parsed is not None:
            return parsed
    return None


# ===========================================================================
# 四、评审器
# ===========================================================================

def _as_score(value: Any) -> float:
    """把模型给的分数规整成 1~5 的浮点数。

    为什么要夹到 [1,5]？因为模型偶尔会给 0 分或 7 分（超出约定范围）。
    不夹的话，一个 7 分会把平均分拉高，看起来像"质量变好了"，
    实际上是模型没按规则输出 —— 这种"沉默的偏差"必须挡掉。
    取不到数值时返回 0，表示"没给分"。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(1.0, min(5.0, number))


# ---------------------------------------------------------------------------
# 一致性守卫用的判词标记（Phase 5 发现 5 的修复）
# ---------------------------------------------------------------------------
# 【为什么要在解析层再做一次校验，而不是只靠 prompt 叮嘱？】
#   Phase 5 的 A 轮里出现过一次自相矛盾：Judge 把某句的 reason 写成
#   「该描述与真值矛盾」，grounded 字段却填了 true（按判据第 3 类本应判 false）。
#   这是**从宽方向**的漏报 —— 判词已经认定有问题，字段却放过了它。
#   prompt 是软约束，实测证明它会被违反；所以解析层补一道硬校验。
#
# 【为什么以 reason 为准，而不是反过来？】
#   reason 是模型"想清楚了才写"的自由文本，grounded 只是一个布尔位。
#   两者冲突时，携带信息更多的那一方更可信 —— 而且 reason 是要给人看的，
#   它说"矛盾"，那就是矛盾。
#
# 【为什么允许这里出现"误报"？】
#   把 reason 说了"矛盾 / 真值中无"的句子翻成 false，方向是**更严**。
#   幻觉检测的第一原则是"宁可误报不可漏报"
#   （见 docs/02_测试复盘/测试复盘记录.txt 里「聚合运算是否纳入对账」的决策），
#   所以这个方向的偏差是可接受的；反过来放过一句才是真问题。
#
# 【已知的假阳性风险（写在这里，别装作没有）】
#   若模型写出「该说法在真值中无直接对应，但属于建议性表述，故判为有依据」
#   这类"先否定再解释"的判词，也会被误翻。这类判词在实测中未出现；
#   真出现时，reason 上会留下 [一致性修正] 标记，人工复核一眼能看到，
#   比"悄悄放过"要好。
_UNGROUNDED_REASON_MARKERS: tuple[str, ...] = (
    "与真值矛盾",
    "真值中无",
    "真值中没有",
    "真值里没有",
    "真值中缺少",
    "真值里缺少",
    "无法由真值",
    "属于无依据",
)


def _reason_says_ungrounded(reason: str) -> bool:
    """判词本身是否已经认定"这句话没有依据"。"""
    return any(marker in reason for marker in _UNGROUNDED_REASON_MARKERS)


class LLMJudge:
    """用 LLM 做句子核验 + 四项打分。

    依赖注入 client，是为了让单元测试能塞一个假客户端，
    在不联网、不花 token 的前提下验证解析与容错逻辑。
    （和 LLMClient / GameDataAgent 的注入风格一致。）
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        # 默认自己建一个客户端；测试时注入 FakeClient 即可完全离线。
        self.client: LLMClient = client if client is not None else LLMClient()

    def judge_case(self, question: str, reference_text: str, answer: str) -> JudgeVerdict:
        """评审一条回答。

        这个方法**不抛异常**：LLM 调用失败、返回内容无法解析，都返回
        JudgeVerdict(error=...) 且分数为 0。
        为什么？因为评测要跑几十条，任何一条评审失败都不该让整轮评测崩掉 ——
        否则你永远拿不到一份完整报告（与 runner.run_case 的立场一致）。
        """
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_judge_prompt(question, reference_text, answer or ""),
            },
        ]
        try:
            response = self.client.chat(messages)
        except Exception as exc:  # noqa: BLE001 - 评审失败必须转成可观察结果
            return JudgeVerdict(error=f"Judge 调用失败：{exc}")

        raw = getattr(response, "content", "") or ""
        data = _extract_json(raw)
        if data is None:
            return JudgeVerdict(raw=raw, error="Judge 返回内容无法解析为 JSON")
        return self._to_verdict(data, raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_verdict(data: Any, raw: str) -> JudgeVerdict:
        """把解析出来的 JSON 值转成 JudgeVerdict。

        为什么兼容「顶层是数组」？
          模型有时会直接把 claims 数组返回回来（省掉了外层对象）。
          这种情况下没有分数可读，就只取 claims —— 句子核验依然有效，
          比整条判失败要划算。
        """
        if isinstance(data, list):
            claims = data
            scores: dict[str, Any] = {}
        elif isinstance(data, dict):
            claims = data.get("claims") or []
            scores = data
        else:
            return JudgeVerdict(raw=raw, error="Judge 返回的 JSON 不是对象也不是数组")

        normalized: list[dict] = []
        for item in claims:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "index": int(item.get("index") or 0),
                    "text": str(item.get("text") or ""),
                    # 只有明确写成 false 才算"无依据"；缺失时按"无依据"处理更保守？
                    # 不 —— 这里按 grounded 的真值判断，缺失时视为 False 会让
                    # 模型漏字段直接变成幻觉指控。所以用 bool() 且默认 True 更稳：
                    # 模型没标 false 就不主动指控（评测里"宁可漏报不误报"）。
                    "grounded": item.get("grounded", True) is not False,
                    "reason": str(item.get("reason") or ""),
                }
            )

        # ---- 一致性守卫：reason 与 grounded 打架时，以 reason 为准 ----
        # 判据见 _UNGROUNDED_REASON_MARKERS 上方的注释。这里只做一件事：
        # 把"判词已认定无依据、字段却填 true"的条目翻成 false，并在 reason 上留痕。
        # ★ 留痕而不是静默修改 —— 否则这份评测报告就变成"看起来更准了，
        #   但没人知道它被改过"，那和篡改数据没有区别。
        consistency_fixed = 0
        for claim in normalized:
            if claim["grounded"] and _reason_says_ungrounded(claim["reason"]):
                claim["grounded"] = False
                claim["reason"] = f"[一致性修正·原判 grounded=true] {claim['reason']}"
                consistency_fixed += 1

        ungrounded = [c for c in normalized if not c["grounded"]]
        total = len(normalized)
        return JudgeVerdict(
            claims=normalized,
            ungrounded_count=len(ungrounded),
            ungrounded_rate=round(len(ungrounded) / total, 4) if total else 0.0,
            task_done=bool(scores.get("task_done", False)),
            intent_score=_as_score(scores.get("intent_score")),
            completeness_score=_as_score(scores.get("completeness_score")),
            relevance_score=_as_score(scores.get("relevance_score")),
            raw=raw,
            error=None,
            consistency_fixed=consistency_fixed,
        )
