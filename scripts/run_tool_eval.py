# -*- coding: utf-8 -*-
"""工具调用准确性专项评测（测试方案 Phase 1 · 维度一）

================================================================================
【这个脚本解决什么问题？】

  Phase 5 的 run_eval.py 回答的是「答得对不对」（数字能不能溯源）；
  本脚本回答的是**更上游的问题**：**工具调对了吗**。

  为什么要把这件事单独拎出来测？
    因为 Agent 的错误是分层的。如果模型选错了指标，那后面的数字对账
    全部失去意义 —— 算得再准，查的也是另一个指标。
    分层定位错误，才能知道该改提示词、改工具描述，还是改参数 schema。

【错误分级参考 MultiCAT-Bench 的六类，逐条给出本项目里的判据】

    GE  生成错误        没有产出可用回答（空回答 / 未正常返回）
    NCA 完全未调用      该查数，但一次 query_metric 都没调用
    NNC 无需调用却调用  不该查数的用例（超范围/危险操作）却成功查了
    WNC 调用数量错误    指标对了、参数也对了，但反复查（≥4 次）在打转
    WTC 调用了错误工具  查了数，但从没命中期望的 metric_id
    WA  参数错误        指标选对了，参数抽错

  ★ 分级优先级：WTC > WA > WNC > OK
    先判「选错指标」再判「参数错」最后判「次数异常」——
    因为选错指标是最严重的错，不该被后面的标签掩盖。

【与评测器的口径保持一致（重要）】
  · 「成功查询」= query_metric 返回 ok=True，复用 eval 层的 metric_calls
  · 指标/参数判定直接复用 check_metric_mapping / check_params，
    不另起一套标准 —— 两套尺子量同一件事，迟早会打架

【产物】
  docs/04_评测原始数据/工具调用测试报告_Phase1.csv   逐例明细，可丢进 Excel 透视
  docs/03_评测报告/工具调用测试报告_Phase1.txt       指标汇总 + 合格线对照 + badcase
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import sys
from pathlib import Path

# 让脚本能直接 `python scripts/run_tool_eval.py` 跑（把项目根目录加进搜索路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台默认是 GBK，直接打印中文/emoji 会炸，这里强制 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src import config as cfg                                     # noqa: E402
from src.agent.react_agent import GameDataAgent                    # noqa: E402
from src.agent.tools import ToolExecutor                           # noqa: E402
from src.eval.eval_set import load_eval_set                        # noqa: E402
from src.eval.metrics import (                                     # noqa: E402
    check_metric_mapping,
    check_params,
    format_rate,
    metric_calls,
)

# ---------------------------------------------------------------------------
# SQL 安全校验的攻击载荷（对应 tests/test_security.py，这里跑一遍是为了让报告
# 自带证据 —— 报告里的每个结论都应该是"刚跑出来的"，而不是抄测试文件的注释）
# ---------------------------------------------------------------------------
SECURITY_PAYLOADS: list[tuple[str, dict]] = [
    ("metric_id 注入（拖库）", {"metric_id": "dau'; DROP TABLE dim_user;--", "params": {}}),
    ("metric_id 注入（联合查询越权读表）",
     {"metric_id": "dau UNION SELECT * FROM hr_recruitment_data", "params": {}}),
    ("metric_id 注入（堆叠语句）",
     {"metric_id": "dau; DELETE FROM user_daily_snapshot;--", "params": {}}),
    ("日期参数注入（永真条件）",
     {"metric_id": "dau", "params": {"start_date": "2026-09-17' OR '1'='1"}}),
    ("日期参数注入（非法日期）",
     {"metric_id": "dau", "params": {"start_date": "not-a-date"}}),
    ("日期参数注入（空串）", {"metric_id": "dau", "params": {"start_date": ""}}),
    ("day_n 注入（堆叠语句）",
     {"metric_id": "retention_rate", "params": {"day_n": "1; DROP TABLE dim_user"}}),
    ("day_n 越界（99999）", {"metric_id": "retention_rate", "params": {"day_n": "99999"}}),
    ("day_n 负数（-1）", {"metric_id": "retention_rate", "params": {"day_n": "-1"}}),
    ("未声明参数键夹带指令",
     {"metric_id": "dau", "params": {"start_date": "2026-09-11", "evil": "'; DROP TABLE dim_user;--"}}),
    ("极端日期范围（1900~2100）",
     {"metric_id": "dau", "params": {"start_date": "1900-01-01", "end_date": "2100-01-01"}}),
    ("未知工具名（幻觉调用）", {"__tool__": "drop_all_tables"}),
]


def run_security_probe() -> tuple[list[dict], dict[str, tuple[int, int]]]:
    """把攻击载荷打一遍，返回 (逐条结果, 数据库完好性核对)。

    为什么报告要自己跑一遍，而不是直接引用 tests/test_security.py 的结论？
    因为"测试通过"是一个布尔值，读者看不到**实际发生了什么**。
    这里把每次调用的真实返回值摊开，报告才有证据力。
    """
    tools = ToolExecutor()
    rows: list[dict] = []

    def count(table: str) -> int:
        # 独立只读连接核对，不用被测代码自己的连接
        with sqlite3.connect(f"file:{cfg.DB_PATH}?mode=ro", uri=True) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def tables() -> set[str]:
        with sqlite3.connect(f"file:{cfg.DB_PATH}?mode=ro", uri=True) as conn:
            return {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }

    before_tables = tables()
    before_counts = {t: count(t) for t in ("dim_user", "user_daily_snapshot", "game_event_log")}

    for label, args in SECURITY_PAYLOADS:
        if "__tool__" in args:                       # 特殊：测未知工具名
            result = tools.execute(args["__tool__"], {})
        else:
            result = tools.execute("query_metric", args)

        row_count = len((result.data or {}).get("rows") or []) if result.ok else 0
        if not result.ok:
            verdict = "已拦截"
        elif row_count <= 1:
            verdict = "已中和（结果未放大）"
        else:
            verdict = "★可疑"
        rows.append(
            {
                "payload": label,
                "verdict": verdict,
                "ok": result.ok,
                "rows": row_count,
                "reason": (result.error or result.content or "").replace("\n", " ")[:90],
            }
        )

    after_tables = tables()
    after_counts = {t: count(t) for t in before_counts}
    intact = {
        "tables_before": len(before_tables),
        "tables_after": len(after_tables),
        "tables_equal": before_tables == after_tables,
        "counts": {t: (before_counts[t], after_counts[t]) for t in before_counts},
        "counts_equal": before_counts == after_counts,
    }
    return rows, intact


# ---------------------------------------------------------------------------
# 阈值：直接来自测试方案的合格标准，写在这里方便一眼对照
# ---------------------------------------------------------------------------
TARGET_MAPPING = 0.90        # 工具（指标）选择准确率 ≥90%
TARGET_PARAMS = 0.85         # 参数提取准确率 ≥85%
TARGET_MISS = 0.05           # 遗漏率 ≤5%（等价于调用完整性 ≥95%）
TARGET_OVER = 0.10           # 过度调用率 ≤10%
TARGET_F1 = 0.85             # 工具调用 F1 ≥0.85

# 「调用数量异常」的阈值：正常一次问答查 1 个指标，最多再查 1~2 个做交叉验证。
# 超过 3 个成功查询，基本可以判定模型在打转（每个指标都要走 口径→取数 两步）。
REDUNDANT_QUERY_THRESHOLD = 4


def tool_sequence(steps) -> list[str]:
    """把执行轨迹里的工具调用按顺序抽出来，用于观察「工作流程」是否合规。

    提示词要求的是：list_metrics（拿不准时）→ get_metric_detail（确认口径）
    → query_metric（取数）。这个序列能直接看出模型有没有跳步。
    """
    return [
        str(getattr(step, "tool_name", "") or "")
        for step in (steps or [])
        if getattr(step, "kind", "") == "tool_call"
    ]


def classify_error(
    *,
    answer_text: str,
    answer_ok: bool,
    expect_query: bool,
    mapping_ok: bool | None,
    params_ok: bool | None,
    calls: list[dict],
    query_attempts: int,
) -> str:
    """给一条用例打上错误分类标签（六选一）。

    参数：
      calls          —— 所有 query_metric 调用（含失败的），来自 metric_calls
      query_attempts —— query_metric 的调用次数（含失败），用于区分
                        「连试都没试」和「试了但没成功」
    """
    # ① GE：连可用回答都没有
    if not answer_ok or not (answer_text or "").strip():
        return "GE"

    succeeded = [call for call in calls if call["ok"]]

    # ② 不该查数的用例（超范围、危险操作）
    if not expect_query:
        return "NNC" if succeeded else "OK"

    # ③ 该查数但一次都没调
    if query_attempts == 0:
        return "NCA"

    # ④ 调了但全失败：区分是「选错指标」还是「指标对但参数有问题」
    if not succeeded:
        return "WA" if mapping_ok else "WTC"

    # ⑤ 指标选错（最严重的错，优先判定）
    if mapping_ok is False:
        return "WTC"

    # ⑥ 参数抽错
    if params_ok is False:
        return "WA"

    # ⑦ 指标参数都对，但查得太多次（打转）
    if len(succeeded) >= REDUNDANT_QUERY_THRESHOLD:
        return "WNC"

    return "OK"


ERROR_LABELS = {
    "OK": "通过",
    "GE": "生成错误（无可用回答）",
    "NCA": "完全未调用（该查却没查）",
    "NNC": "无需调用却调用",
    "WNC": "调用数量错误（反复查）",
    "WTC": "调用了错误工具（指标选错）",
    "WA": "参数错误",
}


def main() -> int:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    eval_set = load_eval_set()
    cases = eval_set.all()
    if limit:
        cases = cases[:limit]

    agent = GameDataAgent()

    print(f"用例数：{len(cases)}    供应商顺序：{getattr(agent.llm_client, 'provider_order', '?')}")
    print("=" * 86)

    rows: list[dict] = []
    for index, case in enumerate(cases, 1):
        expect_query = not case.expect_no_query
        try:
            answer = agent.ask(case.question, history=list(case.history) or None)
            steps = answer.steps
            answer_text = answer.answer or ""
            answer_ok = answer.ok
            error = answer.error or ""
        except Exception as exc:  # noqa: BLE001 - 单条失败不能毁掉整轮评测
            steps, answer_text, answer_ok, error = [], "", False, str(exc)

        calls = metric_calls(steps)
        attempts = sum(
            1
            for step in steps
            if getattr(step, "kind", "") == "tool_call"
            and getattr(step, "tool_name", "") == "query_metric"
        )
        mapping_ok = check_metric_mapping(steps, case.expected_metric_id)
        params_ok, params_detail = check_params(
            steps, case.expected_metric_id, case.expected_params
        )
        label = classify_error(
            answer_text=answer_text,
            answer_ok=answer_ok,
            expect_query=expect_query,
            mapping_ok=mapping_ok,
            params_ok=params_ok,
            calls=calls,
            query_attempts=attempts,
        )

        succeeded = [call for call in calls if call["ok"]]
        usage = getattr(answer, "usage", {}) or {}

        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "question": case.question,
                "expect_query": "是" if expect_query else "否",
                "expected_metric_id": case.expected_metric_id or "-",
                "expected_params": json.dumps(case.expected_params, ensure_ascii=False)
                if case.expected_params
                else "-",
                "actual_metric_ids": " > ".join(c["metric_id"] for c in calls) or "-",
                "actual_params": json.dumps(
                    succeeded[-1]["params"] if succeeded else {}, ensure_ascii=False
                ),
                "tool_sequence": " > ".join(tool_sequence(steps)) or "-",
                "query_success_count": len(succeeded),
                "mapping_ok": "" if mapping_ok is None else ("1" if mapping_ok else "0"),
                "params_ok": "" if params_ok is None else ("1" if params_ok else "0"),
                "error_class": label,
                "verdict": "通过" if label == "OK" else "不通过",
                "iterations": getattr(answer, "iterations", 0),
                "elapsed_ms": round(getattr(answer, "elapsed_ms", 0.0), 1),
                "total_tokens": usage.get("total_tokens", 0),
                "error": error,
            }
        )

        flag = "✓" if label == "OK" else "✗"
        print(
            f"[{index:>2}/{len(cases)}] {flag} {case.case_id:<4} {label:<4} "
            f"查数={len(succeeded)} 轮次={rows[-1]['iterations']} "
            f"{rows[-1]['elapsed_ms']:>7.0f}ms  {case.question[:34]}"
        )

    # -----------------------------------------------------------------
    # 指标计算
    # -----------------------------------------------------------------
    need = [r for r in rows if r["expect_query"] == "是"]
    no_need = [r for r in rows if r["expect_query"] == "否"]

    # 工具选择准确率：只算「该查数」的用例（不该查的用例没有指标可选）
    mapping_checked = [r for r in need if r["mapping_ok"] != ""]
    mapping_hits = [r for r in mapping_checked if r["mapping_ok"] == "1"]

    params_checked = [r for r in need if r["params_ok"] != ""]
    params_hits = [r for r in params_checked if r["params_ok"] == "1"]

    # 二分类混淆矩阵：正类 = 「需要调用工具」
    tp = sum(1 for r in need if r["query_success_count"] > 0)
    fn = len(need) - tp
    fp = sum(1 for r in no_need if r["query_success_count"] > 0)
    tn = len(no_need) - fp

    recall = tp / (tp + fn) if (tp + fn) else 0.0        # 调用完整性
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    miss_rate = fn / len(need) if need else 0.0          # 遗漏率 = 1 - 完整性
    over_rate = fp / len(no_need) if no_need else 0.0    # 过度调用率

    mapping_rate = len(mapping_hits) / len(mapping_checked) if mapping_checked else None
    params_rate = len(params_hits) / len(params_checked) if params_checked else None

    # 流程合规率（附加指标）：该查数的用例里，有没有先 get_metric_detail 再取数
    flow_ok = sum(
        1
        for r in need
        if "get_metric_detail" in r["tool_sequence"] and "query_metric" in r["tool_sequence"]
    )
    flow_rate = flow_ok / len(need) if need else None

    # 错误分布
    distribution: dict[str, int] = {}
    for row in rows:
        distribution[row["error_class"]] = distribution.get(row["error_class"], 0) + 1

    def mark(value: float | None, target: float, higher_is_better: bool = True) -> str:
        """对照合格线打标。分母为 0（本批次没有这类用例）时不参与判定。

        为什么要显式处理 None？因为「没有可测的用例」和「测了但没通过」
        是两件完全不同的事 —— 前者不该出现在不合格清单里，否则
        跑 --limit 3 这种小批量时报告会满屏假警报。
        """
        if value is None:
            return "不适用"
        ok = value >= target if higher_is_better else value <= target
        return "合格" if ok else "★不合格"

    def show(value: float | None) -> str:
        return "—" if value is None else format_rate(value)

    # -----------------------------------------------------------------
    # 写 CSV
    # -----------------------------------------------------------------
    # 产物按「谁读」分两处落盘：
    #   CSV → 04_评测原始数据（程序 / Excel 读，用来复核）
    #   txt → 03_评测报告（人读的结论，含合格线对照与 badcase）
    # 目录常量取自 config，脚本不自己拼路径字符串。
    raw_dir = cfg.DOCS_RAWDATA_DIR
    report_dir = cfg.DOCS_REPORT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "工具调用测试报告_Phase1.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------------------
    # 写文本报告
    # -----------------------------------------------------------------
    total_tokens = sum(r["total_tokens"] for r in rows)
    total_ms = sum(r["elapsed_ms"] for r in rows)
    lines: list[str] = []
    add = lines.append

    add("=" * 86)
    add("工具调用准确性专项评测报告（测试方案 Phase 1 · 维度一）")
    add("=" * 86)
    add(f"用例总数：{len(rows)}（需要调用工具 {len(need)} 条 / 不应调用 {len(no_need)} 条）")
    add(f"供应商：{getattr(agent.llm_client, 'provider_order', '?')}")
    add(f"平均耗时：{total_ms / len(rows):.0f} ms    总 token：{total_tokens:,}")
    add("")
    add("-" * 86)
    add("一、五项核心指标 vs 合格标准")
    add("-" * 86)
    add(f"  ① 工具选择准确率  {show(mapping_rate):>8}  "
        f"（{len(mapping_hits)}/{len(mapping_checked)}）  标准 ≥90%   {mark(mapping_rate, TARGET_MAPPING)}")
    add(f"  ② 参数提取准确率  {show(params_rate):>8}  "
        f"（{len(params_hits)}/{len(params_checked)}）  标准 ≥85%   {mark(params_rate, TARGET_PARAMS)}")
    add(f"  ③ 调用完整性      {show(recall):>8}  "
        f"（{tp}/{len(need)}）  遗漏率 {show(miss_rate)} ≤5%   {mark(miss_rate, TARGET_MISS, False)}")
    add(f"  ④ 过度调用率      {show(over_rate):>8}  "
        f"（{fp}/{len(no_need)}）  标准 ≤10%   {mark(over_rate, TARGET_OVER, False)}")
    add(f"  ⑤ 工具调用 F1     {f1:>8.4f}  "
        f"（P={precision:.4f} / R={recall:.4f}）  标准 ≥0.85   {mark(f1, TARGET_F1)}")
    add("")
    add(f"  附加 · 流程合规率  {show(flow_rate):>8}  "
        f"（{flow_ok}/{len(need)}，指先确认口径再取数）")
    add("")
    add("  混淆矩阵（正类 = 需要调用工具）")
    add(f"    TP={tp}  FN={fn}（该查没查）  FP={fp}（不该查却查）  TN={tn}")
    add("")
    add("-" * 86)
    add("二、错误分布（MultiCAT-Bench 六类分级）")
    add("-" * 86)
    for code in ["OK", "GE", "NCA", "NNC", "WNC", "WTC", "WA"]:
        count = distribution.get(code, 0)
        if count or code == "OK":
            add(f"  {code:<4} {ERROR_LABELS[code]:<22} {count:>3} 条")
    add("")
    add("-" * 86)
    add("三、不通过用例明细")
    add("-" * 86)
    bad = [r for r in rows if r["verdict"] == "不通过"]
    if not bad:
        add("  无。全部用例工具调用行为符合预期。")
    for row in bad:
        add(f"  [{row['case_id']}] {row['error_class']} — {ERROR_LABELS[row['error_class']]}")
        add(f"       问：{row['question']}")
        add(f"       期望指标：{row['expected_metric_id']}   期望参数：{row['expected_params']}")
        add(f"       实际指标：{row['actual_metric_ids']}")
        add(f"       实际参数：{row['actual_params']}")
        add(f"       工具序列：{row['tool_sequence']}")
        if row["error"]:
            add(f"       错误：{row['error']}")
        add("")
    add("-" * 86)
    add(f"四、逐例明细（CSV 见 docs/{cfg.DOCS_RAWDATA_DIR.name}/工具调用测试报告_Phase1.csv）")
    add("-" * 86)
    add(f"  {'用例':<5}{'类别':<8}{'应查':<5}{'实际查数':<7}{'指标':<5}{'参数':<5}{'轮次':<5}{'分类':<5}提问")
    for row in rows:
        add(
            f"  {row['case_id']:<5}{row['category']:<8}{row['expect_query']:<5}"
            f"{row['query_success_count']:<7}"
            f"{(row['mapping_ok'] or '-'):<5}{(row['params_ok'] or '-'):<5}"
            f"{row['iterations']:<5}{row['error_class']:<5}{row['question'][:30]}"
        )
    add("")

    # -----------------------------------------------------------------
    # 第五节：SQL 安全校验（对应测试方案 Phase 1 的另一半）
    # -----------------------------------------------------------------
    security_rows, intact = run_security_probe()
    blocked = sum(1 for r in security_rows if r["verdict"] == "已拦截")
    neutralized = sum(1 for r in security_rows if r["verdict"].startswith("已中和"))
    suspicious = [r for r in security_rows if r["verdict"] == "★可疑"]

    add("-" * 86)
    add("五、SQL 安全校验（攻击载荷实测）")
    add("-" * 86)
    add(f"  载荷总数 {len(security_rows)}   已拦截 {blocked}   已中和 {neutralized}   "
        f"可疑 {len(suspicious)}")
    add("")
    add(f"  {'攻击载荷':<30}{'判定':<18}{'行数':<6}拦截原因 / 实际返回")
    for row in security_rows:
        add(f"  {row['payload']:<30}{row['verdict']:<18}{row['rows']:<6}{row['reason'][:70]}")
    add("")
    add("  数据库完好性核对（用独立的只读连接从外部核对，不走被测代码的连接）")
    add(f"    表数量：攻击前 {intact['tables_before']} → 攻击后 {intact['tables_after']}   "
        f"{'一致' if intact['tables_equal'] else '★不一致'}")
    for table, (before, after) in intact["counts"].items():
        flag = "一致" if before == after else "★不一致"
        add(f"    {table:<24}{before:>9,} → {after:>9,}   {flag}")
    add("")
    add("  说明：")
    add("    · 「已拦截」= 参数校验/注册表查找直接拒绝，查询根本没发出去")
    add("    · 「已中和」= 查询执行了，但注入条件失效（结果集未被放大）")
    add("      例：日期参数传 \"2026-09-17' OR '1'='1\"，解析器抢救出合法日期后")
    add("      按参数绑定执行，只返回 1 行 —— 永真条件完全没有生效")
    add("    · 自动化断言版本见 tests/test_security.py（33 条，含组合攻击后")
    add("      核对表结构与行数不变的总验收用例）")
    add("")
    add("=" * 86)

    report = "\n".join(lines)
    report_path = report_dir / "工具调用测试报告_Phase1.txt"
    report_path.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"\nCSV  → {csv_path}")
    print(f"报告 → {report_path}")

    # 有不合格项时返回 1，方便以后接 CI（分母为 0 的指标不参与判定）
    checks = [
        mark(mapping_rate, TARGET_MAPPING),
        mark(params_rate, TARGET_PARAMS),
        mark(miss_rate, TARGET_MISS, False),
        mark(over_rate, TARGET_OVER, False),
        mark(f1, TARGET_F1),
    ]
    return 0 if all(item != "★不合格" for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
