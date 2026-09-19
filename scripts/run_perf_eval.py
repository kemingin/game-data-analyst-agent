# -*- coding: utf-8 -*-
"""缓存命中率 + 响应性能专项评测（测试方案 Phase 2 · 维度二 + 维度五）

================================================================================
【这个脚本解决什么问题？】

  Phase 5 的评测回答"答得对不对"，Phase 1 回答"工具调对了吗"，
  本脚本回答的是**"快不快、贵不贵"** —— 这是产品能不能上线的问题。

  为什么这两件事要放在一起测？
    因为它们共享同一个根因：**ReAct 循环每多走一轮，就把「系统提示词 + 历史」
    整包重发一次**。所以轮次既推高延迟，又推高 token 成本；
    而提示词前缀能不能被缓存命中，直接决定这部分重复开销是按全价还是 1/10 计价。

【核心口径：缓存命中率怎么算？】

  供应商官方口径（Anthropic 系）：
      cache_hit_rate = cacheRead / (input + cacheRead + cacheWrite)

  DeepSeek 的口径（本项目主供应商）：
      prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens
      cache_hit_rate = prompt_cache_hit_tokens / prompt_tokens

  ★ 两者其实是同一个式子：DeepSeek 没有 cacheWrite（不额外收写入费），
    它的 prompt_cache_miss_tokens 就等于上式的 input。
    所以本项目统一用 **命中 token / 输入 token**，两种口径下结果一致。

【为什么要拦截 LLM 客户端的 chat 方法？】

  生产代码里，usage 是**按一次问答累计**的（一次问答含 2~4 次 LLM 调用），
  累计之后就分不清"哪一次命中、哪一次没命中"。
  而"缓存利用率（有命中的请求占比）"必须按**单次调用**统计才有意义。

  所以这里用测试探针（monkey patch）在客户端外面包一层记录每次调用，
  **不改动任何生产代码** —— 这是评测脚本该有的姿态：
  为了测量而改被测对象，测出来的就不是它了。

【产物】
  docs/04_评测原始数据/性能测试_Phase2_逐次调用.csv   每次 LLM 调用的 usage 与耗时
  docs/04_评测原始数据/性能测试_Phase2_逐用例.csv     每条用例的端到端耗时拆解
  docs/03_评测报告/性能与缓存测试报告_Phase2.txt      指标汇总 + 合格线对照
"""

from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src import config as cfg                                      # noqa: E402
from src.agent.react_agent import GameDataAgent                    # noqa: E402
from src.eval.eval_set import load_eval_set                        # noqa: E402
from src.eval.metrics import format_rate                           # noqa: E402

# ---------------------------------------------------------------------------
# 合格标准（来自测试方案）
# ---------------------------------------------------------------------------
TARGET_CACHE_RATE = 0.70        # 缓存命中率 ≥70%
TARGET_CACHE_UTIL = 0.80        # 缓存利用率（有命中的请求占比）≥80%
TARGET_LATENCY_SIMPLE = 10_000  # 简单查询 ≤10s
TARGET_LATENCY_COMPLEX = 30_000  # 复杂分析 ≤30s
TARGET_SQL_MS = 2_000           # 单次 SQL 执行 ≤2s

# ---------------------------------------------------------------------------
# 单价（元 / 百万 token）—— 仅用于把 token 换算成"量级可感的钱"。
# 价格会变，所以写成常量集中管理；报告里会明确标注这是估算。
# ---------------------------------------------------------------------------
PRICE_CACHE_HIT = 0.5           # 输入·缓存命中
PRICE_CACHE_MISS = 2.0          # 输入·未命中
PRICE_OUTPUT = 8.0              # 输出

WARMUP_CALLS = 2                # 冷启动调用数：前几次必然未命中，稳态口径要排除


def percentile(values: list[float], ratio: float) -> float:
    """最近秩法算分位数（不引入 numpy，几行就够）。

    为什么用最近秩而不是插值？因为延迟数据的样本量只有几十个，
    插值出来的 P99 会是一个"实际不存在"的数，反而误导人。
    最近秩法给出的永远是**真实发生过的一次耗时**。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(ratio * len(ordered))) - 1))
    return ordered[index]


class ChatRecorder:
    """LLM 客户端探针：记录每次调用的 usage 与耗时。

    为什么要记 prompt_tokens / cached_tokens 两个值而不是一个？
      因为"命中率"是比值 —— 只有分子没有分母，就看不出
      "命中率低"到底是"缓存没生效"还是"这一轮输入本来就很小"。
    """

    def __init__(self, client) -> None:
        self.client = client
        self.records: list[dict] = []
        self._original = client.chat
        client.chat = self._wrapper          # 替换实例方法（只影响这个实例）

    def _wrapper(self, messages, tools=None, tool_choice=None):
        start = time.perf_counter()
        response = self._original(messages, tools=tools, tool_choice=tool_choice)
        elapsed = (time.perf_counter() - start) * 1000
        self.records.append(
            {
                "index": len(self.records) + 1,
                "provider": getattr(response, "provider", ""),
                "model": getattr(response, "model", ""),
                "prompt_tokens": getattr(response, "prompt_tokens", 0),
                "cached_tokens": getattr(response, "cached_tokens", 0),
                "completion_tokens": getattr(response, "completion_tokens", 0),
                "total_tokens": getattr(response, "total_tokens", 0),
                "elapsed_ms": round(elapsed, 1),
            }
        )
        return response


def main() -> int:
    limit = 20
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    eval_set = load_eval_set()
    # 只取「需要查数」的用例：不该查数的用例 1 轮就结束，测不出循环开销
    cases = [c for c in eval_set.all() if not c.expect_no_query][:limit]

    agent = GameDataAgent()
    recorder = ChatRecorder(agent.llm_client)

    print(f"用例数：{len(cases)}    供应商顺序：{getattr(agent.llm_client, 'provider_order', '?')}")
    print("=" * 88)

    case_rows: list[dict] = []
    for index, case in enumerate(cases, 1):
        answer = agent.ask(case.question, history=list(case.history) or None)

        # 耗时拆解：工具耗时来自步骤记录，LLM 耗时 = 总耗时 - 工具耗时
        tool_ms = sum(
            getattr(step, "elapsed_ms", 0.0)
            for step in answer.steps
            if getattr(step, "kind", "") == "tool_call"
        )
        sql_ms = sum(
            float((getattr(step, "result", {}) or {}).get("elapsed_ms") or 0.0)
            for step in answer.steps
            if getattr(step, "kind", "") == "tool_call"
        )
        query_count = sum(
            1
            for step in answer.steps
            if getattr(step, "kind", "") == "tool_call"
            and getattr(step, "tool_name", "") == "query_metric"
        )
        usage = answer.usage or {}

        case_rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "question": case.question,
                "iterations": answer.iterations,
                "query_count": query_count,
                "complexity": "简单" if query_count <= 1 else "复杂",
                "elapsed_ms": round(answer.elapsed_ms, 1),
                "llm_ms": round(max(0.0, answer.elapsed_ms - tool_ms), 1),
                "tool_ms": round(tool_ms, 1),
                "sql_ms": round(sql_ms, 1),
                "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                "cached_tokens": usage.get("cached_tokens", 0),
                "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                "total_tokens": usage.get("total_tokens", 0),
                "ok": answer.ok,
                "error": answer.error or "",
            }
        )
        print(
            f"[{index:>2}/{len(cases)}] {case.case_id:<4} 轮次={answer.iterations} "
            f"总={answer.elapsed_ms:>7.0f}ms LLM={case_rows[-1]['llm_ms']:>7.0f}ms "
            f"工具={tool_ms:>6.0f}ms SQL={sql_ms:>5.0f}ms  {case.question[:30]}"
        )

    calls = recorder.records
    # -----------------------------------------------------------------
    # 缓存口径
    # -----------------------------------------------------------------
    total_input = sum(r["prompt_tokens"] for r in calls)
    total_cached = sum(r["cached_tokens"] for r in calls)
    total_output = sum(r["completion_tokens"] for r in calls)

    cache_rate = total_cached / total_input if total_input else 0.0
    hit_calls = [r for r in calls if r["cached_tokens"] > 0]
    cache_util = len(hit_calls) / len(calls) if calls else 0.0
    avg_cached = total_cached / len(calls) if calls else 0.0

    # 稳态口径：排除冷启动（前 WARMUP_CALLS 次必然未命中）
    steady = calls[WARMUP_CALLS:]
    steady_input = sum(r["prompt_tokens"] for r in steady)
    steady_cached = sum(r["cached_tokens"] for r in steady)
    steady_rate = steady_cached / steady_input if steady_input else 0.0
    steady_hits = [r for r in steady if r["cached_tokens"] > 0]
    steady_util = len(steady_hits) / len(steady) if steady else 0.0

    # -----------------------------------------------------------------
    # 性能口径
    # -----------------------------------------------------------------
    latencies = [r["elapsed_ms"] for r in case_rows]
    llm_latencies = [r["elapsed_ms"] for r in calls]
    sql_latencies = [r["sql_ms"] for r in case_rows if r["sql_ms"] > 0]
    simple = [r for r in case_rows if r["complexity"] == "简单"]
    complex_ = [r for r in case_rows if r["complexity"] == "复杂"]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_llm = sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0
    avg_tool = sum(r["tool_ms"] for r in case_rows) / len(case_rows) if case_rows else 0.0
    max_simple = max((r["elapsed_ms"] for r in simple), default=0.0)
    max_complex = max((r["elapsed_ms"] for r in complex_), default=0.0)
    sql_p90 = percentile(sql_latencies, 0.90)
    sql_max = max(sql_latencies, default=0.0)

    # 成本估算：命中部分按命中价，未命中部分按未命中价
    cost = (
        total_cached / 1e6 * PRICE_CACHE_HIT
        + (total_input - total_cached) / 1e6 * PRICE_CACHE_MISS
        + total_output / 1e6 * PRICE_OUTPUT
    )
    cost_per_case = cost / len(case_rows) if case_rows else 0.0

    def mark(value: float, target: float, higher_is_better: bool = True) -> str:
        ok = value >= target if higher_is_better else value <= target
        return "合格" if ok else "★不合格"

    # -----------------------------------------------------------------
    # 写 CSV
    # -----------------------------------------------------------------
    # 产物按「谁读」分两处落盘：CSV 是复核用的原始证据，txt 是人读的结论报告。
    # 目录常量统一取自 config，避免脚本各自拼字符串、改名时漏改。
    raw_dir = cfg.DOCS_RAWDATA_DIR
    report_dir = cfg.DOCS_REPORT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    calls_csv = raw_dir / "性能测试_Phase2_逐次调用.csv"
    with calls_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calls[0].keys()))
        writer.writeheader()
        writer.writerows(calls)

    cases_csv = raw_dir / "性能测试_Phase2_逐用例.csv"
    with cases_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    # -----------------------------------------------------------------
    # 写文本报告
    # -----------------------------------------------------------------
    lines: list[str] = []
    add = lines.append

    add("=" * 88)
    add("缓存命中率 + 响应性能专项评测报告（测试方案 Phase 2 · 维度二 + 维度五）")
    add("=" * 88)
    add(f"用例数：{len(case_rows)}    其中 LLM 调用 {len(calls)} 次"
        f"（平均每用例 {len(calls) / len(case_rows):.2f} 次）")
    add(f"供应商：{getattr(agent.llm_client, 'provider_order', '?')}")
    add("")
    add("-" * 88)
    add("一、缓存命中率（维度二）")
    add("-" * 88)
    add("  口径说明：DeepSeek 的 prompt_tokens = 命中 token + 未命中 token，")
    add("            没有 cacheWrite（不另收写入费），因此")
    add("            命中率 = 命中 token / 输入 token")
    add("            与官方式 cacheRead/(input+cacheRead+cacheWrite) 完全等价。")
    add("")
    add(f"  ① 缓存命中率      {format_rate(steady_rate):>8}  "
        f"（稳态口径，排除前 {WARMUP_CALLS} 次冷启动）  标准 ≥70%   "
        f"{mark(steady_rate, TARGET_CACHE_RATE)}")
    add(f"     全量口径（含冷启动）{format_rate(cache_rate):>8}  "
        f"（{total_cached:,}/{total_input:,} token）")
    add(f"  ② 缓存利用率      {format_rate(steady_util):>8}  "
        f"（{len(steady_hits)}/{len(steady)} 次调用命中）  标准 ≥80%   "
        f"{mark(steady_util, TARGET_CACHE_UTIL)}")
    add(f"  ③ 平均每请求命中  {avg_cached:>8,.0f} token / 次调用")
    add("")
    add("  逐次调用明细（前 8 次，用于观察冷启动 → 稳态的过渡）")
    add(f"    {'序号':<6}{'输入':<9}{'命中':<9}{'命中率':<10}{'耗时':<9}供应商/模型")
    for record in calls[:8]:
        rate = record["cached_tokens"] / record["prompt_tokens"] if record["prompt_tokens"] else 0.0
        add(f"    {record['index']:<6}{record['prompt_tokens']:<9,}{record['cached_tokens']:<9,}"
            f"{format_rate(rate):<10}{record['elapsed_ms']:<9.0f}"
            f"{record['provider']}/{record['model']}")
    add("")
    add("-" * 88)
    add("二、响应性能（维度五）")
    add("-" * 88)
    add(f"  {'指标':<26}{'实测':>12}{'合格线':>12}   判定")
    add(f"  {'端到端·平均':<24}{avg_latency:>12,.0f}ms")
    add(f"  {'端到端·P50':<24}{percentile(latencies, 0.50):>12,.0f}ms")
    add(f"  {'端到端·P90':<24}{percentile(latencies, 0.90):>12,.0f}ms")
    add(f"  {'端到端·P99':<24}{percentile(latencies, 0.99):>12,.0f}ms")
    add(f"  {'简单查询·最慢':<23}{max_simple:>12,.0f}ms{'≤10,000ms':>12}   "
        f"{mark(max_simple, TARGET_LATENCY_SIMPLE, False)}")
    add(f"  {'复杂分析·最慢':<23}{max_complex:>12,.0f}ms{'≤30,000ms':>12}   "
        f"{mark(max_complex, TARGET_LATENCY_COMPLEX, False)}")
    add(f"  {'单次 LLM 调用·平均':<21}{avg_llm:>12,.0f}ms")
    add(f"  {'单次 LLM 调用·P90':<22}{percentile(llm_latencies, 0.90):>12,.0f}ms")
    add(f"  {'单次 SQL·平均':<24}{sum(sql_latencies) / len(sql_latencies) if sql_latencies else 0:>12,.1f}ms")
    add(f"  {'单次 SQL·P90':<25}{sql_p90:>12,.1f}ms{'≤2,000ms':>12}   "
        f"{mark(sql_p90, TARGET_SQL_MS, False)}")
    add(f"  {'单次 SQL·最慢':<24}{sql_max:>12,.1f}ms")
    add("")
    add("  耗时构成（端到端时间花在哪）")
    llm_share = avg_latency - avg_tool
    add(f"    LLM 推理  {llm_share:>8,.0f}ms   占 {format_rate(llm_share / avg_latency if avg_latency else 0)}")
    add(f"    工具+SQL  {avg_tool:>8,.0f}ms   占 {format_rate(avg_tool / avg_latency if avg_latency else 0)}")
    add("")
    add("  简单 / 复杂 分组")
    add(f"    简单（查 1 个指标）{len(simple):>3} 条   平均 "
        f"{sum(r['elapsed_ms'] for r in simple) / len(simple) if simple else 0:>7,.0f}ms   平均轮次 "
        f"{sum(r['iterations'] for r in simple) / len(simple) if simple else 0:.2f}")
    add(f"    复杂（查 ≥2 个指标）{len(complex_):>2} 条   平均 "
        f"{sum(r['elapsed_ms'] for r in complex_) / len(complex_) if complex_ else 0:>7,.0f}ms   平均轮次 "
        f"{sum(r['iterations'] for r in complex_) / len(complex_) if complex_ else 0:.2f}")
    add("")
    add("-" * 88)
    add("三、Token 消耗与成本（维度五）")
    add("-" * 88)
    add(f"  输入 token 合计   {total_input:>10,}   平均每用例 "
        f"{total_input / len(case_rows):>8,.0f}")
    add(f"  其中缓存命中      {total_cached:>10,}   平均每用例 "
        f"{total_cached / len(case_rows):>8,.0f}")
    add(f"  输出 token 合计   {total_output:>10,}   平均每用例 "
        f"{total_output / len(case_rows):>8,.0f}")
    add("")
    add(f"  估算总费用   {cost:>8.4f} 元      单次问答 {cost_per_case:>8.4f} 元")
    add(f"  （按单价：命中 {PRICE_CACHE_HIT} / 未命中 {PRICE_CACHE_MISS} / 输出 "
        f"{PRICE_OUTPUT} 元每百万 token —— 价格会变，仅作量级参考）")
    add("")
    add("  如果完全没有缓存命中，这批用例要多花：")
    no_cache = (
        total_input / 1e6 * PRICE_CACHE_MISS + total_output / 1e6 * PRICE_OUTPUT
    )
    saved = no_cache - cost
    add(f"    未缓存费用 {no_cache:.4f} 元 − 实际 {cost:.4f} 元 = 省下 {saved:.4f} 元"
        f"（{format_rate(saved / no_cache if no_cache else 0)}）")
    add("")
    add("-" * 88)
    add(f"四、逐用例明细（CSV 见 docs/{cfg.DOCS_RAWDATA_DIR.name}/性能测试_Phase2_逐用例.csv）")
    add("-" * 88)
    add(f"  {'用例':<5}{'类型':<7}{'轮次':<5}{'查数':<5}{'总耗时':<9}{'LLM':<9}{'工具':<8}"
        f"{'SQL':<7}{'输入':<8}{'命中':<8}提问")
    for row in case_rows:
        add(f"  {row['case_id']:<5}{row['complexity']:<7}{row['iterations']:<5}"
            f"{row['query_count']:<5}{row['elapsed_ms']:<9.0f}{row['llm_ms']:<9.0f}"
            f"{row['tool_ms']:<8.0f}{row['sql_ms']:<7.0f}{row['input_tokens']:<8,}"
            f"{row['cached_tokens']:<8,}{row['question'][:24]}")
    add("")
    add("=" * 88)

    report = "\n".join(lines)
    report_path = report_dir / "性能与缓存测试报告_Phase2.txt"
    report_path.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"\nCSV → {calls_csv}")
    print(f"CSV → {cases_csv}")
    print(f"报告 → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
