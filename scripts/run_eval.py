# -*- coding: utf-8 -*-
"""
评测命令行入口（Phase 5）
================================================================================
用法：

    # 跑全量评测（18 条，会真实调用大模型，注意 token 消耗）
    python scripts/run_eval.py

    # 先跑 3 条验证流程通了再跑全量 —— 开发期省钱的基本纪律
    python scripts/run_eval.py --limit 3

    # 只跑指定的几条
    python scripts/run_eval.py --cases E01,E07,E17

    # 只打印不落盘
    python scripts/run_eval.py --no-report

    # 顺带把模型原话存成 JSON，便于离线复核「存疑数字」
    python scripts/run_eval.py --dump docs/04_评测原始数据/评测结果_Phase6.json

输出：
    终端实时进度 + 总览；报告默认写入 docs/03_评测报告/评测报告_Phase6.txt

【为什么默认要落盘一份报告？】

  因为评测的价值在于**可对比**。这次 85%，下次改完 prompt 92%，
  这个提升必须有一份文件作为证据。只打印到终端，关掉窗口就没了，
  下次只能凭印象说"好像变好了"—— 那就退化成了没有评测。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，这样直接 `python scripts/run_eval.py` 也能 import src.*
# （和 tests/conftest.py 里做的是同一件事，理由相同：脚本所在目录才是 sys.path[0]）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg                     # noqa: E402
from src.eval.eval_set import EvalCase, load_eval_set   # noqa: E402
from src.eval.metrics import format_rate, summarize     # noqa: E402
from src.eval.runner import CaseResult, EvalRunner, build_report  # noqa: E402

DEFAULT_REPORT_PATH = cfg.DOCS_REPORT_DIR / "评测报告_Phase6.txt"


def print_progress(index: int, total: int, case: EvalCase, result: CaseResult) -> None:
    """实时进度。

    刻意把「映射 / 参数 / 数字对账」三个结果都打出来，而不是只打一个"完成"——
    真机跑一条要十几秒，如果只显示进度条，中间出现问题时你得等全部跑完才知道。
    边跑边看结果，才能在第一条就发现"是不是 prompt 写崩了"。
    """
    mapping = "不适用" if result.metric_mapping_ok is None else ("✓" if result.metric_mapping_ok else "✗")
    params = "不适用" if result.params_ok is None else ("✓" if result.params_ok else "✗")
    if result.number_check is None:
        numbers = "不适用"
    else:
        numbers = f"{result.number_check.matched_count}/{result.number_check.total}"

    print(
        f"  [{index:>2}/{total}] {case.case_id} 映射{mapping} 参数{params} "
        f"数字{numbers} | {result.iterations} 轮 "
        f"{result.usage.get('total_tokens', 0):>6,} tokens "
        f"{result.elapsed_ms / 1000:>5.1f}s | {case.question}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Agent 标准问答评测")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（开发期省钱用）")
    parser.add_argument("--cases", type=str, default=None, help="只跑指定用例，逗号分隔，如 E01,E07")
    parser.add_argument("--out", type=str, default=str(DEFAULT_REPORT_PATH), help="报告输出路径")
    parser.add_argument("--no-report", action="store_true", help="只打印，不写报告文件")
    parser.add_argument("--dump", type=str, default=None,
                        help="把逐例结构化结果（含模型原话）写成 JSON，便于离线复核存疑数字")
    args = parser.parse_args()

    eval_set = load_eval_set()
    print("=" * 78)
    print("游戏数据智能分析师 Agent · 标准问答评测")
    print("=" * 78)
    print(f"评测集：{eval_set.path.name} v{eval_set.meta.get('version', '-')}"
          f"｜共 {len(eval_set)} 条用例")
    print(f"分类覆盖：{eval_set.by_category()}")
    print(f"数据窗口：{cfg.DATA_START} ~ {cfg.DATA_END}")
    print("-" * 78)

    case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None

    runner = EvalRunner(eval_set=eval_set)
    cases = eval_set.all()
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.case_id in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"本次将运行 {len(cases)} 条用例（每条都会真实调用大模型）")
    print("-" * 78)

    results = runner.run_all(cases, progress=print_progress)

    # ---------------- 终端总览 ----------------
    summary = summarize(results)
    print("-" * 78)
    print("总览")
    print("-" * 78)
    print(f"  指标映射准确率    {format_rate(summary.metric_mapping_rate)}"
          f"  ({summary.mapping_correct}/{summary.mapping_checked})")
    print(f"  参数抽取准确率    {format_rate(summary.param_rate)}"
          f"  ({summary.param_correct}/{summary.param_checked})")
    print(f"  答案数字可追溯率  {format_rate(summary.number_rate)}"
          f"  ({summary.number_matched}/{summary.number_total})")
    if summary.refuse_checked:
        print(f"  超范围拒答正确率  {format_rate(summary.refuse_rate)}"
              f"  ({summary.refused_correctly}/{summary.refuse_checked})")
    print(f"  平均 token        {summary.avg_tokens:,.1f}（总计 {summary.total_tokens:,}）")
    if summary.total_cached_tokens:
        print(f"  缓存命中率        {format_rate(summary.cache_hit_rate)}"
              f"  ({summary.total_cached_tokens:,}/{summary.total_prompt_tokens:,} 输入 token)")
    print(f"  平均耗时          {summary.avg_elapsed_ms:,.0f} ms"
          f"｜平均 {summary.avg_iterations} 轮")

    # ---------------- 报告落盘 ----------------
    if not args.no_report:
        report = build_report(results, eval_set=eval_set)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # encoding="utf-8" 必须写：Windows 默认 GBK，报告里有中文和 ✓ ✗ 会直接报错
        out_path.write_text(report, encoding="utf-8")
        print("-" * 78)
        print(f"完整报告已写入：{out_path}")

    # ---------------- 结构化结果落盘（可选） ----------------
    # 为什么要有这一项？因为报告里只会写「存疑数字：25」，
    # 但人复核时需要看到**模型原话的上下文**才能判断它到底是编造还是合理表达。
    # 把模型回答一并存下来，才能事后离线复查，不用重跑（重跑一次要烧 13 万 token）。
    if args.dump:
        import json

        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "eval_set": eval_set.path.name,
                "eval_set_version": eval_set.meta.get("version"),
                "data_window": [str(cfg.DATA_START), str(cfg.DATA_END)],
            },
            "summary": {
                "total": summary.total,
                "metric_mapping_rate": summary.metric_mapping_rate,
                "param_rate": summary.param_rate,
                "number_rate": summary.number_rate,
                "refuse_rate": summary.refuse_rate,
                "avg_tokens": summary.avg_tokens,
                "avg_elapsed_ms": summary.avg_elapsed_ms,
                "avg_iterations": summary.avg_iterations,
            },
            "cases": [r.to_dict() for r in results],
        }
        dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结构化结果已写入：{dump_path}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())