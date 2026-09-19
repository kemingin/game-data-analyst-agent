# -*- coding: utf-8 -*-
"""
评测体系（Phase 5）
================================================================================
回答一个上线前必答的问题：**你怎么知道这个 Agent 答得对？**

    metrics.py     评测算法（纯函数：数字抽取、对账、三类准确率、汇总）
    eval_set.py    评测集加载器（把期望参数解析成真实取值）
    eval_set.json  18 条标准问答用例（数据文件，可被产品/运营直接评审）
    runner.py      执行器（跑 Agent + 独立算真值 + 生成报告）

核心设计：**双路径对账**。
  路径 A 把问题交给 Agent，路径 B 用期望指标+参数直接查一次作为真值。
  两条路径互不相干，所以 Agent 查错了范围时无法"自己印证自己"。

配套脚本：scripts/run_eval.py（命令行跑评测并输出报告）
"""

from src.eval.eval_set import EvalCase, EvalSet, load_eval_set
from src.eval.metrics import (
    EvalSummary,
    NumberCheck,
    check_answer_numbers,
    check_metric_mapping,
    check_params,
    extract_numbers,
    format_rate,
    metric_calls,
    summarize,
)
from src.eval.runner import CaseResult, EvalRunner, build_report, run_eval

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalRunner",
    "EvalSet",
    "EvalSummary",
    "NumberCheck",
    "build_report",
    "check_answer_numbers",
    "check_metric_mapping",
    "check_params",
    "extract_numbers",
    "format_rate",
    "load_eval_set",
    "metric_calls",
    "run_eval",
    "summarize",
]