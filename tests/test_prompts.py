# -*- coding: utf-8 -*-
"""
系统提示词单元测试
================================================================================
【提示词也要写测试吗？】

  要。而且提示词的测试价值比看起来高得多，原因有三个：

    1. 提示词是「产品的行为规范」。删掉「禁止编造」那一句，代码不会报错，
       但 Agent 会开始一本正经地编数字 —— 这种回归完全没有编译期保护，
       只能靠测试守住。

    2. 指标目录是**运行时拼接**进提示词的。加指标、改别名、改参数名，
       都可能悄悄破坏拼接（比如某个指标没被渲染进去），
       模型就再也看不到它了。这类问题必须用「遍历所有指标」的测试来兜。

    3. 它把「提示词里到底该有哪些约束」变成了一份可执行的文档。
       可以直接说：我的提示词有 6 条硬性规则，每条都有对应测试。
"""

from __future__ import annotations

from datetime import date

from src import config as cfg
from src.agent.prompts import build_system_prompt
from src.metrics import get_registry


def test_prompt_contains_data_boundary():
    """数据窗口必须写进提示词 —— 这是 Agent 能「诚实承认知识缺失」的前提。"""
    prompt = build_system_prompt()

    assert cfg.DATA_START.isoformat() in prompt
    assert cfg.DATA_END.isoformat() in prompt
    assert "数据窗口" in prompt
    # 必须教会模型「最近 7 天」具体指哪一段，否则它会自己乱算日期
    assert "最近 7 天" in prompt


def test_prompt_uses_injected_data_window():
    """窗口是可注入的，方便测试与将来做「历史时点回放」的评测。"""
    prompt = build_system_prompt(data_start=date(2026, 1, 1), data_end=date(2026, 1, 31))

    assert "2026-01-01" in prompt
    assert "2026-01-31" in prompt
    # 默认「最近 7 天」的起点应该是 1 月 25 日
    assert "「最近 7 天」指的是 2026-01-25 ~ 2026-01-31" in prompt


def test_prompt_embeds_every_metric_id():
    """每一个指标都必须出现在目录里。

    这条是防回归的关键：将来往注册表加第 13 个指标时，
    如果目录渲染出了问题，这条测试会立刻失败 ——
    否则表现是「模型好像不认识这个指标」，排查起来极其费时。
    """
    prompt = build_system_prompt()

    for metric_id in get_registry().ids():
        assert metric_id in prompt


def test_prompt_forbids_fabricating_numbers():
    """「禁止编造」是本项目的第一原则，必须显式写在提示词里。

    （提示词是软约束，真正的硬约束是「唯一取数工具是 query_metric」。
      软硬结合才可靠 —— 只有工具约束，模型可能仍然用记忆里的数字凑答案。）
    """
    prompt = build_system_prompt()

    assert "禁止编造" in prompt
    assert "query_metric" in prompt


def test_prompt_requires_human_confirmation_for_big_decisions():
    """重大决策要人工确认 —— 这是写进项目硬约束里的产品规则，必须有测试守着。"""
    prompt = build_system_prompt()

    assert "人工复核后执行" in prompt
    assert "版本调优" in prompt


def test_prompt_requires_sample_size_in_comparisons():
    """对比类结论必须报样本量，否则小样本渠道的留存率会被过度解读。"""
    prompt = build_system_prompt()
    assert "cohort_size" in prompt


def test_prompt_mentions_correlation_not_causation():
    prompt = build_system_prompt()
    assert "相关不等于因果" in prompt


def test_prompt_describes_the_workflow_order():
    """工作流程里的步骤必须按「目录 → 口径 → 取数」的顺序出现。

    注意：不能直接在整段提示词里找这三个词 —— 指标目录那一段也提到了
    get_metric_detail（「具体口径必须调用 get_metric_detail 获取」），
    会先于工作流程被匹配到。所以要先把【工作流程】这一节切出来再断言，
    测的才是「流程顺序」这件事本身。
    """
    prompt = build_system_prompt()

    workflow = prompt[prompt.index("【工作流程】"):prompt.index("【硬性规则】")]
    order = [
        workflow.index("list_metrics"),
        workflow.index("get_metric_detail"),
        workflow.index("query_metric"),
    ]
    assert order == sorted(order)
    # 四步都在，且都带序号（序号是给模型看的「分步执行」信号）
    for step in ("1.", "2.", "3.", "4."):
        assert step in workflow


def test_prompt_glossary_is_dynamic_not_hardcoded():
    """目录来自注册表，而不是提示词里手写的。

    验证方式：注入一个「只有 dau 一个指标」的假注册表，
    提示词里就只应该有 dau —— 如果提示词是硬编码的，这条会失败。
    """
    import json
    import tempfile
    from pathlib import Path

    from src.metrics.registry import MetricRegistry

    source = json.loads(
        Path(get_registry().path).read_text(encoding="utf-8")
    )
    source["metrics"] = [m for m in source["metrics"] if m["metric_id"] == "dau"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mini_registry.json"
        path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

        prompt = build_system_prompt(registry=MetricRegistry(path))

    assert "dau" in prompt
    assert "retention_rate" not in prompt
    assert "共 1 个" in prompt


def test_prompt_separates_data_facts_from_external_knowledge():
    """Phase 5 新增的第 7 条：日历归因与行业基准必须标明来源。

    背景：Phase 3 幻觉评测的 11 条「无依据声明」里，有 10 条是同一类问题 ——
    数据表里**没有星期字段**，模型却说「周末效应」「周末拉新节奏好」；
    数据里没有对标样本，模型却断言「LTV/CAC 大于 3 才算健康」「该品类中等偏下」。

    ★ 这条测试刻意做成**双向断言**，两个方向缺一不可：
      · 正向：必须要求模型区分「数据事实」与「外部经验」并标注来源；
      · 反向：必须同时写明「不是让你少给业务解读」。
        少了反向这一句，提示词很容易被优化成"什么都不说"——
        一个不敢做业务解读的分析师，在真实工作里是没用的。
        这和第 4 章 R08「会拒绝 vs 会分级拒绝」是同一个道理：
        优化的目标不是让模型更保守，而是让它的边界更清楚。
    """
    prompt = build_system_prompt()

    assert "外部经验" in prompt            # 要求区分数据事实与外部经验
    assert "周末" in prompt                # 点名「日历归因」这一类
    assert "LTV/CAC" in prompt             # 点名「行业基准」这一类
    assert "需引入外部数据验证" in prompt    # 给出可操作的替代表述
    assert "不是让你少给业务解读" in prompt   # ★ 反向断言：防止优化成过度保守


def test_prompt_is_not_empty_and_has_sections():
    prompt = build_system_prompt()

    assert len(prompt) > 1000                       # 太短说明有段落没拼上
    for section in ("【你的数据边界】", "【可用指标目录】", "【工作流程】", "【硬性规则】", "【回答风格】"):
        assert section in prompt


# ---------------------------------------------------------------------------
# Phase 6 · 提示词固定开销
# ---------------------------------------------------------------------------

def test_catalog_header_appears_exactly_once():
    """指标目录的表头只能出现一次。

    背景：Phase 6 精简时发现的真实缺陷 —— 提示词里手写了一遍
    「【可用指标目录】」，而 to_catalog_text() 自己也会输出同样的表头，
    于是每次请求都白送模型一遍重复信息。
    提示词里的重复不会让模型更听话，只会让每一轮都多付一次 token，
    所以要用测试把它钉死。
    """
    prompt = build_system_prompt()
    assert prompt.count("【可用指标目录】") == 1


def test_catalog_uses_compact_lines_without_category():
    """系统提示词里的目录行用紧凑版（不带「分类」），工具里才用完整版。

    这是「提示词省着放、工具放全」这条设计原则的落地检查。
    """
    prompt = build_system_prompt()
    assert "分类:" not in prompt


def test_prompt_fixed_cost_is_stable():
    """固定开销（字符数）设上限，防止悄悄膨胀。

    为什么用字符数而不是 token 数？因为 token 需要联网分词，测试不能联网。
    中文字符与 token 大致 1:1，字符数完全够用作「预警线」。
    Phase 6 精简后实测 2,722 字符；Phase 5 新增第 7 条（外部经验标注）后
    实测 2,931 字符，仍在上限内。这个上限不是不让加内容，
    而是让「加内容」这个动作必须显式地来改这个数字。
    """
    prompt = build_system_prompt()
    assert len(prompt) < 3000, f"系统提示词固定开销已膨胀到 {len(prompt)} 字符"