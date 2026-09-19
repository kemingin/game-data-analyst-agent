# -*- coding: utf-8 -*-
"""
项目统一异常定义（Phase 2）
================================================================================
【为什么要统一异常，而不是到处 raise ValueError？】

  1. 分层定位问题：
     我们有三层各自可能出错 —— 指标层（找不到指标、参数不合法）、
     SQL 安全层（SQL 越权）、执行层（数据库报错）。
     用不同的异常类，上层就能「按错误类型分流」：
       - MetricNotFoundError → 提示用户「我还不认识这个指标」，并推荐相近指标
       - MetricParamError    → 提示用户「时间区间不对」，让 LLM 重新抽取参数
       - SQLSecurityError    → 直接拦截，记录安全日志，不给用户看数据库细节
       - SQLExecutionError   → 提示「查询超时/执行失败」，让 LLM 换个问法

  2. 设计要点：这是「异常也是 API 的一部分」的思路。
     如果全都抛 Exception，上层只能用字符串匹配来猜错误类型，非常脆弱。

  3. 所有异常都继承自 GameAgentError，这样在上层最外层
     只需要 `except GameAgentError` 一个兜底，就能保证 Agent 不会因为
     一次指标匹配失败而整个崩掉。
"""

from __future__ import annotations


class GameAgentError(Exception):
    """本项目所有业务异常的基类。

    只要继承它，上层就可以安全地统一兜底：
        try:
            ...
        except GameAgentError as e:
            return {"ok": False, "message": e.message}
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # 单独存一份，方便日志/前端直接取（str(e) 也能拿到，但显式一点更清楚）
        self.message = message


class MetricNotFoundError(GameAgentError):
    """指标注册表里找不到对应的指标 ID。

    典型场景：LLM 把用户问题映射成了一个不存在的 metric_id，
    或者用户问的是「昨天食堂吃什么」这种跟数据完全无关的问题。
    """


class MetricParamError(GameAgentError):
    """指标参数不合法：缺失必填参数、类型不对、枚举值不在允许范围内、日期区间倒挂等。"""


class SQLSecurityError(GameAgentError):
    """SQL 触发了安全规则：非 SELECT、写入类关键字、表不在白名单、多语句等。

    注意：这个异常一旦出现，说明「模板被改坏了」或「有人试图注入」，
    应当记录安全日志并告警，而不是静默降级。
    """


class SQLExecutionError(GameAgentError):
    """SQL 语法正确但执行失败：表不存在、超时被中断、数据库文件被占用等。"""


class LLMError(GameAgentError):
    """调用大模型失败。

    覆盖：鉴权失败（Key 错误）、网络异常、限流、所有供应商都不可用等。
    注意：主备降级已经在 LLMClient 内部做完了 —— 只有当**所有**供应商
    都失败时才会抛出这个异常，所以它出现就意味着「本次分析彻底做不了」，
    上层应该给用户一句诚实的说明，而不是继续硬撑。
    """


class LLMNotConfiguredError(LLMError):
    """一个可用的 LLM 供应商都没有（所有 API Key 都为空）。

    这是「部署配置问题」而不是「运行时故障」，所以要跟普通 LLMError 分开：
    前者要提示用户去配 .env，后者只能提示「稍后重试」。
    """