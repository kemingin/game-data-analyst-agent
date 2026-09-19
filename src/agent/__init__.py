# -*- coding: utf-8 -*-
"""Agent 层（Phase 3）。

四个模块，各管一件事，串起来就是一次完整的问答：

    llm_client.py   跟大模型说话（主备降级、重试、用量统计）
    tools.py        让大模型能「动手」的三个工具（目录 / 口径 / 取数）
    prompts.py      系统提示词（角色、流程、硬性规则、输出风格）
    react_agent.py  主循环（想 → 做 → 看 → 再想，直到给出最终回答）

对外只需要记住三个名字：

    GameDataAgent   主入口，调用 agent.ask("问题")
    LLMClient       需要单独调模型时用（比如 Phase 5 评测里的 LLM-as-Judge）
    ToolExecutor    需要单独跑指标时用（不经过大模型）
"""

from src.agent.llm_client import LLMClient, LLMResponse, ToolCall
from src.agent.prompts import build_system_prompt
from src.agent.react_agent import AgentAnswer, AgentStep, GameDataAgent
from src.agent.tools import TOOL_DEFINITIONS, ToolExecutor, ToolResult

__all__ = [
    "TOOL_DEFINITIONS",
    "AgentAnswer",
    "AgentStep",
    "GameDataAgent",
    "LLMClient",
    "LLMResponse",
    "ToolCall",
    "ToolExecutor",
    "ToolResult",
    "build_system_prompt",
]