"""M11 分析代理：有界工具调用循环（无 LLM 也能跑）. """
from spxh.core.agent.loop import BUDGETS, OFFLINE_PLAN, agent_tools_payload, run_agent
from spxh.core.agent.tools import TOOLS, ToolResult, build_evidence, merge_evidence, run_tool, tool_schemas

__all__ = [
    "run_agent",
    "agent_tools_payload",
    "BUDGETS",
    "OFFLINE_PLAN",
    "TOOLS",
    "ToolResult",
    "run_tool",
    "tool_schemas",
    "build_evidence",
    "merge_evidence",
]
