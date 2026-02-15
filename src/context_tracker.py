"""Token counting and /context command breakdown."""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from src.constants import (
    CHARS_PER_TOKEN,
    CONTEXT_USAGE_DANGER_PCT,
    CONTEXT_USAGE_WARNING_PCT,
    MESSAGE_TOKEN_OVERHEAD,
)

try:
    import tiktoken
    _encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoding = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base) with heuristic fallback.

    Args:
        text: The input string to count tokens for.

    Returns:
        Token count (minimum 1).
    """
    if _encoding is not None:
        return max(1, len(_encoding.encode(text)))
    return max(1, len(text) // CHARS_PER_TOKEN)


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """Count approximate tokens in a list of LangChain messages.

    Counts content, tool_calls on AIMessages, and name field on messages.

    Args:
        messages: List of BaseMessage instances.

    Returns:
        Total estimated token count including per-message overhead.
    """
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += count_tokens(content) + MESSAGE_TOKEN_OVERHEAD

        # Count tool_calls in AIMessages (name, args, id)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_text = json.dumps(msg.tool_calls, default=str)
            total += count_tokens(tool_calls_text)

        # Count the name field (present on ToolMessages and some others)
        if hasattr(msg, "name") and msg.name:
            total += count_tokens(msg.name)

    return total


def count_tool_definitions_tokens(tools: list) -> int:
    """Count tokens for tool definitions (schemas sent to the model).

    Extracts the real schema from each tool object when available,
    falling back to str(tool).

    Args:
        tools: List of tool objects (LangChain tools, MCP tools, etc.).

    Returns:
        Total estimated token count for all tool definitions.
    """
    total = 0
    for tool in tools:
        parts: list[str] = []
        name = getattr(tool, "name", None)
        if name:
            parts.append(name)
        description = getattr(tool, "description", None)
        if description:
            parts.append(description)
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            try:
                schema = args_schema.schema()
                parts.append(str(schema))
            except Exception:
                pass
        if parts:
            total += count_tokens(" ".join(parts))
        else:
            total += count_tokens(str(tool))
    return total


def detect_summary_tokens(messages: list[BaseMessage]) -> int:
    """Count tokens from SystemMessages that contain summarization output.

    The SummarizationMiddleware injects a SystemMessage with a summary
    of earlier conversation. This function detects and counts those tokens.

    Args:
        messages: List of messages from the checkpoint state.

    Returns:
        Token count for summary SystemMessages (0 if none found).
    """
    total = 0
    for msg in messages:
        if isinstance(msg, SystemMessage) and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += count_tokens(content)
    return total


@dataclass
class ContextBreakdown:
    """Token usage breakdown for the /context command."""

    system_tokens: int = 0
    memory_tokens: int = 0
    messages_tokens: int = 0
    tools_tokens: int = 0
    mcp_tool_count: int = 0
    summary_tokens: int = 0
    max_tokens: int = 8000

    @property
    def total_tokens(self) -> int:
        return (
            self.system_tokens
            + self.memory_tokens
            + self.messages_tokens
            + self.tools_tokens
            + self.summary_tokens
        )

    @property
    def usage_percent(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return (self.total_tokens / self.max_tokens) * 100

    @property
    def usage_color(self) -> str:
        pct = self.usage_percent
        if pct < CONTEXT_USAGE_WARNING_PCT:
            return "green"
        elif pct < CONTEXT_USAGE_DANGER_PCT:
            return "yellow"
        return "red"


def build_context_breakdown(
    system_prompt: str,
    memories: list[str],
    messages: list[BaseMessage],
    tools: list,
    mcp_tool_count: int,
    max_tokens: int,
) -> ContextBreakdown:
    """Build a token breakdown for the /context command.

    Args:
        system_prompt: The system prompt text.
        memories: List of memory strings injected into context.
        messages: Conversation message history (from checkpoint state).
        tools: List of tool objects (full objects, not just name dicts).
        mcp_tool_count: Number of MCP-provided tools.
        max_tokens: Maximum context window size.

    Returns:
        A populated ContextBreakdown instance.
    """
    memory_text = "\n".join(memories) if memories else ""

    return ContextBreakdown(
        system_tokens=count_tokens(system_prompt),
        memory_tokens=count_tokens(memory_text),
        messages_tokens=count_message_tokens(messages),
        tools_tokens=count_tool_definitions_tokens(tools),
        mcp_tool_count=mcp_tool_count,
        summary_tokens=detect_summary_tokens(messages),
        max_tokens=max_tokens,
    )
