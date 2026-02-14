"""Token counting and /context command breakdown."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage


def count_tokens(text: str) -> int:
    """Approximate token count using whitespace + punctuation heuristic (~4 chars/token)."""
    return max(1, len(text) // 4)


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """Count approximate tokens in a list of messages."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += count_tokens(content) + 4  # overhead per message
    return total


@dataclass
class ContextBreakdown:
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
        if pct < 50:
            return "green"
        elif pct < 80:
            return "yellow"
        return "red"


def build_context_breakdown(
    system_prompt: str,
    memories: list[str],
    messages: list[BaseMessage],
    tool_definitions: list[dict],
    mcp_tool_count: int,
    max_tokens: int,
) -> ContextBreakdown:
    """Build a token breakdown for the /context command."""
    memory_text = "\n".join(memories) if memories else ""
    tools_text = str(tool_definitions) if tool_definitions else ""

    return ContextBreakdown(
        system_tokens=count_tokens(system_prompt),
        memory_tokens=count_tokens(memory_text),
        messages_tokens=count_message_tokens(messages),
        tools_tokens=count_tokens(tools_text),
        mcp_tool_count=mcp_tool_count,
        max_tokens=max_tokens,
    )
