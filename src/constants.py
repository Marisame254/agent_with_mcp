"""Centralized constants and enums for the agent application."""

from __future__ import annotations

from enum import Enum


class ChatCommand(Enum):
    """Commands returned by the chat loop to control thread lifecycle."""

    NEW = "NEW"
    EXIT = "EXIT"


class AgentEventKind(str, Enum):
    """Event types emitted during agent streaming."""

    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    RESPONSE = "response"


# Token estimation
CHARS_PER_TOKEN: int = 4
MESSAGE_TOKEN_OVERHEAD: int = 4

# Display truncation limits
TOOL_INPUT_DISPLAY_LIMIT: int = 100
TOOL_OUTPUT_DISPLAY_LIMIT: int = 200

# Thread history
THREAD_PREVIEW_LIMIT: int = 80
THREAD_LIST_LIMIT: int = 50

# Agent summarization
KEEP_MESSAGES: int = 20

# Tool defaults
TAVILY_MAX_RESULTS: int = 5
MEMORY_MAX_RESULTS: int = 5

# Memory store
MEMORY_NAMESPACE: tuple[str, ...] = ("memories",)

# Context usage thresholds (percentage)
CONTEXT_USAGE_WARNING_PCT: int = 50
CONTEXT_USAGE_DANGER_PCT: int = 80

# UI
CHAT_HISTORY_FILE: str = ".chat_history"
