"""Centralized constants and enums for the agent application."""

from __future__ import annotations

from enum import Enum


class ChatCommand(Enum):
    """Commands returned by the chat loop to control thread lifecycle."""

    NEW = "NEW"
    EXIT = "EXIT"
    MODEL = "MODEL"
    MCP_RELOAD = "MCP_RELOAD"


class AgentEventKind(str, Enum):
    """Event types emitted during agent streaming."""

    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOKEN = "token"
    RESPONSE = "response"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"


# Token estimation
CHARS_PER_TOKEN: int = 4
MESSAGE_TOKEN_OVERHEAD: int = 4

# Display truncation limits
TOOL_INPUT_DISPLAY_LIMIT: int = 100
TOOL_OUTPUT_DISPLAY_LIMIT: int = 200

# Thread history
# NOTE: alist() returns individual checkpoints, not threads. Each conversation
# turn generates several checkpoints (agent node, tool nodes, etc.). This limit
# must be large enough to cover all expected threads × their average checkpoint
# count. At ~20 checkpoints/turn and ~20 turns/thread, 1000 covers ~50 threads.
THREAD_PREVIEW_LIMIT: int = 80
THREAD_LIST_LIMIT: int = 1000

# Agent summarization
KEEP_MESSAGES: int = 20

# Tool defaults
TAVILY_MAX_RESULTS: int = 5
MEMORY_MAX_RESULTS: int = 5

# Memory store
MEMORY_NAMESPACE: tuple[str, ...] = ("memories",)

# Thread names
THREAD_NAMES_NAMESPACE: tuple[str, ...] = ("thread_names",)
THREAD_NAME_PROMPT: str = (
    "Resume en máximo 5 palabras de qué trata este mensaje. "
    "Responde SOLO con el resumen, sin puntuación final ni explicación.\n\n"
    "Mensaje: {message}"
)

# Context usage thresholds (percentage)
CONTEXT_USAGE_WARNING_PCT: int = 50
CONTEXT_USAGE_DANGER_PCT: int = 80

# Tools
ASK_USER_TOOL_NAME: str = "ask_user"

# UI
CHAT_HISTORY_FILE: str = ".chat_history"
