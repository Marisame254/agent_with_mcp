"""Custom tools for the agent."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.tools import tool, BaseTool

from src.constants import ASK_USER_TOOL_NAME


def create_ask_user_tool(
    prompt_fn: Callable[[str], Coroutine[Any, Any, str]],
) -> BaseTool:
    """Create an ask_user tool that delegates input to *prompt_fn*.

    Args:
        prompt_fn: An async callable that receives the question string,
            shows it to the user, waits for a response, and returns it.

    Returns:
        A LangChain tool named ``ask_user``.
    """

    @tool(ASK_USER_TOOL_NAME)
    async def ask_user(question: str) -> str:
        """Ask the user when you need clarification or additional information.

        Use this tool whenever you are unsure about the user's intent, need to
        confirm an action, or require details that were not provided in the
        conversation so far.
        """
        return await prompt_fn(question)

    return ask_user  # type: ignore[return-value]
