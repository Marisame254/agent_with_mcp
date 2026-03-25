"""Custom tools for the agent."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.tools import BaseTool, tool

from src.constants import ASK_USER_TOOL_NAME


def create_ask_user_tool(
    prompt_fn: Callable[
        [str, list[dict] | None, bool],
        Coroutine[Any, Any, str],
    ],
) -> BaseTool:
    """Create an ask_user tool that delegates input to *prompt_fn*.

    Args:
        prompt_fn: An async callable ``(question, options, multi_select)``
            that shows the question to the user and returns the answer.

    Returns:
        A LangChain tool named ``ask_user``.
    """

    @tool(ASK_USER_TOOL_NAME)
    async def ask_user(
        question: str,
        options: list[dict | str] | None = None,
        multi_select: bool = False,
    ) -> str:
        """Ask the user when you need clarification or additional information.

        Use this tool whenever you are unsure about the user's intent, need to
        confirm an action, or require details that were not provided in the
        conversation so far.

        Args:
            question: The question to ask the user.
            options: Optional list of choices to present. Each item can be a
                dict with ``{"title": "...", "description": "..."}`` (description
                is optional) or a plain string. When provided, the user gets an
                interactive selector. They can always choose to type a free-text
                answer instead.
            multi_select: If True the user can pick multiple options. Only
                meaningful when ``options`` is provided.
        """
        # Normalize: strings → dicts
        normalized: list[dict] | None = None
        if options:
            normalized = [
                opt if isinstance(opt, dict) else {"title": str(opt)}
                for opt in options
            ]
        return await prompt_fn(question, normalized, multi_select)

    return ask_user  # type: ignore[return-value]
