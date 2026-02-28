"""Thread management: names, history, and message retrieval."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from src.constants import (
    THREAD_LIST_LIMIT,
    THREAD_NAMES_NAMESPACE,
    THREAD_PREVIEW_LIMIT,
)
from src.prompts import THREAD_NAME_PROMPT

logger = logging.getLogger(__name__)


async def generate_thread_name(llm: BaseChatModel, first_message: str) -> str:
    """Generate a short readable name for a thread using the LLM.

    Args:
        llm: The chat model instance to use for name generation.
        first_message: The user's first message in the thread.

    Returns:
        A short name (roughly 5 words), or a truncated fallback on failure.
    """
    try:
        prompt = THREAD_NAME_PROMPT.format(message=first_message)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        name = response.content.strip().strip(".")
        if name:
            return name[:80]
    except Exception:
        logger.debug("Thread name generation failed", exc_info=True)
    return (first_message[:50] + "...") if len(first_message) > 50 else first_message


async def save_thread_name(
    store: AsyncPostgresStore, thread_id: str, name: str
) -> None:
    """Save a thread name to the store.

    Args:
        store: The Postgres-backed store.
        thread_id: The thread identifier.
        name: The readable name to save.
    """
    await store.aput(THREAD_NAMES_NAMESPACE, thread_id, {"name": name})


async def get_thread_name(
    store: AsyncPostgresStore, thread_id: str
) -> str | None:
    """Retrieve a thread name from the store.

    Args:
        store: The Postgres-backed store.
        thread_id: The thread identifier.

    Returns:
        The stored name, or None if not found.
    """
    try:
        item = await store.aget(THREAD_NAMES_NAMESPACE, thread_id)
        if item and item.value:
            return item.value.get("name")
    except Exception:
        logger.debug("Failed to get thread name for %s", thread_id, exc_info=True)
    return None


async def get_thread_history(
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore | None = None,
) -> list[dict]:
    """List existing conversation threads from the checkpoint store.

    Args:
        checkpointer: The Postgres checkpointer to query.
        store: Optional Postgres store to look up thread names.

    Returns:
        List of dicts with ``thread_id``, ``preview``, and ``name`` keys.
    """
    threads: list[dict] = []
    try:
        async for checkpoint_tuple in checkpointer.alist(None, limit=THREAD_LIST_LIMIT):
            thread_id = checkpoint_tuple.config.get("configurable", {}).get(
                "thread_id", ""
            )
            if thread_id and thread_id not in [t["thread_id"] for t in threads]:
                checkpoint = checkpoint_tuple.checkpoint
                preview = ""
                channel_values = checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])
                for msg in messages:
                    if getattr(msg, "type", None) == "human":
                        content = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        preview = (
                            content[:THREAD_PREVIEW_LIMIT] + "..."
                            if len(content) > THREAD_PREVIEW_LIMIT
                            else content
                        )
                        break

                name = None
                if store:
                    name = await get_thread_name(store, thread_id)

                threads.append({
                    "thread_id": thread_id,
                    "preview": preview,
                    "name": name or preview,
                })
    except Exception:
        logger.debug("Failed to list thread history", exc_info=True)
    return threads


async def get_thread_messages(
    checkpointer: AsyncPostgresSaver,
    thread_id: str,
) -> list[HumanMessage | AIMessage]:
    """Load conversation messages for a specific thread from the checkpoint.

    Args:
        checkpointer: The Postgres checkpointer to query.
        thread_id: The thread ID to load messages for.

    Returns:
        List of HumanMessage and AIMessage instances from the thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            return []
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        return [
            msg for msg in messages
            if getattr(msg, "type", None) in ("human", "ai", "tool")
        ]
    except Exception:
        logger.debug("Failed to load thread messages for %s", thread_id, exc_info=True)
        return []
