"""Long-term memory extraction and retrieval using PostgresStore."""

from __future__ import annotations

import logging
import uuid

from langchain_core.language_models import BaseChatModel
from langgraph.store.base import BaseStore

from src.constants import MEMORY_MAX_RESULTS, MEMORY_NAMESPACE

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """You are a memory extraction assistant. Given a conversation exchange, extract any facts worth remembering about the user for future conversations.

Examples of useful facts:
- User's name, location, profession
- User's preferences and interests
- Important context about their projects or goals
- Technical preferences (languages, tools, frameworks)

If there are no facts worth remembering, respond with exactly: NONE

Otherwise, respond with one fact per line, nothing else. Be concise.

User said: {user_message}
Assistant said: {assistant_message}

Facts to remember:"""


async def extract_memories(
    llm: BaseChatModel,
    user_message: str,
    assistant_message: str,
) -> list[str]:
    """Extract memorable facts from a conversation exchange.

    Args:
        llm: The language model used for extraction.
        user_message: The user's message text.
        assistant_message: The assistant's response text.

    Returns:
        List of extracted fact strings. Empty if nothing worth remembering.
    """
    prompt = EXTRACT_PROMPT.format(
        user_message=user_message,
        assistant_message=assistant_message,
    )
    response = await llm.ainvoke(prompt)
    text = response.content.strip()
    if not text or text.upper() == "NONE":
        return []
    return [line.strip("- ").strip() for line in text.splitlines() if line.strip()]


async def store_memories(
    store: BaseStore,
    user_id: str,
    memories: list[str],
) -> None:
    """Persist extracted memories to the store, skipping exact duplicates.

    Args:
        store: The backing store instance.
        user_id: Owner of the memories.
        memories: List of fact strings to store.
    """
    existing = await list_memories(store, user_id, limit=200)
    existing_texts = {m["text"] for m in existing}
    for memory in memories:
        if memory not in existing_texts:
            key = str(uuid.uuid4())
            await store.aput(
                (*MEMORY_NAMESPACE, user_id),
                key,
                {"text": memory},
            )


async def retrieve_memories(
    store: BaseStore,
    user_id: str,
    query: str,
    max_results: int = MEMORY_MAX_RESULTS,
) -> list[str]:
    """Retrieve memories for a user.

    Uses ``store.asearch`` which, without a vector index configured, returns
    the most recently stored memories (the ``query`` parameter is ignored by
    the plain PostgresStore implementation).

    Args:
        store: The backing store instance.
        user_id: Owner of the memories.
        query: Passed to asearch; has no effect without a vector index.
        max_results: Maximum number of memories to return.

    Returns:
        List of memory text strings.
    """
    try:
        results = await store.asearch(
            (*MEMORY_NAMESPACE, user_id),
            query=query,
            limit=max_results,
        )
        return [item.value["text"] for item in results if "text" in item.value]
    except Exception:
        logger.debug("Memory retrieval failed", exc_info=True)
        return []


async def list_memories(
    store: BaseStore,
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    """List all stored memories for a user.

    Returns:
        List of dicts with 'key' and 'text' fields.
    """
    try:
        results = await store.asearch(
            (*MEMORY_NAMESPACE, user_id),
            limit=limit,
        )
        return [
            {"key": item.key, "text": item.value["text"]}
            for item in results
            if "text" in item.value
        ]
    except Exception:
        logger.debug("Failed to list memories", exc_info=True)
        return []


async def delete_memory(
    store: BaseStore,
    user_id: str,
    key: str,
) -> None:
    """Delete a single memory by its key."""
    await store.adelete((*MEMORY_NAMESPACE, user_id), key)


async def clear_memories(
    store: BaseStore,
    user_id: str,
) -> int:
    """Delete all memories for a user. Returns the count of deleted memories."""
    results = await store.asearch(
        (*MEMORY_NAMESPACE, user_id),
        limit=1000,
    )
    count = 0
    for item in results:
        await store.adelete((*MEMORY_NAMESPACE, user_id), item.key)
        count += 1
    return count


def format_memories_for_prompt(memories: list[str]) -> str:
    """Format retrieved memories into a plain bullet list.

    Args:
        memories: List of memory text strings.

    Returns:
        Bullet-list string of memories, or empty string if none.
    """
    if not memories:
        return ""
    return "\n".join(f"- {m}" for m in memories)
