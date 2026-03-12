"""Long-term memory management using PostgresStore.

Legacy functions (list/delete/clear/store/retrieve) operate on the old
``("memories", user_id)`` namespace used before the CompositeBackend migration.
They are kept for the ``/memory`` CLI command so users can inspect and clean up
old memories.  New memories are managed by the agent itself via the
``/memories/AGENT.md`` file routed through ``StoreBackend``.
"""

from __future__ import annotations

import logging
import uuid

from langgraph.store.base import BaseStore

from src.constants import MEMORY_MAX_RESULTS, MEMORY_NAMESPACE

logger = logging.getLogger(__name__)


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
