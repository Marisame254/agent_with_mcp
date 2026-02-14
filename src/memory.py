"""Long-term memory extraction and retrieval using PostgresStore."""

from __future__ import annotations

import logging
import uuid

from langchain_ollama import ChatOllama
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
    llm: ChatOllama,
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
    """Persist extracted memories to the store.

    Args:
        store: The backing store instance.
        user_id: Owner of the memories.
        memories: List of fact strings to store.
    """
    for memory in memories:
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
    """Retrieve relevant memories for a user.

    Uses keyword search via ``store.asearch``, falling back to an unfiltered
    listing when the store has no embedding support.

    Args:
        store: The backing store instance.
        user_id: Owner of the memories.
        query: Search query for semantic/keyword matching.
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
        logger.debug("Memory search with query failed, trying without query", exc_info=True)
        try:
            results = await store.asearch(
                (*MEMORY_NAMESPACE, user_id),
                limit=max_results,
            )
            return [item.value["text"] for item in results if "text" in item.value]
        except Exception:
            logger.debug("Memory retrieval fallback failed", exc_info=True)
            return []


def format_memories_for_prompt(memories: list[str]) -> str:
    """Format retrieved memories into a system prompt section.

    Args:
        memories: List of memory text strings.

    Returns:
        Formatted string to inject into the system prompt, or empty string.
    """
    if not memories:
        return ""
    memory_lines = "\n".join(f"- {m}" for m in memories)
    return f"\n\nYou remember the following about this user:\n{memory_lines}\n"
