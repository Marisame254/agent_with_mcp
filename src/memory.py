"""Long-term memory extraction and retrieval using PostgresStore."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.store.base import BaseStore

MEMORY_NAMESPACE = ("memories",)

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
    """Extract memorable facts from a conversation exchange."""
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
    """Store extracted memories in the store."""
    for i, memory in enumerate(memories):
        import uuid

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
    max_results: int = 5,
) -> list[str]:
    """Retrieve relevant memories for a user. Uses keyword search via store.asearch."""
    try:
        results = await store.asearch(
            (*MEMORY_NAMESPACE, user_id),
            query=query,
            limit=max_results,
        )
        return [item.value["text"] for item in results if "text" in item.value]
    except Exception:
        # If search fails (e.g., no embeddings configured), try without query
        try:
            results = await store.asearch(
                (*MEMORY_NAMESPACE, user_id),
                limit=max_results,
            )
            return [item.value["text"] for item in results if "text" in item.value]
        except Exception:
            return []


def format_memories_for_prompt(memories: list[str]) -> str:
    """Format retrieved memories into a system prompt section."""
    if not memories:
        return ""
    memory_lines = "\n".join(f"- {m}" for m in memories)
    return f"\n\nYou remember the following about this user:\n{memory_lines}\n"
