"""LangGraph agent with MCP tools, Tavily search, and summarization."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from src.config import (
    DATABASE_URL,
    MAX_CONTEXT_TOKENS,
    MODEL_NAME,
    TAVILY_API_KEY,
)
from src.constants import (
    AgentEventKind,
    KEEP_MESSAGES,
    TAVILY_MAX_RESULTS,
    THREAD_LIST_LIMIT,
    THREAD_PREVIEW_LIMIT,
    TOOL_INPUT_DISPLAY_LIMIT,
    TOOL_OUTPUT_DISPLAY_LIMIT,
)
from src.memory import (
    extract_memories,
    format_memories_for_prompt,
    retrieve_memories,
    store_memories,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant with access to various tools. \
Use the tools available to you to help the user with their requests. \
Be concise and direct in your responses. \
When you use a tool, explain what you found.

Current date and time: {current_time}"""


def get_system_prompt() -> str:
    """Build the system prompt with the current date and time injected."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _SYSTEM_PROMPT_TEMPLATE.format(current_time=now)


async def create_agent_resources() -> tuple[AsyncPostgresSaver, AsyncPostgresStore]:
    """Create and return the checkpointer and store async context managers.

    Returns:
        Tuple of (checkpointer_cm, store_cm) ready to be used as async context managers.
    """
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
    store_cm = AsyncPostgresStore.from_conn_string(DATABASE_URL)
    return checkpointer_cm, store_cm


def build_tools() -> list:
    """Build the list of non-MCP tools (Tavily search)."""
    tools: list = []
    if TAVILY_API_KEY:
        tools.append(TavilySearch(max_results=TAVILY_MAX_RESULTS))
    return tools


async def build_agent(
    mcp_client: MultiServerMCPClient | None,
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
) -> tuple[Any, list, int]:
    """Create the agent with all tools.

    Args:
        mcp_client: Optional MCP client for external tool servers.
        checkpointer: Postgres-backed conversation checkpointer.
        store: Postgres-backed key-value store for memories.

    Returns:
        Tuple of (agent, all_tools, mcp_tool_count).
    """
    mcp_tools = await mcp_client.get_tools() if mcp_client else []
    base_tools = build_tools()
    all_tools = mcp_tools + base_tools
    mcp_tool_count = len(mcp_tools)

    logger.info(
        "Building agent: %d MCP tools, %d base tools, model=%s",
        mcp_tool_count,
        len(base_tools),
        MODEL_NAME,
    )

    llm = ChatOllama(model=MODEL_NAME)

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=get_system_prompt(),
        middleware=[
            SummarizationMiddleware(
                model=llm,
                trigger=("tokens", MAX_CONTEXT_TOKENS),
                keep=("messages", KEEP_MESSAGES),
            ),
        ],
        checkpointer=checkpointer,
        store=store,
    )

    return agent, all_tools, mcp_tool_count


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""

    kind: AgentEventKind
    tool_name: str = ""
    tool_input: str = ""
    tool_output: str = ""
    response: str = ""
    token: str = ""


async def _retrieve_and_build_messages(
    store: AsyncPostgresStore,
    user_message: str,
    user_id: str,
) -> list[SystemMessage | HumanMessage]:
    """Retrieve memories and build the message list for a single turn.

    Args:
        store: The backing store for memory retrieval.
        user_message: The user's current input.
        user_id: The user identifier for memory lookup.

    Returns:
        List of messages to send to the agent.
    """
    memories = await retrieve_memories(store, user_id, user_message)
    memory_section = format_memories_for_prompt(memories)

    messages: list[SystemMessage | HumanMessage] = []
    if memory_section:
        messages.append(SystemMessage(content=memory_section))
    messages.append(HumanMessage(content=user_message))
    return messages


async def _extract_and_store_memories(
    store: AsyncPostgresStore,
    user_message: str,
    response_text: str,
    user_id: str,
) -> None:
    """Extract and persist memories from a completed turn (best-effort).

    Args:
        store: The backing store for memory persistence.
        user_message: The user's input that triggered the response.
        response_text: The agent's final text response.
        user_id: The user identifier for memory storage.
    """
    try:
        llm = ChatOllama(model=MODEL_NAME)
        new_memories = await extract_memories(llm, user_message, response_text)
        if new_memories:
            await store_memories(store, user_id, new_memories)
            logger.info(
                "Stored %d new memories for user=%s", len(new_memories), user_id
            )
    except Exception:
        logger.debug("Memory extraction failed", exc_info=True)


async def stream_agent_turn(
    agent: Any,
    store: AsyncPostgresStore,
    user_message: str,
    thread_id: str,
    user_id: str = "default",
) -> AsyncGenerator[AgentEvent, None]:
    """Stream agent events for a single conversation turn.

    Retrieves relevant memories, streams tool and response events, then
    persists any new memories extracted from the exchange.

    Args:
        agent: The LangGraph agent instance.
        store: The backing store for memories.
        user_message: The user's current input.
        thread_id: Conversation thread identifier.
        user_id: The user identifier.

    Yields:
        AgentEvent instances for tool starts, tool ends, and the final response.
    """
    messages = await _retrieve_and_build_messages(store, user_message, user_id)

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": messages}

    response_text = ""
    streamed_tokens = False

    async for event in agent.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event", "")

        if kind == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = str(event.get("data", {}).get("input", ""))
            if len(tool_input) > TOOL_INPUT_DISPLAY_LIMIT:
                tool_input = tool_input[:TOOL_INPUT_DISPLAY_LIMIT] + "..."
            yield AgentEvent(
                kind=AgentEventKind.TOOL_START,
                tool_name=tool_name,
                tool_input=tool_input,
            )

        elif kind == "on_tool_end":
            tool_name = event.get("name", "unknown")
            output = str(event.get("data", {}).get("output", ""))
            if len(output) > TOOL_OUTPUT_DISPLAY_LIMIT:
                output = output[:TOOL_OUTPUT_DISPLAY_LIMIT] + "..."
            yield AgentEvent(
                kind=AgentEventKind.TOOL_END,
                tool_name=tool_name,
                tool_output=output,
            )

        elif kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if hasattr(chunk, "content") and chunk.content:
                content = chunk.content
                # Skip chunks that are only tool calls with no text
                if isinstance(content, str) and content:
                    response_text += content
                    streamed_tokens = True
                    yield AgentEvent(
                        kind=AgentEventKind.TOKEN, token=content
                    )

        elif kind == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            if isinstance(output, AIMessage) and output.content:
                if not streamed_tokens:
                    # Fallback: model did not support streaming
                    response_text = output.content

    if response_text:
        yield AgentEvent(kind=AgentEventKind.RESPONSE, response=response_text)
        await _extract_and_store_memories(store, user_message, response_text, user_id)


async def get_thread_history(
    checkpointer: AsyncPostgresSaver,
) -> list[dict]:
    """List existing conversation threads from the checkpoint store.

    Args:
        checkpointer: The Postgres checkpointer to query.

    Returns:
        List of dicts with ``thread_id`` and ``preview`` keys.
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
                if messages and len(messages) > 0:
                    first_msg = messages[0]
                    content = (
                        first_msg.content
                        if hasattr(first_msg, "content")
                        else str(first_msg)
                    )
                    preview = (
                        content[:THREAD_PREVIEW_LIMIT] + "..."
                        if len(content) > THREAD_PREVIEW_LIMIT
                        else content
                    )
                threads.append({"thread_id": thread_id, "preview": preview})
    except Exception:
        logger.debug("Failed to list thread history", exc_info=True)
    return threads
