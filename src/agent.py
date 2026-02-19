"""LangGraph agent with MCP tools, Tavily search, and summarization."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command

from src.config import (
    DATABASE_URL,
    MAX_CONTEXT_TOKENS,
    MODEL_NAME,
    TAVILY_API_KEY,
)
from src.constants import (
    ASK_USER_TOOL_NAME,
    AgentEventKind,
    KEEP_MESSAGES,
    TAVILY_MAX_RESULTS,
    THREAD_LIST_LIMIT,
    THREAD_NAME_PROMPT,
    THREAD_NAMES_NAMESPACE,
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
    """Build the base system prompt with the current date and time injected."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _SYSTEM_PROMPT_TEMPLATE.format(current_time=now)


def build_system_prompt(memories: list[str] | None = None) -> str:
    """Build the full system prompt, optionally including user memories.

    Combines the base system prompt with formatted memories into a single
    unified prompt, avoiding multiple SystemMessage injections.

    Args:
        memories: Optional list of memory text strings to include.

    Returns:
        The complete system prompt string.
    """
    base = get_system_prompt()
    memory_block = format_memories_for_prompt(memories or [])
    if not memory_block:
        return base
    return f"{base}\n\nYou remember the following about this user:\n{memory_block}"


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
    *,
    ask_user_tool: BaseTool | None = None,
    model_name: str = MODEL_NAME,
) -> tuple[Any, list, int, list[str]]:
    """Create the agent with all tools and HITL middleware for MCP tools.

    Args:
        mcp_client: Optional MCP client for external tool servers.
        checkpointer: Postgres-backed conversation checkpointer.
        store: Postgres-backed key-value store for memories.
        ask_user_tool: Optional tool that lets the agent ask the user questions.
        model_name: Ollama model name to use for the LLM.

    Returns:
        Tuple of (agent, all_tools, mcp_tool_count, mcp_tool_names).
    """
    mcp_tools = await mcp_client.get_tools() if mcp_client else []
    base_tools = build_tools()

    all_tools: list = mcp_tools + base_tools
    if ask_user_tool is not None:
        all_tools.append(ask_user_tool)

    mcp_tool_count = len(mcp_tools)
    mcp_tool_names = [getattr(t, "name", str(t)) for t in mcp_tools]

    logger.info(
        "Building agent: %d MCP tools, %d base tools, model=%s",
        mcp_tool_count,
        len(base_tools),
        model_name,
    )

    llm = ChatOllama(model=model_name)

    middleware: list = [
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", MAX_CONTEXT_TOKENS),
            keep=("messages", KEEP_MESSAGES),
        ),
    ]

    # Add HITL middleware for MCP tools (require approval before execution)
    if mcp_tool_names:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={name: True for name in mcp_tool_names},
                description_prefix="Aprobación requerida para ejecutar herramienta",
            ),
        )

    agent = create_agent(
        model=llm,
        tools=all_tools,
        middleware=middleware,
        checkpointer=checkpointer,
        store=store,
    )

    return agent, all_tools, mcp_tool_count, mcp_tool_names


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""

    kind: AgentEventKind
    tool_name: str = ""
    tool_input: str = ""
    tool_output: str = ""
    response: str = ""
    token: str = ""
    action_requests: list[dict[str, Any]] | None = None


async def _retrieve_and_build_messages(
    store: AsyncPostgresStore,
    user_message: str,
    user_id: str,
) -> list[SystemMessage | HumanMessage]:
    """Retrieve memories and build the message list for a single turn.

    Builds a single unified SystemMessage containing the base system prompt
    plus any retrieved memories, followed by the user's HumanMessage.

    Args:
        store: The backing store for memory retrieval.
        user_message: The user's current input.
        user_id: The user identifier for memory lookup.

    Returns:
        List of messages to send to the agent.
    """
    memories = await retrieve_memories(store, user_id, user_message)
    system_prompt = build_system_prompt(memories)

    return [
        SystemMessage(content=system_prompt, id="system_prompt"),
        HumanMessage(content=user_message),
    ]


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


async def _stream_and_yield(
    agent: Any,
    inputs: Any,
    config: dict,
) -> AsyncGenerator[AgentEvent, None]:
    """Stream events from the agent and yield parsed AgentEvents.

    After the event stream ends, checks for pending HITL interrupts and
    yields a ``TOOL_APPROVAL_REQUIRED`` event if any are found.

    Args:
        agent: The LangGraph agent instance.
        inputs: The inputs to pass to ``astream_events`` (messages dict or Command).
        config: The LangGraph config with thread_id.

    Yields:
        AgentEvent instances.
    """
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
                if isinstance(content, str) and content:
                    response_text += content
                    streamed_tokens = True
                    yield AgentEvent(kind=AgentEventKind.TOKEN, token=content)

        elif kind == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            if isinstance(output, AIMessage) and output.content:
                if not streamed_tokens:
                    response_text = output.content

    # Check for pending HITL interrupts
    state = await agent.aget_state(config)
    if state and state.tasks:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                action_requests: list[dict[str, Any]] = []
                for intr in task.interrupts:
                    value = intr.value
                    if isinstance(value, dict):
                        for ar in value.get("action_requests", []):
                            action_requests.append(dict(ar))
                if action_requests:
                    yield AgentEvent(
                        kind=AgentEventKind.TOOL_APPROVAL_REQUIRED,
                        action_requests=action_requests,
                    )
                    return  # Don't emit RESPONSE — waiting for approval

    if response_text:
        yield AgentEvent(kind=AgentEventKind.RESPONSE, response=response_text)


async def stream_agent_turn(
    agent: Any,
    store: AsyncPostgresStore,
    user_message: str,
    thread_id: str,
    user_id: str = "default",
    *,
    resume_command: Command | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Stream agent events for a single conversation turn.

    Retrieves relevant memories, streams tool and response events, then
    persists any new memories extracted from the exchange.

    When *resume_command* is provided the agent is resumed from a HITL
    interrupt instead of starting a new turn.

    Args:
        agent: The LangGraph agent instance.
        store: The backing store for memories.
        user_message: The user's current input.
        thread_id: Conversation thread identifier.
        user_id: The user identifier.
        resume_command: Optional Command to resume from a HITL interrupt.

    Yields:
        AgentEvent instances for tool starts, tool ends, and the final response.
    """
    config = {"configurable": {"thread_id": thread_id}}

    if resume_command is not None:
        inputs = resume_command
    else:
        messages = await _retrieve_and_build_messages(store, user_message, user_id)
        inputs = {"messages": messages}

    response_text = ""
    async for event in _stream_and_yield(agent, inputs, config):
        if event.kind == AgentEventKind.RESPONSE:
            response_text = event.response
        yield event

    if response_text:
        await _extract_and_store_memories(store, user_message, response_text, user_id)


async def generate_thread_name(llm: ChatOllama, first_message: str) -> str:
    """Generate a short readable name for a thread using the LLM.

    Args:
        llm: The ChatOllama instance.
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
    # Fallback: truncate the first message
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
