"""LangGraph agent with MCP tools, Tavily search, and summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage
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
from src.memory import (
    extract_memories,
    format_memories_for_prompt,
    retrieve_memories,
    store_memories,
)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools. \
Use the tools available to you to help the user with their requests. \
Be concise and direct in your responses. \
When you use a tool, explain what you found."""


async def create_agent_resources():
    """Create and return the checkpointer, store, and their async context managers."""
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
    store_cm = AsyncPostgresStore.from_conn_string(DATABASE_URL)
    return checkpointer_cm, store_cm


def build_tools() -> list:
    """Build the list of non-MCP tools (Tavily)."""
    tools = []
    if TAVILY_API_KEY:
        tools.append(TavilySearch(max_results=5))
    return tools


async def build_agent(
    mcp_client: MultiServerMCPClient,
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
) -> tuple[Any, list, int]:
    """Create the agent with all tools.

    Returns:
        Tuple of (agent, all_tools, mcp_tool_count)
    """
    mcp_tools = await mcp_client.get_tools()
    base_tools = build_tools()
    all_tools = mcp_tools + base_tools
    mcp_tool_count = len(mcp_tools)

    llm = ChatOllama(model=MODEL_NAME)

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            SummarizationMiddleware(
                model=llm,
                trigger=("tokens", MAX_CONTEXT_TOKENS),
                keep=("messages", 20),
            ),
        ],
        checkpointer=checkpointer,
        store=store,
    )

    return agent, all_tools, mcp_tool_count


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""

    kind: str  # "tool_start", "tool_end", "response"
    tool_name: str = ""
    tool_input: str = ""
    tool_output: str = ""
    response: str = ""


async def stream_agent_turn(
    agent: Any,
    store: AsyncPostgresStore,
    user_message: str,
    thread_id: str,
    user_id: str = "default",
):
    """Stream agent events for a single turn. Yields AgentEvent objects."""
    # Retrieve memories and inject into conversation
    memories = await retrieve_memories(store, user_id, user_message)
    memory_section = format_memories_for_prompt(memories)

    config = {
        "configurable": {"thread_id": thread_id},
    }

    messages = []
    if memory_section:
        from langchain_core.messages import SystemMessage

        messages.append(SystemMessage(content=memory_section))
    messages.append(HumanMessage(content=user_message))

    inputs = {"messages": messages}

    response_text = ""

    async for event in agent.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event", "")

        if kind == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = str(event.get("data", {}).get("input", ""))
            if len(tool_input) > 100:
                tool_input = tool_input[:100] + "..."
            yield AgentEvent(kind="tool_start", tool_name=tool_name, tool_input=tool_input)

        elif kind == "on_tool_end":
            tool_name = event.get("name", "unknown")
            output = str(event.get("data", {}).get("output", ""))
            if len(output) > 200:
                output = output[:200] + "..."
            yield AgentEvent(kind="tool_end", tool_name=tool_name, tool_output=output)

        elif kind == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            if isinstance(output, AIMessage) and output.content:
                response_text = output.content

    if response_text:
        yield AgentEvent(kind="response", response=response_text)

        # Extract and store memories (best-effort)
        try:
            llm = ChatOllama(model=MODEL_NAME)
            new_memories = await extract_memories(llm, user_message, response_text)
            if new_memories:
                await store_memories(store, user_id, new_memories)
        except Exception:
            pass


async def get_thread_history(
    checkpointer: AsyncPostgresSaver,
) -> list[dict]:
    """List existing conversation threads from the checkpoint store."""
    threads = []
    try:
        async for checkpoint_tuple in checkpointer.alist(None, limit=50):
            thread_id = checkpoint_tuple.config.get("configurable", {}).get("thread_id", "")
            if thread_id and thread_id not in [t["thread_id"] for t in threads]:
                # Try to get the first message as a preview
                checkpoint = checkpoint_tuple.checkpoint
                preview = ""
                channel_values = checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])
                if messages and len(messages) > 0:
                    first_msg = messages[0]
                    content = first_msg.content if hasattr(first_msg, "content") else str(first_msg)
                    preview = content[:80] + "..." if len(content) > 80 else content
                threads.append({"thread_id": thread_id, "preview": preview})
    except Exception:
        pass
    return threads
