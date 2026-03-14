"""LangGraph agent with MCP tools, Tavily search, and summarization."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend, StoreBackend
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command, Overwrite

from src.config import (
    DATABASE_URL,
    MODEL_NAME,
    TAVILY_API_KEY,
)
from src.constants import (
    AGENT_MEMORY_FILE,
    AGENT_RECURSION_LIMIT,
    SUMMARIZATION_NODE_NAME,
    TAVILY_MAX_RESULTS,
    TOOL_INPUT_DISPLAY_LIMIT,
    TOOL_OUTPUT_DISPLAY_LIMIT,
    AgentEventKind,
)
from src.prompts import SYSTEM_PROMPT_TEMPLATE
from src.providers import ModelSpec, build_llm

logger = logging.getLogger(__name__)


def _unwrap(value: Any) -> Any:
    """Recursively unwrap LangGraph ``Overwrite`` wrappers."""
    while isinstance(value, Overwrite):
        value = value.value
    return value


def _content_blocks_to_str(blocks: list) -> str:
    """Extract text from a list of MCP content blocks."""
    parts = [
        block.get("text", str(block)) if isinstance(block, dict) else str(block) for block in blocks
    ]
    return "\n".join(filter(None, parts)) or "(no output)"


def _normalize_mcp_tool(tool: BaseTool) -> BaseTool:
    """Wrap an MCP tool to normalize list output to string.

    MCP tools return structured content blocks (list) per the MCP spec.
    OpenAI's API requires ToolMessage.content to be a plain string.
    Ollama is lenient; OpenAI/DeepSeek are strict — without this wrapper
    subsequent LLM calls fail with BadRequestError 400.
    """
    from langchain_core.messages import ToolMessage as _ToolMessage

    original_ainvoke = tool.ainvoke

    async def _normalized(input_data: Any, config: Any = None, **kwargs: Any) -> Any:
        result = await original_ainvoke(input_data, config=config, **kwargs)

        # Case 1: tool returned a ToolMessage with list content
        if isinstance(result, _ToolMessage) and isinstance(result.content, list):
            return result.model_copy(update={"content": _content_blocks_to_str(result.content)})
        # Case 2: tool returned a plain list of content blocks
        if isinstance(result, list):
            return _content_blocks_to_str(result)

        return result

    object.__setattr__(tool, "ainvoke", _normalized)
    return tool


def get_system_prompt() -> str:
    """Build the base system prompt with the current date/time and working directory."""
    # import os

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # cwd = os.getcwd()
    return SYSTEM_PROMPT_TEMPLATE.format(current_time=now)


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
    """Create the agent with all tools and HITL interrupt configuration.

    Args:
        mcp_client: Optional MCP client for external tool servers.
        checkpointer: Postgres-backed conversation checkpointer.
        store: Postgres-backed key-value store for memories.
        ask_user_tool: Optional tool that lets the agent ask the user questions.
        model_name: Model identifier in ``provider/name`` format, or a bare
            Ollama model name for backward compatibility.

    Returns:
        Tuple of (agent, all_tools, mcp_tool_count, mcp_tool_names).
    """
    mcp_tools = []
    if mcp_client:
        try:
            mcp_tools = await mcp_client.get_tools()
        except Exception as e:
            logger.warning("Failed to get MCP tools: %s", e)
            from src.ui import show_error  # noqa: PLC0415

            show_error(f"No se pudieron cargar herramientas MCP: {e}. Continuando sin ellas.")
    mcp_tools = [_normalize_mcp_tool(t) for t in mcp_tools]
    base_tools = build_tools()

    all_tools: list = mcp_tools + base_tools
    if ask_user_tool is not None:
        all_tools.append(ask_user_tool)

    mcp_tool_count = len(mcp_tools)
    mcp_tool_names = [getattr(t, "name", str(t)) for t in mcp_tools]

    spec = ModelSpec.parse(model_name)

    logger.info(
        "Building agent: %d MCP tools, %d base tools, model=%s",
        mcp_tool_count,
        len(base_tools),
        spec,
    )

    llm = build_llm(spec)

    # HITL: always for execute (LocalShellBackend shell tool); also for MCP tools
    # write_file and edit_file are irreversible — require approval like execute
    hitl_tools: dict[str, bool] = {"execute": True, "write_file": True, "edit_file": True}
    hitl_tools.update({name: True for name in mcp_tool_names})

    agent = create_deep_agent(
        model=llm,
        system_prompt=get_system_prompt(),
        tools=all_tools,
        interrupt_on=hitl_tools,
        backend=lambda rt: CompositeBackend(
            default=LocalShellBackend(root_dir=".", inherit_env=True),
            routes={"/memories/": StoreBackend(rt)},
        ),
        checkpointer=checkpointer,
        store=store,
        memory=[AGENT_MEMORY_FILE],
        name="Lyra Code Assistant",
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
    is_subagent: bool = False


async def _stream_and_yield(
    agent: Any,
    inputs: Any,
    config: dict,
    seen_tool_call_ids: set[str] | None = None,
) -> AsyncGenerator[AgentEvent]:
    """Stream events from the agent using the graph-level streaming API.

    Uses ``astream`` with multiple stream modes (messages, updates, custom)
    and ``subgraphs=True`` to capture subagent activity and custom progress
    events from tools.

    HITL interrupts are detected inline from ``updates`` chunks containing
    ``__interrupt__``, with a fallback to ``aget_state()`` post-stream.

    Args:
        agent: The LangGraph ``CompiledStateGraph`` returned by ``create_deep_agent``.
        inputs: Messages dict or ``Command`` for HITL resume.
        config: LangGraph config with ``thread_id``.

    Yields:
        AgentEvent instances.
    """
    response_text = ""
    found_interrupt = False
    if seen_tool_call_ids is None:
        seen_tool_call_ids = set()

    # astream with list stream_mode + subgraphs=True yields tuples: (ns, mode, data)
    async for ns, chunk_type, data in agent.astream(
        inputs,
        config=config,
        stream_mode=["messages", "updates", "custom"],
        subgraphs=True,
    ):
        is_subagent = any(seg.startswith("tools:") for seg in ns)

        if chunk_type == "messages":
            token, metadata = data

            # Filter out summarization middleware tokens
            node = metadata.get("langgraph_node", "")
            if node == SUMMARIZATION_NODE_NAME:
                continue

            # Text tokens (main agent or subagent)
            if hasattr(token, "content") and isinstance(token.content, str) and token.content:
                event_kind = AgentEventKind.SUBAGENT_TOKEN if is_subagent else AgentEventKind.TOKEN
                if not is_subagent:
                    response_text += token.content
                yield AgentEvent(kind=event_kind, token=token.content)

            # Tool result messages
            if getattr(token, "type", None) == "tool":
                tool_name = getattr(token, "name", "unknown")
                output = str(token.content) if token.content else "(no output)"
                is_todo = any(k in tool_name.lower() for k in ("todo", "write_todo"))
                if not is_todo and len(output) > TOOL_OUTPUT_DISPLAY_LIMIT:
                    output = output[:TOOL_OUTPUT_DISPLAY_LIMIT] + "..."
                yield AgentEvent(
                    kind=AgentEventKind.TOOL_END,
                    tool_name=tool_name,
                    tool_output=output,
                    is_subagent=is_subagent,
                )

        elif chunk_type == "updates":
            data = _unwrap(data)
            if not isinstance(data, dict):
                continue

            # Detect tool calls from the agent node's AIMessage (top-level only)
            if not is_subagent:
                for node_name, node_output in data.items():
                    if node_name.startswith("__"):
                        continue
                    node_output = _unwrap(node_output)
                    if not isinstance(node_output, dict):
                        continue
                    messages = _unwrap(node_output.get("messages", []))
                    if not isinstance(messages, list):
                        continue
                    # Only check the LAST AIMessage — updates may contain full
                    # history via Overwrite; older messages would replay old tools.
                    last_ai = next(
                        (m for m in reversed(messages) if isinstance(m, AIMessage)),
                        None,
                    )
                    if not last_ai or not getattr(last_ai, "tool_calls", None):
                        continue
                    for tc in last_ai.tool_calls:
                            tc_id = tc.get("id", "")
                            if tc_id and tc_id in seen_tool_call_ids:
                                continue
                            if tc_id:
                                seen_tool_call_ids.add(tc_id)
                            tool_input = str(tc.get("args", ""))
                            if len(tool_input) > TOOL_INPUT_DISPLAY_LIMIT:
                                tool_input = tool_input[:TOOL_INPUT_DISPLAY_LIMIT] + "..."
                            yield AgentEvent(
                                kind=AgentEventKind.TOOL_START,
                                tool_name=tc.get("name", "unknown"),
                                tool_input=tool_input,
                            )

            # Detect HITL interrupts inline (Interrupt is a dataclass with .value)
            if "__interrupt__" in data:
                action_requests: list[dict[str, Any]] = []
                for intr in data["__interrupt__"]:
                    value = intr.value if hasattr(intr, "value") else intr
                    if isinstance(value, dict):
                        for ar in value.get("action_requests", []):
                            action_requests.append(dict(ar))
                if action_requests:
                    found_interrupt = True
                    yield AgentEvent(
                        kind=AgentEventKind.TOOL_APPROVAL_REQUIRED,
                        action_requests=action_requests,
                    )
                    return

        elif chunk_type == "custom":
            # Progress events emitted by tools via get_stream_writer()
            yield AgentEvent(
                kind=AgentEventKind.CUSTOM_PROGRESS,
                tool_output=str(data),
            )

    # Fallback: check aget_state for HITL interrupts not caught inline
    if not found_interrupt:
        state = await agent.aget_state(config)
        if state and state.tasks:
            for task in state.tasks:
                if not (hasattr(task, "interrupts") and task.interrupts):
                    continue
                action_requests = []
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
                    return

    if response_text:
        yield AgentEvent(kind=AgentEventKind.RESPONSE, response=response_text)


async def stream_agent_turn(
    agent: Any,
    user_message: str,
    thread_id: str,
    *,
    resume_command: Command | None = None,
    seen_tool_call_ids: set[str] | None = None,
) -> AsyncGenerator[AgentEvent]:
    """Stream agent events for a single conversation turn.

    Memory is managed by the agent itself via the CompositeBackend:
    ``/memories/`` paths are routed to a persistent StoreBackend, and
    the ``memory`` parameter on ``create_deep_agent`` auto-loads
    ``AGENT.md`` into the system prompt each turn.

    When *resume_command* is provided the agent is resumed from a HITL
    interrupt instead of starting a new turn.

    Args:
        agent: The LangGraph agent instance.
        user_message: The user's current input.
        thread_id: Conversation thread identifier.
        resume_command: Optional Command to resume from a HITL interrupt.
        seen_tool_call_ids: Set of tool call IDs already displayed, to avoid
            duplicates across HITL resume cycles.

    Yields:
        AgentEvent instances for tool starts, tool ends, and the final response.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": AGENT_RECURSION_LIMIT,
    }

    if resume_command is not None:
        inputs = resume_command
    else:
        inputs = {"messages": [HumanMessage(content=user_message)]}

    async for event in _stream_and_yield(agent, inputs, config, seen_tool_call_ids):
        yield event
