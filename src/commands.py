"""Chat command handlers for the interactive loop.

Each handler corresponds to a slash command available in the chat UI.
All handlers are async and return either a ChatLoopResult (to signal a
lifecycle change) or None (command was handled in-place).
"""

from __future__ import annotations

import asyncio
import logging

from src.constants import ChatCommand
from src.context_tracker import build_context_breakdown
from src.memory import (
    clear_memories,
    delete_memory,
    list_memories,
    retrieve_memories,
    store_memories,
)
from src.providers import DEEPSEEK_MODELS, OPENAI_MODELS, list_ollama_models
from src.ui import (
    MCP_SUBCOMMANDS,
    console,
    show_context_breakdown,
    show_error,
    show_info,
    show_memories_table,
    show_help,
    show_models_table,
    show_mcp_table,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type — defined here to avoid circular imports with main.py
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class ChatLoopResult:
    """Signals what the main loop should do after a command returns.

    Attributes:
        command: The action to take (NEW thread, EXIT, MODEL change, or MCP reload).
        thread_id: Target thread ID when resuming a conversation.
        model_name: New model name when command is MODEL.
        mcp_disabled: Updated set of disabled server names when command is MCP_RELOAD.
    """

    command: ChatCommand
    thread_id: str = ""
    model_name: str = ""
    mcp_disabled: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------


async def handle_context_command(
    agent,
    store,
    thread_id: str,
    user_id: str,
    all_tools: list,
    mcp_tool_count: int,
    max_context_tokens: int,
) -> None:
    """Display token usage breakdown for the current conversation."""
    from src.agent import get_system_prompt
    from src.memory import retrieve_memories

    config = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(config)
    checkpoint_messages = state.values.get("messages", []) if state.values else []

    memories = await retrieve_memories(store, user_id, "")

    breakdown = build_context_breakdown(
        system_prompt=get_system_prompt(),
        memories=memories,
        messages=checkpoint_messages,
        tools=all_tools,
        mcp_tool_count=mcp_tool_count,
        max_tokens=max_context_tokens,
    )
    show_context_breakdown(breakdown)


# ---------------------------------------------------------------------------
# /memory
# ---------------------------------------------------------------------------


async def handle_memory_command(
    parts: list[str],
    store,
    user_id: str,
) -> None:
    """Handle the /memory command and its subcommands."""
    subcmd = parts[1].lower() if len(parts) > 1 else ""

    if subcmd == "help":
        show_help()

    elif subcmd == "search":
        query = parts[2] if len(parts) > 2 else ""
        if not query:
            show_error("Usage: /memory search <query>")
            return
        results = await retrieve_memories(store, user_id, query, max_results=20)
        if not results:
            show_info("No memories found for that query.")
        else:
            show_memories_table([{"key": "", "text": t} for t in results])

    elif subcmd == "delete":
        idx_str = parts[2] if len(parts) > 2 else ""
        if not idx_str:
            show_error("Usage: /memory delete <number>")
            return
        try:
            idx = int(idx_str)
        except ValueError:
            show_error("Please provide a valid memory number.")
            return
        mems = await list_memories(store, user_id)
        if not mems:
            show_info("No memories stored yet.")
            return
        if idx < 1 or idx > len(mems):
            show_error(f"Invalid index. Must be between 1 and {len(mems)}.")
            return
        target = mems[idx - 1]
        await delete_memory(store, user_id, target["key"])
        show_info(f"Deleted memory #{idx}: {target['text']}")

    elif subcmd == "add":
        text = parts[2] if len(parts) > 2 else ""
        if not text:
            show_error("Usage: /memory add <text>")
            return
        await store_memories(store, user_id, [text])
        show_info(f"Memory saved: {text}")

    elif subcmd == "clear":
        try:
            confirm = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: input("Type 'yes' to confirm clearing all memories: "),
            )
        except (EOFError, KeyboardInterrupt):
            return
        if confirm.strip().lower() != "yes":
            show_info("Cancelled.")
            return
        count = await clear_memories(store, user_id)
        show_info(f"Cleared {count} memories.")

    elif subcmd == "":
        mems = await list_memories(store, user_id)
        if not mems:
            show_info("No memories stored yet.")
        else:
            show_memories_table(mems)

    else:
        show_error(f"Unknown subcommand: {subcmd}")
        show_help()


# ---------------------------------------------------------------------------
# /mcp
# ---------------------------------------------------------------------------


async def handle_mcp_command(
    parts: list[str],
    mcp_config: dict,
    disabled_servers: frozenset[str],
) -> ChatLoopResult | None:
    """Handle the /mcp command and its subcommands.

    Returns a ChatLoopResult with MCP_RELOAD when a state change is needed,
    or None when the command only displays information.
    """
    from rich.panel import Panel
    from rich.text import Text

    subcmd = parts[1].lower() if len(parts) > 1 else "list"

    if subcmd == "list":
        show_mcp_table(mcp_config, disabled_servers)
        return None

    if subcmd == "help":
        content = Text()
        content.append("MCP subcommands:\n\n", style="bold")
        for cmd, desc in MCP_SUBCOMMANDS.items():
            content.append(f"  {cmd:<28}", style="bold cyan")
            content.append(f"{desc}\n", style="dim")
        console.print(Panel(content, title="MCP Help", border_style="bright_blue", padding=(1, 2)))
        console.print()
        return None

    if subcmd == "reload":
        return ChatLoopResult(command=ChatCommand.MCP_RELOAD, mcp_disabled=disabled_servers)

    if subcmd == "disable":
        name = parts[2] if len(parts) > 2 else ""
        if not name:
            show_error("Usage: /mcp disable <nombre>")
            return None
        if name not in mcp_config:
            show_error(f"Servidor '{name}' no encontrado en mcp_servers.json.")
            return None
        if name in disabled_servers:
            show_info(f"El servidor '{name}' ya está deshabilitado.")
            return None
        show_info(f"Deshabilitando servidor '{name}'...")
        return ChatLoopResult(
            command=ChatCommand.MCP_RELOAD,
            mcp_disabled=frozenset(disabled_servers | {name}),
        )

    if subcmd == "enable":
        name = parts[2] if len(parts) > 2 else ""
        if not name:
            show_error("Usage: /mcp enable <nombre>")
            return None
        if name not in mcp_config:
            show_error(f"Servidor '{name}' no encontrado en mcp_servers.json.")
            return None
        if name not in disabled_servers:
            show_info(f"El servidor '{name}' ya está activo.")
            return None
        show_info(f"Habilitando servidor '{name}'...")
        return ChatLoopResult(
            command=ChatCommand.MCP_RELOAD,
            mcp_disabled=frozenset(disabled_servers - {name}),
        )

    show_error(f"Subcomando desconocido: '{subcmd}'. Usa /mcp help para ver las opciones.")
    return None


# ---------------------------------------------------------------------------
# /model
# ---------------------------------------------------------------------------


async def handle_model_command(
    parts: list[str],
    current_model: str,
) -> ChatLoopResult | None:
    """Handle the /model command.

    Returns a ChatLoopResult with MODEL when switching models,
    or None when only listing available models.
    """
    new_model = parts[1].strip() if len(parts) > 1 else ""
    if new_model:
        return ChatLoopResult(command=ChatCommand.MODEL, model_name=new_model)

    # List available models
    models_by_provider: dict[str, list[str]] = {}
    try:
        models_by_provider["ollama"] = await list_ollama_models()
    except Exception as e:
        show_error(f"No se pudo obtener modelos de Ollama: {e}")
    models_by_provider["openai"] = list(OPENAI_MODELS)
    models_by_provider["deepseek"] = list(DEEPSEEK_MODELS)
    show_models_table(models_by_provider, current_model)
    return None
