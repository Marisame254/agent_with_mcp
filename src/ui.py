"""Terminal UI using Rich and prompt_toolkit."""

from __future__ import annotations

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import MAX_CONTEXT_TOKENS, MODEL_NAME
from src.constants import CHAT_HISTORY_FILE
from src.context_tracker import ContextBreakdown

console = Console()

COMMANDS = {
    "/new": "Start a new conversation thread",
    "/threads": "List and resume previous conversations",
    "/context": "Show token usage breakdown",
    "/memory": "Manage long-term memories",
    "/mcp [subcmd]": "Gestionar servidores MCP",
    "/model [proveedor/nombre]": "Show available models or switch provider",
    "/help": "Show this help message",
    "/exit": "Exit the application",
}

MEMORY_SUBCOMMANDS = {
    "/memory": "List all stored memories",
    "/memory add <text>": "Save a custom memory",
    "/memory search <query>": "Search memories by keyword",
    "/memory delete <n>": "Delete memory #n from the list",
    "/memory clear": "Delete all memories (asks confirmation)",
}

MCP_SUBCOMMANDS = {
    "/mcp": "Listar todos los servidores configurados",
    "/mcp enable <nombre>": "Activar un servidor deshabilitado",
    "/mcp disable <nombre>": "Deshabilitar un servidor activo",
    "/mcp reload": "Recargar mcp_servers.json y reconectar",
}


def create_prompt_session() -> PromptSession:
    """Create a prompt_toolkit session with file history."""
    return PromptSession(history=FileHistory(CHAT_HISTORY_FILE))


def show_welcome() -> None:
    """Display the welcome screen."""
    content = Text()
    content.append("Agent with MCP\n", style="bold magenta")
    content.append(f"Model: {MODEL_NAME}\n", style="dim")
    content.append(f"Max context: {MAX_CONTEXT_TOKENS} tokens\n\n", style="dim")
    content.append("Commands:\n", style="bold")
    for cmd, desc in COMMANDS.items():
        content.append(f"  {cmd:<12}", style="bold cyan")
        content.append(f"{desc}\n", style="dim")

    console.print(Panel(content, border_style="bright_blue", padding=(1, 2)))
    console.print()


def show_assistant_message(message: str) -> None:
    """Display an assistant message rendered as markdown."""
    console.print()
    console.print("[bold green]Assistant:[/]")
    console.print(Markdown(message))
    console.print()


def _format_tool_name(tool_name: str) -> str:
    """Make namespaced MCP tool names human-readable.

    Converts ``namespace__tool_name`` → ``namespace › tool_name``.
    """
    if "__" in tool_name:
        ns, _, name = tool_name.partition("__")
        return f"{ns} › {name}"
    return tool_name


def _format_tool_summary(tool_name: str, output: str) -> str:
    """Produce a compact one-line summary of tool output.

    Picks a smart format based on the tool name (search, todo, read, write)
    and falls back to the first meaningful line of the raw output.
    """
    if not output or output.strip() in ("", "(no output)"):
        return "done"

    name = tool_name.lower()
    stripped = output.strip()
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    n = len(lines)

    if any(k in name for k in ("search", "tavily", "find", "query")):
        first = lines[0][:70] if lines else stripped[:70]
        suffix = "…" if (lines and len(lines[0]) > 70) else ""
        return f"{n} resultado(s) · {first}{suffix}" if n > 1 else f"{first}{suffix}"

    if any(k in name for k in ("todo", "task", "write_todo")):
        task_lines = [l for l in lines if l.lstrip().startswith(("[", "-", "•", "*", "○", "●"))]
        count = len(task_lines) if task_lines else n
        return f"{count} tarea(s)"

    if any(k in name for k in ("read", "get", "fetch", "load", "open")):
        return f"{n} línea(s)" if n > 1 else (stripped[:80] + "…" if len(stripped) > 80 else stripped)

    if any(k in name for k in ("write", "create", "update", "save", "put", "delete", "remove")):
        return "completado"

    first = lines[0] if lines else stripped
    return (first[:80] + "…") if len(first) > 80 else first


def show_tool_start(tool_name: str, tool_input: str = "") -> None:
    """Display a tool call in Claude Code style: ● ToolName(arg)"""
    display = _format_tool_name(tool_name)
    if tool_input:
        arg = (tool_input[:60] + "…") if len(tool_input) > 60 else tool_input
        console.print(f"[bold orange1]●[/] [bold]{display}[/][dim]({arg})[/]")
    else:
        console.print(f"[bold orange1]●[/] [bold]{display}[/]")


def show_tool_end(tool_name: str, tool_output: str = "") -> None:
    """Display a tool result in Claude Code style: ⎿  summary"""
    summary = _format_tool_summary(tool_name, tool_output)
    console.print(f"  [dim]⎿  {summary}[/]")
    console.print()


def show_error(message: str) -> None:
    """Display an error message."""
    console.print(f"[bold red]Error:[/] {message}")


def show_info(message: str) -> None:
    """Display an info message."""
    console.print(f"[dim]{message}[/]")


def show_threads(threads: list[dict]) -> None:
    """Display a table of conversation threads."""
    if not threads:
        console.print("[dim]No previous conversations found.[/]")
        return

    table = Table(title="Conversation Threads", border_style="bright_blue")
    table.add_column("#", style="bold", width=4)
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Name", style="white")

    for i, thread in enumerate(threads, 1):
        short_id = thread["thread_id"][:8]
        name = thread.get("name") or thread.get("preview", "")
        table.add_row(str(i), short_id, name)

    console.print(table)
    console.print()


def show_help() -> None:
    """Display help for all available commands."""
    content = Text()
    content.append("Commands:\n\n", style="bold")
    for cmd, desc in COMMANDS.items():
        content.append(f"  {cmd:<12}", style="bold cyan")
        content.append(f"{desc}\n", style="dim")

    content.append("\nMemory subcommands:\n\n", style="bold")
    for cmd, desc in MEMORY_SUBCOMMANDS.items():
        content.append(f"  {cmd:<28}", style="bold cyan")
        content.append(f"{desc}\n", style="dim")

    content.append("\nMCP subcommands:\n\n", style="bold")
    for cmd, desc in MCP_SUBCOMMANDS.items():
        content.append(f"  {cmd:<28}", style="bold cyan")
        content.append(f"{desc}\n", style="dim")

    console.print(Panel(content, title="Help", border_style="bright_blue", padding=(1, 2)))
    console.print()


def show_memories_table(memories: list[dict]) -> None:
    """Display a table of stored memories."""
    table = Table(title="Memories", border_style="bright_blue")
    table.add_column("#", style="bold", width=4)
    table.add_column("Memory", style="white")

    for i, mem in enumerate(memories, 1):
        table.add_row(str(i), mem["text"])

    console.print(table)
    console.print()


def show_mcp_table(mcp_config: dict, disabled_servers: frozenset[str]) -> None:
    """Display a table of configured MCP servers with their status."""
    if not mcp_config:
        console.print("[dim]No hay servidores MCP configurados en mcp_servers.json.[/]")
        console.print()
        return

    table = Table(title="Servidores MCP", border_style="bright_blue")
    table.add_column("Nombre", style="bold", width=20)
    table.add_column("Estado", justify="center", width=14)
    table.add_column("Transporte", style="dim", width=10)
    table.add_column("Comando", style="dim")

    for name, cfg in mcp_config.items():
        is_disabled = name in disabled_servers
        status = "[red]deshabilitado[/]" if is_disabled else "[green]activo[/]"
        transport = cfg.get("transport", "")
        command_parts = [cfg.get("command", "")] + list(cfg.get("args", []))
        command_str = " ".join(str(p) for p in command_parts)
        if len(command_str) > 50:
            command_str = command_str[:47] + "..."
        table.add_row(name, status, transport, command_str)

    console.print(table)
    active = len(mcp_config) - len(disabled_servers & mcp_config.keys())
    console.print(f"[dim]{active} activo(s) de {len(mcp_config)} configurado(s)[/]")
    console.print()


def _build_context_blocks(breakdown: ContextBreakdown, n: int = 100) -> list[str]:
    """Generate N colored block characters for the context grid.

    Fills in order: system (dim) → tools (cyan) → memories (orange) →
    messages (magenta) → free (dim ⛶).
    """
    scale = n / max(breakdown.max_tokens, 1)

    categories = [
        (breakdown.system_tokens, "dim white", "⛁"),
        (breakdown.tools_tokens, "cyan", "⛁"),
        (breakdown.memory_tokens, "orange1", "⛁"),
        (breakdown.messages_tokens, "magenta", "⛁"),
    ]

    result: list[str] = []
    for tokens, color, sym in categories:
        for _ in range(round(tokens * scale)):
            if len(result) < n:
                result.append(f"[{color}]{sym}[/]")

    while len(result) < n:
        result.append("[dim]⛶[/]")

    return result[:n]


def show_context_breakdown(breakdown: ContextBreakdown) -> None:
    """Display context usage in Claude Code style: block grid + right column."""
    ROWS, COLS = 10, 10

    def fmt(t: int) -> str:
        return f"{t / 1000:.1f}k" if t >= 1000 else str(t)

    def pct(t: int) -> str:
        return f"{t / breakdown.max_tokens * 100:.1f}%" if breakdown.max_tokens else "0%"

    color = breakdown.usage_color
    free = max(0, breakdown.max_tokens - breakdown.total_tokens)
    model = MODEL_NAME

    right: list[str] = [
        f"[{color}]{model} · {fmt(breakdown.total_tokens)}/{fmt(breakdown.max_tokens)} tokens ({breakdown.usage_percent:.0f}%)[/]",
        "",
        "[dim italic]Uso estimado por categoría[/]",
        f"[dim white]⛁[/] System prompt: [dim]{fmt(breakdown.system_tokens)} tokens ({pct(breakdown.system_tokens)})[/]",
        f"[cyan]⛁[/] Tools: [dim]{fmt(breakdown.tools_tokens)} tokens ({pct(breakdown.tools_tokens)}, {breakdown.mcp_tool_count} MCP)[/]",
        f"[orange1]⛁[/] Memories: [dim]{fmt(breakdown.memory_tokens)} tokens ({pct(breakdown.memory_tokens)})[/]",
        f"[magenta]⛁[/] Messages: [dim]{fmt(breakdown.messages_tokens)} tokens ({pct(breakdown.messages_tokens)})[/]",
        f"[dim]⛶ Espacio libre: {fmt(free)} ({free / breakdown.max_tokens * 100:.1f}%)[/]" if breakdown.max_tokens else "",
    ]
    if breakdown.summary_tokens > 0:
        right.append(
            f"[yellow]⛁[/] Resumen activo: [dim]{fmt(breakdown.summary_tokens)} tokens ({pct(breakdown.summary_tokens)})[/]"
        )

    blocks = _build_context_blocks(breakdown, ROWS * COLS)

    console.print()
    console.print("[bold]❯[/] [dim]/context[/]")
    console.print("  [dim]⎿[/]  [bold]Context Usage[/]")
    console.print()

    indent = "     "
    for row in range(ROWS):
        row_blocks = blocks[row * COLS : (row + 1) * COLS]
        block_str = " ".join(row_blocks)
        right_text = right[row] if row < len(right) else ""
        if right_text:
            console.print(f"{indent}{block_str}  {right_text}")
        else:
            console.print(f"{indent}{block_str}")

    console.print()


def show_conversation_history(messages: list) -> None:
    """Display previous messages when resuming a thread."""
    if not messages:
        return

    console.print(
        Panel("[bold]Previous conversation[/]", border_style="dim", padding=(0, 1))
    )
    for msg in messages:
        msg_type = getattr(msg, "type", None)

        # AI messages that are pure tool invocations (no text content)
        if msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None) or []
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                    console.print(f"[dim orange1]●[/] [dim]{_format_tool_name(name)}[/]")
                if not content.strip():
                    continue
            elif not content.strip():
                continue
            console.print()
            console.print("[bold green]Assistant:[/]")
            console.print(Markdown(content))
        elif msg_type == "human":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            console.print(f"[bold blue]You>[/] {content}")
        elif msg_type == "tool":
            name = getattr(msg, "name", "tool")
            console.print(f"  [dim]⎿  {_format_tool_name(name)}[/]")
    console.print()
    console.rule(style="dim")
    console.print()


def show_agent_question(question: str) -> None:
    """Display a question from the agent that requires user input."""
    console.print()
    console.print(
        Panel(
            question,
            title="[bold yellow]El agente necesita tu respuesta[/]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def show_tool_approval(action_requests: list[dict]) -> None:
    """Display the tool calls that require user approval."""
    table = Table(
        title="Aprobación requerida",
        border_style="yellow",
    )
    table.add_column("#", style="bold", width=4)
    table.add_column("Herramienta", style="cyan")
    table.add_column("Argumentos", style="white")

    for i, ar in enumerate(action_requests, 1):
        name = ar.get("name", "unknown")
        args = str(ar.get("args", {}))
        if len(args) > 120:
            args = args[:120] + "..."
        table.add_row(str(i), name, args)

    console.print()
    console.print(table)
    console.print()


async def prompt_tool_decision() -> str:
    """Ask the user to approve or reject a tool call.

    Returns:
        ``"approve"`` or ``"reject"``.
    """
    console.print("[bold yellow]Aprobar ejecución?[/] [dim](s/n)[/]", end=" ")
    loop = asyncio.get_running_loop()
    try:
        choice = await loop.run_in_executor(None, lambda: input().strip().lower())
    except (EOFError, KeyboardInterrupt):
        return "reject"
    return "approve" if choice in ("s", "si", "sí", "y", "yes") else "reject"


async def prompt_reject_reason() -> str:
    """Ask the user for an optional rejection reason.

    Returns:
        The reason string (may be empty).
    """
    console.print("[dim]Motivo del rechazo (Enter para omitir):[/]", end=" ")
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, lambda: input().strip())
    except (EOFError, KeyboardInterrupt):
        return ""


def show_models_table(models_by_provider: dict[str, list[str]], current_model: str) -> None:
    """Display available models grouped by provider with the active one highlighted.

    Args:
        models_by_provider: Mapping of provider name to list of model names.
        current_model: Active model string in ``provider/name`` or bare format.
    """
    table = Table(title="Modelos disponibles", border_style="bright_blue")
    table.add_column("Proveedor", style="cyan", width=12)
    table.add_column("Modelo", style="white")
    table.add_column("Activo", justify="center", width=8)

    for provider, models in models_by_provider.items():
        for model in models:
            full_name = f"{provider}/{model}"
            # Match both "provider/name" and bare "name" (Ollama backward compat)
            is_active = current_model in (full_name, model)
            active = "[bold green]✓[/]" if is_active else ""
            table.add_row(provider, model, active)

    console.print(table)
    console.print("[dim]Usar /model <proveedor>/<nombre> para cambiar  (ej. openai/gpt-4o)[/]")
    console.print()


async def prompt_thread_selection(threads: list[dict]) -> str | None:
    """Prompt user to select a thread. Returns thread_id or None for new."""
    show_threads(threads)
    if not threads:
        return None

    console.print("Enter thread number to resume, or press Enter for new:")
    try:
        loop = asyncio.get_running_loop()
        choice = await loop.run_in_executor(None, lambda: input(" >> ").strip())
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice:
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(threads):
            return threads[idx]["thread_id"]
    except ValueError:
        pass

    show_error("Invalid selection, starting new thread.")
    return None
