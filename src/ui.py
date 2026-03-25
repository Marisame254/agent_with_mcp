"""Terminal UI using Rich and prompt_toolkit."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import MAX_CONTEXT_TOKENS, MODEL_NAME
from src.constants import CHAT_HISTORY_FILE, ThreadAction
from src.context_tracker import ContextBreakdown

console = Console()

COMMANDS = {
    "/new": "Start a new conversation thread",
    "/threads": "Listar, reanudar, eliminar o renombrar conversaciones",
    "/context": "Show token usage breakdown",
    "/memory": "Manage long-term memories",
    "/mcp [subcmd]": "Gestionar servidores MCP",
    "/model [proveedor/nombre]": "Show available models or switch provider",
    "/help": "Show this help message",
    "/exit": "Exit the application",
}

MEMORY_SUBCOMMANDS = {
    "/memory": "Mostrar memorias del agente (AGENT.md)",
    "/memory clear": "Borrar todas las memorias (pide confirmación)",
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


def _format_todo_summary(output: str) -> str:
    """Parse write_todos output and format as a checklist.

    Expected output format:
        ``Updated todo list to [{'content': '...', 'status': 'pending'}, ...]``
    """
    import ast

    status_marks = {"completed": "x", "in_progress": "-", "pending": " "}
    bracket_pos = output.find("[")
    if bracket_pos == -1:
        return output[:80] if len(output) <= 80 else output[:77] + "…"
    try:
        todos = ast.literal_eval(output[bracket_pos:])
    except (ValueError, SyntaxError):
        return f"{output.count('content')} tarea(s)"
    if not isinstance(todos, list) or not todos:
        return "0 tarea(s)"
    formatted = []
    for t in todos:
        mark = status_marks.get(t.get("status", ""), " ")
        formatted.append(f"[{mark}] {t.get('content', '?')}")
    return "\n".join(formatted)


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

    if "todo" in name:
        return _format_todo_summary(stripped)

    if any(k in name for k in ("write", "create", "update", "save", "put", "delete", "remove")):
        return "completado"

    if any(k in name for k in ("read", "get", "fetch", "load", "open")):
        return f"{n} línea(s)" if n > 1 else (stripped[:80] + "…" if len(stripped) > 80 else stripped)

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
    if "\n" in summary:
        first, *rest = summary.split("\n")
        console.print(f"  [dim]⎿  {first}[/]")
        for line in rest:
            console.print(f"  [dim]   {line}[/]")
    else:
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
            if getattr(msg, "additional_kwargs", {}).get("lc_source") == "summarization":
                console.print()
                console.print(
                    Panel(
                        Markdown(content),
                        title="[yellow bold]Resumen de conversación anterior[/]",
                        border_style="yellow",
                        padding=(0, 2),
                    )
                )
            else:
                console.print(f"[bold blue]You>[/] {content}")
        elif msg_type == "tool":
            name = getattr(msg, "name", "tool")
            console.print(f"  [dim]⎿  {_format_tool_name(name)}[/]")
    console.print()
    console.rule(style="dim")
    console.print()


def prompt_option_selection(
    question: str,
    options: list[dict],
    multi_select: bool = False,
) -> str:
    """Interactive option selector using prompt_toolkit Application.

    Args:
        question: The question to display as header.
        options: List of ``{"title": str, "description": str | None}`` dicts.
        multi_select: Allow multiple selections when True.

    Returns:
        Comma-separated titles of selected options, free-text input,
        or ``"(cancelado por el usuario)"`` on Esc.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.layout import Layout

    # Append "Escribir respuesta..." as the last option
    items = [*options, {"title": "Escribir respuesta...", "_free_text": True}]
    cursor = [0]
    checked: set[int] = set()
    result: list[str] = []
    cancelled = [False]

    def _get_body() -> FormattedText:
        fragments: list[tuple[str, str]] = []
        fragments.append(("bold fg:yellow", f"  {question}\n\n"))
        for i, item in enumerate(items):
            is_cur = i == cursor[0]
            is_free = item.get("_free_text", False)
            title = item.get("title", "")
            desc = item.get("description", "")

            prefix = ") " if is_cur else "  "
            num = f"{i + 1}. "

            mark = ("[✓] " if i in checked else "[ ] ") if multi_select else ""

            # Title line
            title_style = "bold" if is_cur else ""
            if is_free:
                title_style += " italic fg:ansigray"
            fragments.append(("", prefix))
            fragments.append(("fg:ansicyan", num))
            fragments.append((title_style.strip(), f"{mark}{title}\n"))

            # Description line
            if desc and not is_free:
                fragments.append(("fg:ansigray", f"     {desc}\n"))

        # Status bar
        fragments.append(("", "\n"))
        if multi_select:
            fragments.append(
                ("fg:ansigray italic",
                 "  Enter confirmar · Space toggle · ↑↓ navegar · Esc cancelar")
            )
        else:
            fragments.append(
                ("fg:ansigray italic",
                 "  Enter seleccionar · ↑↓ navegar · Esc cancelar")
            )
        return FormattedText(fragments)

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("s-tab")
    def _up(event) -> None:  # noqa: ANN001
        cursor[0] = (cursor[0] - 1) % len(items)

    @kb.add("down")
    @kb.add("tab")
    def _down(event) -> None:  # noqa: ANN001
        cursor[0] = (cursor[0] + 1) % len(items)

    @kb.add("space")
    def _toggle(event) -> None:  # noqa: ANN001
        if not multi_select:
            return
        idx = cursor[0]
        # Don't toggle the free-text option
        if items[idx].get("_free_text"):
            return
        if idx in checked:
            checked.discard(idx)
        else:
            checked.add(idx)

    @kb.add("enter")
    def _select(event) -> None:  # noqa: ANN001
        idx = cursor[0]
        if multi_select:
            # In multi-select, Enter confirms checked items (or free-text if selected)
            if items[idx].get("_free_text") and idx in checked:
                # Free text was toggled — treat like single select free-text
                result.append("__free_text__")
            elif items[idx].get("_free_text") and not checked:
                # Nothing checked and cursor on free-text — go to free text
                result.append("__free_text__")
            elif checked:
                for ci in sorted(checked):
                    result.append(items[ci]["title"])
            elif items[idx].get("_free_text"):
                result.append("__free_text__")
            else:
                # Nothing checked — select current item
                result.append(items[idx]["title"])
        else:
            if items[idx].get("_free_text"):
                result.append("__free_text__")
            else:
                result.append(items[idx]["title"])
        event.app.exit()

    @kb.add("escape")
    def _cancel(event) -> None:  # noqa: ANN001
        cancelled[0] = True
        event.app.exit()

    body = Window(content=FormattedTextControl(_get_body), wrap_lines=True)
    layout = Layout(HSplit([body]))
    app: Application[None] = Application(layout=layout, key_bindings=kb, full_screen=False)
    app.run()

    if cancelled[0]:
        return "(cancelado por el usuario)"

    if result == ["__free_text__"]:
        # Fall back to plain text input
        try:
            answer = input("Tu respuesta> ").strip()
        except (EOFError, KeyboardInterrupt):
            return "(sin respuesta)"
        return answer or "(sin respuesta)"

    return ", ".join(result) if result else "(sin respuesta)"


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


@dataclass
class ThreadManagementResult:
    """Result of the thread management prompt."""

    action: ThreadAction
    thread_id: str = ""
    new_name: str = ""


async def prompt_thread_management(threads: list[dict]) -> ThreadManagementResult:
    """Show thread list and prompt for an action: resume, delete, rename, or new.

    Prompt syntax accepted:
        <number>              — resume thread
        d <number>            — delete thread
        r <number> <name>     — rename thread
        Enter (empty)         — start/stay on new thread

    Loops on invalid input until a valid command is entered.
    """
    show_threads(threads)
    if not threads:
        return ThreadManagementResult(action=ThreadAction.NEW)

    console.print(
        "[dim]  <número> Reanudar  ·  [bold]d <n>[/] Eliminar  ·  "
        "[bold]r <n> <nombre>[/] Renombrar  ·  Enter Nueva conversación[/]"
    )

    loop = asyncio.get_running_loop()
    while True:
        try:
            raw = await loop.run_in_executor(None, lambda: input(" >> ").strip())
        except (EOFError, KeyboardInterrupt):
            return ThreadManagementResult(action=ThreadAction.NEW)

        if not raw:
            return ThreadManagementResult(action=ThreadAction.NEW)

        parts = raw.split(maxsplit=2)
        cmd = parts[0].lower()

        # Resume: bare number
        if cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(threads):
                return ThreadManagementResult(
                    action=ThreadAction.RESUME,
                    thread_id=threads[idx]["thread_id"],
                )
            show_error(f"Número inválido. Ingresa entre 1 y {len(threads)}.")
            continue

        # Delete: d <n>
        if cmd == "d":
            if len(parts) < 2 or not parts[1].isdigit():
                show_error("Uso: d <número>")
                continue
            idx = int(parts[1]) - 1
            if not (0 <= idx < len(threads)):
                show_error(f"Número inválido. Ingresa entre 1 y {len(threads)}.")
                continue
            return ThreadManagementResult(
                action=ThreadAction.DELETE,
                thread_id=threads[idx]["thread_id"],
            )

        # Rename: r <n> <name>
        if cmd == "r":
            if len(parts) < 3 or not parts[1].isdigit():
                show_error("Uso: r <número> <nuevo nombre>")
                continue
            idx = int(parts[1]) - 1
            if not (0 <= idx < len(threads)):
                show_error(f"Número inválido. Ingresa entre 1 y {len(threads)}.")
                continue
            new_name = parts[2].strip()
            if not new_name:
                show_error("El nombre no puede estar vacío.")
                continue
            return ThreadManagementResult(
                action=ThreadAction.RENAME,
                thread_id=threads[idx]["thread_id"],
                new_name=new_name,
            )

        show_error("Comando no reconocido. Usa un número, 'd <n>', 'r <n> <nombre>', o Enter.")


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
