"""Terminal UI using Rich and prompt_toolkit."""

from __future__ import annotations

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
    "/model [nombre]": "Show or change the active Ollama model",
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


def show_tool_start(tool_name: str, tool_input: str = "") -> None:
    """Display a tool call starting."""
    input_hint = f" ({tool_input})" if tool_input else ""
    console.print(f"  [bold yellow]>[/] [yellow]{tool_name}[/][dim]{input_hint}[/]")


def show_tool_end(tool_name: str, tool_output: str = "") -> None:
    """Display a tool call result."""
    output_hint = f" -> {tool_output}" if tool_output else " -> done"
    console.print(f"  [bold green]>[/] [green]{tool_name}[/][dim]{output_hint}[/]")


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


def show_context_breakdown(breakdown: ContextBreakdown) -> None:
    """Display the context token usage breakdown."""
    table = Table(title="Context Usage", border_style="bright_blue")
    table.add_column("Category", style="bold")
    table.add_column("Tokens", justify="right")
    table.add_column("Details", style="dim")

    table.add_row("System prompt", str(breakdown.system_tokens), "")
    table.add_row("Memories", str(breakdown.memory_tokens), "injected from long-term store")
    table.add_row("Messages", str(breakdown.messages_tokens), "conversation history")
    table.add_row(
        "Tools",
        str(breakdown.tools_tokens),
        f"{breakdown.mcp_tool_count} MCP tools",
    )
    if breakdown.summary_tokens > 0:
        table.add_row("Summary", str(breakdown.summary_tokens), "summarized older messages")

    table.add_section()
    color = breakdown.usage_color
    table.add_row(
        "[bold]Total[/]",
        f"[{color} bold]{breakdown.total_tokens}[/]",
        f"[{color}]{breakdown.usage_percent:.0f}% of {breakdown.max_tokens}[/]",
    )

    console.print(table)
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
                    console.print(f"  [yellow]~ {name}[/]")
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
            # Tool result messages — show tool name with result indicator
            name = getattr(msg, "name", "tool")
            console.print(f"  [green]~ {name}[/]")
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


def prompt_tool_decision() -> str:
    """Ask the user to approve or reject a tool call.

    Returns:
        ``"approve"`` or ``"reject"``.
    """
    console.print("[bold yellow]Aprobar ejecución?[/] [dim](s/n)[/]", end=" ")
    try:
        choice = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "reject"
    if choice in ("s", "si", "sí", "y", "yes"):
        return "approve"
    return "reject"


def prompt_reject_reason() -> str:
    """Ask the user for an optional rejection reason.

    Returns:
        The reason string (may be empty).
    """
    console.print("[dim]Motivo del rechazo (Enter para omitir):[/]", end=" ")
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def show_models_table(models: list[str], current_model: str) -> None:
    """Display available Ollama models with the active one highlighted."""
    table = Table(title="Modelos disponibles en Ollama", border_style="bright_blue")
    table.add_column("Modelo", style="white")
    table.add_column("Activo", justify="center", width=8)
    for model in models:
        active = "[bold green]✓[/]" if model == current_model else ""
        table.add_row(model, active)
    console.print(table)
    console.print("[dim]Usar /model <nombre> para cambiar[/]")
    console.print()


def prompt_thread_selection(threads: list[dict]) -> str | None:
    """Prompt user to select a thread. Returns thread_id or None for new."""
    show_threads(threads)
    if not threads:
        return None

    console.print("Enter thread number to resume, or press Enter for new:")
    try:
        choice = input("> ").strip()
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
