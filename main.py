"""Entry point for the Agent with MCP."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.live import Live
from rich.markdown import Markdown

from src.agent import build_agent, create_agent_resources, get_system_prompt, get_thread_history, stream_agent_turn
from src.config import MAX_CONTEXT_TOKENS, load_mcp_servers, setup_logging, validate_config
from src.constants import AgentEventKind, ChatCommand
from src.context_tracker import build_context_breakdown
from src.memory import clear_memories, delete_memory, list_memories, retrieve_memories, store_memories
from src.ui import (
    console,
    create_prompt_session,
    prompt_thread_selection,
    show_assistant_message,
    show_context_breakdown,
    show_error,
    show_info,
    show_tool_end,
    show_tool_start,
    show_memories_table,
    show_help,
    show_welcome,
)

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class ChatLoopResult:
    """Result returned by the chat loop to control thread lifecycle.

    Attributes:
        command: The action to take (NEW thread or EXIT).
        thread_id: Target thread ID when resuming a conversation.
    """

    command: ChatCommand
    thread_id: str = ""


async def handle_context_command(
    store,
    thread_id: str,
    user_id: str,
    all_tools: list,
    mcp_tool_count: int,
    messages: list,
) -> None:
    """Handle the /context command by displaying token usage breakdown."""
    memories = await retrieve_memories(store, user_id, "")
    tool_defs = [{"name": getattr(t, "name", str(t))} for t in all_tools]

    breakdown = build_context_breakdown(
        system_prompt=get_system_prompt(),
        memories=memories,
        messages=messages,
        tool_definitions=tool_defs,
        mcp_tool_count=mcp_tool_count,
        max_tokens=MAX_CONTEXT_TOKENS,
    )
    show_context_breakdown(breakdown)


async def chat_loop(
    agent,
    store,
    checkpointer,
    all_tools: list,
    mcp_tool_count: int,
    thread_id: str,
    user_id: str = "default",
) -> ChatLoopResult | None:
    """Run the main chat loop.

    Returns:
        A ChatLoopResult describing what to do next, or None to exit.
    """
    session = create_prompt_session()
    messages: list[HumanMessage | AIMessage] = []

    show_info(f"Thread: {thread_id}")
    console.print()

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: session.prompt("You> ")
            )
        except (EOFError, KeyboardInterrupt):
            return None

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/exit":
            return None

        if user_input.lower() == "/new":
            return ChatLoopResult(command=ChatCommand.NEW)

        if user_input.lower() == "/threads":
            threads = await get_thread_history(checkpointer)
            selected = prompt_thread_selection(threads)
            if selected:
                return ChatLoopResult(command=ChatCommand.NEW, thread_id=selected)
            continue

        if user_input.lower() == "/help":
            show_help()
            continue

        if user_input.lower() == "/context":
            await handle_context_command(
                store, thread_id, user_id, all_tools, mcp_tool_count, messages
            )
            continue

        if user_input.lower().startswith("/memory"):
            parts = user_input.split(maxsplit=2)
            subcmd = parts[1].lower() if len(parts) > 1 else ""

            if subcmd == "help":
                show_help()
            elif subcmd == "search":
                query = parts[2] if len(parts) > 2 else ""
                if not query:
                    show_error("Usage: /memory search <query>")
                    continue
                results = await retrieve_memories(store, user_id, query, max_results=20)
                if not results:
                    show_info("No memories found for that query.")
                else:
                    show_memories_table([{"key": "", "text": t} for t in results])
            elif subcmd == "delete":
                idx_str = parts[2] if len(parts) > 2 else ""
                if not idx_str:
                    show_error("Usage: /memory delete <number>")
                    continue
                try:
                    idx = int(idx_str)
                except ValueError:
                    show_error("Please provide a valid memory number.")
                    continue
                mems = await list_memories(store, user_id)
                if not mems:
                    show_info("No memories stored yet.")
                    continue
                if idx < 1 or idx > len(mems):
                    show_error(f"Invalid index. Must be between 1 and {len(mems)}.")
                    continue
                target = mems[idx - 1]
                await delete_memory(store, user_id, target["key"])
                show_info(f"Deleted memory #{idx}: {target['text']}")
            elif subcmd == "add":
                text = parts[2] if len(parts) > 2 else ""
                if not text:
                    show_error("Usage: /memory add <text>")
                    continue
                await store_memories(store, user_id, [text])
                show_info(f"Memory saved: {text}")
            elif subcmd == "clear":
                try:
                    confirm = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("Type 'yes' to confirm clearing all memories: ")
                    )
                except (EOFError, KeyboardInterrupt):
                    continue
                if confirm.strip().lower() != "yes":
                    show_info("Cancelled.")
                    continue
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
            continue

        # Regular message
        try:
            response_text = ""
            streaming_started = False
            streamed_text = ""
            live: Live | None = None
            status = console.status("[bold green]Thinking...", spinner="dots")
            status.start()

            async for event in stream_agent_turn(
                agent, store, user_input, thread_id, user_id
            ):
                if event.kind == AgentEventKind.TOOL_START:
                    if live:
                        live.stop()
                        live = None
                        streaming_started = False
                        streamed_text = ""
                    status.stop()
                    show_tool_start(event.tool_name, event.tool_input)
                    status = console.status(
                        f"[bold yellow]Running {event.tool_name}...", spinner="dots"
                    )
                    status.start()
                elif event.kind == AgentEventKind.TOOL_END:
                    status.stop()
                    show_tool_end(event.tool_name, event.tool_output)
                    status = console.status("[bold green]Thinking...", spinner="dots")
                    status.start()
                elif event.kind == AgentEventKind.TOKEN:
                    if not streaming_started:
                        status.stop()
                        console.print()
                        console.print("[bold green]Assistant:[/]")
                        live = Live(
                            Markdown(""),
                            console=console,
                            refresh_per_second=8,
                        )
                        live.start()
                        streaming_started = True
                    streamed_text += event.token
                    live.update(Markdown(streamed_text))
                elif event.kind == AgentEventKind.RESPONSE:
                    response_text = event.response

            status.stop()
            if live:
                live.stop()

            if response_text:
                if not streaming_started:
                    show_assistant_message(response_text)  # Fallback
                else:
                    console.print()
                messages.append(HumanMessage(content=user_input))
                messages.append(AIMessage(content=response_text))
            else:
                show_error("No response from agent.")
        except Exception as e:
            try:
                status.stop()
            except Exception:
                pass
            show_error(f"Agent error: {e}")


async def main() -> None:
    """Main entry point: validate config, connect to services, and run the chat loop."""
    errors = validate_config()
    if errors:
        for err in errors:
            show_error(err)
        return

    show_welcome()

    mcp_config = load_mcp_servers()
    if not mcp_config:
        show_info("No MCP servers configured in mcp_servers.json. Running without MCP tools.")

    checkpointer_cm, store_cm = await create_agent_resources()

    async with checkpointer_cm as checkpointer, store_cm as store:
        await checkpointer.setup()
        await store.setup()

        mcp_client = None
        if mcp_config:
            try:
                mcp_client = MultiServerMCPClient(mcp_config)
                server_names = list(mcp_config.keys())
                logger.info("MCP client initialized with servers: %s", server_names)
                show_info(f"MCP servers: {', '.join(server_names)}")
            except Exception as e:
                logger.warning("Failed to initialize MCP client: %s", e)
                show_error(f"MCP initialization failed: {e}. Continuing without MCP tools.")

        agent, all_tools, mcp_tool_count = await build_agent(
            mcp_client, checkpointer, store
        )

        thread_id = str(uuid.uuid4())

        threads = await get_thread_history(checkpointer)
        if threads:
            selected = prompt_thread_selection(threads)
            if selected:
                thread_id = selected

        while True:
            result = await chat_loop(
                agent, store, checkpointer, all_tools, mcp_tool_count, thread_id
            )

            if result is None:
                break
            elif result.thread_id:
                thread_id = result.thread_id
                show_info(f"Resumed thread: {thread_id}")
            elif result.command == ChatCommand.NEW:
                thread_id = str(uuid.uuid4())
                show_info(f"New thread: {thread_id}")

    console.print("[dim]Goodbye![/]")


if __name__ == "__main__":
    # psycopg requires SelectorEventLoop on Windows (ProactorEventLoop is not supported)
    import sys

    if sys.platform == "win32":
        import selectors

        asyncio.run(
            main(),
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
        )
    else:
        asyncio.run(main())
