"""Entry point for the Agent with MCP."""

from __future__ import annotations

import asyncio
import uuid

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agent import build_agent, create_agent_resources, get_thread_history, stream_agent_turn
from src.config import DATABASE_URL, MAX_CONTEXT_TOKENS, load_mcp_servers, validate_config
from src.context_tracker import ContextBreakdown, build_context_breakdown
from src.memory import retrieve_memories
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
    show_welcome,
)


async def handle_context_command(
    store,
    thread_id: str,
    user_id: str,
    all_tools: list,
    mcp_tool_count: int,
    messages: list,
) -> None:
    """Handle the /context command."""
    from src.agent import SYSTEM_PROMPT

    memories = await retrieve_memories(store, user_id, "")
    tool_defs = [{"name": getattr(t, "name", str(t))} for t in all_tools]

    breakdown = build_context_breakdown(
        system_prompt=SYSTEM_PROMPT,
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
) -> str | None:
    """Run the main chat loop. Returns new thread_id if /new, or None to exit."""
    session = create_prompt_session()
    messages = []

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
            return "NEW"

        if user_input.lower() == "/threads":
            threads = await get_thread_history(checkpointer)
            selected = prompt_thread_selection(threads)
            if selected:
                return f"RESUME:{selected}"
            continue

        if user_input.lower() == "/context":
            await handle_context_command(
                store, thread_id, user_id, all_tools, mcp_tool_count, messages
            )
            continue

        # Regular message (prompt_toolkit already echoes input, no need to repeat)
        try:
            response_text = ""
            status = console.status("[bold green]Thinking...", spinner="dots")
            status.start()

            async for event in stream_agent_turn(
                agent, store, user_input, thread_id, user_id
            ):
                if event.kind == "tool_start":
                    status.stop()
                    show_tool_start(event.tool_name, event.tool_input)
                    status = console.status(
                        f"[bold yellow]Running {event.tool_name}...", spinner="dots"
                    )
                    status.start()
                elif event.kind == "tool_end":
                    status.stop()
                    show_tool_end(event.tool_name, event.tool_output)
                    status = console.status("[bold green]Thinking...", spinner="dots")
                    status.start()
                elif event.kind == "response":
                    response_text = event.response

            status.stop()

            if response_text:
                show_assistant_message(response_text)
                from langchain_core.messages import AIMessage, HumanMessage

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
    """Main entry point."""
    # Validate configuration
    errors = validate_config()
    if errors:
        for err in errors:
            show_error(err)
        return

    show_welcome()

    # Load MCP server config
    mcp_config = load_mcp_servers()
    if not mcp_config:
        show_info("No MCP servers configured in mcp_servers.json. Running without MCP tools.")

    # Initialize database resources
    checkpointer_cm, store_cm = await create_agent_resources()

    async with checkpointer_cm as checkpointer, store_cm as store:
        # Run database migrations
        await checkpointer.setup()
        await store.setup()

        # Start MCP client
        mcp_client = MultiServerMCPClient(mcp_config) if mcp_config else None

        async def run_with_mcp(client):
            agent, all_tools, mcp_tool_count = await build_agent(
                client, checkpointer, store
            )

            # Thread management
            thread_id = str(uuid.uuid4())

            # Offer to resume existing threads
            threads = await get_thread_history(checkpointer)
            if threads:
                selected = prompt_thread_selection(threads)
                if selected:
                    thread_id = selected

            # Main loop with thread switching
            while True:
                result = await chat_loop(
                    agent, store, checkpointer, all_tools, mcp_tool_count, thread_id
                )

                if result is None:
                    break
                elif result == "NEW":
                    thread_id = str(uuid.uuid4())
                    show_info(f"New thread: {thread_id}")
                elif result.startswith("RESUME:"):
                    thread_id = result[7:]
                    show_info(f"Resumed thread: {thread_id}")

        if mcp_client:
            await run_with_mcp(mcp_client)
        else:
            # Create a dummy client that returns no tools
            class NoMCPClient:
                async def get_tools(self):
                    return []

            await run_with_mcp(NoMCPClient())

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
