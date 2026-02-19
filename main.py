"""Entry point for the Agent with MCP."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from rich.live import Live
from rich.markdown import Markdown

from langchain_ollama import ChatOllama

from src.agent import build_agent, create_agent_resources, generate_thread_name, get_system_prompt, get_thread_history, get_thread_messages, get_thread_name, save_thread_name, stream_agent_turn
from src.config import MAX_CONTEXT_TOKENS, MODEL_NAME, load_mcp_servers, setup_logging, validate_config
from src.constants import ASK_USER_TOOL_NAME, AgentEventKind, ChatCommand
from src.context_tracker import build_context_breakdown
from src.memory import clear_memories, delete_memory, list_memories, retrieve_memories, store_memories
from src.tools import create_ask_user_tool
from src.ui import (
    console,
    create_prompt_session,
    prompt_reject_reason,
    prompt_thread_selection,
    prompt_tool_decision,
    show_agent_question,
    show_assistant_message,
    show_context_breakdown,
    show_conversation_history,
    show_error,
    show_info,
    show_models_table,
    show_tool_approval,
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
        command: The action to take (NEW thread, EXIT, or MODEL change).
        thread_id: Target thread ID when resuming a conversation.
        model_name: New model name when command is MODEL.
    """

    command: ChatCommand
    thread_id: str = ""
    model_name: str = ""


async def handle_context_command(
    agent,
    store,
    thread_id: str,
    user_id: str,
    all_tools: list,
    mcp_tool_count: int,
) -> None:
    """Handle the /context command by displaying token usage breakdown.

    Reads real messages from the agent checkpoint (including ToolMessages
    and summarization SystemMessages) instead of relying on a local list.
    """
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
        max_tokens=MAX_CONTEXT_TOKENS,
    )
    show_context_breakdown(breakdown)


async def _ask_user_prompt(question: str) -> str:
    """Prompt handler for the ask_user tool.

    Stops any active spinner, shows the question, and captures the user's
    response via ``run_in_executor`` so the event loop stays unblocked.
    """
    show_agent_question(question)
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, lambda: input("Tu respuesta> ").strip())
    return answer or "(sin respuesta)"


async def handle_model_list_command(current_model: str) -> None:
    """Display available Ollama models and highlight the active one."""
    try:
        from ollama import AsyncClient
        client = AsyncClient()
        response = await client.list()
        models = [m.model for m in response.models]
        show_models_table(models, current_model)
    except Exception as e:
        show_error(f"No se pudo conectar con Ollama: {e}")


async def chat_loop(
    agent,
    store,
    checkpointer,
    all_tools: list,
    mcp_tool_count: int,
    thread_id: str,
    user_id: str = "default",
    resumed: bool = False,
    current_model: str = MODEL_NAME,
) -> ChatLoopResult | None:
    """Run the main chat loop.

    Returns:
        A ChatLoopResult describing what to do next, or None to exit.
    """
    session = create_prompt_session()
    messages: list[HumanMessage | AIMessage] = []
    is_new_thread = not resumed

    show_info(f"Thread: {thread_id}")

    # Load and display previous messages when resuming a thread
    if resumed:
        prev_messages = await get_thread_messages(checkpointer, thread_id)
        if prev_messages:
            show_conversation_history(prev_messages)
            messages.extend(prev_messages)
        else:
            console.print()
    else:
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
            threads = await get_thread_history(checkpointer, store)
            selected = prompt_thread_selection(threads)
            if selected:
                return ChatLoopResult(command=ChatCommand.NEW, thread_id=selected)
            continue

        if user_input.lower() == "/help":
            show_help()
            continue

        if user_input.lower() == "/context":
            await handle_context_command(
                agent, store, thread_id, user_id, all_tools, mcp_tool_count
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

        if user_input.lower().startswith("/model"):
            parts = user_input.split(maxsplit=1)
            new_model = parts[1].strip() if len(parts) > 1 else ""
            if new_model:
                return ChatLoopResult(command=ChatCommand.MODEL, model_name=new_model)
            else:
                await handle_model_list_command(current_model)
            continue

        # Regular message
        try:
            response_text = ""
            streaming_started = False
            streamed_text = ""
            live: Live | None = None
            resume_command: Command | None = None
            status = console.status("[bold green]Thinking...", spinner="dots")
            status.start()

            # Inner loop: process events, handle HITL approvals, resume
            while True:
                needs_resume = False
                async for event in stream_agent_turn(
                    agent, store, user_input, thread_id, user_id,
                    resume_command=resume_command,
                ):
                    if event.kind == AgentEventKind.TOOL_START:
                        if live:
                            live.stop()
                            live = None
                            streaming_started = False
                            streamed_text = ""
                        status.stop()
                        # ask_user handles its own UI — no spinner
                        if event.tool_name == ASK_USER_TOOL_NAME:
                            continue
                        show_tool_start(event.tool_name, event.tool_input)
                        status = console.status(
                            f"[bold yellow]Running {event.tool_name}...",
                            spinner="dots",
                        )
                        status.start()
                    elif event.kind == AgentEventKind.TOOL_END:
                        status.stop()
                        if event.tool_name != ASK_USER_TOOL_NAME:
                            show_tool_end(event.tool_name, event.tool_output)
                        status = console.status(
                            "[bold green]Thinking...", spinner="dots"
                        )
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
                    elif event.kind == AgentEventKind.TOOL_APPROVAL_REQUIRED:
                        status.stop()
                        if live:
                            live.stop()
                            live = None
                        action_requests = event.action_requests or []
                        show_tool_approval(action_requests)

                        # Build decisions for each action request
                        decisions: list[dict] = []
                        for ar in action_requests:
                            decision_type = prompt_tool_decision()
                            if decision_type == "approve":
                                decisions.append({"type": "approve"})
                            else:
                                reason = prompt_reject_reason()
                                msg = reason or (
                                    f"Usuario rechazó la ejecución de "
                                    f"`{ar.get('name', 'unknown')}`"
                                )
                                decisions.append({
                                    "type": "reject",
                                    "message": msg,
                                })

                        resume_command = Command(
                            resume={"decisions": decisions}
                        )
                        # Restart spinner before resuming
                        status = console.status(
                            "[bold green]Thinking...", spinner="dots"
                        )
                        status.start()
                        streaming_started = False
                        streamed_text = ""
                        needs_resume = True
                        break  # exit async for; while True will restart

                if not needs_resume:
                    break  # stream finished without needing resume

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

                # Generate a readable name for new threads after the first response
                if is_new_thread:
                    is_new_thread = False
                    try:
                        llm = ChatOllama(model=current_model)
                        name = await generate_thread_name(llm, user_input)
                        await save_thread_name(store, thread_id, name)
                        show_info(f"Thread: {name}")
                    except Exception:
                        logger.debug("Failed to save thread name", exc_info=True)
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

        ask_user_tool = create_ask_user_tool(_ask_user_prompt)
        current_model = MODEL_NAME
        agent, all_tools, mcp_tool_count, mcp_tool_names = await build_agent(
            mcp_client, checkpointer, store, ask_user_tool=ask_user_tool,
            model_name=current_model,
        )

        thread_id = str(uuid.uuid4())
        is_resumed = False

        threads = await get_thread_history(checkpointer, store)
        if threads:
            selected = prompt_thread_selection(threads)
            if selected:
                thread_id = selected
                is_resumed = True

        while True:
            result = await chat_loop(
                agent, store, checkpointer, all_tools, mcp_tool_count,
                thread_id, resumed=is_resumed, current_model=current_model,
            )

            if result is None:
                break
            elif result.command == ChatCommand.MODEL:
                current_model = result.model_name
                show_info(f"Cambiando modelo a: {current_model}...")
                agent, all_tools, mcp_tool_count, _ = await build_agent(
                    mcp_client, checkpointer, store, ask_user_tool=ask_user_tool,
                    model_name=current_model,
                )
                show_info(f"Modelo activo: {current_model}")
                is_resumed = True
            elif result.thread_id:
                thread_id = result.thread_id
                is_resumed = True
                name = await get_thread_name(store, thread_id)
                label = name or thread_id[:8]
                show_info(f"Resumed thread: {label}")
            elif result.command == ChatCommand.NEW:
                thread_id = str(uuid.uuid4())
                is_resumed = False
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
