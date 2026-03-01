"""Entry point for Lyra — conversational AI agent."""

from __future__ import annotations

import asyncio
import logging
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from rich.live import Live
from rich.markdown import Markdown

from src.agent import build_agent, create_agent_resources, stream_agent_turn
from src.commands import ChatLoopResult, handle_context_command, handle_mcp_command, handle_memory_command, handle_model_command
from src.config import MAX_CONTEXT_TOKENS, MODEL_NAME, TAVILY_API_KEY, load_mcp_servers, set_api_key, setup_logging, validate_config
from src.constants import ASK_USER_TOOL_NAME, AgentEventKind, ChatCommand, ThreadAction
from src.providers import PROVIDER_DEEPSEEK, PROVIDER_OPENAI, ModelSpec, build_llm
from src.threads import delete_thread, generate_thread_name, get_thread_history, get_thread_messages, get_thread_name, save_thread_name
from src.tools import create_ask_user_tool
from src.ui import (
    ThreadManagementResult,
    console,
    create_prompt_session,
    prompt_reject_reason,
    prompt_thread_management,
    prompt_tool_decision,
    show_agent_question,
    show_conversation_history,
    show_error,
    show_info,
    show_tool_approval,
    show_tool_end,
    show_tool_start,
    show_welcome,
)

setup_logging()
logger = logging.getLogger(__name__)


async def _ask_user_prompt(question: str) -> str:
    """Prompt handler for the ask_user tool."""
    show_agent_question(question)
    loop = asyncio.get_running_loop()
    try:
        answer = await loop.run_in_executor(None, lambda: input("Tu respuesta> ").strip())
    except (EOFError, KeyboardInterrupt):
        return "(sin respuesta)"
    return answer or "(sin respuesta)"


async def _run_agent_turn(
    agent,
    store,
    user_input: str,
    thread_id: str,
    user_id: str,
    current_model: str,
) -> str:
    """Stream a single agent turn, rendering tokens and tool calls to the terminal.

    Handles HITL approval loops internally. Returns the final response text,
    or an empty string if the agent produced no response.
    """
    response_text = ""
    streaming_started = False
    streamed_text = ""
    live: Live | None = None
    resume_command: Command | None = None
    status = console.status("[dim]Thinking...[/]", spinner="dots")
    status.start()

    try:
        while True:
            needs_resume = False
            async for event in stream_agent_turn(
                agent,
                store,
                user_input,
                thread_id,
                user_id,
                model_name=current_model,
                resume_command=resume_command,
            ):
                if event.kind == AgentEventKind.TOOL_START:
                    if live:
                        live.stop()
                        live = None
                        streaming_started = False
                        streamed_text = ""
                    status.stop()
                    if event.tool_name == ASK_USER_TOOL_NAME:
                        continue
                    show_tool_start(event.tool_name, event.tool_input)
                    status.update(f"[dim]Running {event.tool_name}...[/]")
                    status.start()

                elif event.kind == AgentEventKind.TOOL_END:
                    status.stop()
                    if event.tool_name != ASK_USER_TOOL_NAME:
                        show_tool_end(event.tool_name, event.tool_output)
                    status.update("[dim]Thinking...[/]")
                    status.start()

                elif event.kind == AgentEventKind.TOKEN:
                    if not streaming_started:
                        status.stop()
                        console.print()
                        console.print("[bold green]Assistant:[/]")
                        live = Live(Markdown(""), console=console, refresh_per_second=8)
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

                    decisions: list[dict] = []
                    for ar in action_requests:
                        decision_type = await prompt_tool_decision()
                        if decision_type == "approve":
                            decisions.append({"type": "approve"})
                        else:
                            reason = await prompt_reject_reason()
                            msg = reason or (
                                f"Usuario rechazó la ejecución de `{ar.get('name', 'unknown')}`"
                            )
                            decisions.append({"type": "reject", "message": msg})

                    resume_command = Command(resume={"decisions": decisions})
                    status = console.status("[bold green]Thinking...", spinner="dots")
                    status.start()
                    streaming_started = False
                    streamed_text = ""
                    needs_resume = True
                    break

            if not needs_resume:
                break
    finally:
        status.stop()
        if live:
            live.stop()

    return response_text


async def _ensure_provider_key(spec: ModelSpec) -> bool:
    """Prompt the user for an API key if the provider requires one and it's missing.

    Updates the config at runtime so the key is available for the current session.

    Args:
        spec: The parsed model spec with provider and name.

    Returns:
        True if we can proceed (key already set or user provided one),
        False if the user cancelled.
    """
    from src.config import DEEPSEEK_API_KEY, OPENAI_API_KEY  # re-import for current values

    required: dict[str, tuple[str, str]] = {
        PROVIDER_OPENAI: ("OPENAI_API_KEY", OPENAI_API_KEY),
        PROVIDER_DEEPSEEK: ("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
    }

    if spec.provider not in required:
        return True

    env_var, current_key = required[spec.provider]
    if current_key:
        return True

    show_info(f"El modelo [bold]{spec}[/] requiere [bold]{env_var}[/] para funcionar.")
    loop = asyncio.get_running_loop()
    try:
        key = await loop.run_in_executor(
            None, lambda: input(f"Introduce tu {env_var} (Enter para cancelar): ").strip()
        )
    except (EOFError, KeyboardInterrupt):
        return False

    if not key:
        show_error("API key vacía. Operación cancelada.")
        return False

    set_api_key(spec.provider, key)
    show_info(f"{env_var} configurada para esta sesión.")
    return True


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
    mcp_config: dict | None = None,
    disabled_servers: frozenset[str] = frozenset(),
) -> ChatLoopResult | None:
    """Run the interactive chat loop for a single thread.

    Returns a ChatLoopResult describing what to do next, or None to exit.
    """
    session = create_prompt_session()
    messages: list[HumanMessage | AIMessage] = []
    is_new_thread = not resumed

    show_info(f"Thread: {thread_id}")

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
            user_input = await asyncio.get_running_loop().run_in_executor(
                None, lambda: session.prompt("You >> ")
            )
        except (EOFError, KeyboardInterrupt):
            return None

        user_input = user_input.strip()
        if not user_input:
            continue

        # --- Lifecycle commands ---
        if user_input.lower() == "/exit":
            return None

        if user_input.lower() == "/new":
            return ChatLoopResult(command=ChatCommand.NEW)

        if user_input.lower() == "/threads":
            managing = True
            while managing:
                threads = await get_thread_history(checkpointer, store)
                mgmt = await prompt_thread_management(threads)
                if mgmt.action == ThreadAction.RESUME:
                    return ChatLoopResult(command=ChatCommand.NEW, thread_id=mgmt.thread_id)
                elif mgmt.action == ThreadAction.DELETE:
                    await delete_thread(checkpointer, store, mgmt.thread_id)
                    show_info("Conversación eliminada.")
                elif mgmt.action == ThreadAction.RENAME:
                    await save_thread_name(store, mgmt.thread_id, mgmt.new_name)
                    show_info(f"Conversación renombrada: {mgmt.new_name}")
                else:  # NEW — stay in current thread
                    managing = False
            continue

        if user_input.lower() == "/help":
            from src.ui import show_help
            show_help()
            continue

        # --- Delegated command handlers ---
        if user_input.lower() == "/context":
            await handle_context_command(
                agent, store, thread_id, user_id, all_tools, mcp_tool_count,
                MAX_CONTEXT_TOKENS,
            )
            continue

        if user_input.lower().startswith("/memory"):
            parts = user_input.split(maxsplit=2)
            await handle_memory_command(parts, store, user_id)
            continue

        if user_input.lower().startswith("/mcp"):
            parts = user_input.split(maxsplit=2)
            result = await handle_mcp_command(parts, mcp_config or {}, disabled_servers)
            if result:
                return result
            continue

        if user_input.lower().startswith("/model"):
            parts = user_input.split(maxsplit=1)
            result = await handle_model_command(parts, current_model)
            if result:
                return result
            continue

        # --- Regular message: run agent turn ---
        try:
            response_text = await _run_agent_turn(
                agent, store, user_input, thread_id, user_id, current_model
            )
        except Exception as e:
            show_error(f"Agent error: {e}")
            continue

        if response_text:
            console.print()
            messages.append(HumanMessage(content=user_input))
            messages.append(AIMessage(content=response_text))

            if is_new_thread:
                is_new_thread = False
                try:
                    llm = build_llm(ModelSpec.parse(current_model))
                    name = await generate_thread_name(llm, user_input)
                    await save_thread_name(store, thread_id, name)
                    show_info(f"Thread: {name}")
                except Exception:
                    logger.debug("Failed to save thread name", exc_info=True)
        else:
            show_error("No response from agent.")


async def main() -> None:
    """Validate config, connect to services, and run the chat loop."""
    errors = validate_config()
    if errors:
        for err in errors:
            show_error(err)
        return

    show_welcome()

    if not TAVILY_API_KEY:
        show_info("TAVILY_API_KEY no configurada — búsqueda web deshabilitada.")

    mcp_config_full = load_mcp_servers()
    disabled_servers: frozenset[str] = frozenset()
    if not mcp_config_full:
        show_info("No MCP servers configured in mcp_servers.json. Running without MCP tools.")

    def _active_mcp_config() -> dict:
        return {k: v for k, v in mcp_config_full.items() if k not in disabled_servers}

    def _make_mcp_client(active: dict) -> MultiServerMCPClient | None:
        if not active:
            return None
        try:
            client = MultiServerMCPClient(active)
            show_info(f"MCP servers: {', '.join(active.keys())}")
            return client
        except Exception as e:
            logger.warning("Failed to initialize MCP client: %s", e)
            show_error(f"MCP initialization failed: {e}. Continuing without MCP tools.")
            return None

    checkpointer_cm, store_cm = await create_agent_resources()

    async with checkpointer_cm as checkpointer, store_cm as store:
        await checkpointer.setup()
        await store.setup()

        mcp_client = _make_mcp_client(_active_mcp_config())
        ask_user_tool = create_ask_user_tool(_ask_user_prompt)
        current_model = MODEL_NAME

        agent, all_tools, mcp_tool_count, mcp_tool_names = await build_agent(
            mcp_client, checkpointer, store,
            ask_user_tool=ask_user_tool,
            model_name=current_model,
        )

        thread_id = str(uuid.uuid4())
        is_resumed = False

        threads = await get_thread_history(checkpointer, store)
        if threads:
            managing = True
            while managing:
                threads = await get_thread_history(checkpointer, store)
                mgmt = await prompt_thread_management(threads)
                if mgmt.action == ThreadAction.RESUME:
                    thread_id = mgmt.thread_id
                    is_resumed = True
                    managing = False
                elif mgmt.action == ThreadAction.DELETE:
                    await delete_thread(checkpointer, store, mgmt.thread_id)
                    show_info("Conversación eliminada.")
                elif mgmt.action == ThreadAction.RENAME:
                    await save_thread_name(store, mgmt.thread_id, mgmt.new_name)
                    show_info(f"Conversación renombrada: {mgmt.new_name}")
                else:  # NEW
                    managing = False

        while True:
            result = await chat_loop(
                agent, store, checkpointer, all_tools, mcp_tool_count,
                thread_id, resumed=is_resumed, current_model=current_model,
                mcp_config=mcp_config_full, disabled_servers=disabled_servers,
            )

            if result is None:
                break

            elif result.command == ChatCommand.MCP_RELOAD:
                disabled_servers = result.mcp_disabled
                mcp_config_full = load_mcp_servers()
                active = _active_mcp_config()
                mcp_client = _make_mcp_client(active)
                show_info("Reconstruyendo agente con nueva configuración MCP...")
                agent, all_tools, mcp_tool_count, _ = await build_agent(
                    mcp_client, checkpointer, store,
                    ask_user_tool=ask_user_tool,
                    model_name=current_model,
                )
                show_info(f"MCP: {len(active)} servidor(es) activo(s)")
                is_resumed = True

            elif result.command == ChatCommand.MODEL:
                spec = ModelSpec.parse(result.model_name)
                if not await _ensure_provider_key(spec):
                    is_resumed = True
                    continue
                current_model = result.model_name
                show_info(f"Cambiando modelo a: {current_model}...")
                agent, all_tools, mcp_tool_count, _ = await build_agent(
                    mcp_client, checkpointer, store,
                    ask_user_tool=ask_user_tool,
                    model_name=current_model,
                )
                show_info(f"Modelo activo: {current_model}")
                is_resumed = True

            elif result.command == ChatCommand.NEW:
                if result.thread_id:
                    thread_id = result.thread_id
                    is_resumed = True
                    name = await get_thread_name(store, thread_id)
                    show_info(f"Resumed thread: {name or thread_id[:8]}")
                else:
                    thread_id = str(uuid.uuid4())
                    is_resumed = False
                    show_info(f"New thread: {thread_id}")

    console.print("[dim]Goodbye![/]")


def run() -> None:
    """Synchronous entry point for the console script."""
    import sys

    if sys.platform == "win32":
        import selectors
        asyncio.run(
            main(),
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
        )
    else:
        asyncio.run(main())


if __name__ == "__main__":
    run()
