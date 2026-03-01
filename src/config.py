"""Application configuration loaded from environment variables."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "WARNING")


def setup_logging() -> None:
    """Configure logging to stderr so it doesn't interfere with Rich UI."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

MODEL_NAME: str = os.environ.get("MODEL_NAME", "kimi-k2.5:cloud")
MAX_CONTEXT_TOKENS: int = int(os.environ.get("MAX_CONTEXT_TOKENS", "90000"))

_PROJECT_ROOT = Path(__file__).parent.parent

MCP_SERVERS_FILE: str = os.environ.get("MCP_SERVERS_FILE", "")


def load_mcp_servers() -> dict[str, dict]:
    """Load MCP server configuration from the JSON config file.

    Resolves the config path relative to the project root (the directory
    containing this package) so that ``lyra`` works correctly regardless
    of the current working directory.

    Returns:
        Dictionary mapping server names to their connection parameters.
        Empty dict if the config file does not exist.
    """
    raw = MCP_SERVERS_FILE or "mcp_servers.json"
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def set_api_key(provider: str, key: str) -> None:
    """Set a provider API key at runtime.

    Updates both ``os.environ`` and the module-level variable so that any
    subsequent call to ``build_llm()`` (which re-imports from this module)
    picks up the new value without restarting the process.

    Args:
        provider: One of ``"openai"`` or ``"deepseek"``.
        key: The API key string.
    """
    import src.config as _self  # noqa: PLC0415

    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = key
        _self.OPENAI_API_KEY = key
    elif provider == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = key
        _self.DEEPSEEK_API_KEY = key


def validate_config() -> list[str]:
    """Check that required environment variables are set.

    Returns:
        List of human-readable error messages. Empty list means valid.
    """
    errors: list[str] = []
    if not DATABASE_URL:
        errors.append("DATABASE_URL is not set. Copy .env.example to .env and configure it.")
    # TAVILY_API_KEY is optional — web search will be disabled without it
    return errors
