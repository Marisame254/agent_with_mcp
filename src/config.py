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

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "WARNING")


def setup_logging() -> None:
    """Configure logging to stderr so it doesn't interfere with Rich UI."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

MODEL_NAME: str = os.environ.get("MODEL_NAME", "qwen3:14b")
MAX_CONTEXT_TOKENS: int = int(os.environ.get("MAX_CONTEXT_TOKENS", "9000"))

MCP_SERVERS_FILE: str = os.environ.get("MCP_SERVERS_FILE", "mcp_servers.json")


def load_mcp_servers() -> dict[str, dict]:
    """Load MCP server configuration from the JSON config file.

    Returns:
        Dictionary mapping server names to their connection parameters.
        Empty dict if the config file does not exist.
    """
    path = Path(MCP_SERVERS_FILE)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def validate_config() -> list[str]:
    """Check that required environment variables are set.

    Returns:
        List of human-readable error messages. Empty list means valid.
    """
    errors: list[str] = []
    if not DATABASE_URL:
        errors.append("DATABASE_URL is not set. Copy .env.example to .env and configure it.")
    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY is not set. Get one at https://tavily.com")
    return errors
