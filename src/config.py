import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")

MODEL_NAME: str = os.environ.get("MODEL_NAME", "qwen3:14b")
MAX_CONTEXT_TOKENS: int = int(os.environ.get("MAX_CONTEXT_TOKENS", "20000"))

MCP_SERVERS_FILE: str = os.environ.get("MCP_SERVERS_FILE", "mcp_servers.json")


def load_mcp_servers() -> dict:
    path = Path(MCP_SERVERS_FILE)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def validate_config() -> list[str]:
    errors = []
    if not DATABASE_URL:
        errors.append("DATABASE_URL is not set. Copy .env.example to .env and configure it.")
    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY is not set. Get one at https://tavily.com")
    return errors
