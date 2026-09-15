"""
Single place for environment configuration. Import this module first: it loads
.env before anything reads os.environ (the old code read DB settings in
routers.py *before* load_dotenv() ran, so it silently used defaults).
"""
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "pia")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "pia")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
# Must match the model the analyst used to write intelligence_records.embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

PIA_API_TOKEN = os.getenv("PIA_API_TOKEN", "")
FRONTEND_ORIGINS = [o.strip() for o in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

_here = os.path.dirname(os.path.abspath(__file__))
# Same folder the document_agent watches (mounted into both containers in compose)
DOC_DIR = os.path.abspath(os.getenv("DOC_DIR", os.path.join(_here, "..", "pia", "data", "documents")))
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024)

# The OpenAI client refuses api_key=None at construction; a placeholder keeps the
# service up (chat/search answer 502) instead of crashing every process at import.
llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY or "missing-openrouter-key",
    default_headers={"HTTP-Referer": "https://github.com/sebastian420-hub/pia", "X-Title": "PIA API Bridge"},
)
