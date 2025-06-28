import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone


# 1) Project root (one level above this file’s folder)
BASE_DIR = Path(__file__).parent.parent

# make src/ importable
sys.path.insert(0, str(Path(__file__).parent))

# load .env
load_dotenv()

# service keys
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")

# Pinecone client instance
PINECONE_CLIENT = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# dropdown indices
INDEX_LIST = [
    "hw5",
    "another-index",
    "some-other-index"
]

# flask config
FLASK_HOST  = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT  = int(os.getenv("FLASK_PORT", "3001"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() in ("1","true","yes")

EMBED_MODEL = "text-embedding-3-large"