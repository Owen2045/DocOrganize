from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "docorganize_laws")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
