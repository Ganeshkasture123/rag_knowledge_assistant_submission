import os
from dotenv import load_dotenv

load_dotenv()


def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


# Azure OpenAI
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = env("AZURE_OPENAI_API_KEY")

CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


# Azure AI Search
SEARCH_ENDPOINT = env("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = env("AZURE_SEARCH_API_KEY")

SEARCH_INDEX = os.getenv(
    "AZURE_SEARCH_INDEX",
    "enterprise-kb",
)

SEMANTIC_CONFIG = os.getenv(
    "AZURE_SEARCH_SEMANTIC_CONFIG",
    "enterprise-semantic",
)


# RAG
TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "50"))

MAX_SUBQUERIES = int(
    os.getenv("MAX_SUBQUERIES", "4")
)

PER_QUERY_RESULTS = int(
    os.getenv("PER_QUERY_RESULTS", "5")
)


# Chunking
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1200")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)


# Retrieval confidence
MIN_RERANKER_SCORE = float(
    os.getenv("MIN_RERANKER_SCORE", "1.5")
)