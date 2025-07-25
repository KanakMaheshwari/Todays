
import os

# Base directory for generator stores
GENERATOR_STORES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "generator_stores"))

# File paths
VECTOR_STORE_PATH = os.path.join(GENERATOR_STORES_DIR, "faiss.index")
PROCESSED_ARTICLES_PATH = os.path.join(GENERATOR_STORES_DIR, "processed_articles.json")
SEEN_URLS_PATH = os.path.join(GENERATOR_STORES_DIR, "seen_articles.json")

# Directory paths
CHUNK_INDEX_DIR = os.path.join(GENERATOR_STORES_DIR, "chunk_indices")
CACHE_DIR = os.path.join(GENERATOR_STORES_DIR, "query_cache")

# Other paths
RESULTS_PATH = os.path.abspath("results.jsonl")
GROUPED_PROCESSED_ARTICLES_PATH = os.path.abspath("grouped_output.json")
