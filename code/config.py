"""Configuration module for the Sanskrit RAG system.

This module exposes all hyperparameters, model paths, directory paths,
and settings used across the ingestion, retrieval, generation,
and evaluation stages.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# # Final chunking configuration selected after ablation testing.
# 4-verse chunks with 1-verse overlap gave the best overall ranking quality
# across Precision@5, MRR, NDCG@5, and MAP.

CHUNK_SIZE: int = 4          # Chunk size in terms of number of verses
CHUNK_OVERLAP: int = 1       # Overlap in terms of number of verses

# Model Paths & Names
MODEL_PATH: str = str(MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
EMBED_MODEL_NAME: str = "intfloat/multilingual-e5-small"
RERANK_MODEL_NAME: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Vector Store and BM25 Index paths
FAISS_INDEX_PATH: str = str(BASE_DIR / "index" / "faiss_index")
BM25_INDEX_PATH: str = str(BASE_DIR / "index" / "bm25_index.pkl")
CHUNKS_METADATA_PATH: str = str(BASE_DIR / "index" / "chunks_metadata.pkl")

# Retrieval & Re-ranking Settings
TOP_K_SEMANTIC: int = 30     # Number of candidate documents retrieved via FAISS
TOP_K_BM25: int = 30         # Number of candidate documents retrieved via BM25
RRF_K: int = 60              # Constant for Reciprocal Rank Fusion
RERANK_ALPHA: float = 0.7     # Weight for semantic score (1 - RERANK_ALPHA for BM25)
TOP_K_RERANK: int = 10       # Number of documents returned after RRF fusion
TOP_K_GENERATION: int = 5    # Number of documents passed to the Generator

# MMR and thresholding are implemented for experimentation,
# but disabled by default because the ablation study showed
# that the baseline 4-verse configuration performed better overall.

USE_MMR: bool = False        # Use Maximal Marginal Relevance for diversity
MMR_LAMBDA: float = 0.7      # Lambda parameter for MMR (relevance vs diversity)
RERANK_THRESHOLD: float = -999.0 # Minimum CrossEncoder score to return a chunk



# Generation settings
MAX_TOKENS: int = 512
TEMPERATURE: float = 0.2
TOP_P: float = 0.9
N_THREADS: int = os.cpu_count() or 4

# Evaluation Settings
EVAL_RESULTS_PATH: str = str(REPORT_DIR / "eval_results.csv")
TEST_SET_PATH: str = str(DATA_DIR / "test_queries.json")
