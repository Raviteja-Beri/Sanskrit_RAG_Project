"""
Retrieval strategy:

1. Normalize Sanskrit queries into Devanagari.
2. Retrieve candidates using both FAISS semantic search and BM25 keyword search.
3. Fuse rankings using Reciprocal Rank Fusion.
4. Re-rank fused candidates with a CrossEncoder.
5. Optionally apply thresholding, overlap deduplication, and MMR.

This file intentionally avoids high-level RAG frameworks so that retrieval
behavior can be inspected, evaluated, and tuned directly.
"""

import os
import re
import pickle
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from indic_transliteration import sanscript

# Import configurations
from config import (
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    FAISS_INDEX_PATH,
    BM25_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    TOP_K_SEMANTIC,
    TOP_K_BM25,
    RRF_K,
    RERANK_ALPHA,
    TOP_K_RERANK,
    TOP_K_GENERATION,
    USE_MMR,
    MMR_LAMBDA,
    RERANK_THRESHOLD,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retriever")

# Global variables for caching loaded models and indexes
_embed_model = None
_rerank_model = None
_faiss_index = None
_bm25_index = None
_chunks = None
_metadata = None


def init_retriever() -> None:
    """Initializes and loads the embedding model and indexes.

    This function is cached and loads the models once at the module level.
    """
    global _embed_model, _rerank_model, _faiss_index, _bm25_index, _chunks, _metadata
    if _embed_model is not None:
        return

    logger.info("Initializing retriever components...")

    # Load embedding model once
    logger.info(f"Loading sentence-transformer model: {EMBED_MODEL_NAME}")
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # Load CrossEncoder model
    logger.info(f"Loading CrossEncoder model: {RERANK_MODEL_NAME}")
    _rerank_model = CrossEncoder(RERANK_MODEL_NAME)

    # Load FAISS index
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index file not found at {FAISS_INDEX_PATH}. Please run ingest.py first."
        )
    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    # Load BM25 index
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError(
            f"BM25 index file not found at {BM25_INDEX_PATH}. Please run ingest.py first."
        )
    logger.info(f"Loading BM25 index from {BM25_INDEX_PATH}")
    with open(BM25_INDEX_PATH, "rb") as f:
        _bm25_index = pickle.load(f)

    # Load Chunks and Metadata
    if not os.path.exists(CHUNKS_METADATA_PATH):
        raise FileNotFoundError(
            f"Chunks metadata not found at {CHUNKS_METADATA_PATH}. Please run ingest.py first."
        )
    logger.info(f"Loading chunks and metadata from {CHUNKS_METADATA_PATH}")
    with open(CHUNKS_METADATA_PATH, "rb") as f:
        data = pickle.load(f)
        _chunks = data["chunks"]
        _metadata = data["metadata"]

    logger.info("Retriever initialized successfully.")


def preprocess_roman_query(query: str) -> str:
    """Preprocesses Romanized Sanskrit queries to handle informal spellings and visargas."""
    # 1. Replace word-final 'h' following a vowel with 'H' (visarga)
    query = re.sub(r'(?<=[aeiouAEIOU])h(?=\b|[^a-zA-Z]|$)', 'H', query)
    
    # 2. Map 'w' to 'v' (informal spelling)
    query = re.sub(r'w', 'v', query)
    query = re.sub(r'W', 'V', query)
    
    # 3. Map informal/ITRANS spellings to HK:
    query = re.sub(r'shh|Sh', 'S', query)
    query = re.sub(r'sh', 'z', query)
    query = re.sub(r'chh', 'ch', query)
    query = re.sub(r'ch', 'c', query)
    
    return query


# Sanskrit queries may arrive in Devanagari, IAST, SLP1, or Harvard-Kyoto.
# Normalizing them into Devanagari avoids cross-script retrieval mismatches.

def detect_scheme(query: str) -> str:
    """Heuristically detects the transliteration scheme of the Roman Sanskrit query."""
    if any('\u0900' <= char <= '\u097F' for char in query):
        return "DEVANAGARI"

    iast_chars = "āīūṛṝḷḹṃḥṅñṭḍṇśṣ"
    iast_score = sum(1 for c in query.lower() if c in iast_chars)

    slp1_score = 0
    hk_score = 0

    slp1_uniques = "wWqQfFxXY"
    slp1_score += sum(2 for c in query if c in slp1_uniques)

    slp1_aspirates = "KGCJTPB"
    slp1_score += sum(1 for c in query if c in slp1_aspirates)

    # HK digraphs: e.g. kh, gh, ch, jh, th, dh, ph, bh, sh
    digraphs = re.findall(r'(kh|gh|ch|jh|th|dh|ph|bh|sh)', query.lower())
    hk_score += len(digraphs) * 3

    if any(c in "zS" for c in query):
        hk_score += 1

    logger.debug(f"Detected scheme scores: IAST={iast_score}, SLP1={slp1_score}, HK={hk_score}")

    if iast_score > 0 and iast_score >= slp1_score and iast_score >= hk_score:
        return "IAST"
    elif slp1_score > hk_score:
        return "SLP1"
    else:
        return "HK"

# Sanskrit queries may arrive in Devanagari, IAST, SLP1, or Harvard-Kyoto.
# Normalizing them into Devanagari avoids cross-script retrieval mismatches.

def transliterate_query_to_devanagari(query: str) -> str:
    """Detects transliteration scheme and converts the Sanskrit query to Devanagari.

    Supported input schemes: Devanagari, IAST, SLP1, Harvard-Kyoto (HK).

    Args:
        query: The raw query string in Devanagari, IAST, SLP1, or HK.

    Returns:
        The transliterated query in Devanagari normalized to NFC.
    """
    query_clean = query.strip()
    if not query_clean:
        return ""

    scheme = detect_scheme(query_clean)
    logger.info(f"Detected script scheme: {scheme}")

    if scheme == "DEVANAGARI":
        result = query_clean
    else:
        preprocessed = preprocess_roman_query(query_clean)
        logger.info(f"Preprocessed Roman query: {preprocessed}")
        if scheme == "IAST":
            result = sanscript.transliterate(preprocessed, sanscript.IAST, sanscript.DEVANAGARI)
        elif scheme == "SLP1":
            result = sanscript.transliterate(preprocessed, sanscript.SLP1, sanscript.DEVANAGARI)
        else:
            result = sanscript.transliterate(preprocessed, sanscript.HK, sanscript.DEVANAGARI)

    # Unicode normalization (NFC)
    return unicodedata.normalize('NFC', result)

# FAISS and BM25 scores are not directly comparable.
# RRF combines ranks instead of raw scores, making fusion stable and simple.

def reciprocal_rank_fusion(
    faiss_results: List[int],
    bm25_results: List[int],
    k: int = 60
) -> List[Tuple[int, float]]:
    """Merges two ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        faiss_results: Ranked list of document/chunk indices from FAISS.
        bm25_results: Ranked list of document/chunk indices from BM25.
        k: The constant parameter for RRF (default 60).

    Returns:
        A list of tuples (chunk_index, rrf_score) sorted in descending order of RRF score.
    """
    rrf_scores = {}

    # Process FAISS results
    for rank, chunk_idx in enumerate(faiss_results, start=1):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    # Process BM25 results
    for rank, chunk_idx in enumerate(bm25_results, start=1):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score descending
    sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


def parse_range(r_str: str) -> set:
    """Helper to parse a range string like 'start-end' to a set of integers."""
    try:
        start, end = map(int, r_str.split('-'))
        return set(range(start, end + 1))
    except Exception:
        return set()


def compute_overlap(range1: str, range2: str) -> float:
    """Computes the overlap ratio relative to the smaller chunk size."""
    set1 = parse_range(range1)
    set2 = parse_range(range2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    return intersection / min(len(set1), len(set2))


def deduplicate_chunks(chunks: List[Dict[str, Any]], overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Deduplicates a list of chunks based on verse overlap."""
    deduped = []
    for chunk in chunks:
        keep = True
        for existing in deduped:
            overlap = compute_overlap(chunk["metadata"]["verse_id"], existing["metadata"]["verse_id"])
            if overlap > overlap_threshold:
                keep = False
                break
        if keep:
            deduped.append(chunk)
    return deduped


def maximal_marginal_relevance(
    candidates: List[Dict[str, Any]],
    query_emb: np.ndarray,
    lambda_param: float = 0.7,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """Selects top_n chunks using Maximal Marginal Relevance (MMR).

    Args:
        candidates: List of retrieved chunk dictionaries.
        query_emb: The query embedding vector.
        lambda_param: Weight factor for trade-off between relevance and diversity (0.0 to 1.0).
        top_n: Number of chunks to select.

    Returns:
        List of selected chunk dictionaries.
    """
    if not candidates:
        return []
    if len(candidates) <= 1:
        return candidates[:top_n]

    # Reconstruct embeddings for all candidates
    def get_embedding(c):
        idx = int(c["metadata"]["chunk_index"])
        return _faiss_index.reconstruct(idx)

    # Sigmoid normalization of CrossEncoder score
    def normalize_score(score):
        return 1.0 / (1.0 + np.exp(-score))

    selected = [candidates[0]]
    remaining = candidates[1:]

    while len(selected) < top_n and remaining:
        best_mmr = -float('inf')
        best_candidate = None

        # Pre-calculate embeddings of selected chunks
        selected_embs = [get_embedding(s) for s in selected]

        for cand in remaining:
            rel = normalize_score(cand["rerank_score"])
            cand_emb = get_embedding(cand)
            
            # Compute max similarity to any selected chunk
            max_sim = max(float(np.dot(cand_emb, sel_emb)) for sel_emb in selected_embs)
            
            # MMR score calculation
            mmr_score = lambda_param * rel - (1.0 - lambda_param) * max_sim
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_candidate = cand

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def retrieve_relevant_chunks(query: str) -> List[Dict[str, Any]]:
    """Performs hybrid retrieval, fusion, and CrossEncoder re-ranking to return top chunks.

    Args:
        query: The user query in any supported script.

    Returns:
        A list of dicts, each representing a chunk.
    """
    init_retriever()

    # Sanskrit queries may arrive in Devanagari, IAST, SLP1, or Harvard-Kyoto.
    # Normalizing them into Devanagari avoids cross-script retrieval mismatches.
    
    # Step 1: Query normalization / transliteration
    query_dev = transliterate_query_to_devanagari(query)
    logger.info(f"Normalized Devanagari Query: {query_dev}")

    # Step 2: Semantic search via FAISS
    # E5 models need "query: " prefix
    prefixed_query = "query: " + query_dev
    query_emb = _embed_model.encode([prefixed_query], normalize_embeddings=True)[0]

    # Query FAISS index
    D, I = _faiss_index.search(np.array([query_emb]).astype("float32"), TOP_K_SEMANTIC)
    faiss_results = [idx for idx in I[0].tolist() if idx != -1]

    # Step 3: Keyword search via BM25
    tokenized_query = query_dev.split()
    bm25_scores = _bm25_index.get_scores(tokenized_query)

    # Get top BM25 results
    bm25_results = np.argsort(bm25_scores)[::-1][:TOP_K_BM25].tolist()

    # Step 4: Hybrid Fusion via Reciprocal Rank Fusion (RRF)
    fused_scores = reciprocal_rank_fusion(faiss_results, bm25_results, k=RRF_K)

    # Top fused chunks as context candidates for rerank
    top_fused = fused_scores[:TOP_K_RERANK]

    # Step 5: Post-RRF Re-ranking using CrossEncoder
    reranked_chunks = []
    
    if top_fused:
        # Prepare pairs for cross-encoder prediction
        pairs = [(query_dev, _chunks[chunk_idx]) for chunk_idx, _ in top_fused]
        cross_scores = _rerank_model.predict(pairs)

        for i, (chunk_idx, rrf_score) in enumerate(top_fused):
            cross_score = float(cross_scores[i])
            
            # Calculate auxiliary similarity features for debugging/telemetry
            chunk_emb = _faiss_index.reconstruct(int(chunk_idx))
            cosine_sim = float(np.dot(query_emb, chunk_emb))

            reranked_chunks.append({
                "text": _chunks[chunk_idx],
                "metadata": _metadata[chunk_idx],
                "rerank_score": cross_score,  # CrossEncoder score is the primary ranking key
                "rrf_score": rrf_score,
                "cosine_sim": cosine_sim,
            })

        # Sort top fused chunks by rerank_score descending
        reranked_chunks = sorted(reranked_chunks, key=lambda x: x["rerank_score"], reverse=True)

    # Step 6: Apply Relevance Threshold Filtering
    filtered_chunks = [c for c in reranked_chunks if c["rerank_score"] >= RERANK_THRESHOLD]
    logger.info(f"Threshold filtered candidates from {len(reranked_chunks)} to {len(filtered_chunks)}")

    # Step 7: Apply Overlap Deduplication (removes redundant/highly overlapping chunks)
    deduped_chunks = deduplicate_chunks(filtered_chunks, overlap_threshold=0.5)
    logger.info(f"Deduplicated candidates from {len(filtered_chunks)} to {len(deduped_chunks)}")

    # Step 8: Apply Maximal Marginal Relevance (MMR) or return top TOP_K_GENERATION
    if USE_MMR:
        final_chunks = maximal_marginal_relevance(
            deduped_chunks,
            query_emb,
            lambda_param=MMR_LAMBDA,
            top_n=TOP_K_GENERATION
        )
        logger.info(f"Applied MMR selecting top {len(final_chunks)} chunks.")
    else:
        final_chunks = deduped_chunks[:TOP_K_GENERATION]
        logger.info(f"Selected top {len(final_chunks)} chunks without MMR.")

    return final_chunks
