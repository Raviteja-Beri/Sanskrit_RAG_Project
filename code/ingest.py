"""Ingestion and preprocessing module for the Sanskrit RAG system.

This module loads Sanskrit documents (.txt and .pdf) from a data directory,
normalizes and cleans the text, splits them into verses, chunks them with
specified overlap, and builds both a FAISS semantic index and a BM25 index.
"""

import os
import re
import sys
import pickle
import logging
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Import configurations
from config import (
    DATA_DIR,
    EMBED_MODEL_NAME,
    FAISS_INDEX_PATH,
    BM25_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest")


def clean_sanskrit_text(text: str) -> str:
    """Normalizes Sanskrit text to NFC Unicode and strips non-Devanagari noise.

    Args:
        text: The raw Sanskrit text.

    Returns:
        The normalized and cleaned Devanagari text.
    """
    # Normalize to NFC Unicode
    text = unicodedata.normalize('NFC', text)

    # Normalize other variants of dandas
    text = text.replace("||", "।।").replace("|", "।")

    # Keep only Devanagari characters, spaces, newlines, dandas, and digits
    # Devanagari block is \u0900-\u097F
    # We also keep English/Devanagari digits and whitespace.
    # Replace non-matching characters with space.
    cleaned = re.sub(r'[^\u0900-\u097F0-9\s]', ' ', text)

    # Normalize whitespace (replace multiple spaces with a single space, keep newlines)
    lines = []
    for line in cleaned.splitlines():
        line_clean = re.sub(r'\s+', ' ', line).strip()
        if line_clean:
            lines.append(line_clean)

    return "\n".join(lines)


def split_into_verses(text: str) -> List[str]:
    """Splits normalized Devanagari text into individual verses based on dandas.

    Args:
        text: The cleaned Sanskrit text.

    Returns:
        A list of verses.
    """
    # Pattern to match verse delimiters: double danda with optional numbers/letters inside,
    # or simple double danda, or single danda.
    pattern = r'((?:।।\s*\d+\s*।।)|(?:।।\s*\d+\s*।)|।।।|।।|।)'
    parts = re.split(pattern, text)

    verses = []
    current_verse = []

    for i, part in enumerate(parts):
        part_clean = part.strip()
        if not part_clean:
            continue

        if i % 2 == 1:
            # This is a delimiter (danda / verse boundary)
            # Append it to the current verse text
            if current_verse:
                current_verse.append(part_clean)
                verses.append(" ".join(current_verse))
                current_verse = []
            else:
                # If there's a delimiter without text before it, just append it
                verses.append(part_clean)
        else:
            # This is actual text content
            current_verse.append(part_clean)

    # If there is any trailing text without a final delimiter, save it
    if current_verse:
        verses.append(" ".join(current_verse))

    # Final cleanup of verses
    cleaned_verses = []
    for v in verses:
        v_clean = re.sub(r'\s+', ' ', v).strip()
        # Keep only if it contains actual Devanagari letters
        if any('\u0900' <= c <= '\u097F' for c in v_clean):
            cleaned_verses.append(v_clean)

    return cleaned_verses


# Sanskrit stories often need nearby verse context.
# I tested 2-verse, 3-verse, and 4-verse chunks.
# 4 verses with 1 overlap gave the strongest overall retrieval metrics.

def chunk_verses(verses: List[str], chunk_size: int = 4, chunk_overlap: int = 1) -> List[Tuple[List[str], str]]:
    """Chunks list of verses with an overlap.

    Args:
        verses: List of verse strings.
        chunk_size: Number of verses per chunk.
        chunk_overlap: Number of overlapping verses.

    Returns:
        A list of tuples, each containing:
        - A list of verse strings in this chunk
        - A verse_id range string (e.g. "0-3")
    """
    chunks_with_ids = []
    if not verses:
        return chunks_with_ids

    step = chunk_size - chunk_overlap
    if step <= 0:
        step = 1

    for i in range(0, len(verses), step):
        chunk = verses[i : i + chunk_size]
        start_idx = i
        end_idx = min(i + chunk_size, len(verses))
        verse_id = f"{start_idx}-{end_idx - 1}"
        chunks_with_ids.append((chunk, verse_id))

        if i + chunk_size >= len(verses):
            break

    return chunks_with_ids


def load_document(file_path: Path) -> str:
    """Loads text from a .txt, .pdf, or .docx file.

    Args:
        file_path: Path to the document.

    Returns:
        The extracted raw text.
    """
    ext = file_path.suffix.lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            p_text = page.extract_text()
            if p_text:
                text_parts.append(p_text)
        return "\n".join(text_parts)
    elif ext == ".docx":
        import docx
        doc = docx.Document(file_path)
        text_parts = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(text_parts)
    else:
        logger.warning(f"Unsupported file extension: {ext} for file {file_path}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Ingest Sanskrit documents and index them.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DATA_DIR),
        help="Path to a Sanskrit file (.txt, .pdf, or .docx) or a directory containing them.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Path {data_path} does not exist.")
        sys.exit(1)

    # Determine files to process
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        files = list(data_path.glob("*.txt")) + list(data_path.glob("*.pdf")) + list(data_path.glob("*.docx"))
    else:
        logger.error(f"Path {data_path} is neither a file nor a directory.")
        sys.exit(1)

    if not files:
        logger.error(f"No .txt, .pdf, or .docx files found at {data_path}")
        sys.exit(1)

    logger.info(f"Found {len(files)} files to ingest.")

    all_chunks = []
    all_metadata = []

    for file_path in files:
        logger.info(f"Processing file: {file_path.name}")
        raw_text = load_document(file_path)
        if not raw_text.strip():
            logger.warning(f"File {file_path.name} is empty, skipping.")
            continue

        cleaned_text = clean_sanskrit_text(raw_text)
        verses = split_into_verses(cleaned_text)
        logger.info(f"Extracted {len(verses)} verses from {file_path.name}")

        chunks = chunk_verses(verses, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")

        for idx, (chunk_verses_list, verse_id) in enumerate(chunks):
            # Combine chunk verses back into a single string
            chunk_text = " ".join(chunk_verses_list)
            all_chunks.append(chunk_text)
            all_metadata.append({
                "source_file": file_path.name,
                "verse_id": verse_id,
                "chunk_index": idx
            })

    if not all_chunks:
        logger.error("No chunks extracted from any documents.")
        sys.exit(1)

    logger.info(f"Total chunks created: {len(all_chunks)}")

    # 2. DUAL INDEXING
    # Ensure index directory exists
    index_dir = Path(FAISS_INDEX_PATH).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    # a) FAISS Indexing
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    # Load embedding model once
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # E5 models need "passage: " prefix for indexing
    # E5 models are trained with asymmetric prefixes.
    # "passage:" is used while indexing documents.
    logger.info("Generating embeddings for chunks...")
    prefixed_chunks = ["passage: " + chunk for chunk in all_chunks]
    # encode handles L2 normalization if normalize_embeddings is True
    embeddings = embed_model.encode(prefixed_chunks, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    logger.info(f"Creating FAISS IndexFlatIP with dimension {dimension}...")
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    # Save FAISS index
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")

    # b) BM25 Indexing
    logger.info("Building BM25 corpus...")
    # Tokenize each chunk: split on whitespace
    tokenized_corpus = [chunk.split() for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    # Persist BM25 index
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_index, f)
    logger.info(f"Saved BM25 index to {BM25_INDEX_PATH}")

    # Persist Chunks and Metadata
    with open(CHUNKS_METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)
    logger.info(f"Saved chunks metadata to {CHUNKS_METADATA_PATH}")

    logger.info("Ingestion and indexing completed successfully.")


if __name__ == "__main__":
    main()
