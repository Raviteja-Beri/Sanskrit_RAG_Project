"""
CLI entry point for testing the full RAG flow.

This prints retrieved passages before generation so that retrieval quality
can be inspected directly instead of treating the system as a black box.
"""

import argparse
import sys
import logging

from retriever import retrieve_relevant_chunks
from generator import generate_answer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def run_pipeline(query: str) -> None:
    """Runs the end-to-end RAG pipeline.

    Args:
        query: User query in Devanagari, IAST, SLP1, or HK.
    """
    logger.info(f"Received query: '{query}'")

    # Step 1: Retrieval and Re-ranking
    try:
        retrieved_chunks = retrieve_relevant_chunks(query)
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        sys.exit(1)

    if not retrieved_chunks:
        logger.warning("No relevant passages were found in the context.")
        context_texts = []
    else:
        logger.info(f"Retrieved {len(retrieved_chunks)} passages.")
        print("\n--- RETRIEVED CONTEXT ---")
        for idx, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk["metadata"]
            print(
                f"\n[{idx}] Score: {chunk['rerank_score']:.4f} | "
                f"Source: {meta['source_file']} (Verse: {meta['verse_id']})"
            )
            print(f"Content: {chunk['text']}")
        print("-------------------------\n")
        context_texts = [chunk["text"] for chunk in retrieved_chunks]

    # Step 2: Generation
    try:
        answer = generate_answer(query, context_texts)
    except Exception as e:
        logger.error(f"Error during answer generation: {e}")
        sys.exit(1)

    print("--- ANSWER ---")
    print(answer)
    print("--------------\n")


def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end Sanskrit RAG pipeline.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Sanskrit query string (Devanagari, IAST, SLP1, or HK).",
    )
    args = parser.parse_args()
    run_pipeline(args.query)


if __name__ == "__main__":
    main()
