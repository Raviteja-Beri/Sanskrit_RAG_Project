"""
The evaluator uses verse-range overlap instead of exact chunk IDs.

This is important because different chunk sizes produce different chunk IDs.
By evaluating at the verse level, the same benchmark can be reused across
2-verse, 3-verse, and 4-verse chunking experiments.
"""


import os
import json
import csv
import logging
import argparse
import math
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Import retriever functions
from retriever import retrieve_relevant_chunks
from config import TEST_SET_PATH, EVAL_RESULTS_PATH, REPORT_DIR, CHUNKS_METADATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluator")


def parse_range(r_str: str) -> set:
    """Helper to parse a range string like 'start-end' to a set of integers."""
    try:
        start, end = map(int, r_str.split('-'))
        return set(range(start, end + 1))
    except Exception:
        return set()


def is_relevant(retrieved_range_str: str, relevant_ranges: List[str]) -> bool:
    """Checks if the retrieved verse range overlaps with any of the ground-truth relevant ranges."""
    retrieved_verses = parse_range(retrieved_range_str)
    if not retrieved_verses:
        return False
    for rel_r in relevant_ranges:
        rel_verses = parse_range(rel_r)
        if retrieved_verses & rel_verses:
            return True
    return False


def precision_at_k(rel_vector: List[int], k: int) -> float:
    """Computes Precision@k from a binary relevance vector."""
    if k <= 0:
        return 0.0
    return sum(rel_vector[:k]) / k


def recall_at_k(rel_vector: List[int], total_relevant: int, k: int) -> float:
    """Computes Recall@k from a binary relevance vector."""
    if total_relevant <= 0:
        return 0.0
    return sum(rel_vector[:k]) / total_relevant


def mrr(rel_vector: List[int]) -> float:
    """Computes Mean Reciprocal Rank (MRR) from a binary relevance vector."""
    for i, val in enumerate(rel_vector, 1):
        if val > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel_vector: List[int], total_relevant: int, k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG@k) from a binary relevance vector."""
    top_k_rel = rel_vector[:k]
    dcg = sum(val / math.log2(i + 2) for i, val in enumerate(top_k_rel))
    ideal_rel = [1] * min(total_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(ideal_rel)))
    return dcg / idcg if idcg > 0.0 else 0.0


def average_precision(rel_vector: List[int], total_relevant: int) -> float:
    """Computes Average Precision (AP) from a binary relevance vector."""
    if total_relevant <= 0:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, val in enumerate(rel_vector, 1):
        if val > 0:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / total_relevant


def evaluate_system(test_set_path: str, k: int = 5) -> Dict[str, Any]:
    """Runs evaluation on the test set and returns the average metrics.

    Args:
        test_set_path: Path to the JSON test set.
        k: Evaluation rank cutoff (default 5).

    Returns:
        A dictionary containing average metrics across all test queries.
    """
    if not os.path.exists(test_set_path):
        raise FileNotFoundError(f"Test set file not found at {test_set_path}")

    with open(test_set_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    logger.info(f"Running evaluation on {len(queries_data)} query-answer pairs at k={k}...")

    # Load corpus metadata to find verse_ids for relevant_verses text mapping
    corpus_chunks = []
    corpus_metadata = []
    if os.path.exists(CHUNKS_METADATA_PATH):
        with open(CHUNKS_METADATA_PATH, "rb") as f:
            corpus_data = pickle.load(f)
            corpus_chunks = corpus_data["chunks"]
            corpus_metadata = corpus_data["metadata"]

    precisions = []
    recalls = []
    rrs = []
    ndcgs = []
    aps = []

    # Store individual query results for the CSV report
    query_results = []

    for idx, item in enumerate(queries_data, 1):
        query = item["query"]

        # Determine ground truth relevant ranges
        relevant_ranges = item.get("relevant_verse_ranges", [])

        # Retrieve top chunks (returns list of dicts)
        retrieved_chunks = retrieve_relevant_chunks(query)
        retrieved = [chunk["metadata"]["verse_id"] for chunk in retrieved_chunks]

        # Calculate binary relevance vector for retrieved chunks
        rel_vector = [1 if is_relevant(vid, relevant_ranges) else 0 for vid in retrieved]

        # Calculate total relevant chunks present in the corpus
        total_relevant = sum(
            1 for chunk_meta in corpus_metadata
            if is_relevant(chunk_meta["verse_id"], relevant_ranges)
        )

        logger.info(f"Evaluating Query {idx}: '{query}' (Relevant Chunks in Corpus: {total_relevant})")

        # Convert retrieved verse ranges into a binary relevance vector.
        # This lets us compute standard IR metrics query by query.

        p_at_k = precision_at_k(rel_vector, k)
        r_at_k = recall_at_k(rel_vector, total_relevant, k)
        rr = mrr(rel_vector)
        ndcg_val = ndcg_at_k(rel_vector, total_relevant, k)
        ap = average_precision(rel_vector, total_relevant)

        precisions.append(p_at_k)
        recalls.append(r_at_k)
        rrs.append(rr)
        ndcgs.append(ndcg_val)
        aps.append(ap)

        query_results.append({
            "query_id": idx,
            "query": query,
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "reciprocal_rank": rr,
            "ndcg_at_k": ndcg_val,
            "average_precision": ap
        })

    # Compute mean metrics
    mean_precision = float(np.mean(precisions))
    mean_recall = float(np.mean(recalls))
    mean_mrr = float(np.mean(rrs))
    mean_ndcg = float(np.mean(ndcgs))
    mean_map = float(np.mean(aps))

    summary = {
        "mean_precision_at_k": mean_precision,
        "mean_recall_at_k": mean_recall,
        "mrr": mean_mrr,
        "mean_ndcg_at_k": mean_ndcg,
        "map": mean_map
    }

    # Save results to CSV
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Path(EVAL_RESULTS_PATH)

    with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "query_id",
            "query",
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "ndcg_at_k",
            "average_precision",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for q_res in query_results:
            writer.writerow(q_res)

        # Write average summary row at the end
        csvfile.write("\n")
        writer_summary = csv.writer(csvfile)
        writer_summary.writerow([
            "Metric Summary",
            f"Mean Precision@{k}",
            f"Mean Recall@{k}",
            "MRR",
            f"Mean NDCG@{k}",
            "MAP",
        ])
        writer_summary.writerow([
            "Average Value",
            f"{mean_precision:.4f}",
            f"{mean_recall:.4f}",
            f"{mean_mrr:.4f}",
            f"{mean_ndcg:.4f}",
            f"{mean_map:.4f}",
        ])

    logger.info(f"Saved evaluation results to {report_path}")

    # Generate PDF report automatically
    try:
        from generate_pdf_report import build_pdf
        pdf_path = Path(EVAL_RESULTS_PATH).with_suffix(".pdf")
        build_pdf(str(report_path), str(pdf_path))
        logger.info(f"Generated PDF evaluation report at {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to generate PDF evaluation report: {e}")

    # Print summary table
    print("\n" + "=" * 50)
    print(f"SANSKRIT RAG EVALUATION SUMMARY (k={k})")
    print("=" * 50)
    print(f"Mean Precision@{k} : {mean_precision:.4f}")
    print(f"Mean Recall@{k}    : {mean_recall:.4f}")
    print(f"MRR                : {mean_mrr:.4f}")
    print(f"Mean NDCG@{k}      : {mean_ndcg:.4f}")
    print(f"MAP                : {mean_map:.4f}")
    print("=" * 50 + "\n")

    return summary



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Sanskrit RAG system using search retrieval metrics."
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default=str(TEST_SET_PATH),
        help="Path to the JSON test set query-answer pairs file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retrieval cutoff k for Precision, Recall, and NDCG (default 5).",
    )
    args = parser.parse_args()

    try:
        evaluate_system(args.test_set, args.k)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
