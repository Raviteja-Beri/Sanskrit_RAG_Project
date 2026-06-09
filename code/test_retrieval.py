# Useful for debugging retrieval without requiring the local LLM model.

import sys
import argparse
import logging
from pathlib import Path

# Set up logging to show the steps
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Add the current directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from retriever import retrieve_relevant_chunks


def run_test(query: str, output_path: str = None):
    output_lines = []
    output_lines.append(f"Executing retrieval for query: '{query}'")
    try:
        results = retrieve_relevant_chunks(query)
    except Exception as e:
        output_lines.append(f"Error during retrieval: {e}")
        output_str = "\n".join(output_lines)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_str)
        try:
            print(output_str)
        except UnicodeEncodeError:
            safe_str = output_str.encode(sys.stdout.encoding or 'ascii', errors='replace').decode(sys.stdout.encoding or 'ascii')
            print(safe_str)
        return

    output_lines.append("\n" + "=" * 60)
    output_lines.append("RETRIEVED CONTEXT PASSAGES")
    output_lines.append("=" * 60)
    if not results:
        output_lines.append("No passages matched the query.")
    else:
        for idx, chunk in enumerate(results, 1):
            meta = chunk["metadata"]
            output_lines.append(
                f"\n[{idx}] Rerank Score: {chunk['rerank_score']:.4f} | "
                f"Source: {meta['source_file']} (Verse Range: {meta['verse_id']})"
            )
            output_lines.append(f"Content: {chunk['text']}")
    output_lines.append("=" * 60 + "\n")

    output_str = "\n".join(output_lines)

    # Save to file first
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"Results successfully saved to {out_p.resolve()}")

    # Print to console safely
    try:
        print(output_str)
    except UnicodeEncodeError:
        safe_str = output_str.encode(sys.stdout.encoding or 'ascii', errors='replace').decode(sys.stdout.encoding or 'ascii')
        print(safe_str)
        print("\n[Note: Some characters could not be displayed in this terminal but are fully preserved in the saved file.]")


def main():
    parser = argparse.ArgumentParser(description="Test Sanskrit RAG retrieval pipeline.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Sanskrit query string (Devanagari, IAST, SLP1, or HK).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the retrieval results in UTF-8 encoding.",
    )
    args = parser.parse_args()

    run_test(args.query, args.output)


if __name__ == "__main__":
    main()
