# Retrieval quality is the focus of this project.
# The generator is deliberately simple: it answers only from retrieved context.

import os
import logging
from typing import List

from llama_cpp import Llama

# Import configurations
from config import (
    MODEL_PATH,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    N_THREADS,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("generator")

# Global variable to cache the Llama model
_llama_model = None


def init_generator() -> None:
    """Initializes and loads the Llama model once at the module level.

    Raises:
        FileNotFoundError: If the GGUF model file is not found at the configured path.
    """
    global _llama_model
    if _llama_model is not None:
        return

    logger.info("Initializing generator component...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Llama model file not found at {MODEL_PATH}. "
            f"Please run the download command: "
            f"huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF "
            f"--include \"Llama-3.2-3B-Instruct-Q4_K_M.gguf\" --local-dir ./models"
        )

    logger.info(f"Loading GGUF model from {MODEL_PATH}...")
    _llama_model = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=N_THREADS,
        verbose=False
    )
    logger.info("Generator model loaded successfully.")


def generate_answer(user_query: str, context_chunks: List[str]) -> str:
    """Generates an answer using the llama-cpp model based ONLY on provided context.

    Args:
        user_query: The raw query from the user.
        context_chunks: Top 5 retrieved Sanskrit chunks.

    Returns:
        The generated answer string.
    """
    init_generator()

    # Format the prompt using the exact template requested
    context_text = ""
    for chunk in context_chunks:
        context_text += f"{chunk}\n"

    prompt = (
        "You are a Sanskrit scholar assistant.\n"
        "Answer the question using ONLY the provided context passages. "
        "If the answer is not in the context, say so explicitly.\n\n"
        f"Context:\n{context_text.strip()}\n\n"
        f"Question: {user_query}\n"
        "Answer:"
    )

    logger.info("Generating answer via Llama model...")
    # Run LLM inference
    response = _llama_model(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=["Question:", "\n\n\n"],  # Simple stop sequences to prevent runaway generation
    )

    answer = response["choices"][0]["text"].strip()
    return answer
