"""Central configuration + paths for the Gemini generation stage."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- paths -----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEM_ROOT = PROJECT_ROOT / "gemini_experiments"

DATA_DIR = PROJECT_ROOT / "data"
NORMALIZED_DIR = DATA_DIR / "normalized"          # Phase 1-2 output
PERTURBATIONS_DIR = DATA_DIR / "perturbations"    # Phase 7 output (released artifact)

WORK_DIR = GEM_ROOT / "work"
BATCH_INPUT_DIR = WORK_DIR / "batch_input"
BATCH_OUTPUT_DIR = WORK_DIR / "batch_output"
GENERATIONS_DIR = WORK_DIR / "generations"
REPORTS_DIR = GEM_ROOT / "reports"

for _d in (
    NORMALIZED_DIR, PERTURBATIONS_DIR, BATCH_INPUT_DIR,
    BATCH_OUTPUT_DIR, GENERATIONS_DIR, REPORTS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- datasets -----
DATASETS = ("kazmmlu", "kazculture", "kazbench_kk")

HF_SOURCES = {
    "kazmmlu": "MBZUAI/KazMMLU",
    "kazculture": "issai/KazCulture",
    "kazbench_kk": "kz-transformers/kk-socio-cultural-bench-mc",
}

# Split actually used for evaluation (KazBench-KK only exposes `train`; we
# document that explicitly rather than silently renaming it).
HF_SPLITS = {
    "kazmmlu": "test",
    "kazculture": "test",
    "kazbench_kk": "train",
}

OPTION_LETTERS = ("A", "B", "C", "D", "E")

# --------------------------------------------------------------- gemini -----
load_dotenv(PROJECT_ROOT / ".env")

# `.env` in this repo uses GEMINI_KEY; GOOGLE_API_KEY / GEMINI_API_KEY are the
# names the google-genai SDK picks up on its own, so accept any of them.
GEMINI_API_KEY = (
    os.getenv("GEMINI_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()

# The Description calls for "Gemini 3.5 Flash-Lite". Model ids move fast, so it
# is overridable without touching code:  export GEMINI_MODEL=...
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

# Generation parameters - frozen and released with the dataset (Phase 32).
# max_output_tokens was raised from 1024 in v1.0: long passage-based KazMMLU
# items truncated mid-string and came back as unparseable JSON.
GEN_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_output_tokens": 2048,
    "response_mime_type": "application/json",
}

# Benchmark items about history, religion and social conflict were being
# refused as harmful; the content is academic evaluation material.
SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
]

# Requests per batch input file, i.e. per batch job.
# 0 = no chunking: one job per dataset (entire Kazakh KazMMLU in one batch,
# entire KazBench-KK in one batch, entire KazCulture in one batch).
# Set BATCH_CHUNK_SIZE=5000 to split large datasets across several jobs instead,
# which localizes failures at the cost of more jobs to track.
BATCH_CHUNK_SIZE = int(os.getenv("BATCH_CHUNK_SIZE", "0"))

BATCH_POLL_SECONDS = int(os.getenv("BATCH_POLL_SECONDS", "60"))

# v1.1: paraphrase rules hardened after round 1 (options appended to the
# paraphrase, digits spelled out, passages dropped, formulas dropped).
PROMPT_VERSION = "v1.1"


def require_api_key() -> str:
    if not GEMINI_API_KEY:
        raise SystemExit(
            "No Gemini API key found.\n"
            f"Set GEMINI_KEY=<your key> in {PROJECT_ROOT / '.env'} "
            "(or export GEMINI_API_KEY / GOOGLE_API_KEY)."
        )
    return GEMINI_API_KEY


def normalized_path(dataset: str) -> Path:
    return NORMALIZED_DIR / f"{dataset}.jsonl"


def generations_path(dataset: str) -> Path:
    return GENERATIONS_DIR / f"{dataset}.jsonl"


def release_full_path(dataset: str) -> Path:
    """Complete record: every generated item, each stamped with its verdict."""
    return PERTURBATIONS_DIR / f"{dataset}_perturbations_full.jsonl"


def release_path(dataset: str) -> Path:
    """Usable items only - this is what the evaluation stage loads."""
    return PERTURBATIONS_DIR / f"{dataset}_perturbations.jsonl"
