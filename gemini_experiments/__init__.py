"""Stage 1: build the unified Kazakh MCQ corpus and generate perturbation
material with Gemini via the Batch API.

This package is intentionally standalone: it is run once, on any machine with
internet access, and produces the release-ready `*_perturbations.jsonl` files
that Stage 2 (`stability_eval`) consumes. No GPU is needed here.
"""
