#!/usr/bin/env bash
# ============================================================================
# Serve one model with vLLM's OpenAI-compatible API.
#
#   MODEL_DIR=/path/to/models/qwen3.5-9b \
#   SERVED_NAME=qwen3.5-9b PORT=8000 CUDA_VISIBLE_DEVICES=0 \
#   REASONING_PARSER=qwen3 MAX_MODEL_LEN=34816 TP=1 \
#   ./serve_model.sh
#
# SERVED_NAME must equal `model_name` in the configuration YAML.
# REASONING_PARSER is listed per model in configs/model_groups.yaml; leave it
# empty for models without a reasoning trace.
# MAX_MODEL_LEN must be at least 34816 (32768 max_tokens + prompt).
#
# Env: MODEL_DIR (required)  SERVED_NAME (required)  PORT (8000)
#      HOST (0.0.0.0)  TP (1)  REASONING_PARSER ("")  MAX_MODEL_LEN (34816)
#      DTYPE (bfloat16)  VLLM_EXTRA_ARGS ("", e.g. "--trust-remote-code")
# ============================================================================
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to a local path or Hugging Face model id}"
: "${SERVED_NAME:?set SERVED_NAME to the model_name used in configs/models/*.yaml}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
REASONING_PARSER="${REASONING_PARSER:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
DTYPE="${DTYPE:-bfloat16}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

ARGS=(--served-model-name "$SERVED_NAME"
      --tensor-parallel-size "$TP"
      --max-model-len "$MAX_MODEL_LEN"
      --dtype "$DTYPE"
      --host "$HOST"
      --port "$PORT")
[[ -n "$REASONING_PARSER" ]] && ARGS+=(--reasoning-parser "$REASONING_PARSER")

# shellcheck disable=SC2086  # VLLM_EXTRA_ARGS is intentionally word-split
exec vllm serve "$MODEL_DIR" "${ARGS[@]}" $VLLM_EXTRA_ARGS
