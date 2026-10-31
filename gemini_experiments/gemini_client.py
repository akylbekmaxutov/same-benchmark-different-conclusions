"""Thin, defensive wrapper around google-genai's Batch API.

Only the pieces this project needs: upload a JSONL request file, create a batch
job, read its state, and download the result file. Written with getattr/dict
fallbacks so it keeps working across minor SDK shape changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .config import GEMINI_MODEL, require_api_key

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def get_client():
    from google import genai

    return genai.Client(api_key=require_api_key())


# ------------------------------------------------------------------ helpers --
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            return default
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def job_state(job: Any) -> str:
    state = _attr(job, "state", default="UNKNOWN")
    return str(_attr(state, "name", default=state))


def job_error(job: Any) -> Optional[str]:
    err = _attr(job, "error")
    return None if err is None else str(err)


def result_file_name(job: Any) -> Optional[str]:
    dest = _attr(job, "dest", "destination")
    name = _attr(dest, "file_name", "fileName", "responses_file")
    return str(name) if name else None


# ------------------------------------------------------------------- calls ---
def upload_batch_file(client, path: Path, display_name: str) -> str:
    uploaded = client.files.upload(
        file=str(path),
        config={"display_name": display_name, "mime_type": "application/jsonl"},
    )
    return str(_attr(uploaded, "name"))


def create_batch_job(client, src_file_name: str, display_name: str, model: str = GEMINI_MODEL):
    return client.batches.create(
        model=model,
        src=src_file_name,
        config={"display_name": display_name},
    )


def get_batch_job(client, job_name: str):
    return client.batches.get(name=job_name)


def cancel_batch_job(client, job_name: str) -> None:
    client.batches.cancel(name=job_name)


def download_result_file(client, file_name: str, out_path: Path) -> Path:
    data = client.files.download(file=file_name)
    if isinstance(data, str):
        data = data.encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


# ------------------------------------------------------------ result parsing --
def extract_text(response: Any) -> Optional[str]:
    """Pull the generated text out of a (batch) GenerateContentResponse."""
    if response is None:
        return None
    direct = _attr(response, "text")
    if isinstance(direct, str) and direct.strip():
        return direct

    candidates = _attr(response, "candidates", default=[]) or []
    for cand in candidates:
        content = _attr(cand, "content")
        parts = _attr(content, "parts", default=[]) or []
        chunks = [p.get("text") if isinstance(p, dict) else getattr(p, "text", None) for p in parts]
        chunks = [c for c in chunks if isinstance(c, str)]
        if chunks:
            joined = "".join(chunks).strip()
            if joined:
                return joined
    return None


def iter_batch_results(result_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield {"key", "text", "error"} for every line of a batch result file."""
    with open(result_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"key": None, "text": None, "error": f"unparseable result line: {exc}"}
                continue

            key = obj.get("key") or obj.get("custom_id") or obj.get("id")
            if obj.get("error"):
                yield {"key": key, "text": None, "error": json.dumps(obj["error"], ensure_ascii=False)}
                continue

            response = obj.get("response") or obj.get("result") or obj
            text = extract_text(response)
            if text is None:
                yield {"key": key, "text": None,
                       "error": "empty generation: " + json.dumps(response, ensure_ascii=False)[:500]}
            else:
                yield {"key": key, "text": text, "error": None}
