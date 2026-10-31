"""Small JSONL / YAML helpers.  orjson everywhere for speed and stable output."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import orjson
import yaml


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "rb") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as exc:  # pragma: no cover - data bug
                raise ValueError(f"{path}:{line_no}: malformed JSON: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "wb") as fh:
        for row in rows:
            fh.write(orjson.dumps(row, option=orjson.OPT_NON_STR_KEYS))
            fh.write(b"\n")
            n += 1
    return n


class JsonlAppender:
    """Line-buffered append-only JSONL writer.  Never holds a run in memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "ab", buffering=0)

    def write(self, row: dict) -> None:
        self._fh.write(orjson.dumps(row, option=orjson.OPT_NON_STR_KEYS) + b"\n")

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception:
            pass
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
        fh.write(b"\n")


def read_json(path: str | Path) -> Any:
    with open(path, "rb") as fh:
        return orjson.loads(fh.read())


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
