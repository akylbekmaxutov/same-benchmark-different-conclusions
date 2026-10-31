#!/usr/bin/env python3
"""A throwaway OpenAI-compatible endpoint for exercising the pipeline offline.

NOT part of the study.  It answers with a deterministic-but-noisy mixture so the
downstream tables have something with a real self-flip floor in them.

    python scripts/dev_fake_server.py --port 8799 --model test-model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


def _answer(prompt: str, seed: int) -> tuple[str, str]:
    """A plausible model: mostly a stable function of the prompt, sometimes
    seed-dependent (the self-flip floor), sometimes malformed output."""
    h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    rng = random.Random(f"{h}:{seed}:{ARGS.noise}")
    letters = [ln[0] for ln in prompt.splitlines() if len(ln) > 2 and ln[1] == ")"]
    letters = letters or ["A", "B", "C", "D"]
    stable = letters[h % len(letters)]
    letter = stable if rng.random() > ARGS.flip_p else rng.choice(letters)
    r = rng.random()
    if r < ARGS.parse_fail_p:
        return "I am not sure about this one.", letter
    if r < ARGS.parse_fail_p + ARGS.fenced_p:
        return '```json\n{"answer": "%s"}\n```' % letter, letter
    return '{"answer": "%s"}' % letter, letter


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [
                {"id": ARGS.model, "object": "model", "owned_by": "fake",
                 "vllm_version": "0.0.0-fake"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt = "\n".join(m.get("content") or "" for m in req.get("messages", []))
        content, letter = _answer(prompt, req.get("seed", 0))
        msg = {"role": "assistant", "content": content}
        if ARGS.reasoning:
            msg["reasoning_content"] = f"Considering the options, {letter} fits best."
        self._send({
            "id": "chatcmpl-fake", "object": "chat.completion", "model": ARGS.model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 12,
                      "total_tokens": len(prompt) // 4 + 12,
                      "completion_tokens_details": {
                          "reasoning_tokens": (ARGS.reasoning_tokens
                                               if ARGS.reasoning else 0)}},
        })


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--model", default="test-model")
    p.add_argument("--flip-p", type=float, default=0.08)
    p.add_argument("--parse-fail-p", type=float, default=0.04)
    p.add_argument("--fenced-p", type=float, default=0.10)
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--reasoning-tokens", type=int, default=180)
    p.add_argument("--noise", default="s0", help="salt: different models differ")
    ARGS = p.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"fake vLLM on http://127.0.0.1:{ARGS.port}/v1  model={ARGS.model}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
