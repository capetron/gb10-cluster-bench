#!/usr/bin/env python3
"""Find an engine's concurrency knee: where adding streams stops adding throughput.

Usage: llm-concurrency-knee.py BASE MODEL OUT.json LEVELS [PAD_WORDS] [MAX_TOKENS]
  BASE        OpenAI-compatible base URL, e.g. http://localhost:8000/v1
  LEVELS      comma-separated concurrency levels, e.g. 1,4,8,16,32
  PAD_WORDS   optional words of unique filler per prompt (models a long-input job)
  MAX_TOKENS  output cap (default 256)
Set OPENAI_API_KEY if the server needs a key.

This is the quick "cold burst" probe: one closed batch per level, no warmup, natural stop.
For publishable numbers use llm-prefill-bench.py steady (windowed, repeated, fixed output
length); docs/BIAS-AUDIT.md explains why burst numbers swing.

For each level N, fire N streaming completions at once and record per-request
time-to-first-token, per-stream speed, and aggregate tokens/s. Prompts differ per
request so prefix caching cannot flatter the result. Thinking is off: this models
the spoken front desk and bounded worker tasks, not long reasoning.
"""
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LEVELS = [int(x) for x in sys.argv[4].split(",")]
# Optional: pad every prompt with N words of distinct filler "page text" and set the output
# length, to model a bulk page job (a long unique input) instead of a short spoken question.
PAD_WORDS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
MAX_TOKENS = int(sys.argv[6]) if len(sys.argv) > 6 else 256
VOCAB = ("network firewall policy backup vendor invoice patient schedule audit license router "
         "password training incident contract storage printer remote office budget review "
         "customer report access control update server email phone tablet camera badge").split()

TOPICS = ["a bakery", "a dental practice", "a machine shop", "a law firm", "a vineyard",
          "a bike courier", "a veterinary clinic", "a print shop", "a marina", "a tutoring service",
          "a locksmith", "a florist", "a brewery", "a surveying firm", "a daycare", "a food truck"]


def filler(i, level):
    if not PAD_WORDS:
        return ""
    rnd = random.Random(level * 100003 + i)   # distinct per request: no prefix-cache flattery
    return "\n\nPAGE TEXT:\n" + " ".join(rnd.choice(VOCAB) for _ in range(PAD_WORDS))


def one(i, level, results):
    topic = TOPICS[i % len(TOPICS)]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      "Request %d of batch %d. Write a practical 8-step checklist for securing the "
                      "office network of %s. Be specific." % (i, level, topic) + filler(i, level)}],
        "max_tokens": MAX_TOKENS, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")})
    t0 = time.monotonic()
    ttft, toks, ptoks, err = None, 0, 0, None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                if d.get("usage"):
                    toks = d["usage"].get("completion_tokens", toks)
                    ptoks = d["usage"].get("prompt_tokens", ptoks)
                ch = d.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content") and ttft is None:
                    ttft = time.monotonic() - t0
    except Exception as e:  # noqa: BLE001 - a failed stream is a data point, not a crash
        err = "%s: %s" % (type(e).__name__, e)
    results[i] = {"ttft": ttft, "tokens": toks, "prompt_tokens": ptoks, "elapsed": time.monotonic() - t0, "error": err}


rows = []
for level in LEVELS:
    results = [None] * level
    threads = [threading.Thread(target=one, args=(i, level, results)) for i in range(level)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    ok = [r for r in results if r and not r["error"] and r["ttft"] is not None]
    errs = [r["error"] for r in results if r and r["error"]]
    ttfts = sorted(r["ttft"] for r in ok)
    total = sum(r["tokens"] for r in ok)
    per_stream = [r["tokens"] / max(r["elapsed"] - r["ttft"], 1e-6) for r in ok if r["tokens"]]
    row = {
        "concurrency": level, "ok": len(ok), "failed": len(errs), "wall_s": round(wall, 2),
        "aggregate_tok_s": round(total / wall, 1) if wall else 0,
        "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in ok) if ok else None,
        "requests_per_min": round(len(ok) / wall * 60, 1) if wall else 0,
        "per_stream_tok_s_median": round(statistics.median(per_stream), 1) if per_stream else None,
        "ttft_p50_s": round(ttfts[len(ttfts) // 2], 3) if ttfts else None,
        "ttft_p95_s": round(ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.95))], 3) if ttfts else None,
        "ttft_max_s": round(ttfts[-1], 3) if ttfts else None,
        "errors": errs[:3],
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
    time.sleep(3)   # let the engine drain between levels

json.dump({"base": BASE, "model": MODEL, "max_tokens": MAX_TOKENS, "pad_words": PAD_WORDS, "thinking": False,
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows},
          open(OUT, "w"), indent=2)
