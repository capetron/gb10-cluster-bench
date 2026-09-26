#!/usr/bin/env python3
"""Long-context / prefill benchmark for an OpenAI-compatible vLLM server.

Short-prompt benchmarks (a few dozen input tokens) measure decode only. This harness measures prompt processing (prefill): TTFT and prompt_tokens/TTFT at exact input
lengths, a RAG-shaped ladder, and long-context user capacity.

Prompt construction (exactly reproducible on any server running the same checkpoint):
  * A deterministic pseudo-English corpus is generated from random.Random(CORPUS_SEED)
    over the fixed VOCAB below (CORPUS_WORDS words, sentences of 8-20 words, paragraphs
    of 6 sentences). It is tokenized by the SERVER (/tokenize, per paragraph, ids
    concatenated) so the token ids come from the model's own tokenizer.
  * Each request gets uid = sha256("<seed>|<phase>|<target>|<conc>|<rep>|<i>")[:16].
    The user message is  "Document <uid>.\n\n" + detok(ids[off:off+F]) +
    "\n\nIn one sentence, what is this document about?"  where off = int(uid,16) mod
    (len(ids) - F). The uid is the first thing in the user turn, so no KV block can be
    shared between two requests: prefix caching cannot hit (the /metrics prefix-cache
    counters are recorded to prove it).
  * F is solved so the full chat-templated prompt (enable_thinking False,
    add_generation_prompt True) counts exactly <target> tokens by /tokenize, iterating
    up to 6 times. The server's usage.prompt_tokens is what gets reported, never the
    target.
  * Output: ignore_eos True with max_tokens fixed, temperature 0, thinking off, so every
    request generates exactly max_tokens tokens.

Timing: TTFT = request send -> first streamed chunk carrying content or reasoning text.
Prefill tok/s (single stream) = usage.prompt_tokens / TTFT. Aggregate prefill for a burst
of N simultaneous requests = sum(prompt_tokens) / (time until the LAST request's first
token). Decode rate per stream = (completion_tokens - 1) / (last token - first token).

Usage:
  llm-prefill-bench.py BASE MODEL OUT.json gate
  llm-prefill-bench.py BASE MODEL OUT.json prefill  [--lengths 1024,4096,...] [--conc 1,4,8] [--reps 3] [--max-tokens 32]
  llm-prefill-bench.py BASE MODEL OUT.json rag      [--input 8192] [--max-tokens 256] [--conc 1,2,4,8,16,32] [--rounds 3]
  llm-prefill-bench.py BASE MODEL OUT.json capacity --context 32768 [--conc 1,4,8,16,24,32,40] [--max-tokens 256]
  llm-prefill-bench.py BASE MODEL OUT.json steady   [--conc 1,4,8,16,32] [--input 256] [--max-tokens 512]
                       [--window 120] [--ramp 20] [--reps 3] [--order-seed 1]
Options for every phase: --run-seed S (prompt uniqueness across runs against a warm server),
--label KEY (key under "phases"). BASE may be a comma-separated list of replica endpoints: requests
are spread round-robin by worker (a client-side load balancer); /tokenize uses the first one.

steady phase (docs/BIAS-AUDIT.md, method fix 1): for each concurrency level, N workers keep
one request each in flight for ramp+window seconds; only tokens produced inside the window count
(each request's tokens are spread linearly between its first and last token). One unrecorded warmup
burst per level precedes the first repeat; level order is shuffled per repeat with --order-seed.
Prompts are local corpus slices (no server round trips inside the loop), unique per request.
Results are merged into OUT.json under "phases".

Authentication: if the server needs an API key, set OPENAI_API_KEY (sent as a Bearer token).
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request

CORPUS_SEED = 20260924
CORPUS_WORDS = 240000
RUN_SEED = "gb10-longctx-2026-09-24"
VOCAB = ("network firewall policy backup vendor invoice patient schedule audit license router "
         "password training incident contract storage printer remote office budget review "
         "customer report access control update server email phone tablet camera badge the a "
         "of and to in for with on by from after before during quarterly annual team manager "
         "client system record request approval change ticket laptop desktop cloud tenant "
         "account user group role permission encryption key certificate domain site building "
         "floor room cabinet rack switch cable power supply battery generator cooling alarm "
         "sensor door lock visitor log retention archive restore test result finding risk "
         "control owner evidence document procedure standard baseline exception waiver plan "
         "milestone deadline status open closed pending escalated resolved verified reviewed "
         "signed shipped received installed replaced patched scanned monitored reported").split()
INSTR = "\n\nIn one sentence, what is this document about?"
# steady phase: ask for long natural output so ignore_eos rarely has to force text past a natural
# stop (post-EOS text would distort speculative-decoding acceptance)
STEADY_INSTR = ("\n\nWrite a detailed, multi-paragraph operations report of at least 800 words based on "
                "this log: summarize the themes, list the risks you see, and recommend next steps.")
# --content code: same input slice, but the output is
# structured code, where draft models (MTP / DFlash2 / DSpark) accept far more tokens than on prose.
# Speculative-decoding gains are content-dependent, so prose and code are reported separately.
STEADY_INSTR_CODE = ("\n\nWrite a complete, production-quality Python 3 module (at least 400 lines, code only, "
                     "no prose outside comments) that parses log records like the ones above into dataclasses, "
                     "validates every field, aggregates counts per category and per status, and emits a JSON "
                     "report. Include type hints, docstrings, a CLI entry point with argparse, and a full "
                     "pytest test suite at the end.")

HEADERS = {"Content-Type": "application/json",
           "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")}

_ids_cache = {}


def base0(base):
    return base.split(",")[0]


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base0(base) + path, data=json.dumps(body).encode(),
                                 headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def root(base):
    base = base0(base)
    return base[:-3] if base.endswith("/v1") else base


def corpus_paragraphs():
    rnd = random.Random(CORPUS_SEED)
    words, paras, sent, para = 0, [], [], []
    while words < CORPUS_WORDS:
        n = rnd.randint(8, 20)
        s = " ".join(rnd.choice(VOCAB) for _ in range(n))
        para.append(s[0].upper() + s[1:] + ".")
        words += n
        if len(para) == 6:
            paras.append(" ".join(para) + "\n\n")
            para = []
    if para:
        paras.append(" ".join(para) + "\n\n")
    return paras


def corpus_ids(base, model):
    if "ids" in _ids_cache:
        return _ids_cache["ids"]
    paras = corpus_paragraphs()
    ids = []
    chunk = 40
    for k in range(0, len(paras), chunk):
        r = post(root(base), "/tokenize", {"model": model, "prompt": "".join(paras[k:k + chunk]),
                                           "add_special_tokens": False})
        ids.extend(r["tokens"])
    _ids_cache["ids"] = ids
    _ids_cache["sha"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return ids


def msgs(text):
    return [{"role": "user", "content": text}]


def count_chat(base, model, text):
    r = post(root(base), "/tokenize", {"model": model, "messages": msgs(text), "add_generation_prompt": True,
                                       "chat_template_kwargs": {"enable_thinking": False}})
    return r["count"]


def build_prompt(base, model, target, key, needle=None):
    """Return (text, templated_token_count, uid, offset)."""
    ids = corpus_ids(base, model)
    uid = hashlib.sha256(("%s|%s" % (RUN_SEED, key)).encode()).hexdigest()[:16]
    base = base0(base)
    head = "Document %s.\n\n" % uid
    tail = INSTR if needle is None else "\n\n" + needle[1]
    overhead = count_chat(base, model, head + tail)
    f = max(0, target - overhead)
    off = int(uid, 16) % max(1, len(ids) - f - 1)
    text, n = None, None
    for _ in range(6):
        body = post(root(base), "/detokenize", {"model": model, "tokens": ids[off:off + f]})["prompt"]
        if needle is not None:
            mid = len(body) // 2
            body = body[:mid] + " " + needle[0] + " " + body[mid:]
        text = head + body + tail
        n = count_chat(base, model, text)
        if n == target:
            break
        f += target - n
    return text, n, uid, off


class MetricsPoller(threading.Thread):
    NAMES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
             "vllm:gpu_cache_usage_perc", "vllm:num_preemptions_total", "vllm:prefix_cache_hits_total",
             "vllm:prefix_cache_queries_total", "vllm:spec_decode_num_drafts_total",
             "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total")

    def __init__(self, base):
        super().__init__(daemon=True)
        self.url = ",".join((b[:-3] if b.endswith("/v1") else b) + "/metrics" for b in base.split(","))
        self.stop = threading.Event()
        self.samples = []

    @classmethod
    def scrape(cls, url):
        out = {}
        txt = ""
        for u in url.split(","):
            try:
                txt += urllib.request.urlopen(u, timeout=10).read().decode() + "\n"
            except Exception:  # noqa: BLE001
                pass
        for line in txt.splitlines():
            if line.startswith("#"):
                continue
            m = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
            if m and m.group(1) in cls.NAMES:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            elif m and m.group(1) == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
                pm = re.search(r'position="(\d+)"', m.group(2) or "")
                if pm:
                    k = "accepted_pos_%s" % pm.group(1)
                    out[k] = out.get(k, 0.0) + float(m.group(3))
        return out

    def run(self):
        while not self.stop.is_set():
            self.samples.append(self.scrape(self.url))
            self.stop.wait(0.5)

    def summary(self, before, after):
        def peak(k):
            v = [s[k] for s in self.samples if k in s]
            return max(v) if v else None
        kv = peak("vllm:kv_cache_usage_perc")
        if kv is None:
            kv = peak("vllm:gpu_cache_usage_perc")

        def delta(k):
            return (after.get(k, 0) - before.get(k, 0)) if k in after else None
        spec = None
        dr = delta("vllm:spec_decode_num_drafts_total")
        if dr:
            acc = delta("vllm:spec_decode_num_accepted_tokens_total") or 0
            dt = delta("vllm:spec_decode_num_draft_tokens_total") or 0
            pos = {k: after[k] - before.get(k, 0) for k in sorted(after) if k.startswith("accepted_pos_")}
            spec = {"drafts": dr, "draft_tokens": dt, "accepted_tokens": acc,
                    "acceptance_rate": r3(acc / dt, 4) if dt else None,
                    "tokens_per_verify_cycle": r3(acc / dr + 1, 3),
                    "acceptance_by_position": {k[13:]: r3(v / dr, 4) for k, v in pos.items()}}
        return {"spec_decode": spec, "peak_running": peak("vllm:num_requests_running"),
                "peak_waiting": peak("vllm:num_requests_waiting"),
                "peak_kv_usage_frac": kv,
                "preemptions": delta("vllm:num_preemptions_total"),
                "prefix_cache_hit_tokens": delta("vllm:prefix_cache_hits_total"),
                "prefix_cache_query_tokens": delta("vllm:prefix_cache_queries_total")}


def stream_one(base, model, text, max_tokens, t_start_barrier=None, timeout=3600):
    body = {"model": model, "messages": msgs(text), "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base0(base) + "/chat/completions", data=json.dumps(body).encode(),
                                 headers=HEADERS)
    if t_start_barrier is not None:
        t_start_barrier.wait()
    t0 = time.monotonic()
    first, last, ctoks, ptoks, err, out = None, None, 0, None, None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                if d.get("usage"):
                    ctoks = d["usage"].get("completion_tokens", ctoks)
                    ptoks = d["usage"].get("prompt_tokens", ptoks)
                ch = d.get("choices") or []
                if ch:
                    dl = ch[0].get("delta") or {}
                    piece = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                    if piece:
                        now = time.monotonic()
                        if first is None:
                            first = now
                        last = now
                        out.append(piece)
    except Exception as e:  # noqa: BLE001 - a failed request is a data point
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
    end = time.monotonic()
    res = {"t0": t0, "ttft_s": (first - t0) if first else None, "end_s": end - t0,
           "prompt_tokens": ptoks, "completion_tokens": ctoks, "error": err, "text": "".join(out)}
    if first and last and ctoks and ctoks > 1 and last > first:
        res["decode_tok_s"] = (ctoks - 1) / (last - first)
    return res


def pct(v, p):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def r3(x, n=3):
    return round(x, n) if isinstance(x, (int, float)) else x


def burst(base, model, prompts, max_tokens):
    """Fire len(prompts) requests at the same instant. Returns (results, wall_start)."""
    n = len(prompts)
    results = [None] * n
    bar = threading.Barrier(n + 1)

    bases = base.split(",")

    def w(i):
        results[i] = stream_one(bases[i % len(bases)], model, prompts[i], max_tokens, bar)
    ths = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    bar.wait()
    t0 = time.monotonic()
    for t in ths:
        t.join()
    return results, t0


def summarize_burst(results, t0):
    ok = [r for r in results if r and not r["error"] and r["ttft_s"] is not None]
    errs = [r["error"] for r in results if r and r["error"]]
    ttfts = [r["ttft_s"] for r in ok]
    ptoks = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
    last_first = max((r["t0"] - t0) + r["ttft_s"] for r in ok) if ok else None
    wall = max((r["t0"] - t0) + r["end_s"] for r in results if r) if results else None
    dec = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    return {"requests": len(results), "ok": len(ok), "failed": len(results) - len(ok),
            "prompt_tokens_each": sorted(set(ptoks)), "prompt_tokens_total": sum(ptoks),
            "ttft_p50_s": r3(pct(ttfts, 50)), "ttft_p95_s": r3(pct(ttfts, 95)),
            "ttft_min_s": r3(min(ttfts)) if ttfts else None, "ttft_max_s": r3(max(ttfts)) if ttfts else None,
            "time_to_last_first_token_s": r3(last_first),
            "aggregate_prefill_tok_s": r3(sum(ptoks) / last_first, 1) if last_first and ok else None,
            "decode_tok_s_per_stream_p50": r3(pct(dec, 50), 1), "wall_s": r3(wall),
            "errors": errs[:5]}


def phase_prefill(a):
    rows = []
    for L in a.lengths:
        for c in a.conc:
            reps = a.reps if c == 1 else 1
            for rep in range(reps):
                prompts, meta = [], []
                for i in range(c):
                    t, n, uid, off = build_prompt(a.base, a.model, L, "prefill|%d|%d|%d|%d" % (L, c, rep, i))
                    prompts.append(t)
                    meta.append({"uid": uid, "offset": off, "templated_tokens": n})
                mp = MetricsPoller(a.base)
                before = mp.scrape(mp.url)
                mp.start()
                res, t0 = burst(a.base, a.model, prompts, a.max_tokens)
                mp.stop.set()
                mp.join()
                after = mp.scrape(mp.url)
                row = {"target_input_tokens": L, "concurrency": c, "rep": rep, "max_tokens": a.max_tokens}
                row.update(summarize_burst(res, t0))
                if c == 1 and res[0] and res[0]["ttft_s"] and res[0]["prompt_tokens"]:
                    row["prefill_tok_s"] = r3(res[0]["prompt_tokens"] / res[0]["ttft_s"], 1)
                row["server_metrics"] = mp.summary(before, after)
                row["prompt_uids"] = [m["uid"] for m in meta]
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in ("target_input_tokens", "concurrency", "rep", "ok",
                      "prompt_tokens_each", "ttft_p50_s", "ttft_max_s", "prefill_tok_s", "aggregate_prefill_tok_s",
                      "errors")}), flush=True)
    return {"lengths": a.lengths, "concurrency": a.conc, "reps_at_c1": a.reps, "max_tokens": a.max_tokens,
            "rows": rows}


def closed_loop(a, conc, target, rounds, max_tokens, tag):
    prompts = [[build_prompt(a.base, a.model, target, "%s|%d|%d|%d|%d" % (tag, target, conc, w, k))[0]
                for k in range(rounds)] for w in range(conc)]
    results = []
    lock = threading.Lock()
    bar = threading.Barrier(conc + 1)

    def worker(w):
        bar.wait()
        for k in range(rounds):
            r = stream_one(a.base.split(",")[w % len(a.base.split(","))], a.model, prompts[w][k], max_tokens)
            r["round"] = k
            with lock:
                results.append(r)
    mp = MetricsPoller(a.base)
    before = mp.scrape(mp.url)
    mp.start()
    ths = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
    for t in ths:
        t.start()
    bar.wait()
    t0 = time.monotonic()
    for t in ths:
        t.join()
    wall = time.monotonic() - t0
    mp.stop.set()
    mp.join()
    after = mp.scrape(mp.url)
    ok = [r for r in results if not r["error"] and r["ttft_s"] is not None]
    ttfts = [r["ttft_s"] for r in ok]
    dec = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    ctoks = sum(r["completion_tokens"] or 0 for r in ok)
    ptoks = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
    return {"concurrency": conc, "rounds_per_worker": rounds, "requests": len(results), "ok": len(ok),
            "failed": len(results) - len(ok), "wall_s": r3(wall),
            "requests_per_min": r3(len(ok) / wall * 60, 1),
            "prompt_tokens_each": sorted(set(ptoks)),
            "ttft_p50_s": r3(pct(ttfts, 50)), "ttft_p95_s": r3(pct(ttfts, 95)), "ttft_max_s": r3(max(ttfts)) if ttfts else None,
            "decode_tok_s_per_stream_p50": r3(pct(dec, 50), 1), "decode_tok_s_per_stream_p5": r3(pct(dec, 5), 1),
            "aggregate_output_tok_s": r3(ctoks / wall, 1),
            "aggregate_input_tok_s": r3(sum(ptoks) / wall, 1),
            "server_metrics": mp.summary(before, after),
            "errors": [r["error"] for r in results if r["error"]][:5]}


def phase_rag(a):
    rows = []
    for c in a.conc:
        row = closed_loop(a, c, a.input, a.rounds, a.max_tokens, "rag")
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("concurrency", "ok", "requests_per_min", "ttft_p50_s", "ttft_p95_s",
                                             "decode_tok_s_per_stream_p50", "errors")}), flush=True)
    return {"input_tokens": a.input, "max_tokens": a.max_tokens, "rounds_per_worker": a.rounds,
            "mode": "closed loop: N workers start together, each sends rounds_per_worker requests back to back",
            "rows": rows}


def phase_capacity(a):
    rows = []
    target = a.context - a.max_tokens
    for c in a.conc:
        prompts = [build_prompt(a.base, a.model, target, "cap|%d|%d|%d" % (a.context, c, i))[0] for i in range(c)]
        mp = MetricsPoller(a.base)
        before = mp.scrape(mp.url)
        mp.start()
        res, t0 = burst(a.base, a.model, prompts, a.max_tokens)
        mp.stop.set()
        mp.join()
        after = mp.scrape(mp.url)
        row = {"context": a.context, "input_tokens_target": target, "concurrency": c, "max_tokens": a.max_tokens}
        row.update(summarize_burst(res, t0))
        row["server_metrics"] = mp.summary(before, after)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("context", "concurrency", "ok", "ttft_p50_s", "ttft_p95_s",
              "ttft_max_s", "aggregate_prefill_tok_s", "decode_tok_s_per_stream_p50", "server_metrics", "errors")}),
              flush=True)
        if row["failed"] == c:
            print("all failed at concurrency %d; stopping ladder" % c, flush=True)
            break
    return {"context": a.context, "max_tokens": a.max_tokens,
            "mode": "burst: N users each send one unique context-sized request at the same instant", "rows": rows}


def steady_level(a, conc, rep, record=True, window=None, ramp=None):
    window = a.window if window is None else window
    ramp = a.ramp if ramp is None else ramp
    paras = corpus_paragraphs()
    bases = a.base.split(",")
    words_per_para = 85.0
    npar = max(1, int(round(a.input / (words_per_para * 1.25))))
    results = []
    lock = threading.Lock()
    stop_at = [None]
    bar = threading.Barrier(conc + 1)

    def prompt(w, k):
        uid = hashlib.sha256(("%s|steady|%d|%d|%d|%d" % (RUN_SEED, conc, rep, w, k)).encode()).hexdigest()[:16]
        off = int(uid, 16) % (len(paras) - npar - 1)
        return "Request %s.\n\n" % uid + "".join(paras[off:off + npar]) + (STEADY_INSTR_CODE if a.content == "code" else STEADY_INSTR)

    def worker(w):
        bar.wait()
        k = 0
        while time.monotonic() < stop_at[0]:
            r = stream_one(bases[w % len(bases)], a.model, prompt(w, k), a.max_tokens)
            r["worker"] = w
            r.pop("text", None)
            with lock:
                results.append(r)
            k += 1
    mp = MetricsPoller(a.base)
    ths = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
    for t in ths:
        t.start()
    t0 = time.monotonic()
    stop_at[0] = t0 + ramp + window
    bar.wait()
    time.sleep(ramp)
    before = mp.scrape(mp.url)
    mp.start()
    w0_epoch = time.time()
    w0 = time.monotonic()
    time.sleep(window)
    w1 = time.monotonic()
    w1_epoch = time.time()
    after = mp.scrape(mp.url)
    mp.stop.set()
    for t in ths:
        t.join()
    mp.join()
    if not record:
        return None
    toks, ok, errs, ttft, dec, ptoks = 0.0, 0, [], [], [], []
    for r in results:
        if r["error"]:
            errs.append(r["error"])
            continue
        if r["ttft_s"] is None:
            continue
        first = r["t0"] + r["ttft_s"]
        last = r["t0"] + r["end_s"]
        n = r["completion_tokens"] or 0
        if last > first and n:
            ov = max(0.0, min(last, w1) - max(first, w0))
            toks += n * ov / (last - first)
        if w0 <= r["t0"] <= w1:
            ttft.append(r["ttft_s"])
            if r.get("prompt_tokens"):
                ptoks.append(r["prompt_tokens"])
        if r.get("decode_tok_s") and first < w1 and last > w0:
            dec.append(r["decode_tok_s"])
        ok += 1
    agg = toks / (w1 - w0)
    row = {"concurrency": conc, "rep": rep, "window_s": r3(w1 - w0, 1), "ramp_s": ramp,
           "window_start_epoch": r3(w0_epoch, 1), "window_end_epoch": r3(w1_epoch, 1),
           "requests": len(results), "ok": ok, "failed": len(errs),
           "prompt_tokens_p50": pct(ptoks, 50), "max_tokens": a.max_tokens,
           "aggregate_output_tok_s": r3(agg, 1),
           "per_user_tok_s_p50": r3(pct(dec, 50), 2), "per_user_tok_s_p5": r3(pct(dec, 5), 2),
           "ttft_p50_s": r3(pct(ttft, 50)), "ttft_p95_s": r3(pct(ttft, 95)),
           "ttft_max_s": r3(max(ttft)) if ttft else None,
           "server_metrics": mp.summary(before, after), "errors": errs[:5]}
    return row


def phase_steady(a):
    rnd = random.Random(a.order_seed)
    print(json.dumps({"warmup": a.conc}), flush=True)
    for c in a.conc:
        burst(a.base, a.model, ["Warmup %d-%d. Say hello." % (c, i) for i in range(c)], 16)
    rows = []
    discovery = []
    if a.stop_below:
        # ceiling discovery: one ascending pass (short windows), keep levels up to and including the
        # first whose per-user p50 falls below --stop-below or that has errors; then 3 randomized reps
        keep = []
        for c in sorted(a.conc):
            row = steady_level(a, c, -1, window=max(30, a.window / 2), ramp=a.ramp)
            discovery.append(row)
            keep.append(c)
            pu = row["per_user_tok_s_p50"]
            print(json.dumps({"discovery": c, "agg": row["aggregate_output_tok_s"], "per_user": pu,
                              "failed": row["failed"]}), flush=True)
            tt = row["ttft_p95_s"]
            if row["failed"] or (pu is not None and pu < a.stop_below) or (tt is not None and tt > a.stop_ttft):
                break
        a.conc = keep
    for rep in range(a.reps):
        order = list(a.conc)
        rnd.shuffle(order)
        for c in order:
            row = steady_level(a, c, rep)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("concurrency", "rep", "ok", "failed", "aggregate_output_tok_s",
                  "per_user_tok_s_p50", "ttft_p50_s", "ttft_p95_s")} | {"spec": (row["server_metrics"].get("spec_decode") or {}).get("tokens_per_verify_cycle")}),
                  flush=True)
    summ = []
    for c in a.conc:
        v = [r["aggregate_output_tok_s"] for r in rows if r["concurrency"] == c]
        u = [r["per_user_tok_s_p50"] for r in rows if r["concurrency"] == c and r["per_user_tok_s_p50"]]
        t = [r["ttft_p95_s"] for r in rows if r["concurrency"] == c and r["ttft_p95_s"] is not None]
        med = statistics.median(v) if v else None
        cv = (statistics.pstdev(v) / statistics.mean(v)) if len(v) > 1 and statistics.mean(v) else None
        summ.append({"concurrency": c, "reps": len(v), "aggregate_tok_s_median": r3(med, 1),
                     "aggregate_tok_s_min": r3(min(v), 1) if v else None,
                     "aggregate_tok_s_max": r3(max(v), 1) if v else None,
                     "cv": r3(cv, 3), "cv_flag": bool(cv and cv > 0.10),
                     "per_user_tok_s_median": r3(statistics.median(u), 2) if u else None,
                     "ttft_p95_s_median": r3(statistics.median(t), 3) if t else None})
    return {"mode": "steady-state closed loop, windowed token count", "input_tokens_approx": a.input,
            "content": a.content, "instruction": (STEADY_INSTR_CODE if a.content == "code" else STEADY_INSTR).strip(),
            "max_tokens": a.max_tokens, "window_s": a.window, "ramp_s": a.ramp, "reps": a.reps,
            "order_seed": a.order_seed, "warmup": "one unrecorded burst per level (max_tokens 16)",
            "replica_endpoints": a.base.split(","), "stop_below_per_user_tok_s": a.stop_below, "stop_ttft_p95_s": a.stop_ttft,
            "discovery_rows": discovery, "summary": summ, "rows": rows}


def phase_gate(a):
    checks = [("What is the capital of France? Answer in one word.", ["paris"]),
              ("What is 17*23? Answer with the number only.", ["391"]),
              ("List the first five prime numbers, comma separated.", ["2", "3", "5", "7", "11"]),
              ("In one sentence, what does a network firewall do?", ["traffic"]),
              ("What is 8347*291? Answer with the number only.", ["2428977"]),
              ("What is 127*43? Answer with the number only.", ["5461"])]
    rows = []
    for q, want in checks:
        r = post(a.base, "/chat/completions", {"model": a.model, "messages": msgs(q), "max_tokens": a.gate_tokens,
                                               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        txt = r["choices"][0]["message"].get("content") or ""
        ans = txt.split("</think>")[-1]
        rows.append({"q": q, "answer": ans.strip()[:200], "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                     "pass": all(w in ans.lower().replace(",", "") if w.isdigit() and len(w) > 3 else w in ans.lower()
                                 for w in want)})
    for L in a.needles:
        needle = ("The vault access code is 48213-KESTREL.", "What is the vault access code mentioned in the "
                  "document? Answer with the code only.")
        t, n, uid, off = build_prompt(a.base, a.model, L, "needle|%d" % L, needle=needle)
        r = stream_one(a.base, a.model, t, a.needle_tokens)
        rows.append({"q": "needle at %d tokens (middle)" % L, "prompt_tokens": r["prompt_tokens"],
                     "answer": r["text"][:200], "ttft_s": r3(r["ttft_s"]), "error": r["error"],
                     "pass": "48213" in r["text"].split("</think>")[-1] and "KESTREL" in r["text"].split("</think>")[-1].upper()})
    for x in rows:
        print(json.dumps(x), flush=True)
    return {"rows": rows, "all_pass": all(x["pass"] for x in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("phase", choices=["gate", "prefill", "rag", "capacity", "steady"])
    p.add_argument("--lengths", default="1024,4096,16384,65536,131040")
    p.add_argument("--conc", default=None)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--input", type=int, default=8192)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--label", default=None, help="key under phases (default: phase name)")
    p.add_argument("--run-seed", default=None, help="override RUN_SEED so prompts differ from earlier runs")
    p.add_argument("--window", type=float, default=120)
    p.add_argument("--stop-ttft", type=float, default=10.0,
                   help="steady discovery also stops at the first level whose TTFT p95 exceeds this (queueing = saturated)")
    p.add_argument("--stop-below", type=float, default=None,
                   help="steady: ascending discovery pass, stop the ladder at the first level whose per-user rate < this")
    p.add_argument("--ramp", type=float, default=20)
    p.add_argument("--order-seed", type=int, default=1)
    p.add_argument("--gate-tokens", type=int, default=64)
    p.add_argument("--needle-tokens", type=int, default=24)
    p.add_argument("--needles", default="16384,120000")
    p.add_argument("--content", choices=["prose", "code"], default="prose",
                   help="steady: prose report (default, unchanged) or structured code output")
    a = p.parse_args()
    global RUN_SEED
    if a.run_seed:
        RUN_SEED = a.run_seed
    a.needles = [int(x) for x in a.needles.split(",") if x]
    a.lengths = [int(x) for x in a.lengths.split(",")]
    defaults = {"prefill": ("1,4,8", 32), "rag": ("1,2,4,8,16,32", 256), "capacity": ("1,4,8,16,24,32,40", 256),
                "gate": ("1", 24), "steady": ("1,4,8,16,32", 512)}
    a.conc = [int(x) for x in (a.conc or defaults[a.phase][0]).split(",")]
    a.max_tokens = a.max_tokens or defaults[a.phase][1]
    t_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if a.phase == "steady" and a.input == 8192:
        a.input = 256
    res = {"gate": phase_gate, "prefill": phase_prefill, "rag": phase_rag, "capacity": phase_capacity,
           "steady": phase_steady}[a.phase](a)
    res["started_at"] = t_start
    res["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    doc = json.load(open(a.out)) if os.path.exists(a.out) else {}
    doc.setdefault("base", a.base)
    doc.setdefault("model", a.model)
    res["run_seed"] = RUN_SEED
    doc["prompt_construction"] = {"corpus_seed": CORPUS_SEED, "corpus_words": CORPUS_WORDS, "run_seed": RUN_SEED,
                                  "corpus_token_count": len(_ids_cache.get("ids", [])) or None,
                                  "corpus_token_ids_sha256": _ids_cache.get("sha"),
                                  "tail_instruction": INSTR.strip(), "chat_template_kwargs": {"enable_thinking": False},
                                  "sampling": {"temperature": 0, "ignore_eos": True}}
    doc.setdefault("phases", {})[a.label or a.phase] = res
    json.dump(doc, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
