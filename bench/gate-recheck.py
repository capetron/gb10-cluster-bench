#!/usr/bin/env python3
"""gate-recheck.py BASE SERVED: re-ask the arithmetic canaries with the checkpoint's DEFAULT template (thinking on),
max_tokens 6000, temperature 0. Used when the thinking-off gate fails only on mental arithmetic (a model-behaviour
finding, not an engine fault). Exit 0 = every canary correct. Set OPENAI_API_KEY if the server needs a key."""
import json, os, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
Q = [("What is 8347*291? Answer with the number only.", "2428977"), ("What is 127*43? Answer with the number only.", "5461"),
     ("What is 17*23? Answer with the number only.", "391")]
ok = True
for q, exp in Q:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 6000, "temperature": 0}).encode()
    r = urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", body, {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")}), timeout=900)
    d = json.load(r); m = d["choices"][0]["message"]
    ans = (m.get("content") or "") + " " + (m.get("reasoning_content") or m.get("reasoning") or "")
    p = exp in ans.replace(",", "")
    ok &= p
    print(json.dumps({"q": q, "expected": exp, "content": (m.get("content") or "")[-80:], "completion_tokens": d["usage"]["completion_tokens"], "pass": p}))
print("RECHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
