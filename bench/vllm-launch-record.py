#!/usr/bin/env python3
"""vllm-launch-record.py HOST CONTAINER [RESULT.json KEY]

Capture every launch setting of a running vLLM container from the ENGINE (not retyped), per the
metadata rule in docs/BIAS-AUDIT.md: image + digest, container env (NCCL etc.), serve argv, and the engine
log lines that state what vLLM actually did (non-default args, engine config, MoE/attention
backend, all-reduce backend, KV cache size, CUDA-graph mode / downgrades, spec-decode config).
Prints JSON; with RESULT.json KEY stores it under doc["launch_records"][KEY].
HOST is anything `ssh` accepts (key-based, BatchMode), or "local" to run docker on this machine.
"""
import json
import re
import subprocess
import sys


def sh(host, cmd):
    argv = ["bash", "-c", cmd] if host == "local" else ["ssh", "-o", "BatchMode=yes", host, cmd]
    return subprocess.run(argv, capture_output=True, text=True).stdout


def main():
    host, name = sys.argv[1], sys.argv[2]
    ins = json.loads(sh(host, "docker inspect %s" % name) or "[{}]")[0]
    log = sh(host, "docker logs %s 2>&1 | grep -v 'GET /\\|POST /' | head -400" % name)
    env = [e for e in ins.get("Config", {}).get("Env", []) if re.match(r"(NCCL|VLLM|GLOO|TP_|TORCH|PYTORCH|CUDA|FLASHINFER|HF_|SPEC|MAX_|GPU_)", e)]

    def grab(pat, n=3):
        return [ln.strip()[:1500] for ln in log.splitlines() if re.search(pat, ln)][:n]
    eng = grab(r"Initializing a V1 LLM engine", 1)
    fields = {}
    if eng:
        for k in ("speculative_config", "enforce_eager", "kv_cache_dtype", "enable_prefix_caching",
                  "enable_chunked_prefill", "tensor_parallel_size", "pipeline_parallel_size",
                  "data_parallel_size", "quantization", "max_seq_len"):
            m = re.search(r"%s=([^,]+(?:\([^)]*\))?)" % k, eng[0])
            if m:
                fields[k] = m.group(1)
        for k in ("cudagraph_mode", "enable_flashinfer_autotune", "moe_backend", "max_cudagraph_capture_size"):
            m = re.search(r"'?%s'?[=:] ?([^,}]+)" % k, eng[0])
            if m:
                fields[k] = m.group(1).strip()
    rec = {"host": host, "container": name,
           "image": ins.get("Config", {}).get("Image"), "image_id": ins.get("Image"),
           "cmd": ins.get("Config", {}).get("Cmd"), "entrypoint": ins.get("Config", {}).get("Entrypoint"),
           "env": env, "started_at": ins.get("State", {}).get("StartedAt"),
           "engine_fields": fields,
           "log_non_default_args": grab(r"non-default args", 1),
           "log_moe_backend": grab(r"MoE backend", 2),
           "log_attention_backend": grab(r"[Aa]ttention backend|Using .*_ATTN|MLA_SPARSE", 3),
           "log_all_reduce": grab(r"all-reduce|allreduce|AllReduce", 3),
           "log_kv_cache": grab(r"Available KV cache memory|GPU KV cache size|Maximum concurrency", 3),
           "log_cudagraph": grab(r"cudagraph_mode|PIECEWISE because|Graph capturing finished|downgrad", 4),
           "log_spec": grab(r"[Ss]peculative|MTP|mtp", 4),
           "log_model_load": grab(r"Model loading took|Loading weights took", 3)}
    out = json.dumps(rec, indent=1)
    if len(sys.argv) > 4:
        import os
        doc = json.load(open(sys.argv[3])) if os.path.exists(sys.argv[3]) else {}
        doc.setdefault("launch_records", {})[sys.argv[4]] = rec
        json.dump(doc, open(sys.argv[3], "w"), indent=1)
    print(out[:3000])


if __name__ == "__main__":
    main()
