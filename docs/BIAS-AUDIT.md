# Bias audit: what went wrong in our first benchmark round

Before the method in [METHODOLOGY.md](METHODOLOGY.md) existed, we ran a first round of GB10, RTX PRO 6000,
GB300 and H200 benchmarks with a simpler burst harness (`bench/llm-concurrency-knee.py`). An audit
of that corpus found the biases below, ranked by how far each one could move a published
conclusion. Every fix in the current harness traces back to a row here. We publish the list
because the same mistakes are easy to make on any cluster.

| # | Bias | Who it flattered | How big |
|---|---|---|---|
| 1 | Comparing all platforms at a fixed 32 streams, although load per unit and per-user speed differed | Small GB10 configurations in per-unit and per-dollar terms | Reversed two headline conclusions |
| 2 | Ladders cut short, and unequal `max_num_seqs` caps (32 on some hosts, 64 or 96 on others) | Platforms whose cap sat near 32 | One GB300 configuration was +55% at 64 streams over 32; "peaks" were not comparable |
| 3 | Speculative decoding inconsistent across hosts and mostly unrecorded | Spec-on runs at low concurrency | Moves single-stream ratios; gain is largest at batch 1 |
| 4 | Burst method: one closed batch per level, no warmup, natural stop | Platforms with low, stable TTFT | 23% run-to-run swing at 32 streams and 66% at 16 on the same GB10 TP8 configuration; about one outlier level per ladder even on the GB300 (31% at 8 streams) |
| 5 | Engine, image and mode mismatch (community images on one side, NGC on the other, eager mode on some) | Whichever side had the better build | Eager mode alone cost 2.4x on one RTX PRO 6000 configuration |
| 6 | Reasoning tokens counted as delivered output | Models that think by default | Overstated the user-visible answer rate |
| 7 | Precision mismatch (NVFP4 vs FP8 vs BF16) | In decode speed: whichever reads fewer bytes | Any FP8-vs-NVFP4 ratio mixes a precision choice with a hardware difference |
| 8 | Decode-only prompts (about 40 input tokens) | Hardware weak at prefill (GB10) | Hid TTFT; long-prompt runs on one RTX PRO 6000 fell to about 330 tok/s aggregate with 11-106 s TTFT |
| 9 | Output length not fixed | Models that run to the cap | Small; per-request counts were not stored, so it could not be checked afterwards |
| 10 | Nodes with different swap and OOM-daemon config | Neither side; added instability | One node was killed first by earlyoom in six consecutive launches, misread as model memory |
| 11 | Mixed GB10 SKUs in the scaling series | Unknown | Low |
| 12 | Prompts unique within a run but identical across runs | Reruns against a warm server with prefix caching on | Large for long padded prompts |
| 13-14 | KV dtype and `max_model_len` differences | Negligible at short context | Matter for capacity, not decode |
| 15 | Shared or production hosts with other load | Unknown, usually against the shared host | Not recorded; needs a load snapshot per run |
| 16 | Missing metadata (no engine args, no spec config, no image digest) | No direction | Made the other biases uncheckable |

## What changed in the harness

1. **Steady-state windows instead of bursts** (`llm-prefill-bench.py steady`): N requests kept in
   flight for a fixed window after a ramp; only tokens inside the window count.
2. **Fixed output length**: `ignore_eos` plus a fixed `max_tokens`, temperature 0, recorded.
3. **Warmup**: one unrecorded pass at every level, so CUDA-graph capture and JIT compilation do
   not land in the first measured level.
4. **Unique prompts across runs**: a run seed in every prompt header.
5. **Ladder to saturation** with a ceiling rule (10 tok/s per user, TTFT p95 10 s) instead of a
   fixed stream count.
6. **Prefill measured separately** at 8k to 128k, and prose vs code content reported separately.
7. **Launch records read from the engine** (`vllm-launch-record.py`) in every result file.
8. **Node parity checks** (`node/parity-check.sh`, `node/verify-versions.sh`,
   `node/gpu-burn.sh`) before any multi-node run.

## Lessons from the GB10 cluster specifically

- **Firmware changes move numbers.** After a firmware and OS update window our 15 s fp16 burn
  dropped about 11% on all eight GB10s at equal or higher SM clocks, while serving decode on the
  same image got 4-8% faster. Decode on GB10 is memory-bandwidth bound and the burn is compute
  bound. Record firmware with every result (`node/verify-versions.sh`) and never mix pre- and
  post-update numbers in one comparison without saying so.
- **Check the kernel command line on every node.** One of our eight nodes was missing NVIDIA's
  `kho=off` hotfix (`node/install-kho-hotfix.py`), and it was rank 0 of every TP4 and TP8 run.
- **A multi-node vLLM job that loses a rank hangs, it does not fail.** Watch every rank's
  container (`cluster/watch-serve.sh`), dump `docker logs` before `docker rm`, and wait for RDMA
  memory regions to clear between launches (`cluster/wait-rdma-clear.sh`).
- **earlyoom kills with SIGTERM and logs to the journal, not dmesg.** No Python traceback, no
  kernel OOM line: check `journalctl -u earlyoom` on every node before blaming the model.
- **Wall power needs a verified outlet map.** Six of our PDU's controller outlet labels were
  wrong; verify each outlet with a coded load before trusting per-node watts.
