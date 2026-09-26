# Results

Raw result files from our lab, measured 2026-09-23 to 2026-09-25 with the harness in `bench/`.
Hostnames and addresses were replaced with role names (`dgx-spark-N`, `edgexpert-N`,
`gb300-station`, `rtx-pro-6000-server`, fabric `192.168.100.x`); internal quality-evaluation blocks were
removed. Everything else is as the harness wrote it: every row, every repeat, the prompt corpus
hash, the server metrics and the full launch records.

**Status: provisional.** These runs were part of a larger contest that was paused for a firmware
and OS update window. Pre-update and post-update numbers are labeled. Configurations that never
ran, or whose correctness gate failed, are not in this table (a failed gate means no speed
numbers, by rule).

## Hardware

| Unit | What it is | Memory |
|---|---|---|
| `dgx-spark-1..4` | NVIDIA DGX Spark (GB10) | 128 GB LPDDR5x unified, 273 GB/s |
| `edgexpert-1..4` | MSI EdgeXpert MS-C931 (GB10) | same GB10 platform |
| `gb300-station` | NVIDIA DGX Station GB300 (MSI XpertStation WS300) | 252 GB HBM3e GPU + 496 GB LPDDR5X CPU |
| `rtx-pro-6000-server` | NVIDIA RTX PRO 6000 Blackwell Max-Q (one 300 W card used) | 96 GB GDDR7 |

GB10 fabric: one 200G ConnectX-7 port per node into a switched RoCE plane, MTU 9000. All eight
GB10s on kernel 7.0.0-1019-nvidia and driver 580.178.04. Memory figures are NVIDIA's published
specifications.

## Decode, steady state (512-token outputs, temperature 0, thinking off, 3 repeats, shuffled order)

Per-user tok/s at 1 user; aggregate tok/s at 32 users; "floor" = most users measured with at least
10 tok/s per user and TTFT p95 of 10 s or less, with the aggregate there. Wall power is the sum of
the PDU outlets feeding the engine's nodes (switch excluded).

| Configuration | 1 user prose / code | 32 users prose / code | Users at floor (aggregate) | Wall W at 32 users | tok/J at 32 users (prose) | File |
|---|---|---|---|---|---|---|
| 1x GB10, Qwen3.8-27B NVFP4, MTP-3 (pre-update) | 23.6 / 31.4 | 303.2 / 402.7 | 16 (222.0) | 145 | 2.09 | `2026-09-25-gb10-1x-qwen3.8-27b-nvfp4-mtp3.json` |
| 1x GB10, same, post firmware update | 25.3 / - | 315.0 / - | 32 (315.0) | 154 | 2.04 | `...-mtp3-post-firmware.json` |
| 2x GB10 TP2, Qwen3.8-Flash-Next NVFP4, 6-seq speed profile | 37.5 / 53.1 | 130.7 / 176.7 | 4 (99.4) | 325 | 0.41 | `2026-09-25-gb10-2x-tp2-...-speed-profile.json` |
| 4x GB10 TP4, Qwen3.8-Flash-Next NVFP4, MTP-4, 128 seqs | 49.4 / 75.1 | 491.1 / 689.0 | 64 (674.6) | 647 | 0.76 | `2026-09-25-gb10-4x-tp4-...-mtp4.json` |
| 8x GB10 as 4 TP2 replicas, same model and flags | 36.0 / - | 623.8 / - | 128 (1,364.5) | 1,312 | 0.48 | `2026-09-25-gb10-8x-...-tp8-vs-4x-tp2.json` |
| 8x GB10 TP8, same model and flags | 48.5 / - | 626.0 / - | 64 (815.9) | 1,200 | 0.52 | same file |

Readings:

- **Speculative decoding gains depend on content.** Every MTP configuration measured on both
  was 33-52% faster on code than on prose for one user. A headline number measured on code-like text overstates prose speed.
- **The TP2 "speed" profile is a single-user profile.** Six sequences by design: requests queue
  from 8 users, and the aggregate stays flat at about 131 tok/s (prose).
- **For many users, replicas beat one big tensor-parallel group.** Eight GB10s as four TP2
  replicas held 128 users at the floor (1,364 tok/s) against 64 users (816 tok/s) as one TP8
  group, and delivered more tokens per joule from 64 users up (0.70 vs 0.68 at 64 users; 1.02 at
  128). TP8 wins single-user speed (48.5 vs 36.0) and is slightly more efficient at 32 users.
- **The firmware update changed numbers.** After the update window the single-node decode was
  4-8% faster, while a compute-bound fp16 burn on the same nodes dropped about 11%
  (docs/BIAS-AUDIT.md). Do not mix pre- and post-update rows.

## Prefill (concurrency 1, unique prompts, prefix cache defeated, median of 3)

Prefill tok/s = prompt tokens / time to first token.

| Configuration | 1k | 4k | 16k | 64k | 128k | File |
|---|---|---|---|---|---|---|
| 2x GB10 TP2, Qwen3.8-Flash-Next NVFP4 | 2,258 | 2,970 | 3,134 | 2,992 | 2,812 | `2026-09-24-gb10-2x-tp2-...-prefill.json` |
| 4x GB10 TP4, same | 2,730 | 3,474 | 3,538 | 3,362 | 3,145 | `2026-09-24-gb10-4x-tp4-...-prefill.json` |
| 8x GB10 TP8, same | 2,682 | 3,578 | 3,678 | 3,468 | 3,218 | `2026-09-24-gb10-8x-tp8-...-prefill.json` |
| 1x DGX Station GB300, same model, TP1 | 8,048 | 15,019 | 41,761 | 44,522 | 46,527 | `2026-09-24-gb300-station-...-prefill.json` |

These four runs used byte-identical prompts (same corpus seed, run seed and token-id sha256).
Prefill is compute bound, and it is where the GB10 is weakest: adding GB10s past two barely moves
single-request prefill, while one GB300 is about 12-15x faster than four GB10s from 16k to 128k
input. A 128k-token prompt takes about 2.8 s to first token on the GB300 and about 42 s on four
GB10s. On the steady-battery runs above, 1x GB10 prefills 2,400 / 2,114 / 1,302 tok/s at 8k /
32k / 128k and 4x GB10 TP4 3,952 / 3,948 / 3,764.

## RTX PRO 6000 reference (burst method, not directly comparable)

`2026-09-23-rtx-pro-6000-maxq-qwen3.8-27b-fp8-burst.json`: Qwen3.8-27B FP8 on one RTX PRO 6000
Blackwell Max-Q (300 W), TP1, MTP-2, max 32 sequences,
measured with the older burst probe (`bench/llm-concurrency-knee.py`, about 40-token prompts,
256-token outputs, one burst per level). 80 tok/s at 1 user and about 1,358 aggregate at 32, the knee (48 users queue behind the
32-sequence cap). It
is here as a reference point and as an example of the burst format; docs/BIAS-AUDIT.md explains why
burst numbers should not be ranked against steady-state numbers.

## Not measured yet

The steady-state battery on the RTX PRO 6000 and on H200 (our H200 runs so far lack the
launch metadata this repo requires, so none are published), GLM-5.3-Flash speed on GB10, DeepSeek-V4-Flash-Vision on GB10, and the
post-update rerun of every GB10 lane. They will be added as result files when they exist.

## Cost basis

NVIDIA list price for a DGX Spark has been $4,699 since February 2026 (hardware only; street
prices vary with market conditions; OEM GB10 systems are priced by their makers). The DGX Station
GB300 has no published list price, so no per-dollar figure is given for it.
