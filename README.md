# gb10-cluster-bench

Benchmark and maintenance tooling for NVIDIA GB10 clusters (DGX Spark and OEM GB10 systems),
with a bias-aware method, the exact launch recipes, and our raw results.

## The problem it solves

Most published GB10 numbers are single-user decode on short prompts, often on code-like text
where speculative decoding shines. That shape hides the two things that decide whether a cluster
of these boxes is worth buying: prefill (time to first token on long prompts), and how many users
it serves before per-user speed collapses. It also hides the cluster itself: nodes on different
firmware, one node without swap, a missing kernel hotfix on rank 0. Each of those moved our
numbers more than the thing we were trying to measure.

This repo is the harness we built after auditing our own first round of results
([docs/BIAS-AUDIT.md](docs/BIAS-AUDIT.md)), plus the node checks and the maintenance tool that
keep an eight-node pod uniform enough to benchmark.

## What is in it

| Path | What it does |
|---|---|
| `bench/llm-prefill-bench.py` | The harness. Phases: `gate` (correctness canaries and 16k/120k needles), `steady` (windowed closed-loop decode ladder, repeats in shuffled order, prose or code content), `prefill` (TTFT at exact input lengths, prefix cache defeated), `rag`, `capacity`. Any OpenAI-compatible server; stdlib only. |
| `bench/speed.sh` | The fixed battery every candidate gets: gate, prose and code ladders, ceiling discovery at a 10 tok/s per-user floor, prefill at 8k/32k/128k, optional wall-power join. |
| `bench/speedgate.sh`, `bench/gate-recheck.py` | Gate policy: arithmetic misses with thinking off get a thinking-on recheck; a failed needle withholds only the contexts it covers. |
| `bench/vllm-launch-record.py` | Reads every launch setting back from a running vLLM container (image digest, env, argv, engine log decisions) into the result file. |
| `bench/llm-concurrency-knee.py` | Quick cold-burst probe. Useful for a first look; not for publication (see the bias audit). |
| `node/gpu-burn.sh` | 15 s fp16 matmul burn plus a memory-copy probe, with clocks and power sampled: catches a clock-latched unit before it skews a run. |
| `node/verify-versions.sh` | Read-only table of kernel, driver, VBIOS, EC and SoC firmware, pending updates and fabric NICs per node. |
| `node/parity-check.sh` | Read-only swap and OOM-daemon parity across nodes. |
| `node/gpu-clock-lock.service`, `node/install-clock-lock.sh` | Boot-time GPU clock lock (`nvidia-smi -lgc 300,2200`) that stopped our long-prefill power-offs, with an idempotent installer and `--check`, `--verify` (burn test) and `--uninstall`. See [docs/power-offs-and-clock-lock.md](docs/power-offs-and-clock-lock.md). |
| `node/memfrag-check.sh` | Read-only FRESH / FRAGMENTED verdict per node: has an engine run since boot left the small free blocks that cost 3-5% decode? See [docs/memory-bandwidth-reboot-before-launch.md](docs/memory-bandwidth-reboot-before-launch.md). |
| `node/install-kho-hotfix.py` | Installs NVIDIA's `kho=off` GB10 kernel hotfix on one node, reboots, verifies. |
| `node/fwupd-pending-list.py` | Pending firmware updates, one line per device. |
| `cluster/watch-serve.sh` | Waits for a multi-node vLLM API and fails fast the moment any rank dies (a lost rank otherwise hangs forever). |
| `cluster/wait-rdma-clear.sh` | Blocks until no node holds RDMA memory regions, between teardown and relaunch. |
| `power/pdu-sample-unifi.sh`, `power/pdu-join.py` | Read-only wall-power sampler for a UniFi PDU, and the join that puts watts and tokens per joule on every benchmark row. The join works with any sampler that writes the same JSONL. |
| `fleet-maint/` | Deterministic OS and firmware maintenance for a GPU fleet: read-only collector, policy classes, approval-gated applier with canary, busy detection, maintenance windows and post-update verification. No model makes any decision. 78 unit tests. Busy detection reads running containers, GPU utilization and Ollama by default; the optional `lab_hub` hook also asks a job-scheduler HTTP endpoint (`/api/services`, `/api/jobs` returning JSON) whether a node holds a GPU lock or a live job. Leave it unset and set `busy.llm_lab: false` if you have no scheduler. |
| `recipes/` | The vLLM launch commands and NCCL environment behind every result. |
| `results/` | Raw result JSONs and a summary table. |
| `docs/` | [Methodology](docs/METHODOLOGY.md), the [bias audit](docs/BIAS-AUDIT.md), and the two stability and bandwidth write-ups below. |

## Quick start

```
git clone https://github.com/capetron/gb10-cluster-bench && cd gb10-cluster-bench
./setup.sh                       # checks prerequisites, runs the offline tests

# 1. node health (hosts are ssh aliases with key-based login)
export CLUSTER_HOSTS="node1 node2 node3 node4"
node/verify-versions.sh          # firmware and driver identical?
node/parity-check.sh             # swap and OOM daemon identical?
node/install-clock-lock.sh --check   # clock lock active? (see docs/power-offs-and-clock-lock.md)
node/memfrag-check.sh            # FRESH? if FRAGMENTED, reboot before launching
ssh node1 'bash -s' < node/gpu-burn.sh

# 2. the battery against a running engine (OpenAI-compatible)
bench/speedgate.sh out.json http://node1:8000/v1 my-model 131072 run1 run1.log
#    or step by step:
bench/llm-prefill-bench.py http://node1:8000/v1 my-model out.json gate
bench/llm-prefill-bench.py http://node1:8000/v1 my-model out.json steady --conc 1,4,8,16,32
bench/llm-prefill-bench.py http://node1:8000/v1 my-model out.json prefill --lengths 8192,32768,131040 --conc 1

# 3. record what the engine actually ran with
bench/vllm-launch-record.py node1 my-vllm-container out.json main
```

Set `OPENAI_API_KEY` if your server requires a key. `speed.sh` joins wall power when
`PDU_TRACE` and `PDU_OUTLETS` are set (see `power/`).

## Hardware we ran it on

- 8x GB10: 4x NVIDIA DGX Spark and 4x MSI EdgeXpert, one 200G ConnectX-7 port each into a
  switched RoCE fabric, tested as 1, 2 (TP2), 4 (TP4) and 8 (TP8, and 4x TP2 replicas) units.
- NVIDIA DGX Station GB300 (as a prefill and decode reference).
- NVIDIA RTX PRO 6000 Blackwell Max-Q, one card (burst reference).

## Results (provisional)

Qwen3.8 on GB10 with vLLM nightly, steady state, 512-token outputs, temperature 0. Full tables,
prefill and caveats in [results/README.md](results/README.md).

| Configuration | 1 user tok/s, prose / code | Users at 10 tok/s floor (aggregate tok/s) | Prefill at 128k, tok/s |
|---|---|---|---|
| 1x GB10, Qwen3.8-27B NVFP4, MTP-3 | 23.6 / 31.4 | 16 (222) | 1,302 |
| 2x GB10 TP2, Qwen3.8-Flash-Next NVFP4, 6-seq speed profile | 37.5 / 53.1 | 4 (99) | 3,300 |
| 4x GB10 TP4, Qwen3.8-Flash-Next NVFP4, MTP-4 | 49.4 / 75.1 | 64 (675) | 3,764 |
| 8x GB10 as 4x TP2 replicas, same model | 36.0 / - | 128 (1,365) | - |
| 8x GB10 TP8, same model | 48.5 / - | 64 (816) | 3,218 |
| 1x DGX Station GB300, Qwen3.8-Flash-Next NVFP4 | - | - | 46,527 |

What the data says so far: GB10 clusters scale users well and single-user speed modestly;
replicas beat one large tensor-parallel group for many users; prefill is the GB10's weak point
(one GB300 prefills 12-15x faster than four GB10s at 16k-128k input); and speculative decoding is
33-52% faster on code than on prose, so check which one a headline number used.

## Stability and bandwidth findings (2026-09)

- [Power-offs at long-prefill onset](docs/power-offs-and-clock-lock.md): hot GB10 units switched
  themselves off, with nothing in the logs, a few seconds into a 131k-token prefill. Locking the
  GPU clock at 300-2200 MHz took one unit from 2 losses in 2 to 0 in 6 on the same sequence, then
  seven loaded units ran 66 long-input onsets with no loss, at 1-3% single-user decode on the
  test unit. A workaround pending NVIDIA, not a proven root cause.
- [Reboot before you launch](docs/memory-bandwidth-reboot-before-launch.md): identical units read
  262 GB/s or 238 GB/s depending on how many engines had run since boot, because the driver
  hands the next engine the small free blocks the last one left. A reboot restored 262 GB/s and
  4-5% decode; `drop_caches`, `compact_memory` and `cma=128M` did not.
- Blog write-up: https://petronellatech.com/blog/dgx-spark-shutdown-under-load-our-8-unit-gb10-diagnosis/
- Firmware field report: https://petronellatech.com/blog/dgx-spark-firmware-update-what-we-learned-on-8-gb10-units/

## Method in one paragraph

Correctness gate before any speed number. Steady-state windows, not bursts: N users each keep one
request in flight, only tokens produced inside the window count, one warmup per level, three
repeats in shuffled order. Fixed output length (`ignore_eos`), temperature 0, thinking off. A unique
header on every prompt so the prefix cache cannot hit, with the hit counters recorded to prove it.
Prose and code reported separately. Ladders run to saturation on every platform. Launch settings
read from the engine. Wall power per outlet. Details and sources in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Related

- [gb10-cluster-guide](https://github.com/capetron/gb10-cluster-guide): cabling and networking two,
  three and four-plus GB10 nodes.
- [gb10-cluster-check](https://github.com/capetron/gb10-cluster-check): read-only ConnectX-7 fabric
  check for GB10 clusters.
- [Self-hosted LLM benchmarks for private AI](https://petronellatech.com/ai/llm-benchmarks/)
  (Petronella Technology Group, Inc.)
- [NVIDIA DGX Spark](https://petronellatech.com/hardware/dgx-spark/): the hardware, and our notes
  on running it (Petronella Technology Group, Inc.)
- [DGX Spark cluster cable](https://petronellatech.com/hardware/dgx-spark-cluster-cable/): the
  0.5 m QSFP112 DAC we use between GB10 nodes.

Disclosure: Petronella Technology Group, Inc. sells AI hardware, including GB10 systems and
cluster cabling. That is why the scripts, launch commands and raw results are all here: rerun them
on your own hardware and tell us where your numbers differ.

## Contributing

Issues and pull requests are welcome, especially result files from other GB10 configurations.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE). Maintained by Petronella Technology Group, Inc.
