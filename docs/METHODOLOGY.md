# Methodology

How the numbers in `results/` were produced, and the rules we hold every platform to. The
short version: gate for correctness first, measure steady state rather than bursts, keep output
length fixed, defeat the prefix cache, report prose and code separately, measure prefill as well
as decode, repeat with shuffled order, and record every launch setting from the engine itself.

## Why decode-only benchmarks mislead

Most quick LLM benchmarks send a prompt of a few dozen tokens and read back a few hundred. That
shape is almost pure decode. Decode is memory-bandwidth bound and prefill is compute bound
(Splitwise, ISCA 2024 [16]; Usami et al. 2026 [19]), so a decode-only suite flatters hardware with
little compute and hides time to first token (TTFT) for RAG, long documents and agent loops. A
DGX Spark has 273 GB/s of LPDDR5x bandwidth [21]; a DGX Station GB300 has 7.1 TB/s of HBM3e [22].
Those two facts alone predict the decode ranking. Prefill, long context and power are where the
interesting differences are, so we measure them.

## The battery (`bench/speed.sh`)

Every candidate gets the same battery, in this order:

1. **Node health first.** `node/gpu-burn.sh` on every unit (clock-latch check),
   `node/parity-check.sh` (swap and OOM daemon identical), `node/verify-versions.sh` (kernel,
   driver, firmware identical). A benchmark across nodes that differ here measures config drift.
2. **Correctness gate** (`llm-prefill-bench.py gate`): four short-context canaries (a capital, a
   product, the first five primes, a definition), two harder arithmetic canaries, and a needle
   retrieval at 16k and (when the context allows) 120k tokens. A failed gate means no throughput
   numbers for that configuration. `bench/speedgate.sh` applies the policy: arithmetic misses with
   thinking off get a thinking-on recheck and are recorded as a quality finding; a needle failure
   withholds only the contexts it covers.
3. **Steady-state decode ladder** at 1, 4, 8, 16 and 32 users, prose content and code content
   run separately. For each level, N workers each keep one request in flight for a ramp plus a
   fixed window; only tokens produced inside the window count (each request's tokens are spread
   linearly between its first and last token). One unrecorded warmup burst per level. Three
   repeats with the level order shuffled per repeat.
4. **Ceiling discovery**: if the top of the ladder still delivers 10 tok/s per user, continue
   to 48, 64, 96, 128, 192 users and stop at the first level whose per-user median drops below
   10 tok/s or whose TTFT p95 exceeds 10 s. The ceiling is reported as "users at the 10 tok/s
   floor" plus the aggregate at that level.
5. **Prefill** at concurrency 1 at 8k, 32k and 128k input tokens, 3 repeats, 32 output tokens.
   Prefill tok/s = `usage.prompt_tokens / TTFT`.
6. **Power** (optional): the wall-power trace of the outlets feeding the engine's nodes is joined
   to every steady row (`power/pdu-join.py`), giving watts and tokens per joule per level.

Fixed request settings for throughput runs: temperature 0, `ignore_eos: true`, a fixed
`max_tokens` (512 for the ladder), thinking disabled through the chat template. With
`ignore_eos` every request produces exactly `max_tokens` tokens, so verbosity cannot change the
result. The steady prompt asks for a long answer so that `ignore_eos` rarely has to force text past
a natural stop, which would distort speculative-decoding acceptance.

## Prompt construction (reproducible, cache-proof)

- A deterministic pseudo-English corpus is generated from a fixed seed over a fixed vocabulary,
  then tokenized **by the server** (`/tokenize`), so token ids come from the model's own
  tokenizer. The sha256 of the corpus token ids is written into every result file, so two runs
  can prove they used byte-identical prompts.
- Each request starts with `Document <uid>.` where the uid is a hash of the run seed, phase,
  length, concurrency, repeat and index. No KV block can be shared between two requests, so the
  prefix cache cannot hit. The server's prefix-cache counters are recorded per row to prove it.
- Prompt length is solved so the fully templated prompt counts exactly the target length by
  `/tokenize`; the server's own `usage.prompt_tokens` is what gets reported.
- `--run-seed` changes every prompt, so a rerun against a warm server cannot hit cached blocks.

## Speculative decoding

Multi-token prediction (MTP), EAGLE-style and draft-model decoding accept far more tokens on
structured text (code, counting) than on prose. Published headline numbers are often measured on
the easy case. We report prose and code separately for every spec-decode configuration, and
record acceptance (`tokens_per_verify_cycle`) from the server's `/metrics` per row. Never measure a
speculative speedup on random-token prompts: acceptance depends on real text [43].

## Launch records

Every result file carries `launch_records`: the image and digest, the container environment
(NCCL and vLLM variables), the serve argv as the container actually ran it, and the engine log
lines that state what vLLM decided (non-default args, attention and MoE backend, all-reduce
backend, KV cache size, CUDA-graph mode and any downgrade, speculative config). These are read
back from the engine with `bench/vllm-launch-record.py`, never retyped by hand.

## Fairness rules we apply

Drawn from MLPerf Inference [2], SemiAnalysis InferenceX [10][11], Artificial Analysis [15] and
the vLLM benchmark docs [18]:

1. **Two tracks.** A same-config track (identical checkpoint, quantization, KV dtype, context,
   sampling) isolates hardware. A best-per-platform track (each platform on its best engine,
   precision and parallelism, with the quality gate) shows what a buyer would deploy. Label which
   one a number belongs to.
2. **Per-platform tuning is allowed and disclosed**: every final launch command is published.
3. **Written warmup rule, applied to everyone**, and a measured window long enough to hold
   several power samples.
4. **Repeats and variance**: three repeats minimum (five for published headline figures), median
   and min-max, coefficient of variation per level; any level with CV above 10% is investigated
   before it is quoted. At least one repeat should follow a full engine restart, because the
   largest swings we saw were between launches, not within one.
5. **Sweep to saturation on every platform.** Never stop one platform's ladder earlier than
   another's, and never compare at a fixed stream count when the platforms have different
   `max_num_seqs` caps.
6. **Quality paired with speed.** A configuration that fails the correctness gate has no
   published speed numbers.
7. **Power**: wall AC for the outlets that feed the engine, measured during the same run, idle
   and engine-resident baselines reported separately, switch power listed as its own line. GPU
   software telemetry (`nvidia-smi power.draw`) is secondary; it samples only part of the
   runtime [33].
8. **Cost**: one published basis applied to every system: list price (MSRP) where the vendor
   publishes one, with a market-conditions disclaimer, hardware only. "No MSRP published" is
   stated rather than filled with a street figure.
9. **Publish losing and "does not fit" results**, with the raw JSON.
10. **Disclose interest.** Petronella Technology Group, Inc. sells AI hardware, including GB10
    systems and cluster cabling. That is why every script and raw result is here: rerun them.

## Metrics definitions

- **TTFT**: request send to the first streamed chunk carrying content or reasoning text.
- **Per-user tok/s**: `(completion_tokens - 1) / (last token time - first token time)`, median
  over requests in the window.
- **Aggregate tok/s (steady)**: output tokens produced inside the window / window seconds.
- **Prefill tok/s**: `usage.prompt_tokens / TTFT` at concurrency 1.
- **Users at the 10 tok/s floor**: the highest measured level whose per-user median stays at or
  above 10 tok/s with TTFT p95 at or below 10 s.
- **Tokens per joule**: aggregate tok/s / mean wall watts over the same window.
- **Per $1k**: tok/s divided by (units x list price / 1000), hardware only.

## Sources

2. MLCommons, MLPerf Inference Rules. https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
10. SemiAnalysis, "InferenceMAX: Open Source Inference Benchmarking". https://inferencex.semianalysis.com/blog/inferencemax-open-source-inference-benchmarking
11. SemiAnalysis, InferenceX About page. https://inferencex.semianalysis.com/about
15. Artificial Analysis, Performance Benchmarking Methodology. https://artificialanalysis.ai/methodology/performance-benchmarking
16. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024. https://arxiv.org/abs/2311.18677
18. vLLM docs, `vllm bench serve`. https://docs.vllm.ai/en/latest/cli/bench/serve.html
19. Usami, Vishwanath, Bethel, "Prefill/Decode-Aware Evaluation of LLM Inference on Emerging AI Accelerators," 2026. https://arxiv.org/abs/2606.17104
21. NVIDIA, DGX Spark Hardware Overview. https://docs.nvidia.com/dgx/dgx-spark/hardware.html
22. NVIDIA, DGX Station product page. https://www.nvidia.com/en-us/products/workstations/dgx-station/
33. Yang, Adamek, Armour, "Part-time Power Measurements: nvidia-smi's Lack of Attention." https://arxiv.org/abs/2312.02741
43. vLLM issue #42508 (speculative decoding acceptance varies by framework and data). https://github.com/vllm-project/vllm/issues/42508

Source numbers follow our internal research notes, so they are not consecutive.
