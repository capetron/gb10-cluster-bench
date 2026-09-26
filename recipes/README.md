# Launch recipes

The exact vLLM launches behind the result files, read back from the running containers with
`bench/vllm-launch-record.py` (the full records, including engine log lines, are in each result
JSON under `launch_records`). Addresses are replaced with documentation placeholders: the fabric
plane is `192.168.100.0/24`, rank 0 is `192.168.100.1`.

All GB10 runs used the same image: `vllm/vllm-openai:nightly-aarch64`, image id
`sha256:4d1d2cc1d6c2...`, vLLM build commit `0961bbae2894d574be790d219651824eb199318e`. Model
checkpoints: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (snapshot `7b719225`) and an NVIDIA NVFP4 build of
Qwen3.8-27B. GB10 platform: kernel 7.0.0-1019-nvidia, driver 580.178.04, MTU 9000 on the fabric.

## Common multi-node environment (GB10 over 200G RoCE)

One ConnectX-7 port per node (`enp1s0f0np0`, RDMA device `rocep1s0f0`), all nodes on one switched
L2 plane. These are the variables the engine actually ran with:

```
NCCL_SOCKET_IFNAME=enp1s0f0np0  GLOO_SOCKET_IFNAME=enp1s0f0np0  TP_SOCKET_IFNAME=enp1s0f0np0
NCCL_NET=IB  NCCL_IB_DISABLE=0  NCCL_IB_HCA=rocep1s0f0  NCCL_IB_ROCE_VERSION_NUM=2
NCCL_IB_ADDR_FAMILY=AF_INET  NCCL_IB_ADDR_RANGE=192.168.100.0/24
NCCL_IB_QPS_PER_CONNECTION=1  NCCL_IB_MERGE_NICS=0  NCCL_CROSS_NIC=0  NCCL_MAX_NCHANNELS=8
NCCL_NVLS_ENABLE=0  NCCL_CUMEM_ENABLE=0  NCCL_IGNORE_CPU_AFFINITY=1  NCCL_DEBUG=WARN
TORCH_NCCL_ASYNC_ERROR_HANDLING=1  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_HOST_IP=<this node's fabric address>  VLLM_ENGINE_READY_TIMEOUT_S=3600  HF_HUB_OFFLINE=1
```

`NCCL_IB_GID_INDEX` was left unset: recent NCCL (2.30.7 in our tests) picks the RoCE v2 GID on
its own; older NCCL needed it set, and the right index differed between GB10 models in our rack
(check with `show_gids`). Container flags on every rank:

```
docker run -d --name vllm_qwen --network host --ipc host --shm-size 32g --gpus all \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit nofile=1048576 \
  -v $HOME/.cache/huggingface:/hf:ro -e <env above> vllm/vllm-openai:nightly-aarch64 ...
```

Rank 0 runs the API server, engine core and a worker, so it is the tightest node on memory;
workers add `--headless` and their own `--node-rank`.

## 1x GB10: Qwen3.8-27B NVFP4, MTP-3

```
vllm serve /model --served-model-name qwen3.8-27b --trust-remote-code \
  --max-model-len 262144 --gpu-memory-utilization 0.80 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
# env: TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a
```

## 2x GB10, TP2: Qwen3.8-Flash-Next NVFP4, single-user "speed" profile

Six sequences, FP8 KV, decode-only CUDA graphs, prefix caching off. Fast for one to four users,
queues beyond six by design.

```
vllm serve <snapshot> --served-model-name qwen3.8-flash-next --quantization modelopt_fp4 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 --master-addr 192.168.100.1 --master-port 29541 \
  --distributed-executor-backend mp --load-format safetensors --no-enable-flashinfer-autotune \
  --max-model-len 262144 --max-num-seqs 6 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.70 \
  --kv-cache-dtype fp8_e4m3 --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24]}' \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3
# extra env: VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_DEEP_GEMM=0 TORCH_CUDA_ARCH_LIST=12.1a
```

## 4x GB10, TP4 (and 2x TP2 replicas, and 8x TP8): Qwen3.8-Flash-Next NVFP4, serving profile

128 sequences, expert parallel, eager mode, MTP-4. The same flags ran at TP2 (`--gpu-memory-utilization
0.78`, four independent replicas behind a client-side round robin) and TP8 (`0.65`).

```
vllm serve <snapshot> --served-model-name qwen3.8-flash-next --quantization modelopt_fp4 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --nnodes 4 --node-rank 0 \
  --master-addr 192.168.100.1 --master-port 29501 --distributed-executor-backend mp \
  --max-model-len 163840 --max-num-seqs 128 --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --gpu-memory-utilization 0.65 --kv-cache-dtype auto --load-format safetensors --enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' --enable-prefix-caching \
  --no-enable-flashinfer-autotune --enable-expert-parallel \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3
```

Prefix caching is on in the serving profile; the harness defeats it with a unique header per
request and records the server prefix-cache hit counters in every row, so you can check.

## DGX Station GB300: Qwen3.8-Flash-Next NVFP4 (prefill comparison)

```
vllm serve <checkpoint> --tensor-parallel-size 1 --max-model-len 131072 --max-num-seqs 64 \
  --enable-chunked-prefill --enable-prefix-caching --gpu-memory-utilization 0.92 --kv-cache-dtype auto
# env: TORCH_CUDA_ARCH_LIST=10.3a ; image vllm/vllm-openai:nightly-aarch64 (vLLM 0.29.1rc1 dev build)
# FP8 KV was rejected on sm_103 by FlashAttention in this build, so KV is bf16.
```

## Operational notes

- Copy images and weights to the other nodes over the fabric, not from the internet, and turn
  off ssh compression for the copy (`-o Compression=no`); on our nodes compression alone was
  about a 20x slowdown.
- Mount the whole Hugging Face hub directory, not a single snapshot directory: snapshot files are
  symlinks into `../../blobs` and break when mounted alone.
- Drop the page cache before launching very large models; watch `MemAvailable` during load, since
  a GB10 that runs out of unified memory can wedge the whole host.
