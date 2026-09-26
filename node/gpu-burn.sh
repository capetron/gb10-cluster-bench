#!/bin/bash
# gpu-burn.sh [IMAGE] - quick per-node health check before any benchmark (run ON the node).
#
# 1. 15 s fp16 4096x4096 matmul burn (compute-bound), nvidia-smi sampled inside the burn window.
# 2. 20 s device-memory copy of 1 GiB fp16 (bandwidth-bound), so a firmware memory-clock change
#    shows up even when the compute number does not move.
#
# Why: some GB10 units have been reported to latch at a low SM clock (630-950 MHz, under 20 W)
# until the power adapter is unplugged. Healthy GB10 readings we observed: roughly 2.2-2.3 GHz
# and 90 W+ under this burn, 78-91 fp16 TFLOPS depending on firmware. Run it on every unit
# and compare units against each other, not against a spec sheet.
#
# IMAGE: any image with CUDA torch (default: NGC PyTorch). Needs docker + NVIDIA container toolkit.
IMG="${1:-nvcr.io/nvidia/pytorch:25.09-py3}"
OUTF="$(mktemp)"; trap 'rm -f "$OUTF"' EXIT
idle=$(nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_event_reasons.active --format=csv,noheader)
locked=$(nvidia-smi -q -d CLOCK | grep -A3 -i "locked" | tr -s " " | tr "\n" " ")
( docker run --rm --gpus all --network none --entrypoint python3 "$IMG" -c "
import torch, time
a = torch.randn(4096, 4096, dtype=torch.float16, device='cuda'); b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
for _ in range(10): c = a @ b
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 15:
    c = a @ b; n += 1
torch.cuda.synchronize(); print(f'fp16 {2*4096**3*n/(time.time()-t0)/1e12:.1f} TFLOPS')
x = torch.empty(512*1024*1024, dtype=torch.float16, device='cuda'); y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 20:
    y.copy_(x); n += 1
torch.cuda.synchronize(); print(f'copy {2*x.numel()*2*n/(time.time()-t0)/1e9:.1f} GB/s')
" 2>&1 | grep -E "TFLOPS|GB/s" ) > "$OUTF" &
# container start takes a few seconds; sample twice inside the burn window
sleep 14; s1=$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv,noheader)
sleep 3;  s2=$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader)
wait
echo "host=$(hostname) image=$IMG idle=[$idle] locked=[$locked] load1=[$s1] load2=[$s2] $(tr '\n' ' ' < "$OUTF")"
