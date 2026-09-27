# GB10 memory bandwidth: reboot before you launch

Eight identical GB10 units (DGX Spark 1-4 and MSI EdgeXpert 1-4) split into two GPU
memory-bandwidth tiers, and decode speed followed the tier. The cause is memory state, not
silicon, and a reboot before launching an engine fixes it. Labels: **[measured]** = read from a
saved file; **[inferred]** = our reasoning from measured data.

## Short version

- A GPU read probe (4 GiB fp32 buffer) reads **about 262 GB/s on a freshly booted unit** and
  **about 238 GB/s on a unit that has already run an inference engine** since boot [measured].
- Decode tracks it: Pearson r = 0.88 at 32 users and 0.83 at 1 user across the 8 unit means
  [measured]. Rebooting one slow unit gave **+4.2% decode at 1 user and +5.0% at 32 users**
  [measured]. Prefill barely moves.
- The cause: the NVIDIA driver backs GPU allocations with ordinary unmovable host pages, taken
  smallest free block first, and a buffer built from small blocks reads slower. Every engine run
  leaves 12-15 GiB of small free blocks behind, and the next engine's weights land in them.
- `drop_caches`, `compact_memory` and `cma=128M` do not fix it. A reboot does.
- Check before a launch or a benchmark with `node/memfrag-check.sh` (read-only).

## The parity study [measured]

All 8 units were checked first for config drift: same kernel, driver, VBIOS, firmware versions (one unit has an older BIOS build date with the
same version string; the slow tier includes units with both dates, so it is not the cause), clock
lock (see [power-offs-and-clock-lock.md](power-offs-and-clock-lock.md)), CPU governor, THP,
swap, 128 GB of LPDDR5 at 8533 MT/s (per SMBIOS), container image, model bytes (sha256) and engine
config. Spec-decode acceptance was 0.444-0.448 on every unit. Under load the SM clock median
(2,171-2,184 MHz), GPU power median (47.9-50.4 W) and GPU temperature were the same. CPU memory
bandwidth (STREAM triad 116.4-118.8 GB/s) was the same on all 8.

What differed was GPU read bandwidth, identical whether all 8 were probed at once or one at a
time (so not rack heat or a shared resource):

| Unit | Uptime at test | Free in 32 MiB blocks | GPU read / copy GB/s | Decode c=1 | Decode c=32 | Prefill 32k / 131k |
|---|---|---|---|---|---|---|
| DGX Spark 1 | 19.9 h | 43 GB | 242.2 / 231.7 | 85.2 | 591.5 | 6,211 / 4,441 |
| DGX Spark 2 | 5.9 h | 68 GB | 261.9 / 244.3 | 88.1 | 608.1 | 6,229 / 4,464 |
| DGX Spark 3 | 8.7 h | 45 GB | 238.8 / 230.7 | 86.2 | 592.7 | 6,230 / 4,477 |
| DGX Spark 4 | 2.4 h | 90 GB | 263.2 / 244.4 | 88.3 | 615.0 | 6,266 / 4,496 |
| MSI EdgeXpert 1 | 19.9 h | 3 GB | 238.9 / 230.7 | 86.8 | 599.8 | 6,203 / 4,459 |
| MSI EdgeXpert 2 | 21.6 h | 0 GB | 238.7 / 226.6 | 83.8 | 581.5 | 6,204 / 4,445 |
| MSI EdgeXpert 3 | 1.9 h | 87 GB | 262.9 / 244.5 | 87.6 | 607.2 | 6,200 / 4,459 |
| MSI EdgeXpert 4 | 21.8 h | 2 GB | 237.8 / 227.7 | 85.1 | 590.0 | 6,221 / 4,458 |

Nemotron-3.5-Lightning-30B-A3B NVFP4, single unit, `gpu-memory-utilization 0.85`, 262,144
context, speculative decoding k=3; decode = mean of 4 windows of 60 s after a 180 s heat soak,
aggregate tok/s, the same run seed (so the same prompts) on every unit. The bandwidth probe:
4 GiB fp32 tensor, read = `sum()`, copy counted as read + write, median of 5 trials.

The fast tier averaged 610 tok/s at 32 users vs 591 for the rest (+3.2%) on a 9.8% bandwidth
gap. [inferred] Decode here is only partly bandwidth bound (speculative decoding, CPU
scheduling), which fits the smaller effect.

At first it looked like uptime: the fast units were the three most recently booted. The real
variable was the number of engine launches since boot (next section).

## Mechanism [measured unless marked]

1. The driver backs `cudaMalloc` with ordinary kernel pages (`CmaTotal 0`, no hugetlb), taken
   from the buddy allocator as unmovable pages, smallest free block first. Holding 10 x 4 GiB on
   one unit consumed 64 KiB-8 MiB blocks and did not touch its 53 GiB of 32 MiB blocks.
2. Read bandwidth of a buffer rises with the size of the blocks it came from (290 chunks on 5
   units, each 90%+ from one block size; our own CUDA kernel, which peaks about 4 GB/s below
   torch):

   | Source block size | Chunks | Read GB/s median (min-max) |
   |---|---|---|
   | 512 KiB-1 MiB | 6 | 234-236 |
   | 2 MiB | 11 | 236.9 (234.1-239.8) |
   | 4 MiB | 16 | 237.4 (235.4-254.8) |
   | 8 MiB | 13 | 239.3 (234.6-256.3) |
   | 16 MiB | 7 | 250.4 (240.3-257.9) |
   | 32 MiB | 237 | 258.2 (244.5-260.5) |

3. A fresh boot has 0.1-0.5 GiB free in small blocks inside Unmovable pageblocks. One engine run
   leaves 12-15 GiB of them; a second run about 28 GiB. The next engine's first allocations,
   its weights, land in those fragments.
4. What does NOT fragment: reading 191 GB of model files through the page cache and dropping it
   returned the buddy lists to their previous state; a unit left idle after a reboot read
   260.6, 262.2 and 263.1 GB/s at 0.0, 0.4 and 0.7 h.
5. [inferred] Why compaction cannot help: after GPU use most 2 MiB pageblocks are typed
   Unmovable, and kernel compaction only migrates pages into free space of Movable pageblocks.
   Why contiguity matters to the GPU is also inferred: most likely address-translation reach
   (larger physical runs mapped with larger GPU pages) or DRAM channel spread. We could not
   separate the two from user space.

## Remedies tested

| Remedy | Effect | Verdict |
|---|---|---|
| Reboot before launching | 239 -> 262 GB/s; Nemotron c=1 85.1 -> 88.7 (+4.2%), c=32 592.8 -> 622.5 (+5.0%), same spec acceptance [measured] | **Works.** About 1 minute plus the engine's load time. |
| Relaunch the engine without a reboot | c=32 on three units: 622.5 -> 608.1, 615.0 -> 604.8, 604.5 -> 598.1 [measured] | Loses 1-2.5% per relaunch |
| `drop_caches` + `compact_memory` | no change on three units [measured] | Does not work |
| Boot parameter `cma=128M` | fresh boot 263.2 vs 262.6 GB/s for a control unit; same 238 GB/s after one run [measured] | No benefit, reverted |
| Hugepages reserved at boot | not run; the driver does not allocate from hugetlb [inferred] | Not worth a boot change |
| Hold the fragments with a placeholder during engine start | not run; at 0.85 the engine needs about 101.7 GB free, so at most about 18 GB could be held [inferred] | Not viable at these settings |

## Recommendations

1. **Reboot before launching** any engine that should run at full speed, and before any
   benchmark that compares units or configurations. Record the `node/memfrag-check.sh` verdict
   next to every result.
2. **Long-running serving needs no periodic reboot** [inferred]: a running engine keeps the
   placement it got at launch. Make any planned engine restart a reboot plus launch.
3. **Tensor-parallel runs:** reboot every participating unit first; the slowest unit sets the
   pace of the group.
4. Treat a read below about 200 GB/s as a different problem (see the outside reports).

## Caveats

- The idle-uptime control ran 42 minutes on one unit; it rules out only a fast idle decay.
- The effect was measured on one model and engine recipe; other models will see a different
  share of decode that is bandwidth bound.
- Soft-reboot time to ssh was 38-52 s on the Sparks and one MSI unit, 1.5-2.2 min on two others.

## Outside reports

- [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks issue #23](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/23):
  model loading on a two-Spark pair fragmented memory until the driver's contiguous allocations
  failed (`NV_ERR_NO_MEMORY`). `compact_memory` helped there, an allocation-failure case; it did
  not help bandwidth here. Same root: driver allocations are unmovable host pages.
- [kreuzhofer/dgx-manager issue #88](https://github.com/kreuzhofer/dgx-manager/issues/88): a
  Spark read 165 GB/s vs 254-261 on its peers after 30 h of serving, restored after an AC drain.
  That is a different, half-speed fault, not this 238 vs 262 tier.
- [NVIDIA forum 383926](https://forums.developer.nvidia.com/t/upgrade-to-7-6-0-kernel-7-0-0-1019-nvidia-system-wide-slowdown-50-60/383926):
  a 50-60% slowdown on the 7.6.0 image with several causes discussed there (among them a CMA
  footprint problem, an SM clock clamp and a 30 W power-delivery fallback); `cma=128M kho=off` was
  suggested, and a moderator points to the `kho=off` hotfix. We tested `cma=128M` and saw no benefit
  on this tier; `kho=off` was already installed on all 8 (see `node/install-kho-hotfix.py`).
