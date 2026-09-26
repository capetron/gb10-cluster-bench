# GB10 power-offs at long-prefill onset, and the clock-lock workaround

What we saw on an eight-unit GB10 pod (DGX Spark 1-4 and MSI EdgeXpert 1-4) in September 2026,
how we narrowed it down, and the workaround we now run on every unit. Every number is labelled
**[measured]** (read from a saved log or telemetry file) or **[inferred]** (our reasoning from
the measured data). The cause below is the best-supported explanation, not a proven one.

## Symptom

A unit goes OFF, not rebooting, a few seconds into a long-context prefill (131k tokens) while it
is already hot from decode load. It stays off until its AC input is cycled, and draws 4.1-4.4 W
at the outlet, standby level [measured]. The journal ends with routine lines: no shutdown
sequence, no panic, no OOM, no MCE, no thermal message, and pstore is empty [measured].

Workload: Nemotron-3.5-Lightning-30B-A3B NVFP4 on vLLM, `--gpu-memory-utilization 0.85`,
`--max-model-len 262144`, 64 sequences, 8,192 batched tokens, fp8 KV cache, prefix caching,
speculative decoding (k=3), with 32k and 131k prefill tests. Three of three contest runs with
the whole pod loaded ended this way.

One trap first: **the unit's own logs cannot show the last seconds.** The on-disk journal lost
the last 6-14 s and the on-disk `docker logs` the last 26-29 s before each loss [measured].
For any long-context run you may need to diagnose, stream telemetry to another machine (ssh plus
`journalctl -f`, `docker logs -f`, a 250 ms `nvidia-smi` sampler and a ping watch).

## What we ruled out

| Candidate | Verdict | Evidence |
|---|---|---|
| Host memory exhaustion at 0.85 | Ruled out [measured] | MemAvailable flat at 9.2-9.4 GB and PSI memory 0.00 until under 1.5 s before both induced losses; swap untouched. |
| Process-spawn burst (AppArmor DENIED lines) | Ruled out [measured] | A constant 95-100 per minute on an idle unit with no model loaded; process count flat at 555-578 through every run. |
| OS / ACPI thermal trip | Ruled out as the trigger [measured] | All 7 ACPI zones trip at 104.8 C; the last readings before the losses were 93.3 and 96.2 C, and the same unit survived 98.1 C that day. A kernel critical trip would also shut down cleanly. |
| PDU, circuit, voltage | Ruled out [measured] | Relay stayed on and voltage steady; the outlet read 4.06-4.39 W after every loss. |
| Average wall power too high | Ruled out [measured] | The unit held 204-212 W at the wall (30 s averages) for 3 minutes with 64-user decode, a CPU burn and three 131k prefills together, and survived. The two induced losses happened at 169 W and 145 W. |

[inferred] The unit went to a controlled soft-off rather than losing its supply. A Spark powers
itself back on when its DC input returns, so a brick dropout would have looked like a reboot.
The brick kept supplying power and the platform chose to turn off.

## Reproducing it, and what stopped it

All runs on one DGX Spark ("Spark A") with the contest launch settings.

| Run | Sequence | Clocks | 131k onsets | Result |
|---|---|---|---|---|
| 1a-1d | cold; 131k x8 back to back; 12 min decode soak; decode + CPU burn + 131k | default | 17 | survived |
| 2a | fresh engine, the exact contest sequence | default | 3 (+1 at 120k) | survived |
| 2b | 131k x3 | locked 300-2200 MHz | 3 | survived |
| 2c | decode at 1 then 32 users (60 s each), prefill 32k x2, then 131k | default | 1 | **power off about 11 s into the first 131k** |
| 3a | fresh engine, run 2c's sequence 6 times | locked 300-2200 MHz | 12 | survived; no clock-event reason in 5,458 samples |
| 3b | same engine, the same sequence | default | 1 | **power off about 8 s into the first 131k** |

[measured] The 2c/3 sequence ended in a power loss 2 of 2 times at default clocks and 0 of 6
times with the lock (Fisher exact, one-sided, p = 1/28 = 0.036). Overall at 131k: default
clocks, 2 losses in 22 onsets; locked, 0 in 15. This is one engine on one unit on one day.

### Telemetry signature in the last 3 seconds [measured]

At 250 ms sampling before each loss: GPU power 57-102 W, swinging by up to 45 W between samples;
SM clock hopping between 1,872 and 2,431 MHz; the GPU T.limit margin
(`temperature.gpu.tlimit`) jumping between -7 and +13 C from one sample to the next. With the
lock, the same workload peaked at 83.9 W, stayed at 2,177-2,200 MHz, kept the T.limit margin at
2 C or more, and logged no slowdown events.

[inferred] The trip is a fast event: power steps and hotspot spikes on the order of 100 ms at
uncapped boost, on an already hot chassis. The OS-visible sensors never reach a limit and the
averaged wall power is lower than in runs that survived, so the protection acts on something
the OS cannot read (instantaneous current in the USB-C power path, or an internal hotspot).
Removing the top boost bins removes the transient.

## Loaded confirmation across the pod [measured]

With the lock on every running unit (7 of 8; one unit was off), seven units ran the full
workload at once for two phases of about 20 minutes: five single-unit Nemotron lanes plus
GLM-5.3-Flash NVFP4 TP2 on two unit pairs, at 0.85 memory, 262k context, 32k and 131k prefill
and a 120k needle.

| | Phase A | Phase B |
|---|---|---|
| Long-input onsets (131k prefills + 120k needles) | 36 | 30 |
| Failed requests / ping failures / power losses | 0 / 0 / 0 | 0 / 0 / 0 |
| Pod wall power, max | 1,107.2 W | 1,111.5 W |
| SM clock under load | 2,073-2,190 MHz | 2,138-2,190 MHz |
| Max GPU instant power | 87.5 W | 87.1 W |
| Max ACPI zone (trip 104.8 C) | 96.1 C | 95.9 C |

66 long-input onsets, 0 losses. The contest losses happened with the pod drawing 1,013-1,361 W,
so the confirmation load sat inside that range.

[inferred] At the default-clock loss rate from the single-unit test (2 in 22), 48 Nemotron 131k
onsets passing clean has a probability of about (20/22)^48, roughly 1%, before counting the TP2
onsets. This assumes independent onsets with equal risk.

## What the lock costs [measured]

Nemotron single unit, aggregate tok/s.

| Metric | Default clocks | Locked 300-2200 MHz |
|---|---|---|
| Decode, 1 user | 91.0-92.1 | 89.2 median on the test unit; 84.0-90.2 across units under pod load |
| Decode, 32 users | 614.6-618.8 | 619.0 median |
| Prefill 32k | 6,665 cold, 6,012-6,517 warm | 6,261 median |
| Prefill 131k | 4,612-4,795 cold, 4,040-4,414 after decode heat | 4,488 median |

On the test unit the lock costs 1-3% at one user and nothing at 32 users. After any warm-up it
is equal or faster at 131k, because default clocks spend time in thermal and power slowdown.
One other unit showed 7-9% at one user against a figure measured a day earlier with a longer
window; part of that gap is unit-to-unit memory state (see
[memory-bandwidth-reboot-before-launch.md](memory-bandwidth-reboot-before-launch.md)).

## Install and verify

`node/gpu-clock-lock.service` is a oneshot unit that runs `nvidia-smi -lgc 300,2200` after
`nvidia-persistenced` at boot. `node/install-clock-lock.sh` installs it on each host
(idempotent) and has `--check`, `--verify` and `--uninstall`.

```
node/install-clock-lock.sh node1 node2        # install, enable, apply now
node/install-clock-lock.sh --check node1      # enabled, active, journal line this boot
node/install-clock-lock.sh --verify node1     # 20 s GPU burn; LOCK_OK if the SM clock stays capped
```

Verifying the lock is less obvious than it should be. On driver 580.178.04,
`nvidia-smi -q -d CLOCK` shows nothing different when the lock is set, and NVML has no getter
for locked clocks [measured]. Check instead:

1. the unit is enabled and active;
2. its journal line for the current boot: `GPU clocks set to "(gpuClkMin 300, gpuClkMax 2200)"`;
3. a burn: default clocks reached 2,398-2,405 MHz and locked runs 2,177-2,210 MHz (the SM clock
   can sit one bin above the cap), so the pass line is the cap plus 50 MHz. Burn throughput was
   94-96 TFLOPS locked vs 96 at default.

The lock survives a reboot (the unit re-applies it) and does not touch running containers.

## Caveats and open items

- **The cause is not proven.** The evidence points at platform power protection acting on a
  transient; the embedded controller is not visible from the OS.
- **This is a workaround pending NVIDIA.** A support case with our telemetry is drafted, not yet
  sent.
- **Power bricks.** During these tests the 240 W USB-C power bricks were taped together in pairs.
  Paired bricks run hotter, so a brick's own protection is an open factor. A re-test with the
  bricks separated is pending: default clocks first, then 2300 and 2400 MHz caps, stopping at
  the first loss.
- Nothing above 2200 MHz was tested. Other owners report caps from 2100 to 2400 MHz.
- TP4 and TP8 across the pod have not yet run under the lock; the TP2 pairs and the seven-unit
  load are the closest test.
- The GLM-5.3-Flash TP2 recipe at 0.85 leaves rank 0 with 1.3-1.8 GiB MemAvailable. It held for
  40 minutes under load, but do not raise it.

## Outside reports

Several owners report the same symptom (unit off, not rebooting, no crash lines, at a load step):

- [tonyd2wild/dgx-spark-hard-poweroff-fix](https://github.com/tonyd2wild/dgx-spark-hard-poweroff-fix):
  "No OOM-killer output, no kernel panic, no hung-task, no MCE, no Xid"; attributes it to the
  embedded controller cutting power faster than Linux can log.
- [NVIDIA forum 373251](https://forums.developer.nvidia.com/t/dgx-spark-gb10-reproducibly-hard-powers-off-under-gpu-load-fully-updated-zero-crash-capture/373251):
  reproducible hard power-off under GPU load, empty pstore, journal truncated mid-line.
- [NVIDIA forum 377478](https://forums.developer.nvidia.com/t/spark-abruptly-shuts-down/377478):
  NVIDIA staff say shutdown issues "are being investigated"; a user reports
  `nvidia-smi -lgc 300,2400` as part of a fix.
- [ai-muninn GX10 write-up](https://ai-muninn.com/en/blog/gx10-thermal-hard-poweroff): hard
  power-offs reproduced on purpose; a 2200 MHz clock lock stopped them at about 9% throughput
  cost.
- [note.com write-up](https://note.com/nob75note/n/ned4cc56ead79): power swinging 18-95 W in step
  with vLLM prefill; the author concludes over-current rather than heat.
- [NVIDIA forum 362585](https://forums.developer.nvidia.com/t/msi-edgexpert-suddenly-power-off-during-llama-benchy-possible-pd-firmware-issue/362585):
  an MSI EdgeXpert power-off that a power adapter swap did not fix; cites SoC FW 2.152.15 and PD
  FW 5.22 as the fix. Our test unit already ran newer SoC firmware and PD 5.22.

[inferred] More than one fault (over-current, a hidden thermal sensor, firmware) may produce the
same log-less power-off. The clock cap is the mitigation reported most consistently.
