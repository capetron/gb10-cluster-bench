#!/bin/bash
# install-clock-lock.sh [--uninstall|--check|--verify] [HOST ...] - install the GB10 GPU clock lock
# (node/gpu-clock-lock.service: nvidia-smi -lgc CLOCK_MIN,CLOCK_MAX at boot) on each node.
#
# Why this exists: GB10 units can power themselves off, with nothing in the logs, a few seconds
# into a long prefill on a hot unit at the uncapped boost clock. Capping the SM clock at 2200 MHz
# stopped it in our tests at a cost of 1-3% single-user decode. See docs/power-offs-and-clock-lock.md.
#
# Modes:
#   (default)    install, enable and start the unit. Idempotent: the unit file is rewritten only
#                if it differs, and the lock is re-applied (systemctl restart). Nothing else restarts.
#   --check      read-only: is-enabled, is-active, the unit's "GPU clocks set" journal line for
#                this boot, current SM clock.
#   --verify     functional proof. `nvidia-smi -q` shows nothing different at idle when a lock is
#                set, and NVML has no getter for locked clocks, so this runs a 20 s bf16 matmul burn
#                in VERIFY_IMAGE (--network none) while sampling the SM clock every 250 ms.
#                LOCK_OK if the max stays at or below CLOCK_MAX + 50 MHz (the clock can sit one bin
#                above the cap; default clocks reach about 2,400 MHz). Skips if the GPU is in use.
#                Needs docker on the node, and runs as your ssh user (no sudo).
#   --uninstall  stop, disable and remove the unit, then nvidia-smi -rgc (default clocks).
#
# Hosts: arguments or $CLUSTER_HOSTS; key-based ssh. CLOCK_MIN / CLOCK_MAX (default 300 / 2200).
# VERIFY_IMAGE (default vllm/vllm-openai:nightly-aarch64; any image with CUDA torch works).
# Sudo (install and uninstall only): SUDO_PASSWORD_FILE is a path to a file holding the node's sudo
# password, piped to `sudo -S` over stdin and never put on a command line; SUDO_PASSWORD_FILE=none
# uses `sudo -n` (passwordless sudo). Unset: ssh -t, and sudo prompts you on the terminal.
set -u
MODE=install
case "${1:-}" in --uninstall) MODE=uninstall; shift;; --check) MODE=check; shift;; --verify) MODE=verify; shift;;
  -h|--help) sed -n '2,26p' "$0"; exit 0;; esac
HOSTS="${*:-${CLUSTER_HOSTS:?pass hosts as arguments or set CLUSTER_HOSTS}}"
LO="${CLOCK_MIN:-300}"; HI="${CLOCK_MAX:-2200}"
case "$LO$HI" in *[!0-9]*) echo "CLOCK_MIN and CLOCK_MAX must be integers (MHz)"; exit 2;; esac
IMG="${VERIFY_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
PASSLINE=$((HI + 50))
UNIT=gpu-clock-lock.service
HERE="$(cd "$(dirname "$0")" && pwd)"
PWF="${SUDO_PASSWORD_FILE:-}"

TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT
sed -e "s/-lgc [0-9]*,[0-9]*/-lgc $LO,$HI/" -e "s/at [0-9]*-[0-9]* MHz/at $LO-$HI MHz/" "$HERE/$UNIT" > "$TMPD/$UNIT"
cat > "$TMPD/helper.sh" <<'REMOTE'
#!/bin/bash
# remote helper for install-clock-lock.sh; $1 = install | uninstall | status | verify
u=gpu-clock-lock.service
case "$1" in
install)
  if ! cmp -s /tmp/$u /etc/systemd/system/$u; then install -m 0644 /tmp/$u /etc/systemd/system/$u && systemctl daemon-reload && echo -n "unit-written "; else echo -n "unit-unchanged "; fi
  systemctl enable $u >/dev/null 2>&1 && echo -n "enable-ok "
  systemctl restart $u && echo "start-ok"
  rm -f /tmp/$u;;
uninstall)
  systemctl disable --now $u >/dev/null 2>&1
  rm -f /etc/systemd/system/$u; systemctl daemon-reload
  nvidia-smi -rgc | head -1;;
status)
  j=$(journalctl -u $u -b --no-pager -o cat 2>/dev/null | grep -o "gpuClkMin [0-9]*, gpuClkMax [0-9]*" | tail -1)
  echo "enabled=$(systemctl is-enabled $u 2>/dev/null) active=$(systemctl is-active $u 2>/dev/null) journal=[${j:-not readable or none}] sm=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | tr -d ' ')";;
verify)
  img="$2"; pass="$3"
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "SKIP gpu busy"; exit 0; fi
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -lms 250 > /tmp/gpu-lockv.csv 2>&1 & S=$!
  docker run --rm --name gpu_lockverify --gpus all --network none --entrypoint python3 "$img" -c '
import torch, time
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); t = time.time(); n = 0
while time.time() - t < 20:
    a @ a; n += 1
torch.cuda.synchronize(); print("burn_tflops=%.0f" % (2 * 8192**3 * n / (time.time() - t) / 1e12))' 2>&1 | grep burn_tflops | tr '\n' ' '
  kill $S; m=$(grep -E '^[0-9]+$' /tmp/gpu-lockv.csv | sort -n | tail -1); rm -f /tmp/gpu-lockv.csv
  if [ -n "$m" ] && [ "$m" -le "$pass" ]; then echo "LOCK_OK max_sm=${m}MHz"; else echo "LOCK_NOT_IN_EFFECT max_sm=${m:-none}MHz"; fi;;
esac
REMOTE

# run the helper as root on host $1 with action $2, following the SUDO_PASSWORD_FILE convention
as_root() {
  if [ -z "$PWF" ]; then ssh -t "$1" "sudo bash /tmp/gpu-clock-lock-helper.sh $2"
  elif [ "$PWF" = none ]; then ssh "$1" "sudo -n bash /tmp/gpu-clock-lock-helper.sh $2"
  else ssh "$1" "sudo -S -p '' bash /tmp/gpu-clock-lock-helper.sh $2" < "${PWF/#\~/$HOME}"; fi
}

# interactive sudo must reach the terminal, so its output is not captured or reformatted
root_run() {
  if [ -z "$PWF" ]; then echo "$1: (sudo may prompt)"; as_root "$1" "$2"
  else echo "$1: $(as_root "$1" "$2" 2>&1 | tr '\n' ' ')"; fi
}

if [ -n "$PWF" ] && [ "$PWF" != none ] && [ ! -r "${PWF/#\~/$HOME}" ]; then echo "SUDO_PASSWORD_FILE not readable"; exit 2; fi
rc=0
for H in $HOSTS; do
  scp -q "$TMPD/helper.sh" "$H:/tmp/gpu-clock-lock-helper.sh" || { echo "$H: scp failed"; rc=1; continue; }
  case $MODE in
  check)  echo "$H: $(ssh "$H" 'bash /tmp/gpu-clock-lock-helper.sh status' 2>&1)";;
  verify) echo "$H: $(ssh "$H" "bash /tmp/gpu-clock-lock-helper.sh verify '$IMG' $PASSLINE" 2>&1 | tr '\n' ' ')";;
  install)
    scp -q "$TMPD/$UNIT" "$H:/tmp/$UNIT" || { echo "$H: scp unit failed"; rc=1; continue; }
    root_run "$H" install
    echo "$H: $(ssh "$H" 'bash /tmp/gpu-clock-lock-helper.sh status' 2>&1)";;
  uninstall)
    root_run "$H" uninstall
    echo "$H: $(ssh "$H" 'bash /tmp/gpu-clock-lock-helper.sh status' 2>&1)";;
  esac
  ssh "$H" 'rm -f /tmp/gpu-clock-lock-helper.sh' 2>/dev/null
done
exit $rc
