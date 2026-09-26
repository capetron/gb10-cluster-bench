#!/bin/bash
# parity-check.sh [HOST ...] - READ-ONLY swap and OOM-daemon parity across a GB10 cluster.
#
# Why this exists: in our cluster one node had no swap (its fstab line was commented out and the
# swapfile deleted) and was the only node running earlyoom. Under a large tensor-parallel model
# that node's worker was SIGTERMed first, six launches in a row, and every failure looked like
# "out of memory on the model". earlyoom kills with SIGTERM (no Python traceback) and logs to
# the journal, not dmesg (`journalctl -u earlyoom`). A benchmark on nodes with different swap or
# OOM policy measures config drift, not hardware.
#
# Prints one row per node; exits 1 if any node differs from the expected values.
# Hosts: arguments or $CLUSTER_HOSTS. EXPECT_SWAP_BYTES (default 16 GiB) and SWAPFILE
# (default /swap.img) set the target. Fixing drift is a human action (swapoff/fallocate/mkswap
# and `apt-get purge earlyoom` under sudo); this script never changes anything.
set -u
HOSTS="${*:-${CLUSTER_HOSTS:?pass hosts as arguments or set CLUSTER_HOSTS}}"
SZ="${EXPECT_SWAP_BYTES:-17179869184}"
SWF="${SWAPFILE:-/swap.img}"
bad=0
printf "%-12s %-16s %-10s %-9s %-6s %s\n" HOST SWAPFILE_BYTES SWAPON EARLYOOM FSTAB STATUS
for h in $HOSTS; do
  row=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "SWF='$SWF' bash -s" <<'EOF' 2>/dev/null
sz=$(stat -c %s "$SWF" 2>/dev/null || echo MISSING)
on=$(swapon --show=SIZE --noheadings 2>/dev/null | head -1); on=${on:-none}
eo=$(dpkg -l earlyoom 2>/dev/null | grep -c ^ii)
fs=$(grep -c "^$SWF[[:space:]]" /etc/fstab)
echo "$sz $on $eo $fs"
EOF
)
  read -r sz on eo fs <<< "${row:-UNREACHABLE - - -}"
  st=ok
  if [ "$sz" != "$SZ" ] || [ "$on" = none ] || [ "$eo" != 0 ] || [ "$fs" != 1 ]; then st=DRIFT; bad=1; fi
  printf "%-12s %-16s %-10s %-9s %-6s %s\n" "$h" "$sz" "$on" "$eo" "$fs" "$st"
done
exit $bad
