#!/bin/bash
# memfrag-check.sh [HOST ...] - READ-ONLY check of GB10 host-memory fragmentation before a benchmark
# or an engine (re)launch.
#
# Why this exists: on GB10 the NVIDIA driver backs GPU allocations with unmovable host pages and
# takes the SMALLEST free blocks first. GPU read bandwidth of a buffer depends on the physical block
# size behind it: about 234-240 GB/s from blocks of 2 MiB or less, about 250 from 16 MiB blocks and
# about 258-263 from 32 MiB blocks. A fresh boot has almost no small free blocks in Unmovable
# pageblocks, so the next engine's weights land in 32 MiB blocks (about 262 GB/s, +4-5% decode in our
# tests). Every engine run leaves 12-15 GiB of small free blocks behind in Unmovable pageblocks;
# drop_caches and compact_memory do not remove them; a reboot does.
# See docs/memory-bandwidth-reboot-before-launch.md.
#
# Verdict per node: FRESH if free memory in Unmovable pageblocks in blocks smaller than 32 MiB is
# under FRESH_GIB (default 2), else FRAGMENTED (expect the next engine at about 238-250 GB/s; reboot
# before launching for full speed). Changes nothing.
#
# Hosts: arguments or $CLUSTER_HOSTS; key-based ssh. /proc/pagetypeinfo is root-only, so sudo is
# needed for that one read. SUDO_PASSWORD_FILE is a path to a file holding the node's sudo password,
# piped to `sudo -S` over stdin; SUDO_PASSWORD_FILE=none uses `sudo -n` (passwordless sudo). Unset:
# ssh -t, and sudo prompts you on the terminal.
set -u
case "${1:-}" in -h|--help) sed -n '2,21p' "$0"; exit 0;; esac
HOSTS="${*:-${CLUSTER_HOSTS:?pass hosts as arguments or set CLUSTER_HOSTS}}"
FRESH="${FRESH_GIB:-2}"
PWF="${SUDO_PASSWORD_FILE:-}"
if [ -n "$PWF" ] && [ "$PWF" != none ] && [ ! -r "${PWF/#\~/$HOME}" ]; then echo "SUDO_PASSWORD_FILE not readable"; exit 2; fi
RD='grep -E "zone +Normal, type +Unmovable" /proc/pagetypeinfo'

# prints three lines: /proc/uptime, the Normal/Unmovable pagetypeinfo row, the Normal buddyinfo row
read_host() {
  local h=$1 pti
  if [ -z "$PWF" ]; then
    # the sudo prompt goes to the terminal; only the file content is captured
    ssh -t -o ConnectTimeout=5 "$h" "sudo cat /proc/pagetypeinfo > /tmp/memfrag-pti.txt" >/dev/tty || return
    pti=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" 'grep -E "zone +Normal, type +Unmovable" /tmp/memfrag-pti.txt; rm -f /tmp/memfrag-pti.txt')
  elif [ "$PWF" = none ]; then
    pti=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" "sudo -n bash -c '$RD'")
  else
    pti=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" "sudo -S -p '' bash -c '$RD'" < "${PWF/#\~/$HOME}")
  fi
  [ -z "$pti" ] && return
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" 'cat /proc/uptime'
  echo "$pti"
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" 'grep Normal /proc/buddyinfo'
}

printf "%-12s %-9s %-26s %-30s %s\n" node uptime "free GiB in 32 MiB blocks" "Unmovable free GiB < 32 MiB" verdict
for h in $HOSTS; do
  out=$(read_host "$h" 2>/dev/null)
  [ "$(echo "$out" | grep -c .)" -lt 3 ] && { printf "%-12s unreachable or no root read\n" "$h"; continue; }
  echo "$out" | python3 -c '
import sys
l = sys.stdin.read().splitlines(); up = float(l[0].split()[0]) / 3600
um = [int(x) for x in l[1].split()[6:]]; bd = [int(x) for x in l[2].split()[4:]]
g = lambda v, o: v * 4096 * 2**o / 2**30
frag = sum(g(n, o) for o, n in enumerate(um[:-1])); big = g(bd[-1], len(bd) - 1)
ok = frag < float(sys.argv[2])
print("%-12s %-9s %-26.1f %-30.1f %s" % (sys.argv[1], "%.1f h" % up, big, frag,
      "FRESH" if ok else "FRAGMENTED (reboot before launch for full bandwidth)"))' "$h" "$FRESH"
done
