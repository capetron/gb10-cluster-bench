#!/bin/bash
# verify-versions.sh [HOST ...] - READ-ONLY version table for a GB10 cluster.
#
# One row per node: kernel, driver, VBIOS, embedded-controller firmware, UEFI/SoC firmware,
# pending fwupd updates, pending apt installs, uptime, fabric NIC addresses and MTU.
# Run it before and after every update window and diff the output: benchmark numbers from nodes
# on different firmware are not comparable (see docs/BIAS-AUDIT.md).
#
# Hosts come from the arguments or $CLUSTER_HOSTS (space separated). Each must be reachable
# with key-based ssh (BatchMode). No sudo is used.
# FABRIC_NICS: regex of the netdev names to report (default: the GB10 ConnectX-7 names).
set -u
HOSTS="${*:-${CLUSTER_HOSTS:?pass hosts as arguments or set CLUSTER_HOSTS}}"
NICRE="${FABRIC_NICS:-enp1s0f0np0|enP2p1s0f0np0|enP7s7}"
MTUNIC="${MTU_NIC:-enp1s0f0np0}"
printf "%-12s %-20s %-10s %-16s %-12s %-27s %-5s %-5s %-6s %s\n" host kernel driver vbios EC UEFI/SoC fwupd apt uptime nics
for h in $HOSTS; do
  out=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" "NICRE='$NICRE' MTUNIC='$MTUNIC' bash -s" <<'EOF'
k=$(uname -r)
read drv vb < <(nvidia-smi --query-gpu=driver_version,vbios_version --format=csv,noheader | tr -d , | head -1)
fw=$(fwupdmgr get-devices --json 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)["Devices"]
ec=[x["Version"] for x in d if x.get("Name")=="Embedded Controller"]
uefi=[x["Version"] for x in d if x.get("Name")=="UEFI Device Firmware"]
print((ec or ["-"])[0], ",".join(uefi) or "-")')
pend=$(fwupdmgr get-updates --json --no-unreported-check --no-metadata-check 2>/dev/null | python3 -c 'import json,sys
try: print(len(json.load(sys.stdin).get("Devices",[])))
except Exception: print(0)')
apt=$(apt-get -s -o Debug::NoLocking=1 full-upgrade 2>/dev/null | grep -c "^Inst ")
up=$(awk '{printf "%dm", $1/60}' /proc/uptime)
nics=$(ip -4 -br addr show | awk -v re="^($NICRE)\$" '$1 ~ re {print $1":"$3}' | tr '\n' ' ')
mtu=$(cat /sys/class/net/$MTUNIC/mtu 2>/dev/null)
echo "$k|$drv|$vb|$fw|$pend|$apt|$up|$nics mtu=$mtu"
EOF
)
  IFS='|' read -r k drv vb fw pend apt up nics <<< "$out"
  read -r ec uefi <<< "$fw"
  printf "%-12s %-20s %-10s %-16s %-12s %-27s %-5s %-5s %-6s %s\n" "$h" "$k" "$drv" "$vb" "$ec" "$uefi" "$pend" "$apt" "$up" "$nics"
done
