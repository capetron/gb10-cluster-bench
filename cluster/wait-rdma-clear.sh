#!/bin/bash
# wait-rdma-clear.sh [HOST ...]
# Blocks until every node reports zero RDMA memory regions (mr) in `rdma resource show`.
# Run it between tearing down a multi-node engine and launching the next one: a relaunch while
# the previous job's memory regions are still registered can fail NCCL init or hang.
# Exits 0 when clear, 1 after 5 minutes with the busy nodes listed. Hosts: args or $CLUSTER_HOSTS.
set -u
HOSTS="${*:-${CLUSTER_HOSTS:?pass hosts or set CLUSTER_HOSTS}}"
busy=""
for i in $(seq 1 60); do
  busy=""
  for h in $HOSTS; do
    mrs=$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" "rdma resource show 2>/dev/null | grep -o 'mr [0-9]*' | head -1 | awk '{print \$2}'" 2>/dev/null)
    [ "${mrs:-0}" != "0" ] && busy="$busy $h(mr=$mrs)"
  done
  [ -z "$busy" ] && { echo "clear after ${i}x5s"; exit 0; }
  sleep 5
done
echo "STILL BUSY:$busy"; exit 1
