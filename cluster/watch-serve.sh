#!/bin/bash
# watch-serve.sh CONTAINER API_URL [HOST ...]
# Exits 0 when the API answers, 1 the moment any rank's container has exited, 2 on timeout
# (40 minutes). A multi-node vLLM job that loses one rank hangs forever instead of failing, so
# watch the containers on every node, not just the endpoint.
#   CONTAINER  container name used on every node
#   API_URL    e.g. http://<rank0-host>:8000/v1/models
#   HOST ...   nodes running a rank (default: $CLUSTER_HOSTS)
set -u
C="${1:?container}"; URL="${2:?api url, e.g. http://rank0:8000/v1/models}"; shift 2
HOSTS="${*:-${CLUSTER_HOSTS:?pass hosts or set CLUSTER_HOSTS}}"
for i in $(seq 1 120); do
  curl -sf --max-time 5 "$URL" >/dev/null 2>&1 && { echo "READY after ~$((i*20))s"; exit 0; }
  for h in $HOSTS; do
    st=$(ssh -o ConnectTimeout=6 -o BatchMode=yes "$h" "docker ps -a --filter name=$C --format '{{.Status}}'" 2>/dev/null)
    case "$st" in Exited*|"") echo "DEAD $h ($st) - dump its logs before removing the container: ssh $h docker logs $C"; exit 1;; esac
  done
  sleep 20
done
echo TIMEOUT; exit 2
