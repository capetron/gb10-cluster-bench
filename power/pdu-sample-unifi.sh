#!/usr/bin/env bash
# pdu-sample-unifi.sh OUT.jsonl [INTERVAL_S=5] - READ-ONLY wall-power sampler for a UniFi
# power distribution unit (tested with the USP-PDU-Pro, model code USPPDUP) managed by a UniFi
# OS console (UDM / UDM Pro / Cloud Key).
#
# One login POST, then GET stat/device every INTERVAL_S; writes one JSON line per NEW device
# record. The controller refreshes PDU readings about every 30 s, so records whose last_seen did
# not change are skipped. Never PUTs or POSTs anything except the login. Stop it with kill;
# it logs in again if the session expires.
#
# Output line: {"t": <epoch>, "last_seen": ..., "total_w": ..., "outlets": [{"i": <index>,
#   "name": ..., "w": ..., "a": ..., "v": ..., "pf": ...}, ...]}
# power/pdu-join.py attaches these to benchmark windows.
#
# Configuration (environment):
#   UNIFI_URL        console base URL, e.g. https://192.168.1.1          (required)
#   UNIFI_USER       a LOCAL read-only admin account on the console        (required)
#   UNIFI_PASS_FILE  file holding that account's password (mode 600)      (required)
#   UNIFI_SITE       site name (default: default)
#   PDU_MODEL        model code to select (default: USPPDUP)
#   OUTLET_MIN / OUTLET_MAX  outlet index range to record (default 1..32)
#
# Record the outlet INDEX, not the controller label: in our rack six of the labels were wrong.
# Verify the index-to-node map once with a coded load (for example the node/gpu-burn.sh burn on
# one node at a time) and keep that map next to your results.
# Uses -k because consoles ship a self-signed certificate; point it at a console you trust.
set -u
OUT="${1:?out.jsonl}"; IV="${2:-5}"
: "${UNIFI_URL:?set UNIFI_URL}" "${UNIFI_USER:?set UNIFI_USER}" "${UNIFI_PASS_FILE:?set UNIFI_PASS_FILE}"
SITE="${UNIFI_SITE:-default}"; MODEL="${PDU_MODEL:-USPPDUP}"
OMIN="${OUTLET_MIN:-1}"; OMAX="${OUTLET_MAX:-32}"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
U="$UNIFI_URL/proxy/network/api/s/$SITE"
login() {
  jq -n --arg u "$UNIFI_USER" --rawfile p "$UNIFI_PASS_FILE" '{username:$u,password:($p|rtrimstr("\n"))}' > "$W/l.json"
  curl -sk -c "$W/jar" -o /dev/null -H 'Content-Type: application/json' \
    -X POST "$UNIFI_URL/api/auth/login" --data @"$W/l.json"; rm -f "$W/l.json"
}
login; last=""
while true; do
  j=$(curl -sk -b "$W/jar" "$U/stat/device" 2>/dev/null)
  if ! echo "$j" | jq -e '.data' >/dev/null 2>&1; then login; sleep "$IV"; continue; fi
  rec=$(echo "$j" | jq -c --arg now "$(date +%s.%N)" --arg m "$MODEL" --argjson lo "$OMIN" --argjson hi "$OMAX" \
    '.data[] | select(.model==$m) |
     {t: ($now|tonumber), last_seen: .last_seen, total_w: .outlet_ac_power_consumption,
      outlets: [.outlet_table[] | select(.index>=$lo and .index<=$hi) |
        {i: .index, name: .name, w: ((.outlet_power // "0")|tonumber), a: ((.outlet_current // "0")|tonumber),
         v: ((.outlet_voltage // "0")|tonumber), pf: ((.outlet_power_factor // "0")|tonumber)}]}' | head -1)
  ls=$(echo "$rec" | jq -r '.last_seen' 2>/dev/null)
  if [ -n "$rec" ] && [ "$ls" != "$last" ]; then echo "$rec" >> "$OUT"; last="$ls"; fi
  sleep "$IV"
done
