#!/bin/bash
# speedgate.sh RESULT.json BASE SERVED MAXLEN SEED [LOG]
# Runs the correctness gate ONCE, applies the gate policy, then hands off to speed.sh.
#
# Gate policy: gate on the contexts actually measured.
#  - short-context checks (capital, primes, firewall, 17*23) must pass, else no throughput (exit 3).
#  - mental-arithmetic misses with thinking off (8347*291, 127*43) -> a think-on recheck
#    (gate-recheck.py) must pass; the miss is then recorded as a quality finding.
#  - needle failures withhold ONLY the contexts they cover: speed is published with a caveat, and
#    the prefill sweep is capped at the largest passing needle context (none passing -> no prefill).
#    Recorded as <PFX>gate_policy in the JSON.
# speed.sh then runs with SKIP_GATE, so the judged gate is the one on record.
set -u
J="$1"; B="$2"; M="$3"; ML="$4"; SEED="$5"; LOG="${6:-/dev/stdout}"
S="$(cd "$(dirname "$0")" && pwd)"; P="${PFX:-}"; H="$S/llm-prefill-bench.py"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
NEEDLES=16384; [ "$ML" -ge 126000 ] && NEEDLES=16384,120000
log "gate ($NEEDLES) via speedgate"
python3 "$H" "$B" "$M" "$J" gate --label "${P}gate" --needles $NEEDLES --gate-tokens "${GATE_TOKENS:-64}" --run-seed "$SEED" >> "$LOG" 2>&1
V=$(python3 - "$J" "${P}gate" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))["phases"][sys.argv[2]]
rows = g.get("rows") or g.get("checks") or []
bad = [r for r in rows if not r.get("pass")]
arith = [r for r in bad if r.get("q", "").startswith(("What is 8347", "What is 127"))]
needle_ok = [r.get("prompt_tokens") or int(r["q"].split()[2]) for r in rows if "needle" in r.get("q", "") and r.get("pass")]
needle_bad = [r.get("prompt_tokens") or int(r["q"].split()[2]) for r in rows if "needle" in r.get("q", "") and not r.get("pass")]
other = [r for r in bad if r not in arith and "needle" not in r.get("q", "")]
print("OTHER" if other else "OK", 1 if arith else 0, max(needle_ok or [0]), ",".join(str(x) for x in needle_bad) or "-", "%d/%d" % (len(rows) - len(bad), len(rows)))
PY
)
set -- $V; STATE=$1; ARITH=$2; LP=$3; NB=$4; SCORE=$5
log "gate $SCORE: state=$STATE arith_miss=$ARITH max_passing_needle=$LP failing_needles=$NB"
[ "$STATE" = OTHER ] && { log "GATE FAILED on a short-context check - no throughput numbers"; exit 3; }
NOTE="gate $SCORE"
if [ "$ARITH" = 1 ]; then
  RL="${LOG%.log}-gate-recheck.log"; [ "$LOG" = /dev/stdout ] && RL="$(mktemp)"
  if python3 "$S/gate-recheck.py" "$B" "$M" > "$RL" 2>&1; then NOTE="$NOTE; thinking-off arithmetic miss, think-on recheck of all three canaries passed"
  else log "think-on arithmetic recheck FAILED - no throughput numbers"; exit 3; fi
fi
PL=""
if [ "$NB" != "-" ]; then
  PL=$(python3 -c "
lp=$LP; ml=$ML; c=[x for x in (8192,32768,131040) if x<=ml and x<=lp*1.1]
if lp and lp not in c and all(abs(lp-x)>lp*0.1 for x in c): c.append(lp)
print(','.join(str(x) for x in sorted(c)))")
  NOTE="$NOTE; FAILS long-context retrieval at ${NB} tokens; speed valid up to ${LP} tokens; prefill sweep capped at ${PL:-none}"
  [ -z "$PL" ] && export SKIP_PREFILL=1
fi
python3 - "$J" "${P}gate_policy" "$NOTE" "$LP" "$NB" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
d[sys.argv[2]] = {"note": sys.argv[3], "max_passing_needle_tokens": int(sys.argv[4]), "failing_needles": sys.argv[5],
                  "policy": "needle failures withhold only the contexts they cover"}
json.dump(d, open(p, "w"), indent=1)
PY
log "gate judged: $NOTE"
SKIP_GATE="$NOTE" PREFILL_LENGTHS="${PL:-${PREFILL_LENGTHS:-}}" bash "$S/speed.sh" "$J" "$B" "$M" "$ML" "$SEED" >> "$LOG" 2>&1
