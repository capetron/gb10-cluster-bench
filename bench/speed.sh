#!/bin/bash
# speed.sh: the ONE speed battery every candidate gets, so results from different models,
# engines and cluster sizes are comparable.
#
# usage: speed.sh RESULT.json BASE SERVED [MAXLEN] [SEED]
#   RESULT.json  result file (phases are merged in, never overwritten)
#   BASE         OpenAI-compatible base URL of the engine, e.g. http://localhost:8000/v1
#   SERVED       served model name
#   MAXLEN       engine max_model_len (decides whether the 120k needle and the 128k prefill run)
#   SEED         run seed (prompt uniqueness across runs); default speed-<date>
#
# Env knobs:
#   SKIP_GATE=<reason>      gate already run and judged (speedgate.sh sets this)
#   GATE_TOKENS=64          raise for models that think despite enable_thinking=false
#   PREFILL_LENGTHS=8192,32768  cap the prefill sweep
#   SKIP_CEILING=1 SKIP_CODE=1 SKIP_PREFILL=1
#   LADDER="1,4,8,16,32"    steady-state ladder (users)
#   CEIL="48,64,96,128,192" ceiling discovery levels
#   PFX=<label prefix>      labels become <PFX>steady_prose etc. (A/B arms in one file)
#   PDU_TRACE=<file.jsonl>  optional wall-power trace (power/pdu-sample-unifi.sh or your own
#                           sampler in the same format); joined to every steady row
#   PDU_OUTLETS=5,6         outlets that feed the units in this engine (with PDU_TRACE)
#   OPENAI_API_KEY          sent as a Bearer token if set
#
# Method (docs/METHODOLOGY.md): correctness gate first; one warmup burst per level; steady-state
# closed loop, tokens counted only inside the window; 3 reps with shuffled level order;
# temperature 0, ignore_eos, thinking off, 512 output tokens; prose AND code content reported
# separately (speculative-decoding gains are content-dependent); ceiling = highest load whose
# per-user p50 stays >= 10 tok/s with TTFT p95 <= 10 s; prefill at c=1 at 8k/32k/128k x3 with a
# unique uid per prompt (prefix cache defeated).
set -uo pipefail
OUT="$1"; BASE="$2"; M="$3"; MAXLEN="${4:-131072}"; SEED="${5:-speed-$(date +%Y%m%d)}"
P="${PFX:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
H="$HERE/llm-prefill-bench.py"
J="$HERE/../power/pdu-join.py"
LADDER="${LADDER:-1,4,8,16,32}"
CEIL="${CEIL:-48,64,96,128,192}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
join_power() {
  [ -n "${PDU_TRACE:-}" ] || return 0
  python3 "$J" "$OUT" "$1" "$PDU_TRACE" --outlets "${PDU_OUTLETS:?set PDU_OUTLETS with PDU_TRACE}" --write | tail -3
}

NEEDLES=16384; [ "$MAXLEN" -ge 126000 ] && NEEDLES=16384,120000
if [ -n "${SKIP_GATE:-}" ]; then log "gate already run and judged (${SKIP_GATE})"; else
log "gate ($NEEDLES)"
python3 "$H" "$BASE" "$M" "$OUT" gate --label "${P}gate" --needles $NEEDLES --gate-tokens "${GATE_TOKENS:-64}" --run-seed "$SEED" 2>&1 | tail -3
if ! python3 - "$OUT" "${P}gate" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); g = d["phases"][sys.argv[2]]
rows = g.get("rows") or g.get("checks") or []
bad = [r for r in rows if not r.get("pass")]
print("gate %d/%d" % (len(rows) - len(bad), len(rows)))
sys.exit(1 if bad else 0)
EOF
then if [ -n "${GATE_OVERRIDE:-}" ]; then log "GATE FAILED but GATE_OVERRIDE set: ${GATE_OVERRIDE}"; else log "GATE FAILED - no throughput numbers for this arm"; exit 3; fi; fi
fi

log "steady prose $LADDER"
python3 "$H" "$BASE" "$M" "$OUT" steady --label "${P}steady_prose" --conc "$LADDER" --window 90 --ramp 20 \
  --reps 3 --max-tokens 512 --content prose --run-seed "$SEED-prose" --order-seed 25 2>&1 | grep -v '^{"warmup' | tail -16
join_power "${P}steady_prose"

if [ -z "${SKIP_CODE:-}" ]; then
  log "steady code $LADDER"
  python3 "$H" "$BASE" "$M" "$OUT" steady --label "${P}steady_code" --conc "$LADDER" --window 60 --ramp 20 \
    --reps 3 --max-tokens 512 --content code --run-seed "$SEED-code" --order-seed 26 2>&1 | grep -v '^{"warmup' | tail -16
  join_power "${P}steady_code"
fi

if [ -z "${SKIP_CEILING:-}" ]; then
  top=$(python3 -c "
import json, sys; d=json.load(open(sys.argv[1])); s=d['phases'][sys.argv[2]]['summary']
s=sorted(s,key=lambda r:r['concurrency']); print(s[-1]['per_user_tok_s_median'] or 0)" "$OUT" "${P}steady_prose")
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 10 else 1)" "$top"; then
    log "ceiling discovery $CEIL (per-user at top of ladder $top)"
    python3 "$H" "$BASE" "$M" "$OUT" steady --label "${P}ceiling_prose" --conc "$CEIL" --window 60 --ramp 30 \
      --reps 3 --max-tokens 512 --content prose --stop-below 10 --stop-ttft 10 --run-seed "$SEED-ceil" --order-seed 27 2>&1 \
      | grep -v '^{"warmup' | tail -14
    join_power "${P}ceiling_prose"
  else
    log "ceiling is inside the ladder (per-user at top level $top < 10)"
  fi
fi

if [ -z "${SKIP_PREFILL:-}" ]; then
  L=8192,32768; [ "$MAXLEN" -ge 133000 ] && L=8192,32768,131040
  [ -n "${PREFILL_LENGTHS:-}" ] && L="$PREFILL_LENGTHS"
  log "prefill $L"
  python3 "$H" "$BASE" "$M" "$OUT" prefill --label "${P}prefill" --lengths "$L" --conc 1 --reps 3 --max-tokens 32 \
    --run-seed "$SEED-prefill" 2>&1 | tail -10
fi
log "SPEEDDONE $OUT"
