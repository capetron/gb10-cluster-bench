#!/usr/bin/env python3
"""pdu-join.py RESULT.json PHASE_LABEL PDU.jsonl --outlets 5,6,... [--names 5=node-a,6=node-b] [--write]

Attach wall power from a PDU trace (power/pdu-sample-unifi.sh, or any sampler that writes the
same JSONL shape: {"t": epoch, "outlets": [{"i": index, "w": watts}, ...]}) to every row of a
steady-phase result from bench/llm-prefill-bench.py (rows carry window_start_epoch and
window_end_epoch).

For each row: mean watts per outlet over PDU records whose sample time falls inside the window,
the sum over the chosen outlets, tokens per joule (aggregate_output_tok_s / watts) and the sample
count. A PDU that refreshes about every 30 s gives about 4 samples in a 120 s window; windows with
fewer than 2 samples are flagged. Scope: wall AC of the listed outlets only; switch power is
excluded unless you list the switch outlet.

With --write the power block is stored back into the result under phases[LABEL]["power"].
"""
import argparse
import json
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result")
    p.add_argument("label")
    p.add_argument("pdu")
    p.add_argument("--outlets", required=True, help="comma-separated outlet indexes feeding the engine's nodes")
    p.add_argument("--names", default="", help="optional outlet=name pairs, e.g. 5=node-a,6=node-b")
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    outs = [int(x) for x in a.outlets.split(",")]
    names = {int(k): v for k, v in (x.split("=", 1) for x in a.names.split(",") if "=" in x)}
    recs = [json.loads(line) for line in open(a.pdu) if line.strip()]
    doc = json.load(open(a.result))
    ph = doc["phases"][a.label]
    rows_out = []
    for r in ph["rows"]:
        t0, t1 = r["window_start_epoch"], r["window_end_epoch"]
        inwin = [x for x in recs if t0 <= x["t"] <= t1]
        per = {}
        for o in outs:
            v = [y["w"] for x in inwin for y in x["outlets"] if y["i"] == o]
            per[names.get(o, "outlet-%d" % o)] = round(statistics.mean(v), 1) if v else None
        tot = sum(v for v in per.values() if v is not None) if inwin else None
        agg = r.get("aggregate_output_tok_s")
        rows_out.append({"concurrency": r["concurrency"], "rep": r["rep"], "samples": len(inwin),
                         "low_sample_flag": len(inwin) < 2, "watts_total": round(tot, 1) if tot else None,
                         "watts_per_outlet": per, "aggregate_output_tok_s": agg,
                         "tokens_per_joule": round(agg / tot, 4) if tot and agg else None})
    summ = []
    for c in sorted({x["concurrency"] for x in rows_out}):
        w = [x["watts_total"] for x in rows_out if x["concurrency"] == c and x["watts_total"]]
        j = [x["tokens_per_joule"] for x in rows_out if x["concurrency"] == c and x["tokens_per_joule"]]
        summ.append({"concurrency": c, "watts_median": round(statistics.median(w), 1) if w else None,
                     "tokens_per_joule_median": round(statistics.median(j), 4) if j else None})
    block = {"source": "PDU outlet trace %s" % a.pdu.split("/")[-1],
             "scope": "wall AC, sum of outlets %s" % a.outlets,
             "summary": summ, "rows": rows_out}
    print(json.dumps(summ, indent=1))
    if a.write:
        ph["power"] = block
        json.dump(doc, open(a.result, "w"), indent=1)


if __name__ == "__main__":
    main()
