#!/usr/bin/env python3
"""fleet-maint.py - deterministic fleet OS + firmware maintenance (no LLM decisions).

  fleet-maint.py collect [--hosts a,b] [--class c] [--notify auto|always|never] [--json]
      READ-ONLY. Collect every host (or a subset), write ~/.local/state/fleet-maint/
      hosts/<host>.json + fleet.json, print the state table. --notify auto calls the
      FLEET_MAINT_NOTIFY hook only when the alert set changed since the last run, and
      always on Mondays (the weekly table).
  fleet-maint.py report [--markdown]
      Render the saved state (no ssh).
  fleet-maint.py plan --class c
      Dry run of the applier: gate verdict, plus what WOULD happen if the gate were open
      (fresh read-only collect). Touches nothing. Review it before writing an approval tag.
  fleet-maint.py apply --class c [--now]
      The privileged applier. Fails closed without the class approval tag. --now runs
      outside the window only with a separate live tag "<approval_tag>-now".
  fleet-maint.py hold <host|class-<name>|ALL> --reason R [--hours N]
  fleet-maint.py unhold <host|class-<name>|ALL>
      Central maintenance holds on the runner (the remote convention is in the docs).
  fleet-maint.py check-policy

Exit codes: 0 ok / nothing to do / report-only, 1 failed, 3 canary failed,
4 kill switch or pause, 5 refused by the approval gate, 6 deferred (busy / outside window).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fleetmaint as fm  # noqa: E402

EXIT = {"completed": 0, "nothing-to-do": 0, "report-only": 0, "dry-run": 0,
        "failed": 1, "parity-drift": 1, "canary-failed": 3, "killswitch": 4,
        "refused": 5, "deferred": 6, "outside-window": 6}


def _hosts(policy, args) -> list[str]:
    if args.hosts:
        hs = [h.strip() for h in args.hosts.split(",") if h.strip()]
        unknown = [h for h in hs if h not in policy["hosts"]]
        if unknown:
            raise SystemExit(f"not in policy: {unknown}")
        return hs
    if args.cls:
        return fm.class_hosts(policy, args.cls)
    return [h for h, hc in policy["hosts"].items() if hc.get("collect", True)]


def cmd_collect(policy, args) -> int:
    hosts = _hosts(policy, args)
    runner = fm.SSHRunner(policy)
    states, _ = fm.gather(policy, hosts, runner, fm.fetch_lab)
    fm.write_states(states)
    allst = fm.read_states()
    al = fm.alerts(policy, allst)
    fleet = {"collected_at": time.time(), "hosts": sorted(allst), "alerts": al}
    (fm.state_dir() / "fleet.json").write_text(json.dumps(fleet, indent=1))
    table = fm.state_table(policy, allst)
    if args.json:
        print(json.dumps({h: states[h] for h in hosts}, indent=1, default=str))
    else:
        print(table)
        print("Alerts:\n" + ("\n".join(f"- {a}" for a in al) if al else "- none"))
    weekly = _dt.date.today().weekday() == 0
    changed = fm.alerts_changed(al)
    if args.notify == "always" or (args.notify == "auto" and (weekly or (changed and al))):
        subject = (f"Fleet maintenance {'weekly' if weekly else 'alert'} "
                   f"{_dt.date.today().isoformat()}: {len(al)} alert(s)")
        body = ("Read-only fleet maintenance collector (fleet-maint). Nothing was changed.\n\n"
                + ("Alerts:\n" + "\n".join(f"- {a}" for a in al) + "\n\n" if al else "")
                + table + f"\nState: {fm.state_dir()}\n")
        print(fm.notify(subject, body, "warning" if al else "info"))
    return 0


def cmd_report(policy, args) -> int:
    st = fm.read_states()
    print(fm.state_table(policy, st))
    al = fm.alerts(policy, st)
    print("Alerts:\n" + ("\n".join(f"- {a}" for a in al) if al else "- none"))
    for cname, c in policy["classes"].items():
        if c.get("as_set"):
            pt = fm.parity_table(st, fm.class_hosts(policy, cname))
            print(f"\nParity {cname}: drift in {pt['drift'] or 'nothing'}")
            cols = [c_ for c_ in pt["columns"] if c_ not in ("apt_pending",)]
            print("| host | " + " | ".join(cols) + " |")
            print("|---" * (len(cols) + 1) + "|")
            for h, row in pt["rows"].items():
                print(f"| {h} | " + " | ".join(str(row.get(c_, "")) for c_ in cols) + " |")
    return 0


def _post_hook(cls_name: str, hosts: list[str]) -> dict:
    """Set classes that verify parity=gb10: run the read-only node/parity-check.sh over the
    class hosts after the update (it never fixes anything)."""
    policy = fm.load_policy()
    if (policy["classes"][cls_name].get("verify") or {}).get("parity") != "gb10":
        return {}
    script = Path(__file__).resolve().parent.parent / "node" / "parity-check.sh"
    try:
        p = subprocess.run(["bash", str(script), *hosts], capture_output=True, text=True, timeout=300)
        return {"gb10_parity_rc": p.returncode, "gb10_parity": p.stdout[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"gb10_parity_error": str(e)}


def _record(result: dict) -> Path:
    d = fm.state_dir() / "runs" / f"{result['class']}-{result['run_id']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(result, indent=1, default=str))
    latest = fm.state_dir() / "runs" / f"{result['class']}-latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(d.name)
    except OSError:
        pass
    for h, r in (result.get("hosts") or {}).items():
        if r.get("after"):
            fm.write_states({h: r["after"]})
    return d


def _summary(result: dict) -> str:
    lines = [f"fleet-maint {result['class']}: {result['status']} - {result.get('reason')}"]
    if result.get("approval"):
        lines.append(f"  approval: {result['approval']}")
    plan = result.get("probe") or result.get("plan") or {}
    if plan.get("hosts"):
        lines.append(f"  plan ({'if the gate were open' if result.get('probe') else 'live'}): "
                     f"{plan.get('verdict')} - {plan.get('reason')}")
        for h, e in plan["hosts"].items():
            extra = f" pkgs={len(e['packages'])} fw={len(e['firmware'])}" if e["state"] == "ready" else ""
            lines.append(f"    {h:<11} {e['state']:<10}{extra} {'; '.join(e['reasons'])[:150]}")
        if plan.get("order"):
            lines.append(f"    order: {' -> '.join(plan['order'])} (canary {plan.get('canary')})")
    for h, r in (result.get("hosts") or {}).items():
        lines.append(f"  {h:<11} {r.get('status')}: {r.get('reason')}")
    for h in result.get("skipped") or []:
        lines.append(f"  {h:<11} skipped")
    if result.get("parity"):
        lines.append(f"  parity drift: {result['parity']['drift'] or 'none'}")
    return "\n".join(lines)


def cmd_plan(policy, args) -> int:
    res = fm.run_class(policy, args.cls, fm.SSHRunner(policy), dry_run=True)
    print(_summary(res))
    return 0


def cmd_apply(policy, args) -> int:
    cls = policy["classes"][args.cls]
    override = False
    if args.now:
        tag = str(cls.get("approval_tag") or "") + "-now"
        ok, why, _ = fm.check_approval(tag, fm.class_hosts(policy, args.cls), cls["as_set"])
        if not ok:
            print(f"--now refused: {why}")
            return 5
        override = True
    try:
        with fm.RunLock():
            res = fm.run_class(policy, args.cls, fm.SSHRunner(policy),
                               override_window=override, post_hook=_post_hook)
    except RuntimeError as e:
        print(str(e))
        return 6
    touched = bool(res.get("hosts"))
    if touched:
        d = _record(res)
        text = _summary(res) + f"\n\nRun record: {d}/run.json"
        sev = "info" if res["status"] == "completed" else "error"
        print(fm.notify(f"fleet-maint {args.cls} {res['status']}", text, sev))
        if res["status"] in ("canary-failed", "failed", "parity-drift"):
            print(fm.notify(f"ACTION NEEDED: fleet-maint {args.cls} stopped ({res['status']})",
                            text + "\n\nConsider `fleet-maint.py hold ALL --reason ...` or the pause file "
                            "before the next window.", "error"))
    else:
        (fm.state_dir()).mkdir(parents=True, exist_ok=True)
        with open(fm.state_dir() / "apply-log.jsonl", "a") as fh:
            fh.write(json.dumps({"at": time.time(), "class": args.cls, "status": res["status"],
                                 "reason": res.get("reason")}) + "\n")
    print(_summary(res))
    return EXIT.get(res["status"], 1)


def cmd_hold(policy, args) -> int:
    d = fm.central_holds_dir()
    d.mkdir(parents=True, exist_ok=True)
    exp = int(time.time() + args.hours * 3600) if args.hours else None
    body = f"reason={args.reason} by={os.environ.get('USER')} at={int(time.time())}" + \
        (f" expires={exp}" if exp else "")
    (d / args.target).write_text(body + "\n")
    print(f"hold written: {d / args.target}: {body}")
    return 0


def cmd_unhold(policy, args) -> int:
    f = fm.central_holds_dir() / args.target
    if f.exists():
        f.unlink()
        print(f"removed {f}")
    else:
        print(f"no hold at {f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--hosts")
    c.add_argument("--class", dest="cls")
    c.add_argument("--notify", choices=("auto", "always", "never"), default="never")
    c.add_argument("--json", action="store_true")
    sub.add_parser("report").add_argument("--markdown", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("--class", dest="cls", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--class", dest="cls", required=True)
    a.add_argument("--now", action="store_true")
    h = sub.add_parser("hold")
    h.add_argument("target")
    h.add_argument("--reason", required=True)
    h.add_argument("--hours", type=float)
    u = sub.add_parser("unhold")
    u.add_argument("target")
    sub.add_parser("check-policy")
    args = ap.parse_args(argv)
    try:
        policy = fm.load_policy()
    except (OSError, fm.PolicyError, ValueError) as e:
        print(f"policy error: {e}", file=sys.stderr)
        return 1
    if getattr(args, "cls", None) and args.cls not in policy["classes"]:
        print(f"unknown class {args.cls}; have {sorted(policy['classes'])}", file=sys.stderr)
        return 1
    if args.cmd == "check-policy":
        for n, c_ in policy["classes"].items():
            print(f"{n:<20} auto_apply={c_['auto_apply']!s:<5} updates={c_['updates']:<8} "
                  f"firmware={c_['firmware']!s:<5} reboot={c_['reboot']:<11} "
                  f"hosts={','.join(fm.class_hosts(policy, n))}")
        return 0
    if args.cmd in ("apply",):
        stop = fm.stopped()
        if stop:
            print(stop)
            return 4
    return {"collect": cmd_collect, "report": cmd_report, "plan": cmd_plan, "apply": cmd_apply,
            "hold": cmd_hold, "unhold": cmd_unhold}[args.cmd](policy, args)


if __name__ == "__main__":
    sys.exit(main())
