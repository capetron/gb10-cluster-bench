#!/usr/bin/env python3
"""fleetmaint.py - deterministic OS + firmware maintenance for a small GPU fleet.

Code, not agents: every decision here is a rule in this file or in the policy YAML
($FLEET_MAINT_POLICY, default policy.yaml next to this file; start from
policy.example.yaml). No model is consulted. The model to copy is a plain NixOS
system.autoUpgrade timer (build nightly, activate at next boot, never reboot on its own,
mail on failure), not an LLM deciding what to do.

Three layers:
  collect  READ-ONLY. ssh <host> bash -s < remote/collect.sh, parsed into one JSON
           state per host under ~/.local/state/fleet-maint/hosts/<host>.json.
  plan     Pure functions over (policy, states, approvals, clock): which class may run,
           which host is blocked and why. Unit tested in test_fleetmaint.py.
  apply    For a class whose policy allows it AND whose approval tag is live AND whose
           window is open AND whose hosts are idle AND with no kill switch: canary
           first, then the rest; non-interactive apt full-upgrade (or an explicit
           security package list) and fwupdmgr update; reboot, wait, verify, record
           before and after. A canary failure stops the class. Never power-cycles.

Fails closed everywhere: an unreadable approvals file, a hub that does not answer, a
docker that cannot be listed, a hold file with a past expiry - all of them block.

Stdlib + PyYAML only. CLI: fleet-maint.py. Tests: test_fleetmaint.py.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import yaml

HERE = Path(__file__).resolve().parent
HOME = Path(os.path.expanduser("~"))
COLLECT_SH = HERE / "remote" / "collect.sh"

UPDATE_TYPES = ("none", "security", "full")
REBOOT_MODES = ("never", "if-required")
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# A standing approval longer than this is refused: approvals must be renewed, never
# left to authorise next quarter's run by accident.
MAX_APPROVAL_DAYS = 92
TERMINAL_JOB_STATES = {"done", "failed", "stopped", "exited", "cancelled", "canceled", "error"}
PKG_RE = re.compile(r"^[a-z0-9][a-z0-9+.\-]{0,127}$")
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,63}$")
DKMS_FAIL_RE = re.compile(r"(?i)(dkms[^\n]*(error|fail))|Bad return status for module build|^Error! ", re.M)

CLASS_KEYS = {
    "description", "auto_apply", "approval_tag", "updates", "firmware", "reboot", "window",
    "max_parallel", "canary", "as_set", "busy", "forbid_packages", "require_reboot_tested",
    "verify", "reboot_timeout_s", "manual_only_reason", "apply_supported_os",
    "uu_blacklist_required",
}
BUSY_KEYS = {"any_container", "container_patterns", "container_ignore", "gpu_util_pct",
             "gpu_processes", "llm_lab", "ollama_loaded"}
VERIFY_KEYS = {"nics", "gpu", "failed_units", "services", "holds", "dkms", "parity",
               "services_ignore", "nics_ignore", "firmware_cleared"}
HOST_KEYS = {"class", "sudo_file", "lab_hub", "reboot_tested", "collect", "note", "ssh"}

DEFAULT_SERVICES_IGNORE = [
    "fwupd.service", "packagekit.service", "apt-daily*.service", "unattended-upgrades.service",
    "user@*.service", "systemd-*.service", "getty@*.service", "serial-getty@*.service",
    "session-*.scope", "udisks2.service", "polkit.service", "ModemManager.service",
    "snapd.service", "fwupd-refresh.service", "man-db.service", "motd-news.service",
]
DEFAULT_NICS_IGNORE = ["lo", "docker*", "veth*", "br-*", "virbr*", "vnet*", "cni*",
                       "flannel*", "tap*", "tun*", "podman*"]


class PolicyError(ValueError):
    pass


# ---------------------------------------------------------------------------
# paths and switches
# ---------------------------------------------------------------------------

def policy_path() -> Path:
    return Path(os.environ.get("FLEET_MAINT_POLICY", HERE / "policy.yaml"))


def state_dir() -> Path:
    return Path(os.environ.get("FLEET_MAINT_STATE", HOME / ".local" / "state" / "fleet-maint"))


def kill_switch_path() -> Path:
    return Path(os.environ.get("FLEET_MAINT_KILL", HOME / ".config" / "fleet-maint" / "kill"))


def pause_path() -> Path:
    return Path(os.environ.get("FLEET_MAINT_PAUSE", HOME / ".config" / "fleet-maint" / "pause"))


def central_holds_dir() -> Path:
    return Path(os.environ.get("FLEET_MAINT_HOLDS", HOME / ".config" / "fleet-maint" / "holds"))


def approvals_path() -> Path:
    return Path(os.environ.get(
        "FLEET_MAINT_APPROVALS", HOME / ".config" / "fleet-maint" / "approvals.json"))


def stopped() -> str | None:
    """Reason maintenance must not run at all, or None."""
    if kill_switch_path().exists():
        return f"kill switch present at {kill_switch_path()}"
    if pause_path().exists():
        return f"fleet-maint paused ({pause_path()})"
    return None


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

def parse_window(w: Any) -> dict[str, Any] | None:
    if w is None:
        return None
    if not isinstance(w, dict):
        raise PolicyError("window must be a mapping {days, start, end, tz}")
    days = w.get("days") or []
    if not days or any(d not in DAYS for d in days):
        raise PolicyError(f"window.days must be a non-empty subset of {DAYS}")
    out = {"days": list(days), "tz": str(w.get("tz") or "America/New_York")}
    for k in ("start", "end"):
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(w.get(k) or ""))
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
            raise PolicyError(f"window.{k} must be HH:MM")
        out[k] = (int(m.group(1)), int(m.group(2)))
    if out["end"] <= out["start"]:
        raise PolicyError("window.end must be after window.start (same day)")
    try:
        ZoneInfo(out["tz"])
    except Exception:
        raise PolicyError(f"window.tz unknown: {out['tz']}") from None
    return out


def validate_policy(p: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(p, dict) or not isinstance(p.get("classes"), dict) \
            or not isinstance(p.get("hosts"), dict):
        raise PolicyError("policy needs `classes:` and `hosts:` mappings")
    for name, c in p["classes"].items():
        if not isinstance(c, dict):
            raise PolicyError(f"class {name}: must be a mapping")
        extra = set(c) - CLASS_KEYS
        if extra:
            raise PolicyError(f"class {name}: unknown keys {sorted(extra)} (typos must not "
                              "silently weaken a policy)")
        c.setdefault("auto_apply", False)
        c.setdefault("updates", "none")
        c.setdefault("firmware", False)
        c.setdefault("reboot", "never")
        c.setdefault("max_parallel", 1)
        c.setdefault("as_set", False)
        c.setdefault("busy", {})
        c.setdefault("forbid_packages", [])
        c.setdefault("require_reboot_tested", False)
        c.setdefault("verify", {})
        c.setdefault("reboot_timeout_s", 1200)
        c.setdefault("apply_supported_os", ["ubuntu", "debian"])
        if c["updates"] not in UPDATE_TYPES:
            raise PolicyError(f"class {name}: updates must be one of {UPDATE_TYPES}")
        if c["reboot"] not in REBOOT_MODES:
            raise PolicyError(f"class {name}: reboot must be one of {REBOOT_MODES}")
        if c["auto_apply"] and not str(c.get("approval_tag") or "").strip():
            raise PolicyError(f"class {name}: auto_apply needs an approval_tag")
        if c["auto_apply"] and not c.get("window"):
            raise PolicyError(f"class {name}: auto_apply needs a window")
        c["window"] = parse_window(c.get("window"))
        if int(c["max_parallel"]) < 1:
            raise PolicyError(f"class {name}: max_parallel must be >= 1")
        extra = set(c["busy"]) - BUSY_KEYS
        if extra:
            raise PolicyError(f"class {name}: unknown busy keys {sorted(extra)}")
        extra = set(c["verify"]) - VERIFY_KEYS
        if extra:
            raise PolicyError(f"class {name}: unknown verify keys {sorted(extra)}")
        for rx in c["forbid_packages"]:
            try:
                re.compile(rx)
            except re.error as e:
                raise PolicyError(f"class {name}: bad forbid regex {rx!r}: {e}") from None
    for h, hc in p["hosts"].items():
        if not HOST_RE.match(h):
            raise PolicyError(f"host {h!r}: not a valid ssh destination")
        if not isinstance(hc, dict) or hc.get("class") not in p["classes"]:
            raise PolicyError(f"host {h}: class must be one of {sorted(p['classes'])}")
        extra = set(hc) - HOST_KEYS
        if extra:
            raise PolicyError(f"host {h}: unknown keys {sorted(extra)}")
        hc.setdefault("collect", True)
    for name, c in p["classes"].items():
        members = class_hosts(p, name)
        if c.get("canary") and c["canary"] not in members:
            raise PolicyError(f"class {name}: canary {c['canary']} is not a member")
    p.setdefault("defaults", {})
    return p


def load_policy(path: Path | None = None) -> dict[str, Any]:
    path = path or policy_path()
    with open(path) as fh:
        return validate_policy(yaml.safe_load(fh))


def class_hosts(policy: dict[str, Any], cls: str) -> list[str]:
    return [h for h, hc in policy["hosts"].items() if hc.get("class") == cls]


# ---------------------------------------------------------------------------
# approval gate (same file and shape as fleetfanout.approval_ok / prod_deploy_guard)
# ---------------------------------------------------------------------------

def check_approval(tag: str, hosts: list[str], as_set: bool, now: float | None = None,
                   path: Path | None = None) -> tuple[bool, str, list[str]]:
    """(allowed, reason, approved_hosts).

    An entry is {"contains": tag, "expires": epoch, "note": str, "hosts": [..]}. The
    `hosts` list is REQUIRED for fleet-maint: a standing approval names the machines it
    covers, so adding a host to the policy never silently extends the operator's approval to it.
    Fails closed on a missing, unreadable or malformed file, a missing tag, an expired
    tag, an expiry further out than MAX_APPROVAL_DAYS, or (as_set) partial coverage.
    """
    now = time.time() if now is None else now
    if not tag:
        return False, "class has no approval_tag", []
    path = path or approvals_path()
    try:
        entries = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return False, f"no approvals file at {path}", []
    except (OSError, ValueError) as e:
        return False, f"approvals file unreadable ({e}); failing closed", []
    if not isinstance(entries, list):
        return False, "approvals file is not a list; failing closed", []
    reasons = []
    for e in entries:
        if not isinstance(e, dict) or str(e.get("contains") or "") != tag:
            continue
        try:
            exp = float(e.get("expires") or 0)
        except (TypeError, ValueError):
            reasons.append("expires not a number")
            continue
        if exp <= now:
            reasons.append("expired")
            continue
        if exp > now + MAX_APPROVAL_DAYS * 86400:
            reasons.append(f"expiry more than {MAX_APPROVAL_DAYS} days out")
            continue
        eh = e.get("hosts")
        if not isinstance(eh, list) or not eh:
            reasons.append("entry has no hosts allowlist")
            continue
        ok_hosts = [h for h in hosts if h in eh]
        if not ok_hosts:
            reasons.append("hosts allowlist covers none of the class")
            continue
        if as_set and len(ok_hosts) != len(hosts):
            missing = sorted(set(hosts) - set(ok_hosts))
            reasons.append(f"set class but allowlist misses {missing}")
            continue
        return True, f"approved: {e.get('note') or tag}", ok_hosts
    if reasons:
        return False, f"approval {tag!r} refused: " + "; ".join(reasons), []
    return False, f"no approval for {tag!r} in {path}", []


def in_window(window: dict[str, Any] | None, now: float | None = None) -> bool:
    if not window:
        return False
    now = time.time() if now is None else now
    t = _dt.datetime.fromtimestamp(now, ZoneInfo(window["tz"]))
    if DAYS[t.weekday()] not in window["days"]:
        return False
    return window["start"] <= (t.hour, t.minute) < window["end"]


# ---------------------------------------------------------------------------
# collect: parse the remote script's output
# ---------------------------------------------------------------------------

def split_sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip()
            out.setdefault(cur, [])
        elif cur is not None:
            out[cur].append(line)
    return out


def _kv(lines: list[str]) -> dict[str, str]:
    d: dict[str, str] = {}
    for ln in lines:
        if "=" in ln:
            k, v = ln.split("=", 1)
            d[k.strip()] = v.strip()   # last one wins (dgx-release repeats OTA)
    return d


def _json(lines: list[str]) -> Any:
    try:
        return json.loads("\n".join(lines)) if lines else None
    except ValueError:
        return None


def _int(v: Any) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


APT_LINE = re.compile(r"^(?P<name>[^/\s]+)/(?P<suites>\S+)\s+(?P<new>\S+)\s+\S+"
                      r"(?:\s+\[upgradable from: (?P<old>[^\]]+)\])?")


def parse_apt_upgradable(lines: list[str]) -> list[dict[str, Any]]:
    pkgs = []
    for ln in lines:
        m = APT_LINE.match(ln.strip())
        if not m:
            continue
        suites = m.group("suites").split(",")
        pkgs.append({"name": m.group("name"), "new": m.group("new"), "old": m.group("old"),
                     "suites": suites,
                     "security": any(s.endswith("-security") for s in suites)})
    return pkgs


def parse_fwupd(updates: Any, devices: Any) -> dict[str, Any]:
    pending = []
    for d in (updates or {}).get("Devices", []) if isinstance(updates, dict) else []:
        rel = d.get("Releases") or [{}]
        pending.append({"device": d.get("Name"), "current": d.get("Version"),
                        "new": rel[0].get("Version"), "plugin": d.get("Plugin")})
    devs = []

    def walk(items):
        for d in items or []:
            if isinstance(d, dict):
                flags = d.get("Flags") or []
                if d.get("Version") and ("updatable" in flags or "updatable-hidden" in flags):
                    devs.append({"device": d.get("Name"), "version": d.get("Version")})
                walk(d.get("Children"))
    if isinstance(devices, dict):
        walk(devices.get("Devices"))
    return {"pending": pending, "devices": devs}


def parse_collect(host: str, text: str, rc: int = 0, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    s = split_sections(text)
    st: dict[str, Any] = {"host": host, "collected_at": now, "reachable": "end" in s,
                          "errors": []}
    if not st["reachable"]:
        tail = text.strip().splitlines()[-1:] if text.strip() else []
        st["errors"].append(f"collect failed rc={rc}: {' '.join(tail)[:200]}")
        return st
    meta = _kv(s.get("meta", []))
    st["os"] = {"kind": meta.get("os_kind"), "id": meta.get("os_id"),
                "version": meta.get("os_version"), "pretty": meta.get("os_pretty"),
                "nixos": meta.get("nixos") == "1", "fips": meta.get("fips")}
    st["hostname"] = meta.get("hostname")
    st["model"] = " ".join(x for x in (meta.get("dmi_vendor"), meta.get("dmi_product")) if x) or None
    st["ssh_user"] = meta.get("user")
    st["boot_id"] = meta.get("boot_id")
    up = _int(meta.get("uptime_s"))
    st["uptime_days"] = round(up / 86400, 1) if up is not None and up < 10 ** 9 else None
    st["kernel"] = {"running": meta.get("kernel"),
                    "latest_installed": meta.get("kernel_latest_installed")}
    st["dgx"] = {"name": meta.get("DGX_NAME"), "ota": meta.get("DGX_OTA_VERSION"),
                 "swbuild": meta.get("DGX_SWBUILD_VERSION")} if meta.get("DGX_NAME") else None
    reboot_reasons = []

    # packages
    pkg: dict[str, Any] = {"manager": None, "pending": None, "security": None,
                           "packages": [], "holds": [], "errors": []}
    if meta.get("has_apt-get") == "1":
        pkgs = parse_apt_upgradable(s.get("apt_upgradable", []))
        am = _kv(s.get("apt_meta", []))
        lm = _int(am.get("lists_mtime"))
        pkg.update(manager="apt", packages=pkgs, pending=len(pkgs),
                   security=sum(1 for p in pkgs if p["security"]),
                   holds=[h for h in s.get("apt_holds", []) if h.strip()],
                   errors=[e for e in s.get("apt_errors", []) if e.strip()],
                   lists_age_h=round((now - lm) / 3600, 1) if lm else None,
                   uu={"installed": am.get("uu_installed") == "1",
                       "enabled": am.get("uu_enabled") == "1",
                       "blacklist": am.get("uu_blacklist", "").split()})
        if am.get("reboot_required") == "1":
            reboot_reasons.append("reboot-required: " + am.get("reboot_required_pkgs", "").strip())
        if pkg["errors"]:
            pkg["pending"] = None   # a broken apt cannot be trusted to report a count
    elif st["os"]["nixos"]:
        pkg["manager"] = "nixos"
    elif meta.get("has_pacman") == "1":
        lines = s.get("pacman", [])
        method = next((ln.split("=", 1)[1] for ln in lines if ln.startswith("method=")), "")
        ups = [ln for ln in lines if ln and not ln.startswith("method=")]
        pkg.update(manager="pacman", pending=len(ups), method=method,
                   packages=[{"name": ln.split()[0]} for ln in ups])
    elif meta.get("os_kind") == "Darwin":
        pkg["manager"] = "macos"
    st["pkg"] = pkg

    kr, kl = st["kernel"]["running"], st["kernel"]["latest_installed"]
    if pkg["manager"] == "apt" and kr and kl and kr != kl:
        reboot_reasons.append(f"newer kernel installed ({kl}) than running ({kr})")

    if "nixos" in s:
        n = _kv(s["nixos"])
        nx = {"version": n.get("version"), "running": n.get("running"), "booted": n.get("booted"),
              "staged": n.get("staged"), "upgrade_result": n.get("Result"),
              "upgrade_status": n.get("ExecMainStatus"),
              "upgrade_last": n.get("ExecMainExitTimestamp"),
              "autoupgrade_timer": n.get("autoupgrade_timer")}
        sm = _int(n.get("staged_mtime"))
        nx["staged_age_days"] = round((now - sm) / 86400, 1) if sm else None
        nx["staged_not_running"] = bool(nx["staged"] and nx["running"] and nx["staged"] != nx["running"])
        nx["kernel_change_pending"] = bool(n.get("booted_kernel") and n.get("staged_kernel")
                                           and n["booted_kernel"] != n["staged_kernel"])
        if nx["staged_not_running"]:
            reboot_reasons.append("nixos generation staged (operation=boot), not yet booted")
        st["nixos"] = nx

    fw = parse_fwupd(_json(s.get("fwupd_updates", [])), _json(s.get("fwupd_devices", [])))
    fm = _kv(s.get("fwupd_meta", []))
    mm = _int(fm.get("metadata_mtime"))
    fw["available"] = meta.get("has_fwupdmgr") == "1"
    fw["metadata_age_days"] = round((now - mm) / 86400, 1) if mm else None
    fw["refresh_timer"] = fm.get("refresh_timer")
    st["firmware"] = fw

    gpus = []
    for ln in s.get("gpu", []):
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) >= 6 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1], "driver": parts[2],
                         "vbios": parts[3], "util": _int(parts[4].rstrip(" %")),
                         "mem_used": parts[5]})
    samples = [_int(x) for x in s.get("gpu_util_samples", []) if _int(x) is not None]
    st["gpu"] = {"present": meta.get("has_nvidia-smi") == "1", "gpus": gpus,
                 "util_max": max(samples + [g["util"] for g in gpus if g["util"] is not None],
                                 default=None),
                 "procs": [p for p in s.get("gpu_procs", []) if p.strip()],
                 "smi_ok": bool(gpus) or meta.get("has_nvidia-smi") != "1"}
    st["dkms"] = [d for d in s.get("dkms", []) if d.strip()] if "dkms" in s else None

    dk = s.get("docker")
    if dk is None:
        st["containers"] = None if meta.get("has_docker") == "1" else []
        st["docker_error"] = "docker present but section missing" if meta.get("has_docker") == "1" else None
    elif dk and dk[0].startswith("ERROR|"):
        st["containers"], st["docker_error"] = None, dk[0][6:]
    else:
        st["containers"] = [dict(zip(("name", "image", "status"), ln.split("|", 2)))
                            for ln in dk if ln.strip()]
        st["docker_error"] = None
    ol = _json(s.get("ollama", []))
    st["ollama_loaded"] = [m.get("name") for m in (ol or {}).get("models", [])] if isinstance(ol, dict) else []
    st["holds"] = [ln for ln in s.get("holds", []) if ln.strip()]
    st["failed_units"] = [u for u in s.get("failed_units", []) if u.strip()]
    st["running_services"] = [u for u in s.get("running_services", []) if u.strip()]
    nics: dict[str, list[str]] = {}
    for ln in s.get("nics", []):
        parts = ln.split()
        if len(parts) == 2:
            nics.setdefault(parts[0], []).append(parts[1])
    st["nics"] = nics
    links = {}
    for ln in s.get("links", []):
        parts = ln.split()
        if len(parts) == 3:
            links[parts[0]] = {"mtu": parts[1], "state": parts[2]}
    st["links"] = links
    if "swap_parity" in s:
        st["swap_parity"] = _kv(s["swap_parity"])
    st["reboot"] = {"required": bool(reboot_reasons), "reasons": reboot_reasons}
    return st


# ---------------------------------------------------------------------------
# busy check
# ---------------------------------------------------------------------------

DEFAULT_CONTAINER_PATTERNS = ["*vllm*", "*sglang*", "*ollama*", "*trtllm*", "*tensorrt*",
                              "lab-*", "*llama*", "*nim*", "*triton*"]


def fetch_lab(hub: str, timeout: float = 5.0) -> dict[str, Any]:
    """GET a llm-lab hub's /api/services and /api/jobs. Errors are returned, not raised."""
    out: dict[str, Any] = {"hub": hub, "ok": True, "error": None, "services": None, "jobs": None}
    for path, key in (("/api/services", "services"), ("/api/jobs", "jobs")):
        try:
            with urllib.request.urlopen(hub.rstrip("/") + path, timeout=timeout) as r:
                out[key] = json.load(r)
        except urllib.error.HTTPError as e:
            if key == "jobs" and e.code == 404:
                out[key] = {"jobs": []}   # "this lab has no cluster jobs"
            else:
                out["ok"], out["error"] = False, f"{path}: HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            out["ok"], out["error"] = False, f"{path}: {e}"
    return out


def lab_busy(host: str, lab: dict[str, Any] | None) -> tuple[bool | None, list[str]]:
    """(busy, reasons); busy None means the hub could not be read (caller fails closed)."""
    if not lab or not lab.get("ok"):
        return None, [f"llm-lab hub unreadable: {(lab or {}).get('error')}"]
    reasons = []
    svc = lab.get("services") or {}
    for g in (svc.get("gpus") or {}).values():
        if g.get("node") != host:
            continue
        if g.get("lock"):
            reasons.append(f"llm-lab GPU lock: {str(g['lock'])[:80]}")
        if g.get("processes"):
            reasons.append(f"llm-lab GPU processes: {len(g['processes'])}")
    for s_ in svc.get("services") or []:
        if isinstance(s_, dict) and host in (s_.get("node"), s_.get("host")):
            reasons.append(f"llm-lab service {s_.get('name') or s_.get('id')}")
    for j in (lab.get("jobs") or {}).get("jobs") or []:
        if host in (j.get("nodes") or []) and str(j.get("state")) not in TERMINAL_JOB_STATES:
            reasons.append(f"llm-lab job {j.get('id')} state {j.get('state')}")
    return bool(reasons), reasons


def parse_hold(line: str, now: float) -> tuple[str, bool]:
    """(description, stale). A hold whose `expires=<epoch>` is past is STALE but still
    blocks: a forgotten hold costs a skipped week, a wrongly ignored one costs a benchmark."""
    m = re.search(r"expires=(\d+)", line)
    stale = bool(m and float(m.group(1)) < now)
    return line[:200], stale


def central_holds(host: str, cls: str, now: float | None = None) -> list[str]:
    now = time.time() if now is None else now
    d = central_holds_dir()
    out = []
    for name in (host, f"class-{cls}", "ALL"):
        f = d / name
        if f.exists():
            try:
                body = f.read_text()[:200].replace("\n", " ")
            except OSError:
                body = "(unreadable)"
            out.append(f"{f}|{body}")
    return out


def evaluate_busy(st: dict[str, Any], rules: dict[str, Any], lab: dict[str, Any] | None,
                  extra_holds: list[str] | None = None,
                  now: float | None = None) -> dict[str, Any]:
    """Pure. busy True/False; unknowns that a rule depends on count as busy."""
    now = time.time() if now is None else now
    reasons: list[str] = []
    if not st.get("reachable"):
        return {"busy": True, "reasons": ["unreachable"], "stale_holds": []}
    stale_holds = []
    for h in (st.get("holds") or []) + (extra_holds or []):
        desc, stale = parse_hold(h, now)
        reasons.append(f"hold file {desc}")
        if stale:
            stale_holds.append(desc)
    if rules.get("any_container") or rules.get("container_patterns"):
        if st.get("containers") is None and st.get("docker_error") is not None:
            reasons.append(f"docker unreadable ({st.get('docker_error')})")
        pats = rules.get("container_patterns") or DEFAULT_CONTAINER_PATTERNS
        ignore = [p.lower() for p in rules.get("container_ignore") or []]
        for c in st.get("containers") or []:
            if any(fnmatch.fnmatch(c.get("name", "").lower(), p) for p in ignore):
                continue   # infrastructure that never blocks maintenance (e.g. cloudflared)
            if rules.get("any_container") or any(
                    fnmatch.fnmatch(c.get("name", "").lower(), p) or
                    fnmatch.fnmatch(c.get("image", "").lower(), p) for p in pats):
                reasons.append(f"container running: {c.get('name')} ({c.get('image')})")
    lim = rules.get("gpu_util_pct")
    gpu = st.get("gpu") or {}
    if lim is not None and gpu.get("present"):
        if not gpu.get("smi_ok"):
            reasons.append("nvidia-smi failed")
        elif gpu.get("util_max") is not None and gpu["util_max"] > lim:
            reasons.append(f"GPU utilization {gpu['util_max']}% > {lim}%")
    if rules.get("gpu_processes") and gpu.get("procs"):
        reasons.append(f"GPU compute processes: {len(gpu['procs'])}")
    if rules.get("ollama_loaded") and st.get("ollama_loaded"):
        reasons.append(f"ollama models loaded: {', '.join(st['ollama_loaded'])}")
    if rules.get("llm_lab"):
        b, r = lab_busy(st["host"], lab)
        if b is None or b:
            reasons.extend(r)
    return {"busy": bool(reasons), "reasons": reasons, "stale_holds": stale_holds}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def selected_packages(st: dict[str, Any], updates: str) -> list[dict[str, Any]]:
    pk = (st.get("pkg") or {}).get("packages") or []
    if updates == "full":
        return list(pk)
    if updates == "security":
        return [p for p in pk if p.get("security")]
    return []


def forbidden_hits(names: list[str], patterns: list[str]) -> list[str]:
    return sorted({n for n in names for rx in patterns if re.search(rx, n)})


def host_eligibility(host: str, st: dict[str, Any], cls: dict[str, Any],
                     hcfg: dict[str, Any], busy: dict[str, Any],
                     sudo_ok: bool) -> dict[str, Any]:
    """Pure. -> {state: ready|up-to-date|blocked|manual, reasons, packages, firmware}."""
    r: dict[str, Any] = {"host": host, "reasons": [], "packages": [], "firmware": []}
    if not st.get("reachable"):
        return dict(r, state="blocked", reasons=["unreachable"])
    if (st.get("os") or {}).get("id") not in cls["apply_supported_os"] or \
            (st.get("pkg") or {}).get("manager") != "apt":
        return dict(r, state="manual", reasons=[
            f"no deterministic applier for {(st.get('os') or {}).get('id')}/"
            f"{(st.get('pkg') or {}).get('manager')}"])
    if busy.get("busy"):
        return dict(r, state="blocked", reasons=busy["reasons"])
    pk = st.get("pkg") or {}
    if pk.get("errors"):
        return dict(r, state="blocked", reasons=["apt broken: " + "; ".join(pk["errors"])[:200]])
    if cls.get("require_reboot_tested") and not hcfg.get("reboot_tested"):
        return dict(r, state="blocked", reasons=["reboot not yet tested by hand (policy)"])
    sel = selected_packages(st, cls["updates"])
    hits = forbidden_hits([p["name"] for p in sel], cls["forbid_packages"])
    if hits:
        return dict(r, state="manual", reasons=[f"forbidden packages pending: {', '.join(hits)}"])
    fw = (st.get("firmware") or {}).get("pending") or [] if cls.get("firmware") else []
    needs_reboot = bool(fw) or (st.get("reboot") or {}).get("required") or any(
        p["name"].startswith(("linux-image", "linux-modules", "linux-firmware")) for p in sel)
    if needs_reboot and cls["reboot"] == "never" and fw:
        return dict(r, state="manual", reasons=["firmware pending but class never reboots"])
    r["packages"] = [p["name"] for p in sel]
    r["firmware"] = fw
    if not sel and not fw:
        return dict(r, state="up-to-date")
    if not sudo_ok:
        return dict(r, state="blocked", reasons=["no sudo credential file configured/readable"])
    return dict(r, state="ready")


def plan_class(policy: dict[str, Any], cls_name: str, states: dict[str, dict[str, Any]],
               busy: dict[str, dict[str, Any]], sudo_ok: dict[str, bool],
               approval: tuple[bool, str, list[str]] | None, window_open: bool,
               stop_reason: str | None = None) -> dict[str, Any]:
    """Pure. The whole decision for one class, before anything is touched."""
    cls = policy["classes"][cls_name]
    hosts = class_hosts(policy, cls_name)
    plan: dict[str, Any] = {"class": cls_name, "hosts": {}, "order": [], "canary": None}
    if stop_reason:
        return dict(plan, verdict="killswitch", reason=stop_reason)
    if not cls["auto_apply"]:
        return dict(plan, verdict="report-only",
                    reason=cls.get("manual_only_reason") or "auto_apply is false for this class")
    if not window_open:
        return dict(plan, verdict="outside-window", reason="class window is closed")
    if approval is None or not approval[0]:
        return dict(plan, verdict="refused", reason=(approval or (False, "no approval checked"))[1])
    approved = set(approval[2])
    for h in hosts:
        e = host_eligibility(h, states.get(h) or {"reachable": False}, cls,
                             policy["hosts"][h], busy.get(h) or {"busy": True, "reasons": ["no busy data"]},
                             sudo_ok.get(h, False))
        if h not in approved:
            e = dict(e, state="blocked", reasons=["not in the approval's hosts allowlist"])
        plan["hosts"][h] = e
    blocked = {h: e for h, e in plan["hosts"].items() if e["state"] in ("blocked", "manual")}
    work = [h for h in hosts if plan["hosts"][h]["state"] == "ready"]
    if cls["as_set"] and blocked:
        return dict(plan, verdict="deferred",
                    reason="set class: " + "; ".join(f"{h}: {e['reasons'][0]}"
                                                     for h, e in sorted(blocked.items())))
    if not work:
        return dict(plan, verdict="nothing-to-do", reason="every eligible host is up to date")
    canary = cls.get("canary")
    if canary not in work:
        # The canary exists to try the change first. If it has nothing to do (already
        # current) the first host WITH work becomes the canary; if it is blocked in a
        # non-set class, the class waits - the canary is never skipped silently.
        if canary and plan["hosts"][canary]["state"] in ("blocked", "manual"):
            return dict(plan, verdict="deferred", reason=f"canary {canary} blocked: "
                        + plan["hosts"][canary]["reasons"][0])
        canary = work[0]
    plan["canary"] = canary
    plan["order"] = [canary] + [h for h in work if h != canary]
    return dict(plan, verdict="run", reason=f"{len(work)} host(s) to update, canary {canary}")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def _ignored(name: str, pats: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in pats)


def verify_host(before: dict[str, Any], after: dict[str, Any], cls: dict[str, Any]) -> list[str]:
    """Pure. Problems found comparing the pre- and post-update states of one host."""
    v = {"nics": True, "gpu": True, "failed_units": True, "services": True, "holds": True,
         "dkms": False, "firmware_cleared": True, **(cls.get("verify") or {})}
    probs: list[str] = []
    if not after.get("reachable"):
        return ["host not reachable after update"]
    if v["nics"]:
        ign = DEFAULT_NICS_IGNORE + list(v.get("nics_ignore") or [])
        for nic, addrs in (before.get("nics") or {}).items():
            if _ignored(nic, ign):
                continue
            if sorted(addrs) != sorted((after.get("nics") or {}).get(nic, [])):
                probs.append(f"NIC {nic} address changed {addrs} -> {(after.get('nics') or {}).get(nic)}")
        for nic, lk in (before.get("links") or {}).items():
            if _ignored(nic, ign) or lk.get("state") != "UP":
                continue
            la = (after.get("links") or {}).get(nic) or {}
            if la.get("state") != "UP" or la.get("mtu") != lk.get("mtu"):
                probs.append(f"link {nic} was UP mtu {lk.get('mtu')}, now {la.get('state')} mtu {la.get('mtu')}")
    if v["gpu"] and (before.get("gpu") or {}).get("gpus"):
        nb, na = len(before["gpu"]["gpus"]), len((after.get("gpu") or {}).get("gpus") or [])
        if na < nb:
            probs.append(f"GPUs visible {nb} -> {na}")
    if v["failed_units"]:
        new = sorted(set(after.get("failed_units") or []) - set(before.get("failed_units") or []))
        if new:
            probs.append(f"new failed units: {', '.join(new)}")
    if v["services"]:
        ign = DEFAULT_SERVICES_IGNORE + list(v.get("services_ignore") or [])
        gone = sorted(s for s in set(before.get("running_services") or [])
                      - set(after.get("running_services") or []) if not _ignored(s, ign))
        if gone:
            probs.append(f"services that were running are not: {', '.join(gone)}")
    if v["holds"]:
        hb, ha = sorted((before.get("pkg") or {}).get("holds") or []), \
            sorted((after.get("pkg") or {}).get("holds") or [])
        if hb != ha:
            probs.append(f"apt holds changed {hb} -> {ha}")
    if v["dkms"] and (before.get("dkms") or []) != (after.get("dkms") or []):
        probs.append(f"dkms status changed {before.get('dkms')} -> {after.get('dkms')}")
    if v["firmware_cleared"] and cls.get("firmware"):
        left = (after.get("firmware") or {}).get("pending") or []
        if left:
            probs.append("firmware still pending after reboot (an embedded controller may need "
                         "a full AC power cycle; that is a human decision): "
                         + ", ".join(f"{f['device']} {f['current']}->{f['new']}" for f in left))
    if (after.get("pkg") or {}).get("errors"):
        probs.append("apt reports errors after update")
    return probs


# Facts that must match across a whole set class (pod-wide). Everything else - VBIOS and
# every fwupd device - is only comparable between hosts of the SAME hardware model: an MSI
# EdgeXpert reports its embedded controller as 10800 while a DGX Spark reports 0x03000508,
# and neither is "drift".
POD_WIDE = ("kernel", "driver", "ota", "swapimg", "earlyoom")


def parity_table(states: dict[str, dict[str, Any]], hosts: list[str]) -> dict[str, Any]:
    """Rows of version facts per host plus the columns that differ across the set."""
    rows = {}
    for h in hosts:
        st = states.get(h) or {}
        g = ((st.get("gpu") or {}).get("gpus") or [{}])[0]
        row = {"model": st.get("model"), "kernel": (st.get("kernel") or {}).get("running"),
               "driver": g.get("driver"), "vbios": g.get("vbios"),
               "ota": (st.get("dgx") or {}).get("ota"),
               "swapimg": (st.get("swap_parity") or {}).get("swapimg_bytes"),
               "earlyoom": (st.get("swap_parity") or {}).get("earlyoom"),
               "apt_pending": (st.get("pkg") or {}).get("pending"),
               "fw_pending": len((st.get("firmware") or {}).get("pending") or [])}
        seen: dict[str, int] = {}
        for d in (st.get("firmware") or {}).get("devices") or []:
            name = str(d["device"]).strip()
            seen[name] = seen.get(name, 0) + 1
            key = "fw:" + name + (f" #{seen[name]}" if seen[name] > 1 else "")
            row[key] = str(d["version"]).strip()
        rows[h] = row
    cols = sorted({k for r in rows.values() for k in r} - {"model"})
    drift = []
    for c in cols:
        if c in ("apt_pending", "fw_pending"):
            continue
        groups: dict[Any, set] = {}
        for r in rows.values():
            if r.get(c) in (None, ""):
                continue
            key = "all" if c in POD_WIDE else r.get("model")
            groups.setdefault(key, set()).add(r[c])
        bad = [str(k) for k, v in groups.items() if len(v) > 1]
        if bad:
            drift.append(c if c in POD_WIDE else f"{c} [{', '.join(sorted(bad))}]")
    return {"rows": rows, "columns": ["model"] + cols, "drift": drift}


# ---------------------------------------------------------------------------
# ssh runner (the only place a process is spawned; tests replace it)
# ---------------------------------------------------------------------------

class SSHRunner:
    def __init__(self, policy: dict[str, Any]):
        d = policy.get("defaults") or {}
        self.opts = list(d.get("ssh_options") or [
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"])
        self.policy = policy

    def _argv(self, host: str, remote: str) -> list[str]:
        extra = list((self.policy["hosts"].get(host) or {}).get("ssh") or [])
        return ["ssh", *self.opts, *extra, host, remote]

    def run(self, host: str, remote: str, stdin: str | None = None,
            timeout: float = 120) -> tuple[int, str]:
        try:
            p = subprocess.run(self._argv(host, remote), input=stdin, capture_output=True,
                               text=True, timeout=timeout)
            return p.returncode, p.stdout + (("\n" + p.stderr) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout}s"
        except OSError as e:
            return 127, str(e)

    def collect(self, host: str, timeout: float = 240) -> dict[str, Any]:
        rc, out = self.run(host, "bash -s", stdin=COLLECT_SH.read_text(), timeout=timeout)
        return parse_collect(host, out, rc)

    def sudo(self, host: str, cmd: str, timeout: float = 3600) -> tuple[int, str]:
        """Run cmd as root. The password goes over stdin to `sudo -S`, never into argv,
        a log or the environment."""
        f = (self.policy["hosts"].get(host) or {}).get("sudo_file")
        pw = None
        if f:
            try:
                pw = Path(os.path.expanduser(f)).read_text().rstrip("\n") + "\n"
            except OSError as e:
                return 126, f"sudo file unreadable: {e.__class__.__name__}"
        remote = "sudo -S -p '' bash -c " + shlex.quote(cmd) if pw else \
            "sudo -n bash -c " + shlex.quote(cmd)
        rc, out = self.run(host, remote, stdin=pw, timeout=timeout)
        if pw:
            out = out.replace(pw.strip(), "***")
        return rc, out


def sudo_file_ok(policy: dict[str, Any], host: str) -> bool:
    f = (policy["hosts"].get(host) or {}).get("sudo_file")
    return bool(f) and os.access(os.path.expanduser(f), os.R_OK)


# ---------------------------------------------------------------------------
# apply one host
# ---------------------------------------------------------------------------

APT_ENV = ("DEBIAN_FRONTEND=noninteractive apt-get -y -o DPkg::Lock::Timeout=600 "
           "-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold")


APT_SIM = "apt-get -s -o Debug::NoLocking=1"


def apt_command(updates: str, packages: list[str], simulate: bool = False) -> str:
    """A non-interactive apt full-upgrade keeping existing config files (full), or an
    explicit --only-upgrade list of the security packages (security). simulate=True
    gives the same transaction as an unprivileged `apt-get -s` dry run."""
    base = APT_SIM if simulate else APT_ENV
    if updates == "full":
        return f"{base} full-upgrade"
    bad = [p for p in packages if not PKG_RE.match(p)]
    if bad or not packages:
        raise ValueError(f"refusing package list {bad or packages}")
    return f"{base} install --only-upgrade " + " ".join(packages)


def simulate_installs(sim_out: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"^Inst (\S+)", sim_out, re.M)]


def apply_host(host: str, cls: dict[str, Any], runner: Any, policy: dict[str, Any],
               lab_fetch: Callable[[str], dict[str, Any]],
               sleep: Callable[[float], None] = time.sleep,
               now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Update one host. Returns a record with status ok|failed|needs-hands|skipped."""
    rec: dict[str, Any] = {"host": host, "started": now(), "steps": [], "status": "failed"}

    def step(name: str, rc: int, out: str) -> None:
        rec["steps"].append({"step": name, "rc": rc, "out": out[-4000:]})

    before = runner.collect(host)
    rec["before"] = before
    hcfg = policy["hosts"][host]
    lab = lab_fetch(hcfg["lab_hub"]) if hcfg.get("lab_hub") and cls["busy"].get("llm_lab") else None
    busy = evaluate_busy(before, cls["busy"], lab, central_holds(host, hcfg["class"]))
    elig = host_eligibility(host, before, cls, hcfg, busy, sudo_file_ok(policy, host))
    rec["eligibility"] = elig
    if elig["state"] != "ready":
        # Re-checked right before touching the host: it may have become busy since the
        # plan. "blocked" halts the rest of the class like a failure does.
        rec["status"] = "skipped" if elig["state"] == "up-to-date" else "blocked"
        rec["reason"] = "; ".join(elig["reasons"]) or elig["state"]
        return rec

    rc, out = runner.sudo(host, f"apt-get -o DPkg::Lock::Timeout=600 update", timeout=900)
    step("apt-get update", rc, out)
    if rc != 0:
        rec["reason"] = "apt-get update failed"
        return rec
    # Re-read the pending list after the refresh; the collector's view may be a day old.
    rc, out = runner.run(host, "apt list --upgradable 2>/dev/null", timeout=120)
    sel = selected_packages({"pkg": {"packages": parse_apt_upgradable(out.splitlines())}},
                            cls["updates"])
    names = [p["name"] for p in sel]
    cmd = apt_command(cls["updates"], names) if names else None
    if cmd:
        # Simulate the SAME transaction first: `apt list --upgradable` never shows NEW
        # packages a full-upgrade pulls in (a new kernel image, a new nvidia module),
        # which are exactly the ones forbid_packages exists for.
        rc, sim = runner.run(host, apt_command(cls["updates"], names, simulate=True), timeout=300)
        step("apt simulate", rc, sim)
        installs = simulate_installs(sim)
        hits = forbidden_hits(installs, cls["forbid_packages"])
        if rc != 0 or hits:
            rec["status"] = "failed"
            rec["reason"] = (f"forbidden packages in the transaction: {hits}" if hits
                             else "apt simulation failed")
            return rec
        rc, out = runner.sudo(host, cmd, timeout=3600)
        step("apt upgrade", rc, out)
        if DKMS_FAIL_RE.search(out):
            rec["status"] = "needs-hands"
            rec["reason"] = "dkms build failure reported by apt; not rebooting"
            return rec
        if rc != 0:
            rec["reason"] = "apt upgrade failed"
            return rec
    fw_applied = False
    if cls.get("firmware") and before.get("firmware", {}).get("available"):
        rc, out = runner.sudo(host, "fwupdmgr refresh --force", timeout=300)
        step("fwupdmgr refresh", rc, out)
        rc, out = runner.sudo(host, "fwupdmgr update -y --no-reboot-check", timeout=1800)
        step("fwupdmgr update", rc, out)
        if rc not in (0, 2):   # 2 = nothing to do
            rec["reason"] = "fwupdmgr update failed"
            return rec
        fw_applied = rc == 0
    mid = runner.collect(host)
    need_reboot = fw_applied or (mid.get("reboot") or {}).get("required")
    rec["reboot_needed"] = bool(need_reboot)
    if need_reboot and cls["reboot"] == "if-required":
        old_boot = mid.get("boot_id") or before.get("boot_id")
        t0 = now()
        rc, out = runner.sudo(host, "systemctl reboot", timeout=60)
        step("reboot", rc, out)
        sleep(60)
        back = False
        while now() - t0 < cls["reboot_timeout_s"]:
            rc, out = runner.run(host, "cat /proc/sys/kernel/random/boot_id", timeout=20)
            if rc == 0 and out.strip() and out.strip().splitlines()[0] != old_boot:
                back = True
                break
            sleep(30)
        rec["reboot_s"] = round(now() - t0)
        if not back:
            # No BMC on a GB10: a machine that does not come back needs hands. Never
            # power-cycle from here; the PDU is a separate, human-approved action.
            rec["status"] = "needs-hands"
            rec["reason"] = f"did not come back within {cls['reboot_timeout_s']}s of reboot"
            return rec
        sleep(90)   # let units settle before judging failed/running services
    elif need_reboot:
        rec["reboot_deferred"] = True
    after = runner.collect(host)
    rec["after"] = after
    probs = verify_host(before, after, cls)
    rec["problems"] = probs
    rec["status"] = "ok" if not probs else "failed"
    rec["reason"] = "; ".join(probs) if probs else "verified"
    rec["finished"] = now()
    return rec


# ---------------------------------------------------------------------------
# run a class
# ---------------------------------------------------------------------------

def gather(policy: dict[str, Any], hosts: list[str], runner: Any,
           lab_fetch: Callable[[str], dict[str, Any]], parallel: int = 8
           ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Collect hosts in parallel; evaluate busy against their class rules."""
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        states = dict(zip(hosts, pool.map(runner.collect, hosts)))
    hubs = sorted({policy["hosts"][h].get("lab_hub") for h in hosts
                   if policy["hosts"][h].get("lab_hub")})
    labs = {hb: lab_fetch(hb) for hb in hubs}
    busy = {}
    for h in hosts:
        hc = policy["hosts"][h]
        cls = policy["classes"][hc["class"]]
        lab = labs.get(hc.get("lab_hub")) if hc.get("lab_hub") else None
        rules = dict(cls["busy"])
        if rules.get("llm_lab") and not hc.get("lab_hub"):
            rules["llm_lab"] = False
        busy[h] = evaluate_busy(states[h], rules, lab, central_holds(h, hc["class"]))
        states[h]["class"] = hc["class"]
        states[h]["busy"] = busy[h]
    return states, busy


def run_class(policy: dict[str, Any], cls_name: str, runner: Any,
              lab_fetch: Callable[[str], dict[str, Any]] = fetch_lab,
              dry_run: bool = False, now: float | None = None,
              override_window: bool = False,
              sleep: Callable[[float], None] = time.sleep,
              apply_fn: Callable[..., dict[str, Any]] = apply_host,
              post_hook: Callable[[str, list[str]], dict[str, Any]] | None = None
              ) -> dict[str, Any]:
    now_f = time.time() if now is None else now
    cls = policy["classes"][cls_name]
    hosts = class_hosts(policy, cls_name)
    result: dict[str, Any] = {"class": cls_name, "run_id": _dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
                              "dry_run": dry_run, "started": now_f, "hosts": {}, "skipped": []}
    stop = stopped()
    approval = check_approval(str(cls.get("approval_tag") or ""), hosts, cls["as_set"], now_f) \
        if cls["auto_apply"] else None
    window_open = in_window(cls["window"], now_f) or override_window
    if stop or not cls["auto_apply"] or not window_open or not (approval and approval[0]):
        plan = plan_class(policy, cls_name, {}, {}, {}, approval, window_open, stop)
        if dry_run and not stop:
            # Show the operator the whole picture even though the gate is shut.
            states, busy = gather(policy, hosts, runner, lab_fetch)
            probe = plan_class(policy, cls_name, states, busy,
                               {h: sudo_file_ok(policy, h) for h in hosts},
                               (True, "dry-run probe", hosts), True, None) \
                if cls["auto_apply"] else None
            result["probe"] = probe
            result["states"] = {h: _brief(states[h]) for h in hosts}
        result.update(plan=plan, status=plan["verdict"], reason=plan["reason"],
                      approval=approval[1] if approval else None)
        return result
    states, busy = gather(policy, hosts, runner, lab_fetch)
    plan = plan_class(policy, cls_name, states, busy,
                      {h: sudo_file_ok(policy, h) for h in hosts}, approval, window_open, None)
    result.update(plan=plan, approval=approval[1], states={h: _brief(states[h]) for h in hosts})
    if plan["verdict"] != "run" or dry_run:
        result.update(status=plan["verdict"] if not dry_run else "dry-run", reason=plan["reason"])
        return result

    halt = threading.Event()
    order = plan["order"]
    canary = order[0]
    r = apply_fn(canary, cls, runner, policy, lab_fetch, sleep=sleep)
    result["hosts"][canary] = r
    if r["status"] != "ok":
        result.update(status="canary-failed", skipped=order[1:],
                      reason=f"canary {canary} {r['status']}: {r.get('reason')}; rest not started")
        return result

    def one(h: str) -> dict[str, Any]:
        if halt.is_set():
            return {"host": h, "status": "skipped", "reason": "an earlier host failed"}
        if not override_window and not in_window(cls["window"]):
            return {"host": h, "status": "skipped", "reason": "window closed"}
        if stopped():
            return {"host": h, "status": "skipped", "reason": stopped()}
        try:
            rr = apply_fn(h, cls, runner, policy, lab_fetch, sleep=sleep)
        except Exception as e:  # noqa: BLE001 - one host never takes the run down
            rr = {"host": h, "status": "failed", "reason": repr(e)}
        if rr["status"] not in ("ok", "skipped"):
            halt.set()
        return rr

    with ThreadPoolExecutor(max_workers=int(cls["max_parallel"])) as pool:
        for h, rr in zip(order[1:], pool.map(one, order[1:])):
            result["hosts"][h] = rr
    bad = [h for h, rr in result["hosts"].items() if rr["status"] not in ("ok", "skipped")]
    if cls["as_set"] or (cls.get("verify") or {}).get("parity"):
        post, _ = gather(policy, hosts, runner, lab_fetch)
        result["parity"] = parity_table(post, hosts)
        result["post_states"] = {h: _brief(post[h]) for h in hosts}
        if post_hook:
            result["post_hook"] = post_hook(cls_name, hosts)
    result["status"] = "failed" if bad else (
        "parity-drift" if result.get("parity", {}).get("drift") else "completed")
    result["reason"] = ("hosts not ok: " + ", ".join(sorted(bad))) if bad else \
        (f"drift in {result['parity']['drift']}" if result["status"] == "parity-drift" else "all verified")
    return result


def _brief(st: dict[str, Any]) -> dict[str, Any]:
    pk = st.get("pkg") or {}
    return {"reachable": st.get("reachable"), "pending": pk.get("pending"),
            "security": pk.get("security"), "fw_pending": len((st.get("firmware") or {}).get("pending") or []),
            "kernel": (st.get("kernel") or {}).get("running"),
            "reboot": (st.get("reboot") or {}).get("required"),
            "busy": (st.get("busy") or {}).get("reasons")}


# ---------------------------------------------------------------------------
# state files, alerts, report
# ---------------------------------------------------------------------------

def write_states(states: dict[str, dict[str, Any]]) -> None:
    d = state_dir() / "hosts"
    d.mkdir(parents=True, exist_ok=True)
    for h, st in states.items():
        tmp = d / f".{h}.json.tmp"
        tmp.write_text(json.dumps(st, indent=1, default=str))
        tmp.replace(d / f"{h}.json")


def read_states() -> dict[str, dict[str, Any]]:
    d = state_dir() / "hosts"
    out = {}
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            out[f.stem] = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
    return out


def alerts(policy: dict[str, Any], states: dict[str, dict[str, Any]]) -> list[str]:
    """Conditions worth the operator's attention. Deterministic strings so they dedupe."""
    out = []
    for h, st in sorted(states.items()):
        hc = policy["hosts"].get(h) or {}
        if hc.get("class") == "unmanaged":
            continue
        if not st.get("reachable"):
            out.append(f"{h}: unreachable for collection")
            continue
        pk = st.get("pkg") or {}
        if pk.get("errors"):
            out.append(f"{h}: apt is broken ({pk['errors'][0][:120]})")
        uu = pk.get("uu") or {}
        cls = policy["classes"].get(hc.get("class"), {})
        if cls.get("uu_blacklist_required") and uu.get("installed") and uu.get("enabled") \
                and not uu.get("blacklist"):
            # The class says kernel/driver changes are manual-only, but the host's own
            # unattended-upgrades can still pull them from -security with nothing
            # blacklisted. fleet-maint never changes that config itself. Only security-
            # pocket packages count: that is all Ubuntu's default UU origins take.
            hits = forbidden_hits([p["name"] for p in pk.get("packages") or [] if p.get("security")],
                                  cls["forbid_packages"])
            out.append(f"{h}: unattended-upgrades ON with an empty Package-Blacklist on a "
                       f"manual-only class" + (f"; security-pocket hits now: {', '.join(hits[:4])}"
                                               if hits else ""))
        nx = st.get("nixos") or {}
        if nx and nx.get("upgrade_result") not in (None, "", "success"):
            out.append(f"{h}: nixos-upgrade last result {nx['upgrade_result']}")
        if nx.get("staged_not_running") and (nx.get("staged_age_days") or 0) > 7:
            out.append(f"{h}: NixOS generation staged {nx['staged_age_days']}d ago, not booted")
        if (st.get("reboot") or {}).get("required") and (st.get("uptime_days") or 0) > 14:
            out.append(f"{h}: reboot pending, up {st['uptime_days']}d")
        fw = st.get("firmware") or {}
        if fw.get("available") and (fw.get("metadata_age_days") or 0) > 30:
            out.append(f"{h}: fwupd metadata {fw['metadata_age_days']}d old")
        for sh in (st.get("busy") or {}).get("stale_holds") or []:
            out.append(f"{h}: stale maintenance hold {sh[:80]}")
    for cname, c in policy["classes"].items():
        if not c.get("as_set"):
            continue
        members = [h for h in class_hosts(policy, cname) if (states.get(h) or {}).get("reachable")]
        if len(members) > 1:
            pt = parity_table(states, members)
            if pt["drift"]:
                out.append(f"{cname}: parity drift in {', '.join(pt['drift'][:6])}")
    return out


def state_table(policy: dict[str, Any], states: dict[str, dict[str, Any]]) -> str:
    hdr = ("| host | class | OS | kernel | apt pending | security | held | firmware pending "
           "| reboot | up (d) | busy |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for h in policy["hosts"]:
        st = states.get(h)
        cls = policy["hosts"][h]["class"]
        if st is None:
            rows.append(f"| {h} | {cls} | not collected | | | | | | | | |")
            continue
        if not st.get("reachable"):
            rows.append(f"| {h} | {cls} | UNREACHABLE: {(st.get('errors') or [''])[0][:60]} | | | | | | | | |")
            continue
        pk, fw = st.get("pkg") or {}, st.get("firmware") or {}
        osv = f"{(st.get('os') or {}).get('id')} {(st.get('os') or {}).get('version')}"
        if pk.get("manager") == "nixos":
            nx = st.get("nixos") or {}
            pend = "gen staged" if nx.get("staged_not_running") else "current"
            sec = f"upgrade {nx.get('upgrade_result')}"
        elif pk.get("errors"):
            pend, sec = "APT BROKEN", "?"
        else:
            pend = "n/a" if pk.get("pending") is None else str(pk["pending"])
            sec = "n/a" if pk.get("security") is None else str(pk["security"])
        fwp = ", ".join(f"{f['device']} {f['current']}->{f['new']}" for f in fw.get("pending") or []) \
            or ("0" if fw.get("available") else "no fwupd")
        busy = "; ".join((st.get("busy") or {}).get("reasons") or []) or "idle"
        rows.append(f"| {h} | {cls} | {osv} | {(st.get('kernel') or {}).get('running')} | {pend} | "
                    f"{sec} | {' '.join(pk.get('holds') or []) or '-'} | {fwp[:90]} | "
                    f"{'yes' if (st.get('reboot') or {}).get('required') else 'no'} | "
                    f"{st.get('uptime_days')} | {busy[:70]} |")
    return hdr + "\n".join(rows) + "\n"


def notify(subject: str, body: str, severity: str = "info") -> str:
    """Hand a message to the operator's notification hook.

    If FLEET_MAINT_NOTIFY is set it is run as a command with the message on stdin and
    FLEET_MAINT_SUBJECT / FLEET_MAINT_SEVERITY in its environment (wire it to mail, a chat
    webhook or a ticket queue). Without it the message is only printed by the caller.
    """
    cmd = os.environ.get("FLEET_MAINT_NOTIFY")
    if not cmd:
        return "notify: FLEET_MAINT_NOTIFY not set (message printed only)"
    env = dict(os.environ, FLEET_MAINT_SUBJECT=subject[:200], FLEET_MAINT_SEVERITY=severity)
    try:
        r = subprocess.run(shlex.split(cmd), input=body[:19000], capture_output=True, text=True,
                           timeout=120, env=env)
        return f"notify rc={r.returncode} {r.stderr.strip()[:200]}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"notify failed: {e}"


class RunLock:
    """One writer: only one applier touches the fleet at a time."""

    def __init__(self, name: str = "apply"):
        state_dir().mkdir(parents=True, exist_ok=True)
        self.path = state_dir() / f"{name}.lock"
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            raise RuntimeError(f"another fleet-maint apply holds {self.path}") from None
        return self

    def __exit__(self, *a):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def alerts_changed(al: list[str]) -> bool:
    f = state_dir() / "alerts-last.json"
    h = hashlib.sha256("\n".join(sorted(al)).encode()).hexdigest()
    try:
        prev = json.loads(f.read_text()).get("hash")
    except (OSError, ValueError):
        prev = None
    state_dir().mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"hash": h, "alerts": al, "at": time.time()}))
    return h != prev
