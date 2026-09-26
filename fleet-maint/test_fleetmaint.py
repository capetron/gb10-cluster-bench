#!/usr/bin/env python3
"""Tests for fleetmaint.py. Stdlib unittest + PyYAML. Run:
    python3 fleet-maint/test_fleetmaint.py
Nothing here touches the network or a real host: the runner, the lab hub and the clock
are all injected.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fleetmaint as fm  # noqa: E402

ET = ZoneInfo("America/New_York")
SUN_0300 = dt.datetime(2026, 9, 27, 3, 0, tzinfo=ET).timestamp()   # a Sunday
SUN_0700 = dt.datetime(2026, 9, 27, 7, 0, tzinfo=ET).timestamp()
MON_0300 = dt.datetime(2026, 9, 28, 3, 0, tzinfo=ET).timestamp()

BUSY_ALL = {"any_container": True, "gpu_util_pct": 5, "gpu_processes": True,
            "llm_lab": True, "ollama_loaded": True}


def base_policy() -> dict:
    return {
        "classes": {
            "pod": {"auto_apply": True, "approval_tag": "# fleet-maint-pod", "updates": "full",
                    "firmware": True, "reboot": "if-required", "max_parallel": 1,
                    "canary": "a1", "as_set": True, "busy": dict(BUSY_ALL),
                    "forbid_packages": ["^nvidia-dkms-"],
                    "window": {"days": ["Sun"], "start": "02:00", "end": "06:00",
                               "tz": "America/New_York"}},
            "loose": {"auto_apply": True, "approval_tag": "# fleet-maint-loose", "updates": "security",
                      "reboot": "never", "max_parallel": 1, "canary": "b1", "as_set": False,
                      "busy": {"any_container": True},
                      "window": {"days": ["Sun"], "start": "02:00", "end": "06:00"}},
            "prod": {"auto_apply": False, "updates": "security", "busy": {}},
        },
        "hosts": {
            "a1": {"class": "pod", "sudo_file": "~/x", "lab_hub": "http://hub"},
            "a2": {"class": "pod", "sudo_file": "~/x", "lab_hub": "http://hub"},
            "a3": {"class": "pod", "sudo_file": "~/x", "lab_hub": "http://hub"},
            "b1": {"class": "loose", "sudo_file": "~/x"},
            "b2": {"class": "loose", "sudo_file": "~/x"},
            "p1": {"class": "prod"},
        },
    }


def policy() -> dict:
    return fm.validate_policy(base_policy())


def idle_state(host: str, pkgs=None, fw=None, **kw) -> dict:
    st = {
        "host": host, "reachable": True, "boot_id": "boot-1",
        "os": {"id": "ubuntu", "kind": "Linux"},
        "kernel": {"running": "7.0.0-1019-nvidia", "latest_installed": "7.0.0-1019-nvidia"},
        "pkg": {"manager": "apt", "packages": pkgs if pkgs is not None else [
            {"name": "curl", "security": True}, {"name": "gnome-shell", "security": False}],
            "holds": [], "errors": []},
        "firmware": {"available": True, "pending": fw or [], "devices": []},
        "gpu": {"present": True, "gpus": [{"driver": "580", "vbios": "9A", "util": 0}],
                "util_max": 0, "procs": [], "smi_ok": True},
        "containers": [], "docker_error": None, "ollama_loaded": [], "holds": [],
        "failed_units": [], "running_services": ["ssh.service", "docker.service"],
        "nics": {"enP7s7": ["192.168.1.41/24"], "enp1s0f0np0": ["192.168.100.1/24"]},
        "links": {"enP7s7": {"mtu": "1500", "state": "UP"}, "enp1s0f0np0": {"mtu": "9000", "state": "UP"}},
        "reboot": {"required": False, "reasons": []},
    }
    st.update(kw)
    return st


IDLE_LAB = {"ok": True, "services": {"gpus": {}, "services": []}, "jobs": {"jobs": []}}


class TmpEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {
            "FLEET_MAINT_KILL": str(t / "kill"), "FLEET_MAINT_PAUSE": str(t / "pause"),
            "FLEET_MAINT_HOLDS": str(t / "holds"), "FLEET_MAINT_STATE": str(t / "state"),
            "FLEET_MAINT_APPROVALS": str(t / "approvals.json")})
        self.env.start()
        self.t = t

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def approve(self, entries):
        (self.t / "approvals.json").write_text(json.dumps(entries))


# ---------------------------------------------------------------------------

class PolicyTests(unittest.TestCase):
    def test_valid(self):
        p = policy()
        self.assertEqual(fm.class_hosts(p, "pod"), ["a1", "a2", "a3"])
        self.assertEqual(p["classes"]["pod"]["window"]["start"], (2, 0))

    def test_unknown_class_key_rejected(self):
        p = base_policy()
        p["classes"]["pod"]["auto_aply"] = True
        with self.assertRaises(fm.PolicyError):
            fm.validate_policy(p)

    def test_auto_apply_needs_tag_and_window(self):
        p = base_policy()
        del p["classes"]["pod"]["approval_tag"]
        with self.assertRaises(fm.PolicyError):
            fm.validate_policy(p)
        p = base_policy()
        del p["classes"]["pod"]["window"]
        with self.assertRaises(fm.PolicyError):
            fm.validate_policy(p)

    def test_canary_must_be_member(self):
        p = base_policy()
        p["classes"]["pod"]["canary"] = "b1"
        with self.assertRaises(fm.PolicyError):
            fm.validate_policy(p)

    def test_bad_window(self):
        for w in ({"days": ["Sunday"], "start": "02:00", "end": "03:00"},
                  {"days": ["Sun"], "start": "05:00", "end": "03:00"},
                  {"days": ["Sun"], "start": "2am", "end": "03:00"}):
            p = base_policy()
            p["classes"]["pod"]["window"] = w
            with self.assertRaises(fm.PolicyError):
                fm.validate_policy(p)

    def test_shipped_policy_is_valid_and_report_only_where_required(self):
        p = fm.load_policy(Path(__file__).resolve().parent / "policy.example.yaml")
        for c in ("dgx-station", "gpu-inference-prod", "workstation", "unmanaged"):
            self.assertFalse(p["classes"][c]["auto_apply"], c)
        self.assertEqual(p["classes"]["gb10-pod"]["canary"], "edge4")
        self.assertTrue(p["classes"]["gb10-pod"]["as_set"])
        self.assertEqual(p["classes"]["dgx-station"]["reboot"], "never")
        self.assertFalse(p["hosts"]["station1"]["reboot_tested"])
        self.assertEqual(sorted(fm.class_hosts(p, "gb10-pod")),
                         ["dgxs1", "dgxs2", "dgxs3", "dgxs4", "edge1", "edge2", "edge3", "edge4"])


class ApprovalTests(TmpEnv):
    H = ["a1", "a2", "a3"]

    def test_missing_file(self):
        ok, why, _ = fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)
        self.assertFalse(ok)
        self.assertIn("no approvals file", why)

    def test_bad_json(self):
        (self.t / "approvals.json").write_text("{nope")
        self.assertFalse(fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)[0])

    def test_not_a_list(self):
        self.approve({"contains": "# fleet-maint-pod"})
        self.assertFalse(fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)[0])

    def test_wrong_tag(self):
        self.approve([{"contains": "# gb10-fw-0925", "expires": SUN_0300 + 3600, "hosts": self.H}])
        self.assertFalse(fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)[0])

    def test_expired(self):
        self.approve([{"contains": "# fleet-maint-pod", "expires": SUN_0300 - 1, "hosts": self.H}])
        ok, why, _ = fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)
        self.assertFalse(ok)
        self.assertIn("expired", why)

    def test_too_long(self):
        self.approve([{"contains": "# fleet-maint-pod", "expires": SUN_0300 + 200 * 86400,
                       "hosts": self.H}])
        ok, why, _ = fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)
        self.assertFalse(ok)
        self.assertIn("92 days", why)

    def test_hosts_list_required(self):
        self.approve([{"contains": "# fleet-maint-pod", "expires": SUN_0300 + 3600}])
        self.assertFalse(fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)[0])

    def test_set_partial_coverage_refused(self):
        self.approve([{"contains": "# fleet-maint-pod", "expires": SUN_0300 + 3600,
                       "hosts": ["a1", "a2"]}])
        ok, why, _ = fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)
        self.assertFalse(ok)
        self.assertIn("a3", why)

    def test_non_set_partial_coverage_narrows(self):
        self.approve([{"contains": "# fleet-maint-loose", "expires": SUN_0300 + 3600, "hosts": ["b1"]}])
        ok, _, hosts = fm.check_approval("# fleet-maint-loose", ["b1", "b2"], False, SUN_0300)
        self.assertTrue(ok)
        self.assertEqual(hosts, ["b1"])

    def test_ok(self):
        self.approve([{"contains": "# fleet-maint-pod", "expires": SUN_0300 + 30 * 86400,
                       "hosts": self.H, "note": "weekly"}])
        ok, why, hosts = fm.check_approval("# fleet-maint-pod", self.H, True, SUN_0300)
        self.assertTrue(ok)
        self.assertEqual(hosts, self.H)


class WindowTests(unittest.TestCase):
    def test_window(self):
        w = fm.parse_window({"days": ["Sun"], "start": "02:00", "end": "06:00", "tz": "America/New_York"})
        self.assertTrue(fm.in_window(w, SUN_0300))
        self.assertFalse(fm.in_window(w, SUN_0700))
        self.assertFalse(fm.in_window(w, MON_0300))
        self.assertFalse(fm.in_window(None, SUN_0300))


class ParseTests(unittest.TestCase):
    SAMPLE = "\n".join([
        "@@meta", "hostname=edge4", "kernel=7.0.0-1019-nvidia", "os_kind=Linux", "os_id=ubuntu",
        "os_version=24.04", "uptime_s=172800", "boot_id=abc", "has_apt-get=1", "has_fwupdmgr=1",
        "has_nvidia-smi=1", "has_docker=1", "DGX_NAME=DGX Spark", "DGX_OTA_VERSION=7.5.0",
        "DGX_OTA_VERSION=7.6.0", "kernel_latest_installed=7.0.0-1020-nvidia",
        "dmi_vendor=MSI", "dmi_product=MS-C931",
        "@@apt_upgradable",
        "curl/noble-updates,noble-security 8.5.0-2ubuntu10.15 arm64 [upgradable from: 8.5.0-2ubuntu10.13]",
        "gnome-shell/noble-updates 46.0-0ubuntu6~24.04.15 arm64 [upgradable from: 46.0-0ubuntu6~24.04.14]",
        "@@apt_errors", "@@apt_holds", "snapd", "@@apt_meta", "lists_mtime=1790372045",
        "reboot_required=1", "reboot_required_pkgs=linux-image-7.0.0-1020-nvidia",
        "@@fwupd_updates",
        json.dumps({"Devices": [{"Name": "Embedded Controller", "Version": "10700",
                                 "Releases": [{"Version": "10800"}]}]}),
        "@@fwupd_devices",
        json.dumps({"Devices": [{"Name": "Embedded Controller", "Version": "10700", "Flags": ["updatable"]},
                                {"Name": "Disk", "Version": "1", "Flags": []}]}),
        "@@gpu", "0, NVIDIA GB10, 580.178.04, 9A.0B.2D.00.00, 3 %, 4880 MiB",
        "@@gpu_util_samples", "3", "40", "2",
        "@@gpu_procs", "@@docker", "lab-j1-r0|vllm:latest|Up 2 hours",
        "@@ollama", '{"models":[]}',
        "@@holds", "/home/ops/.fleet-maint-hold|bench tp8 expires=1",
        "@@failed_units", "@@running_services", "ssh.service",
        "@@nics", "enP7s7 192.168.1.44/24", "@@links", "enP7s7 1500 UP",
        "@@swap_parity", "swapimg_bytes=17179869184", "earlyoom=0",
        "@@end", "ok"])

    def test_parse(self):
        st = fm.parse_collect("edge4", self.SAMPLE, 0, now=1790375594)
        self.assertTrue(st["reachable"])
        self.assertEqual(st["pkg"]["pending"], 2)
        self.assertEqual(st["pkg"]["security"], 1)
        self.assertEqual(st["pkg"]["holds"], ["snapd"])
        self.assertEqual(st["dgx"]["ota"], "7.6.0")
        self.assertEqual(st["model"], "MSI MS-C931")
        self.assertEqual(st["firmware"]["pending"][0]["new"], "10800")
        self.assertEqual(st["firmware"]["devices"], [{"device": "Embedded Controller", "version": "10700"}])
        self.assertEqual(st["gpu"]["util_max"], 40)
        self.assertEqual(st["containers"][0]["name"], "lab-j1-r0")
        self.assertTrue(st["reboot"]["required"])
        self.assertEqual(len(st["reboot"]["reasons"]), 2)   # reboot-required file + newer kernel
        self.assertEqual(st["uptime_days"], 2.0)

    def test_unreachable(self):
        st = fm.parse_collect("x", "ssh: connect to host x port 22: No route to host", 255)
        self.assertFalse(st["reachable"])
        self.assertIn("No route", st["errors"][0])

    def test_apt_errors_void_the_count(self):
        s = self.SAMPLE.replace("@@apt_errors", "@@apt_errors\nE: Problem with MergeList /var/lib/apt/lists/x")
        st = fm.parse_collect("h", s, 0)
        self.assertIsNone(st["pkg"]["pending"])

    def test_nixos(self):
        s = "\n".join(["@@meta", "kernel=7.2.7", "os_kind=Linux", "os_id=nixos", "nixos=1",
                       "@@nixos", "running=/nix/store/a-sys", "booted=/nix/store/a0-sys",
                       "staged=/nix/store/b-sys", "booted_kernel=/k1", "staged_kernel=/k1",
                       "staged_mtime=1790323995", "Result=success", "@@end"])
        st = fm.parse_collect("ws1", s, 0, now=1790375594)
        self.assertEqual(st["pkg"]["manager"], "nixos")
        self.assertTrue(st["nixos"]["staged_not_running"])
        self.assertFalse(st["nixos"]["kernel_change_pending"])
        self.assertTrue(st["reboot"]["required"])


class BusyTests(unittest.TestCase):
    def test_idle(self):
        b = fm.evaluate_busy(idle_state("a1"), BUSY_ALL, IDLE_LAB, [], now=SUN_0300)
        self.assertFalse(b["busy"], b)

    def test_any_container(self):
        st = idle_state("a1", containers=[{"name": "x", "image": "busybox"}])
        self.assertTrue(fm.evaluate_busy(st, BUSY_ALL, IDLE_LAB)["busy"])

    def test_container_ignore(self):
        st = idle_state("a1", containers=[{"name": "cloudflared", "image": "cloudflare/cloudflared"}])
        self.assertFalse(fm.evaluate_busy(st, dict(BUSY_ALL, container_ignore=["cloudflared"]), IDLE_LAB)["busy"])
        self.assertTrue(fm.evaluate_busy(st, BUSY_ALL, IDLE_LAB)["busy"])

    def test_pattern_container_only(self):
        st = idle_state("a1", containers=[{"name": "watchtower", "image": "w"}])
        self.assertFalse(fm.evaluate_busy(st, {"container_patterns": ["*vllm*"]}, None)["busy"])
        st = idle_state("a1", containers=[{"name": "srv", "image": "vllm/vllm-openai"}])
        self.assertTrue(fm.evaluate_busy(st, {"container_patterns": ["*vllm*"]}, None)["busy"])

    def test_docker_unreadable_is_busy(self):
        st = idle_state("a1", containers=None, docker_error="permission denied")
        self.assertTrue(fm.evaluate_busy(st, BUSY_ALL, IDLE_LAB)["busy"])

    def test_gpu_util(self):
        st = idle_state("a1")
        st["gpu"]["util_max"] = 60
        b = fm.evaluate_busy(st, BUSY_ALL, IDLE_LAB)
        self.assertTrue(b["busy"])
        self.assertIn("60%", b["reasons"][0])

    def test_gpu_procs(self):
        st = idle_state("a1")
        st["gpu"]["procs"] = ["123, VLLM::EngineCore, 234966 MiB"]
        self.assertTrue(fm.evaluate_busy(st, BUSY_ALL, IDLE_LAB)["busy"])

    def test_lab_lock_and_jobs(self):
        lab = copy.deepcopy(IDLE_LAB)
        lab["services"]["gpus"] = {"g": {"node": "a1", "lock": {"by": "bench"}, "processes": []}}
        self.assertTrue(fm.evaluate_busy(idle_state("a1"), BUSY_ALL, lab)["busy"])
        self.assertFalse(fm.evaluate_busy(idle_state("a2"), BUSY_ALL, lab)["busy"])
        lab = copy.deepcopy(IDLE_LAB)
        lab["jobs"]["jobs"] = [{"id": "j1", "state": "running", "nodes": ["a1", "a2"]},
                               {"id": "j0", "state": "done", "nodes": ["a3"]}]
        self.assertTrue(fm.evaluate_busy(idle_state("a2"), BUSY_ALL, lab)["busy"])
        self.assertFalse(fm.evaluate_busy(idle_state("a3"), BUSY_ALL, lab)["busy"])
        lab["jobs"]["jobs"] = [{"id": "j2", "state": "some-new-state", "nodes": ["a3"]}]
        self.assertTrue(fm.evaluate_busy(idle_state("a3"), BUSY_ALL, lab)["busy"])

    def test_lab_unreadable_is_busy(self):
        b = fm.evaluate_busy(idle_state("a1"), BUSY_ALL, {"ok": False, "error": "timeout"})
        self.assertTrue(b["busy"])

    def test_hold_files(self):
        st = idle_state("a1", holds=["/home/ops/.fleet-maint-hold|bench"])
        self.assertTrue(fm.evaluate_busy(st, {}, None)["busy"])
        b = fm.evaluate_busy(idle_state("a1"), {}, None, ["/c/holds/a1|reason=x expires=100"], now=SUN_0300)
        self.assertTrue(b["busy"])                # a stale hold still blocks
        self.assertEqual(len(b["stale_holds"]), 1)

    def test_central_holds(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"FLEET_MAINT_HOLDS": d}):
            self.assertEqual(fm.central_holds("a1", "pod"), [])
            Path(d, "class-pod").write_text("reason=bench")
            self.assertEqual(len(fm.central_holds("a1", "pod")), 1)
            Path(d, "ALL").write_text("reason=freeze")
            self.assertEqual(len(fm.central_holds("zz", "other")), 1)

    def test_unreachable(self):
        self.assertTrue(fm.evaluate_busy({"host": "x", "reachable": False}, {}, None)["busy"])


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.p = policy()
        self.cls = self.p["classes"]["pod"]
        self.idle = {"busy": False, "reasons": []}

    def e(self, st, sudo=True, busy=None, host="a1"):
        return fm.host_eligibility(host, st, self.cls, self.p["hosts"][host], busy or self.idle, sudo)

    def test_ready(self):
        r = self.e(idle_state("a1"))
        self.assertEqual(r["state"], "ready")
        self.assertEqual(r["packages"], ["curl", "gnome-shell"])

    def test_up_to_date(self):
        self.assertEqual(self.e(idle_state("a1", pkgs=[]))["state"], "up-to-date")

    def test_firmware_only_is_work(self):
        r = self.e(idle_state("a1", pkgs=[], fw=[{"device": "EC", "current": "1", "new": "2"}]))
        self.assertEqual(r["state"], "ready")

    def test_forbidden_is_manual(self):
        r = self.e(idle_state("a1", pkgs=[{"name": "nvidia-dkms-580", "security": False}]))
        self.assertEqual(r["state"], "manual")

    def test_busy_blocks(self):
        self.assertEqual(self.e(idle_state("a1"), busy={"busy": True, "reasons": ["x"]})["state"], "blocked")

    def test_apt_broken_blocks(self):
        st = idle_state("a1")
        st["pkg"]["errors"] = ["E: MergeList"]
        self.assertEqual(self.e(st)["state"], "blocked")

    def test_no_sudo_blocks(self):
        self.assertEqual(self.e(idle_state("a1"), sudo=False)["state"], "blocked")

    def test_non_apt_is_manual(self):
        st = idle_state("a1")
        st["pkg"]["manager"] = "nixos"
        self.assertEqual(self.e(st)["state"], "manual")

    def test_security_only_selection(self):
        cls = self.p["classes"]["loose"]
        r = fm.host_eligibility("b1", idle_state("b1"), cls, self.p["hosts"]["b1"], self.idle, True)
        self.assertEqual(r["packages"], ["curl"])

    def test_reboot_tested_gate(self):
        cls = dict(self.cls, require_reboot_tested=True)
        r = fm.host_eligibility("a1", idle_state("a1"), cls, self.p["hosts"]["a1"], self.idle, True)
        self.assertEqual(r["state"], "blocked")
        r = fm.host_eligibility("a1", idle_state("a1"), cls, dict(self.p["hosts"]["a1"], reboot_tested=True),
                                self.idle, True)
        self.assertEqual(r["state"], "ready")


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.p = policy()
        self.ok = (True, "approved", ["a1", "a2", "a3"])
        self.states = {h: idle_state(h) for h in ("a1", "a2", "a3")}
        self.busy = {h: {"busy": False, "reasons": []} for h in self.states}
        self.sudo = {h: True for h in self.states}

    def plan(self, **kw):
        a = dict(states=self.states, busy=self.busy, sudo_ok=self.sudo, approval=self.ok,
                 window_open=True, stop_reason=None)
        a.update(kw)
        return fm.plan_class(self.p, "pod", a["states"], a["busy"], a["sudo_ok"], a["approval"],
                             a["window_open"], a["stop_reason"])

    def test_run_canary_first(self):
        pl = self.plan()
        self.assertEqual(pl["verdict"], "run")
        self.assertEqual(pl["order"], ["a1", "a2", "a3"])

    def test_gates(self):
        self.assertEqual(self.plan(stop_reason="kill")["verdict"], "killswitch")
        self.assertEqual(self.plan(window_open=False)["verdict"], "outside-window")
        self.assertEqual(self.plan(approval=(False, "no", []))["verdict"], "refused")
        self.assertEqual(self.plan(approval=None)["verdict"], "refused")
        self.assertEqual(fm.plan_class(self.p, "prod", {}, {}, {}, None, True)["verdict"], "report-only")

    def test_set_defers_when_one_member_busy(self):
        self.busy["a3"] = {"busy": True, "reasons": ["container running: bench"]}
        pl = self.plan()
        self.assertEqual(pl["verdict"], "deferred")
        self.assertIn("a3", pl["reason"])

    def test_set_defers_when_member_unreachable(self):
        self.states["a2"] = {"host": "a2", "reachable": False}
        self.assertEqual(self.plan()["verdict"], "deferred")

    def test_canary_up_to_date_hands_over(self):
        self.states["a1"] = idle_state("a1", pkgs=[])
        pl = self.plan()
        self.assertEqual(pl["verdict"], "run")
        self.assertEqual(pl["canary"], "a2")
        self.assertEqual(pl["order"], ["a2", "a3"])

    def test_nothing_to_do(self):
        for h in self.states:
            self.states[h] = idle_state(h, pkgs=[])
        self.assertEqual(self.plan()["verdict"], "nothing-to-do")

    def test_non_set_blocked_canary_defers(self):
        states = {"b1": idle_state("b1"), "b2": idle_state("b2")}
        busy = {"b1": {"busy": True, "reasons": ["x"]}, "b2": {"busy": False, "reasons": []}}
        pl = fm.plan_class(self.p, "loose", states, busy, {"b1": True, "b2": True},
                           (True, "ok", ["b1", "b2"]), True)
        self.assertEqual(pl["verdict"], "deferred")
        self.assertIn("canary b1", pl["reason"])

    def test_host_outside_allowlist_blocked(self):
        states = {"b1": idle_state("b1"), "b2": idle_state("b2")}
        busy = {h: {"busy": False, "reasons": []} for h in states}
        pl = fm.plan_class(self.p, "loose", states, busy, {"b1": True, "b2": True},
                           (True, "ok", ["b1"]), True)
        self.assertEqual(pl["verdict"], "run")
        self.assertEqual(pl["order"], ["b1"])
        self.assertEqual(pl["hosts"]["b2"]["state"], "blocked")


class FakeRunner:
    def __init__(self, states):
        self.states = states
        self.collects = []

    def collect(self, host):
        self.collects.append(host)
        return copy.deepcopy(self.states[host])


class RunClassTests(TmpEnv):
    def setUp(self):
        super().setUp()
        self.p = policy()
        self.approve([{"contains": "# fleet-maint-pod", "expires": time.time() + 86400,
                       "hosts": ["a1", "a2", "a3"]}])
        self.runner = FakeRunner({h: idle_state(h) for h in ("a1", "a2", "a3")})
        self.called = []
        mock.patch.object(fm, "sudo_file_ok", lambda p, h: True).start()
        self.addCleanup(mock.patch.stopall)

    def apply_fn(self, results):
        def f(host, cls, runner, policy, lab_fetch, sleep=None):
            self.called.append(host)
            return {"host": host, "status": results.get(host, "ok"), "reason": "x"}
        return f

    def run_(self, results=None, **kw):
        return fm.run_class(self.p, "pod", self.runner, lab_fetch=lambda hub: IDLE_LAB,
                            override_window=True, apply_fn=self.apply_fn(results or {}),
                            sleep=lambda s: None, **kw)

    def test_canary_failure_stops_class(self):
        r = self.run_({"a1": "failed"})
        self.assertEqual(r["status"], "canary-failed")
        self.assertEqual(self.called, ["a1"])
        self.assertEqual(r["skipped"], ["a2", "a3"])

    def test_canary_needs_hands_stops_class(self):
        r = self.run_({"a1": "needs-hands"})
        self.assertEqual(r["status"], "canary-failed")
        self.assertEqual(self.called, ["a1"])

    def test_all_ok(self):
        r = self.run_()
        self.assertEqual(self.called, ["a1", "a2", "a3"])
        self.assertIn(r["status"], ("completed", "parity-drift"))
        self.assertIn("parity", r)

    def test_failure_after_canary_halts_rest(self):
        r = self.run_({"a2": "failed"})
        self.assertEqual(self.called, ["a1", "a2"])   # max_parallel 1: a3 never starts
        self.assertEqual(r["hosts"]["a3"]["status"], "skipped")
        self.assertEqual(r["status"], "failed")

    def test_kill_switch(self):
        (self.t / "kill").write_text("")
        r = self.run_()
        self.assertEqual(r["status"], "killswitch")
        self.assertEqual(self.called, [])

    def test_pause(self):
        (self.t / "pause").write_text("")
        self.assertEqual(self.run_()["status"], "killswitch")
        self.assertEqual(self.called, [])

    def test_no_approval(self):
        (self.t / "approvals.json").unlink()
        r = self.run_()
        self.assertEqual(r["status"], "refused")
        self.assertEqual(self.called, [])
        self.assertEqual(self.runner.collects, [])    # the gate runs before any ssh

    def test_busy_member_defers_whole_set(self):
        self.runner.states["a3"]["containers"] = [{"name": "bench", "image": "vllm"}]
        r = self.run_()
        self.assertEqual(r["status"], "deferred")
        self.assertEqual(self.called, [])

    def test_dry_run_touches_nothing(self):
        r = self.run_(dry_run=True)
        self.assertEqual(r["status"], "dry-run")
        self.assertEqual(self.called, [])

    def test_dry_run_with_closed_gate_still_probes(self):
        (self.t / "approvals.json").unlink()
        r = self.run_(dry_run=True)
        self.assertEqual(r["status"], "refused")
        self.assertEqual(r["probe"]["verdict"], "run")
        self.assertEqual(self.called, [])

    def test_window_enforced_without_override(self):
        r = fm.run_class(self.p, "pod", self.runner, lab_fetch=lambda hub: IDLE_LAB,
                         now=MON_0300, apply_fn=self.apply_fn({}))
        self.assertEqual(r["status"], "outside-window")
        self.assertEqual(self.called, [])


class ScriptedRunner:
    """Replays collect states in order and answers run/sudo by substring."""

    def __init__(self, collects, answers, boot_ids=None):
        self.collects = list(collects)
        self.answers = answers
        self.boot_ids = list(boot_ids or [])
        self.sudo_cmds = []
        self.run_cmds = []

    def collect(self, host):
        return copy.deepcopy(self.collects.pop(0) if len(self.collects) > 1 else self.collects[0])

    def _answer(self, cmd):
        for k, v in self.answers.items():
            if k in cmd:
                return v
        return (0, "")

    def run(self, host, cmd, stdin=None, timeout=120):
        self.run_cmds.append(cmd)
        if "boot_id" in cmd:
            return (0, self.boot_ids.pop(0)) if self.boot_ids else (255, "")
        return self._answer(cmd)

    def sudo(self, host, cmd, timeout=3600):
        self.sudo_cmds.append(cmd)
        return self._answer(cmd)


UPG = ("curl/noble-updates,noble-security 8.5.0-2ubuntu10.15 arm64 [upgradable from: 8.5.0-2ubuntu10.13]\n"
       "gnome-shell/noble-updates 46.0-0ubuntu6~24.04.15 arm64 [upgradable from: 46.0-0ubuntu6~24.04.14]")


class ApplyHostTests(TmpEnv):
    def setUp(self):
        super().setUp()
        self.p = policy()
        self.cls = self.p["classes"]["pod"]
        mock.patch.object(fm, "sudo_file_ok", lambda p, h: True).start()
        self.addCleanup(mock.patch.stopall)
        self.clock = [0.0]

    def now(self):
        return self.clock[0]

    def sleep(self, s):
        self.clock[0] += s

    def go(self, runner):
        return fm.apply_host("a1", self.cls, runner, self.p, lambda hub: IDLE_LAB,
                             sleep=self.sleep, now=self.now)

    def test_happy_path_with_reboot(self):
        before = idle_state("a1", fw=[{"device": "EC", "current": "1", "new": "2"}])
        after = idle_state("a1", pkgs=[], boot_id="boot-2")
        r = ScriptedRunner([before, before, after],
                           {"apt list --upgradable": (0, UPG), "fwupdmgr update": (0, "done")},
                           boot_ids=["boot-1", "boot-2"])
        rec = self.go(r)
        self.assertEqual(rec["status"], "ok", rec.get("reason"))
        self.assertTrue(any("full-upgrade" in c and "--force-confold" in c for c in r.sudo_cmds))
        self.assertIn("systemctl reboot", r.sudo_cmds)
        sim = [c for c in r.run_cmds if "apt-get -s" in c]
        self.assertEqual(len(sim), 1)
        self.assertIn("Debug::NoLocking=1", sim[0])

    def test_forbidden_package_in_simulation(self):
        st = idle_state("a1")
        r = ScriptedRunner([st], {"apt list --upgradable": (0, UPG),
                                  "apt-get -s": (0, "Inst nvidia-dkms-580 (580.1 Ubuntu)\nInst curl (8.5)")})
        rec = self.go(r)
        self.assertEqual(rec["status"], "failed")
        self.assertIn("nvidia-dkms-580", rec["reason"])
        self.assertFalse(any("full-upgrade" in c for c in r.sudo_cmds))

    def test_dkms_failure_no_reboot(self):
        st = idle_state("a1")
        r = ScriptedRunner([st], {"apt list --upgradable": (0, UPG), "apt-get -s": (0, "Inst curl (8.5)"),
                                  "full-upgrade": (100, "Error! Bad return status for module build on kernel")})
        rec = self.go(r)
        self.assertEqual(rec["status"], "needs-hands")
        self.assertNotIn("systemctl reboot", r.sudo_cmds)

    def test_never_comes_back(self):
        st = idle_state("a1", fw=[{"device": "EC", "current": "1", "new": "2"}])
        r = ScriptedRunner([st], {"apt list --upgradable": (0, UPG), "fwupdmgr update": (0, "")},
                           boot_ids=[])
        rec = self.go(r)
        self.assertEqual(rec["status"], "needs-hands")
        self.assertIn("did not come back", rec["reason"])
        # never a power action of any kind
        self.assertFalse(any("pdu" in c.lower() or "poweroff" in c for c in r.sudo_cmds + r.run_cmds))

    def test_became_busy_is_blocked(self):
        st = idle_state("a1", containers=[{"name": "bench", "image": "x"}])
        r = ScriptedRunner([st], {})
        rec = self.go(r)
        self.assertEqual(rec["status"], "blocked")
        self.assertEqual(r.sudo_cmds, [])

    def test_verify_failure(self):
        before = idle_state("a1", fw=[{"device": "EC", "current": "1", "new": "2"}])
        after = idle_state("a1", pkgs=[], boot_id="boot-2", nics={"enP7s7": ["192.168.1.99/24"]})
        r = ScriptedRunner([before, before, after], {"apt list --upgradable": (0, UPG),
                                                     "fwupdmgr update": (0, "")},
                           boot_ids=["boot-2"])
        rec = self.go(r)
        self.assertEqual(rec["status"], "failed")
        self.assertIn("enp1s0f0np0", rec["reason"])

    def test_security_command_is_explicit_list(self):
        self.assertIn("install --only-upgrade curl libssl3", fm.apt_command("security", ["curl", "libssl3"]))
        with self.assertRaises(ValueError):
            fm.apt_command("security", ["curl; rm -rf /"])
        with self.assertRaises(ValueError):
            fm.apt_command("security", [])


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.cls = policy()["classes"]["pod"]

    def test_clean(self):
        b = idle_state("a1")
        a = idle_state("a1", pkgs=[])
        self.assertEqual(fm.verify_host(b, a, self.cls), [])

    def test_problems(self):
        b = idle_state("a1")
        a = idle_state("a1", failed_units=["nvidia-persistenced.service"],
                       running_services=["ssh.service"],
                       fw=[{"device": "EC", "current": "10700", "new": "10800"}])
        a["pkg"]["holds"] = []
        b["pkg"]["holds"] = ["snapd"]
        a["links"]["enp1s0f0np0"]["mtu"] = "1500"
        a["gpu"]["gpus"] = []
        probs = " | ".join(fm.verify_host(b, a, self.cls))
        for needle in ("nvidia-persistenced", "docker.service", "holds changed", "mtu",
                       "GPUs visible", "AC power cycle"):
            self.assertIn(needle, probs)

    def test_ignored_nics_and_services(self):
        b = idle_state("a1", nics={"docker0": ["172.17.0.1/16"]}, running_services=["fwupd.service"])
        a = idle_state("a1", nics={}, running_services=[])
        self.assertEqual(fm.verify_host(b, a, dict(self.cls, firmware=False)), [])


class ParityTests(unittest.TestCase):
    def st(self, model, ec, kernel="7.0.0-1019-nvidia"):
        s = idle_state("x", model=model)
        s["kernel"]["running"] = kernel
        s["firmware"]["devices"] = [{"device": "Embedded Controller", "version": ec},
                                    {"device": "UEFI Device Firmware", "version": "10900"},
                                    {"device": "UEFI Device Firmware", "version": "522"}]
        return s

    def test_vendor_difference_is_not_drift(self):
        states = {"edge4": self.st("MSI MS-C931", "10800"), "dgxs3": self.st("NVIDIA DGX Spark", "0x03000508")}
        self.assertEqual(fm.parity_table(states, ["edge4", "dgxs3"])["drift"], [])

    def test_same_model_difference_is_drift(self):
        states = {"edge1": self.st("MSI MS-C931", "10700"), "edge4": self.st("MSI MS-C931", "10800")}
        d = fm.parity_table(states, ["edge1", "edge4"])["drift"]
        self.assertEqual(len(d), 1)
        self.assertIn("Embedded Controller", d[0])

    def test_kernel_is_pod_wide(self):
        states = {"edge4": self.st("MSI MS-C931", "10800"),
                  "dgxs3": self.st("NVIDIA DGX Spark", "0x03000508", kernel="7.0.0-1020-nvidia")}
        self.assertIn("kernel", fm.parity_table(states, ["edge4", "dgxs3"])["drift"])

    def test_duplicate_device_names_are_distinct_columns(self):
        pt = fm.parity_table({"a": self.st("M", "1")}, ["a"])
        self.assertIn("fw:UEFI Device Firmware #2", pt["columns"])


class AlertTests(unittest.TestCase):
    def test_uu_on_manual_only_class(self):
        p = fm.validate_policy({
            "classes": {"dgx": {"forbid_packages": ["^nvidia-"], "busy": {}, "uu_blacklist_required": True}},
            "hosts": {"station1": {"class": "dgx"}}})
        st = idle_state("station1")
        st["pkg"]["uu"] = {"installed": True, "enabled": True, "blacklist": []}
        self.assertTrue(any("unattended-upgrades" in a for a in fm.alerts(p, {"station1": st})))
        st["pkg"]["uu"]["blacklist"] = ['"nvidia-"', '"linux-"']
        self.assertFalse(any("unattended-upgrades" in a for a in fm.alerts(p, {"station1": st})))


class SudoSecrecyTests(unittest.TestCase):
    def test_password_never_in_argv_and_redacted(self):
        with tempfile.TemporaryDirectory() as d:
            pw = Path(d, "pw")
            pw.write_text("s3cret-value\n")
            p = policy()
            p["hosts"]["a1"]["sudo_file"] = str(pw)
            seen = {}

            def fake_run(argv, input=None, capture_output=True, text=True, timeout=None):
                seen["argv"], seen["input"] = argv, input
                return mock.Mock(returncode=0, stdout="echo s3cret-value leaked", stderr="")
            with mock.patch.object(fm.subprocess, "run", fake_run):
                rc, out = fm.SSHRunner(p).sudo("a1", "apt-get update")
            self.assertNotIn("s3cret-value", " ".join(seen["argv"]))
            self.assertEqual(seen["input"], "s3cret-value\n")
            self.assertNotIn("s3cret-value", out)
            self.assertIn("sudo -S -p ''", seen["argv"][-1])


if __name__ == "__main__":
    unittest.main(verbosity=1)
