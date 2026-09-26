"""Offline tests for the bench and power tools. No server, no network, no GPU.

    python3 -m unittest discover -s tests
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*argv):
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, cwd=ROOT)


class PduJoinTests(unittest.TestCase):
    def test_join_attaches_power_to_steady_rows(self):
        with tempfile.TemporaryDirectory() as t:
            res = os.path.join(t, "r.json")
            pdu = os.path.join(t, "p.jsonl")
            json.dump({"phases": {"steady_prose": {"rows": [
                {"concurrency": 1, "rep": 0, "window_start_epoch": 100, "window_end_epoch": 200,
                 "aggregate_output_tok_s": 50.0},
                {"concurrency": 1, "rep": 1, "window_start_epoch": 300, "window_end_epoch": 400,
                 "aggregate_output_tok_s": 52.0}]}}}, open(res, "w"))
            with open(pdu, "w") as f:
                for ts, w in ((110, 100), (150, 110), (190, 120), (320, 130)):
                    f.write(json.dumps({"t": ts, "outlets": [{"i": 5, "w": w}, {"i": 6, "w": 10}]}) + "\n")
            r = run("power/pdu-join.py", res, "steady_prose", pdu, "--outlets", "5,6",
                    "--names", "5=node-a", "--write")
            self.assertEqual(r.returncode, 0, r.stderr)
            block = json.load(open(res))["phases"]["steady_prose"]["power"]
            first, second = block["rows"]
            self.assertEqual(first["samples"], 3)
            self.assertEqual(first["watts_per_outlet"], {"node-a": 110.0, "outlet-6": 10.0})
            self.assertEqual(first["watts_total"], 120.0)
            self.assertAlmostEqual(first["tokens_per_joule"], 50.0 / 120.0, places=4)
            self.assertTrue(second["low_sample_flag"])


class CliTests(unittest.TestCase):
    def test_prefill_bench_help(self):
        r = run("bench/llm-prefill-bench.py", "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("steady", r.stdout)

    def test_fwupd_list_handles_empty_and_bad_input(self):
        for stdin, want in (('{"Devices": []}', "no updates"), ("not json", "no json")):
            r = subprocess.run([sys.executable, "node/fwupd-pending-list.py"], input=stdin,
                               capture_output=True, text=True, cwd=ROOT)
            self.assertIn(want, r.stdout)

    def test_result_files_parse_and_carry_prompt_hash(self):
        d = os.path.join(ROOT, "results")
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            doc = json.load(open(os.path.join(d, name)))
            self.assertTrue("phases" in doc or "rows" in doc, name)


class NodeScriptTests(unittest.TestCase):
    def test_clock_lock_unit_and_installer(self):
        unit = open(os.path.join(ROOT, "node/gpu-clock-lock.service")).read()
        self.assertIn("ExecStart=/usr/bin/nvidia-smi -lgc 300,2200", unit)
        self.assertIn("ExecStop=/usr/bin/nvidia-smi -rgc", unit)
        r = subprocess.run(["bash", "node/install-clock-lock.sh", "--help"], capture_output=True,
                           text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--verify", r.stdout)

    def test_memfrag_verdicts_with_fake_ssh(self):
        # a fake ssh on PATH answers the three reads; no network, no root
        pti = "Node    0, zone   Normal, type    Unmovable " + " ".join(["%d"] * 14)
        with tempfile.TemporaryDirectory() as t:
            fake = os.path.join(t, "ssh")
            with open(fake, "w") as f:
                f.write("#!/bin/bash\n"
                        'case "$*" in\n'
                        '  *pagetypeinfo*) if [[ "$*" == *frag* ]]; then echo "%s"; else echo "%s"; fi;;\n'
                        '  *uptime*) echo "7200.00 100.00";;\n'
                        '  *buddyinfo*) echo "Node 0, zone   Normal 1 1 1 1 1 1 1 1 1 1 1 1 1 3000";;\n'
                        "esac\n" % (pti % ((0,) * 10 + (5000, 0, 0, 10)),
                                     pti % ((10,) + (0,) * 12 + (10,))))
            os.chmod(fake, 0o755)
            env = dict(os.environ, PATH=t + os.pathsep + os.environ["PATH"], SUDO_PASSWORD_FILE="none")
            r = subprocess.run(["bash", "node/memfrag-check.sh", "fresh-node", "frag-node"],
                               capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            lines = {ln.split()[0]: ln for ln in r.stdout.splitlines()[1:]}
            self.assertIn("FRESH", lines["fresh-node"])
            self.assertIn("FRAGMENTED", lines["frag-node"])
            self.assertIn("93.8", lines["fresh-node"])  # 3000 blocks of 32 MiB


if __name__ == "__main__":
    unittest.main()
