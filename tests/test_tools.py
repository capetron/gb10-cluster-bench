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


if __name__ == "__main__":
    unittest.main()
