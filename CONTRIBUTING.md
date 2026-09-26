# Contributing

Thanks for helping. The most useful contributions are:

1. **Result files from your own GB10 (or other) hardware**, produced with `bench/speed.sh` or
   `bench/llm-prefill-bench.py`, with `launch_records` filled in by `bench/vllm-launch-record.py`.
   Replace hostnames and addresses before you submit; keep everything else as the harness wrote it.
2. **Bug reports with evidence**: the command you ran, the output, and the engine version.
3. **Method fixes**: if you find a bias we missed, open an issue with the measurement that shows it.

Ground rules:

- Tools under `node/`, `cluster/` and `power/` stay read-only unless the script name says
  otherwise. The exceptions are `install-kho-hotfix.py` (it refuses to run on a busy node) and
  `install-clock-lock.sh` (it installs a boot unit and changes nothing else).
- No new runtime dependencies for `bench/` (Python standard library only). `fleet-maint/` may use
  PyYAML.
- Run `./setup.sh` before opening a pull request; it runs every offline test.
- Never put credentials, internal hostnames or addresses in an issue, a result file or a commit.

By contributing you agree that your contribution is licensed under the MIT License of this repo.
