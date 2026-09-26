#!/usr/bin/env bash
# setup.sh - check prerequisites and run the offline test suites. Changes nothing on the system.
set -u
cd "$(dirname "$0")"
ok=1
need() { if command -v "$1" >/dev/null 2>&1; then echo "found   $1"; else echo "MISSING $1 ($2)"; ok=0; fi; }
need python3 "all tools; 3.9 or newer"
need bash    "shell tools"
need ssh     "node/, cluster/ and fleet-maint/ reach nodes over key-based ssh"
need curl    "cluster/watch-serve.sh, power/pdu-sample-unifi.sh"
need jq      "power/pdu-sample-unifi.sh only"
python3 - <<'EOF' || ok=0
import sys
if sys.version_info < (3, 9):
    sys.exit("python3 is %d.%d; 3.9 or newer is required" % sys.version_info[:2])
try:
    import yaml  # noqa: F401
    print("found   PyYAML (fleet-maint)")
except ImportError:
    print("MISSING PyYAML (fleet-maint only): python3 -m pip install --user pyyaml")
    sys.exit(1)
EOF
for f in bench/*.sh node/*.sh cluster/*.sh power/*.sh; do bash -n "$f" || ok=0; done
echo "--- tests"
python3 -m unittest discover -s tests || ok=0
python3 fleet-maint/test_fleetmaint.py || ok=0
if [ "$ok" = 1 ]; then echo "setup: OK"; else echo "setup: see MISSING / FAIL lines above"; exit 1; fi
