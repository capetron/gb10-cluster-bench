#!/usr/bin/env python3
"""fwupd-pending-list.py - print pending firmware updates, one line per device.

    fwupdmgr get-updates --json | python3 fwupd-pending-list.py
    ssh node 'fwupdmgr get-updates --json' | python3 fwupd-pending-list.py

Shows device name, current version -> offered version, urgency and summary. Read-only.
"""
import json
import sys

try:
    d = json.load(sys.stdin)
except Exception as e:
    print("  (no json:", str(e)[:80], ")")
    sys.exit()
devs = d.get("Devices", [])
if not devs:
    print("  no updates")
for dev in devs:
    for r in dev.get("Releases", []):
        print("  %-38s %s -> %s  [%s] %s" % (dev.get("Name", "?")[:38], dev.get("Version", "?"),
              r.get("Version", "?"), r.get("Urgency", "?"), r.get("Summary", "")[:60]))
        break
