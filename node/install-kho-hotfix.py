#!/usr/bin/env python3
"""install-kho-hotfix.py HOST REPORT.md - install NVIDIA's DGX Spark `kho=off` hotfix on ONE GB10,
reboot it, and verify the running kernel command line.

Background: NVIDIA developer forum thread 383926 (a slowdown on the 7.6.0 image / kernel
7.0.0-1019; NVIDIA staff: the hotfix package nvidia-spark-grub-kho adds kho=off to the kernel
command line). In our cluster one node out of eight was missing the package, so every run with
that node as rank 0 carried an uncontrolled difference. Check all nodes with:
    for h in $CLUSTER_HOSTS; do ssh $h 'grep -o kho=off /proc/cmdline || echo MISSING'; done

Deterministic, no model. One host at a time. Refuses if any container is running (except names
listed in IGNORE_CONTAINERS, comma separated). Never power-cycles: if the node does not return
within 20 minutes it stops and says so.

Needs SUDO_PASSWORD_FILE (path to a file holding the node's sudo password; it is piped to
`sudo -S` over stdin, never put on a command line) unless the node has passwordless sudo, in
which case set SUDO_PASSWORD_FILE=none. FABRIC_NICS (regex) and MTU_NIC pick the NICs recorded
in the report. The report ends in RESULT: MATCH or RESULT: MISMATCH; exit code 0 only on MATCH.
"""
import os
import subprocess
import sys
import time

host, report = sys.argv[1], os.path.expanduser(sys.argv[2])
pwf = os.environ.get("SUDO_PASSWORD_FILE", "none")
pw = None if pwf == "none" else open(os.path.expanduser(pwf)).read().strip()
ignore = {x for x in os.environ.get("IGNORE_CONTAINERS", "").split(",") if x}
nicre = os.environ.get("FABRIC_NICS", "enp1s0f0np0|enP2p1s0f0np0|enP7s7")
mtunic = os.environ.get("MTU_NIC", "enp1s0f0np0")
log = []


def ssh(cmd, sudo=False, timeout=600, check=True):
    if sudo:
        full = (f"sudo -S -p '' bash -c {subprocess.list2cmdline([cmd])}" if pw
                else f"sudo -n bash -c {subprocess.list2cmdline([cmd])}")
    else:
        full = cmd
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, full],
                       input=(pw + "\n") if (sudo and pw) else None, capture_output=True, text=True,
                       timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd!r} rc={r.returncode}: {r.stderr.strip()[:400]}")
    return r.stdout.strip()


def rec(label, value):
    log.append(f"- **{label}:** `{value}`")


def finish(result):
    os.makedirs(os.path.dirname(report) or ".", exist_ok=True)
    with open(report, "w") as f:
        f.write(f"# nvidia-spark-grub-kho on {host} ({time.strftime('%Y-%m-%d %H:%M %Z')})\n\n")
        f.write("\n".join(log) + f"\n\nRESULT: {result}\n")
    print(f"RESULT: {result}")
    sys.exit(0 if result == "MATCH" else 1)


try:
    running = [n for n in ssh("docker ps --format '{{.Names}}'").split() if n not in ignore]
    rec("running containers", running or "none")
    if running:
        finish(f"MISMATCH - busy ({', '.join(running)})")
    rec("before cmdline", ssh("cat /proc/cmdline"))
    rec("before package", ssh("dpkg-query -W -f='${Status} ${Version}' nvidia-spark-grub-kho 2>/dev/null || echo not-installed", check=False))
    ssh("DEBIAN_FRONTEND=noninteractive apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold nvidia-spark-grub-kho", sudo=True, timeout=900)
    rec("package after install", ssh("dpkg-query -W -f='${Status} ${Version}' nvidia-spark-grub-kho"))
    grub = ssh("grep -h kho= /etc/default/grub /etc/default/grub.d/*.cfg 2>/dev/null || true", check=False)
    rec("grub config kho line", grub or "MISSING")
    if "kho=off" not in grub:
        finish("MISMATCH - package installed but no kho=off in grub config")
    ssh("update-grub", sudo=True, timeout=300)
    t0 = time.time()
    ssh("systemctl reboot", sudo=True, check=False, timeout=30)
    time.sleep(60)
    while time.time() - t0 < 1200:
        try:
            if ssh("echo up", timeout=15) == "up":
                break
        except Exception:
            pass
        time.sleep(30)
    else:
        finish("MISMATCH - did not return within 20 min of reboot (needs hands on the node)")
    rec("reboot took", f"{int(time.time() - t0)} s")
    time.sleep(30)
    cmdline = ssh("cat /proc/cmdline")
    rec("after cmdline", cmdline)
    rec("GPU", ssh("nvidia-smi --query-gpu=name,driver_version,clocks.sm --format=csv,noheader"))
    rec("NICs", ssh(f"ip -4 -br addr show | grep -E '{nicre}' | tr -s ' '", check=False).replace("\n", " ; "))
    rec("fabric MTU", ssh(f"cat /sys/class/net/{mtunic}/mtu", check=False))
    rec("systemctl --failed", ssh("systemctl --failed --no-legend | wc -l"))
    finish("MATCH" if "kho=off" in cmdline else "MISMATCH - kho=off not in running cmdline")
except Exception as e:
    rec("error", str(e))
    finish(f"MISMATCH - {str(e)[:200]}")
