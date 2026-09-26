#!/bin/bash
# fleet-maint remote collector. READ-ONLY. Never uses sudo, never writes outside /tmp.
#
# Sent over stdin by fleet-maint.py (`ssh <host> bash -s`), so it runs under bash no
# matter what the login shell is (fish, zsh on macOS). Must stay bash 3.2
# compatible for macOS: no mapfile, no associative arrays, no ${var,,}.
#
# Output: sections delimited by "@@<name>" lines; fleetmaint.parse_collect() turns them
# into the per-host JSON. Every probe is wrapped in `timeout` where one exists so a hung
# daemon (fwupd, docker, nvidia-smi) cannot stall the fleet run.
export LC_ALL=C
export PATH="$PATH:/usr/sbin:/sbin:/run/current-system/sw/bin"
T() { if command -v timeout >/dev/null 2>&1; then timeout "$@"; else shift; "$@"; fi; }
sec() { echo "@@$1"; }

sec meta
echo "hostname=$(hostname 2>/dev/null)"
echo "uname=$(uname -srm 2>/dev/null)"
echo "kernel=$(uname -r 2>/dev/null)"
echo "os_kind=$(uname -s 2>/dev/null)"
echo "user=$(id -un 2>/dev/null)"
if [ -r /etc/os-release ]; then
  ( . /etc/os-release; echo "os_id=${ID:-}"; echo "os_version=${VERSION_ID:-}"; echo "os_pretty=${PRETTY_NAME:-}" )
fi
if [ "$(uname -s)" = Darwin ]; then
  echo "os_id=macos"; echo "os_version=$(sw_vers -productVersion 2>/dev/null)"
  echo "uptime_s=$(( $(date +%s) - $(sysctl -n kern.boottime 2>/dev/null | sed -E 's/^\{ sec = ([0-9]+),.*/\1/') ))"
else
  echo "uptime_s=$(cut -d. -f1 /proc/uptime 2>/dev/null)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
fi
echo "now=$(date +%s)"
for m in apt-get nixos-version pacman softwareupdate fwupdmgr nvidia-smi docker dkms; do
  command -v "$m" >/dev/null 2>&1 && echo "has_$m=1"
done
[ -e /etc/NIXOS ] && echo "nixos=1"
[ -r /sys/class/dmi/id/product_name ] && echo "dmi_vendor=$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null)" && echo "dmi_product=$(cat /sys/class/dmi/id/product_name 2>/dev/null)"
[ -r /etc/dgx-release ] && grep -E '^(DGX_NAME|DGX_OTA_VERSION|DGX_SWBUILD_VERSION)=' /etc/dgx-release | tr -d '"'
[ -e /proc/sys/crypto/fips_enabled ] && echo "fips=$(cat /proc/sys/crypto/fips_enabled)"
if [ -d /boot ]; then
  latest=$(ls /boot/vmlinuz-* 2>/dev/null | sed 's|/boot/vmlinuz-||' | sort -V | tail -1)
  [ -n "$latest" ] && echo "kernel_latest_installed=$latest"
fi

if command -v apt-get >/dev/null 2>&1; then
  sec apt_upgradable
  T 60 apt list --upgradable 2>/dev/null | grep -v '^Listing'
  sec apt_errors
  T 60 apt-cache policy 2>&1 >/dev/null | grep -E '^(E|W):' | head -5
  sec apt_holds
  apt-mark showhold 2>/dev/null
  sec apt_meta
  for f in /var/lib/apt/periodic/update-success-stamp /var/lib/apt/lists/partial /var/cache/apt/pkgcache.bin; do
    [ -e "$f" ] && echo "lists_mtime=$(stat -c %Y "$f")" && break
  done
  [ -e /var/run/reboot-required ] && echo "reboot_required=1"
  [ -r /var/run/reboot-required.pkgs ] && echo "reboot_required_pkgs=$(sort -u /var/run/reboot-required.pkgs | tr '\n' ' ')"
  dpkg-query -W -f='${Status}' unattended-upgrades 2>/dev/null | grep -q 'install ok installed' && echo "uu_installed=1"
  grep -hE 'APT::Periodic::Unattended-Upgrade' /etc/apt/apt.conf.d/* 2>/dev/null | grep -q '"1"' && echo "uu_enabled=1"
  if [ -r /etc/apt/apt.conf.d/50unattended-upgrades ]; then
    bl=$(sed -n '/Package-Blacklist/,/};/p' /etc/apt/apt.conf.d/50unattended-upgrades | grep -vE '^\s*//' | grep -oE '"[^"]+"' | tr '\n' ' ')
    echo "uu_blacklist=$bl"
  fi
  sec swap_parity
  echo "swapimg_bytes=$(stat -c %s /swap.img 2>/dev/null || echo missing)"
  echo "swap_total_kb=$(awk '/SwapTotal/{print $2}' /proc/meminfo)"
  echo "earlyoom=$(dpkg -l earlyoom 2>/dev/null | grep -c '^ii')"
  echo "fstab_swap=$(grep -c '^/swap.img' /etc/fstab 2>/dev/null)"
fi

if [ -e /etc/NIXOS ]; then
  sec nixos
  echo "version=$(nixos-version 2>/dev/null)"
  echo "running=$(readlink -f /run/current-system 2>/dev/null)"
  echo "booted=$(readlink -f /run/booted-system 2>/dev/null)"
  echo "staged=$(readlink -f /nix/var/nix/profiles/system 2>/dev/null)"
  echo "booted_kernel=$(readlink -f /run/booted-system/kernel 2>/dev/null)"
  echo "staged_kernel=$(readlink -f /nix/var/nix/profiles/system/kernel 2>/dev/null)"
  echo "staged_mtime=$(stat -c %Y /nix/var/nix/profiles/system 2>/dev/null)"
  systemctl show nixos-upgrade.service -p Result -p ExecMainStatus -p ExecMainExitTimestamp 2>/dev/null
  echo "autoupgrade_timer=$(systemctl is-enabled nixos-upgrade.timer 2>/dev/null)"
fi

if command -v pacman >/dev/null 2>&1; then
  sec pacman
  if command -v checkupdates >/dev/null 2>&1; then
    echo "method=checkupdates"; T 90 checkupdates 2>/dev/null
  else
    echo "method=pacman-Qu-stale-db"; pacman -Qu 2>/dev/null
  fi
fi

if command -v fwupdmgr >/dev/null 2>&1; then
  sec fwupd_updates
  T 30 fwupdmgr get-updates --json 2>/dev/null
  sec fwupd_devices
  T 30 fwupdmgr get-devices --json 2>/dev/null
  sec fwupd_meta
  m=$(ls -t /var/lib/fwupd/metadata/*/firmware.xml* /var/lib/fwupd/metadata/*/metadata.xml* 2>/dev/null | grep -v jcat | head -1)
  [ -n "$m" ] && echo "metadata_mtime=$(stat -c %Y "$m" 2>/dev/null)"
  echo "refresh_timer=$(systemctl is-enabled fwupd-refresh.timer 2>/dev/null)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  sec gpu
  T 20 nvidia-smi --query-gpu=index,name,driver_version,vbios_version,utilization.gpu,memory.used --format=csv,noheader 2>&1
  sec gpu_util_samples
  for i in 1 2 3; do
    T 10 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1
    [ $i -lt 3 ] && sleep 2
  done
  sec gpu_procs
  T 20 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
fi
if command -v dkms >/dev/null 2>&1; then
  sec dkms
  T 20 dkms status 2>&1
fi

if command -v docker >/dev/null 2>&1; then
  sec docker
  out=$(T 20 docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}' 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "ERROR|$(echo "$out" | head -1)"; else echo "$out"; fi
fi

if command -v curl >/dev/null 2>&1; then
  sec ollama
  T 5 curl -s -m 3 http://127.0.0.1:11434/api/ps 2>/dev/null
fi

sec holds
# The maintenance hold convention (docs/fleet-maintenance.md): any of these files present
# blocks automated maintenance on this host. Contents are printed so the report can say who.
for f in "$HOME/.fleet-maint-hold" /etc/fleet-maint/hold /run/fleet-maint.hold; do
  [ -e "$f" ] && echo "$f|$(head -c 300 "$f" 2>/dev/null | tr '\n' ' ')"
done
for f in /tmp/fleet-maint-hold.d/*; do
  [ -e "$f" ] && echo "$f|$(head -c 300 "$f" 2>/dev/null | tr '\n' ' ')"
done

if command -v systemctl >/dev/null 2>&1; then
  sec failed_units
  systemctl --failed --no-legend --plain 2>/dev/null | awk '{print $1}'
  sec running_services
  systemctl list-units --type=service --state=running --no-legend --plain 2>/dev/null | awk '{print $1}'
fi

if command -v ip >/dev/null 2>&1; then
  sec nics
  ip -o -4 addr show 2>/dev/null | awk '{print $2, $4}'
  sec links
  ip -o link show 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="mtu") m=$(i+1); st="?"; for(i=1;i<=NF;i++) if($i=="state") st=$(i+1); sub(":","",$2); sub("@.*","",$2); print $2, m, st}'
fi
sec end
echo ok
