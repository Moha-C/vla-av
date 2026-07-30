#!/usr/bin/env bash
set -euo pipefail

SINCE="${1:-2026-07-01 15:20:00}"
UNTIL="${2:-2026-07-01 15:45:00}"

echo "[pc-shutdown-audit] window: $SINCE -> $UNTIL"
echo
echo "== last reboot/shutdown history =="
last -x shutdown reboot | head -n 40 || true
echo
echo "== journal boots =="
journalctl --list-boots --no-pager || true
echo

if ! sudo -n true 2>/dev/null; then
  cat <<EOF
[pc-shutdown-audit] sudo is required for full system logs.

Run this command from a normal terminal and enter your password:

  sudo bash $(realpath "$0") "$SINCE" "$UNTIL"

EOF
  exit 0
fi

echo "== previous boot around shutdown =="
sudo journalctl -b -1 --since "$SINCE" --until "$UNTIL" -o short-iso --no-pager || true
echo

echo "== previous boot kernel tail =="
sudo journalctl -k -b -1 -o short-iso --no-pager | tail -n 200 || true
echo

echo "== shutdown/suspend/power/thermal signals =="
sudo journalctl -b -1 \
  --grep 'shutdown|power|Power|suspend|Suspend|sleep|Sleep|thermal|Thermal|critical|Critical|battery|Battery|ACPI|logind|reboot|halt|hibernate|lid|Lid|watchdog|Watchdog|panic|Oops|NMI|temperature|overheat|oom|OOM|nvidia|NVRM' \
  -o short-iso --no-pager || true
echo

echo "== syslog/kern fallback =="
sudo grep -Ehi \
  'Jul  1 15:(2[0-9]|3[0-9])|shutdown|power|Power|suspend|Suspend|sleep|Sleep|thermal|Thermal|critical|Critical|battery|Battery|ACPI|logind|reboot|halt|hibernate|lid|Lid|watchdog|Watchdog|panic|Oops|NMI|temperature|overheat|oom|OOM|nvidia|NVRM' \
  /var/log/syslog /var/log/syslog.1 /var/log/kern.log /var/log/kern.log.1 2>/dev/null | tail -n 300 || true
