#!/usr/bin/env bash
# scripts/trigger_link_failure.sh

set -euo pipefail

SW1="${1:-}"
SW2="${2:-}"
OUTFILE="${3:-}"

if [[ -z "$SW1" || -z "$SW2" || -z "$OUTFILE" ]]; then
  echo "Usage: $0 <sw1> <sw2> <fault_log_file>"
  echo "Example: $0 s1 s2 \$HOME/sdn-selfhealing/results/phase3_fault_begin.txt"
  exit 1
fi

FAULT_TIME="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"

mkdir -p "$(dirname "$OUTFILE")"
echo "$FAULT_TIME" > "$OUTFILE"

echo "FAULT_BEGIN_UNIX_TIME=$FAULT_TIME (saved to $OUTFILE)"
echo
echo "NOW, in your mininet> CLI, run this EXACT command:"
echo "  link ${SW1} ${SW2} down"
echo
echo "To restore later:"
echo "  link ${SW1} ${SW2} up"
