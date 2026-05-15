#!/usr/bin/env bash
# scripts/start_iperf_client.sh
set -euo pipefail

SERVER_IP="${1:-10.0.0.2}"
DURATION="${2:-25}"
OUTFILE="${3:-results/phase3_iperf.txt}"
STARTFILE="${4:-results/phase3_iperf_start.txt}"

mkdir -p "$(dirname "$OUTFILE")"
mkdir -p "$(dirname "$STARTFILE")"

IPERF_START_UNIX="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"

echo "$IPERF_START_UNIX" > "$STARTFILE"
echo "IPERF_START_UNIX=$IPERF_START_UNIX (saved to $STARTFILE)"

# iperf output contains intervals like 3.00-4.00 sec; we will convert later using IPERF_START_UNIX
iperf3 -c "$SERVER_IP" -t "$DURATION" -i 1 | tee "$OUTFILE"
