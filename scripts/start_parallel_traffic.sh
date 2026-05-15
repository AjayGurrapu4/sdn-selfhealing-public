#!/usr/bin/env bash
# scripts/start_parallel_traffic.sh

set -euo pipefail

SERVER_IP="${1:-10.0.0.2}"
DURATION="${2:-30}"
PARALLEL_STREAMS="${3:-5}"
BANDWIDTH_PER_STREAM="${4:-0}"
OUTFILE="${5:-results/parallel_traffic.txt}"
STARTFILE="${6:-results/parallel_traffic_start.txt}"

mkdir -p "$(dirname "$OUTFILE")"
mkdir -p "$(dirname "$STARTFILE")"

START_UNIX="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"

echo "$START_UNIX" > "$STARTFILE"
echo "PARALLEL_TRAFFIC_START_UNIX=$START_UNIX (saved to $STARTFILE)"

if [[ "$BANDWIDTH_PER_STREAM" == "0" ]]; then
  iperf3 -c "$SERVER_IP" -t "$DURATION" -P "$PARALLEL_STREAMS" -i 1 | tee "$OUTFILE"
else
  iperf3 -c "$SERVER_IP" -t "$DURATION" -P "$PARALLEL_STREAMS" -b "$BANDWIDTH_PER_STREAM" -i 1 | tee "$OUTFILE"
fi
