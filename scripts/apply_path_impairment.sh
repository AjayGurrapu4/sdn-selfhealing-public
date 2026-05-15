#!/usr/bin/env bash
# scripts/apply_path_impairment.sh

set -euo pipefail

IFACE="${1:-}"
MODE="${2:-}"
VALUE1="${3:-}"
VALUE2="${4:-}"
STARTFILE="${5:-}"

if [[ -z "$IFACE" || -z "$MODE" || -z "$VALUE1" || -z "$STARTFILE" ]]; then
  echo "Usage examples:"
  echo "  $0 s1-eth2 tbf 10mbit 50ms results/impairment_start.txt"
  echo "  $0 s1-eth2 netem delay 40ms results/impairment_start.txt"
  echo "  $0 s1-eth2 netem loss 5% results/impairment_start.txt"
  exit 1
fi

START_UNIX="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"

mkdir -p "$(dirname "$STARTFILE")"
echo "$START_UNIX" > "$STARTFILE"
echo "IMPAIRMENT_START_UNIX=$START_UNIX (saved to $STARTFILE)"

sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true

if [[ "$MODE" == "tbf" ]]; then
  RATE="$VALUE1"
  LATENCY="${VALUE2:-50ms}"
  sudo tc qdisc add dev "$IFACE" root tbf rate "$RATE" burst 32kbit latency "$LATENCY"
  echo "Applied TBF rate limit on $IFACE: rate=$RATE latency=$LATENCY"

elif [[ "$MODE" == "netem" ]]; then
  KIND="$VALUE1"
  METRIC="${VALUE2:-}"
  if [[ -z "$METRIC" ]]; then
    echo "For netem mode you must provide kind + metric"
    echo "Examples:"
    echo "  $0 s1-eth2 netem delay 40ms results/impairment_start.txt"
    echo "  $0 s1-eth2 netem loss 5% results/impairment_start.txt"
    exit 1
  fi

  if [[ "$KIND" == "delay" ]]; then
    sudo tc qdisc add dev "$IFACE" root netem delay "$METRIC"
    echo "Applied netem delay on $IFACE: $METRIC"
  elif [[ "$KIND" == "loss" ]]; then
    sudo tc qdisc add dev "$IFACE" root netem loss "$METRIC"
    echo "Applied netem loss on $IFACE: $METRIC"
  elif [[ "$KIND" == "jitter" ]]; then
    sudo tc qdisc add dev "$IFACE" root netem delay "$METRIC" 10ms distribution normal
    echo "Applied netem jitter-style delay on $IFACE: base=$METRIC variation=10ms"
  else
    echo "Unsupported netem kind: $KIND"
    exit 1
  fi
else
  echo "Unsupported mode: $MODE"
  exit 1
fi
