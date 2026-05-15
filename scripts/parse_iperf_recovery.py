# scripts/parse_iperf_recovery.py
import argparse
import re
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iperf", required=True, help="iperf3 output file")
    p.add_argument("--iperf_start", required=True, help="file containing iperf client start UNIX time")
    p.add_argument("--fault", required=True, help="file containing fault begin UNIX time")
    p.add_argument("--threshold_mbps", type=float, default=0.5, help="Recovery threshold in Mbits/sec")
    p.add_argument("--consecutive", type=int, default=1, help="Consecutive intervals >= threshold")
    return p.parse_args()


def read_float(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        return float(f.read().strip())


def parse_iperf_intervals(text: str):
    """
    Parse lines like:
    [  5]   3.00-4.00   sec  11.2 MBytes  94.1 Mbits/sec
    [  5]   3.01-4.00   sec  50.9 KBytes   418 Kbits/sec
    [  5]   4.00-5.00   sec  0.00 Bytes  0.00 bits/sec

    Returns list: (start_sec, end_sec, mbps)
    """
    intervals = []

    rx = re.compile(
        r"\[\s*\d+\]\s+"
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s+sec\s+"
        r".+?\s+"
        r"(\d+(?:\.\d+)?)\s+"
        r"([KMG]?)bits/sec"
    )

    for line in text.splitlines():
        m = rx.search(line)
        if not m:
            continue

        start = float(m.group(1))
        end = float(m.group(2))
        val = float(m.group(3))
        unit = m.group(4)

        # Convert everything to Mbps
        if unit == "G":
            val *= 1000.0
        elif unit == "K":
            val /= 1000.0
        elif unit == "":
            val /= 1_000_000.0

        intervals.append((start, end, val))

    return intervals


def main():
    args = parse_args()

    fault_unix = read_float(args.fault)
    iperf_start_unix = read_float(args.iperf_start)

    with open(args.iperf, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    intervals = parse_iperf_intervals(text)
    if not intervals:
        print("ERROR: Could not parse iperf intervals.", file=sys.stderr)
        sys.exit(1)

    streak = 0
    recovered_unix = None
    recovered_rel_end = None

    for (start_sec, end_sec, mbps) in intervals:
        interval_end_unix = iperf_start_unix + end_sec

        if interval_end_unix < fault_unix:
            continue

        if mbps >= args.threshold_mbps:
            streak += 1
        else:
            streak = 0

        if streak >= args.consecutive:
            recovered_unix = interval_end_unix
            recovered_rel_end = end_sec
            break

    print("=== Phase 3 Recovery Estimation (UNIX-aligned) ===")
    print(f"Fault begin time (UNIX):      {fault_unix:.6f}")
    print(f"iperf start time (UNIX):      {iperf_start_unix:.6f}")
    print(f"Threshold:                    {args.threshold_mbps:.2f} Mbits/sec")
    print(f"Consecutive intervals:        {args.consecutive}")

    if recovered_unix is None:
        print("Recovery not detected with given threshold/consecutive.")
        sys.exit(0)

    mttr = recovered_unix - fault_unix
    print(f"Throughput recovered (UNIX):  {recovered_unix:.6f}")
    print(f"Recovery interval end (sec):  {recovered_rel_end:.2f} after iperf start")
    print(f"Estimated MTTR (sec):         {mttr:.3f}")


if __name__ == "__main__":
    main()
