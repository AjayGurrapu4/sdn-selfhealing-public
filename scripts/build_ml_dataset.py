#!/usr/bin/env python3
"""
scripts/build_ml_dataset.py

Project:
AI-Driven Self-Healing Software-Defined Networking (SDN) for Predictive Fault Management

Purpose:
Build an ML-ready predictive dataset from raw telemetry collected by the Ryu controller.

What this script does:
1. Reads raw telemetry CSV from data/raw/telemetry.csv
2. Reads one or more recorded fault timestamps
3. Computes engineered features per (dpid, port)
4. Applies predictive labels:
      label = 1 if fault occurs within the next prediction window
      label = 0 otherwise
5. Saves processed dataset to data/processed/ml_dataset.csv

Expected raw telemetry columns:
- timestamp
- dpid
- port
- rx_bytes_rate
- tx_bytes_rate
- rx_packets_rate
- tx_packets_rate
- rx_drop_rate
- tx_drop_rate
- rx_error_rate
- tx_error_rate
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "dpid",
    "port",
    "rx_bytes_rate",
    "tx_bytes_rate",
    "rx_packets_rate",
    "tx_packets_rate",
    "rx_drop_rate",
    "tx_drop_rate",
    "rx_error_rate",
    "tx_error_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a predictive ML dataset from SDN telemetry."
    )
    parser.add_argument(
        "--telemetry",
        required=True,
        help="Path to raw telemetry CSV, e.g. data/raw/telemetry.csv",
    )
    parser.add_argument(
        "--fault_time",
        required=True,
        nargs="+",
        help=(
            "One or more files containing UNIX fault timestamps. "
            "Example: results/phase5_fault_begin.txt"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output processed dataset CSV, e.g. data/processed/ml_dataset.csv",
    )
    parser.add_argument(
        "--prediction-window-sec",
        type=float,
        default=10.0,
        help="Label rows as positive if a fault will occur within this many seconds.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window size used for engineered features.",
    )
    parser.add_argument(
        "--run-id",
        default="run_1",
        help="Optional run identifier to store in the dataset.",
    )
    return parser.parse_args()


def read_fault_times(paths: List[str]) -> List[float]:
    fault_times: List[float] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise ValueError(f"Fault time file is empty: {path}")

        try:
            fault_times.append(float(content))
        except ValueError as exc:
            raise ValueError(f"Invalid fault timestamp in {path}: {content}") from exc

    return sorted(fault_times)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Telemetry CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def safe_slope(values: pd.Series) -> float:
    """
    Compute simple linear slope across a short rolling window.
    Used for byte-rate trend estimation.
    """
    arr = values.to_numpy(dtype=float)
    if len(arr) < 2:
        return 0.0

    x = np.arange(len(arr), dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()

    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0

    numer = np.sum((x - x_mean) * (arr - y_mean))
    return float(numer / denom)


def add_engineered_features(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """
    Create predictive features per switch-port telemetry stream.
    Grouping by (dpid, port) preserves per-port behaviour over time.
    """
    df = df.copy()

    # Base composite features
    df["utilisation_now"] = df["rx_bytes_rate"] + df["tx_bytes_rate"]
    df["drop_rate_now"] = df["rx_drop_rate"] + df["tx_drop_rate"]
    df["error_rate_now"] = df["rx_error_rate"] + df["tx_error_rate"]

    group_cols = ["dpid", "port"]

    # Rolling mean / std / slope for utilisation and drop behaviour
    df["utilisation_mean_5"] = (
        df.groupby(group_cols)["utilisation_now"]
        .transform(lambda s: s.rolling(window=rolling_window, min_periods=1).mean())
    )

    df["utilisation_std_5"] = (
        df.groupby(group_cols)["utilisation_now"]
        .transform(lambda s: s.rolling(window=rolling_window, min_periods=1).std())
        .fillna(0.0)
    )

    df["drop_rate_mean_5"] = (
        df.groupby(group_cols)["drop_rate_now"]
        .transform(lambda s: s.rolling(window=rolling_window, min_periods=1).mean())
    )

    df["byte_rate_slope_5"] = (
        df.groupby(group_cols)["utilisation_now"]
        .transform(
            lambda s: s.rolling(window=rolling_window, min_periods=2).apply(
                safe_slope, raw=False
            )
        )
        .fillna(0.0)
    )

    # Optional supporting stability features
    df["utilisation_max_5"] = (
        df.groupby(group_cols)["utilisation_now"]
        .transform(lambda s: s.rolling(window=rolling_window, min_periods=1).max())
    )

    df["drop_rate_max_5"] = (
        df.groupby(group_cols)["drop_rate_now"]
        .transform(lambda s: s.rolling(window=rolling_window, min_periods=1).max())
    )

    df["packets_total_rate"] = df["rx_packets_rate"] + df["tx_packets_rate"]

    return df


def apply_predictive_labels(
    df: pd.DataFrame,
    fault_times: List[float],
    prediction_window_sec: float,
) -> pd.DataFrame:
    """
    Predictive label strategy:
    label = 1 if any fault_time satisfies:
        fault_time - prediction_window_sec <= timestamp < fault_time
    else 0
    """
    df = df.copy()
    timestamps = df["timestamp"].to_numpy(dtype=float)

    label = np.zeros(len(df), dtype=int)

    for fault_time in fault_times:
        in_window = (
            (timestamps >= (fault_time - prediction_window_sec))
            & (timestamps < fault_time)
        )
        label[in_window] = 1

    df["label"] = label
    return df


def main() -> None:
    args = parse_args()

    telemetry_path = Path(args.telemetry)
    output_path = Path(args.output)

    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

    df = pd.read_csv(telemetry_path)
    validate_columns(df)

    # Sort to ensure rolling features work correctly per port stream
    df = df.sort_values(by=["dpid", "port", "timestamp"]).reset_index(drop=True)

    # Force numeric types where needed
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove malformed rows
    before_drop = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    dropped_rows = before_drop - len(df)

    fault_times = read_fault_times(args.fault_time)

    df = add_engineered_features(df=df, rolling_window=args.rolling_window)
    df = apply_predictive_labels(
        df=df,
        fault_times=fault_times,
        prediction_window_sec=args.prediction_window_sec,
    )

    # Add useful metadata
    df["run_id"] = args.run_id
    df["fault_window_sec"] = args.prediction_window_sec

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    total_rows = len(df)
    fault_rows = int(df["label"].sum())
    normal_rows = int((df["label"] == 0).sum())

    print("=== ML Dataset Generation Complete ===")
    print(f"Input telemetry:           {telemetry_path}")
    print(f"Fault timestamp files:     {args.fault_time}")
    print(f"Output dataset:            {output_path}")
    print(f"Run ID:                    {args.run_id}")
    print(f"Prediction window (sec):   {args.prediction_window_sec}")
    print(f"Rolling window:            {args.rolling_window}")
    print(f"Rows dropped as invalid:   {dropped_rows}")
    print(f"Total rows:                {total_rows}")
    print(f"Fault-labelled rows:       {fault_rows}")
    print(f"Normal rows:               {normal_rows}")
    print(f"Fault times used:          {fault_times}")
    print()
    print("Columns in output dataset:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
