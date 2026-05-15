#!/usr/bin/env python3
"""
scripts/build_predictive_fault_dataset.py

Phase 6 — Dataset Engineering and Predictive Labelling

Purpose:
    Build a single ML-ready dataset from multiple telemetry experiment runs.

Input:
    A manifest CSV describing each run, with columns:
        run_id,scenario_type,telemetry_file,event_time_file

    Example:
        run_id,scenario_type,telemetry_file,event_time_file
        hard_run1,hard,data/raw/telemetry_hard_run1.csv,results/hard_run1_event.txt
        hard_run2,hard,data/raw/telemetry_hard_run2.csv,results/hard_run2_event.txt
        cong_run1,congestion,data/raw/telemetry_cong_run1.csv,results/cong_run1_event.txt
        normal_run1,normal,data/raw/telemetry_normal_run1.csv,

Output:
    A processed dataset CSV containing:
        - original telemetry columns
        - engineered rolling/statistical features
        - predictive binary label
        - run_id
        - scenario_type

Predictive label rule:
    label = 1 if an event will occur within the next T seconds
    label = 0 otherwise

    For row timestamp t and event time E:
        label = 1 if (E - T) <= t < E
"""

import argparse
import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


REQUIRED_TELEMETRY_COLUMNS = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build predictive ML dataset from multiple telemetry runs."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV file listing runs: run_id,scenario_type,telemetry_file,event_time_file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset CSV path, e.g. data/processed/ml_dataset.csv"
    )
    parser.add_argument(
        "--prediction-window",
        type=float,
        default=10.0,
        help="Seconds before event to label as positive (default: 10)"
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window size in samples for feature engineering (default: 5)"
    )
    parser.add_argument(
        "--min-positive-fraction-warning",
        type=float,
        default=0.005,
        help="Warn if positive ratio falls below this value (default: 0.005 = 0.5%%)"
    )
    return parser.parse_args()


def read_event_time(path: Optional[str]) -> Optional[float]:
    """
    Read a single UNIX timestamp from a file.
    Returns None if path is empty / missing / NaN-like.
    """
    if path is None:
        return None

    path = str(path).strip()
    if not path or path.lower() == "nan":
        return None

    with open(path, "r", encoding="utf-8") as f:
        return float(f.read().strip())


def validate_manifest(df: pd.DataFrame) -> None:
    required = ["run_id", "scenario_type", "telemetry_file", "event_time_file"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")


def validate_telemetry_columns(df: pd.DataFrame, telemetry_file: str) -> None:
    missing = [c for c in REQUIRED_TELEMETRY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Telemetry file '{telemetry_file}' is missing required columns: {missing}"
        )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Avoid divide-by-zero and inf values.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_engineered_features(group: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """
    Add predictive features per (run_id, dpid, port) stream.
    """
    g = group.sort_values("timestamp").copy()

    # Current combined throughput/utilisation proxy
    g["bytes_rate_total"] = g["rx_bytes_rate"] + g["tx_bytes_rate"]
    g["packets_rate_total"] = g["rx_packets_rate"] + g["tx_packets_rate"]
    g["drop_rate_total"] = g["rx_drop_rate"] + g["tx_drop_rate"]
    g["error_rate_total"] = g["rx_error_rate"] + g["tx_error_rate"]

    # Ratios
    g["drop_to_packet_ratio"] = safe_divide(g["drop_rate_total"], g["packets_rate_total"])
    g["error_to_packet_ratio"] = safe_divide(g["error_rate_total"], g["packets_rate_total"])

    # Rolling means
    g["bytes_rate_total_mean"] = g["bytes_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).mean()
    g["packets_rate_total_mean"] = g["packets_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).mean()
    g["drop_rate_total_mean"] = g["drop_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).mean()
    g["error_rate_total_mean"] = g["error_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).mean()

    # Rolling standard deviation
    g["bytes_rate_total_std"] = g["bytes_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).std().fillna(0.0)
    g["packets_rate_total_std"] = g["packets_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).std().fillna(0.0)
    g["drop_rate_total_std"] = g["drop_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).std().fillna(0.0)

    # Short-term slopes/trends
    g["bytes_rate_total_diff"] = g["bytes_rate_total"].diff().fillna(0.0)
    g["packets_rate_total_diff"] = g["packets_rate_total"].diff().fillna(0.0)
    g["drop_rate_total_diff"] = g["drop_rate_total"].diff().fillna(0.0)
    g["error_rate_total_diff"] = g["error_rate_total"].diff().fillna(0.0)

    # Rolling max/min
    g["bytes_rate_total_max"] = g["bytes_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).max()
    g["bytes_rate_total_min"] = g["bytes_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).min()
    g["drop_rate_total_max"] = g["drop_rate_total"].rolling(
        window=rolling_window, min_periods=1
    ).max()

    # Normalised instability indicators
    g["bytes_instability"] = safe_divide(
        g["bytes_rate_total_std"], g["bytes_rate_total_mean"] + 1e-9
    )
    g["drop_instability"] = safe_divide(
        g["drop_rate_total_std"], g["drop_rate_total_mean"] + 1e-9
    )

    # Lag features
    g["bytes_rate_total_lag1"] = g["bytes_rate_total"].shift(1).fillna(0.0)
    g["drop_rate_total_lag1"] = g["drop_rate_total"].shift(1).fillna(0.0)
    g["error_rate_total_lag1"] = g["error_rate_total"].shift(1).fillna(0.0)

    return g


def apply_predictive_label(
    df: pd.DataFrame,
    event_time: Optional[float],
    prediction_window: float,
    scenario_type: str
) -> pd.DataFrame:
    """
    Apply predictive labels.
    For normal runs or runs without event_time, label all rows as 0.
    """
    out = df.copy()

    if event_time is None or str(scenario_type).lower() == "normal":
        out["event_time"] = np.nan
        out["time_to_event"] = np.nan
        out["label"] = 0
        return out

    out["event_time"] = event_time
    out["time_to_event"] = event_time - out["timestamp"]

    out["label"] = (
        (out["timestamp"] >= (event_time - prediction_window)) &
        (out["timestamp"] < event_time)
    ).astype(int)

    return out


def process_single_run(
    run_id: str,
    scenario_type: str,
    telemetry_file: str,
    event_time_file: Optional[str],
    prediction_window: float,
    rolling_window: int
) -> pd.DataFrame:
    """
    Load one telemetry file, engineer features, and label it.
    """
    if not os.path.exists(telemetry_file):
        raise FileNotFoundError(f"Telemetry file not found: {telemetry_file}")

    telemetry_df = pd.read_csv(telemetry_file)
    validate_telemetry_columns(telemetry_df, telemetry_file)

    telemetry_df = telemetry_df.copy()
    telemetry_df["run_id"] = run_id
    telemetry_df["scenario_type"] = scenario_type

    # Ensure numeric types
    numeric_cols = REQUIRED_TELEMETRY_COLUMNS
    for col in numeric_cols:
        telemetry_df[col] = pd.to_numeric(telemetry_df[col], errors="coerce")

    telemetry_df = telemetry_df.dropna(subset=["timestamp", "dpid", "port"])
    telemetry_df = telemetry_df.sort_values(["dpid", "port", "timestamp"]).reset_index(drop=True)

    # Feature engineering per switch-port stream
    processed_parts: List[pd.DataFrame] = []
    grouped = telemetry_df.groupby(["run_id", "dpid", "port"], group_keys=False)

    for _, group in grouped:
        feat_group = add_engineered_features(group, rolling_window=rolling_window)
        processed_parts.append(feat_group)

    processed_df = pd.concat(processed_parts, ignore_index=True)

    # Apply predictive label
    event_time = read_event_time(event_time_file)
    processed_df = apply_predictive_label(
        processed_df,
        event_time=event_time,
        prediction_window=prediction_window,
        scenario_type=scenario_type
    )

    return processed_df


def print_dataset_summary(df: pd.DataFrame, min_positive_fraction_warning: float) -> None:
    total_rows = len(df)
    positive_rows = int(df["label"].sum())
    negative_rows = total_rows - positive_rows
    positive_fraction = (positive_rows / total_rows) if total_rows > 0 else 0.0

    print("=== Phase 6 Predictive Dataset Summary ===")
    print(f"Rows:                 {total_rows}")
    print(f"Positive samples:     {positive_rows}")
    print(f"Negative samples:     {negative_rows}")
    print(f"Positive fraction:    {positive_fraction:.6f}")
    print()

    print("Rows by run_id:")
    print(df["run_id"].value_counts().sort_index())
    print()

    print("Rows by scenario_type:")
    print(df["scenario_type"].value_counts().sort_index())
    print()

    print("Positive labels by run_id:")
    print(df.groupby("run_id")["label"].sum().sort_index())
    print()

    print("Positive labels by scenario_type:")
    print(df.groupby("scenario_type")["label"].sum().sort_index())
    print()

    if positive_fraction < min_positive_fraction_warning:
        print(
            f"WARNING: Positive fraction ({positive_fraction:.6f}) is very low. "
            "Class imbalance may be severe."
        )


def main():
    args = parse_args()

    manifest_df = pd.read_csv(args.manifest)
    validate_manifest(manifest_df)

    all_runs: List[pd.DataFrame] = []

    for _, row in manifest_df.iterrows():
        run_id = str(row["run_id"]).strip()
        scenario_type = str(row["scenario_type"]).strip()
        telemetry_file = str(row["telemetry_file"]).strip()
        event_time_file = row["event_time_file"]

        print(f"Processing run_id={run_id}, scenario_type={scenario_type}")
        processed_run_df = process_single_run(
            run_id=run_id,
            scenario_type=scenario_type,
            telemetry_file=telemetry_file,
            event_time_file=event_time_file,
            prediction_window=args.prediction_window,
            rolling_window=args.rolling_window
        )
        all_runs.append(processed_run_df)

    if not all_runs:
        raise ValueError("No runs were processed. Check your manifest file.")

    final_df = pd.concat(all_runs, ignore_index=True)

    # Sort final dataset cleanly
    final_df = final_df.sort_values(
        by=["run_id", "dpid", "port", "timestamp"]
    ).reset_index(drop=True)

    # Fill any remaining NaNs from rolling or lag operations
    final_df = final_df.fillna(0.0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_df.to_csv(args.output, index=False)

    print_dataset_summary(
        final_df,
        min_positive_fraction_warning=args.min_positive_fraction_warning
    )

    print(f"Dataset written to: {args.output}")


if __name__ == "__main__":
    main()
