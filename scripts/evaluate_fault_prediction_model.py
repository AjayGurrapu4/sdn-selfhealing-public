#!/usr/bin/env python3
"""
scripts/evaluate_fault_prediction_model.py

Load a saved model bundle and evaluate it on a labelled dataset.
Useful for repeatable reporting and validation.
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved fault prediction model on a labelled dataset."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to saved model bundle .pkl",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to labelled dataset CSV",
    )
    parser.add_argument(
        "--output-json",
        default="results/fault_prediction_evaluation.json",
        help="Path to save evaluation JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    bundle = joblib.load(args.model)
    df = pd.read_csv(args.dataset)

    label_col = bundle["label_column"]
    feature_cols = bundle["feature_columns"]
    threshold = float(bundle["threshold"])

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing expected feature columns: {missing_features}")

    x = df[feature_cols].copy()
    y = df[label_col].astype(int).to_numpy()

    pipeline = bundle["pipeline"]
    y_prob = pipeline.predict_proba(x)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    result = {
        "model_name": bundle["model_name"],
        "threshold": threshold,
        "dataset": args.dataset,
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== Saved Model Evaluation ===")
    print(f"Model:       {result['model_name']}")
    print(f"Dataset:     {result['dataset']}")
    print(f"Rows:        {result['rows']}")
    print(f"Threshold:   {result['threshold']:.2f}")
    print(f"Accuracy:    {result['accuracy']:.4f}")
    print(f"Precision:   {result['precision']:.4f}")
    print(f"Recall:      {result['recall']:.4f}")
    print(f"F1-score:    {result['f1_score']:.4f}")
    print(f"ROC-AUC:     {result['roc_auc']:.4f}")
    print(f"Confusion:   TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Saved JSON:  {args.output_json}")


if __name__ == "__main__":
    main()
