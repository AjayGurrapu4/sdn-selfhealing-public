#!/usr/bin/env python3
"""
scripts/train_fault_prediction_model.py

Train and select an ML model for predictive fault management in the
self-healing SDN project.

Models:
- Logistic Regression (baseline)
- Random Forest (stronger nonlinear model)

Outputs:
- Saved model bundle (.pkl)
- Metrics JSON
- Confusion matrix CSV
- ROC curve CSV
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and select a fault prediction model for the SDN self-healing project."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to processed labelled dataset CSV, e.g. data/processed/ml_dataset.csv",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Target column name. Default: label",
    )
    parser.add_argument(
        "--output-model",
        default="models/fault_prediction_model.pkl",
        help="Path to save final trained model bundle",
    )
    parser.add_argument(
        "--output-metrics",
        default="results/fault_prediction_metrics.json",
        help="Path to save evaluation metrics JSON",
    )
    parser.add_argument(
        "--output-confusion",
        default="results/fault_prediction_confusion_matrix.csv",
        help="Path to save confusion matrix CSV",
    )
    parser.add_argument(
        "--output-roc",
        default="results/fault_prediction_roc_curve.csv",
        help="Path to save ROC curve points CSV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split fraction. Default: 0.3",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--fault-threshold",
        type=float,
        default=0.50,
        help="Classification threshold on predicted probability. Default: 0.50",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_dataset(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def select_feature_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    excluded = {
        label_col,
        "timestamp",
        "run_id",
        "fault_type",
        "fault_time",
        "port",
        "dpid",
    }

    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No numeric feature columns found for training.")

    return feature_cols


def build_logistic_pipeline(feature_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def build_random_forest_pipeline(feature_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )
    return metrics


def save_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, path: str) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    ensure_parent_dir(path)
    df_cm.to_csv(path, index=True)


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: str) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    df_roc = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    ensure_parent_dir(path)
    df_roc.to_csv(path, index=False)


def train_and_evaluate(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold: float,
) -> Tuple[Pipeline, Dict[str, float], np.ndarray]:
    pipeline.fit(x_train, y_train)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_prob, threshold)
    metrics["model_name"] = name
    return pipeline, metrics, y_prob


def main():
    args = parse_args()

    df = load_dataset(args.dataset, args.label_col)
    feature_cols = select_feature_columns(df, args.label_col)

    x = df[feature_cols].copy()
    y = df[args.label_col].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    logistic_pipeline = build_logistic_pipeline(feature_cols)
    rf_pipeline = build_random_forest_pipeline(feature_cols)

    trained_logistic, logistic_metrics, logistic_prob = train_and_evaluate(
        name="LogisticRegression",
        pipeline=logistic_pipeline,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        threshold=args.fault_threshold,
    )

    trained_rf, rf_metrics, rf_prob = train_and_evaluate(
        name="RandomForest",
        pipeline=rf_pipeline,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        threshold=args.fault_threshold,
    )

    # Selection rule:
    # Prefer higher F1. If tied, prefer higher recall. If still tied, prefer higher ROC-AUC.
    candidates = [logistic_metrics, rf_metrics]
    best_metrics = sorted(
        candidates,
        key=lambda m: (m["f1_score"], m["recall"], m["roc_auc"]),
        reverse=True,
    )[0]

    if best_metrics["model_name"] == "LogisticRegression":
        best_pipeline = trained_logistic
        best_prob = logistic_prob
    else:
        best_pipeline = trained_rf
        best_prob = rf_prob

    model_bundle = {
        "model_name": best_metrics["model_name"],
        "pipeline": best_pipeline,
        "feature_columns": feature_cols,
        "threshold": args.fault_threshold,
        "label_column": args.label_col,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    ensure_parent_dir(args.output_model)
    ensure_parent_dir(args.output_metrics)
    ensure_parent_dir(args.output_confusion)
    ensure_parent_dir(args.output_roc)

    joblib.dump(model_bundle, args.output_model)

    full_metrics = {
        "dataset": args.dataset,
        "label_column": args.label_col,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "class_distribution": {
            "normal_0": int((y == 0).sum()),
            "fault_1": int((y == 1).sum()),
        },
        "test_size": args.test_size,
        "random_state": args.random_state,
        "models": {
            "LogisticRegression": logistic_metrics,
            "RandomForest": rf_metrics,
        },
        "selected_model": best_metrics["model_name"],
        "selected_metrics": best_metrics,
    }

    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    save_confusion_matrix(y_test.to_numpy(), best_prob, args.fault_threshold, args.output_confusion)
    save_roc_curve(y_test.to_numpy(), best_prob, args.output_roc)

    print("=== Fault Prediction Model Training ===")
    print(f"Dataset:               {args.dataset}")
    print(f"Rows:                  {len(df)}")
    print(f"Features used:         {len(feature_cols)}")
    print(f"Train rows:            {len(x_train)}")
    print(f"Test rows:             {len(x_test)}")
    print()
    print("Baseline Model: LogisticRegression")
    print(f"  Precision:           {logistic_metrics['precision']:.4f}")
    print(f"  Recall:              {logistic_metrics['recall']:.4f}")
    print(f"  F1-score:            {logistic_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:             {logistic_metrics['roc_auc']:.4f}")
    print()
    print("Improved Model: RandomForest")
    print(f"  Precision:           {rf_metrics['precision']:.4f}")
    print(f"  Recall:              {rf_metrics['recall']:.4f}")
    print(f"  F1-score:            {rf_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:             {rf_metrics['roc_auc']:.4f}")
    print()
    print(f"Selected model:        {best_metrics['model_name']}")
    print(f"Decision threshold:    {args.fault_threshold:.2f}")
    print(f"Saved model bundle:    {args.output_model}")
    print(f"Saved metrics JSON:    {args.output_metrics}")
    print(f"Saved confusion CSV:   {args.output_confusion}")
    print(f"Saved ROC CSV:         {args.output_roc}")


if __name__ == "__main__":
    main()
