#!/usr/bin/env python3
"""
train_classifier.py
--------------------
Trains a Random Forest classifier on the flow-feature dataset produced by
extract_features.py, evaluates it, and saves the trained model + a
predict_traffic_type() helper for handoff to the FastAPI teammate.

Usage:
    python3 train_classifier.py --input dataset.csv --output model.joblib

Requires: scikit-learn, pandas, joblib
    pip install scikit-learn pandas joblib --break-system-packages
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "packet_count",
    "total_bytes",
    "avg_packet_size",
    "std_packet_size",
    "avg_inter_arrival",
    "std_inter_arrival",
    "flow_duration",
    "bitrate_bps",
]


def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest traffic-type classifier.")
    parser.add_argument("--input", default="dataset.csv", help="CSV produced by extract_features.py")
    parser.add_argument("--output", default="model.joblib", help="Where to save the trained model")
    parser.add_argument("--test-size", type=float, default=0.25,
                         help="Fraction of rows held out for testing (default: 0.25)")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees (default: 200)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows across {df['label'].nunique()} classes:")
    print(df["label"].value_counts().to_string())

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # Stratify keeps class proportions similar between train/test, but with
    # very small classes (e.g. only 4 icmp rows) this can fail outright --
    # fall back to a plain random split if stratifying isn't possible.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"\nStratified split failed ({e})")
        print("Falling back to a plain random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )

    print(f"\nTrain size: {len(X_train)}   Test size: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",  # compensates for icmp/voip/web having far fewer rows than video
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y.unique())
    print("=== Confusion Matrix ===")
    print("Labels (row=actual, col=predicted):", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    print("\n=== Feature Importances ===")
    importances = sorted(zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda x: -x[1])
    for name, importance in importances:
        print(f"  {name}: {importance:.3f}")

    joblib.dump({"model": clf, "feature_columns": FEATURE_COLUMNS}, args.output)
    print(f"\nSaved trained model to {args.output}")


def predict_traffic_type(model_path, packet_count, total_bytes, avg_packet_size,
                          std_packet_size, avg_inter_arrival, std_inter_arrival,
                          flow_duration, bitrate_bps):
    """Handoff helper for the FastAPI teammate: loads the saved model and
    predicts a traffic-type label from a single flow-window's features.

    Example:
        label = predict_traffic_type(
            "model.joblib",
            packet_count=12, total_bytes=1816, avg_packet_size=151.3,
            std_packet_size=29.8, avg_inter_arrival=0.094,
            std_inter_arrival=0.293, flow_duration=1.037, bitrate_bps=14012.5,
        )
    """
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    columns = bundle["feature_columns"]
    row = pd.DataFrame(
        [[packet_count, total_bytes, avg_packet_size, std_packet_size,
          avg_inter_arrival, std_inter_arrival, flow_duration, bitrate_bps]],
        columns=columns,
    )
    return clf.predict(row)[0]


if __name__ == "__main__":
    main()
