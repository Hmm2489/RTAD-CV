"""Train the lightweight attention classifier on extracted features."""

import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rtad.features import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/features.csv")
    parser.add_argument("--out", default="models/classifier.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y, groups = df[FEATURE_NAMES], df["label"], df["session"]

    # Split by session so frames from the same clip don't leak between train and test
    train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(X, y, groups))

    # TODO: try windowed/aggregated features and a small GBM (e.g. HistGradientBoosting)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    print(classification_report(y.iloc[test_idx], model.predict(X.iloc[test_idx])))
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
