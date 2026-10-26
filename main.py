#!/usr/bin/env python3
"""Download MNIST and evaluate PCA + logistic regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/openml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 100 or args.components < 1:
        raise ValueError("Use at least 100 samples and one PCA component.")
    dataset = fetch_openml(
        data_id=554,
        as_frame=False,
        data_home=str(args.data_dir),
        parser="liac-arff",
    )
    features = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int64)
    sample_count = min(args.samples, labels.size)
    sample_indexes, _ = train_test_split(
        np.arange(labels.size),
        train_size=sample_count,
        random_state=args.seed,
        stratify=labels,
    )
    features, labels = features[sample_indexes], labels[sample_indexes]

    train_validation, test = train_test_split(
        np.arange(labels.size), test_size=0.20, random_state=args.seed, stratify=labels
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.20,
        random_state=args.seed,
        stratify=labels[train_validation],
    )
    components = min(args.components, features.shape[1], len(train) - 1)
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=components, svd_solver="randomized", random_state=args.seed)),
            ("classifier", LogisticRegression(max_iter=1_000, solver="lbfgs")),
        ]
    )
    model.fit(features[train], labels[train])

    def score(indexes: np.ndarray) -> dict[str, float | int]:
        prediction = model.predict(features[indexes])
        return {
            "samples": int(indexes.size),
            "accuracy": float(accuracy_score(labels[indexes], prediction)),
            "macro_f1": float(f1_score(labels[indexes], prediction, average="macro")),
        }

    explained_variance = float(np.sum(model.named_steps["pca"].explained_variance_ratio_))
    metrics = {
        "dataset": "OpenML MNIST (data_id=554)",
        "seed": args.seed,
        "features": int(features.shape[1]),
        "pca_components": components,
        "explained_variance_ratio_sum": explained_variance,
        "split_sizes": {"train": int(train.size), "validation": int(validation.size), "test": int(test.size)},
        "train": score(train),
        "validation": score(validation),
        "test": score(test),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "pca_model.joblib")
    plot_indexes = test[: min(2_000, test.size)]
    projection = model.named_steps["pca"].transform(
        model.named_steps["scale"].transform(features[plot_indexes])
    )
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    scatter = axis.scatter(projection[:, 0], projection[:, 1], c=labels[plot_indexes], s=8, cmap="tab10", alpha=0.7)
    axis.set(title="MNIST test samples projected onto the first two PCA components", xlabel="PC1", ylabel="PC2")
    figure.colorbar(scatter, ax=axis, ticks=range(10), label="Digit class")
    figure.savefig(args.output_dir / "pca_projection.png", dpi=160)
    plt.close(figure)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"MNIST rows / PCA dimensions: {sample_count} / {components}")
    print(f"Explained variance retained: {explained_variance:.3f}")
    print(f"Held-out test accuracy / macro-F1: {metrics['test']['accuracy']:.3f} / {metrics['test']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
