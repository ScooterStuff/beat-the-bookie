"""Evaluation helpers — classification reports and confusion-matrix plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def report_model(name: str, y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\n[{name}] accuracy={acc:.4f}  f1={f1:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"model": name, "accuracy": acc, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, model_name: str, label_classes=None, save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=label_classes if label_classes is not None else np.unique(y_true),
        yticklabels=label_classes if label_classes is not None else np.unique(y_true),
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def summary_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    return df.sort_values(by="f1", ascending=False).reset_index(drop=True)
