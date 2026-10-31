"""Gráficos salvos como artefatos no MLflow."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def confusion_plot(cm, labels, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdadeiro")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def reliability_plot(y, probs, n_bins, title):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    acc = [correct[bins == b].mean() if (bins == b).any() else np.nan for b in range(n_bins)]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(edges[:-1], acc, width=1 / n_bins, align="edge", edgecolor="black", alpha=0.7, label="acurácia no bin")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="calibração perfeita")
    ax.set_xlabel("confiança (prob. máxima)")
    ax.set_ylabel("acurácia")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def at_k_plot(results: dict, ks, title):
    """results: {"baseline": metrics, "finetuned": metrics}"""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, name in zip(axes, ("precision", "recall", "ndcg")):
        for label, m in results.items():
            ax.plot(ks, [m[f"{name}_at_{k}"] for k in ks], marker="o", label=label)
        ax.set_xticks(ks)
        ax.set_xlabel("k")
        ax.set_title(f"{name}@k")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
