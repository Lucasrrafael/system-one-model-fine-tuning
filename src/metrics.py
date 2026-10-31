"""Métricas para perguntas `score` (ordinais) do Laya.

Entrada comum a todas as funções:
  y      -> array (n,) com o nível verdadeiro, 0..K-1
  probs  -> array (n, K) com a distribuição prevista sobre os níveis

As referências de cada métrica estão no README (seção "Métricas").
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_level(probs: np.ndarray) -> np.ndarray:
    """Nível esperado E[y] = sum_i i * p_i (é o `score` que o Laya devolve)."""
    return probs @ np.arange(probs.shape[1])


def classification_metrics(y, probs, labels):
    pred = probs.argmax(1)
    k = probs.shape[1]
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "cohen_kappa": cohen_kappa_score(y, pred, labels=list(range(k))),
    }
    for avg in ("macro", "weighted"):
        out[f"precision_{avg}"] = precision_score(y, pred, labels=list(range(k)), average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y, pred, labels=list(range(k)), average=avg, zero_division=0)
        out[f"f1_{avg}"] = f1_score(y, pred, labels=list(range(k)), average=avg, zero_division=0)
    per_p = precision_score(y, pred, labels=list(range(k)), average=None, zero_division=0)
    per_r = recall_score(y, pred, labels=list(range(k)), average=None, zero_division=0)
    per_f = f1_score(y, pred, labels=list(range(k)), average=None, zero_division=0)
    for i, name in enumerate(labels):
        out[f"precision_{name}"] = per_p[i]
        out[f"recall_{name}"] = per_r[i]
        out[f"f1_{name}"] = per_f[i]
    return out


def ordinal_metrics(y, probs):
    pred = probs.argmax(1)
    ev = expected_level(probs)
    err = np.abs(pred - y)
    # MAE macro (Baccianella et al., 2009): média do MAE de cada classe verdadeira,
    # para que as classes raras (very_low, critical) pesem tanto quanto as comuns.
    macro_mae = np.mean([err[y == c].mean() for c in np.unique(y)])
    k = probs.shape[1]
    return {
        "mae": err.mean(),
        "macro_mae": macro_mae,
        "mae_expected": np.abs(ev - y).mean(),
        "within_1": (err <= 1).mean(),
        "qwk": cohen_kappa_score(y, pred, labels=list(range(k)), weights="quadratic"),
        "spearman": spearmanr(ev, y).statistic,
        "kendall_tau": kendalltau(ev, y).statistic,
    }


def probabilistic_metrics(y, probs, n_bins=15):
    n, k = probs.shape
    eps = 1e-12
    onehot = np.eye(k)[y]
    # RPS (Epstein, 1969): Brier sobre as distribuições acumuladas -> pune mais
    # errar por 3 níveis do que por 1. É a regra que o RLCD do Laya usa no score.
    cdf_gap = np.cumsum(probs, 1)[:, :-1] - np.cumsum(onehot, 1)[:, :-1]
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    # ECE com bins de largura igual (Guo et al., 2017 usam 15 bins).
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    gaps = [abs(correct[bins == b].mean() - conf[bins == b].mean()) for b in range(n_bins) if (bins == b).any()]
    weights = [(bins == b).mean() for b in range(n_bins) if (bins == b).any()]
    return {
        "nll": -np.log(probs[np.arange(n), y] + eps).mean(),
        "brier": ((probs - onehot) ** 2).sum(1).mean(),
        "rps": (cdf_gap ** 2).sum(1).mean() / (k - 1),
        "ece": float(np.dot(weights, gaps)),
        "mce": float(max(gaps)),
        "mean_confidence": conf.mean(),
    }


def topk_class_accuracy(y, probs, ks):
    """Top-k clássico: o nível verdadeiro está entre os k níveis mais prováveis?

    Só é registrado para k < K: com k >= K todo nível entra no top-k e o valor é 1.0.
    """
    order = np.argsort(-probs, 1)
    k_levels = probs.shape[1]
    return {
        f"top{k}_accuracy": (order[:, :k] == y[:, None]).any(1).mean()
        for k in ks
        if k < k_levels
    }


def ranking_score(probs, relevant_min_level, rank_by):
    """Nota usada para ordenar os tickets do 'mais provável' ao 'menos provável'."""
    if rank_by == "expected":
        return expected_level(probs)
    return probs[:, relevant_min_level:].sum(1)  # P(y >= nível relevante)


def _at_k(y, score, ks, relevant_min_level):
    rel = (y >= relevant_min_level).astype(int)
    order = np.argsort(-score, kind="stable")
    gains = 2.0 ** y - 1
    ideal = np.sort(gains)[::-1]
    discount = 1.0 / np.log2(np.arange(2, len(y) + 2))
    out = {}
    for k in ks:
        top = order[:k]
        out[f"precision_at_{k}"] = rel[top].mean()
        out[f"recall_at_{k}"] = rel[top].sum() / max(rel.sum(), 1)
        idcg = (ideal[:k] * discount[: len(ideal[:k])]).sum()
        out[f"ndcg_at_{k}"] = (gains[top] * discount[: len(top)]).sum() / idcg if idcg > 0 else 0.0
    return out


def ranking_metrics(y, probs, ks, relevant_min_level, rank_by, pool_size=0, n_pools=1, seed=0):
    """Ordena os tickets do 'mais provável' ao 'menos provável' e olha os k primeiros.

    precision@k -> dos k primeiros, quantos são relevantes (nível >= relevant_min_level)
    recall@k    -> dos relevantes que existem, quantos apareceram nos k primeiros
    ndcg@k      -> qualidade da ordem com relevância graduada (ganho 2^nível - 1)

    Com pool_size > 0, simula n_pools "caixas de entrada" de pool_size tickets sorteados do
    teste e tira a média; no teste inteiro (~2 mil tickets) os 20 primeiros saturam em 1.0.
    """
    score = ranking_score(probs, relevant_min_level, rank_by)
    rel = (y >= relevant_min_level).astype(int)
    if pool_size and pool_size < len(y):
        rng = np.random.default_rng(seed)
        pools = [rng.choice(len(y), pool_size, replace=False) for _ in range(n_pools)]
    else:
        pools = [np.arange(len(y))]
    per_pool = [_at_k(y[idx], score[idx], ks, relevant_min_level) for idx in pools]
    out = {key: np.mean([p[key] for p in per_pool]) for key in per_pool[0]}
    out["relevant_rate"] = rel.mean()
    if 0 < rel.sum() < len(rel):
        out["rank_auroc"] = roc_auc_score(rel, score)
        out["rank_average_precision"] = average_precision_score(rel, score)
    return out


def all_metrics(y, probs, labels, ks, relevant_min_level, rank_by, ece_bins,
                pool_size=0, n_pools=1, seed=0):
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=float)
    out = {}
    out.update(classification_metrics(y, probs, labels))
    out.update(ordinal_metrics(y, probs))
    out.update(probabilistic_metrics(y, probs, ece_bins))
    out.update(topk_class_accuracy(y, probs, ks))
    out.update(ranking_metrics(y, probs, ks, relevant_min_level, rank_by, pool_size, n_pools, seed))
    return {k: float(v) for k, v in out.items()}


def confusion(y, probs):
    return confusion_matrix(y, np.asarray(probs).argmax(1), labels=range(np.asarray(probs).shape[1]))
