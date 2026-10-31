"""Probabilidades por nível no conjunto de teste, pelo mesmo caminho de inferência do Laya."""
from __future__ import annotations

import json
import time

import numpy as np
import torch


def read_split(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    qid = next(iter(rows[0]["questions"]))
    states = [r["state"] for r in rows]
    y = np.array([int(r["expected"][qid]) for r in rows])
    return states, y, {qid: rows[0]["questions"][qid]}


def agent_probs(model_dir, states, questions, batch_size):
    """Carrega o checkpoint com `laya.load` (aplica a temperatura salva) e prevê em lote."""
    import laya

    agent = laya.load(str(model_dir), device="cuda" if torch.cuda.is_available() else "cpu")
    qid = next(iter(questions))
    k = len(questions[qid]["criteria"])
    start = time.perf_counter()
    results = agent.predict_batch(states, questions, batch_size=batch_size, sort_by_length=True)
    elapsed = time.perf_counter() - start
    probs = np.array([[r["answers"][qid]["probabilities"][str(i)] for i in range(k)] for r in results])
    probs = probs / probs.sum(1, keepdims=True)  # o Laya arredonda para 4 casas
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs, {"latency_ms_per_item": 1000 * elapsed / len(states), "eval_seconds": elapsed}


def records_probs(records):
    """Probabilidades a partir de `laya.train.calibration_records` (temperatura 1)."""
    rows = []
    for _qt, logits, _target, _k in records:
        z = np.asarray(logits, dtype=float)
        p = np.exp(z - z.max())
        rows.append(p / p.sum())
    return np.array(rows), np.array([int(np.argmax(t)) for _q, _l, t, _k in records])
