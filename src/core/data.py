"""Leitura dos splits JSONL gerados pelo prepare_data.py."""
from __future__ import annotations

import json

import numpy as np


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_split(path):
    """(estados, nível verdadeiro, perguntas) da pergunta única de cada linha."""
    rows = read_rows(path)
    qid = next(iter(rows[0]["questions"]))
    states = [r["state"] for r in rows]
    y = np.array([int(r["expected"][qid]) for r in rows])
    return states, y, {qid: rows[0]["questions"][qid]}
