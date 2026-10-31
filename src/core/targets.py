"""Alvo de treino por `option_id`.

Cada modelo apresenta as opções numa ordem própria (o clef ordena `choice` alfabeticamente e
põe `noul` como true, false; o Laya usa false, true). Por isso o alvo é montado pelo id da
opção que o encoder do modelo devolve, nunca pela posição.
"""
from __future__ import annotations

import math


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "sim")
    return bool(value)


def target_vector(question: dict, option_ids, expected=None, gold=None, label_smoothing: float = 0.0) -> list[float]:
    """Distribuição alvo alinhada com `option_ids`.

    expected: noul -> bool; choice -> chave da opção; score -> índice (fracionário divide entre vizinhos).
    gold:     {option_id: probabilidade}, como no formato `gold` do laya-train.
    """
    ids = [str(o) for o in option_ids]
    k = len(ids)
    target = [0.0] * k
    qtype = question["type"]
    if gold is not None:
        for key, p in gold.items():
            target[ids.index(str(key))] = float(p)
        total = sum(target)
        target = [t / total for t in target] if total > 0 else [1.0 / k] * k
    elif qtype == "noul":
        target[ids.index("true" if _truthy(expected) else "false")] = 1.0
    elif qtype == "choice":
        target[ids.index(str(expected))] = 1.0
    elif qtype == "score":
        level = float(expected)
        if not 0 <= level <= k - 1:
            raise ValueError(f"score fora da escala 0..{k - 1}: {expected}")
        lo, hi = math.floor(level), math.ceil(level)
        target[ids.index(str(lo))] += 1.0 - (level - lo) if hi != lo else 1.0
        if hi != lo:
            target[ids.index(str(hi))] += level - lo
    else:
        raise ValueError(f"tipo de pergunta desconhecido: {qtype}")
    if label_smoothing:
        target = [(1 - label_smoothing) * t + label_smoothing / k for t in target]
    return target
