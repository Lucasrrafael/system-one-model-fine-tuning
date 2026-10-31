"""Uma temperatura por tipo de pergunta (temperature scaling, Guo et al., 2017).

Ajustada por NLL numa fatia do treino separada antes de treinar. O piso de 1 segue o
Inner Jev para backbones Qwen: só deixa o modelo menos confiante, nunca mais.
"""
from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def fit_temperatures(records, min_t=1.0, max_t=10.0, min_n=10):
    """records: [(tipo, logits, alvo)] -> {tipo: T}. Tipos com menos de min_n itens ficam em max(1, min_t)."""
    by_type = {}
    for qtype, logits, target in records:
        by_type.setdefault(qtype, []).append((logits, target))
    out = {}
    for qtype, rows in by_type.items():
        if len(rows) < min_n:
            out[qtype] = max(1.0, min_t)
            continue
        # Perguntas com número diferente de opções: o preenchimento fica num valor finito e muito
        # baixo (com -inf o gradiente de T vira NaN); como o alvo ali é 0, ele não entra na NLL.
        logits = pad_sequence([torch.as_tensor(z, dtype=torch.float64) for z, _ in rows],
                              batch_first=True, padding_value=-1e4)
        target = pad_sequence([torch.as_tensor(t, dtype=torch.float64) for _, t in rows], batch_first=True)
        log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

        def nll():
            optimizer.zero_grad()
            logp = torch.log_softmax(logits / log_t.exp(), -1)
            loss = -(target * logp).sum(-1).mean()
            loss.backward()
            return loss

        optimizer.step(nll)
        out[qtype] = float(min(max_t, max(min_t, log_t.exp().item())))
    return out
