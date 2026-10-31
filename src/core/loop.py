"""Loop de treino genérico para modelos System One (usado pela família clef).

A família entrega `step(chunk) -> (logits[Q, K], alvo[Q, K], máscara[Q, K], tipo_laya[Q])`, uma
linha por pergunta. O loop cuida de micro-batch, acumulação, autocast, agenda de LR (warmup
linear + cosseno), clip, log de progresso e do callback de fim de época (validação e checkpoint).
As losses são as mesmas do laya, que só dependem dos logits.
"""
from __future__ import annotations

import math
import random
import time

import torch
from laya.train import rlcd_loss, sigma_at, soft_ce_loss

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def decision_loss(logits, target, mask, qtype, s, sigma):
    if s.loss == "rlcd":
        return rlcd_loss(logits, target, mask, qtype, sigma, s.rl_samples, s.w_sph, s.w_rps)
    loss = soft_ce_loss(logits, target, mask)
    if s.loss == "soft-ce+brier":
        p = torch.softmax(logits.masked_fill(~mask, -1e4), -1)
        loss = loss + s.brier_weight * (((p - target) ** 2) * mask).sum(-1).mean()
    elif s.loss != "soft-ce":
        raise ValueError(f"LOSS desconhecida: {s.loss} (use rlcd, soft-ce ou soft-ce+brier)")
    return loss


def updates_per_epoch(n_items, micro_batch, grad_accum):
    return math.ceil(math.ceil(n_items / micro_batch) / grad_accum)


def _lr_lambdas(groups, total_updates, s):
    warmup = int(total_updates * s.warmup_fraction)

    def make(base_lr):
        floor = min(1.0, s.min_lr / base_lr) if base_lr > 0 else 0.0

        def factor(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total_updates - warmup))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        return factor

    return [make(g["lr"]) for g in groups]


def train(step, groups, items, s, device, on_epoch_end=None, resume=None):
    """Treina e devolve a loss média de cada época.

    on_epoch_end(epoch, mean_loss, state) recebe o estado do otimizador para salvar o checkpoint;
    `resume` é esse mesmo estado, lido de um checkpoint anterior.
    """
    params = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=s.weight_decay)
    total = updates_per_epoch(len(items), s.micro_batch, s.grad_accum) * s.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambdas(groups, total, s))
    dtype = DTYPES[s.dtype]
    use_autocast = dtype != torch.float32
    scaler = torch.amp.GradScaler(device.type) if dtype == torch.float16 else None

    start_epoch, history, global_step = 0, [], 0
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch, history, global_step = resume["epoch"] + 1, list(resume["history"]), resume["global_step"]
        print(f"retomando da época {start_epoch + 1} (passo {global_step})", flush=True)

    torch.manual_seed(s.seed)
    for epoch in range(start_epoch, s.epochs):
        order = list(items)
        random.Random(s.seed + epoch).shuffle(order)
        sigma = sigma_at(epoch, s.epochs, s.sigma_start, s.sigma_end)
        n_micro = math.ceil(len(order) / s.micro_batch)
        total_loss, started = 0.0, time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for i in range(n_micro):
            chunk = order[i * s.micro_batch:(i + 1) * s.micro_batch]
            with torch.autocast(device.type, dtype=dtype, enabled=use_autocast):
                logits, target, mask, qtype = step(chunk)
            loss = decision_loss(logits.float(), target, mask, qtype, s, sigma)
            window = min(s.grad_accum, n_micro - (i // s.grad_accum) * s.grad_accum)
            scaled = loss / window
            (scaler.scale(scaled) if scaler else scaled).backward()
            if (i + 1) % s.grad_accum == 0 or i + 1 == n_micro:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, s.grad_clip)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            total_loss += loss.item()
            if s.log_every and (i + 1) % s.log_every == 0:
                per_step = (time.perf_counter() - started) / (i + 1)
                print(f"época {epoch + 1}/{s.epochs} passo {i + 1}/{n_micro} loss {loss.item():.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {per_step:.2f}s/passo "
                      f"faltam ~{per_step * (n_micro - i - 1) / 60:.0f} min", flush=True)
        mean = total_loss / n_micro
        history.append(mean)
        print(f"época {epoch + 1}/{s.epochs} loss média {mean:.4f}", flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean, {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                       "epoch": epoch, "global_step": global_step, "history": history})
    return history
