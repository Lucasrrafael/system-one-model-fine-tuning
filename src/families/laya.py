"""Família laya: checkpoints no formato do Laya (rl_agent_config.json), treinados pelo loop do próprio laya.

Os três modos de treino (lora, head, full) seguem `laya.train.finetune`: separa a fatia de
calibração antes do treino, treina, ajusta uma temperatura por tipo de pergunta e salva no
formato que `laya.load` lê. A diferença é o LoRA (que o laya-train não tem) e a validação a
cada época para o MLflow.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from laya.calibrate import fit_abstention_thresholds, fit_temperature_map
from laya.common import uses_parallel_layout
from laya.train import (
    TrainConfig,
    calibration_records,
    calibration_report,
    items_from_rows,
    load_checkpoint,
    optimizer_updates,
    read_data,
    resolve_device,
    save_checkpoint,
    split_calibration,
    train_model,
)

from src import hub


def download(s):
    return hub.download_model(s.hf_model_id, s.hf_model_subfolder, s.hf_model_revision, s.models_dir, s.hf_token)


def predict_probs(s, model_dir, states, questions, base_dir=None):
    """Carrega o checkpoint com `laya.load` (aplica a temperatura salva) e prevê em lote.

    `base_dir` não é usado: um checkpoint do Laya ajustado já é completo.
    """
    import laya

    agent = laya.load(str(model_dir), device="cuda" if torch.cuda.is_available() else "cpu")
    qid = next(iter(questions))
    k = len(questions[qid]["criteria"])
    start = time.perf_counter()
    results = agent.predict_batch(states, questions, batch_size=s.eval_batch_size, sort_by_length=True)
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


def make_config(s) -> TrainConfig:
    return TrainConfig(
        epochs=s.epochs,
        micro_batch=s.micro_batch,
        grad_accum=s.grad_accum,
        # No modo lora os únicos parâmetros treináveis do encoder são os adaptadores,
        # que pedem uma taxa maior que a do fine-tuning completo (Hu et al., 2022).
        encoder_lr=s.lora_lr if s.train_mode == "lora" else s.encoder_lr,
        head_lr=s.head_lr,
        min_lr=s.min_lr,
        weight_decay=s.weight_decay,
        grad_clip=s.grad_clip,
        loss=s.loss,
        rl_samples=s.rl_samples,
        sigma_start=s.sigma_start,
        sigma_end=s.sigma_end,
        w_sph=s.w_sph,
        w_rps=s.w_rps,
        freeze_encoder=s.train_mode == "head",
        calib_max=s.calib_max,
        calib_frac=s.calib_frac,
        seed=s.seed,
        max_len=s.max_len,
        head_max_len=s.head_max_len,
        amp=s.amp,
        gradient_checkpointing=s.gradient_checkpointing,
        log_every=s.log_every,
        label_smoothing=s.label_smoothing,
        target_error=s.target_error,
    )


def add_lora(model, s):
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(r=s.lora_r, lora_alpha=s.lora_alpha, lora_dropout=s.lora_dropout,
                      target_modules=s.lora_target_modules, bias="none")
    # get_peft_model congela o encoder e deixa só os adaptadores treináveis; os nomes continuam
    # começando por "encoder.", então o train_model do Laya os põe no grupo do encoder.
    model.encoder = get_peft_model(model.encoder, lora)
    return model


def param_counts(model, mode):
    total = sum(p.numel() for p in model.parameters())
    if mode == "head":
        trainable = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder."))
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}


def train(s, base_dir: Path, out_dir: Path, on_epoch=None) -> dict:
    """Treina a partir de `base_dir` e salva o checkpoint calibrado em `out_dir`.

    on_epoch(epoch, mean_loss, val_probs, val_y) é chamado no fim de cada época.
    """
    config = make_config(s)
    config.validate()
    dev = resolve_device("auto")
    if s.train_mode == "full" and dev.type == "cuda":
        vram = torch.cuda.get_device_properties(dev).total_memory / 2**30
        if vram < 12:
            print(f"AVISO: TRAIN_MODE=full com {vram:.1f} GB de VRAM provavelmente não cabe; use lora ou head.")

    model, tok, cfg = load_checkpoint(str(base_dir))
    max_len = config.max_len or cfg.get("max_len", 512)
    head_max_len = config.head_max_len or cfg.get("head_max_len", 192)
    parallel = uses_parallel_layout(cfg)

    items, skipped = items_from_rows(tok, read_data(str(s.data_dir / "train.jsonl")), max_len, head_max_len,
                                     label_smoothing=config.label_smoothing)
    train_items, calib_items = split_calibration(items, config.calib_max, config.calib_frac, config.calib_seed)
    val_items = []
    if s.eval_every_epoch and on_epoch is not None:
        val_items, _ = items_from_rows(tok, read_data(str(s.data_dir / "val.jsonl")), max_len, head_max_len)

    if s.train_mode == "lora":
        add_lora(model, s)
    counts = param_counts(model, s.train_mode)
    model.to(dev)

    def epoch_end(epoch, mean_loss):
        probs, y = None, None
        if val_items:
            probs, y = records_probs(calibration_records(model, tok, val_items, dev, max_len, head_max_len,
                                                         batch_size=s.eval_batch_size, parallel=parallel))
        if on_epoch is not None:
            on_epoch(epoch, mean_loss, probs, y)
        # calibration_records deixa o modelo em eval(); o loop do Laya não volta para train().
        model.train()
        if config.freeze_encoder:
            model.encoder.eval()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    history = train_model(model, tok, train_items, config, dev, max_len, head_max_len,
                          on_epoch_end=epoch_end, parallel=parallel)
    train_seconds = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0

    if s.train_mode == "lora":
        # Funde os adaptadores nos pesos: o checkpoint final é um Laya normal, sem dependência do peft.
        model.encoder = model.encoder.merge_and_unload()

    records = calibration_records(model, tok, calib_items, dev, max_len, head_max_len, parallel=parallel)
    fitted = fit_temperature_map(records)
    calibration = calibration_report(records, fitted["temperature"])
    try:
        abstention = fit_abstention_thresholds(records, temperature=fitted["temperature"],
                                               temperature_by_options=fitted["temperature_by_options"],
                                               target_error=config.target_error,
                                               min_bucket_n=config.min_abstain_n)
    except Exception as e:  # mesmo tratamento do finetune do Laya
        print(f"aviso: limiares de abstenção não ajustados: {e}")
        abstention = {}

    out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                   temperature=fitted["temperature"])
    out_cfg.pop("temperature_by_options", None)
    if fitted["temperature_by_options"]:
        out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
    out_cfg["training"] = dict(out_cfg.get("training") or {}, laya_train=asdict(config),
                               laya_train_calibration=calibration, train_mode=s.train_mode)
    if abstention:
        out_cfg["training"]["abstention_thresholds"] = abstention
    save_checkpoint(model, tok, out_cfg, str(out_dir))
    questions = read_data(str(s.data_dir / "train.jsonl"))[0]["questions"]
    (Path(out_dir) / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "train_items": len(train_items),
        "calibration_items": len(calib_items),
        "val_items": len(val_items),
        "skipped": skipped,
        "optimizer_updates": optimizer_updates(len(train_items), config.micro_batch, config.grad_accum, config.epochs),
        "epoch_loss": history,
        "temperature": fitted["temperature"],
        "calibration": calibration,
        "train_seconds": train_seconds,
        "peak_vram_gb": peak_vram,
        **counts,
    }
