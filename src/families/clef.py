"""Família clef: Cloudflare/clef-flash e afins (Qwen3.5 + cabeça de esquema conjunta).

Codificação, batch e modelo vêm do `joint_schema_model.py` do próprio repositório do modelo,
na revisão fixada em HF_MODEL_REVISION. O treino usa o loop genérico (`src/core/loop.py`):
backbone congelado, LoRA nas camadas de texto e a cabeça inteira, como a Cloudflare descreve.

Checkpoint ajustado (não guarda o backbone de 19 GB, só o que foi treinado):
  lora/                    adaptador peft (TRAIN_MODE=lora)
  joint_head.safetensors   cabeça treinada
  joint_head_config.json
  calibration.json         {"temperature": {"noul": T, "choice": T, "score": T}}
  base.json                modelo e revisão de origem
  questions.json
  trainer_state.pt         só em checkpoint_latest/, para retomar com RESUME_FROM
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from laya.train import split_calibration
from safetensors.torch import load_file, save_file
from torch.nn.utils.rnn import pad_sequence

from src import hub
from src.core import calibration, loop
from src.core.data import read_rows
from src.core.targets import target_vector

QTYPE_NAMES = {0: "noul", 1: "choice", 2: "score"}  # códigos do clef
# rlcd_loss do laya usa outros códigos (choice 0, score 1, noul 2) para saber onde aplicar o RPS.
TO_LAYA_QTYPE = {0: 2, 1: 0, 2: 1}
CALIB_SEED = 20260922
IGNORE = (".eval_results/*",)
_MODULES: dict[str, object] = {}


def download(s):
    if not s.hf_model_revision:
        print("AVISO: HF_MODEL_REVISION vazio: o joint_schema_model.py vem da main, sem versão fixa.")
    return hub.download_snapshot(s.hf_model_id, s.hf_model_revision, s.models_dir, s.hf_token, IGNORE)


def _jsm(base_dir):
    """Importa o joint_schema_model.py da pasta baixada (uma vez por pasta)."""
    path = (Path(base_dir) / "joint_schema_model.py").resolve()
    if str(path) not in _MODULES:
        spec = importlib.util.spec_from_file_location(f"joint_schema_model_{len(_MODULES)}", path)
        module = importlib.util.module_from_spec(spec)
        # Os @dataclass do arquivo procuram o próprio módulo em sys.modules ao serem criados.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _MODULES[str(path)] = module
    return _MODULES[str(path)]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_id(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def _load(s, base_dir, ckpt_dir=None):
    jsm = _jsm(base_dir)
    model, processor = jsm.load_release_model(str(base_dir), device=_device(), dtype=loop.DTYPES[s.dtype])
    if ckpt_dir is not None:
        if (Path(ckpt_dir) / "lora").exists():
            from peft import PeftModel

            model.language_model = PeftModel.from_pretrained(model.language_model, str(Path(ckpt_dir) / "lora"))
        model.head.load_state_dict(load_file(str(Path(ckpt_dir) / "joint_head.safetensors")))
    return jsm, model.eval(), processor


def _prepare_training(model, s):
    """Congela o backbone, põe LoRA nas camadas de texto e deixa a cabeça treinável em fp32."""
    for p in model.parameters():
        p.requires_grad_(False)
    if s.train_mode == "lora":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(r=s.lora_r, lora_alpha=s.lora_alpha, lora_dropout=s.lora_dropout, bias="none",
                          target_modules=s.lora_target_regex or s.lora_target_modules)
        model.language_model = get_peft_model(model.language_model, lora)
    model.head.float().requires_grad_(True)
    backbone = model.language_model.get_base_model() if hasattr(model.language_model, "get_base_model") \
        else model.language_model
    if s.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.config.use_cache = False
    model.train()
    lora_params = [p for p in model.language_model.parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": s.lora_lr}, {"params": list(model.head.parameters()), "lr": s.head_lr}]
    return [g for g in groups if g["params"]]


def _canonical_order(question, option_ids):
    """Posição de cada opção na ordem do nosso JSONL (score 0..K-1, choice na ordem dos criteria)."""
    if question["type"] == "score":
        return [int(o) for o in option_ids]
    if question["type"] == "noul":
        return [["false", "true"].index(o) for o in option_ids]
    keys = [str(k) for k in question["criteria"]]
    return [keys.index(o) for o in option_ids]


def _encode_rows(jsm, tok, rows, s, label_smoothing=0.0, with_targets=True):
    items = []
    for row in rows:
        enc = jsm.encode_record(tok, {"state": row["state"], "questions": row["questions"]},
                                max_length=s.clef_max_length)
        item = {"enc": enc, "qtypes": [q.question_type for q in enc.questions]}
        if with_targets:
            item["targets"] = [
                target_vector(row["questions"][q.question_id], q.option_ids,
                              expected=(row.get("expected") or {}).get(q.question_id),
                              gold=((row.get("gold") or {}).get(q.question_id) or {}).get("probabilities"),
                              label_smoothing=label_smoothing)
                for q in enc.questions
            ]
        items.append(item)
    return items


def _forward(jsm, model, tok, chunk, dev):
    """Uma linha por pergunta: logits com preenchimento, alvo, máscara e tipo (códigos do laya)."""
    batch = jsm.collate_records([it["enc"] for it in chunk], _pad_id(tok), dev)
    flat = [q for record in model(batch) for q in record]
    lengths = torch.tensor([len(q) for q in flat], device=dev)
    logits = pad_sequence([q.float() for q in flat], batch_first=True, padding_value=-1e4)
    mask = torch.arange(logits.shape[1], device=dev)[None, :] < lengths[:, None]
    target = pad_sequence([torch.tensor(t, device=dev, dtype=torch.float32) for it in chunk for t in it["targets"]],
                          batch_first=True)
    qtype = torch.tensor([TO_LAYA_QTYPE[q] for it in chunk for q in it["qtypes"]], device=dev)
    return logits, target, mask, qtype


@torch.inference_mode()
def _logits(jsm, model, tok, items, s):
    """Logits (numpy, temperatura 1) por item e por pergunta, na ordem de option_ids."""
    model.eval()
    dev = _device()
    dtype = loop.DTYPES[s.dtype]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["enc"].input_ids))
    out = [None] * len(items)
    for start in range(0, len(order), s.eval_batch_size):
        idx = order[start:start + s.eval_batch_size]
        batch = jsm.collate_records([items[i]["enc"] for i in idx], _pad_id(tok), dev)
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
            nested = model(batch)
        for i, record in zip(idx, nested):
            out[i] = [q.float().cpu().numpy() for q in record]
    return out


def _probs(items, logits, questions, temperatures=None):
    """Probabilidades da primeira pergunta, na ordem canônica, com a temperatura do tipo."""
    qid = next(iter(questions))
    rows, ys = [], []
    for item, record_logits in zip(items, logits):
        j = [q.question_id for q in item["enc"].questions].index(qid)
        q = item["enc"].questions[j]
        t = (temperatures or {}).get(QTYPE_NAMES[q.question_type], 1.0)
        z = record_logits[j] / t
        p = np.exp(z - z.max())
        p /= p.sum()
        canon = _canonical_order(questions[qid], q.option_ids)
        row = np.zeros(len(p))
        row[canon] = p
        rows.append(row)
        if "targets" in item:
            ys.append(canon[int(np.argmax(item["targets"][j]))])
    return np.array(rows), (np.array(ys) if ys else None)


def predict_probs(s, model_dir, states, questions, base_dir=None):
    """Modelo base (model_dir = pasta baixada) ou ajustado (model_dir = checkpoint, base_dir = base)."""
    finetuned = base_dir is not None and Path(model_dir) != Path(base_dir)
    jsm, model, processor = _load(s, base_dir if finetuned else model_dir, model_dir if finetuned else None)
    temperatures = None
    if finetuned and (Path(model_dir) / "calibration.json").exists():
        temperatures = json.loads((Path(model_dir) / "calibration.json").read_text())["temperature"]
    tok = processor.tokenizer
    items = _encode_rows(jsm, tok, [{"state": st, "questions": questions} for st in states], s, with_targets=False)
    start = time.perf_counter()
    probs, _ = _probs(items, _logits(jsm, model, tok, items, s), questions, temperatures)
    elapsed = time.perf_counter() - start
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs, {"latency_ms_per_item": 1000 * elapsed / len(states), "eval_seconds": elapsed}


def _save(model, base_dir, dst, s, questions, temperatures=None, trainer_state=None, head_dtype=torch.bfloat16):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if hasattr(model.language_model, "peft_config"):
        model.language_model.save_pretrained(str(dst / "lora"))
    save_file({k: v.detach().to(head_dtype).cpu().contiguous() for k, v in model.head.state_dict().items()},
              str(dst / "joint_head.safetensors"))
    shutil.copy(Path(base_dir) / "joint_head_config.json", dst / "joint_head_config.json")
    (dst / "base.json").write_text(json.dumps({"hf_model_id": s.hf_model_id, "revision": s.hf_model_revision,
                                               "train_mode": s.train_mode}, indent=2), encoding="utf-8")
    (dst / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
    if temperatures is not None:
        (dst / "calibration.json").write_text(json.dumps({"temperature": temperatures}, indent=2), encoding="utf-8")
    if trainer_state is not None:
        torch.save(trainer_state, dst / "trainer_state.pt")


def _restore_trainable(model, ckpt_dir):
    from peft import set_peft_model_state_dict

    if (Path(ckpt_dir) / "lora").exists():
        set_peft_model_state_dict(model.language_model,
                                  load_file(str(Path(ckpt_dir) / "lora" / "adapter_model.safetensors")))
    model.head.load_state_dict(load_file(str(Path(ckpt_dir) / "joint_head.safetensors")))


def train(s, base_dir, out_dir, on_epoch=None) -> dict:
    """Treina a partir de `base_dir` e salva o checkpoint calibrado em `out_dir`.

    on_epoch(epoch, mean_loss, val_probs, val_y) é chamado no fim de cada época, depois de salvar
    `out_dir/checkpoint_latest` (use RESUME_FROM=<essa pasta> para continuar se o Colab cair).
    """
    out_dir, dev = Path(out_dir), _device()
    jsm, model, processor = _load(s, base_dir)
    tok = processor.tokenizer
    groups = _prepare_training(model, s)
    resume = None
    if s.resume_from:
        _restore_trainable(model, s.resume_from)
        resume = torch.load(Path(s.resume_from) / "trainer_state.pt", map_location=dev, weights_only=False)

    rows = read_rows(s.data_dir / "train.jsonl")
    questions = rows[0]["questions"]
    items = _encode_rows(jsm, tok, rows, s, label_smoothing=s.label_smoothing)
    train_items, calib_items = split_calibration(items, s.calib_max, s.calib_frac, CALIB_SEED)
    val_items = []
    if s.eval_every_epoch and on_epoch is not None:
        val_items = _encode_rows(jsm, tok, read_rows(s.data_dir / "val.jsonl"), s)
    counts = {"params_total": sum(p.numel() for p in model.parameters()),
              "params_trainable": sum(p.numel() for g in groups for p in g["params"])}
    print(f"clef: {len(train_items)} itens de treino, {len(calib_items)} de calibração, "
          f"{counts['params_trainable'] / 1e6:.1f}M parâmetros treináveis", flush=True)

    def epoch_end(epoch, mean_loss, state):
        probs, y = (_probs(val_items, _logits(jsm, model, tok, val_items, s), questions) if val_items
                    else (None, None))
        _save(model, base_dir, out_dir / "checkpoint_latest", s, questions,
              trainer_state=state, head_dtype=torch.float32)
        if on_epoch is not None:
            on_epoch(epoch, mean_loss, probs, y)
        model.train()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    history = loop.train(lambda chunk: _forward(jsm, model, tok, chunk, dev), groups, train_items, s, dev,
                         epoch_end, resume)
    train_seconds = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0

    records = []
    for item, record_logits in zip(calib_items, _logits(jsm, model, tok, calib_items, s)):
        for qtype, z, t in zip(item["qtypes"], record_logits, item["targets"]):
            records.append((QTYPE_NAMES[qtype], z, t))
    temperatures = calibration.fit_temperatures(records, min_t=s.calib_min_temperature)
    print(f"temperaturas: {temperatures}", flush=True)
    _save(model, base_dir, out_dir, s, questions, temperatures=temperatures)

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "train_items": len(train_items),
        "calibration_items": len(calib_items),
        "val_items": len(val_items),
        "optimizer_updates": loop.updates_per_epoch(len(train_items), s.micro_batch, s.grad_accum) * s.epochs,
        "epoch_loss": history,
        "temperature": temperatures,
        "train_seconds": train_seconds,
        "peak_vram_gb": peak_vram,
        **counts,
    }
