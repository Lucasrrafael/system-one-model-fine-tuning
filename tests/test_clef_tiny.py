"""Todo o caminho da família clef num Qwen3.5 minúsculo com pesos aleatórios.

Usa o joint_schema_model.py e o tokenizer reais do clef-flash (revisão fixada), então cobre
codificação, LoRA, loop de treino, checkpoint por época, retomada, calibração e previsão,
sem baixar os 19 GB do modelo.
"""
import json
import shutil

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

REGEX = (r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
         r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")
TEXTS = {
    0: ["How do I change my avatar?", "Feature request: dark mode for the dashboard.", "Question about invoices."],
    1: ["Reports load slowly since Monday.", "Export fails sometimes with error 500.", "Sync is intermittent."],
    2: ["Production is down, nobody can log in.", "We found a data breach in our account.", "All orders are lost."],
}


def _tiny_base(clef_files, path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from src.families import clef

    cfg = json.loads((clef_files / "config.json").read_text())
    cfg["dtype"] = "float32"
    cfg["text_config"].update(
        hidden_size=64, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, intermediate_size=128,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        dtype="float32")
    cfg["text_config"]["rope_parameters"]["mrope_section"] = [2, 1, 1]  # soma = head_dim * 0,25 / 2
    cfg["vision_config"].update(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64,
                                dtype="float32")
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(cfg)).save_pretrained(path)
    for name in ["joint_schema_model.py", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "processor_config.json"]:
        shutil.copy(clef_files / name, path / name)
    head_cfg = {"hidden_size": 64, "width": 32, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 64}
    (path / "joint_head_config.json").write_text(json.dumps(head_cfg))
    head = clef._jsm(path).JointSchemaHead(**head_cfg)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(path / "joint_head.safetensors"))


def _write_data(data_dir, criteria):
    data_dir.mkdir(parents=True)
    question = {"priority": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": criteria}}
    sizes = {"train": 24, "val": 6, "test": 6}
    for split, n in sizes.items():
        with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i in range(n):
                level = i % 3
                row = {"state": f"{TEXTS[level][i % 3]} ({split} #{i})", "questions": question,
                       "expected": {"priority": level}}
                f.write(json.dumps(row) + "\n")


@pytest.fixture
def setup(clef_files, tmp_path, monkeypatch):
    env = {
        "MODEL_FAMILY": "clef", "WORK_DIR": str(tmp_path / "work"), "RUNS_DIR": str(tmp_path / "runs"),
        "HF_MODEL_ID": "test/tiny-clef", "HF_MODEL_REVISION": "", "TRAIN_MODE": "lora", "LOSS": "rlcd",
        "EPOCHS": "1", "MICRO_BATCH": "4", "GRAD_ACCUM": "2", "DTYPE": "fp32", "LORA_R": "4", "LORA_ALPHA": "8",
        "LORA_DROPOUT": "0", "LORA_TARGET_REGEX": REGEX, "LORA_LR": "1e-3", "HEAD_LR": "1e-3",
        "CALIB_MAX": "10", "CALIB_FRAC": "0.5", "EVAL_BATCH_SIZE": "4", "LOG_EVERY": "0",
        "MAX_TRAIN_SAMPLES": "0", "MAX_EVAL_SAMPLES": "0", "RESUME_FROM": "", "EVAL_EVERY_EPOCH": "true",
        "LABEL_SMOOTHING": "0", "WARMUP_FRACTION": "0.1", "GRADIENT_CHECKPOINTING": "true",
        "CLEF_MAX_LENGTH": "512", "WEIGHT_DECAY": "0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from src.config import load_settings

    s = load_settings()
    base = tmp_path / "base"
    base.mkdir()
    _tiny_base(clef_files, base)
    _write_data(s.data_dir, s.score_criteria)
    return s, base


def test_train_checkpoint_predict_and_resume(setup, tmp_path, monkeypatch):
    from src.config import load_settings
    from src.core.data import read_split
    from src.families import clef

    s, base = setup
    epochs = []
    out = tmp_path / "ckpt"
    report = clef.train(s, base, out, lambda e, loss, probs, y: epochs.append((e, loss, probs, y)))

    assert (report["train_items"], report["calibration_items"], report["val_items"]) == (14, 10, 6)
    assert 0 < report["params_trainable"] < report["params_total"]
    (epoch, loss, probs_v, y_v), = epochs
    assert epoch == 0 and np.isfinite(loss)
    assert probs_v.shape == (6, 3) and np.allclose(probs_v.sum(1), 1) and list(y_v) == [0, 1, 2, 0, 1, 2]
    for name in ["lora/adapter_model.safetensors", "joint_head.safetensors", "joint_head_config.json",
                 "calibration.json", "base.json", "questions.json", "checkpoint_latest/trainer_state.pt"]:
        assert (out / name).exists(), name
    temps = json.loads((out / "calibration.json").read_text())["temperature"]
    assert set(temps) == {"score"} and 1.0 <= temps["score"] <= 10.0

    states, _, questions = read_split(s.data_dir / "test.jsonl")
    p_base, timing = clef.predict_probs(s, base, states, questions)
    p_ft, _ = clef.predict_probs(s, out, states, questions, base_dir=base)
    assert p_base.shape == p_ft.shape == (6, 3)
    assert np.allclose(p_base.sum(1), 1) and np.allclose(p_ft.sum(1), 1)
    assert not np.allclose(p_base, p_ft), "o checkpoint ajustado deveria mudar as previsões"
    assert timing["latency_ms_per_item"] > 0

    # Retomada: com EPOCHS=2 e RESUME_FROM, só a segunda época roda.
    monkeypatch.setenv("EPOCHS", "2")
    monkeypatch.setenv("RESUME_FROM", str(out / "checkpoint_latest"))
    epochs.clear()
    report2 = clef.train(load_settings(), base, tmp_path / "ckpt2", lambda e, *_: epochs.append((e,)))
    assert [e for e, in epochs] == [1]
    assert len(report2["epoch_loss"]) == 2


def test_head_only_mode_has_no_adapter(setup, tmp_path, monkeypatch):
    from src.config import load_settings
    from src.families import clef

    monkeypatch.setenv("TRAIN_MODE", "head")
    monkeypatch.setenv("EVAL_EVERY_EPOCH", "false")
    _, base = setup
    out = tmp_path / "ckpt-head"
    report = clef.train(load_settings(), base, out)
    assert not (out / "lora").exists() and (out / "joint_head.safetensors").exists()
    head_params = sum(p.numel() for p in clef._jsm(base).JointSchemaHead(
        **json.loads((base / "joint_head_config.json").read_text())).parameters())
    assert report["params_trainable"] == head_params
