"""Lê o .env da raiz do projeto. Cada campo aqui tem uma linha correspondente no .env."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SECRET_FIELDS = {"hf_token"}

# Variáveis já definidas no terminal têm prioridade sobre o .env (útil para testes rápidos).
load_dotenv(ROOT / ".env", override=False)
# Precisam existir antes de importar huggingface_hub/transformers/mlflow, por isso ficam aqui.
_work = os.getenv("WORK_DIR", "D:/laya-treino").strip()
os.environ.setdefault("HF_HOME", str(Path(_work) / "hf-home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    return int(_get(name) or default)


def _float(name: str, default: float) -> float:
    return float(_get(name) or default)


def _bool(name: str, default: bool) -> bool:
    value = _get(name)
    return default if not value else value.lower() in ("1", "true", "yes", "sim")


def _list(name: str, default: str = "", sep: str = ",") -> list[str]:
    return [x.strip() for x in (_get(name) or default).split(sep) if x.strip()]


@dataclass
class Settings:
    # pastas
    work_dir: Path
    # modelo
    hf_model_id: str
    hf_model_subfolder: str
    hf_model_revision: str
    hf_token: str
    delete_model_after_run: bool
    # dataset
    dataset_id: str
    dataset_files: list[str]
    dataset_language: str
    label_column: str
    question_id: str
    score_instructions: str
    score_labels: list[str]
    score_criteria: list[str]
    val_fraction: float
    test_fraction: float
    split_seed: int
    max_train_samples: int
    max_eval_samples: int
    # treino
    train_mode: str
    loss: str
    epochs: int
    micro_batch: int
    grad_accum: int
    encoder_lr: float
    head_lr: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    label_smoothing: float
    max_len: int
    head_max_len: int
    amp: bool
    gradient_checkpointing: bool
    seed: int
    rl_samples: int
    sigma_start: float
    sigma_end: float
    w_sph: float
    w_rps: float
    calib_max: int
    calib_frac: float
    target_error: float
    log_every: int
    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_lr: float
    lora_target_modules: list[str]
    # avaliação
    eval_baseline: bool
    eval_every_epoch: bool
    eval_batch_size: int
    top_k: list[int]
    relevant_min_level: int
    rank_by: str
    rank_pool_size: int
    rank_pools: int
    ece_bins: int
    # mlflow
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_run_name: str
    keep_finetuned_checkpoint: bool
    log_checkpoint_to_mlflow: bool

    @property
    def data_dir(self) -> Path:
        # Recortes menores (teste rápido) ganham pasta própria e não sobrescrevem os dados completos.
        cut = ""
        if self.max_train_samples or self.max_eval_samples:
            cut = f"-train{self.max_train_samples or 'all'}-eval{self.max_eval_samples or 'all'}"
        return self.work_dir / "data" / f"{self.dataset_language}{cut}"

    @property
    def models_dir(self) -> Path:
        return self.work_dir / "models"

    @property
    def runs_dir(self) -> Path:
        return self.work_dir / "runs"

    @property
    def hf_cache_dir(self) -> Path:
        return self.work_dir / "hf-cache"

    def public_params(self) -> dict:
        """Tudo que vai para o MLflow como parâmetro (sem o token)."""
        return {k: (",".join(map(str, v)) if isinstance(v, list) else str(v))
                for k, v in asdict(self).items() if k not in SECRET_FIELDS}


def load_settings() -> Settings:
    s = Settings(
        work_dir=Path(_get("WORK_DIR", "D:/laya-treino")),
        hf_model_id=_get("HF_MODEL_ID", "convaiinnovations/laya"),
        hf_model_subfolder=_get("HF_MODEL_SUBFOLDER"),
        hf_model_revision=_get("HF_MODEL_REVISION"),
        hf_token=_get("HF_TOKEN"),
        delete_model_after_run=_bool("DELETE_MODEL_AFTER_RUN", True),
        dataset_id=_get("DATASET_ID", "Tobi-Bueck/customer-support-tickets"),
        dataset_files=_list("DATASET_FILES"),
        dataset_language=_get("DATASET_LANGUAGE", "en"),
        label_column=_get("LABEL_COLUMN", "priority"),
        question_id=_get("QUESTION_ID", "priority"),
        score_instructions=_get("SCORE_INSTRUCTIONS"),
        score_labels=_list("SCORE_LABELS"),
        score_criteria=_list("SCORE_CRITERIA", sep="|"),
        val_fraction=_float("VAL_FRACTION", 0.1),
        test_fraction=_float("TEST_FRACTION", 0.1),
        split_seed=_int("SPLIT_SEED", 42),
        max_train_samples=_int("MAX_TRAIN_SAMPLES", 0),
        max_eval_samples=_int("MAX_EVAL_SAMPLES", 0),
        train_mode=_get("TRAIN_MODE", "lora").lower(),
        loss=_get("LOSS", "rlcd"),
        epochs=_int("EPOCHS", 4),
        micro_batch=_int("MICRO_BATCH", 8),
        grad_accum=_int("GRAD_ACCUM", 8),
        encoder_lr=_float("ENCODER_LR", 2.5e-5),
        head_lr=_float("HEAD_LR", 1e-4),
        min_lr=_float("MIN_LR", 1e-6),
        weight_decay=_float("WEIGHT_DECAY", 0.01),
        grad_clip=_float("GRAD_CLIP", 1.0),
        label_smoothing=_float("LABEL_SMOOTHING", 0.0),
        max_len=_int("MAX_LEN", 512),
        head_max_len=_int("HEAD_MAX_LEN", 192),
        amp=_bool("AMP", True),
        gradient_checkpointing=_bool("GRADIENT_CHECKPOINTING", True),
        seed=_int("SEED", 0),
        rl_samples=_int("RL_SAMPLES", 4),
        sigma_start=_float("SIGMA_START", 0.4),
        sigma_end=_float("SIGMA_END", 0.1),
        w_sph=_float("W_SPH", 0.75),
        w_rps=_float("W_RPS", 1.0),
        calib_max=_int("CALIB_MAX", 400),
        calib_frac=_float("CALIB_FRAC", 0.1),
        target_error=_float("TARGET_ERROR", 0.10),
        log_every=_int("LOG_EVERY", 100),
        lora_r=_int("LORA_R", 8),
        lora_alpha=_int("LORA_ALPHA", 16),
        lora_dropout=_float("LORA_DROPOUT", 0.1),
        lora_lr=_float("LORA_LR", 3e-4),
        lora_target_modules=_list("LORA_TARGET_MODULES", "Wqkv,Wo,Wi"),
        eval_baseline=_bool("EVAL_BASELINE", True),
        eval_every_epoch=_bool("EVAL_EVERY_EPOCH", True),
        eval_batch_size=_int("EVAL_BATCH_SIZE", 32),
        top_k=[int(k) for k in _list("TOP_K", "1,3,5,10,15,20")],
        relevant_min_level=_int("RELEVANT_MIN_LEVEL", 2),
        rank_by=_get("RANK_BY", "prob"),
        rank_pool_size=_int("RANK_POOL_SIZE", 100),
        rank_pools=_int("RANK_POOLS", 500),
        ece_bins=_int("ECE_BINS", 15),
        mlflow_tracking_uri=_get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment=_get("MLFLOW_EXPERIMENT", "laya-tickets-priority"),
        mlflow_run_name=_get("MLFLOW_RUN_NAME"),
        keep_finetuned_checkpoint=_bool("KEEP_FINETUNED_CHECKPOINT", True),
        log_checkpoint_to_mlflow=_bool("LOG_CHECKPOINT_TO_MLFLOW", False),
    )
    if len(s.score_labels) != len(s.score_criteria):
        raise ValueError("SCORE_LABELS e SCORE_CRITERIA precisam ter o mesmo número de níveis")
    if s.train_mode not in ("lora", "head", "full"):
        raise ValueError("TRAIN_MODE deve ser lora, head ou full")
    if s.rank_by not in ("prob", "expected"):
        raise ValueError("RANK_BY deve ser prob ou expected")
    if s.hf_token:
        os.environ.setdefault("HF_TOKEN", s.hf_token)
    return s
