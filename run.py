"""Um experimento completo: baixa o modelo, mede o antes, treina, mede o depois e registra no MLflow.

Uso:  python run.py            (lê tudo do .env)
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import time
import traceback
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from src.config import load_settings

# src.config vem antes de laya/transformers/mlflow: ele aponta o HF_HOME para o D: ao ser importado.
S = load_settings()
# O console do Windows (cp1252) quebra com os emojis que o MLflow imprime no fim do run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import hub, metrics, plots  # noqa: E402
from src.core.data import read_split  # noqa: E402
from src.families import get_family  # noqa: E402

# A validação por época do Laya roda em fp32; TF32 (Ampere+) deixou ela ~35% mais rápida na 3050.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Métricas que entram na tabela de comparação antes x depois (todas as outras também vão pro MLflow).
HEADLINE = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "mcc",
            "qwk", "macro_mae", "within_1", "spearman", "nll", "brier", "rps", "ece",
            "rank_auroc", "rank_average_precision"]


def log_figure(fig, path):
    mlflow.log_figure(fig, path)
    plots.plt.close(fig)


def evaluate(tag, y, probs, extra=None):
    m = metrics.all_metrics(y, probs, S.score_labels, S.top_k, S.relevant_min_level, S.rank_by,
                            S.ece_bins, S.rank_pool_size, S.rank_pools, S.split_seed)
    if extra:
        m.update(extra)
    mlflow.log_metrics({f"{tag}/{k}": v for k, v in m.items()})
    cm = metrics.confusion(y, probs)
    log_figure(plots.confusion_plot(cm, S.score_labels, f"{tag}: matriz de confusão"), f"plots/{tag}_confusion.png")
    log_figure(plots.reliability_plot(y, probs, S.ece_bins, f"{tag}: calibração"), f"plots/{tag}_reliability.png")
    mlflow.log_dict({"labels": S.score_labels, "matrix": cm.tolist()}, f"metrics/{tag}_confusion.json")
    mlflow.log_dict(m, f"metrics/{tag}_metrics.json")
    return m


def predictions_frame(states, y, probs):
    df = pd.DataFrame(probs, columns=[f"p_{n}" for n in S.score_labels])
    df.insert(0, "y_true", [S.score_labels[i] for i in y])
    df.insert(1, "y_pred", [S.score_labels[i] for i in probs.argmax(1)])
    df.insert(2, "score_expected", metrics.expected_level(probs))
    df.insert(3, "rank_score", metrics.ranking_score(probs, S.relevant_min_level, S.rank_by))
    df["state"] = [s[:300] for s in states]
    return df.sort_values("rank_score", ascending=False)


def comparison_table(before, after):
    rows = []
    for k in HEADLINE + [f"{n}_at_{k}" for n in ("precision", "recall", "ndcg") for k in S.top_k]:
        if k in before and k in after:
            rows.append({"metric": k, "baseline": before[k], "finetuned": after[k], "delta": after[k] - before[k]})
    return pd.DataFrame(rows)


def run_name():
    model = S.hf_model_id.split("/")[-1] + (f"-{S.hf_model_subfolder}" if S.hf_model_subfolder else "")
    return S.mlflow_run_name or f"{model}-{S.train_mode}-{time.strftime('%Y%m%d-%H%M%S')}"


def sqlite_path(uri):
    return Path(uri[len("sqlite:///"):]) if uri.startswith("sqlite:///") else None


def setup_mlflow():
    db = sqlite_path(S.mlflow_tracking_uri)
    if db:
        db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(S.mlflow_tracking_uri)
    # Com SQLite (Colab) os artefatos vão para uma pasta conhecida, que o sync_mlflow.py remapeia no PC.
    if S.mlflow_artifact_location and mlflow.get_experiment_by_name(S.mlflow_experiment) is None:
        mlflow.create_experiment(S.mlflow_experiment,
                                 artifact_location=f"{S.mlflow_artifact_location.rstrip('/')}/{S.mlflow_experiment}")
    mlflow.set_experiment(S.mlflow_experiment)


def mirror(run_dir):
    """Copia o run e o MLflow (SQLite + artefatos) para MIRROR_DIR, ex.: o Google Drive no Colab."""
    if not S.mirror_dir:
        return
    dst = Path(S.mirror_dir)
    db = sqlite_path(S.mlflow_tracking_uri)
    if db and db.exists():
        (dst / "mlflow").mkdir(parents=True, exist_ok=True)
        # backup() tira uma cópia consistente mesmo com o MLflow escrevendo no banco.
        with closing(sqlite3.connect(db)) as src, closing(sqlite3.connect(dst / "mlflow" / db.name)) as out:
            src.backup(out)
        if S.mlflow_artifact_location and Path(S.mlflow_artifact_location).exists():
            shutil.copytree(S.mlflow_artifact_location, dst / "mlflow" / "artifacts", dirs_exist_ok=True)
    if run_dir.exists():
        shutil.copytree(run_dir, dst / "runs" / run_dir.name, dirs_exist_ok=True)
    print(f"  espelhado em {dst}", flush=True)


def main():
    for split in ("train", "val", "test"):
        if not (S.data_dir / f"{split}.jsonl").exists():
            raise SystemExit(f"Falta {S.data_dir / f'{split}.jsonl'}: rode antes  python prepare_data.py")

    name = run_name()
    run_dir = S.runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    states, y_test, questions = read_split(S.data_dir / "test.jsonl")
    family = get_family(S.model_family)

    setup_mlflow()
    downloaded = None
    with mlflow.start_run(run_name=name):
        mlflow.log_params(S.public_params())
        mlflow.set_tags({
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "laya": version("laya"),
            "family": S.model_family,
            "model": S.hf_model_id + (f"/{S.hf_model_subfolder}" if S.hf_model_subfolder else ""),
        })
        stats_path = S.data_dir / "data_stats.json"
        if stats_path.exists():
            mlflow.log_artifact(str(stats_path), "data")
        try:
            print(f"Baixando {S.hf_model_id} {S.hf_model_subfolder} para {S.models_dir} ...")
            downloaded, base_dir = family.download(S)
            mlflow.log_metric("model_download_mb", hub.folder_size_mb(downloaded))

            before = None
            if S.eval_baseline:
                print("Avaliando o modelo base no teste ...")
                probs_b, timing = family.predict_probs(S, base_dir, states, questions)
                before = evaluate("baseline", y_test, probs_b, timing)
                predictions_frame(states, y_test, probs_b).to_csv(run_dir / "predictions_baseline.csv", index=False)
                mlflow.log_artifact(str(run_dir / "predictions_baseline.csv"), "predictions")

            def on_epoch(epoch, loss, probs_v, y_v):
                mlflow.log_metric("train/loss", loss, step=epoch + 1)
                if probs_v is not None:
                    m = metrics.all_metrics(y_v, probs_v, S.score_labels, S.top_k, S.relevant_min_level, S.rank_by,
                                            S.ece_bins, S.rank_pool_size, S.rank_pools, S.split_seed)
                    mlflow.log_metrics({f"val/{k}": v for k, v in m.items()}, step=epoch + 1)
                    print(f"  val época {epoch + 1}: acc {m['accuracy']:.4f}  f1_macro {m['f1_macro']:.4f}  "
                          f"qwk {m['qwk']:.4f}  p@10 {m.get('precision_at_10', float('nan')):.3f}")
                mirror(run_dir)

            print(f"Treinando (modo {S.train_mode}) ...")
            ckpt_dir = run_dir / "checkpoint"
            report = family.train(S, base_dir, ckpt_dir, on_epoch)
            mlflow.log_params({k: report[k] for k in ("train_items", "calibration_items", "val_items",
                                                      "optimizer_updates", "params_total", "params_trainable")})
            mlflow.log_metrics({"train/seconds": report["train_seconds"], "train/peak_vram_gb": report["peak_vram_gb"]})
            mlflow.log_dict(report, "train_report.json")

            print("Avaliando o modelo ajustado no teste ...")
            probs_a, timing = family.predict_probs(S, ckpt_dir, states, questions, base_dir=base_dir)
            after = evaluate("finetuned", y_test, probs_a, timing)
            predictions_frame(states, y_test, probs_a).to_csv(run_dir / "predictions_finetuned.csv", index=False)
            mlflow.log_artifact(str(run_dir / "predictions_finetuned.csv"), "predictions")

            curves = {"finetuned": after} if before is None else {"baseline": before, "finetuned": after}
            log_figure(plots.at_k_plot(curves, S.top_k, "ranking no teste"), "plots/ranking_at_k.png")
            if before is not None:
                table = comparison_table(before, after)
                table.to_csv(run_dir / "comparison.csv", index=False)
                mlflow.log_artifact(str(run_dir / "comparison.csv"))
                print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

            if S.log_checkpoint_to_mlflow:
                mlflow.log_artifacts(str(ckpt_dir), "checkpoint")
            if not S.keep_finetuned_checkpoint:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        except Exception:
            mlflow.log_text(traceback.format_exc(), "error.txt")
            raise
        finally:
            if downloaded is not None and S.delete_model_after_run:
                hub.delete_model(downloaded)
                print(f"Modelo baixado apagado: {downloaded}")
    mirror(run_dir)
    print(f"\nPronto. Resultados em {S.mlflow_tracking_uri} (experimento {S.mlflow_experiment}, run {name}).")


if __name__ == "__main__":
    main()
