"""Importa os runs feitos no Colab (MLflow em SQLite, espelhado no Drive) para o seu MLflow local.

Baixe do Drive a pasta  system-one/mlflow  (mlflow.db + artifacts/) e rode:

    python sync_mlflow.py --source "C:\\caminho\\para\\mlflow"

Recria cada run no servidor de destino com params, tags, histórico completo das métricas e
artefatos. Rodar de novo não duplica: cada run importado leva a tag sync.source_run_id.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from src.config import load_settings  # primeiro: carrega o .env

from mlflow.entities import Metric, Param  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

SOURCE_TAG = "sync.source_run_id"
# MLflow limita o tamanho de cada log_batch.
BATCH = {"metrics": 1000, "params": 100, "tags": 100}


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _plain_path(uri: str) -> str:
    uri = uri.removeprefix("file://")
    if re.match(r"^/[A-Za-z]:", uri):  # file:///C:/... no Windows
        uri = uri[1:]
    return uri[0].upper() + uri[1:] if re.match(r"^[A-Za-z]:", uri) else uri  # o MLflow às vezes grava c:/


def _local_artifacts(artifact_uri: str, colab_root: str, local_root: Path) -> Path:
    """`/content/mlflow/artifacts/<exp>/<run>/artifacts` -> `<local_root>/<exp>/<run>/artifacts`."""
    uri, root = _plain_path(artifact_uri), _plain_path(colab_root).rstrip("/")
    if not uri.startswith(root):
        raise ValueError(f"artefato fora de {colab_root}: {artifact_uri} (ajuste --colab-artifacts)")
    return local_root / uri[len(root):].lstrip("/")


def sync(source_db: Path, source_artifacts: Path, target_uri: str, colab_root: str = "/content/mlflow/artifacts",
         experiments: list[str] | None = None) -> list[tuple[str, str]]:
    """Copia os runs. Devolve [(run_id de origem, run_id novo)] dos que foram importados agora."""
    src = MlflowClient(tracking_uri=f"sqlite:///{Path(source_db).resolve().as_posix()}")
    dst = MlflowClient(tracking_uri=target_uri)
    imported = []
    for exp in src.search_experiments():
        if experiments and exp.name not in experiments:
            continue
        dst_exp = dst.get_experiment_by_name(exp.name)
        dst_exp_id = dst_exp.experiment_id if dst_exp else dst.create_experiment(exp.name)
        done = {r.data.tags.get(SOURCE_TAG) for r in dst.search_runs([dst_exp_id], max_results=50000)}
        for run in src.search_runs([exp.experiment_id], max_results=50000):
            if run.info.run_id in done:
                print(f"  já importado: {run.info.run_name}")
                continue
            tags = {k: v for k, v in run.data.tags.items() if k != "mlflow.runName"}
            tags[SOURCE_TAG] = run.info.run_id
            new = dst.create_run(dst_exp_id, start_time=run.info.start_time, run_name=run.info.run_name, tags=tags)
            new_id = new.info.run_id
            for part in _chunks([Param(k, v) for k, v in run.data.params.items()], BATCH["params"]):
                dst.log_batch(new_id, params=part)
            history = [m for key in run.data.metrics for m in src.get_metric_history(run.info.run_id, key)]
            for part in _chunks([Metric(m.key, m.value, m.timestamp, m.step) for m in history], BATCH["metrics"]):
                dst.log_batch(new_id, metrics=part)
            local = _local_artifacts(run.info.artifact_uri, colab_root, Path(source_artifacts))
            if local.exists() and any(local.iterdir()):
                dst.log_artifacts(new_id, str(local))
            dst.set_terminated(new_id, status=run.info.status, end_time=run.info.end_time)
            imported.append((run.info.run_id, new_id))
            print(f"  importado: {exp.name} / {run.info.run_name} ({len(history)} pontos de métrica)")
    return imported


def main():
    s = load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path,
                        help="pasta baixada do Drive com mlflow.db e artifacts/")
    parser.add_argument("--target", default=s.mlflow_tracking_uri, help="MLflow de destino (padrão: o do .env)")
    parser.add_argument("--colab-artifacts", default="/content/mlflow/artifacts",
                        help="MLFLOW_ARTIFACT_LOCATION usado no Colab")
    parser.add_argument("--experiment", action="append", help="só estes experimentos (pode repetir)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    imported = sync(args.source / "mlflow.db", args.source / "artifacts", args.target,
                    args.colab_artifacts, args.experiment)
    print(f"\n{len(imported)} run(s) importado(s) para {args.target}")


if __name__ == "__main__":
    main()
