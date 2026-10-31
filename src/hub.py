"""Baixa um checkpoint do Hugging Face para o disco de trabalho e apaga depois do run."""
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# Pastas do repositório que não fazem parte do checkpoint (imagens, resultados de benchmark).
_SKIP_DIRS = ("assets/", "eval/")


def _checkpoint_files(repo_id: str, subfolder: str, revision: str | None, token: str | None) -> list[str]:
    """Só os arquivos do checkpoint pedido.

    O repo raiz do Laya guarda outros checkpoints em subpastas (multilingual/, typed-decisions/);
    sem esse filtro o download traria ~1,5 GB a mais.
    """
    files = HfApi().list_repo_files(repo_id, revision=revision, token=token)
    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    nested = {
        f[: -len("rl_agent_config.json")]
        for f in files
        if f.endswith("/rl_agent_config.json") and f != f"{prefix}rl_agent_config.json"
    }
    keep = []
    for f in files:
        if not f.startswith(prefix):
            continue
        rel = f[len(prefix):]
        if rel.startswith(_SKIP_DIRS) or any(f.startswith(n) for n in nested if n != prefix):
            continue
        if rel.endswith((".py", ".md", ".gitattributes")):
            continue
        keep.append(f)
    if f"{prefix}rl_agent_config.json" not in keep:
        raise ValueError(
            f"{repo_id}/{subfolder} não parece um checkpoint Laya (falta rl_agent_config.json). "
            "Este projeto treina checkpoints no formato do Laya."
        )
    return keep


def download_model(repo_id: str, subfolder: str, revision: str, models_dir: Path, token: str = "") -> tuple[Path, Path]:
    """Retorna (pasta baixada, pasta do checkpoint). A primeira é a que deve ser apagada no fim."""
    revision = revision or None
    token = token or None
    name = repo_id.replace("/", "__") + (f"__{subfolder.strip('/')}" if subfolder else "")
    target = Path(models_dir) / name
    files = _checkpoint_files(repo_id, subfolder, revision, token)
    snapshot_download(repo_id, revision=revision, local_dir=target, allow_patterns=files, token=token)
    model_dir = target / subfolder if subfolder else target
    return target, model_dir


def download_snapshot(repo_id: str, revision: str, models_dir: Path, token: str = "",
                      ignore: tuple[str, ...] = ()) -> tuple[Path, Path]:
    """Repo inteiro, para famílias que não seguem o layout do Laya. Retorna (pasta baixada, pasta do modelo)."""
    target = Path(models_dir) / repo_id.replace("/", "__")
    snapshot_download(repo_id, revision=revision or None, local_dir=target,
                      ignore_patterns=list(ignore) or None, token=token or None)
    return target, target


def folder_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 2**20


def delete_model(target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
