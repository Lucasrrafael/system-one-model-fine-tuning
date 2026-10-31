"""Converte o dataset de tickets para o JSONL que o laya-train lê.

Uso:  python prepare_data.py
Saída (em DATA_DIR): train.jsonl, val.jsonl, test.jsonl e data_stats.json
"""
from __future__ import annotations

import hashlib
import json

from src.config import load_settings  # primeiro: define HF_HOME antes do huggingface_hub

import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def score_question(s):
    return {
        "type": "score",
        "instructions": s.score_instructions,
        "criteria": s.score_criteria,
    }


def load_raw(s) -> pd.DataFrame:
    frames = []
    for name in s.dataset_files:
        path = hf_hub_download(s.dataset_id, name, repo_type="dataset", cache_dir=s.hf_cache_dir)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def build_frame(s) -> tuple[pd.DataFrame, dict]:
    df = load_raw(s)
    stats = {"rows_raw": len(df)}
    if s.dataset_language:
        df = df[df["language"] == s.dataset_language]
    stats["rows_language"] = len(df)

    df = df[df[s.label_column].isin(s.score_labels)].copy()
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str).str.replace("\\n", "\n", regex=False).str.strip()
    df = df[df["body"] != ""]
    df["state"] = ("Subject: " + df["subject"] + "\n\n" + df["body"]).str.strip()
    df["level"] = df[s.label_column].map({name: i for i, name in enumerate(s.score_labels)})

    # Os dois CSVs do dataset se sobrepõem: o mesmo ticket aparece repetido. Sem remover,
    # cópias cairiam no treino e no teste ao mesmo tempo e inflariam as métricas.
    conflicts = df.groupby("state")["level"].nunique()
    stats["duplicate_rows"] = int(df.duplicated("state").sum())
    stats["states_with_conflicting_labels"] = int((conflicts > 1).sum())
    df = df[~df["state"].isin(conflicts[conflicts > 1].index)]
    df = df.drop_duplicates("state").reset_index(drop=True)
    stats["rows_clean"] = len(df)
    return df[["state", "level"]], stats


def stratified_splits(df: pd.DataFrame, s):
    rest, test = train_test_split(df, test_size=s.test_fraction, stratify=df["level"], random_state=s.split_seed)
    val_share = s.val_fraction / (1 - s.test_fraction)
    train, val = train_test_split(rest, test_size=val_share, stratify=rest["level"], random_state=s.split_seed)

    def cap(part, n):
        if n and len(part) > n:
            part, _ = train_test_split(part, train_size=n, stratify=part["level"], random_state=s.split_seed)
        return part

    return {
        "train": cap(train, s.max_train_samples),
        "val": cap(val, s.max_eval_samples),
        "test": cap(test, s.max_eval_samples),
    }


def write_jsonl(part: pd.DataFrame, path, s) -> None:
    q = score_question(s)
    with open(path, "w", encoding="utf-8") as f:
        for state, level in zip(part["state"], part["level"]):
            row = {"state": state, "questions": {s.question_id: q}, "expected": {s.question_id: int(level)}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    s = load_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    df, stats = build_frame(s)
    splits = stratified_splits(df, s)
    for name, part in splits.items():
        write_jsonl(part, s.data_dir / f"{name}.jsonl", s)
        counts = part["level"].value_counts().sort_index()
        sha256 = hashlib.sha256((s.data_dir / f"{name}.jsonl").read_bytes()).hexdigest()
        stats[name] = {"rows": len(part), **{s.score_labels[i]: int(c) for i, c in counts.items()}, "sha256": sha256}
    (s.data_dir / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
