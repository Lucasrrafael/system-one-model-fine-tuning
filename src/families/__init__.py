"""Uma família por formato de modelo System One.

Cada módulo expõe:
  download(s)                                              -> (pasta baixada, pasta do modelo)
  predict_probs(s, model_dir, states, questions, base_dir) -> (probs[n, K], tempos)
  train(s, base_dir, out_dir, on_epoch)                    -> relatório do treino
"""
from importlib import import_module


def get_family(name: str):
    return import_module(f"src.families.{name}")
