# utils/expl_summary.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import Optional
import pandas as pd


COLUMNS = [
    "model",
    "task",
    "split",
    "prompt_mode",
    "accuracy",          # z.B. 0.812
    "n_samples",         # z.B. 100
    "valid_expl_pct",    # z.B. 20.0
    "valid_expl_count",  # z.B. 20
    "total_samples",     # z.B. 100
]


def _expl_table_path(results_root: str) -> str:
    """
    results_root ist der Ordner:
      - results/with_explanation
      (für without_explanation wollen wir ja KEINE Tabelle)
    """
    return os.path.join(results_root, "explanation_stats.csv")


def init_expl_table(results_root: str) -> str:
    """
    Initialisiert explanation_stats.csv, falls nicht vorhanden.
    Gibt immer den Pfad zurück.
    """
    path = _expl_table_path(results_root)
    if os.path.exists(path):
        # Falls Struktur mal geändert wurde, hier migrieren
        df = pd.read_csv(path)
        # fehlende Spalten hinzufügen
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[COLUMNS]
        df.to_csv(path, index=False)
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([], columns=COLUMNS)
    df.to_csv(path, index=False)
    return path


def append_expl_run(
    results_root: str,
    model: str,
    task: str,
    split: str,
    prompt_mode: str,
    accuracy: Optional[float],
    n_samples: int,
    valid_expl_pct: float,
    valid_expl_count: int,
    total_samples: int,
) -> str:
    """
    Hängt eine neue Zeile an explanation_stats.csv an.
    Eine Zeile pro (model, task, split, prompt_mode)-Run.
    """
    path = _expl_table_path(results_root)
    if not os.path.exists(path):
        init_expl_table(results_root)

    df = pd.read_csv(path)

    row = {
        "model": model,
        "task": task,
        "split": split,
        "prompt_mode": prompt_mode,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "n_samples": int(n_samples),
        "valid_expl_pct": float(valid_expl_pct),
        "valid_expl_count": int(valid_expl_count),
        "total_samples": int(total_samples),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return path