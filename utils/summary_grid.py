from __future__ import annotations
import os
import pandas as pd
from typing import Optional, List

TASKS  = ["VQA-X", "ACT-X", "ESNLI-VE", "VCR"]
MODELS = ["llava", "qwen"]
SPLITS = ["train", "test", "val"]

def _grid_path(project_root: str) -> str:
    return os.path.join(project_root, "results", "accuracy_grid.csv")

def _columns() -> List[str]:
    cols = []
    for m in MODELS:
        for s in SPLITS:
            cols.append(f"{m}_{s}")
    return cols

def init_grid(project_root: str, force: bool = False) -> str:
    """
    Create accuracy grid if missing (or rebuild if force=True).
    Columns: task | llava_train | llava_test | llava_val | qwen_train | qwen_test | qwen_val
    Cells initialized with None.
    """
    path = _grid_path(project_root)
    if os.path.exists(path) and not force:
        # If file exists but schema is outdated, migrate
        df = pd.read_csv(path)
        df = _ensure_schema(df)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return path

    data = {"task": TASKS}
    for col in _columns():
        data[col] = [None] * len(TASKS)
    df = pd.DataFrame(data, columns=["task"] + _columns())

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the grid has all expected columns; add missing ones with None.
    Ensure 'task' column exists and contains all TASKS rows (add missing rows).
    Keep any extra user columns untouched.
    """
    # ensure 'task'
    if "task" not in df.columns:
        df.insert(0, "task", None)

    # add missing task rows
    existing_tasks = set(df["task"].dropna().astype(str))
    for t in TASKS:
        if t not in existing_tasks:
            new_row = {"task": t}
            for c in _columns():
                new_row[c] = None
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # add missing expected columns
    for c in _columns():
        if c not in df.columns:
            df[c] = None

    # reorder: task + expected columns first, then any extras
    ordered = ["task"] + _columns()
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]
    return df

def _load_grid(project_root: str) -> pd.DataFrame:
    path = _grid_path(project_root)
    if not os.path.exists(path):
        init_grid(project_root)
    df = pd.read_csv(path)
    return _ensure_schema(df)

def _save_grid(project_root: str, df: pd.DataFrame) -> str:
    path = _grid_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

def _parse_acc(cell_value) -> Optional[float]:
    """Extract numeric accuracy from a cell like '0.81 (50)'."""
    if cell_value is None or (isinstance(cell_value, float) and pd.isna(cell_value)):
        return None
    if isinstance(cell_value, (float, int)):
        return float(cell_value)
    try:
        s = str(cell_value).strip()
        if "(" in s:
            s = s.split("(", 1)[0]
        return float(s)
    except Exception:
        return None

def update_best(
    project_root: str,
    task: str,
    model: str,
    split: str,
    n_samples: int,
    accuracy: Optional[float],
) -> str:
    """
    Update cell [task, f"{model}_{split}"] with 'accuracy (n_samples)'
    only if the new accuracy is better than the existing numeric part.
    """
    if accuracy is None:
        return _grid_path(project_root)

    model = model.lower()
    split = split.lower()
    col = f"{model}_{split}"

    df = _load_grid(project_root)  # ensures schema
    if col not in df.columns:
        # as a safeguard; should be handled by _ensure_schema already
        df[col] = None

    # ensure task row exists
    if task not in df["task"].astype(str).values:
        new_row = {"task": task}
        for c in _columns():
            new_row[c] = None
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    idx = df.index[df["task"] == task][0]
    old_acc = _parse_acc(df.at[idx, col])

    if (old_acc is None) or (float(accuracy) > old_acc):
        df.at[idx, col] = f"{accuracy:.3f} ({int(n_samples)})"

    return _save_grid(project_root, df)