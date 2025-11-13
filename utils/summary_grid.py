# utils/summary_grid.py
from __future__ import annotations
import os
import pandas as pd
from typing import Optional, List

# Rows = tasks, columns = model_split
TASKS  = ["VQA-X", "ACT-X", "ESNLI-VE", "VCR"]
MODELS = ["llava", "qwen"]
SPLITS = ["train", "test", "val"]


def _grid_path(results_root: str) -> str:
    """
    Path to the accuracy grid CSV for a given results root.

    Example:
        results_root = "<proj>/results/with_explanation"
        -> "<proj>/results/with_explanation/accuracy_grid.csv"
    """
    return os.path.join(results_root, "accuracy_grid.csv")


def _grid_excel_path(results_root: str) -> str:
    """
    Path to the accuracy grid Excel for a given results root.
    """
    return os.path.join(results_root, "accuracy_grid.xlsx")


def _columns() -> List[str]:
    cols = []
    for m in MODELS:
        for s in SPLITS:
            cols.append(f"{m}_{s}")
    return cols


def init_grid(results_root: str, force: bool = False) -> str:
    """
    Create accuracy grid if missing (or rebuild if force=True).

    The grid lives directly under `results_root`, i.e.:

        <results_root>/accuracy_grid.csv
        <results_root>/accuracy_grid.xlsx

    Columns: task | llava_train | llava_test | llava_val | qwen_train | qwen_test | qwen_val
    Cells initialized with None.
    """
    path = _grid_path(results_root)
    if os.path.exists(path) and not force:
        # ensure schema up to date
        df = pd.read_csv(path)
        df = _ensure_schema(df)
        _save_grid(results_root, df)
        return path

    data = {"task": TASKS}
    for col in _columns():
        data[col] = [None] * len(TASKS)
    df = pd.DataFrame(data, columns=["task"] + _columns())

    _save_grid(results_root, df)
    return path


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the grid has all expected columns and rows; keep extra user columns.
    """
    if "task" not in df.columns:
        df.insert(0, "task", None)

    existing_tasks = set(df["task"].dropna().astype(str))
    for t in TASKS:
        if t not in existing_tasks:
            new_row = {"task": t}
            for c in _columns():
                new_row[c] = None
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for c in _columns():
        if c not in df.columns:
            df[c] = None

    ordered = ["task"] + _columns()
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]
    return df


def _load_grid(results_root: str) -> pd.DataFrame:
    path = _grid_path(results_root)
    if not os.path.exists(path):
        init_grid(results_root)
    df = pd.read_csv(path)
    return _ensure_schema(df)


def _save_grid(results_root: str, df: pd.DataFrame) -> str:
    """
    Save grid to CSV and (if possible) Excel inside results_root.
    """
    csv_path = _grid_path(results_root)
    excel_path = _grid_excel_path(results_root)
    os.makedirs(results_root, exist_ok=True)

    df.to_csv(csv_path, index=False)

    try:
        df.to_excel(excel_path, index=False, engine="openpyxl")
    except ImportError:
        import warnings
        warnings.warn(
            "openpyxl is not installed. Excel export skipped. "
            "Install with: pip install openpyxl",
            UserWarning,
        )
    except Exception as e:
        import warnings
        warnings.warn(
            f"Failed to save Excel file: {e}. CSV file saved successfully.",
            UserWarning,
        )

    return csv_path


def _parse_acc(cell_value) -> Optional[float]:
    """
    Extract numeric accuracy from cells like '0.81 (50)'.
    """
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
    results_root: str,
    task: str,
    model: str,
    split: str,
    n_samples: int,
    accuracy: Optional[float],
) -> str:
    """
    Update cell [task, f"{model}_{split}"] in the grid stored under `results_root`.

    Cell content format: "acc (n_samples)".
    Only overwritten if the new accuracy is higher than the existing numeric part.
    """
    if accuracy is None:
        return _grid_path(results_root)

    model = model.lower()
    split = split.lower()
    col = f"{model}_{split}"

    df = _load_grid(results_root)
    if col not in df.columns:
        df[col] = None

    if task not in df["task"].astype(str).values:
        new_row = {"task": task}
        for c in _columns():
            new_row[c] = None
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    idx = df.index[df["task"] == task][0]
    old_acc = _parse_acc(df.at[idx, col])

    if (old_acc is None) or (float(accuracy) > old_acc):
        df.at[idx, col] = f"{accuracy:.3f} ({int(n_samples)})"

    return _save_grid(results_root, df)