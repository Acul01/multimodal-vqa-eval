# utils/summary_grid.py
from __future__ import annotations
import os
import pandas as pd
from typing import Optional, List, Any

# Tasks, models, splits we track in the grid
TASKS  = ["VQA-X", "ACT-X", "ESNLI-VE", "VCR"]
MODELS = ["llava", "qwen"]
SPLITS = ["train", "test", "val"]


def _grid_path(results_root: str) -> str:
    """
    Path to the accuracy grid for a given results_root.
    results_root will typically be:
      - <project_root>/results/with_explanation
      - <project_root>/results/without_explanation
    """
    return os.path.join(results_root, "accuracy_grid.csv")


def _grid_excel_path(results_root: str) -> str:
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

    Columns:
        task | llava_train | llava_test | llava_val | qwen_train | qwen_test | qwen_val

    Cells are initialized with None.
    """
    path = _grid_path(results_root)
    if os.path.exists(path) and not force:
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
    Ensure the grid has all expected columns and all TASK rows.
    Keep any extra user columns untouched.
    """
    # Ensure 'task'
    if "task" not in df.columns:
        df.insert(0, "task", None)

    # Add missing rows for tasks
    existing_tasks = set(df["task"].dropna().astype(str))
    for t in TASKS:
        if t not in existing_tasks:
            new_row = {"task": t}
            for c in _columns():
                new_row[c] = None
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Add missing expected columns
    for c in _columns():
        if c not in df.columns:
            df[c] = None

    # Reorder: task + expected columns first, then extras
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
    Save grid to CSV and (if possible) Excel.
    """
    csv_path = _grid_path(results_root)
    excel_path = _grid_excel_path(results_root)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

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


def _parse_acc(cell_value: Any) -> Optional[float]:
    """
    Extract the *hard* accuracy number from a cell string.

    The cell format may be:
      - "0.812 (100)"
      - "0.812, soft:0.873 (100)"

    We always parse only the first numeric part, i.e. the hard accuracy.
    """
    if cell_value is None or (isinstance(cell_value, float) and pd.isna(cell_value)):
        return None
    if isinstance(cell_value, (float, int)):
        return float(cell_value)

    try:
        s = str(cell_value).strip()

        # Remove sample count
        if "(" in s:
            s = s.split("(", 1)[0].strip()

        # If soft accuracy is present, keep only the part before the comma
        if "," in s:
            s = s.split(",", 1)[0].strip()

        return float(s)
    except Exception:
        return None


def update_best(
    results_root: str,
    task: str,
    model: str,
    split: str,
    n_samples: Optional[int],
    accuracy: Optional[float],
    soft_accuracy: Optional[float] = None,
) -> str:
    """
    Update cell [task, f"{model}_{split}"] with either:
      - "accuracy (n)"                     if soft_accuracy is None
      - "accuracy, soft:soft_acc (n)"     if soft_accuracy is given

    Only overwrite if the new *hard* accuracy is better than the old one.
    """
    if accuracy is None:
        return _grid_path(results_root)

    model = model.lower()
    split = split.lower()
    col = f"{model}_{split}"

    df = _load_grid(results_root)

    if col not in df.columns:
        df[col] = None

    # Ensure task row exists
    if task not in df["task"].astype(str).values:
        new_row = {"task": task}
        for c in _columns():
            new_row[c] = None
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    idx = df.index[df["task"] == task][0]

    old_acc = _parse_acc(df.at[idx, col])
    new_acc = float(accuracy)
    n = int(n_samples) if n_samples is not None else 0

    if (old_acc is None) or (new_acc > old_acc):
        if soft_accuracy is not None:
            cell = f"{new_acc:.3f}, soft:{soft_accuracy:.3f} ({n})"
        else:
            cell = f"{new_acc:.3f} ({n})"
        df.at[idx, col] = cell

    return _save_grid(results_root, df)