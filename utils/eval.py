from __future__ import annotations
import os
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from utils.load_data import load_vqax, load_actx, load_esnlive, load_vcr


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
_ARTICLES = {"a", "an", "the"}


def _normalize_answer(s: Optional[str]) -> str:
    """Normalize a short answer string for robust comparison."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def _split_pred_expl(text: str) -> Tuple[str, str]:
    """
    Split a model output into (<prediction>, <explanation>).

    Expected format: "<prediction> because <explanation>"
    Fallback: enforce this format heuristically.
    """
    if not isinstance(text, str):
        return "", ""
    t = text.strip()
    if " because " in t:
        pred, expl = t.split(" because ", 1)
    else:
        words = t.split()
        if len(words) >= 2:
            pred = " ".join(words[:2])
            expl = " ".join(words[2:]) or ""
        elif words:
            pred, expl = words[0], ""
        else:
            pred, expl = "", ""
    pred = re.sub(r"[^a-zA-Z0-9 ]+", " ", pred).strip()
    pred = " ".join(pred.split()[:2])
    return pred, expl.strip()


# ------------------------------------------------------------
# Evaluation for VQA-X
# ------------------------------------------------------------
def evaluate_vqax_to_csv(
    results: List[Dict],
    images_root: str,
    nle_root_vqax: str,
    split: str = "val",
    save_dir: str = "results",
    save_name: Optional[str] = None,
) -> str:
    """
    Evaluate and export VQA-X results to a CSV file.

    Columns:
        image_id, question, gt_answer, generated_answer,
        gt_explanation, generated_explanation, correct (0/1)
    """
    gt_samples = load_vqax(images_root, nle_root_vqax, split=split, require_image=False)
    gt_by_path = {}

    for s in gt_samples:
        key_full = s.image_path or ""
        key_base = os.path.basename(key_full) if key_full else ""
        gt_by_path[(key_full, key_base)] = {
            "gt_expl": s.explanation or "",
            "image_id": s.image_id,
        }

    rows = []
    for r in results:
        img_path = r.get("image", "") or ""
        key = (img_path, os.path.basename(img_path))
        gt_info = gt_by_path.get(key, {"gt_expl": "", "image_id": None})

        pred_full = r.get("prediction", "") or ""
        pred_answer, pred_expl = _split_pred_expl(pred_full)
        gt_answer = r.get("ground_truth", "") or ""

        correct = int(_normalize_answer(pred_answer) == _normalize_answer(gt_answer))

        image_id = gt_info.get("image_id")
        if image_id is None:
            m = re.search(r"(\d+)", os.path.basename(img_path))
            image_id = int(m.group(1)) if m else None

        rows.append({
            "image_id": image_id,
            "question": r.get("question", ""),
            "gt_answer": gt_answer,
            "generated_answer": pred_answer,
            "gt_explanation": gt_info.get("gt_expl", ""),
            "generated_explanation": pred_expl,
            "correct": correct,
        })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    save_name = save_name or f"vqax_{split}_eval.csv"
    out_path = os.path.join(save_dir, save_name)
    df.to_csv(out_path, index=False)
    return out_path


# ------------------------------------------------------------
# Evaluation for ACT-X
# ------------------------------------------------------------
def evaluate_actx_to_csv(
    results: List[Dict],
    images_root: str,
    nle_root_actx: str,
    split: str = "test",
    save_dir: str = "results",
    save_name: Optional[str] = None,
) -> str:
    """
    Evaluate and export ACT-X results to a CSV file.

    Columns:
        image_id, gt_label, generated_label,
        gt_explanation, generated_explanation, correct (0/1)
    """
    gt_samples = load_actx(images_root, nle_root_actx, split=split, require_image=False)
    gt_by_path = {}

    for s in gt_samples:
        key_full = s.image_path or ""
        key_base = os.path.basename(key_full) if key_full else ""
        gt_by_path[(key_full, key_base)] = {
            "gt_expl": s.explanation or "",
            "image_id": s.sample_id,
            "gt_label": s.label or "",
        }

    rows = []
    for r in results:
        img_path = r.get("image", "") or ""
        key = (img_path, os.path.basename(img_path))
        info = gt_by_path.get(key, {"gt_expl": "", "image_id": None, "gt_label": ""})

        pred_full = r.get("prediction", "") or ""
        pred_answer, pred_expl = _split_pred_expl(pred_full)
        pred_answer, pred_expl = pred_answer.lower(), pred_expl.lower()

        gt_label = r.get("ground_truth", "") or info.get("gt_label", "")
        correct = int(_normalize_answer(pred_answer) == _normalize_answer(gt_label))

        image_id = info.get("image_id")
        if image_id is None:
            m = re.search(r"(\d+)", os.path.basename(img_path))
            image_id = int(m.group(1)) if m else None

        rows.append({
            "image_id": image_id,
            "gt_label": gt_label,
            "generated_label": pred_answer,
            "gt_explanation": info.get("gt_expl", ""),
            "generated_explanation": pred_expl,
            "correct": correct,
        })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    save_name = save_name or f"actx_{split}_eval.csv"
    out_path = os.path.join(save_dir, save_name)
    df.to_csv(out_path, index=False)
    return out_path


# ------------------------------------------------------------
# Evaluation for e-SNLI-VE
# ------------------------------------------------------------
def evaluate_esnlive_to_csv(
    results: List[Dict],
    images_root: str,
    nle_root_esnlive: str,
    split: str = "test",
    save_dir: str = "results",
    save_name: Optional[str] = None,
) -> str:
    """
    Evaluate and export e-SNLI-VE results to CSV.

    Columns:
        image_name, image_numeric_id, hypothesis,
        gt_label, generated_label, gt_explanation, generated_explanation, correct
    """
    gt_samples = load_esnlive(images_root, nle_root_esnlive, split=split, require_image=False)
    gt_by_key = {}

    for s in gt_samples:
        full_path = s.image_path or ""
        base_name = os.path.basename(full_path) if full_path else (s.raw.get("image_name") or "")
        gt_by_key[(full_path, str(base_name))] = {
            "gt_expl": s.explanation or "",
            "gt_label": s.label or "",
            "hypothesis": s.hypothesis or "",
            "image_name": str(base_name),
        }

    rows = []
    for r in results:
        img_path = r.get("image", "") or ""
        base_name = os.path.basename(img_path)
        info = gt_by_key.get((img_path, base_name)) or next(
            (v for (k_full, k_base), v in gt_by_key.items() if k_base == base_name),
            {"gt_expl": "", "gt_label": "", "hypothesis": "", "image_name": base_name},
        )

        pred_full = r.get("prediction", "") or ""
        if " because " in pred_full:
            gen_label, gen_expl = pred_full.split(" because ", 1)
        else:
            parts = pred_full.strip().split()
            gen_label = " ".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")
            gen_expl = " ".join(parts[2:]) if len(parts) > 2 else ""

        gen_label, gen_expl = gen_label.lower().strip(), gen_expl.lower().strip()
        gt_label = (r.get("ground_truth") or info.get("gt_label") or "").strip()

        def _norm(s: str) -> str:
            s = (s or "").lower().strip()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            toks = [t for t in s.split() if t not in _ARTICLES]
            return " ".join(toks)

        correct = int(_norm(gen_label) == _norm(gt_label)) if gt_label else 0

        m = re.search(r"(\d+)", base_name)
        numeric_id = int(m.group(1)) if m else None

        rows.append({
            "image_name": info.get("image_name"),
            "image_numeric_id": numeric_id,
            "hypothesis": info.get("hypothesis", ""),
            "gt_label": gt_label,
            "generated_label": gen_label,
            "gt_explanation": info.get("gt_expl", ""),
            "generated_explanation": gen_expl,
            "correct": correct,
        })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    save_name = save_name or f"esnlive_{split}_eval.csv"
    out_path = os.path.join(save_dir, save_name)
    df.to_csv(out_path, index=False)
    return out_path


# ------------------------------------------------------------
# Evaluation for VCR
# ------------------------------------------------------------
def evaluate_vcr_to_csv(
    results: List[Dict],
    images_root: str,
    nle_root_vcr: str,
    split: str = "val",
    save_dir: str = "results",
    save_name: Optional[str] = None,
) -> str:
    """
    Evaluate and export VCR results to CSV.

    Columns:
        image_name, question, gt_answer, generated_answer,
        gt_explanation, generated_explanation, correct
    """
    gt_samples = load_vcr(images_root, nle_root_vcr, split=split, require_image=False)
    gt_by_key = {}

    for s in gt_samples:
        full = s.image_path or ""
        base = os.path.basename(full) if full else ""
        gt_by_key[(full, base)] = {
            "gt_expl": s.explanation or "",
            "gt_ans": s.answer or "",
            "image_name": base,
            "question": s.question or "",
        }

    rows = []
    for r in results:
        img_path = r.get("image", "") or ""
        base = os.path.basename(img_path)
        info = gt_by_key.get((img_path, base), {
            "gt_expl": "",
            "gt_ans": r.get("ground_truth", ""),
            "image_name": base,
            "question": r.get("question", ""),
        })

        gen_full = r.get("prediction", "") or ""
        if " because " in gen_full:
            gen_ans, gen_expl = gen_full.split(" because ", 1)
        else:
            toks = gen_full.split()
            gen_ans = " ".join(toks[:2]) if len(toks) >= 2 else (toks[0] if toks else "")
            gen_expl = " ".join(toks[2:]) if len(toks) > 2 else ""

        gen_ans, gen_expl = gen_ans.lower().strip(), gen_expl.lower().strip()
        gt_ans = (r.get("ground_truth") or info["gt_ans"] or "").strip()

        def _norm(s: str) -> str:
            s = (s or "").lower().strip()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            toks = [t for t in s.split() if t not in _ARTICLES]
            return " ".join(toks)

        correct = int(_norm(gen_ans) == _norm(gt_ans)) if gt_ans else 0

        rows.append({
            "image_name": info["image_name"],
            "question": info["question"],
            "gt_answer": gt_ans,
            "generated_answer": gen_ans,
            "gt_explanation": info["gt_expl"],
            "generated_explanation": gen_expl,
            "correct": correct,
        })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    save_name = save_name or f"vcr_{split}_eval.csv"
    out_path = os.path.join(save_dir, save_name)
    df.to_csv(out_path, index=False)
    return out_path