# utils/load_data.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _search_recursively(root: str, filename: str) -> Optional[str]:
    """Recursively search for exact 'filename' under 'root'."""
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None

def _norm_expl(example: Dict[str, Any]) -> Optional[str]:
    """Extract explanation robustly (string or list → first entry)."""
    for k in ("explanation", "explanations", "rationale", "nle", "exp"):
        if k in example and example[k] is not None:
            v = example[k]
            if isinstance(v, list):
                return v[0] if v else None
            return str(v)
    return None

def _first_nonempty(*candidates):
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c
    return None

def _coco_filename_from_id(image_id: int, split_tag: str) -> str:
    # split_tag ∈ {"train","val"} → train2014/val2014
    return f"COCO_{split_tag}2014_{int(image_id):012d}.jpg"

# -----------------------------------------------------------------------------
# Dataclasses (unified return format)
# -----------------------------------------------------------------------------

@dataclass
class VQAXExample:
    image_path: str
    image_id: Optional[int]
    question: str
    answer: Optional[str]
    explanation: Optional[str]
    sample_id: Optional[str]
    raw: Dict[str, Any]

@dataclass
class ACTXExample:
    image_path: str
    label: Optional[str]
    explanation: Optional[str]
    sample_id: Optional[str]
    raw: Dict[str, Any]

@dataclass
class ESNLIVEExample:
    image_path: str
    premise: Optional[str]   
    hypothesis: str
    label: Optional[str]
    explanation: Optional[str]
    sample_id: Optional[str]
    raw: Dict[str, Any]

@dataclass
class VCRExample:
    image_path: str
    question: Optional[str]
    answer: Optional[str]
    choices: Optional[List[str]]   
    rationale: Optional[str]       
    explanation: Optional[str]
    sample_id: Optional[str]
    raw: Dict[str, Any]

# -----------------------------------------------------------------------------
# VQA-X  (Top-level = dict; answers = list of dicts; image_name present)
# -----------------------------------------------------------------------------

def load_vqax(
    images_root: str,
    ann_root: str,
    split: str = "train",
    require_image: bool = True,
) -> List[VQAXExample]:
    split = split.lower()
    fname_map = {"train": "vqaX_train.json", "val": "vqaX_val.json", "test": "vqaX_test.json"}
    if split not in fname_map:
        raise ValueError("VQA-X split must be 'train' | 'val' | 'test'.")
    ann_path = os.path.join(ann_root, fname_map[split])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"VQA-X not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # dict: {sample_id: { ... }, ...}

    # Convert to list of examples
    examples = list(data.values()) if isinstance(data, dict) else (data if isinstance(data, list) else [])

    results: List[VQAXExample] = []
    # Primary image subdirectory based on split:
    coco_sub = "train2014" if split == "train" else "val2014"
    default_dir = os.path.join(images_root, coco_sub)

    for ex in examples:
        # Question
        question = str(ex.get("question", ""))

        # Answer (answers = list of dicts with key "answer")
        answer = None
        ans_field = ex.get("answers")
        if isinstance(ans_field, list) and ans_field:
            # Take the most common/first – here: first
            first = ans_field[0]
            if isinstance(first, dict):
                answer = first.get("answer") or first.get("text") or None
            elif isinstance(first, str):
                answer = first

        # image_id 
        image_id = ex.get("image_id")
        image_id_int: Optional[int] = None
        if image_id is not None:
            try:
                image_id_int = int(image_id)
            except Exception:
                m = re.search(r"(\d+)", str(image_id))
                if m:
                    image_id_int = int(m.group(1))

        # Primary: image_name (full COCO filename)
        img_name = _first_nonempty(ex.get("image_name"), ex.get("image"), ex.get("filename"))

        # Determine image path
        img_path = ""
        tried = False
        if img_name:
            tried = True
            # If image_name already contains "COCO_val2014_...": decide subdirectory based on name
            if "train2014" in img_name:
                img_path = os.path.join(images_root, "train2014", os.path.basename(img_name))
            elif "val2014" in img_name:
                img_path = os.path.join(images_root, "val2014", os.path.basename(img_name))
            else:
                # Otherwise use standard subdirectory according to split
                img_path = os.path.join(default_dir, os.path.basename(img_name))

        if (not img_path or not os.path.exists(img_path)) and image_id_int is not None:
            # Fallback via image_id → construct COCO filename
            fname = _coco_filename_from_id(image_id_int, "train" if coco_sub == "train2014" else "val")
            cand = os.path.join(default_dir, fname)
            if os.path.exists(cand):
                img_path = cand

        if (not img_path or not os.path.exists(img_path)) and img_name:
            # Recursive search as last attempt
            alt = _search_recursively(images_root, os.path.basename(img_name))
            if alt:
                img_path = alt

        if require_image and (not img_path or not os.path.exists(img_path)):
            # Image required: skip
            continue

        results.append(VQAXExample(
            image_path=img_path if img_path else (os.path.join(default_dir, os.path.basename(img_name)) if img_name else ""),
            image_id=image_id_int,
            question=question,
            answer=answer,
            explanation=_norm_expl(ex),
            sample_id=str(ex.get("id") or ex.get("sample_id") or ex.get("question_id") or ""),
            raw=ex,
        ))

    return results

# -----------------------------------------------------------------------------
# ACT-X  (Top-level = dict; answers = label string; image_name)
# -----------------------------------------------------------------------------

def load_actx(
    images_root: str,
    ann_root: str,
    split: str = "train",
    require_image: bool = True,
) -> List[ACTXExample]:
    split = split.lower()
    if split in ("val", "valid", "validation"):
        split = "test"  # no val split available
    fname_map = {"train": "actX_train.json", "test": "actX_test.json"}
    if split not in fname_map:
        raise ValueError("ACT-X split must be 'train' | 'test' (val→test).")
    ann_path = os.path.join(ann_root, fname_map[split])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ACT-X not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # dict

    examples = list(data.values()) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    results: List[ACTXExample] = []
    img_dir = os.path.join(images_root, "mpi")

    for ex in examples:
        img_name = _first_nonempty(ex.get("image_name"), ex.get("image"), ex.get("filename")) or ""
        img_path = os.path.join(img_dir, img_name) if img_name else ""
        if img_name and not os.path.exists(img_path):
            alt = _search_recursively(img_dir, img_name) or _search_recursively(images_root, img_name)
            if alt:
                img_path = alt

        if require_image and (not img_path or not os.path.exists(img_path)):
            continue

        label = _first_nonempty(ex.get("answers"), ex.get("label"), ex.get("activity"), ex.get("class"), ex.get("label_name"))
        results.append(ACTXExample(
            image_path=img_path if img_path else (os.path.join(img_dir, img_name) if img_name else ""),
            label=label,
            explanation=_norm_expl(ex),
            sample_id=str(ex.get("image_id") or ex.get("id") or ""),
            raw=ex,
        ))
    return results

# -----------------------------------------------------------------------------
# e-SNLI-VE  (Top-level = list; answers = label string; image_name)
# -----------------------------------------------------------------------------

def load_esnlive(
    images_root: str,
    ann_root: str,
    split: str = "train",
    require_image: bool = True,
    max_samples: Optional[int] = None,
) -> List[ESNLIVEExample]:
    split = split.lower()
    if split in ("val", "valid", "validation"):
        split = "test"  # no val split available
    fname_map = {"train": "esnlive_train.json", "test": "esnlive_test.json"}
    if split not in fname_map:
        raise ValueError("e-SNLI-VE split must be 'train' | 'test' (val→test).")
    ann_path = os.path.join(ann_root, fname_map[split])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"e-SNLI-VE not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # list

    examples = data if isinstance(data, list) else (list(data.values()) if isinstance(data, dict) else [])
    results: List[ESNLIVEExample] = []
    img_dir = os.path.join(images_root, "flickr30k")

    for ex in examples:
        img_name = _first_nonempty(ex.get("image_name"), ex.get("image"), ex.get("filename")) or ""
        img_path = os.path.join(img_dir, img_name) if img_name else ""
        if img_name and not os.path.exists(img_path):
            alt = _search_recursively(img_dir, img_name) or _search_recursively(images_root, img_name)
            if alt:
                img_path = alt

        if require_image and (not img_path or not os.path.exists(img_path)):
            continue

        label = _first_nonempty(ex.get("answers"), ex.get("label"), ex.get("gold_label"), ex.get("relation"))
        results.append(ESNLIVEExample(
            image_path=img_path if img_path else (os.path.join(img_dir, img_name) if img_name else ""),
            premise=None,  # not available in this version
            hypothesis=str(ex.get("hypothesis", "")),
            label=label,
            explanation=_norm_expl(ex),
            sample_id=str(ex.get("image_name") or ex.get("id") or ""),
            raw=ex,
        ))

        if max_samples is not None and len(results) >= max_samples:
            break
    return results

# -----------------------------------------------------------------------------
# VCR  (Top-level = list; img_name = relative path under vcr1images/)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# VCR (original)  - Multi-choice with 4 answer options
# Folder: VCR_original with
#   - vcr_train_split.json
#   - vcr_dev_split.json    (used as 'val')
#   - vcr_valtest.json      (used as 'test')
# -----------------------------------------------------------------------------

def load_vcr(
    images_root: str,
    ann_root: str,
    split: str = "train",
    require_image: bool = True,
) -> List[VCRExample]:
    """
    Load the original VCR multi-choice annotations.

    Expected JSON structure per example (simplified):

        {
          "img_id": "vcr_val_...@0.npz",
          "raw_img_id": "lsmdc_.../....@0.jpg",
          "sent": "Does <det1> live in this house ?",
          "answer_choices": [
              "No, <det1> lives nowhere close.",
              "Yes, <det1> works there.",
              "No, <det1> is a visitor.",
              "No <det2> does not belong here."
          ],
          "label": 2,
          "explanation": "<det1> is wearing outerwear, holding an umbrella ...",
          "objects": [...],
          "question_id": "val-1",
          ...
        }

    We return:
      - question: ex["sent"]
      - choices: list of 4 answer choices
      - answer: the correct answer text = choices[label]
      - explanation: textual explanation (if present)
      - image_path: resolved under images_root/vcr1images
    """
    split = split.lower()
    fname_map = {
        "train": "vcr_train_split.json",
        "val":   "vcr_dev_split.json",
        "test":  "vcr_valtest.json",
    }
    if split not in fname_map:
        raise ValueError("VCR split must be 'train' | 'val' | 'test'.")

    ann_path = os.path.join(ann_root, fname_map[split])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"VCR original annotations not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level is usually a list
    examples = data if isinstance(data, list) else (list(data.values()) if isinstance(data, dict) else [])
    results: List[VCRExample] = []

    vcr_root = os.path.join(images_root, "vcr1images")

    for ex in examples:
        # ---- Image path resolution ----
        # Prefer the explicit raw_img_id (relative path under vcr1images/)
        img_rel = _first_nonempty(ex.get("raw_img_id"), ex.get("raw_img"), "")

        if img_rel:
            img_rel = img_rel.lstrip("/")
        else:
            # Fallback: derive from .npz img_id (replace .npz → .jpg)
            img_id = ex.get("img_id") or ""
            if img_id:
                guess = img_id.replace(".npz", ".jpg").lstrip("/")
                img_rel = guess

        # strip potential "vcr1images/" prefix if present
        if img_rel.startswith("vcr1images/"):
            img_rel = img_rel[len("vcr1images/"):]

        img_path = os.path.join(vcr_root, img_rel) if img_rel else ""

        if img_rel and not os.path.exists(img_path):
            # Fallback: recursively search by basename
            alt = _search_recursively(vcr_root, os.path.basename(img_rel))
            if alt:
                img_path = alt

        if require_image and (not img_path or not os.path.exists(img_path)):
            # Skip samples without a resolvable image if required
            continue

        # ---- Question ----
        question = ex.get("sent") or ex.get("question") or ""

        # ---- Answer choices + correct answer ----
        choices = ex.get("answer_choices") or ex.get("answers") or []
        if not isinstance(choices, list):
            choices = list(choices)  # be defensive

        label_idx = ex.get("label")
        answer_text: Optional[str] = None
        if isinstance(label_idx, int) and 0 <= label_idx < len(choices):
            answer_text = choices[label_idx]

        # ---- Explanation / rationale ----
        explanation = _norm_expl(ex)  # will pick ex["explanation"] if present

        results.append(
            VCRExample(
                image_path=img_path if img_path else (os.path.join(vcr_root, img_rel) if img_rel else ""),
                question=str(question),
                answer=answer_text,
                choices=choices,
                rationale=None,       # separate rationale choices are not provided here
                explanation=explanation,
                sample_id=str(
                    ex.get("question_id")
                    or ex.get("img_id")
                    or ex.get("id")
                    or ""
                ),
                raw=ex,
            )
        )

    return results

# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

def load_task(
    task: str,
    images_root: str,
    ann_root: str,
    split: str = "train",
    require_image: bool = True,
):
    t = task.replace("_", "").replace("-", "").lower()
    if t in ("vqax", "vqa", "vqaxnle"):
        return load_vqax(images_root, ann_root, split, require_image=require_image)
    if t in ("actx", "act"):
        return load_actx(images_root, ann_root, split, require_image=require_image)
    if t in ("esnlive", "esnliveve", "esnlivev", "esnlive_nle"):
        return load_esnlive(images_root, ann_root, split, require_image=require_image)
    if t == "vcr":
        return load_vcr(images_root, ann_root, split, require_image=require_image)
    raise ValueError(f"Unknown task: {task}")