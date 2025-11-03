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
    """Rekursiv nach exakt 'filename' unter 'root' suchen."""
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None

def _norm_expl(example: Dict[str, Any]) -> Optional[str]:
    """Erklärung robuster extrahieren (String oder Liste → erster Eintrag)."""
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
# Dataclasses (vereinheitlichte Rückgabe)
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
    premise: Optional[str]     # bei dir nicht vorhanden → bleibt None
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
    choices: Optional[List[str]]   # in deiner Version nicht vorhanden
    rationale: Optional[str]       # in deiner Version nicht vorhanden
    explanation: Optional[str]
    sample_id: Optional[str]
    raw: Dict[str, Any]

# -----------------------------------------------------------------------------
# VQA-X  (Top-Level = dict; answers = Liste von Dicts; image_name vorhanden)
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
        raise ValueError("VQAX split muss 'train' | 'val' | 'test' sein.")
    ann_path = os.path.join(ann_root, fname_map[split])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"VQA-X nicht gefunden: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # dict: {sample_id: { ... }, ...}

    # in Liste der Beispiele überführen
    examples = list(data.values()) if isinstance(data, dict) else (data if isinstance(data, list) else [])

    results: List[VQAXExample] = []
    # Primärer Bild-Unterordner nach Split:
    coco_sub = "train2014" if split == "train" else "val2014"
    default_dir = os.path.join(images_root, coco_sub)

    for ex in examples:
        # Frage
        question = str(ex.get("question", ""))

        # Antwort (answers = Liste von Dicts mit Key "answer")
        answer = None
        ans_field = ex.get("answers")
        if isinstance(ans_field, list) and ans_field:
            # Nimm die häufigste/erste – hier: erste
            first = ans_field[0]
            if isinstance(first, dict):
                answer = first.get("answer") or first.get("text") or None
            elif isinstance(first, str):
                answer = first

        # image_id (optional numerisch)
        image_id = ex.get("image_id")
        image_id_int: Optional[int] = None
        if image_id is not None:
            try:
                image_id_int = int(image_id)
            except Exception:
                m = re.search(r"(\d+)", str(image_id))
                if m:
                    image_id_int = int(m.group(1))

        # Primär: image_name (voller COCO-Dateiname)
        img_name = _first_nonempty(ex.get("image_name"), ex.get("image"), ex.get("filename"))

        # Bildpfad bestimmen
        img_path = ""
        tried = False
        if img_name:
            tried = True
            # wenn image_name schon "COCO_val2014_..." enthält: entscheide Unterordner anhand des Namens
            if "train2014" in img_name:
                img_path = os.path.join(images_root, "train2014", os.path.basename(img_name))
            elif "val2014" in img_name:
                img_path = os.path.join(images_root, "val2014", os.path.basename(img_name))
            else:
                # sonst Standard-Unterordner gemäß Split
                img_path = os.path.join(default_dir, os.path.basename(img_name))

        if (not img_path or not os.path.exists(img_path)) and image_id_int is not None:
            # Fallback über image_id → COCO-Dateiname konstruieren
            fname = _coco_filename_from_id(image_id_int, "train" if coco_sub == "train2014" else "val")
            cand = os.path.join(default_dir, fname)
            if os.path.exists(cand):
                img_path = cand

        if (not img_path or not os.path.exists(img_path)) and img_name:
            # Rekursive Suche als letzter Versuch
            alt = _search_recursively(images_root, os.path.basename(img_name))
            if alt:
                img_path = alt

        if require_image and (not img_path or not os.path.exists(img_path)):
            # Bild zwingend: überspringen
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
# ACT-X  (Top-Level = dict; answers = Label-String; image_name)
# -----------------------------------------------------------------------------

def load_actx(
    images_root: str,
    ann_root: str,
    split: str = "train",
    require_image: bool = True,
) -> List[ACTXExample]:
    split = split.lower()
    if split in ("val", "valid", "validation"):
        split = "test"  # es gibt bei dir kein val
    fname_map = {"train": "actX_train.json", "test": "actX_test.json"}
    if split not in fname_map:
        raise ValueError("ACT-X split muss 'train' | 'test' sein (val→test).")
    ann_path = os.path.join(ann_root, fname_map[split])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ACT-X nicht gefunden: {ann_path}")

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
# e-SNLI-VE  (Top-Level = Liste; answers = Label-String; image_name)
# -----------------------------------------------------------------------------

def load_esnlive(
    images_root: str,
    ann_root: str,
    split: str = "train",
    require_image: bool = True,
) -> List[ESNLIVEExample]:
    split = split.lower()
    if split in ("val", "valid", "validation"):
        split = "test"  # bei dir kein val
    fname_map = {"train": "esnlive_train.json", "test": "esnlive_test.json"}
    if split not in fname_map:
        raise ValueError("eSNLI-VE split muss 'train' | 'test' sein (val→test).")
    ann_path = os.path.join(ann_root, fname_map[split])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"eSNLI-VE nicht gefunden: {ann_path}")

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
            premise=None,  # in deiner Version nicht vorhanden
            hypothesis=str(ex.get("hypothesis", "")),
            label=label,
            explanation=_norm_expl(ex),
            sample_id=str(ex.get("image_name") or ex.get("id") or ""),
            raw=ex,
        ))
    return results

# -----------------------------------------------------------------------------
# VCR  (Top-Level = Liste; img_name = relativer Pfad unter vcr1images/)
# -----------------------------------------------------------------------------

def load_vcr(
    images_root: str,
    ann_root: str,
    split: str = "train",
    require_image: bool = True,
) -> List[VCRExample]:
    split = split.lower()
    fname_map = {"train": "vcr_train.json", "val": "vcr_val.json", "test": "vcr_test.json"}
    if split not in fname_map:
        raise ValueError("VCR split muss 'train' | 'val' | 'test' sein.")
    ann_path = os.path.join(ann_root, fname_map[split])
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"VCR nicht gefunden: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # list

    examples = data if isinstance(data, list) else (list(data.values()) if isinstance(data, dict) else [])
    results: List[VCRExample] = []
    vcr_root = os.path.join(images_root, "vcr1images")

    for ex in examples:
        img_name = _first_nonempty(ex.get("img_name"), ex.get("image"), ex.get("filename")) or ""
        img_rel = img_name.lstrip("/")
        if img_rel.startswith("vcr1images/"):
            img_rel = img_rel[len("vcr1images/"):]
        img_path = os.path.join(vcr_root, img_rel) if img_rel else ""

        if img_rel and not os.path.exists(img_path):
            # Fallback: nur Basename rekursiv suchen
            alt = _search_recursively(vcr_root, os.path.basename(img_rel))
            if alt:
                img_path = alt

        if require_image and (not img_path or not os.path.exists(img_path)):
            continue

        results.append(VCRExample(
            image_path=img_path if img_path else (os.path.join(vcr_root, img_rel) if img_rel else ""),
            question=ex.get("question"),
            answer=_first_nonempty(ex.get("answers"), ex.get("label")),
            choices=None,
            rationale=None,
            explanation=_norm_expl(ex),
            sample_id=str(ex.get("image_id") or ex.get("id") or ""),
            raw=ex,
        ))
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
    raise ValueError(f"Unbekannter Task: {task}")