# utils/run_vqa_tasks_noEx.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
from typing import Optional, List, Dict
from PIL import Image
import torch, copy

from utils.load_data import (
    load_vqax, load_actx, load_esnlive, load_vcr
)

# import unified answer-only prompting templates
from utils.prompting_templates import (
    prompt_vqax_answer_only,
    prompt_actx_answer_only,
    prompt_esnlive_answer_only,
    prompt_vcr_answer_only,
)

# -----------------------------
# Canonical task mapping
# -----------------------------
TASK_CANON = {
    "vqax": "VQA-X",
    "vqa-x": "VQA-X",
    "vqa": "VQA-X",
    "actx": "ACT-X",
    "act-x": "ACT-X",
    "esnlive": "ESNLI-VE",
    "esnli-ve": "ESNLI-VE",
    "esnli_ve": "ESNLI-VE",
    "vcr": "VCR",
}

# canonical task -> annotation folder name on disk (case-sensitive on Linux)
ANN_DIR_MAP = {
    "VQA-X":    "VQA-X",
    "ACT-X":    "ACT-X",
    "ESNLI-VE": "eSNLI-VE",   # lowercase 'e' on disk
    "VCR":      "VCR",
}

# -----------------------------
# Normalizers / helpers
# -----------------------------
_ARTICLES = {"a", "an", "the"}

def normalize_ans(s: str) -> str:
    """Lowercase, strip punctuation, remove articles."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)

def majority_vqa_answer(raw_answers):
    """Majority-vote over the 10 VQA answers."""
    import collections
    if not isinstance(raw_answers, list) or not raw_answers:
        return None
    texts = []
    for a in raw_answers:
        if isinstance(a, dict) and "answer" in a:
            texts.append(a["answer"])
        elif isinstance(a, str):
            texts.append(a)
    if not texts:
        return None
    cnt = collections.Counter([normalize_ans(t) for t in texts])
    return cnt.most_common(1)[0][0]

def _force_label_space(label: str) -> str:
    """Restrict ESNLI-VE labels to one of the 3 valid options."""
    l = (label or "").lower().strip()
    if "entail" in l:
        return "entailment"
    if "contradict" in l:
        return "contradiction"
    if "neutral" in l:
        return "neutral"
    return "unknown"

def _is_qwen_model(model) -> bool:
    return model.__class__.__name__.startswith("Qwen3VLForConditionalGeneration")

def _inject_image_into_messages(messages, pil_image: Image.Image):
    """
    Return a deep-copied messages list where the LAST {'type':'image'} placeholder
    is replaced by {'type':'image', 'image': pil_image}.
    """
    msgs = copy.deepcopy(messages)
    # Find last user turn and replace a placeholder image token
    for turn in reversed(msgs):
        if turn.get("role") == "user" and isinstance(turn.get("content"), list):
            for item in turn["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    item["image"] = pil_image
                    return msgs
    # If no placeholder found, append an image item to last user turn
    for turn in reversed(msgs):
        if turn.get("role") == "user" and isinstance(turn.get("content"), list):
            turn["content"].append({"type": "image", "image": pil_image})
            return msgs
    # Fallback: add a new user turn
    msgs.append({"role": "user", "content": [{"type": "image", "image": pil_image}]})
    return msgs

def _clean_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^(assistant:|response:|answer:|question:)\s*", "", t, flags=re.I)
    t = t.replace("\n", " ").strip()
    return t

def _answer_only_from_freeform(text: str, max_tokens: int = 3) -> str:
    """
    Heuristic: take only the first 1–3 tokens as the answer.
    Useful if the model ignores the 'answer only' instruction.
    """
    t = _clean_text(text)
    toks = t.split()
    if not toks:
        return ""
    return " ".join(toks[:max_tokens])

# -----------------------------
# VCR-specific helpers
# -----------------------------
_LETTER_TO_IDX = {"a": 0, "b": 1, "c": 2, "d": 3}

def _parse_vcr_letter(text: str) -> Optional[str]:
    """
    Parse model output like 'C', 'answer: B', etc.
    Returns 'a'/'b'/'c'/'d' or None.
    """
    t = _clean_text(text)
    toks = t.split()
    if not toks:
        return None
    cand = toks[0].lower().strip(":.")
    return cand if cand in _LETTER_TO_IDX else None

# -----------------------------
# Core generation
# -----------------------------
def generate_answer(
    model,
    processor,
    image_path: str,
    conversation,
    max_new_tokens: int = 20,
    device: str = "cuda",
) -> str:
    """
    Unified generator for LLaVA and Qwen3-VL.
    `conversation` is a list of chat turns as produced by your prompt builders.
    """
    img = Image.open(image_path).convert("RGB")

    if _is_qwen_model(model):
        # Qwen: image must be embedded into messages
        messages = _inject_image_into_messages(conversation, img)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
            )

        # trim prompt tokens
        gen_only = []
        if out.dim() == 2:
            seqs = [out[0]]
        else:
            seqs = list(out)
        for in_ids, out_ids in zip(inputs["input_ids"], seqs):
            gen_only.append(out_ids[len(in_ids):])
        text = processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return text

    else:
        # LLaVA: prompt + image separately
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device, torch.float16)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        gen_only = out[0, input_len:]
        text = processor.decode(gen_only, skip_special_tokens=True).strip()
        return text

# -----------------------------
# Main unified runner (answers only)
# -----------------------------
def run_vqa_task(
    task: str,
    model,
    processor,
    images_root: str,
    nle_root: str,
    split: str = "val",
    n_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Unified entry point for VQA-X, ACT-X, ESNLI-VE, VCR (answers only).
    Returns a list of dicts with keys:
      task, split, idx, question, prediction, ground_truth, image, correct
    """
    # canonicalize task
    key = TASK_CANON.get(task.replace(" ", "").lower())
    if not key:
        raise ValueError(f"Unknown task: {task}")
    task = key

    loader_map = {
        "VQA-X":    load_vqax,
        "ACT-X":    load_actx,
        "ESNLI-VE": load_esnlive,
        "VCR":      load_vcr,
    }

    ann_root_task = os.path.join(nle_root, ANN_DIR_MAP[task])
    dataset = loader_map[task](images_root, ann_root_task, split, require_image=True)
    if n_samples:
        dataset = dataset[:n_samples]

    print(f"Running {task} on {len(dataset)} samples...")

    results: List[Dict] = []

    for i, s in enumerate(dataset, 1):

        if task == "VQA-X":
            gt = majority_vqa_answer(s.raw.get("answers"))

            # unified answer-only prompt
            prompt = prompt_vqax_answer_only(s.question)
            raw_pred = generate_answer(model, processor, s.image_path, prompt)
            pred_ans = _answer_only_from_freeform(raw_pred, max_tokens=2)
            hit = int(normalize_ans(pred_ans) == normalize_ans(gt)) if gt else None

            results.append({
                "task": task,
                "split": split,
                "idx": i,
                "image": s.image_path,
                "question": s.question,
                "ground_truth": gt,
                "prediction": pred_ans,
                "correct": hit,
            })
            continue

        elif task == "ACT-X":
            gt = s.label

            # unified answer-only prompt
            prompt = prompt_actx_answer_only()
            raw_pred = generate_answer(model, processor, s.image_path, prompt)
            pred_ans = _answer_only_from_freeform(raw_pred, max_tokens=3)
            hit = int(normalize_ans(pred_ans) == normalize_ans(gt)) if gt else None

            results.append({
                "task": task,
                "split": split,
                "idx": i,
                "image": s.image_path,
                "question": None,
                "ground_truth": gt,
                "prediction": pred_ans,
                "correct": hit,
            })
            continue

        elif task == "ESNLI-VE":
            gt = s.label

            # unified answer-only prompt
            prompt = prompt_esnlive_answer_only(s.hypothesis)
            raw_pred = generate_answer(model, processor, s.image_path, prompt)
            label = _force_label_space(raw_pred)
            pred_ans = label
            hit = int(normalize_ans(label) == normalize_ans(gt)) if gt else None

            results.append({
                "task": task,
                "split": split,
                "idx": i,
                "image": s.image_path,
                "question": s.hypothesis,
                "ground_truth": gt,
                "prediction": pred_ans,
                "correct": hit,
            })
            continue

        elif task == "VCR":
            # Ground truth: label (0..3)
            gt_label = s.raw.get("label")
            if isinstance(gt_label, list):
                gt_label = gt_label[0]
            gt_label = int(gt_label)

            # 4 options
            choice_texts = s.choices or []
            if len(choice_texts) != 4:
                choice_texts = (choice_texts + ["Option missing"] * 4)[:4]

            # unified answer-only prompt (letter only)
            prompt = prompt_vcr_answer_only(s.question or "", choice_texts)

            raw_pred = generate_answer(model, processor, s.image_path, prompt)
            letter = _parse_vcr_letter(raw_pred)

            if letter is None:
                pred_label = None
            else:
                pred_label = _LETTER_TO_IDX.get(letter)

            hit = int(pred_label == gt_label)

            option_dict = {
                "0": choice_texts[0],
                "1": choice_texts[1],
                "2": choice_texts[2],
                "3": choice_texts[3],
            }

            results.append({
                "task": task,
                "split": split,
                "idx": i,
                "image": s.image_path,
                "question": s.question,
                "gt_answer": gt_label,
                "pred_answer": pred_label,
                "correct": hit,
                "options": option_dict,
            })
            continue

    # --- Accuracy summary ---
    valid_hits = [r["correct"] for r in results if r["correct"] is not None]
    if valid_hits:
        acc = sum(valid_hits) / len(valid_hits)
        print(f"\n{task} {split} Accuracy: {acc:.3f}")
    else:
        print(f"\n{task} {split}: keine evaluierbaren Ground-Truths gefunden.")

    return results