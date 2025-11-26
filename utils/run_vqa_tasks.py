# utils/run_vqa_tasks.py
# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Optional
from PIL import Image
import torch
import copy

from utils.load_data import (
    load_vqax, load_actx, load_esnlive, load_vcr
)

# Prompts (Answer + Explanation) aus utils/prompting_templates
from utils.prompting_templates import (
    prompt_vqax_expl,
    prompt_actx_expl,
    prompt_esnlive_expl,
    prompt_vcr_expl,
)

# Unified postprocessing
from utils.postprocessing import (
    postprocess_prediction,
    normalize_answer as normalize_ans,
)

# Token entropy functions
from utils.token_entropies import (
    extract_token_entropies,
    filter_entropy_to_explanation,
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
    "ESNLI-VE": "eSNLI-VE",     # lowercase 'e' on disk
    "VCR":      "VCR",
}


def majority_vqa_answer(raw_answers):
    """Mehrheitsantwort aus den 10 VQA-Answers."""
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


def _is_qwen_model(model) -> bool:
    return model.__class__.__name__.startswith("Qwen3VLForConditionalGeneration")


def _inject_image_into_messages(messages, pil_image: Image.Image):
    """
    Return a deep-copied messages list where the LAST {'type':'image'} placeholder
    is replaced by {'type':'image', 'image': pil_image}.
    """
    msgs = copy.deepcopy(messages)
    for turn in reversed(msgs):
        if turn.get("role") == "user" and isinstance(turn.get("content"), list):
            for item in turn["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    item["image"] = pil_image
                    return msgs
    for turn in reversed(msgs):
        if turn.get("role") == "user" and isinstance(turn.get("content"), list):
            turn["content"].append({"type": "image", "image": pil_image})
            return msgs
    msgs.append({"role": "user", "content": [{"type": "image", "image": pil_image}]})
    return msgs


# -----------------------------
# Core generation (returns text + token_entropy)
# -----------------------------
def generate_answer(
    model,
    processor,
    image_path: str,
    conversation,
    max_new_tokens: int = 40,
    device: str = "cuda",
):
    """
    Unified generator for LLaVA and Qwen3-VL.
    Returns:
        text (str), token_entropy (Dict[str, float])
    """
    img = Image.open(image_path).convert("RGB")

    # --------------------------------------------------
    # Qwen3-VL-Zweig
    # --------------------------------------------------
    if _is_qwen_model(model):
        messages = _inject_image_into_messages(conversation, img)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                eos_token_id=getattr(processor, "eos_token_id", None)
                if hasattr(processor, "eos_token_id") else None,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Nur generierten Teil dekodieren
        gen_only = []
        if out.sequences.dim() == 2:
            seqs = [out.sequences[0]]
        else:
            seqs = list(out.sequences)
        for in_ids, out_ids in zip(inputs["input_ids"], seqs):
            gen_only.append(out_ids[len(in_ids):])

        text = processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all

    # --------------------------------------------------
    # LLaVA-Zweig
    # --------------------------------------------------
    else:
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
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_only = out.sequences[0, input_len:]
        text = processor.decode(gen_only, skip_special_tokens=True).strip()

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all


# -----------------------------
# Main unified runner
# -----------------------------
def run_vqa_task(
    task: str,
    model,
    processor,
    images_root: str,
    nle_root: str,
    split: str = "val",
    n_samples: Optional[int] = None,
    prompt_mode: str = "zero",
):
    """
    Unified entry point for VQA-X, ACT-X, ESNLI-VE, VCR (answer+explanation).
    """
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

    print(f"Running {task} on {len(dataset)} samples with prompt_mode='{prompt_mode}'...")

    results = []
    for i, s in enumerate(dataset, 1):

        if task == "VQA-X":
            gt = majority_vqa_answer(s.raw.get("answers"))
            prompt = prompt_vqax_expl(s.question, prompt_mode)
            raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)

            result = postprocess_prediction(raw_pred, "VQA-X")
            pred_full = result["full_text"]
            pred_only = result["answer"]
            expl = result["explanation"]

            token_entropy = filter_entropy_to_explanation(token_entropy_raw, expl)

            hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        elif task == "ACT-X":
            gt = s.label
            prompt = prompt_actx_expl(prompt_mode)
            raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)

            result = postprocess_prediction(raw_pred, "ACT-X")
            pred_full = result["full_text"]
            pred_only = result["answer"]
            expl = result["explanation"]

            token_entropy = filter_entropy_to_explanation(token_entropy_raw, expl)

            hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        elif task == "ESNLI-VE":
            gt = s.label
            prompt = prompt_esnlive_expl(s.hypothesis, prompt_mode)
            raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)

            result = postprocess_prediction(raw_pred, "ESNLI-VE")
            pred_full = result["full_text"]
            label = result["answer"]
            explanation = result["explanation"]

            token_entropy = filter_entropy_to_explanation(token_entropy_raw, explanation)

            hit = int(normalize_ans(label) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        elif task == "VCR":
            choices = s.choices or []
            gt = s.answer or ""
            prompt = prompt_vcr_expl(s.question or "", choices, prompt_mode)

            raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)
            result = postprocess_prediction(raw_pred, "VCR", vcr_choices=choices)
            
            pred_full = result["full_text"]
            pred_answer_text = result["answer"]
            expl = result["explanation"]

            token_entropy = filter_entropy_to_explanation(token_entropy_raw, expl)

            hit = int(normalize_ans(pred_answer_text) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        else:
            raise ValueError(f"Unsupported task: {task}")

        results.append({
            "task": task,
            "split": split,
            "idx": i,
            "question": getattr(s, "question", None) or getattr(s, "hypothesis", None),
            "prediction": pred_to_store,
            "ground_truth": gt,
            "image": s.image_path,
            "correct": hit,
            "prompt_mode": prompt_mode,
            "token_entropy": token_entropy,
        })

    valid_hits = [r["correct"] for r in results if r["correct"] is not None]
    if valid_hits:
        acc = sum(valid_hits) / len(valid_hits)
        print(f"\n{task} {split} Accuracy ({prompt_mode}): {acc:.3f}")
    else:
        print(f"\n{task} {split} ({prompt_mode}): keine evaluierbaren Ground-Truths gefunden.")

    return results