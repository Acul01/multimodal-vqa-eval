# utils/run_vqa_tasks.py
# -*- coding: utf-8 -*-
import os
import re
from typing import List, Dict, Optional
from PIL import Image
import torch
import torch.nn.functional as F
import copy

from utils.pixelshap_integration import run_pixelshap_for_token, run_pixelshap_for_image, VLMConfig 

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

# -----------------------------
# Normalizers / helpers
# -----------------------------

# einfache manuelle Stopword-Liste, wird für Entropie-Filterung genutzt
MANUAL_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "is", "are",
    "to", "and", "or", "because", "that", "this", "it",
    "they", "he", "she", "we", "you", "i", "them",
    "his", "her", "their", "there", "here",
}

def _clean_token_for_entropy(t: str) -> Optional[str]:
    """
    Normalise a decoded token for entropy logging:

    - strip whitespace
    - lowercase
    - drop if empty
    - drop if in MANUAL_STOPWORDS

    Wichtig: kein Regex-Strippen, damit Subwords (z.B. '▁skate') erhalten bleiben.
    """
    if not t:
        return None
    t = t.strip()
    if not t:
        return None

    t_low = t.lower()
    if t_low in MANUAL_STOPWORDS:
        return None

    return t_low


_ARTICLES = {"a", "an", "the"}

def normalize_ans(s: str) -> str:
    """Lowercase, Strip, Punctuation raus, Artikel entfernen."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


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


def _postprocess_pred_because_expl(text: str) -> str:
    """
    Normalisiere Modell-Output in das Format:
    '<prediction> because <explanation>' (lowercase).
    """
    t = (text or "").strip()
    # Prefixe wie "assistant:", "answer:" etc. entfernen
    t = re.sub(r'^(?:assistant:|response:|answer:|question:)\s*', "", t, flags=re.I)
    t = t.replace("\n", " ").strip()

    if "because" not in t.lower():
        toks = t.split()
        if toks and toks[0].lower() in ["entailment", "contradiction", "neutral"]:
            return f"{toks[0].lower()} because explanation missing"
        elif toks:
            return f"{toks[0].lower()} because explanation missing"
        else:
            return "unknown because explanation missing"

    parts = re.split(r"\bbecause\b", t, maxsplit=1, flags=re.I)
    label = parts[0].strip().lower() if len(parts) >= 1 else "unknown"
    explanation = parts[1].strip().rstrip(".").lower() if len(parts) >= 2 else "explanation missing"

    if not label:
        label = "unknown"
    if not explanation:
        explanation = "explanation missing"

    return f"{label} because {explanation}"


def _force_label_space(label: str) -> str:
    """Restrict ESNLI-VE label to one of the 3 valid options."""
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


# ---- VQA-X specific postprocessor ----
_BECAUSE_RE = re.compile(r"\bbecause\b", flags=re.I)

def _split_on_because(text: str):
    m = _BECAUSE_RE.search(text or "")
    if not m:
        return (text or "").strip(), ""
    return (text[:m.start()].strip(), text[m.end():].strip())


def _postprocess_vqax(text: str) -> str:
    """
    Normalize VQA-X to '<prediction> because <explanation>' (lowercase).
    - prediction: 1–2 words
    - wenn 'because' fehlt: prediction aus den ersten Tokens ableiten
    """
    t = (text or "").strip()
    t = re.sub(r'^(?:assistant:|response:|answer:|question:)\s*', "", t, flags=re.I)
    t = t.replace("\n", " ").strip()

    print(f"[DBG] generated_text: {t}")

    left, right = _split_on_because(t)

    if right == "":
        words = left.split()
        pred = " ".join(words[:2]).lower() if words else "unknown"
        expl = " ".join(words[2:]).lower() if len(words) > 2 else "no further details"
        return f"{pred} because {expl}"

    pred = re.sub(r"[^a-zA-Z0-9 ]+", " ", left).strip()
    pred = " ".join(pred.split()[:2]).lower() or "unknown"
    expl = right.strip().rstrip(".").lower() or "no further details"
    return f"{pred} because {expl}"


# -----------------------------
# VCR-specific helpers
# -----------------------------
_LETTER_TO_IDX = {"a": 0, "b": 1, "c": 2, "d": 3}

def _parse_vcr_letter_and_expl(text: str):
    """
    Parse model output like 'C because ...' or 'answer: B because ...'
    Returns (letter or None, explanation string).
    """
    t = (text or "").strip()
    t = re.sub(r'^(assistant:|response:|answer:)\s*', "", t, flags=re.I)
    t = t.replace("\n", " ").strip()

    tokens = t.split()
    letter = None
    if tokens:
        cand = tokens[0].lower().strip(":.")
        if cand in _LETTER_TO_IDX:
            letter = cand

    m = _BECAUSE_RE.search(t)
    if m:
        expl = t[m.end():].strip()
    else:
        expl = ""

    if not expl:
        expl = "no further details"

    return letter, expl


# -----------------------------
# Helper: extract token-wise entropies
# -----------------------------
def _extract_token_entropies(
    out,
    tokenizer,
    input_len: int,
) -> Dict[str, float]:
    """
    Berechne Entropie pro generiertem Token (nach dem Prompt).
    - Token werden mit _clean_token_for_entropy normalisiert
    - Stopwords (MANUAL_STOPWORDS) werden entfernt
    - Entropien auf 3 Nachkommastellen gerundet
    - wenn derselbe Token mehrfach vorkommt -> maximale Entropie behalten
    """
    token_entropy: Dict[str, float] = {}

    # Keine Scores -> keine Entropien
    if not hasattr(out, "scores") or out.scores is None or len(out.scores) == 0:
        return token_entropy

    sequences = out.sequences              # (batch, total_len)
    gen_ids = sequences[0, input_len:]     # nur generierte Tokens
    scores = out.scores                    # Liste von Logits pro Schritt
    T = min(len(gen_ids), len(scores))

    for t in range(T):
        logits_step = scores[t][0]         # (vocab_size,)
        probs = F.softmax(logits_step, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        H = float(-(probs * log_probs).sum().item())
        H_round = round(H, 3)

        tok_id = gen_ids[t].item()
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)

        tok_clean = _clean_token_for_entropy(tok_str)
        if tok_clean is None:
            continue

        if tok_clean in token_entropy:
            token_entropy[tok_clean] = max(token_entropy[tok_clean], H_round)
        else:
            token_entropy[tok_clean] = H_round

    return token_entropy


def _filter_entropy_to_explanation(
    token_entropy: Dict[str, float],
    explanation: str,
) -> Dict[str, float]:
    """
    Behalte nur Entropien für Tokens, die in der Explanation vorkommen
    (nach dem gleichen Cleaning wie in _clean_token_for_entropy).
    """
    if not explanation:
        return {}

    # rohe "Wörter" aus der Explanation holen
    words_raw = re.findall(r"[A-Za-z0-9]+", explanation)
    keep: set[str] = set()

    for w in words_raw:
        cleaned = _clean_token_for_entropy(w)
        if cleaned is not None:
            keep.add(cleaned)

    if not keep:
        return {}

    return {
        tok: H for tok, H in token_entropy.items()
        if tok in keep
    }


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

        token_entropy_all = _extract_token_entropies(
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

        token_entropy_all = _extract_token_entropies(
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
    pixel_shap=None,
    pixelshap_out_dir: Optional[str] = None,
    max_tokens_pixelshap: Optional[int] = 3,
):
    """
    Unified entry point for VQA-X, ACT-X, ESNLI-VE, VCR (answer+explanation).
    Uses prompting_templates.py for prompts.
    Optionally computes PixelSHAP overlays for explanation tokens.
    If max_tokens_pixelshap is None, uses all tokens from the explanation.
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

        pixelshap_paths = None  # will hold (token, overlay_path) tuples if used

        if task == "VQA-X":
            gt = majority_vqa_answer(s.raw.get("answers"))
            prompt = prompt_vqax_expl(s.question, prompt_mode)
            raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)
        
            pred_full = _postprocess_vqax(raw_pred)  # full answer: "<answer> because <explanation>"
            pred_only, _, expl = pred_full.partition(" because ")
        
            token_entropy = _filter_entropy_to_explanation(token_entropy_raw, expl)
        
            hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full
        
            # --------------------------------------------------------
            # PixelSHAP integration: per-image folder + meta.json
            # --------------------------------------------------------
            # Use token_entropy_raw (all tokens) instead of filtered token_entropy
            if pixel_shap is not None and pixelshap_out_dir is not None and token_entropy_raw:
        
                # sample index from the outer loop (for i, s in enumerate(..., 1))
                sample_idx = i
        
                # select tokens: all tokens if max_tokens_pixelshap is None, otherwise top-K
                # Use token_entropy_raw (all tokens from complete answer) for selection
                sorted_tokens = sorted(
                    token_entropy_raw.items(), key=lambda kv: kv[1], reverse=True
                )
                if max_tokens_pixelshap is None:
                    # Use all tokens from the complete answer
                    selected_tokens = [t for t, H in sorted_tokens]
                else:
                    # Use top-K tokens
                    selected_tokens = [t for t, H in sorted_tokens[:max_tokens_pixelshap]]
        
                # we use only the question as base prompt for PixelSHAP
                base_prompt_for_pixelshap = s.question
        
                img_base = os.path.splitext(os.path.basename(s.image_path))[0]
                img_id = getattr(s, "image_id", None)
        
                # create per-image directory
                if img_id is not None:
                    img_dir_name = f"{img_base}_id{img_id}"
                else:
                    img_dir_name = img_base
        
                img_out_dir = os.path.join(pixelshap_out_dir, img_dir_name)
                os.makedirs(img_out_dir, exist_ok=True)
        
                # meta.json with full answer + all tokens with entropy
                meta_path = os.path.join(img_out_dir, "meta.json")
                meta = {
                    "sample_index": sample_idx,
                    "image_path": s.image_path,
                    "image_id": img_id,
                    "question": s.question,
                    "model_answer": pred_full,          # full "<answer> because <expl>"
                    "ground_truth_answer": gt,
                    "all_tokens": list(token_entropy_raw.keys()),  # all tokens from complete answer
                    "token_entropy": token_entropy_raw,  # all tokens with entropy values (complete answer)
                    "explanation_tokens": list(token_entropy.keys()) if token_entropy else [],  # filtered tokens (explanation only)
                }
                try:
                    import json
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[WARN] Could not write meta.json: {e}")
        
                # Create one overlay per image using the most important token (highest entropy)
                # This represents the overall explanation for the image
                pixelshap_paths = []
                if selected_tokens:
                    # Use the token with highest entropy for the overlay
                    most_important_token = selected_tokens[0]
                    try:
                        # Determine device from model
                        device = next(model.parameters()).device
                        device_str = "cuda" if device.type == "cuda" else "cpu"
                        
                        # Create VLMConfig for PixelSHAP
                        vlm_cfg = VLMConfig(
                            model=model,
                            processor=processor,
                            device=device_str,
                            max_new_tokens=40,
                            task=task,
                        )
                        
                        # Get temp_dir from pixel_shap if available, otherwise use default
                        temp_dir = getattr(pixel_shap, 'temp_dir', 'pixelshap_tmp') if pixel_shap else 'pixelshap_tmp'
                        
                        # Pass complete generated answer to build_segmentation_model
                        out_path = run_pixelshap_for_image(
                            vlm_cfg=vlm_cfg,
                            generated_answer=pred_full,  # complete answer: "<answer> because <explanation>"
                            image_path=s.image_path,
                            base_prompt=base_prompt_for_pixelshap,
                            token=most_important_token,
                            out_dir=img_out_dir,
                            image_id=img_id,
                            question=s.question,
                            model_answer=pred_full,
                            gt_answer=gt,
                            temp_dir=temp_dir,
                        )
                        pixelshap_paths.append((most_important_token, out_path))
                    except Exception as e:
                        print(
                            f"[WARN] PixelSHAP failed for sample {sample_idx}, "
                            f"image {s.image_path}, token '{most_important_token}': {e}"
                        )
                
        elif task == "ACT-X":
            gt = s.label
            prompt = prompt_actx_expl(prompt_mode)
            raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)

            pred_full = _postprocess_pred_because_expl(raw_pred)
            pred_only, _, expl = pred_full.partition(" because ")

            token_entropy = _filter_entropy_to_explanation(token_entropy_raw, expl)

            hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

            if pixel_shap is not None and pixelshap_out_dir is not None and token_entropy:
                sorted_tokens = sorted(
                    token_entropy.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                top_tokens = [t for t, H in sorted_tokens[:max_tokens_pixelshap]]

                base_prompt_for_pixelshap = "Describe the human activity in this image."

                pixelshap_paths = []
                for tok in top_tokens:
                    out_path = run_pixelshap_for_token(
                        pixel_shap=pixel_shap,
                        image_path=s.image_path,
                        base_prompt=base_prompt_for_pixelshap,
                        token=tok,
                        out_dir=pixelshap_out_dir,
                    )
                    pixelshap_paths.append((tok, out_path))

        elif task == "ESNLI-VE":
            gt = s.label
            prompt = prompt_esnlive_expl(s.hypothesis, prompt_mode)
            raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)

            pred_full = raw_pred.strip()
            if "because" not in pred_full.lower():
                pred_full = pred_full.replace(",", " because ", 1)

            parts = re.split(r"\bbecause\b", pred_full, maxsplit=1, flags=re.I)
            label = parts[0].strip().lower()
            explanation = parts[1].strip().lower() if len(parts) > 1 else "explanation missing"

            label = _force_label_space(label)
            pred_full = f"{label} because {explanation}"

            token_entropy = _filter_entropy_to_explanation(token_entropy_raw, explanation)

            hit = int(normalize_ans(label) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

            if pixel_shap is not None and pixelshap_out_dir is not None and token_entropy:
                sorted_tokens = sorted(
                    token_entropy.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                top_tokens = [t for t, H in sorted_tokens[:max_tokens_pixelshap]]

                base_prompt_for_pixelshap = s.hypothesis

                pixelshap_paths = []
                for tok in top_tokens:
                    out_path = run_pixelshap_for_token(
                        pixel_shap=pixel_shap,
                        image_path=s.image_path,
                        base_prompt=base_prompt_for_pixelshap,
                        token=tok,
                        out_dir=pixelshap_out_dir,
                    )
                    pixelshap_paths.append((tok, out_path))

        elif task == "VCR":
            choices = s.choices or []
            gt = s.answer or ""
            prompt = prompt_vcr_expl(s.question or "", choices, prompt_mode)

            raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)
            letter, expl = _parse_vcr_letter_and_expl(raw_pred)

            token_entropy = _filter_entropy_to_explanation(token_entropy_raw, expl)

            if letter is not None:
                idx = _LETTER_TO_IDX.get(letter)
            else:
                idx = None

            if idx is not None and 0 <= idx < len(choices):
                pred_answer_text = choices[idx]
            else:
                pred_answer_text = "unknown"

            pred_full = f"{pred_answer_text.lower()} because {expl.lower()}"
            hit = int(normalize_ans(pred_answer_text) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

            if pixel_shap is not None and pixelshap_out_dir is not None and token_entropy:
                sorted_tokens = sorted(
                    token_entropy.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                top_tokens = [t for t, H in sorted_tokens[:max_tokens_pixelshap]]

                base_prompt_for_pixelshap = s.question or ""

                pixelshap_paths = []
                for tok in top_tokens:
                    out_path = run_pixelshap_for_token(
                        pixel_shap=pixel_shap,
                        image_path=s.image_path,
                        base_prompt=base_prompt_for_pixelshap,
                        token=tok,
                        out_dir=pixelshap_out_dir,
                    )
                    pixelshap_paths.append((tok, out_path))

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
            "pixelshap_overlays": pixelshap_paths,
        })

    valid_hits = [r["correct"] for r in results if r["correct"] is not None]
    if valid_hits:
        acc = sum(valid_hits) / len(valid_hits)
        print(f"\n{task} {split} Accuracy ({prompt_mode}): {acc:.3f}")
    else:
        print(f"\n{task} {split} ({prompt_mode}): no evaluable ground truths found.")

    return results