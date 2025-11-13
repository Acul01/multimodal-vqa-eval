import os
import re
from typing import List, Dict, Optional
from PIL import Image
import torch, copy

from utils.load_data import (
    load_vqax, load_actx, load_esnlive, load_vcr
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
    "VCR":      "VCR", # <- NEU: echter VCR-Ordner
}

# -----------------------------
# Normalizers / helpers
# -----------------------------
_ARTICLES = {"a", "an", "the"}

def normalize_ans(s: str):
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)

def majority_vqa_answer(raw_answers):
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
    Normalize model output into the canonical format:
    '<prediction> because <explanation>' (lowercase).
    Only fixes minimal structure, keeps explanation text intact if available.
    """
    t = (text or "").strip()
    t = re.sub(r'^(?:assistant:|response:|answer:|question:)\s*', '', t, flags=re.I)
    t = t.replace("\n", " ").strip()

    if " because " not in t.lower():
        toks = t.split()
        if toks and toks[0].lower() in ["entailment", "contradiction", "neutral"]:
            return f"{toks[0].lower()} because explanation missing"
        else:
            return "unknown because explanation missing"

    label, _, explanation = re.split(r"\bbecause\b", t, maxsplit=1, flags=re.I)
    label = label.lower().strip().split()[0] if label else "unknown"
    explanation = explanation.strip().rstrip(".").lower()
    if not explanation:
        explanation = "explanation missing"
    return f"{label} because {explanation}"

def _force_label_space(label: str) -> str:
    """Restrict label to one of the 3 valid options."""
    l = label.lower().strip()
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
    - prediction: 1–2 words, no punctuation
    - if 'because' missing: split heuristically (first 1–2 words = prediction)
    """
    t = (text or "").strip()
    t = re.sub(r'^(?:assistant:|response:|answer:|question:)\s*', '', t, flags=re.I)
    t = t.replace("\n", " ").strip()

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
# Prompt builders
# -----------------------------
POSTPROCESSORS = {
    "VQA-X": _postprocess_vqax,
    "ACT-X": _postprocess_pred_because_expl,
    "ESNLI-VE": _postprocess_pred_because_expl,
    "VCR": _postprocess_pred_because_expl,   # für generische Fälle, VCR macht unten eigenes Handling
}

def build_prompt_vqax(question: str):
    instructions = (
        "You are a visual question answering assistant.\n"
        "Answer in EXACTLY one sentence using this strict format:\n"
        "<prediction> because <explanation>\n"
        "- <prediction> = first 1 or 2 words ONLY (no punctuation)\n"
        "- use lowercase 'because' exactly once as the separator\n"
        "- keep the explanation short (<= 20 words)\n"
        "Do NOT add any extra words before or after the sentence.\n"
        f"Question: {question}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": instructions},
            {"type": "image"},
        ],
    }]

def build_prompt_actx(_unused: str = ""):
    instructions = (
        "You are an activity recognition assistant.\n"
        "Answer in EXACTLY one sentence using this strict format:\n"
        "<activity> because <explanation>\n"
        "- <activity> = first 1 or 2 words ONLY (no punctuation)\n"
        "- separator must be exactly 'because' (lowercase)\n"
        "- keep the explanation short (<= 20 words)\n"
        "Do NOT add any extra words before or after the sentence."
    )
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": instructions},
            {"type": "image"},
        ],
    }]

def build_prompt_esnlive(hypothesis: str):
    """
    Few-shot prompt for e-SNLI-VE.
    The few-shot examples are text-only; the actual sample includes one image.
    """
    intro = (
        "You are a visual entailment assistant. "
        "Given an IMAGE and a HYPOTHESIS, decide the relationship and explain briefly. "
        "Answer in EXACTLY this format: <label> because <explanation>. "
        "The <label> must be one of: entailment, contradiction, or neutral. "
        "Use 'because' exactly once, and keep the explanation under 15 words. "
        "Do not add extra words or restate the hypothesis."
    )

    fewshot_examples = [
        {
            "user": "Hypothesis: A person is skiing down a snowy hill.",
            "assistant": "entailment because the image shows a person skiing on snow",
        },
        {
            "user": "Hypothesis: The man is riding a bicycle indoors.",
            "assistant": "contradiction because the bicycle is being ridden outside on a street",
        },
        {
            "user": "Hypothesis: Someone might be preparing a meal.",
            "assistant": "neutral because the kitchen scene could involve cooking or cleaning",
        },
    ]

    conversation = []
    for ex in fewshot_examples:
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": ex["user"]}],
        })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ex["assistant"]}],
        })

    conversation.append({
        "role": "user",
        "content": [
            {"type": "text",
             "text": f"{intro}\nNow analyze the following sample:\nHypothesis: {hypothesis}"},
            {"type": "image"},
        ],
    })
    return conversation

def build_prompt_vcr(question: str, choices: List[str]):
    """
    Multi-choice prompt for original VCR (4 answer choices).
    The model is asked to pick A/B/C/D and may optionally give an explanation.
    """
    # Ensure we have exactly 4 options (pad if dataset is weird)
    pad_texts = ["Option missing"] * max(0, 4 - len(choices))
    opts = (choices[:4] + pad_texts)[:4]

    instructions = (
        "You are a visual commonsense reasoning assistant.\n"
        "You will be given an IMAGE, a QUESTION about the scene, and four answer options.\n"
        "Choose the BEST answer option.\n\n"
        "Respond in exactly this format:\n"
        "<letter> because <explanation>\n"
        "where <letter> is one of: A, B, C, or D.\n"
        "Keep the explanation short (<= 20 words).\n"
        "Do NOT repeat the question.\n"
    )

    question_block = (
        f"Question: {question}\n\n"
        "Options:\n"
        f"A) {opts[0]}\n"
        f"B) {opts[1]}\n"
        f"C) {opts[2]}\n"
        f"D) {opts[3]}"
    )

    full_text = instructions + "\n" + question_block

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_text},
                {"type": "image"},
            ],
        }
    ]

PROMPT_BUILDERS = {
    "VQA-X":    build_prompt_vqax,
    "ACT-X":    build_prompt_actx,
    "ESNLI-VE": build_prompt_esnlive,
    # VCR nutzt build_prompt_vcr(question, choices) direkt im Runner
}

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
    t = re.sub(r'^(assistant:|response:|answer:)\s*', '', t, flags=re.I)
    t = t.replace("\n", " ").strip()

    # Find a/b/c/d as first meaningful token
    tokens = t.split()
    letter = None
    if tokens:
        cand = tokens[0].lower().strip(":.")
        if cand in _LETTER_TO_IDX:
            letter = cand

    # Extract explanation if 'because' present
    m = _BECAUSE_RE.search(t)
    if m:
        expl = t[m.end():].strip()
    else:
        expl = ""

    if not expl:
        expl = "no further details"

    return letter, expl

# -----------------------------
# Core generation
# -----------------------------
def generate_answer(model, processor, image_path: str, conversation, max_new_tokens: int = 20, device="cuda"):
    """
    Unified generator for LLaVA and Qwen3-VL.
    `conversation` is a list of chat turns as produced by your prompt builders.
    """
    img = Image.open(image_path).convert("RGB")

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

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None,
                pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None,
            )

        gen_only = []
        if out.dim() == 2:
            seqs = [out[0]]
        else:
            seqs = list(out)
        for in_ids, out_ids in zip(inputs["input_ids"], seqs):
            gen_only.append(out_ids[len(in_ids):])
        text = processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return text

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
            )
        gen_only = out[0, input_len:]
        text = processor.decode(gen_only, skip_special_tokens=True).strip()
        return text

# -----------------------------
# Main unified runner
# -----------------------------
def run_vqa_task(task: str, model, processor, images_root: str, nle_root: str,
                 split: str = "val", n_samples: Optional[int] = None):
    """Unified entry point for VQA-X, ACT-X, ESNLI-VE, VCR."""
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

    results = []
    for i, s in enumerate(dataset, 1):

        if task == "VQA-X":
            gt = majority_vqa_answer(s.raw.get("answers"))
            prompt = build_prompt_vqax(s.question)
            raw_pred = generate_answer(model, processor, s.image_path, prompt)
            pred_full = _postprocess_vqax(raw_pred)
            pred_only, _, _ = pred_full.partition(" because ")
            hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        elif task == "ACT-X":
            gt = s.label
            prompt = build_prompt_actx("")
            raw_pred = generate_answer(model, processor, s.image_path, prompt)
            pred_full = _postprocess_pred_because_expl(raw_pred)
            pred_only, _, _ = pred_full.partition(" because ")
            hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        elif task == "ESNLI-VE":
            gt = s.label
            prompt = build_prompt_esnlive(s.hypothesis)
            raw_pred = generate_answer(model, processor, s.image_path, prompt)
            pred_full = raw_pred.strip()

            if "because" not in pred_full.lower():
                pred_full = pred_full.replace(",", " because ", 1)
            parts = re.split(r"\bbecause\b", pred_full, maxsplit=1, flags=re.I)
            label = parts[0].strip().lower()
            explanation = parts[1].strip().lower() if len(parts) > 1 else "explanation missing"

            label = _force_label_space(label)
            pred_full = f"{label} because {explanation}"

            hit = int(normalize_ans(label) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        elif task == "VCR":
            # Multi-choice: s.choices = list of 4 strings; s.answer = correct choice text
            choices = s.choices or []
            gt = s.answer or ""
            prompt = build_prompt_vcr(s.question or "", choices)

            raw_pred = generate_answer(model, processor, s.image_path, prompt)
            letter, expl = _parse_vcr_letter_and_expl(raw_pred)

            if letter is not None:
                idx = _LETTER_TO_IDX.get(letter)
            else:
                idx = None

            if idx is not None and 0 <= idx < len(choices):
                pred_answer_text = choices[idx]
            else:
                pred_answer_text = "unknown"

            # store in unified "<answer> because <explanation>" Format
            pred_full = f"{pred_answer_text.lower()} because {expl.lower()}"
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
        })

    valid_hits = [r["correct"] for r in results if r["correct"] is not None]
    if valid_hits:
        acc = sum(valid_hits) / len(valid_hits)
        print(f"\n{task} {split} Accuracy: {acc:.3f}")
    else:
        print(f"\n{task} {split}: keine evaluierbaren Ground-Truths gefunden.")

    return results