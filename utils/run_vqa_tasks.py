# utils/run_vqa_tasks.py
# -*- coding: utf-8 -*-
import os
import re
from typing import List, Dict, Optional
from PIL import Image
import torch, copy
import torch.nn.functional as F

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
    "VCR":      "VCR",
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
    # remove prefixes like "assistant:", "answer:", ...
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
    """Restrict label to one of the 3 valid options."""
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
    - prediction: 1–2 words, no punctuation
    - if 'because' missing: derive prediction from first 1–2 tokens
    """
    t = (text or "").strip()
    t = re.sub(r'^(?:assistant:|response:|answer:|question:)\s*', '', t, flags=re.I)
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
# Prompt builders (with modes)
# -----------------------------
def _resolve_shot_count(prompt_mode: str) -> int:
    """
    Map prompt_mode to number of few-shot examples:
      'zero'  -> 0
      '1shot' -> 1
      '3shot' -> 3
    """
    m = (prompt_mode or "zero").lower()
    if m.startswith("1"):
        return 1
    if m.startswith("3"):
        return 3
    return 0

def build_prompt_vqax(question: str, prompt_mode: str = "zero"):
    fewshot_examples = [
        {
            "q": "What is the man holding?",
            "a": "guitar because he is holding a stringed instrument across his body",
        },
        {
            "q": "What color is the bus?",
            "a": "yellow because the vehicle is painted bright yellow",
        },
        {
            "q": "What is the woman doing?",
            "a": "cooking because she is standing in front of a stove with pans",
        },
    ]
    k = _resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    for ex in fewshot_examples[:k]:
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": f"Question: {ex['q']}"}],
        })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ex["a"]}],
        })

    instructions = (
        "You see an IMAGE and a QUESTION.\n"
        "Answer the question in exactly one sentence using this format:\n"
        "<answer> because <explanation>\n"
        f"Question: {question}"
    )
    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": instructions},
            {"type": "image"},
        ],
    })
    return conversation

def build_prompt_actx(_unused: str = "", prompt_mode: str = "zero"):
    fewshot_examples = [
        {
            "d": "A person is moving quickly on a red athletics track.",
            "a": "running because the person is moving fast along the track",
        },
        {
            "d": "A woman is standing at a stove holding a pan with steam rising.",
            "a": "cooking because she is preparing food on the stove",
        },
        {
            "d": "A man is sitting on a chair holding an acoustic guitar.",
            "a": "playing guitar because he is holding and strumming a guitar",
        },
    ]

    k = _resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    for ex in fewshot_examples[:k]:
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Image description: {ex['d']}"},
            ],
        })
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": ex["a"]},
            ],
        })

    instructions = (
        "You will be shown an IMAGE.\n"
        "Identify the activity shown in the image and answer in EXACTLY this format:\n"
        "<activity> because <explanation>\n"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": instructions},
            {"type": "image"},
        ],
    })

    return conversation

def build_prompt_esnlive(hypothesis: str, prompt_mode: str = "zero"):
    intro = (
        "You are a visual entailment assistant. "
        "Given an IMAGE and a HYPOTHESIS, decide the relationship and explain briefly. "
        "Answer in EXACTLY this format: <label> because <explanation>. "
        "The <label> must be one of: entailment, contradiction, or neutral. "
        "Use 'because' exactly once, explanation <= 15 words."
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
    k = _resolve_shot_count(prompt_mode)

    conversation: List[Dict] = []

    for ex in fewshot_examples[:k]:
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

def build_prompt_vcr(question: str, choices: List[str], prompt_mode: str = "zero"):
    pad_texts = ["Option missing"] * max(0, 4 - len(choices))
    opts = (choices[:4] + pad_texts)[:4]
    k = _resolve_shot_count(prompt_mode)

    fewshot_examples = [
        {
            "q": "Why is the man holding the microphone?",
            "opts": ["He is singing", "He is painting", "He is sleeping", "He is cooking"],
            "a": "A because he appears to be performing or singing on stage",
        },
        {
            "q": "What are the people likely doing?",
            "opts": ["Watching a movie", "Riding bicycles", "Swimming", "Sleeping"],
            "a": "A because they are seated and facing a large screen",
        },
        {
            "q": "Why is the woman wearing a helmet?",
            "opts": ["She is biking", "She is reading", "She is cooking", "She is sleeping"],
            "a": "A because she is on a bicycle outdoors",
        },
    ]

    conversation: List[Dict] = []

    for ex in fewshot_examples[:k]:
        q_block = (
            f"Question: {ex['q']}\n\n"
            "Options:\n"
            f"A) {ex['opts'][0]}\n"
            f"B) {ex['opts'][1]}\n"
            f"C) {ex['opts'][2]}\n"
            f"D) {ex['opts'][3]}"
        )
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": q_block}],
        })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ex["a"]}],
        })

    instructions = (
        "You are a visual commonsense reasoning assistant.\n"
        "You will be given an IMAGE, a QUESTION about the scene, and four answer options.\n"
        "Choose the BEST answer option.\n\n"
        "Respond in exactly this format:\n"
        "<letter> because <explanation>\n"
        "where <letter> is one of: A, B, C, or D.\n"
        "Keep the explanation short (<= 20 words).\n"
        "Do NOT repeat the question."
    )

    question_block = (
        f"Question: {question}\n\n"
        "Options:\n"
        f"A) {opts[0]}\n"
        f"B) {opts[1]}\n"
        f"C) {opts[2]}\n"
        f"D) {opts[3]}"
    )

    full_text = instructions + "\n\n" + question_block

    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": full_text},
            {"type": "image"},
        ],
    })

    return conversation

PROMPT_BUILDERS = {
    "VQA-X":    build_prompt_vqax,
    "ACT-X":    build_prompt_actx,
    "ESNLI-VE": build_prompt_esnlive,
    # VCR uses build_prompt_vcr(question, choices, prompt_mode) directly
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
def _extract_token_entropies(out, tokenizer, input_len: int) -> Dict[str, float]:
    """
    Given a `generate` output with scores, compute per-step entropy
    for the generated tokens (after the prompt).
    Returns dict: "000:<token>" -> entropy_value, ...
    """
    token_entropy: Dict[str, float] = {}

    if not hasattr(out, "scores") or out.scores is None or len(out.scores) == 0:
        return token_entropy

    sequences = out.sequences  # (batch, total_len)
    gen_ids = sequences[0, input_len:]  # generated tokens only
    scores = out.scores           # list length = #generated steps

    # align generated tokens with scores (take min length just to be safe)
    T = min(len(gen_ids), len(scores))

    for t in range(T):
        logits_step = scores[t][0]             # (vocab_size,)
        probs = F.softmax(logits_step, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        H = float(-torch.sum(probs * log_probs).item())

        tok_id = gen_ids[t].item()
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
        key = f"{t:03d}:{tok_str}"
        token_entropy[key] = H

    return token_entropy

# -----------------------------
# Core generation (now returns text + token_entropy)
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
      text: decoded generated string
      token_entropy: dict { "000:<token>": entropy, ... }
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
                eos_token_id=getattr(processor, "eos_token_id", None)
                if hasattr(processor, "eos_token_id") else None,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequences = out.sequences
        input_len = inputs["input_ids"].shape[1]
        gen_ids = sequences[0, input_len:]
        text = processor.batch_decode(
            [gen_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        token_entropy = _extract_token_entropies(out, tokenizer, input_len)
        return text, token_entropy

    else:
        # LLaVA branch
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

        sequences = out.sequences
        gen_ids = sequences[0, input_len:]
        text = processor.decode(gen_ids, skip_special_tokens=True).strip()

        tokenizer = processor.tokenizer
        token_entropy = _extract_token_entropies(out, tokenizer, input_len)
        return text, token_entropy

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
    Adds 'token_entropy' dict to each result row.
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
            prompt = build_prompt_vqax(s.question, prompt_mode=prompt_mode)
            raw_pred, token_entropy = generate_answer(model, processor, s.image_path, prompt)
            pred_full = _postprocess_vqax(raw_pred)
            pred_only, _, _ = pred_full.partition(" because ")
            hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        elif task == "ACT-X":
            gt = s.label
            prompt = build_prompt_actx("", prompt_mode=prompt_mode)
            raw_pred, token_entropy = generate_answer(model, processor, s.image_path, prompt)
            pred_full = _postprocess_pred_because_expl(raw_pred)
            pred_only, _, _ = pred_full.partition(" because ")
            hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
            pred_to_store = pred_full

        elif task == "ESNLI-VE":
            gt = s.label
            prompt = build_prompt_esnlive(s.hypothesis, prompt_mode=prompt_mode)
            raw_pred, token_entropy = generate_answer(model, processor, s.image_path, prompt)
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
            choices = s.choices or []
            gt = s.answer or ""
            prompt = build_prompt_vcr(s.question or "", choices, prompt_mode=prompt_mode)

            raw_pred, token_entropy = generate_answer(model, processor, s.image_path, prompt)
            letter, expl = _parse_vcr_letter_and_expl(raw_pred)

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
            "token_entropy": token_entropy,   # NEW: dict token -> entropy
        })

    valid_hits = [r["correct"] for r in results if r["correct"] is not None]
    if valid_hits:
        acc = sum(valid_hits) / len(valid_hits)
        print(f"\n{task} {split} Accuracy ({prompt_mode}): {acc:.3f}")
    else:
        print(f"\n{task} {split} ({prompt_mode}): keine evaluierbaren Ground-Truths gefunden.")

    return results