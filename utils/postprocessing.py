# utils/postprocessing.py
# -*- coding: utf-8 -*-
"""
Unified postprocessing functions for all VQA tasks.
Handles normalization, splitting into answer/explanation, and task-specific logic.
"""
import re
from typing import Tuple, Optional, Dict, Literal

# Task type constants
TaskType = Literal["VQA-X", "ACT-X", "ESNLI-VE", "VCR"]

# Constants
_ARTICLES = {"a", "an", "the"}
_BECAUSE_RE = re.compile(r"\bbecause\b", flags=re.I)
_LETTER_TO_IDX = {"a": 0, "b": 1, "c": 2, "d": 3}

# Export for use in other modules
LETTER_TO_IDX = _LETTER_TO_IDX


def normalize_generated_text(text: str) -> str:
    """
    Normalize generated text:
    - Remove prefixes (assistant:, response:, answer:, question:)
    - Lowercase
    - Remove non-alphanumeric characters (except spaces)
    - Normalize multiple spaces
    - Remove stopwords (answer, question, explanation)
    """
    t = (text or "").strip()

    # Remove prefixes
    t = re.sub(r'^(?:assistant:|response:|answer:|question:)\s*', "", t, flags=re.I)

    # Lowercase
    t = t.lower()

    # Remove non-alphanumeric (except spaces)
    t = re.sub(r"[^a-z0-9\s]+", " ", t)

    # Normalize multiple spaces
    t = re.sub(r"\s+", " ", t).strip()

    # Remove stopwords
    remove_words = {"answer", "question", "explanation"}
    toks = [w for w in t.split() if w not in remove_words]

    return " ".join(toks).strip()


def normalize_answer(s: str) -> str:
    """
    Normalize answer for comparison:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    """
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def _force_label_space(label: str) -> str:
    """
    Force e-SNLI-VE label to one of: entailment, contradiction, neutral.
    """
    l = (label or "").lower().strip()
    if "entail" in l:
        return "entailment"
    if "contradict" in l:
        return "contradiction"
    if "neutral" in l:
        return "neutral"
    return "unknown"


def _parse_vcr_letter(text: str) -> Optional[str]:
    """
    Parse VCR letter (A, B, C, or D) from text.
    Returns the letter if found, None otherwise.
    """
    t = (text or "").strip()
    t = re.sub(r'^(assistant:|response:|answer:)\s*', "", t, flags=re.I)
    t = t.replace("\n", " ").strip()

    tokens = t.split()
    if tokens:
        cand = tokens[0].lower().strip(":.")
        if cand in _LETTER_TO_IDX:
            return cand
    return None


def _is_yes_no_or_number(word: str) -> bool:
    """
    Check if a word is "yes", "no", or a number.
    Used for VQA-X to determine if only the first word should be used as answer.
    """
    if not word:
        return False
    word_lower = word.lower().strip()
    # Check for yes/no
    if word_lower in ("yes", "no"):
        return True
    # Check if it's a number (integer or decimal)
    try:
        float(word_lower)
        return True
    except ValueError:
        # Check if it's a word representation of a number (e.g., "one", "two", etc.)
        # For now, we'll just check for numeric strings
        if word_lower.isdigit():
            return True
    return False


def clean_text(text: str) -> str:
    """
    Clean text for answer-only postprocessing:
    - Remove prefixes (assistant:, response:, answer:, question:)
    - Remove newlines
    - Strip whitespace
    """
    t = (text or "").strip()
    t = re.sub(r"^(assistant:|response:|answer:|question:)\s*", "", t, flags=re.I)
    t = t.replace("\n", " ").strip()
    return t


def postprocess_answer_only(
    raw_text: str,
    task: TaskType,
    max_tokens: int = 3,
) -> str:
    """
    Postprocess answer-only predictions (no explanations).
    
    Args:
        raw_text: Raw model output text
        task: Task type ("VQA-X", "ACT-X", "ESNLI-VE", "VCR")
        max_tokens: Maximum number of tokens to keep (default: 3)
    
    Returns:
        Cleaned answer string
    """
    if not raw_text:
        return ""
    
    # Clean text
    t = clean_text(raw_text)
    toks = t.split()
    if not toks:
        return ""
    
    # Task-specific handling
    if task == "VQA-X":
        # Special case: If first word is yes/no/number, use only the first word
        # (cut after first word, regardless of how many words follow)
        if _is_yes_no_or_number(toks[0]):
            return toks[0]
        # Default: take first max_tokens (typically 2 for VQA-X)
        return " ".join(toks[:max_tokens])
    
    elif task == "ESNLI-VE":
        # e-SNLI-VE: Force label to valid space
        # Take first word/token and force to valid label
        label = _force_label_space(toks[0] if toks else "")
        return label
    
    elif task == "VCR":
        # VCR: Parse letter (A, B, C, or D)
        letter = _parse_vcr_letter(raw_text)
        return letter if letter else ""
    
    else:
        # ACT-X and others: take first max_tokens
        return " ".join(toks[:max_tokens])


def postprocess_prediction(
    raw_text: str,
    task: TaskType,
    vcr_choices: Optional[list] = None,
) -> Dict[str, str]:
    """
    Unified postprocessing function for all tasks.
    
    Args:
        raw_text: Raw model output text
        task: Task type ("VQA-X", "ACT-X", "ESNLI-VE", "VCR")
        vcr_choices: For VCR task, list of choice texts to map letter to answer
    
    Returns:
        Dictionary with keys:
            - "answer": The answer/label/letter (normalized)
            - "explanation": The explanation text
            - "full_text": Complete formatted text "<answer> because <explanation>"
            - "raw_answer": For VCR, the letter; for others, same as "answer"
    """
    if not raw_text:
        return {
            "answer": "unknown",
            "explanation": "explanation missing",
            "full_text": "unknown because explanation missing",
            "raw_answer": "unknown",
        }

    # Task-specific preprocessing
    if task == "ESNLI-VE":
        # e-SNLI-VE: Replace comma with "because" if "because" not present
        pred_full = raw_text.strip()
        if "because" not in pred_full.lower():
            pred_full = pred_full.replace(",", " because ", 1)
    else:
        pred_full = raw_text.strip()

    # Normalize text
    normalized = normalize_generated_text(pred_full)

    if not normalized:
        return {
            "answer": "unknown",
            "explanation": "explanation missing",
            "full_text": "unknown because explanation missing",
            "raw_answer": "unknown",
        }

    # VQA-X special handling: if first word is yes/no/number, use only first word
    if task == "VQA-X":
        toks = normalized.split()
        if not toks:
            return {
                "answer": "unknown",
                "explanation": "explanation missing",
                "full_text": "unknown because explanation missing",
                "raw_answer": "unknown",
            }
        
        first_word = toks[0]
        if _is_yes_no_or_number(first_word):
            # For yes/no/number: only first word is answer, rest is explanation
            answer_raw = first_word
            expl_raw = " ".join(toks[1:]).strip()
        else:
            # For other answers: use standard because-splitting logic
            if " because " in f" {normalized} ":
                parts = normalized.split("because", 1)
                answer_raw = parts[0].strip()
                expl_raw = parts[1].strip() if len(parts) > 1 else ""
            else:
                # No "because" found: first word/token = answer, rest = explanation
                answer_raw = first_word
                expl_raw = " ".join(toks[1:]).strip()
    else:
        # For other tasks: standard because-splitting
        if " because " in f" {normalized} ":
            parts = normalized.split("because", 1)
            answer_raw = parts[0].strip()
            expl_raw = parts[1].strip() if len(parts) > 1 else ""
        else:
            # No "because" found: first word/token = answer, rest = explanation
            toks = normalized.split()
            if not toks:
                return {
                    "answer": "unknown",
                    "explanation": "explanation missing",
                    "full_text": "unknown because explanation missing",
                    "raw_answer": "unknown",
                }
            answer_raw = toks[0]
            expl_raw = " ".join(toks[1:]).strip()

    # Task-specific postprocessing
    if task == "VCR":
        # VCR: Extract letter and map to answer text
        letter = _parse_vcr_letter(raw_text)
        if letter is not None and vcr_choices and 0 <= _LETTER_TO_IDX[letter] < len(vcr_choices):
            answer_text = vcr_choices[_LETTER_TO_IDX[letter]]
        else:
            answer_text = "unknown"
        
        # Use the explanation from normalized text, or default
        if not expl_raw:
            expl_raw = "no further details"
        
        return {
            "answer": answer_text.lower(),
            "explanation": expl_raw,
            "full_text": f"{answer_text.lower()} because {expl_raw}",
            "raw_answer": letter or "unknown",
        }
    
    elif task == "ESNLI-VE":
        # e-SNLI-VE: Force label to valid space
        label = _force_label_space(answer_raw)
        explanation = expl_raw if expl_raw else "explanation missing"
        
        return {
            "answer": label,
            "explanation": explanation,
            "full_text": f"{label} because {explanation}",
            "raw_answer": label,
        }
    
    elif task == "VQA-X":
        # VQA-X: answer already processed (yes/no/number handling done above)
        answer = answer_raw if answer_raw else "unknown"
        explanation = expl_raw if expl_raw else "explanation missing"
        
        return {
            "answer": answer,
            "explanation": explanation,
            "full_text": f"{answer} because {explanation}",
            "raw_answer": answer,
        }
    
    else:
        # ACT-X: Standard processing
        answer = answer_raw if answer_raw else "unknown"
        explanation = expl_raw if expl_raw else "explanation missing"
        
        return {
            "answer": answer,
            "explanation": explanation,
            "full_text": f"{answer} because {explanation}",
            "raw_answer": answer,
        }

