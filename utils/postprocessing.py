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
_THEREFORE_RE = re.compile(r"\btherefore\b", flags=re.I)
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


def _parse_vcr_letter(text: str, vcr_choices: Optional[list] = None) -> Optional[str]:
    """
    Parse VCR letter (A, B, C, or D) from text.
    
    First tries to find a letter ('a', 'b', 'c', 'd') at the start of the text.
    If not found and vcr_choices are provided, tries to match the generated text
    with one of the choices to determine the letter.
    
    Returns the letter if found, None otherwise.
    """
    t = (text or "").strip()
    
    # DEBUG
    print(f"[DEBUG _parse_vcr_letter] input: {repr(text)}")
    
    t = re.sub(r'^(assistant:|response:|answer:)\s*', "", t, flags=re.I)
    t = t.replace("\n", " ").strip()
    
    # DEBUG
    print(f"[DEBUG _parse_vcr_letter] after cleaning: {repr(t)}")

    # First, try to find a letter at the start
    tokens = t.split()
    if tokens:
        cand = tokens[0].lower().strip(":.")
        print(f"[DEBUG _parse_vcr_letter] first token: {repr(tokens[0])}, candidate: {repr(cand)}")
        if cand in _LETTER_TO_IDX:
            print(f"[DEBUG _parse_vcr_letter] Found letter: {cand}")
            return cand
        else:
            print(f"[DEBUG _parse_vcr_letter] '{cand}' not in {list(_LETTER_TO_IDX.keys())}")
    else:
        print(f"[DEBUG _parse_vcr_letter] No tokens found")
    
    # If no letter found and choices are provided, try to match with choices
    if vcr_choices:
        print(f"[DEBUG _parse_vcr_letter] Trying to match with choices: {vcr_choices}")
        # Normalize the generated text for comparison
        # Remove "because" and everything after it
        t_normalized = re.split(r'\bbecause\b', t, flags=re.I)[0].strip().lower()
        # Remove trailing punctuation
        t_normalized = re.sub(r'[.,;:!?]+$', '', t_normalized).strip()
        # Normalize multiple spaces
        t_normalized = re.sub(r'\s+', ' ', t_normalized)
        print(f"[DEBUG _parse_vcr_letter] normalized text (before 'because'): {repr(t_normalized)}")
        
        # Try to match with each choice
        for idx, choice in enumerate(vcr_choices):
            if not choice:
                continue
            # Normalize choice text
            choice_normalized = choice.strip().lower()
            # Remove trailing punctuation
            choice_normalized = re.sub(r'[.,;:!?]+$', '', choice_normalized).strip()
            # Normalize multiple spaces
            choice_normalized = re.sub(r'\s+', ' ', choice_normalized)
            print(f"[DEBUG _parse_vcr_letter] comparing with choice {idx} ({list(_LETTER_TO_IDX.keys())[idx]}): {repr(choice_normalized)}")
            
            # Check if normalized text starts with choice (exact match at start)
            if t_normalized.startswith(choice_normalized):
                letter = list(_LETTER_TO_IDX.keys())[idx]
                print(f"[DEBUG _parse_vcr_letter] Matched (starts with) choice {idx}, returning letter: {letter}")
                return letter
            
            # Check if choice is contained in normalized text (for cases where there's extra text)
            if choice_normalized in t_normalized and len(choice_normalized) > 20:  # Only if substantial match
                letter = list(_LETTER_TO_IDX.keys())[idx]
                print(f"[DEBUG _parse_vcr_letter] Matched (contains) choice {idx}, returning letter: {letter}")
                return letter
            
            # Also try reverse: check if choice starts with normalized text (for partial matches)
            if choice_normalized.startswith(t_normalized) and len(t_normalized) > 20:  # Only if substantial match
                letter = list(_LETTER_TO_IDX.keys())[idx]
                print(f"[DEBUG _parse_vcr_letter] Reverse match with choice {idx}, returning letter: {letter}")
                return letter
    
    print(f"[DEBUG _parse_vcr_letter] No match found, returning None")
    return None


def _is_yes_no_or_number(word: str) -> bool:
    """
    Check if a word is "yes", "no", or a number.
    Used for VQA-X to determine if only the first word should be used as answer.
    """
    if not word:
        return False
    # Strip punctuation from the word before checking
    word_clean = word.strip().rstrip(".,;:!?")
    word_lower = word_clean.lower()
    
    # DEBUG
    print(f"[DEBUG _is_yes_no_or_number] input: {repr(word)}, cleaned: {repr(word_clean)}, lower: {repr(word_lower)}")
    
    # Check for yes/no
    if word_lower in ("yes", "no"):
        print(f"[DEBUG _is_yes_no_or_number] Matched yes/no: {word_lower}")
        return True
    # Check if it's a number (integer or decimal)
    try:
        float(word_lower)
        print(f"[DEBUG _is_yes_no_or_number] Matched number: {word_lower}")
        return True
    except ValueError:
        # Check if it's a word representation of a number (e.g., "one", "two", etc.)
        # For now, we'll just check for numeric strings
        if word_lower.isdigit():
            print(f"[DEBUG _is_yes_no_or_number] Matched digit: {word_lower}")
            return True
    print(f"[DEBUG _is_yes_no_or_number] No match for: {word_lower}")
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
    vcr_choices: Optional[list] = None,
) -> str:
    """
    Postprocess answer-only predictions (no explanations).
    
    Args:
        raw_text: Raw model output text
        task: Task type ("VQA-X", "ACT-X", "ESNLI-VE", "VCR")
        max_tokens: Maximum number of tokens to keep (default: 3)
        vcr_choices: For VCR task, list of choice texts to map letter to answer
    
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
    
    # DEBUG: Print for VQA-X
    if task == "VQA-X":
        print(f"[DEBUG postprocess_answer_only] raw_text: {repr(raw_text)}")
        print(f"[DEBUG postprocess_answer_only] cleaned_text: {repr(t)}")
        print(f"[DEBUG postprocess_answer_only] tokens: {toks}")
        print(f"[DEBUG postprocess_answer_only] first_token: {repr(toks[0])}")
        is_yes_no_num = _is_yes_no_or_number(toks[0])
        print(f"[DEBUG postprocess_answer_only] is_yes_no_or_number('{toks[0]}'): {is_yes_no_num}")
    
    # Task-specific handling
    if task == "VQA-X":
        # Special case: If first word is yes/no/number, use only the first word
        # (cut after first word, regardless of how many words follow)
        if _is_yes_no_or_number(toks[0]):
            result = toks[0]
            print(f"[DEBUG postprocess_answer_only] Returning only first word: {repr(result)}")
            return result
        # Default: take first max_tokens (typically 2 for VQA-X)
        result = " ".join(toks[:max_tokens])
        print(f"[DEBUG postprocess_answer_only] Returning first {max_tokens} tokens: {repr(result)}")
        return result
    
    elif task == "ESNLI-VE":
        # e-SNLI-VE: Force label to valid space
        # Take first word/token and force to valid label
        label = _force_label_space(toks[0] if toks else "")
        return label
    
    elif task == "VCR":
        # VCR: Parse letter (A, B, C, or D)
        letter = _parse_vcr_letter(raw_text, vcr_choices)
        return letter if letter else ""
    
    else:
        # ACT-X and others: take first max_tokens
        return " ".join(toks[:max_tokens])


def postprocess_prediction(
    raw_text: str,
    task: TaskType,
    vcr_choices: Optional[list] = None,
    generation_mode: str = "posthoc",
) -> Dict[str, str]:
    """
    Unified postprocessing function for all tasks.
    
    Args:
        raw_text: Raw model output text
        task: Task type ("VQA-X", "ACT-X", "ESNLI-VE", "VCR")
        vcr_choices: For VCR task, list of choice texts to map letter to answer
        generation_mode: "posthoc" (answer because explanation) or "cot" (Therefore-based)
    
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

    # Handle CoT vs post-hoc parsing
    if generation_mode == "cot":
        # CoT: Extract answer from "Therefore, the answer is: <answer>" pattern
        # Extract everything before "Therefore" as explanation, and answer after "Therefore"
        text_lower = raw_text.lower()
        therefore_match = _THEREFORE_RE.search(text_lower)
        
        if therefore_match:
            # Split at "therefore"
            therefore_pos = therefore_match.start()
            expl_raw = raw_text[:therefore_pos].strip()
            
            # Extract answer from after "therefore"
            after_therefore = raw_text[therefore_match.end():].strip()
            
            # Remove leading punctuation and whitespace (e.g., ", " after "Therefore,")
            after_therefore = re.sub(r'^[,\s]+', '', after_therefore)
            
            # Look for patterns like "the answer is: X" or "the activity is: X" or "the label is: X"
            # Also handle "Therefore, the answer is" as a complete separator
            answer_patterns = [
                r"the\s+answer\s+is\s*:?\s*(.+)",
                r"the\s+activity\s+is\s*:?\s*(.+)",
                r"the\s+label\s+is\s*:?\s*(.+)",
                r"answer\s+is\s*:?\s*(.+)",
                r"activity\s+is\s*:?\s*(.+)",
                r"label\s+is\s*:?\s*(.+)",
            ]
            
            answer_raw = None
            for pattern in answer_patterns:
                match = re.search(pattern, after_therefore, flags=re.I)
                if match:
                    answer_raw = match.group(1).strip().rstrip(".,;:!?")
                    break
            
            # If no pattern matched, try to extract first word/token after "therefore"
            # (skip articles)
            if not answer_raw:
                tokens = after_therefore.split()
                # Skip articles at the beginning
                token_idx = 0
                while token_idx < len(tokens) and tokens[token_idx].lower() in _ARTICLES:
                    token_idx += 1
                if token_idx < len(tokens):
                    answer_raw = tokens[token_idx].strip().rstrip(".,;:!?")
            
            if not answer_raw:
                answer_raw = "unknown"
            
            if not expl_raw:
                expl_raw = "explanation missing"
            
            # Normalize answer
            answer_normalized = normalize_generated_text(answer_raw)
            expl_normalized = normalize_generated_text(expl_raw)
            
            # For CoT, we still format as "answer because explanation" for consistency
            if not answer_normalized:
                answer_normalized = "unknown"
            if not expl_normalized:
                expl_normalized = "explanation missing"
            
            # Continue with task-specific processing below
            normalized = f"{answer_normalized} because {expl_normalized}"
        else:
            # No "therefore" found - fallback to post-hoc parsing
            generation_mode = "posthoc"
            pred_full = raw_text.strip()
            normalized = normalize_generated_text(pred_full)
    else:
        # Post-hoc: standard processing
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
        
        # Skip articles at the beginning
        first_word_idx = 0
        while first_word_idx < len(toks) and toks[first_word_idx] in _ARTICLES:
            first_word_idx += 1
        
        if first_word_idx >= len(toks):
            # Only articles found, use first token anyway
            first_word_idx = 0
        
        first_word = toks[first_word_idx]
        if _is_yes_no_or_number(first_word):
            # For yes/no/number: use the word (and skip articles before it)
            answer_raw = first_word
            expl_raw = " ".join(toks[first_word_idx + 1:]).strip()
        else:
            # For other answers: use standard because-splitting logic
            if " because " in f" {normalized} ":
                parts = normalized.split("because", 1)
                answer_raw = parts[0].strip()
                expl_raw = parts[1].strip() if len(parts) > 1 else ""
            else:
                # No "because" found: skip articles, then first word/token = answer, rest = explanation
                answer_raw = first_word
                expl_raw = " ".join(toks[first_word_idx + 1:]).strip()
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
        print(f"[DEBUG postprocess_prediction VCR] raw_text: {repr(raw_text)}")
        print(f"[DEBUG postprocess_prediction VCR] vcr_choices: {vcr_choices}")
        letter = _parse_vcr_letter(raw_text, vcr_choices)
        print(f"[DEBUG postprocess_prediction VCR] parsed letter: {letter}")
        if letter is not None and vcr_choices and 0 <= _LETTER_TO_IDX[letter] < len(vcr_choices):
            answer_text = vcr_choices[_LETTER_TO_IDX[letter]]
            print(f"[DEBUG postprocess_prediction VCR] mapped to answer_text: {repr(answer_text)}")
        else:
            answer_text = "unknown"
            print(f"[DEBUG postprocess_prediction VCR] No valid letter found, using 'unknown'")
        
        # Use the explanation from normalized text, or default
        if not expl_raw:
            expl_raw = "no further details"
        
        result = {
            "answer": answer_text.lower(),
            "explanation": expl_raw,
            "full_text": f"{answer_text.lower()} because {expl_raw}",
            "raw_answer": letter or "unknown",
        }
        print(f"[DEBUG postprocess_prediction VCR] returning: {result}")
        return result
    
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

