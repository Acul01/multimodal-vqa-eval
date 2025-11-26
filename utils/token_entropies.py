# utils/token_entropies.py
# -*- coding: utf-8 -*-
"""
Token-level entropy extraction and filtering functions.
Handles entropy computation from model outputs and filtering to explanation tokens.
"""
import re
from typing import Dict, Optional
import torch
import torch.nn.functional as F

# Stopword list for entropy filtering
MANUAL_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "is", "are",
    "to", "and", "or", "because", "that", "this", "it",
    "they", "he", "she", "we", "you", "i", "them",
    "his", "her", "their", "there", "here",
}


def clean_token_for_entropy(t: str) -> Optional[str]:
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


def extract_token_entropies(
    out,
    tokenizer,
    input_len: int,
) -> Dict[str, float]:
    """
    Berechne Entropie pro generiertem Token (nach dem Prompt).
    - Token werden mit clean_token_for_entropy normalisiert
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

        tok_clean = clean_token_for_entropy(tok_str)
        if tok_clean is None:
            continue

        if tok_clean in token_entropy:
            token_entropy[tok_clean] = max(token_entropy[tok_clean], H_round)
        else:
            token_entropy[tok_clean] = H_round

    return token_entropy


def filter_entropy_to_explanation(
    token_entropy: Dict[str, float],
    explanation: str,
) -> Dict[str, float]:
    """
    Behalte nur Entropien für Tokens, die in der Explanation vorkommen
    (nach dem gleichen Cleaning wie in clean_token_for_entropy).
    """
    if not explanation:
        return {}

    # rohe "Wörter" aus der Explanation holen
    words_raw = re.findall(r"[A-Za-z0-9]+", explanation)
    keep: set[str] = set()

    for w in words_raw:
        cleaned = clean_token_for_entropy(w)
        if cleaned is not None:
            keep.add(cleaned)

    if not keep:
        return {}

    return {
        tok: H for tok, H in token_entropy.items()
        if tok in keep
    }

