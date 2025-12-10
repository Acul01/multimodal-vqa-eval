from typing import List, Dict


# -------------------------------------------------------------------
# Helper: map prompt_mode string to k-shot count
# -------------------------------------------------------------------
def resolve_shot_count(mode: str) -> int:
    m = (mode or "zero").lower()
    if m.startswith("1"):
        return 1
    if m.startswith("3"):
        return 3
    if m.startswith("6"):
        return 6
    return 0  # default: zero-shot


# -------------------------------------------------------------------
# Generic few-shot injection
# -------------------------------------------------------------------
def add_fewshot_examples(conversation: List[Dict], examples: List[Dict], k: int):
    """
    examples: list of {"user": "...", "assistant": "..."} items
    conversation: the final chat history
    k: number of examples to include
    """
    for ex in examples[:k]:
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": ex["user"]}],
        })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ex["assistant"]}],
        })


# =====================
# Answer + Explanation
# =====================

# -------------------------------------------------------------
# VQA-X examples: question/answer typer: yes/no, number, other
# -------------------------------------------------------------
VQAX_FEWSHOT = [
    {
        "user": "Question: Is there a dog in the image?",
        "assistant": "yes because a dog is visible in the picture",
    },
    {
        "user": "Question: How many people are in the image?",
        "assistant": "3 because there are three individuals visible",
    },
    {
        "user": "Question: What color is the bus?",
        "assistant": "yellow because the vehicle is painted bright yellow",
    },
    {
        "user": "Question: Is the person holding something?",
        "assistant": "yes because the person has a mobile phone in his hand",
    },
    {
        "user": "Question: How many dogs are in the park?",
        "assistant": "3 because three canines are present in the park area",
    },
    {
        "user": "Question: What type of vehicle is shown?",
        "assistant": "car because the image displays an automobile with four wheels",
    },
]


def prompt_vqax_expl(question: str, prompt_mode: str):
    k = resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    add_fewshot_examples(conversation, VQAX_FEWSHOT, k)

    instructions = (
        "Given an IMAGE and a QUESTION, answer and explain in this format:\n"
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


# ---------------------------------------------------------
# ACT-X examples
# ---------------------------------------------------------
ACTX_FEWSHOT = [
    {
        "user": "Description: A person is sitting with a guitar.",
        "assistant": "playing guitar because the person is strumming the instrument",
    },
    {
        "user": "Description: A person rides a bicycle outside.",
        "assistant": "riding bike because they are moving on a bicycle",
    },
    {
        "user": "Description: Someone stands in a kitchen using pans.",
        "assistant": "cooking because they are working with pans on a stove",
    },
    {
        "user": "Description: A person is running on a track.",
        "assistant": "running because the person is moving quickly on foot",
    },
    {
        "user": "Description: Someone is reading a book while sitting.",
        "assistant": "reading because the person is looking at an open book",
    },
    {
        "user": "Description: A person is swimming in a pool.",
        "assistant": "swimming because they are moving through water",
    },
]


def prompt_actx_expl(prompt_mode: str):
    k = resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    add_fewshot_examples(conversation, ACTX_FEWSHOT, k)

    instructions = (
        "Given an IMAGE, identify the activity and explain in this format:\n"
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


# ---------------------------------------------------------
# e-SNLI-VE examples
# ---------------------------------------------------------
ESNLIVE_FEWSHOT = [
    {
        "user": "Hypothesis: A person is skiing.",
        "assistant": "entailment because the image shows someone skiing",
    },
    {
        "user": "Hypothesis: The man is riding a bicycle indoors.",
        "assistant": "contradiction because the bicycle is outdoors",
    },
    {
        "user": "Hypothesis: Someone might be preparing a meal.",
        "assistant": "neutral because the kitchen scene is ambiguous",
    },
    {
        "user": "Hypothesis: There is a dog in the image.",
        "assistant": "entailment because a canine is clearly visible",
    },
    {
        "user": "Hypothesis: The car is blue.",
        "assistant": "contradiction because the vehicle is red",
    },
    {
        "user": "Hypothesis: It could be morning or evening.",
        "assistant": "neutral because the lighting does not clearly indicate the time",
    },
]


def prompt_esnlive_expl(hypothesis: str, prompt_mode: str):
    k = resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    add_fewshot_examples(conversation, ESNLIVE_FEWSHOT, k)

    instructions = (
        "Given an IMAGE and a HYPOTHESIS, answer and explain in this format:\n"
        "<label> because <explanation>\n"
        "Label must be: entailment, contradiction, or neutral.\n"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"{instructions}\nHypothesis: {hypothesis}"},
            {"type": "image"},
        ],
    })
    return conversation


# ---------------------------------------------------------
# VCR examples (multiple-choice)
# ---------------------------------------------------------
VCR_FEWSHOT = [
    {
        "user": "Question: Why is the person smiling? Options: A) happy B) sad C) angry D) tired",
        "assistant": "A because the person shows clear signs of happiness",
    },
    {
        "user": "Question: What is the woman doing? Options: A) sleeping B) reading C) running D) dancing",
        "assistant": "B because she is holding an open book",
    },
    {
        "user": "Question: What is the child looking at? Options: A) dog B) bird C) phone D) ball",
        "assistant": "C because the child is facing the device in their hands",
    },
    {
        "user": "Question: What color is the car? Options: A) red B) blue C) green D) yellow",
        "assistant": "D because the vehicle is painted yellow",
    },
    {
        "user": "Question: How many people are visible? Options: A) one B) two C) three D) four",
        "assistant": "A because there is just one individual in the scene",
    },
    {
        "user": "Question: Where is the scene taking place? Options: A) indoors B) outdoors C) in a car D) at a beach",
        "assistant": "B because the environment shows an outdoor setting",
    },
]


def prompt_vcr_expl(question: str, choices: List[str], prompt_mode: str):
    k = resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    add_fewshot_examples(conversation, VCR_FEWSHOT, k)

    padded = (choices + ["missing"] * 4)[:4]

    instruction_block = (
        "Given an IMAGE, a QUESTION and four options, answer and explain in this format:\n"
        "<letter> because <explanation>\n"
        "Letter must be A, B, C, or D."
    )

    text = (
        f"{instruction_block}\n"
        f"Question: {question}\n"
        f"A) {padded[0]}\n"
        f"B) {padded[1]}\n"
        f"C) {padded[2]}\n"
        f"D) {padded[3]}"
    )

    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},
        ],
    })
    return conversation


# ============
# Answer Only
# ============

def prompt_vqax_answer_only(question: str):
    instructions = (
        "Given an IMAGE and a QUESTION, answer in this format:\n"
        "<answer>\n"
        "Use at most three words and do not add any explanation."
        f"Question: {question}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": instructions},
            {"type": "image"},
        ],
    }]


def prompt_actx_answer_only():
    instructions = (
        "Given an IMAGE, identify the activity in this format:\n"
        "<activity>\n"
        "Use at most two words and do not add any explanation."
    )
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": instructions},
            {"type": "image"},
        ],
    }]


def prompt_esnlive_answer_only(hypothesis: str):
    instructions = (
        "Given an IMAGE and a HYPOTHESIS, classify the relation in this format:\n"
        "<label>\n"
        "Label must be exactly one of: entailment, contradiction, neutral.\n"
        "Use only the label and do not add any explanation."
        f"Hypothesis: {hypothesis}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": instructions},
            {"type": "image"},
        ],
    }]


def prompt_vcr_answer_only(question: str, choices: List[str]):
    padded = (choices + ["missing"] * 4)[:4]
    instructions = (
        "Given an IMAGE, a QUESTION and four options, choose the correct answer in this format:\n"
        "<letter>\n"
        "Letter must be A, B, C, or D."
        "Use only the letter and do not add any explanation."
    )
    text = (
        f"{instructions}\n"
        f"Question: {question}\n"
        f"A) {padded[0]}\n"
        f"B) {padded[1]}\n"
        f"C) {padded[2]}\n"
        f"D) {padded[3]}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},
        ],
    }]