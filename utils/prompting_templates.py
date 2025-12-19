from typing import List, Dict, Optional

# Import CoT prompting templates from the cot_prompting_templates package
from .cot_prompting_templates import (
    prompt_image_description_cot as _prompt_image_description_cot_base,
    prompt_question_from_description_cot_vqax,
    prompt_question_from_description_cot_actx,
    prompt_question_from_description_cot_esnlive,
    prompt_question_from_description_cot_vcr,
)


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


# -------------------------------------------------------------------
# CoT Prompting Templates Wrapper
# -------------------------------------------------------------------
# Wrapper function that injects helper functions into the CoT template
def prompt_image_description_cot(
    prompt_mode: str = "zero",
    question: Optional[str] = None,
    hypothesis: Optional[str] = None,
    choices: Optional[List[str]] = None,
    task: Optional[str] = None,
    use_question_in_stage1: bool = False,
):
    """
    Stage 1 of CoT: Generate a description of the image.
    Returns a conversation with just the image description prompt.
    
    This is a wrapper that injects the helper functions into the CoT template.
    
    Args:
        prompt_mode: Prompt mode (zero, 1shot, 3shot, 6shot)
        question: Question to answer (for VQA-X, VCR) - used in Variant 2
        hypothesis: Hypothesis to evaluate (for ESNLI-VE) - used in Variant 2
        choices: Answer choices (for VCR) - used in Variant 2
        task: Task name (VQA-X, ACT-X, ESNLI-VE, VCR) - used in Variant 2
        use_question_in_stage1: If True, use Variant 2 (question in Stage 1), else Variant 1 (default)
    """
    if use_question_in_stage1:
        # Import Variant 2
        from .cot_prompting_templates.cot_2stage_with_question import (
            prompt_image_description_cot as _prompt_image_description_cot_v2,
        )
        return _prompt_image_description_cot_v2(
            prompt_mode=prompt_mode,
            question=question,
            hypothesis=hypothesis,
            choices=choices,
            task=task,
            resolve_shot_count=resolve_shot_count,
            add_fewshot_examples=add_fewshot_examples,
        )
    else:
        # Use default Variant 1
        return _prompt_image_description_cot_base(
            prompt_mode=prompt_mode,
            resolve_shot_count=resolve_shot_count,
            add_fewshot_examples=add_fewshot_examples,
        )


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

VQAX_FEWSHOT_COT = [
    {
        "user": "Question: Is there a dog in the image?",
        "assistant": "I can see a four-legged creature with fur. It has the characteristic features of a dog: floppy ears, a tail, and a snout. Therefore, the answer is: yes",
    },
    {
        "user": "Question: How many people are in the image?",
        "assistant": "I observe one person on the left side, another person in the center, and a third person on the right side. That makes three individuals. Therefore, the answer is: 3",
    },
    {
        "user": "Question: What color is the bus?",
        "assistant": "I can see a large vehicle with windows and wheels. The body of the vehicle is painted in a bright, saturated color. Looking closely, the color appears to be yellow. Therefore, the answer is: yellow",
    },
    {
        "user": "Question: Is the person holding something?",
        "assistant": "I observe the person has an object in their hand. Looking at the shape and size, it appears to be a mobile phone or similar device. Therefore, the answer is: yes",
    },
    {
        "user": "Question: How many dogs are in the park?",
        "assistant": "I can see one dog near the tree, another dog by the bench, and a third dog in the distance. That makes three canines in total. Therefore, the answer is: 3",
    },
    {
        "user": "Question: What type of vehicle is shown?",
        "assistant": "I see a motorized vehicle with four wheels, a windshield, and doors. It has the typical structure of an automobile. The size and shape indicate it is a car, not a truck or bus. Therefore, the answer is: car",
    },
]


def prompt_vqax_expl(question: str, prompt_mode: str, generation_mode: str = "posthoc"):
    k = resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    if generation_mode == "cot":
        add_fewshot_examples(conversation, VQAX_FEWSHOT_COT, k)
        instructions = (
            "Given an IMAGE and a QUESTION, think step by step and then provide your answer in this formate:\n"
            "<explanation> Therefore, the answer is: <answer>\n"
            f"Question: {question}"
        )
    else:
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

ACTX_FEWSHOT_COT = [
    {
        "user": "Description: A person is sitting with a guitar.",
        "assistant": "I observe a person in a seated position. They are holding a musical instrument - specifically a guitar. The person's hands are positioned on the strings, suggesting they are actively playing. The posture and hand position indicate strumming or plucking motions. Therefore, the activity is: playing guitar",
    },
    {
        "user": "Description: A person rides a bicycle outside.",
        "assistant": "I see a person on a two-wheeled vehicle. The person is in a riding position with their feet on pedals. The vehicle has handlebars and is clearly a bicycle. The person is outdoors, as indicated by the environment. The motion and positioning suggest active cycling. Therefore, the activity is: riding bike",
    },
    {
        "user": "Description: Someone stands in a kitchen using pans.",
        "assistant": "I see a person in a kitchen environment. They are standing near a stove. The person is holding pans and appears to be cooking. The context of being in a kitchen with cooking utensils indicates food preparation. Therefore, the activity is: cooking",
    },
    {
        "user": "Description: A person is running on a track.",
        "assistant": "I observe a person on a track or running surface. The person's body position shows forward motion with legs in a running stride. The environment is clearly a track, designed for running. The speed and posture indicate running rather than walking. Therefore, the activity is: running",
    },
    {
        "user": "Description: Someone is reading a book while sitting.",
        "assistant": "I see a person in a seated position. They are holding an open book. The person's eyes are directed at the book, and their hands are positioned to hold pages. This indicates active reading. Therefore, the activity is: reading",
    },
    {
        "user": "Description: A person is swimming in a pool.",
        "assistant": "I observe a person in water. The person is in a swimming pool. Their body is horizontal in the water, and their arms and legs are in swimming motions. The environment and body position clearly indicate swimming. Therefore, the activity is: swimming",
    },
]


def prompt_actx_expl(prompt_mode: str, generation_mode: str = "posthoc"):
    k = resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    if generation_mode == "cot":
        add_fewshot_examples(conversation, ACTX_FEWSHOT_COT, k)
        instructions = (
            "Given an IMAGE, analyze the activity step by step and then identify it in this format.\n"
            "<explanation> Therefore, the answer is: <answer>"
        )
    else:
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

ESNLIVE_FEWSHOT_COT = [
    {
        "user": "Hypothesis: A person is skiing.",
        "assistant": "First, I check if there is a person in the image - yes, I can see a person. Next, I examine what activity they are doing. I observe the person is on skis, positioned on a snowy slope, with poles in hand. The hypothesis states 'A person is skiing' which directly matches what I observe. Therefore, the answer is: entailment",
    },
    {
        "user": "Hypothesis: The man is riding a bicycle indoors.",
        "assistant": "I see a person on a bicycle. The hypothesis claims they are indoors. However, when I examine the background, I can see trees, sky, and an outdoor environment. The setting is clearly outdoors, not indoors. This contradicts the hypothesis. Therefore, the answer is: contradiction",
    },
    {
        "user": "Hypothesis: Someone might be preparing a meal.",
        "assistant": "I see a kitchen scene with various items. The hypothesis says 'might be preparing a meal' which suggests uncertainty. Looking at the scene, there are cooking utensils and ingredients visible, but I cannot definitively confirm active meal preparation is happening. The scene is ambiguous. Therefore, the answer is: neutral",
    },
    {
        "user": "Hypothesis: There is a dog in the image.",
        "assistant": "I can clearly see a four-legged animal with fur, ears, and a tail. The characteristics match a dog. The hypothesis 'There is a dog in the image' is directly supported by what I observe. Therefore, the answer is: entailment",
    },
    {
        "user": "Hypothesis: The car is blue.",
        "assistant": "I see a car in the image. The hypothesis claims it is blue. However, when I look at the actual color of the vehicle, it appears to be red, not blue. This directly contradicts the hypothesis. Therefore, the answer is: contradiction",
    },
    {
        "user": "Hypothesis: It could be morning or evening.",
        "assistant": "I examine the lighting in the image. The hypothesis suggests it could be morning or evening, indicating uncertainty. Looking at the lighting conditions, I cannot definitively determine whether it is morning or evening based on the available visual cues. The scene is ambiguous regarding time. Therefore, the answer is: neutral",
    },
]


def prompt_esnlive_expl(hypothesis: str, prompt_mode: str, generation_mode: str = "posthoc"):
    k = resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    if generation_mode == "cot":
        add_fewshot_examples(conversation, ESNLIVE_FEWSHOT_COT, k)
        instructions = (
            "Given an IMAGE and a HYPOTHESIS, evaluate the relationship step by step.\n"
            "<explanation> Therefore, the answer is: <label>\n"
            "Label must be: entailment, contradiction, or neutral.\n"
        )
    else:
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

VCR_FEWSHOT_COT = [
    {
        "user": "Question: Why is the person smiling? Options: A) happy B) sad C) angry D) tired",
        "assistant": "First, I observe the person's facial expression - they are smiling. A smile typically indicates positive emotion. Looking at the context, the person appears relaxed and content. Option A (happy) aligns with the smiling expression. Options B (sad), C (angry), and D (tired) do not match a smiling face. Therefore, the answer is: A",
    },
    {
        "user": "Question: What is the woman doing? Options: A) sleeping B) reading C) running D) dancing",
        "assistant": "I see a woman in the image. She is holding an object in her hands. Looking closely, the object appears to be a book - it has pages and a rectangular shape. Her eyes are directed at the book, suggesting she is looking at it. Her posture is seated and relaxed, which is consistent with reading. This matches option B. Therefore, the answer is: B",
    },
    {
        "user": "Question: What is the child looking at? Options: A) dog B) bird C) phone D) ball",
        "assistant": "I observe a child in the image. The child's gaze and body position indicate they are looking at something in their hands. The object appears to be a rectangular device with a screen, which matches the description of a phone. Options A (dog), B (bird), and D (ball) do not match what the child is holding. Therefore, the answer is: C",
    },
    {
        "user": "Question: What color is the car? Options: A) red B) blue C) green D) yellow",
        "assistant": "I see a car in the image. I need to identify its color. Looking at the vehicle's body, the paint appears to be a bright, saturated color. The specific hue matches yellow rather than red, blue, or green. Therefore, the answer is: D",
    },
    {
        "user": "Question: How many people are visible? Options: A) one B) two C) three D) four",
        "assistant": "I scan the scene for individuals. I can see one person clearly visible in the frame. There do not appear to be additional people in the scene. Options B (two), C (three), and D (four) do not match the count. Therefore, the answer is: A",
    },
    {
        "user": "Question: Where is the scene taking place? Options: A) indoors B) outdoors C) in a car D) at a beach",
        "assistant": "I examine the environment in the image. I can see natural elements like trees, sky, and open space. There are no walls, ceilings, or indoor structures visible. The setting is clearly an outdoor environment. Options A (indoors), C (in a car), and D (at a beach) do not match the general outdoor setting. Therefore, the answer is: B",
    },
]


def prompt_vcr_expl(
    question: str,
    choices: List[str],
    prompt_mode: str,
    generation_mode: str = "posthoc",
    include_image: bool = True,
):
    k = resolve_shot_count(prompt_mode)
    conversation: List[Dict] = []

    padded = (choices + ["missing"] * 4)[:4]

    if generation_mode == "cot":
        add_fewshot_examples(conversation, VCR_FEWSHOT_COT, k)
        instruction_block = (
            ("Given an IMAGE, " if include_image else "Given NO IMAGE (text-only), ")
            + "a QUESTION and four options, analyze step by step and choose the correct answer in this format:\n"
            "<explanation> Therefore, the answer is: <letter>\n"
            "Letter must be A, B, C, or D."
        )
    else:
        add_fewshot_examples(conversation, VCR_FEWSHOT, k)
        instruction_block = (
            ("Given an IMAGE, " if include_image else "Given NO IMAGE (text-only), ")
            + "a QUESTION and four options, answer and explain in this format:\n"
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

    user_content = [{"type": "text", "text": text}]
    if include_image:
        user_content.append({"type": "image"})

    conversation.append({"role": "user", "content": user_content})
    return conversation


# ============
# Answer Only
# ============

def prompt_vqax_answer_only(question: str, include_image: bool = True):
    instructions = (
        ("Given an IMAGE and a QUESTION" if include_image else "Given NO IMAGE (text-only) and a QUESTION")
        + ", answer in this format:\n"
        "<answer>\n"
        "Use at most three words and do not add any explanation."
        f"Question: {question}"
    )
    user_content = [{"type": "text", "text": instructions}]
    if include_image:
        user_content.append({"type": "image"})
    return [{
        "role": "user",
        "content": user_content,
    }]


def prompt_actx_answer_only(include_image: bool = True):
    instructions = (
        ("Given an IMAGE" if include_image else "Given NO IMAGE (text-only)")
        + ", identify the activity in this format:\n"
        "<activity>\n"
        "Use at most two words and do not add any explanation."
    )
    user_content = [{"type": "text", "text": instructions}]
    if include_image:
        user_content.append({"type": "image"})
    return [{
        "role": "user",
        "content": user_content,
    }]


def prompt_esnlive_answer_only(hypothesis: str, include_image: bool = True):
    instructions = (
        ("Given an IMAGE and a HYPOTHESIS" if include_image else "Given NO IMAGE (text-only) and a HYPOTHESIS")
        + ", classify the relation in this format:\n"
        "<label>\n"
        "Label must be exactly one of: entailment, contradiction, neutral.\n"
        "Use only the label and do not add any explanation."
        f"Hypothesis: {hypothesis}"
    )
    user_content = [{"type": "text", "text": instructions}]
    if include_image:
        user_content.append({"type": "image"})
    return [{
        "role": "user",
        "content": user_content,
    }]


def prompt_vcr_answer_only(question: str, choices: List[str], include_image: bool = True):
    padded = (choices + ["missing"] * 4)[:4]
    instructions = (
        ("Given an IMAGE, a QUESTION and four options" if include_image else "Given NO IMAGE (text-only), a QUESTION and four options")
        + ", choose the correct answer in this format:\n"
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
    user_content = [{"type": "text", "text": text}]
    if include_image:
        user_content.append({"type": "image"})
    return [{
        "role": "user",
        "content": user_content,
    }]