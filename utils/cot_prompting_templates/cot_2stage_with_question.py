"""
CoT 2-Stage Prompting Template (Variant 2)

This variant uses a two-stage approach where the question is provided in Stage 1:
- Stage 1: Generate image description (with image AND question)
- Stage 2: Answer question based on description (text only, no image)

Difference from Variant 1: The question is included in Stage 1, allowing the model
to generate a more focused description that is relevant to the question.
"""

from typing import List, Dict, Optional


def prompt_image_description_cot(
    prompt_mode: str = "zero",
    question: Optional[str] = None,
    hypothesis: Optional[str] = None,
    choices: Optional[List[str]] = None,
    task: Optional[str] = None,
    resolve_shot_count=None,
    add_fewshot_examples=None,
):
    """
    Stage 1 of CoT: Generate a description of the image, considering the question.
    Returns a conversation with image description prompt that includes the question.
    
    Args:
        prompt_mode: Prompt mode (zero, 1shot, 3shot, 6shot)
        question: Question to answer (for VQA-X, VCR)
        hypothesis: Hypothesis to evaluate (for ESNLI-VE)
        choices: Answer choices (for VCR)
        task: Task name (VQA-X, ACT-X, ESNLI-VE, VCR)
        resolve_shot_count: Function to resolve shot count from prompt_mode
        add_fewshot_examples: Function to add few-shot examples to conversation
    """
    k = resolve_shot_count(prompt_mode) if resolve_shot_count else 0
    conversation: List[Dict] = []
    
    # Few-shot examples for image description with question
    if task == "VQA-X" and question:
        description_examples = [
            {
                "user": "Question: Is there a dog in the image? Describe what you see in this image, focusing on elements relevant to the question.",
                "assistant": "I see a person sitting on a bench in a park. There is a dog nearby with brown fur, and trees in the background. The scene appears to be outdoors during daytime. The dog is clearly visible in the foreground.",
            },
            {
                "user": "Question: How many people are in the image? Describe what you see in this image, focusing on elements relevant to the question.",
                "assistant": "I see three people in the scene. They are standing together outdoors. There are trees and open space in the background. All three individuals are clearly visible.",
            },
            {
                "user": "Question: What color is the bus? Describe what you see in this image, focusing on elements relevant to the question.",
                "assistant": "I observe a large vehicle on a road. The vehicle has windows and wheels, indicating it is a bus. The body of the vehicle is painted in a bright yellow color. There are buildings in the background and the scene is outdoors.",
            },
        ]
    elif task == "ACT-X":
        description_examples = [
            {
                "user": "Describe what you see in this image, focusing on the human activity.",
                "assistant": "I see a person in a seated position holding a musical instrument - specifically a guitar. The person's hands are positioned on the strings, suggesting they are actively playing. The setting appears to be indoors.",
            },
            {
                "user": "Describe what you see in this image, focusing on the human activity.",
                "assistant": "I observe a person on a two-wheeled vehicle - a bicycle. The person is in a riding position with their feet on pedals. The vehicle has handlebars and the person is outdoors, moving on a road.",
            },
        ]
    elif task == "ESNLI-VE" and hypothesis:
        description_examples = [
            {
                "user": "Hypothesis: A person is skiing. Describe what you see in this image, focusing on elements relevant to the hypothesis.",
                "assistant": "I see a person on a snowy slope. The person is wearing skis and holding poles. The environment is clearly a winter mountain setting with snow-covered ground. The person is positioned in a skiing stance.",
            },
            {
                "user": "Hypothesis: The man is riding a bicycle indoors. Describe what you see in this image, focusing on elements relevant to the hypothesis.",
                "assistant": "I observe a person on a bicycle. The person is in a riding position. However, the background shows trees, sky, and an outdoor environment, not indoor structures. The setting is clearly outdoors.",
            },
        ]
    elif task == "VCR" and question and choices:
        padded = (choices + ["missing"] * 4)[:4]
        description_examples = [
            {
                "user": f"Question: {question} Options: A) {padded[0]} B) {padded[1]} C) {padded[2]} D) {padded[3]}\nDescribe what you see in this image, focusing on elements relevant to the question.",
                "assistant": "I see a person in the image. The person's facial expression shows a smile, indicating positive emotion. The person appears relaxed and content. The setting appears to be outdoors with natural lighting.",
            },
        ]
    else:
        # Generic examples if no task-specific context
        description_examples = [
            {
                "user": "Describe what you see in this image in a few sentences.",
                "assistant": "I see a person sitting on a bench in a park. There is a dog nearby, and trees in the background. The scene appears to be outdoors during daytime.",
            },
        ]
    
    if add_fewshot_examples:
        add_fewshot_examples(conversation, description_examples, k)
    
    # Build instructions based on task
    if task == "VQA-X" and question:
        instructions = (
            f"Question: {question}\n\n"
            f"Describe what you see in this image, focusing on elements relevant to the question above. "
            f"Provide a detailed description that will help answer the question."
        )
    elif task == "ACT-X":
        instructions = (
            "Describe what you see in this image, focusing on the human activity. "
            "Provide a detailed description of what the person is doing."
        )
    elif task == "ESNLI-VE" and hypothesis:
        instructions = (
            f"Hypothesis: {hypothesis}\n\n"
            f"Describe what you see in this image, focusing on elements relevant to the hypothesis above. "
            f"Provide a detailed description that will help evaluate whether the hypothesis is true."
        )
    elif task == "VCR" and question and choices:
        padded = (choices + ["missing"] * 4)[:4]
        instructions = (
            f"Question: {question}\n"
            f"Options: A) {padded[0]} B) {padded[1]} C) {padded[2]} D) {padded[3]}\n\n"
            f"Describe what you see in this image, focusing on elements relevant to the question above. "
            f"Provide a detailed description that will help choose the correct answer."
        )
    else:
        instructions = "Describe what you see in this image in a few sentences. Focus on the main objects, people, activities, and setting."
    
    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": instructions},
            {"type": "image"},
        ],
    })
    return conversation


def prompt_question_from_description_cot_vqax(description: str, question: str):
    """
    Stage 2 of CoT for VQA-X: Answer question based on image description.
    Returns a conversation continuation (text only, no image).
    """
    instructions = (
        f"Based on the following image description, answer the question:\n\n"
        f"Image description: {description}\n\n"
        f"Question: {question}\n\n"
        f"Think step by step and provide your answer in this format:\n"
        f"<explanation> Therefore, the answer is: <answer>"
    )
    
    return {
        "role": "user",
        "content": [{"type": "text", "text": instructions}],
    }


def prompt_question_from_description_cot_actx(description: str):
    """
    Stage 2 of CoT for ACT-X: Identify activity based on image description.
    Returns a conversation continuation (text only, no image).
    """
    instructions = (
        f"Based on the following image description, identify the activity:\n\n"
        f"Image description: {description}\n\n"
        f"Think step by step and provide your answer in this format:\n"
        f"<explanation> Therefore, the activity is: <activity>"
    )
    
    return {
        "role": "user",
        "content": [{"type": "text", "text": instructions}],
    }


def prompt_question_from_description_cot_esnlive(description: str, hypothesis: str):
    """
    Stage 2 of CoT for ESNLI-VE: Evaluate hypothesis based on image description.
    Returns a conversation continuation (text only, no image).
    """
    instructions = (
        f"Based on the following image description, evaluate the hypothesis:\n\n"
        f"Image description: {description}\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"Think step by step and provide your answer in this format:\n"
        f"<explanation> Therefore, the answer is: <label>\n"
        f"Label must be: entailment, contradiction, or neutral."
    )
    
    return {
        "role": "user",
        "content": [{"type": "text", "text": instructions}],
    }


def prompt_question_from_description_cot_vcr(description: str, question: str, choices: List[str]):
    """
    Stage 2 of CoT for VCR: Answer question based on image description.
    Returns a conversation continuation (text only, no image).
    """
    padded = (choices + ["missing"] * 4)[:4]
    
    instructions = (
        f"Based on the following image description, answer the question:\n\n"
        f"Image description: {description}\n\n"
        f"Question: {question}\n"
        f"A) {padded[0]}\n"
        f"B) {padded[1]}\n"
        f"C) {padded[2]}\n"
        f"D) {padded[3]}\n\n"
        f"Think step by step and provide your answer in this format:\n"
        f"<explanation> Therefore, the answer is: <letter>\n"
        f"Letter must be A, B, C, or D."
    )
    
    return {
        "role": "user",
        "content": [{"type": "text", "text": instructions}],
    }

