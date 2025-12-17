"""
CoT 2-Stage Prompting Template (Variant 1)

This variant uses a two-stage approach:
- Stage 1: Generate image description (with image)
- Stage 2: Answer question based on description (text only, no image)
"""

from typing import List, Dict


def prompt_image_description_cot(prompt_mode: str = "zero", resolve_shot_count=None, add_fewshot_examples=None):
    """
    Stage 1 of CoT: Generate a description of the image.
    Returns a conversation with just the image description prompt.
    
    Args:
        prompt_mode: Prompt mode (zero, 1shot, 3shot, 6shot)
        resolve_shot_count: Function to resolve shot count from prompt_mode
        add_fewshot_examples: Function to add few-shot examples to conversation
    """
    k = resolve_shot_count(prompt_mode) if resolve_shot_count else 0
    conversation: List[Dict] = []
    
    # Few-shot examples for image description
    description_examples = [
        {
            "user": "Describe what you see in this image in a few sentences.",
            "assistant": "I see a person sitting on a bench in a park. There is a dog nearby, and trees in the background. The scene appears to be outdoors during daytime.",
        },
        {
            "user": "Describe what you see in this image in a few sentences.",
            "assistant": "The image shows a kitchen with a person standing near a stove. There are cooking utensils and pans visible. The setting is clearly indoors.",
        },
        {
            "user": "Describe what you see in this image in a few sentences.",
            "assistant": "I observe a car on a road. The vehicle is yellow in color. There are buildings in the background and the scene is outdoors.",
        },
        {
            "user": "Describe what you see in this image in a few sentences.",
            "assistant": "The image contains a person reading a book while seated. The person appears focused on the book. The background is indoors with furniture visible.",
        },
        {
            "user": "Describe what you see in this image in a few sentences.",
            "assistant": "I see three people in the scene. They are standing together outdoors. There are trees and open space in the background.",
        },
        {
            "user": "Describe what you see in this image in a few sentences.",
            "assistant": "The image shows a child holding a phone. The child is looking at the device. The setting appears to be indoors with a room visible in the background.",
        },
    ]
    
    if add_fewshot_examples:
        add_fewshot_examples(conversation, description_examples, k)
    
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

