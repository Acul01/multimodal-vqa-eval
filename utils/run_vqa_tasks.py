# utils/run_vqa_tasks.py
# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Optional
from PIL import Image
import torch
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
    prompt_image_description_cot,
    prompt_question_from_description_cot_vqax,
    prompt_question_from_description_cot_actx,
    prompt_question_from_description_cot_esnlive,
    prompt_question_from_description_cot_vcr,
)

# Unified postprocessing
from utils.postprocessing import (
    postprocess_prediction,
    normalize_answer as normalize_ans,
)

# Token entropy functions
from utils.token_entropies import (
    extract_token_entropies,
    filter_entropy_to_explanation,
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


# -----------------------------
# CoT 2-Stage generation (returns text + token_entropy)
# -----------------------------
def generate_answer_cot_2stage(
    model,
    processor,
    image_path: str,
    task: str,
    prompt_mode: str,
    question: Optional[str] = None,
    hypothesis: Optional[str] = None,
    choices: Optional[List[str]] = None,
    max_new_tokens: int = 40,
    device: str = "cuda",
    use_question_in_stage1: bool = False,
):
    """
    CoT 2-Stage generation with proper message list building:
    
    Stage 1: [{system message}, {user message 1, image}] -> model -> assistant response 1 -> append
    Stage 2: [{system message}, {user message 1, image}, {assistant response 1}, {user message 2}] -> model -> assistant response 2
    
    Args:
        use_question_in_stage1: If True, use Variant 2 (question in Stage 1), else Variant 1 (default)
    
    Returns:
        text (str), token_entropy (Dict[str, float])
    """
    # Build initial message list for Stage 1: [{system message}, {user message 1, image}]
    stage1_conv = prompt_image_description_cot(
        prompt_mode=prompt_mode,
        question=question,
        hypothesis=hypothesis,
        choices=choices,
        task=task,
        use_question_in_stage1=use_question_in_stage1,
    )
    
    # Stage 1: Generate image description
    # Input: [{system message}, {user message 1, image}]
    description, _ = generate_answer(model, processor, image_path, stage1_conv, max_new_tokens=60, device=device)
    
    # Append assistant response 1 to message list
    # Now we have: [{system message}, {user message 1, image}, {assistant response 1}]
    stage2_conv = stage1_conv.copy()
    stage2_conv.append({
        "role": "assistant",
        "content": [{"type": "text", "text": description}],
    })
    
    # Stage 2: Generate answer based on description
    # Build user message 2 (text only, no image)
    if task == "VQA-X":
        user_message_2 = prompt_question_from_description_cot_vqax(description, question or "")
    elif task == "ACT-X":
        user_message_2 = prompt_question_from_description_cot_actx(description)
    elif task == "ESNLI-VE":
        user_message_2 = prompt_question_from_description_cot_esnlive(description, hypothesis or "")
    elif task == "VCR":
        user_message_2 = prompt_question_from_description_cot_vcr(description, question or "", choices or [])
    else:
        raise ValueError(f"Unknown task for CoT 2-stage: {task}")
    
    # Append user message 2
    # Now we have: [{system message}, {user message 1, image}, {assistant response 1}, {user message 2}]
    stage2_conv.append(user_message_2)
    
    # Stage 2: Generate final answer
    # Input: [{system message}, {user message 1, image}, {assistant response 1}, {user message 2}]
    # Output: assistant response 2
    answer, token_entropy = generate_answer_with_message_list(
        model, processor, image_path, stage2_conv, max_new_tokens=max_new_tokens, device=device
    )
    
    # Return both description (Stage 1) and answer (Stage 2) for CoT
    # Format: ((description, answer), token_entropy)
    return (description, answer), token_entropy


def generate_answer_with_message_list(
    model,
    processor,
    image_path: str,
    conversation: List[Dict],
    max_new_tokens: int = 40,
    device: str = "cuda",
):
    """
    Generate answer from a conversation message list that may already contain assistant responses.
    This is used for Stage 2 where the conversation already includes the Stage 1 response.
    
    The conversation should have the structure:
    [{system message}, {user message 1, image}, {assistant response 1}, {user message 2}]
    
    Returns:
        text (str), token_entropy (Dict[str, float])
    """
    img = Image.open(image_path).convert("RGB")

    # --------------------------------------------------
    # Qwen3-VL-Zweig
    # --------------------------------------------------
    if _is_qwen_model(model):
        # Convert conversation to messages format, preserving structure
        messages = []
        for item in conversation:
            if item["role"] == "system":
                messages.append({"role": "system", "content": item.get("content", "")})
            elif item["role"] == "user":
                # Extract text and image from user content
                text_parts = []
                has_image = False
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                    elif content_item.get("type") == "image":
                        has_image = True
                
                if has_image:
                    # User message with image
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": " ".join(text_parts)},
                            {"type": "image", "image": img},
                        ],
                    })
                else:
                    # User message without image (text only)
                    messages.append({"role": "user", "content": " ".join(text_parts)})
            elif item["role"] == "assistant":
                # Extract text from assistant content
                text_parts = []
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                if text_parts:
                    messages.append({"role": "assistant", "content": " ".join(text_parts)})
        
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

        gen_only = out.sequences[0, input_len:]
        text = processor.decode(gen_only, skip_special_tokens=True).strip()

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all

    # --------------------------------------------------
    # LLaVA-Zweig
    # --------------------------------------------------
    else:
        # For LLaVA, we need to build the prompt from the conversation
        # LLaVA expects a text prompt and an image
        # We'll use the original image for the prompt
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

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all


def generate_answer_text_only(
    model,
    processor,
    conversation,
    max_new_tokens: int = 40,
    device: str = "cuda",
):
    """
    Generate answer from text-only conversation (no image).
    Used for Stage 2 of CoT.
    
    Returns:
        text (str), token_entropy (Dict[str, float])
    """
    # --------------------------------------------------
    # Qwen3-VL-Zweig
    # --------------------------------------------------
    if _is_qwen_model(model):
        # For Qwen, convert conversation to messages format (text only, no images)
        messages = []
        for item in conversation:
            if item["role"] == "user":
                # Extract text from user content (skip images)
                text_parts = []
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                    # Skip image content items
                if text_parts:
                    messages.append({"role": "user", "content": " ".join(text_parts)})
            elif item["role"] == "assistant":
                # Extract text from assistant content
                text_parts = []
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                if text_parts:
                    messages.append({"role": "assistant", "content": " ".join(text_parts)})
        
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

        gen_only = out.sequences[0, input_len:]
        text = processor.decode(gen_only, skip_special_tokens=True).strip()

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all

    # --------------------------------------------------
    # LLaVA-Zweig
    # --------------------------------------------------
    else:
        # For LLaVA, we need to provide an image, but we can use a minimal dummy image
        # since the conversation is text-only. Alternatively, we could reuse the original image.
        # For now, use a small white image as placeholder
        dummy_img = Image.new('RGB', (224, 224), color='white')
        
        # Convert conversation to text prompt (extract text, ignore images)
        text_parts = []
        for item in conversation:
            if item["role"] == "user":
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
            elif item["role"] == "assistant":
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
        
        # Build a simple text prompt from the conversation
        prompt = "\n".join(text_parts)
        
        inputs = processor(images=dummy_img, text=prompt, return_tensors="pt").to(device, torch.float16)
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

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all


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
        # Convert conversation to messages format, preserving system messages
        messages = []
        for item in conversation:
            if item["role"] == "system":
                # System message
                content = item.get("content", "")
                if isinstance(content, list):
                    # Extract text from content list
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else ""
                messages.append({"role": "system", "content": content})
            elif item["role"] == "user":
                # User message - ensure content is a list format for Qwen3-VL
                user_content = item.get("content", [])
                if isinstance(user_content, str):
                    # Convert string to list format
                    user_content = [{"type": "text", "text": user_content}]
                elif not isinstance(user_content, list):
                    # Fallback: wrap in list
                    user_content = [{"type": "text", "text": str(user_content)}]
                messages.append({"role": "user", "content": user_content})
            elif item["role"] == "assistant":
                # Assistant message
                content = item.get("content", "")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else ""
                messages.append({"role": "assistant", "content": content})
        
        # Inject image into messages
        messages = _inject_image_into_messages(messages, img)
        
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

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all

    # --------------------------------------------------
    # LLaVA-Zweig
    # --------------------------------------------------
    else:
        # LLaVA's apply_chat_template should handle system messages automatically
        # Filter out system messages if needed, or let the processor handle them
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

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all


def generate_answer_text_only(
    model,
    processor,
    conversation,
    max_new_tokens: int = 40,
    device: str = "cuda",
):
    """
    Text-only generation (no image provided).
    Useful for ablations like VCR --no_images.
    Returns:
        text (str), token_entropy (Dict[str, float])
    """
    # --------------------------------------------------
    # Qwen3-VL-Zweig (text-only)
    # --------------------------------------------------
    if _is_qwen_model(model):
        messages = []
        for item in conversation:
            if item["role"] == "system":
                content = item.get("content", "")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else ""
                messages.append({"role": "system", "content": content})
            elif item["role"] == "user":
                text_parts = []
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                messages.append({"role": "user", "content": " ".join(text_parts).strip()})
            elif item["role"] == "assistant":
                content = item.get("content", "")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else ""
                messages.append({"role": "assistant", "content": content})

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

        gen_only = out.sequences[0, input_len:]
        text = processor.decode(gen_only, skip_special_tokens=True).strip()

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all

    # --------------------------------------------------
    # LLaVA-Zweig (text-only)
    # --------------------------------------------------
    else:
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors="pt").to(device, torch.float16)
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

        token_entropy_all = extract_token_entropies(
            out,
            tokenizer=processor.tokenizer,
            input_len=input_len,
        )

        return text, token_entropy_all


def generate_answer_batch_text_only(
    model,
    processor,
    conversations: List,
    max_new_tokens: int = 40,
    device: str = "cuda",
):
    """
    Batch text-only generation.
    Returns list of (text, token_entropy) tuples.
    """
    if _is_qwen_model(model):
        # Qwen3-VL: do sequential (safe) but avoid any image processing
        results = []
        eos_token_id = getattr(processor, "eos_token_id", None) or processor.tokenizer.eos_token_id

        # left padding for decoder-only
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
            original_padding_side = processor.tokenizer.padding_side
            processor.tokenizer.padding_side = "left"

        try:
            for conv in conversations:
                text, token_entropy = generate_answer_text_only(
                    model, processor, conv, max_new_tokens=max_new_tokens, device=device
                )
                results.append((text, token_entropy))
        finally:
            if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
                processor.tokenizer.padding_side = original_padding_side

        return results

    else:
        # LLaVA: true batching with text-only processor call
        prompts = [processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
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

        batch_size = out.sequences.shape[0]
        results = []
        for i in range(batch_size):
            out_seq = out.sequences[i]
            gen_only = out_seq[input_len:]
            text = processor.decode(gen_only, skip_special_tokens=True).strip()

            single_out = type("obj", (object,), {
                "sequences": out_seq.unsqueeze(0),
                "scores": tuple(s[i:i+1] for s in out.scores) if hasattr(out, "scores") and out.scores else None,
            })()
            token_entropy = extract_token_entropies(single_out, processor.tokenizer, input_len)
            results.append((text, token_entropy))

        return results

# -----------------------------
# Batch generation (returns list of (text, token_entropy) tuples)
# -----------------------------
def generate_answer_batch(
    model,
    processor,
    image_paths: List[str],
    conversations: List,
    max_new_tokens: int = 40,
    device: str = "cuda",
):
    """
    Batch version of generate_answer - processes multiple images at once.
    Returns:
        List of (text, token_entropy) tuples, one per input
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]
    batch_size = len(images)
    
    if _is_qwen_model(model):
        # Qwen3-VL: Process sequentially (Qwen3-VL requires image_grid_thw which is complex to batch)
        # But we organize it efficiently by pre-loading images
        results = []
        eos_token_id = getattr(processor, "eos_token_id", None) or processor.tokenizer.eos_token_id
        
        # Set tokenizer to left-padding for decoder-only models
        if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'padding_side'):
            original_padding_side = processor.tokenizer.padding_side
            processor.tokenizer.padding_side = 'left'
        
        try:
            for img, conv in zip(images, conversations):
                messages = _inject_image_into_messages(conv, img)
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]
                
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        eos_token_id=eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                gen_only = out.sequences[0, input_len:]
                text = processor.decode(gen_only, skip_special_tokens=True).strip()
                token_entropy = extract_token_entropies(out, processor.tokenizer, input_len)
                results.append((text, token_entropy))
        finally:
            # Restore original padding side
            if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'padding_side'):
                processor.tokenizer.padding_side = original_padding_side
        
        return results
    
    else:
        # LLaVA: Native batch support
        prompts = [processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
        
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]  # After padding, all sequences have same length
        
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
        
        # Decode each sequence separately and extract entropies
        results = []
        for i in range(batch_size):
            out_seq = out.sequences[i]
            gen_only = out_seq[input_len:]
            text = processor.decode(gen_only, skip_special_tokens=True).strip()
            
            # Extract entropy for this specific sequence
            # Create a single-sequence output object for entropy extraction
            single_out = type('obj', (object,), {
                'sequences': out_seq.unsqueeze(0),
                'scores': tuple(s[i:i+1] for s in out.scores) if hasattr(out, 'scores') and out.scores else None
            })()
            
            token_entropy = extract_token_entropies(single_out, processor.tokenizer, input_len)
            results.append((text, token_entropy))
        
        return results


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
    generation_mode: str = "posthoc",
    pixel_shap=None,
    pixelshap_out_dir: Optional[str] = None,
    max_tokens_pixelshap: Optional[int] = 3,
    batch_size: int = 1,
    cot_variant: str = "1",
    no_images: bool = False,
):
    """
    Unified entry point for VQA-X, ACT-X, ESNLI-VE, VCR (answer+explanation).
    Uses prompting_templates.py for prompts.
    Optionally computes PixelSHAP overlays for explanation tokens.
    If max_tokens_pixelshap is None, uses all tokens from the explanation.
    
    Args:
        batch_size: Number of samples to process in parallel (1 = sequential processing).
                    Recommended: 8-16 for V100-32GB GPU.
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

    # For VCR ablations: allow running without images
    require_image = True
    if task == "VCR" and no_images:
        require_image = False

    if task == "ESNLI-VE" and n_samples:
        # Important: ESNLI-VE loader may be slow due to recursive image search on network FS.
        # Apply the limit during loading to avoid scanning the full dataset.
        dataset = load_esnlive(images_root, ann_root_task, split, require_image=require_image, max_samples=n_samples)
    else:
        dataset = loader_map[task](images_root, ann_root_task, split, require_image=require_image)
        if n_samples:
            dataset = dataset[:n_samples]

    print(f"Running {task} on {len(dataset)} samples with prompt_mode='{prompt_mode}', generation_mode='{generation_mode}' and batch_size={batch_size}...")
    
    # Get device from model
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if device.type == "cuda" else "cpu"

    results = []
    
    # Process in batches if batch_size > 1
    if batch_size > 1:
        # Process in batches
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]
            actual_batch_size = len(batch)
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} "
                  f"(samples {batch_start+1}-{batch_end})...")
            
            # Prepare batch data
            batch_paths = [s.image_path for s in batch]
            batch_conversations = []
            batch_gts = []
            batch_samples = []  # Store sample objects for later processing
            
            for s in batch:
                batch_samples.append(s)
                if task == "VQA-X":
                    batch_gts.append(majority_vqa_answer(s.raw.get("answers")))
                    if generation_mode == "cot":
                        # CoT 2-stage: will be handled separately
                        batch_conversations.append(None)
                    else:
                        batch_conversations.append(prompt_vqax_expl(s.question, prompt_mode, generation_mode))
                elif task == "ACT-X":
                    batch_gts.append(s.label)
                    if generation_mode == "cot":
                        batch_conversations.append(None)
                    else:
                        batch_conversations.append(prompt_actx_expl(prompt_mode, generation_mode))
                elif task == "ESNLI-VE":
                    batch_gts.append(s.label)
                    if generation_mode == "cot":
                        batch_conversations.append(None)
                    else:
                        batch_conversations.append(prompt_esnlive_expl(s.hypothesis, prompt_mode, generation_mode))
                elif task == "VCR":
                    batch_gts.append(s.answer or "")
                    if generation_mode == "cot":
                        batch_conversations.append(None)
                    else:
                        batch_conversations.append(
                            prompt_vcr_expl(
                                s.question or "",
                                s.choices or [],
                                prompt_mode,
                                generation_mode,
                                include_image=(not no_images),
                            )
                        )
            
            # Generate answers in batch
            if generation_mode == "cot":
                if task == "VCR" and no_images:
                    # Text-only VCR ablation: no 2-stage image description possible
                    convs = [
                        prompt_vcr_expl(
                            s.question or "",
                            s.choices or [],
                            prompt_mode,
                            generation_mode,
                            include_image=False,
                        )
                        for s in batch_samples
                    ]
                    batch_results = generate_answer_batch_text_only(
                        model, processor, convs, max_new_tokens=40, device=device_str
                    )
                else:
                    # CoT 2-stage: process sequentially (each sample needs 2 generations)
                    batch_results = []
                    for i, s in enumerate(batch_samples):
                        if task == "VQA-X":
                            (description, answer), token_entropy_raw = generate_answer_cot_2stage(
                                model, processor, batch_paths[i], task, prompt_mode,
                                question=s.question, max_new_tokens=40, device=device_str,
                                use_question_in_stage1=(cot_variant == "2")
                            )
                        elif task == "ACT-X":
                            (description, answer), token_entropy_raw = generate_answer_cot_2stage(
                                model, processor, batch_paths[i], task, prompt_mode,
                                max_new_tokens=40, device=device_str,
                                use_question_in_stage1=(cot_variant == "2")
                            )
                        elif task == "ESNLI-VE":
                            (description, answer), token_entropy_raw = generate_answer_cot_2stage(
                                model, processor, batch_paths[i], task, prompt_mode,
                                hypothesis=s.hypothesis, max_new_tokens=40, device=device_str,
                                use_question_in_stage1=(cot_variant == "2")
                            )
                        elif task == "VCR":
                            (description, answer), token_entropy_raw = generate_answer_cot_2stage(
                                model, processor, batch_paths[i], task, prompt_mode,
                                question=s.question or "", choices=s.choices or [],
                                max_new_tokens=40, device=device_str,
                                use_question_in_stage1=(cot_variant == "2")
                            )
                        # For CoT, we pass both description and answer to postprocessing
                        batch_results.append(((description, answer), token_entropy_raw))
            else:
                if task == "VCR" and no_images:
                    batch_results = generate_answer_batch_text_only(
                        model, processor, batch_conversations, max_new_tokens=40, device=device_str
                    )
                else:
                    batch_results = generate_answer_batch(
                        model, processor, batch_paths, batch_conversations,
                        max_new_tokens=40, device=device_str
                    )
            
            # Process each result in batch
            for i, (raw_pred_or_tuple, token_entropy_raw) in enumerate(batch_results):
                s = batch_samples[i]
                gt = batch_gts[i]
                pixelshap_paths = None
                sample_idx = batch_start + i + 1

                # Handle CoT vs post-hoc format
                if generation_mode == "cot":
                    if task == "VCR" and no_images:
                        # Single-stage text-only (no description available)
                        raw_pred = raw_pred_or_tuple
                        cot_description = None
                    else:
                        # CoT: raw_pred_or_tuple is (description, answer)
                        description, answer = raw_pred_or_tuple
                        print(f"Raw CoT Output - Description: {description}")
                        print(f"Raw CoT Output - Answer: {answer}")
                        raw_pred = answer  # Use answer for postprocessing
                        cot_description = description  # Pass description separately
                else:
                    # Post-hoc: raw_pred_or_tuple is just (raw_pred, token_entropy)
                    # For batch processing, raw_pred_or_tuple is already the raw_pred string
                    raw_pred = raw_pred_or_tuple
                    cot_description = None
                
                # Postprocess
                if task == "VQA-X":
                    result = postprocess_prediction(raw_pred, "VQA-X", generation_mode=generation_mode, description=cot_description)
                    pred_full = result["full_text"]
                    pred_only = result["answer"]
                    expl = result["explanation"]
                elif task == "ACT-X":
                    result = postprocess_prediction(raw_pred, "ACT-X", generation_mode=generation_mode, description=cot_description)
                    pred_full = result["full_text"]
                    pred_only = result["answer"]
                    expl = result["explanation"]
                elif task == "ESNLI-VE":
                    result = postprocess_prediction(raw_pred, "ESNLI-VE", generation_mode=generation_mode, description=cot_description)
                    pred_full = result["full_text"]
                    pred_only = result["answer"]
                    expl = result["explanation"]
                elif task == "VCR":
                    choices = s.choices or []
                    result = postprocess_prediction(raw_pred, "VCR", vcr_choices=choices, generation_mode=generation_mode, description=cot_description)
                    pred_full = result["full_text"]
                    pred_only = result["answer"]
                    expl = result["explanation"]
                
                token_entropy = filter_entropy_to_explanation(token_entropy_raw, expl)
                
                hit = int(normalize_ans(pred_only) == normalize_ans(gt)) if gt else None
                pred_to_store = pred_full
                
                # PixelSHAP handling (if needed) - same as before
                if task == "VQA-X" and pixel_shap is not None and pixelshap_out_dir is not None and token_entropy_raw:
                    sorted_tokens = sorted(
                        token_entropy_raw.items(), key=lambda kv: kv[1], reverse=True
                    )
                    if max_tokens_pixelshap is None:
                        selected_tokens = [t for t, H in sorted_tokens]
                    else:
                        selected_tokens = [t for t, H in sorted_tokens[:max_tokens_pixelshap]]
                    
                    base_prompt_for_pixelshap = s.question
                    img_base = os.path.splitext(os.path.basename(s.image_path))[0]
                    img_id = getattr(s, "image_id", None)
                    
                    if img_id is not None:
                        img_dir_name = f"{img_base}_id{img_id}"
                    else:
                        img_dir_name = img_base
                    
                    img_out_dir = os.path.join(pixelshap_out_dir, img_dir_name)
                    os.makedirs(img_out_dir, exist_ok=True)
                    
                    meta_path = os.path.join(img_out_dir, "meta.json")
                    meta = {
                        "sample_index": sample_idx,
                        "image_path": s.image_path,
                        "image_id": img_id,
                        "question": s.question,
                        "model_answer": pred_full,
                        "ground_truth_answer": gt,
                        "all_tokens": list(token_entropy_raw.keys()),
                        "token_entropy": token_entropy_raw,
                        "explanation_tokens": list(token_entropy.keys()) if token_entropy else [],
                    }
                    try:
                        import json
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"[WARN] Could not write meta.json: {e}")
                    
                    pixelshap_paths = []
                    if selected_tokens:
                        most_important_token = selected_tokens[0]
                        try:
                            vlm_cfg = VLMConfig(
                                model=model,
                                processor=processor,
                                device=device_str,
                                max_new_tokens=40,
                                task=task,
                            )
                            temp_dir = getattr(pixel_shap, 'temp_dir', 'pixelshap_tmp') if pixel_shap else 'pixelshap_tmp'
                            
                            out_path = run_pixelshap_for_image(
                                vlm_cfg=vlm_cfg,
                                generated_answer=pred_full,
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
                
                elif task in ["ACT-X", "ESNLI-VE", "VCR"] and pixel_shap is not None and pixelshap_out_dir is not None and token_entropy:
                    sorted_tokens = sorted(
                        token_entropy.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                    top_tokens = [t for t, H in sorted_tokens[:max_tokens_pixelshap]]
                    
                    if task == "ACT-X":
                        base_prompt_for_pixelshap = "Describe the human activity in this image."
                    elif task == "ESNLI-VE":
                        base_prompt_for_pixelshap = s.hypothesis
                    elif task == "VCR":
                        base_prompt_for_pixelshap = s.question or ""
                    
                    pixelshap_paths = []
                    for tok in top_tokens:
                        try:
                            out_path = run_pixelshap_for_token(
                                pixel_shap=pixel_shap,
                                image_path=s.image_path,
                                base_prompt=base_prompt_for_pixelshap,
                                token=tok,
                                out_dir=pixelshap_out_dir,
                            )
                            pixelshap_paths.append((tok, out_path))
                        except Exception as e:
                            print(f"[WARN] PixelSHAP failed for token '{tok}': {e}")
                
                result_dict = {
                    "task": task,
                    "split": split,
                    "idx": sample_idx,
                    "question": getattr(s, "question", None) or getattr(s, "hypothesis", None),
                    "prediction": pred_to_store,
                    "ground_truth": gt,
                    "image": s.image_path,
                    "correct": hit,
                    "prompt_mode": prompt_mode,
                    "token_entropy": token_entropy,
                    "pixelshap_overlays": pixelshap_paths,
                }
                if task == "VCR":
                    result_dict["choices"] = getattr(s, "choices", [])
                    result_dict["sample_id"] = getattr(s, "sample_id", None)
                    result_dict["no_images"] = bool(no_images)
                
                results.append(result_dict)
    else:
        # Original sequential processing (batch_size == 1)
        for i, s in enumerate(dataset, 1):

            pixelshap_paths = None  # will hold (token, overlay_path) tuples if used

            if task == "VQA-X":
                gt = majority_vqa_answer(s.raw.get("answers"))
                if generation_mode == "cot":
                    (description, answer), token_entropy_raw = generate_answer_cot_2stage(
                        model, processor, s.image_path, task, prompt_mode,
                        question=s.question, max_new_tokens=40, device=device_str,
                        use_question_in_stage1=(cot_variant == "2")
                    )
                    raw_pred = answer
                    cot_description = description
                else:
                    prompt = prompt_vqax_expl(s.question, prompt_mode, generation_mode)
                    raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)
                    cot_description = None

                result = postprocess_prediction(raw_pred, "VQA-X", generation_mode=generation_mode, description=cot_description)
                pred_full = result["full_text"]
                pred_only = result["answer"]
                expl = result["explanation"]

                token_entropy = filter_entropy_to_explanation(token_entropy_raw, expl)

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
                if generation_mode == "cot":
                    (description, answer), token_entropy_raw = generate_answer_cot_2stage(
                        model, processor, s.image_path, task, prompt_mode,
                        max_new_tokens=40, device=device_str,
                        use_question_in_stage1=(cot_variant == "2")
                    )
                    raw_pred = answer
                    cot_description = description
                else:
                    prompt = prompt_actx_expl(prompt_mode, generation_mode)
                    raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)
                    cot_description = None

                result = postprocess_prediction(raw_pred, "ACT-X", generation_mode=generation_mode, description=cot_description)
                pred_full = result["full_text"]
                pred_only = result["answer"]
                expl = result["explanation"]

                token_entropy = filter_entropy_to_explanation(token_entropy_raw, expl)

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
                if generation_mode == "cot":
                    (description, answer), token_entropy_raw = generate_answer_cot_2stage(
                        model, processor, s.image_path, task, prompt_mode,
                        hypothesis=s.hypothesis, max_new_tokens=40, device=device_str,
                        use_question_in_stage1=(cot_variant == "2")
                    )
                    raw_pred = answer
                    cot_description = description
                else:
                    prompt = prompt_esnlive_expl(s.hypothesis, prompt_mode, generation_mode)
                    raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)
                    cot_description = None

                result = postprocess_prediction(raw_pred, "ESNLI-VE", generation_mode=generation_mode, description=cot_description)
                pred_full = result["full_text"]
                label = result["answer"]
                explanation = result["explanation"]

                token_entropy = filter_entropy_to_explanation(token_entropy_raw, explanation)

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
                if generation_mode == "cot":
                    if no_images:
                        # Text-only VCR ablation: no 2-stage image description possible
                        prompt = prompt_vcr_expl(
                            s.question or "",
                            choices,
                            prompt_mode,
                            generation_mode,
                            include_image=False,
                        )
                        raw_pred, token_entropy_raw = generate_answer_text_only(
                            model, processor, prompt, max_new_tokens=40, device=device_str
                        )
                        cot_description = None
                    else:
                        (description, answer), token_entropy_raw = generate_answer_cot_2stage(
                            model, processor, s.image_path, task, prompt_mode,
                            question=s.question or "", choices=choices,
                            max_new_tokens=40, device=device_str,
                            use_question_in_stage1=(cot_variant == "2")
                        )
                        raw_pred = answer
                        cot_description = description
                else:
                    prompt = prompt_vcr_expl(
                        s.question or "",
                        choices,
                        prompt_mode,
                        generation_mode,
                        include_image=(not no_images),
                    )
                    if no_images:
                        raw_pred, token_entropy_raw = generate_answer_text_only(
                            model, processor, prompt, max_new_tokens=40, device=device_str
                        )
                    else:
                        raw_pred, token_entropy_raw = generate_answer(model, processor, s.image_path, prompt)
                    cot_description = None
                print(f"[DEBUG run_vqa_task VCR] raw_pred: {repr(raw_pred)}")
                print(f"[DEBUG run_vqa_task VCR] gt: {repr(gt)}")
                print(f"[DEBUG run_vqa_task VCR] choices: {choices}")

                result = postprocess_prediction(raw_pred, "VCR", vcr_choices=choices, generation_mode=generation_mode, description=cot_description)
                
                pred_full = result["full_text"]
                pred_answer_text = result["answer"]
                expl = result["explanation"]

                print(f"[DEBUG run_vqa_task VCR] pred_answer_text: {repr(pred_answer_text)}")
                print(f"[DEBUG run_vqa_task VCR] normalized pred: {repr(normalize_ans(pred_answer_text))}")
                print(f"[DEBUG run_vqa_task VCR] normalized gt: {repr(normalize_ans(gt))}")

                token_entropy = filter_entropy_to_explanation(token_entropy_raw, expl)

                hit = int(normalize_ans(pred_answer_text) == normalize_ans(gt)) if gt else None
                print(f"[DEBUG run_vqa_task VCR] hit: {hit}")
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

            result_dict = {
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
            }
            # Add task-specific fields
            if task == "VCR":
                result_dict["choices"] = choices
                # Add sample_id for VCR to enable unique mapping (multiple questions per image)
                result_dict["sample_id"] = getattr(s, "sample_id", None)
                result_dict["no_images"] = bool(no_images)
            results.append(result_dict)

    valid_hits = [r["correct"] for r in results if r["correct"] is not None]
    if valid_hits:
        acc = sum(valid_hits) / len(valid_hits)
        print(f"\n{task} {split} Accuracy ({prompt_mode}): {acc:.3f}")
    else:
        print(f"\n{task} {split} ({prompt_mode}): no evaluable ground truths found.")

    return results