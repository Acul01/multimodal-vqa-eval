# utils/pixelshap_setup.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional

from token_shap.image_utils import (
    DinoSam2SegmentationModel,
    BlackoutSegmentationManipulator,
)


def build_segmentation_model(generated_answer: str, device: str = "cuda"):
    """
    Build and return the segmentation model used by PixelSHAP.
    This follows the PixelSHAP example: DinoSam2SegmentationModel
    wraps GroundingDINO + SAM2.

    The 'text_prompt' can be either:
    - A complete generated answer string (e.g., "<answer> because <explanation>") - tokens will be extracted
    - A comma-separated string of tokens (e.g., "red, car, is") - used directly

    Args:
        generated_answer: Either complete answer string or comma-separated tokens
        device: Device to run the model on (default: "cuda")
    
    Returns:
        DinoSam2SegmentationModel instance
    """
    import re
    
    # Check if it's already a comma-separated list of tokens
    # (simple heuristic: if it contains commas and no "because", treat as token list)
    if "," in generated_answer and "because" not in generated_answer.lower():
        # Already a comma-separated token list
        text_prompt = generated_answer
    else:
        # Extract tokens from complete answer
        tokens = re.findall(r"[A-Za-z0-9]+", generated_answer)
        
        # Filter out empty tokens and convert to lowercase
        filtered_tokens = [t.lower().strip() for t in tokens if t and t.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in filtered_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        # Join tokens with commas
        text_prompt = ", ".join(unique_tokens)
    
    print(f"Text prompt: {text_prompt}")

    segmentation_model = DinoSam2SegmentationModel(
        text_prompt=text_prompt,
        device=device,
    )
    return segmentation_model


def build_manipulator(segmentation_model=None, device: str = "cuda"):
    """
    Build and return the image manipulator used for PixelSHAP.
    IMPORTANT:
      - BlackoutSegmentationManipulator does NOT accept a segmentation_model
        argument. It only takes mask configuration parameters.
    """

    manipulator = BlackoutSegmentationManipulator(
        mask_type="bbox",
        preserve_overlapping=True,
        # add other parameters here if required by your TokenSHAP version
    )

    return manipulator