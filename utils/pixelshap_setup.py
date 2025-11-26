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

    The 'text_prompt' is created from the complete generated answer.
    Tokens are extracted from the answer and joined by commas to form
    a search query for object/concept detection.

    Args:
        generated_answer: Complete generated answer string (e.g., "<answer> because <explanation>")
        device: Device to run the model on (default: "cuda")
    
    Returns:
        DinoSam2SegmentationModel instance
    """
    import re
    
    # Extract all alphanumeric tokens from the complete answer
    # This includes tokens from both the answer and explanation parts
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
    
    # Fallback if no tokens found
    if not text_prompt:
        text_prompt = "person, man, woman, child, dog, cat, car, bus, bike"

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