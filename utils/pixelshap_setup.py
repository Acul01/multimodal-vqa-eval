# utils/pixelshap_setup.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional

from token_shap.image_utils import (
    DinoSam2SegmentationModel,
    BlackoutSegmentationManipulator,
)


def build_segmentation_model(answer_tokens: list, device: str = "cuda"):
    """
    Build and return the segmentation model used by PixelSHAP.
    This follows the PixelSHAP example: DinoSam2SegmentationModel
    wraps GroundingDINO + SAM2.

    The 'text_prompt' is created from the answer tokens, which are
    joined by commas to form a search query for object/concept detection.

    Args:
        answer_tokens: List of tokens from the generated answer/explanation
        device: Device to run the model on (default: "cuda")
    
    Returns:
        DinoSam2SegmentationModel instance
    """
    # Convert tokens to comma-separated string
    # Filter out empty tokens and join with commas
    filtered_tokens = [str(t).strip() for t in answer_tokens if t and str(t).strip()]
    
    text_prompt = ", ".join(filtered_tokens)

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