# utils/pixelshap_setup.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional

from token_shap.image_utils import (
    DinoSam2SegmentationModel,
    BlackoutSegmentationManipulator,
)


def build_segmentation_model(device: str = "cuda"):
    """
    Build and return the segmentation model used by PixelSHAP.
    This follows the PixelSHAP example: DinoSam2SegmentationModel
    wraps GroundingDINO + SAM2.

    The 'text_prompt' controls which objects/concepts are detected.
    You can adjust it later to better fit your datasets.
    """
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