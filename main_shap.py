#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from utils.run_vqa_tasks import run_vqa_task
from utils.pixelshap_integration import VLMConfig, build_pixelshap
from utils.pixelshap_setup import build_segmentation_model, build_manipulator


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    images_root = os.path.join(project_root, "images")
    nle_root = os.path.join(project_root, "nle_data")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----- Load VLM (example: LLaVA 1.5 7B) -----
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model_dtype = torch.float16 if device == "cuda" else torch.float32

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    # ----- Note: Segmentation model is now created per-image in run_pixelshap_for_image -----
    # ----- based on the generated answer tokens. The initial pixel_shap object is -----
    # ----- kept for compatibility but a new one is created per image. -----
    
    # ----- Wrap VLM for PixelSHAP -----
    vlm_cfg = VLMConfig(
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=40,
        task="VQA-X",
    )

    # Create a dummy pixel_shap object for compatibility (not used, but kept for API)
    # The actual pixel_shap is created per-image in run_pixelshap_for_image
    dummy_segmentation_model = build_segmentation_model(
        answer_tokens=["person", "man", "woman"],  # dummy tokens
        device=device
    )
    manipulator = build_manipulator(device=device)
    pixel_shap = build_pixelshap(
        vlm_cfg=vlm_cfg,
        segmentation_model=dummy_segmentation_model,
        manipulator=manipulator,
        vectorizer=None,  # as recommended for VLM-style models
        temp_dir=os.path.join(project_root, "pixelshap_tmp"),
        debug=False,
    )

    # ----- Where to save overlay images -----
    pixelshap_out_dir = os.path.join(
        project_root,
        "results",
        "llava_vqax_3shot",
    )

    # ----- Run VQA-X SHAP experiment -----
    results = run_vqa_task(
        task="VQA-X",
        model=model,
        processor=processor,
        images_root=images_root,
        nle_root=nle_root,
        split="val",
        n_samples=5,
        prompt_mode="3shot",
        pixel_shap=pixel_shap,
        pixelshap_out_dir=pixelshap_out_dir,
        max_tokens_pixelshap=None,  # None means use all tokens
    )

    print(f"Finished SHAP run with {len(results)} samples.")
    print(f"Overlays saved under: {pixelshap_out_dir}")


if __name__ == "__main__":
    main()