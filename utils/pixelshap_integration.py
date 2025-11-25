# utils/pixelshap_integration.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Optional, Any

import torch
from PIL import Image
import numpy as np

from sentence_transformers import SentenceTransformer


@dataclass
class VLMConfig:
    model: Any
    processor: Any
    device: str = "cuda"
    max_new_tokens: int = 40
    task: str = "VQA-X"


def _is_qwen_model(model) -> bool:
    """
    Minimal helper to detect Qwen3-VL models, if you want to extend later.
    """
    return model.__class__.__name__.startswith("Qwen3VLForConditionalGeneration")


class SentenceTransformerVectorizer:
    """
    Simple vectorizer wrapper for PixelSHAP.

    Exposes:
      - vectorize(texts) -> np.ndarray
      - calculate_similarity(base_vector, comparison_vectors) -> np.ndarray
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda",
    ):
        # SentenceTransformer handles "cuda" / "cpu" internally
        self.model = SentenceTransformer(model_name, device=device)

    def vectorize(self, texts):
        """
        texts: list of strings or a single string.

        Returns:
            NumPy array of shape (N, D).
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def calculate_similarity(self, base_vector, comparison_vectors):
        """
        Compute cosine similarity between a single base vector and
        a set of comparison vectors.

        Args:
            base_vector: np.ndarray or tensor of shape (D,) or (1, D)
            comparison_vectors: np.ndarray or tensor of shape (N, D)

        Returns:
            np.ndarray of shape (N,) with cosine similarities.
        """
        base = np.asarray(base_vector).reshape(1, -1)
        comps = np.asarray(comparison_vectors)

        # Ensure 2D for comparison vectors
        if comps.ndim == 1:
            comps = comps.reshape(1, -1)

        # Normalize
        base_norm = np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
        comps_norm = np.linalg.norm(comps, axis=1, keepdims=True) + 1e-12

        base_unit = base / base_norm
        comps_unit = comps / comps_norm

        # Cosine similarity: (N, D) @ (D, 1) -> (N, 1)
        sims = np.matmul(comps_unit, base_unit.T).reshape(-1)
        return sims


class VLMWrapper:
    """
    Wrapper that adapts your VLM (LLaVA / Qwen) to the interface expected by PixelSHAP.

    PixelSHAP expects the model object to support:
        - __call__(image_path=str, prompt=str) -> str
        - generate(...) -> str
    """

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self.model = cfg.model
        self.processor = cfg.processor
        self.device = cfg.device
        self.max_new_tokens = cfg.max_new_tokens

    def __call__(self, image_path: str, prompt: str) -> str:
        """
        Run VLM inference: image + prompt -> generated text.

        Minimal LLaVA path for text generation only.
        """

        img = Image.open(image_path).convert("RGB")

        # For now, only support LLaVA here.
        if _is_qwen_model(self.model):
            raise NotImplementedError(
                "PixelSHAP VLMWrapper currently only supports LLaVA. "
                "Qwen support can be added by extending this method."
            )

        # ----------------------
        # LLaVA path
        # ----------------------
        # IMPORTANT: include an image placeholder so that apply_chat_template
        # inserts the <image> token. Otherwise LLaVA sees features but no image tokens.
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},  # image placeholder -> <image> token
                ],
            }
        ]

        chat_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        # dtype: float16 on GPU, float32 on CPU
        model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        inputs = self.processor(
            images=img,
            text=chat_prompt,
            return_tensors="pt",
        ).to(self.device, model_dtype)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=True,
            )

        gen_only = out.sequences[0, input_len:]
        text = self.processor.decode(gen_only, skip_special_tokens=True).strip()
        return text

    def generate(self, *args, **kwargs) -> str:
        """
        TokenSHAP / BaseSHAP call `model.generate(...)`.
        We forward to __call__() using the same argument names.
        """
        # PixelSHAP usually passes image_path and prompt as keyword args
        image_path = kwargs.get("image_path")
        prompt = kwargs.get("prompt")

        # Fallback: positional args if provided as (image_path, prompt)
        if image_path is None and len(args) >= 1:
            image_path = args[0]
        if prompt is None and len(args) >= 2:
            prompt = args[1]

        if image_path is None or prompt is None:
            raise ValueError(
                "VLMWrapper.generate expects at least (image_path, prompt). "
                f"Got image_path={image_path}, prompt={prompt}"
            )

        return self.__call__(image_path=image_path, prompt=prompt)


def build_pixelshap(
    vlm_cfg: VLMConfig,
    segmentation_model: Any,
    manipulator: Any,
    vectorizer: Optional[Any] = None,
    temp_dir: str = "pixelshap_tmp",
    debug: bool = False,
):
    """
    Construct and return a PixelSHAP object.
    """

    try:
        from token_shap.pixel_shap import PixelSHAP
    except ImportError as e:
        raise ImportError(
            "PixelSHAP is not available. Ensure the TokenSHAP repo is cloned and installed:\n"
            "    pip install -e TokenSHAP"
        ) from e

    vlm_wrapper = VLMWrapper(vlm_cfg)

    # If no vectorizer is provided, use a SentenceTransformer-based one.
    if vectorizer is None:
        vectorizer = SentenceTransformerVectorizer(
            model_name="sentence-transformers/all-mpnet-base-v2",
            device=vlm_cfg.device,
        )

    pixel_shap = PixelSHAP(
        model=vlm_wrapper,
        segmentation_model=segmentation_model,
        manipulator=manipulator,
        vectorizer=vectorizer,
        temp_dir=temp_dir,
        debug=debug,
    )
    return pixel_shap


def run_pixelshap_for_token(
    pixel_shap,
    image_path: str,
    base_prompt: str,
    token: str,
    out_dir: str,
    sampling_ratio: float = 0.5,
    max_combinations: int = 20,
    image_id: Optional[Any] = None,
    question: Optional[str] = None,
    model_answer: Optional[str] = None,
    gt_answer: Optional[str] = None,
) -> str:
    """
    Run PixelSHAP for a specific explanation token and save an overlay image.

    The overlays are stored in a per-image subdirectory of `out_dir`.
    Additionally, a meta.json file with image/question/answer information
    is written once per image.

    Returns:
        Path to the saved overlay image.
    """
    # base output dir for all images (already created by caller)
    os.makedirs(out_dir, exist_ok=True)

    # create per-image directory
    image_basename = os.path.splitext(os.path.basename(image_path))[0]

    if image_id is not None:
        image_dir_name = f"{image_basename}_id{image_id}"
    else:
        image_dir_name = image_basename

    image_dir = os.path.join(out_dir, image_dir_name)
    os.makedirs(image_dir, exist_ok=True)

    # write / update metadata file once per image
    meta_path = os.path.join(image_dir, "meta.json")
    if not os.path.exists(meta_path):
        meta = {
            "image_path": image_path,
            "image_id": image_id,
            "question": question,
            "model_answer": model_answer,
            "ground_truth_answer": gt_answer,
        }
        try:
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Could not write meta.json for {image_dir}: {e}")

    # build prompt for this specific token
    prompt = (
        base_prompt
        + f"\n\nExplain the image and especially the concept '{token}'."
    )

    # run PixelSHAP
    results_df, shapley_values = pixel_shap.analyze(
        image_path=image_path,
        prompt=prompt,
        sampling_ratio=sampling_ratio,
        max_combinations=max_combinations,
        cleanup_temp_files=True,
    )

    # output path for the overlay image
    out_path = os.path.join(
        image_dir,
        f"{image_basename}__token_{token}.png",
    )

    # visualize overlay
    pixel_shap.visualize(
        background_opacity=0.5,
        show_original_side_by_side=True,
        show_labels=False,
        show_model_output=True,
        output_path=out_path,
    )

    return out_path


def run_pixelshap_for_image(
    vlm_cfg: VLMConfig,
    answer_tokens: list,
    image_path: str,
    base_prompt: str,
    token: str,
    out_dir: str,
    sampling_ratio: float = 0.5,
    max_combinations: int = 20,
    image_id: Optional[Any] = None,
    question: Optional[str] = None,
    model_answer: Optional[str] = None,
    gt_answer: Optional[str] = None,
    vectorizer: Optional[Any] = None,
    temp_dir: str = "pixelshap_tmp",
    debug: bool = False,
) -> str:
    """
    Run PixelSHAP for an image and save an overlay image.
    Creates a new segmentation model for each image based on the answer tokens.
    Similar to run_pixelshap_for_token, but uses a generic filename (overlay.png)
    instead of token-specific naming.

    The overlay is stored in the provided directory along with meta.json.

    Args:
        vlm_cfg: VLMConfig containing model, processor, device, etc.
        answer_tokens: List of tokens from the generated answer/explanation
        image_path: Path to the image
        base_prompt: Base prompt for the VLM
        token: Specific token to focus on in the explanation
        out_dir: Output directory for the overlay
        sampling_ratio: Sampling ratio for PixelSHAP
        max_combinations: Maximum combinations for PixelSHAP
        image_id: Optional image ID
        question: Optional question text
        model_answer: Optional model answer
        gt_answer: Optional ground truth answer
        vectorizer: Optional vectorizer (if None, uses default)
        temp_dir: Temporary directory for PixelSHAP
        debug: Debug mode flag

    Returns:
        Path to the saved overlay image.
    """
    from utils.pixelshap_setup import build_segmentation_model, build_manipulator
    
    # base output dir for all images (already created by caller)
    os.makedirs(out_dir, exist_ok=True)

    # write / update metadata file once per image
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        meta = {
            "image_path": image_path,
            "image_id": image_id,
            "question": question,
            "model_answer": model_answer,
            "ground_truth_answer": gt_answer,
        }
        try:
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Could not write meta.json for {out_dir}: {e}")

    # Build segmentation model with answer tokens
    segmentation_model = build_segmentation_model(
        answer_tokens=answer_tokens,
        device=vlm_cfg.device,
    )
    
    # Build manipulator
    manipulator = build_manipulator(device=vlm_cfg.device)
    
    # Build PixelSHAP with the new segmentation model
    pixel_shap = build_pixelshap(
        vlm_cfg=vlm_cfg,
        segmentation_model=segmentation_model,
        manipulator=manipulator,
        vectorizer=vectorizer,
        temp_dir=temp_dir,
        debug=debug,
    )

    # build prompt for this specific token
    prompt = (
        base_prompt
        + f"\n\nExplain the image and especially the concept '{token}'."
    )

    # run PixelSHAP
    results_df, shapley_values = pixel_shap.analyze(
        image_path=image_path,
        prompt=prompt,
        sampling_ratio=sampling_ratio,
        max_combinations=max_combinations,
        cleanup_temp_files=True,
    )

    # output path for the overlay image (generic name: overlay.png)
    out_path = os.path.join(out_dir, "overlay.png")

    # visualize overlay
    pixel_shap.visualize(
        background_opacity=0.5,
        show_original_side_by_side=True,
        show_labels=False,
        show_model_output=True,
        output_path=out_path,
    )

    return out_path
