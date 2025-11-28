# utils/pixelshap_integration.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Optional, Any, List, Dict

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
    # Include the generated answer as context to ensure attribution is for the specific output
   
    prompt = (
        base_prompt
        + f"\n\nYour previous answer was: '{model_answer}'. "
        + f"Focus on explaining the concept '{token}' in this answer."
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


def run_pixelshap_for_tokens(
    vlm_cfg: VLMConfig,
    text_prompt: str,
    image_path: str,
    question: str,
    baseline_answer: str,
    tokens: List[str],
    token_entropy_dict: Dict[str, float],
    out_dir: str,
    image_id: Optional[Any] = None,
    sampling_ratio: float = 0.5,
    max_combinations: int = 20,
    vectorizer: Optional[Any] = None,
    temp_dir: str = "pixelshap_tmp",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run PixelSHAP for multiple tokens and save overlays.
    Creates one overlay per token showing which image segments are important for that token.
    Also creates a segmentation visualization showing all detected segments.
    
    Args:
        vlm_cfg: VLMConfig containing model, processor, device, etc.
        text_prompt: Comma-separated tokens for segmentation (e.g., "red, car, is")
        image_path: Path to the image
        question: Original question
        baseline_answer: Original answer from the model
        tokens: List of tokens to analyze
        token_entropy_dict: Dictionary mapping tokens to entropy values
        out_dir: Output directory for overlays
        image_id: Optional image ID
        sampling_ratio: Sampling ratio for PixelSHAP
        max_combinations: Maximum combinations for PixelSHAP
        vectorizer: Optional vectorizer (if None, uses default)
        temp_dir: Temporary directory for PixelSHAP
        debug: Debug mode flag
    
    Returns:
        Dictionary with:
            - "segmentation_overlay": path to segmentation visualization
            - "token_overlays": list of (token, overlay_path) tuples
            - "token_analyses": list of dicts with token, answer, similarity_diff for each token
    """
    from utils.pixelshap_setup import build_segmentation_model, build_manipulator
    import json
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Build segmentation model with tokens that have entropy
    segmentation_model = build_segmentation_model(
        generated_answer=text_prompt,  # Use text_prompt directly (comma-separated tokens)
        device=vlm_cfg.device,
    )
    
    # Build manipulator
    manipulator = build_manipulator(device=vlm_cfg.device)
    
    # Build PixelSHAP
    pixel_shap = build_pixelshap(
        vlm_cfg=vlm_cfg,
        segmentation_model=segmentation_model,
        manipulator=manipulator,
        vectorizer=vectorizer,
        temp_dir=temp_dir,
        debug=debug,
    )
    
    # Build vectorizer if not provided
    if vectorizer is None:
        vectorizer = SentenceTransformerVectorizer(device=vlm_cfg.device)
    
    # Get baseline embedding for comparison
    baseline_embedding = vectorizer.vectorize(baseline_answer)
    
    # Create segmentation visualization (all segments)
    seg_overlay_path = None
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        boxes, labels, conf, masks = segmentation_model.segment(image_path)
        seg_overlay_path = os.path.join(out_dir, "segmentation_overlay.png")
        
        # Create visualization
        img = Image.open(image_path).convert("RGB")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title("Detected Segments", fontsize=14)
        
        # Draw bounding boxes
        for box, label, c in zip(boxes, labels, conf):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{label} ({c:.2f})", color='red', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(seg_overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not create segmentation overlay: {e}")
    
    # Analyze each token
    token_overlays = []
    token_analyses = []
    
    # Use simple prompt (just the question)
    prompt = question
    
    # Build PixelSHAP once (shared for all tokens)
    pixel_shap = build_pixelshap(
        vlm_cfg=vlm_cfg,
        segmentation_model=segmentation_model,
        manipulator=manipulator,
        vectorizer=vectorizer,
        temp_dir=temp_dir,
        debug=debug,
    )
    
    # Run baseline first to get baseline output
    baseline_output = pixel_shap.model(image_path, prompt)
    
    # Get all segments from segmentation model
    boxes, labels, conf, masks = segmentation_model.segment(image_path)
    
    # Create a mapping from token to segments (find segments that match the token)
    token_to_segments = {}
    for token in tokens:
        matching_segments = []
        for i, label in enumerate(labels):
            # Check if token appears in label (case-insensitive)
            if token.lower() in label.lower():
                matching_segments.append(i)
        token_to_segments[token] = matching_segments if matching_segments else list(range(len(labels)))  # Fallback: all segments
    
    # Create overall overlay once (for all tokens combined)
    overall_overlay_path = os.path.join(out_dir, "overlay.png")
    try:
        # Run PixelSHAP once to get overall overlay
        results_df_overall, shapley_values_overall = pixel_shap.analyze(
            image_path=image_path,
            prompt=prompt,
            sampling_ratio=sampling_ratio,
            max_combinations=max_combinations,
            cleanup_temp_files=False,  # Keep temp files for now
        )
        pixel_shap.visualize(
            background_opacity=0.5,
            show_original_side_by_side=True,
            show_labels=False,
            show_model_output=True,
            output_path=overall_overlay_path,
        )
    except Exception as e:
        print(f"[WARN] Could not create overall overlay: {e}")
        overall_overlay_path = None
    
    for token in tokens:
        try:
            # Find segments that match this token
            token_segments = token_to_segments.get(token, [])
            
            # Create manipulated image with this token's segments blacked out
            manipulated_image_path = None
            perturbed_answer = None
            similarity_diff = None
            
            if token_segments:
                try:
                    # Load original image
                    original_img = Image.open(image_path).convert("RGB")
                    img_array = np.array(original_img)
                    
                    # Create mask for this token's segments
                    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool)
                    for seg_idx in token_segments:
                        if seg_idx < len(masks):
                            seg_mask = masks[seg_idx]
                            if seg_mask is not None:
                                # Resize mask if needed using PIL
                                if seg_mask.shape != mask.shape:
                                    from PIL import Image as PILImage
                                    seg_mask_img = PILImage.fromarray((seg_mask * 255).astype(np.uint8))
                                    seg_mask_resized = seg_mask_img.resize(
                                        (mask.shape[1], mask.shape[0]),
                                        resample=PILImage.NEAREST
                                    )
                                    seg_mask_array = np.array(seg_mask_resized) / 255.0
                                    mask = mask | (seg_mask_array > 0.5)
                                else:
                                    mask = mask | (seg_mask > 0.5)
                    
                    # Apply mask (blackout)
                    img_array[mask] = 0
                    manipulated_img = Image.fromarray(img_array)
                    
                    # Save manipulated image
                    manipulated_image_path = os.path.join(out_dir, f"manipulated_token_{token}.png")
                    manipulated_img.save(manipulated_image_path)
                    
                    # Run VLM with manipulated image to get perturbed answer
                    perturbed_answer = pixel_shap.model(manipulated_image_path, prompt)
                    
                    # Calculate similarity difference
                    if perturbed_answer:
                        perturbed_embedding = vectorizer.vectorize(perturbed_answer)
                        similarity = vectorizer.calculate_similarity(
                            baseline_embedding,
                            perturbed_embedding
                        )
                        if len(similarity) > 0:
                            similarity_value = float(similarity[0])
                            similarity_diff = 1.0 - similarity_value  # Difference from baseline
                
                except Exception as e:
                    print(f"[WARN] Could not create manipulated image for token '{token}': {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
            
            # Run PixelSHAP analysis for overlay (this creates the heatmap)
            results_df, shapley_values = pixel_shap.analyze(
                image_path=image_path,
                prompt=prompt,  # Just the question
                sampling_ratio=sampling_ratio,
                max_combinations=max_combinations,
                cleanup_temp_files=True,
            )
            
            # Create overlay for this token
            token_overlay_path = os.path.join(out_dir, f"overlay_token_{token}.png")
            pixel_shap.visualize(
                background_opacity=0.5,
                show_original_side_by_side=True,
                show_labels=False,
                show_model_output=True,
                output_path=token_overlay_path,
            )
            
            token_overlays.append((token, token_overlay_path))
            
            token_analyses.append({
                "token": token,
                "answer": perturbed_answer if perturbed_answer else baseline_output,
                "similarity_diff": float(similarity_diff) if similarity_diff is not None else None,
                "manipulated_image": manipulated_image_path,
            })
            
        except Exception as e:
            print(f"[WARN] PixelSHAP failed for token '{token}': {e}")
            import traceback
            if debug:
                traceback.print_exc()
            continue
    
    return {
        "segmentation_overlay": seg_overlay_path,
        "overall_overlay": overall_overlay_path,
        "token_overlays": token_overlays,
        "token_analyses": token_analyses,
    }


def run_pixelshap_for_image(
    vlm_cfg: VLMConfig,
    generated_answer: str,
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
    Creates a new segmentation model for each image based on the complete generated answer.
    Similar to run_pixelshap_for_token, but uses a generic filename (overlay.png)
    instead of token-specific naming.

    The overlay is stored in the provided directory along with meta.json.

    Args:
        vlm_cfg: VLMConfig containing model, processor, device, etc.
        generated_answer: Complete generated answer string (e.g., "<answer> because <explanation>")
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

    # Build segmentation model with complete generated answer
    segmentation_model = build_segmentation_model(
        generated_answer=generated_answer,
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
    # Include the generated answer as context to ensure attribution is for the specific output
    prompt = (
        base_prompt
        + f"\n\nYour previous answer was: '{generated_answer}'. "
        + f"Focus on explaining the concept '{token}' in this answer."
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
