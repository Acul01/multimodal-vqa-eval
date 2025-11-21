# utils/pixelshap_integration.py
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any
from utils.run_vqa_tasks import generate_answer  # local import to avoid circular deps

@dataclass
class VLMConfig:
    model: Any
    processor: Any
    device: str = "cuda"
    max_new_tokens: int = 40
    task: str = "VQA-X"


class VLMWrapper:
    """
    Minimal black-box wrapper: adapts your VLM (LLaVA / Qwen3-VL)
    to the interface expected by PixelSHAP.
    """
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device

    def __call__(self, image_path: str, prompt_text: str) -> str:
        """
        PixelSHAP calls this to get the model output.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},  # placeholder, image is injected in generate_answer
                ],
            }
        ]

        text, _ = generate_answer(
            model=self.cfg.model,
            processor=self.cfg.processor,
            image_path=image_path,
            conversation=conversation,
            max_new_tokens=self.cfg.max_new_tokens,
            device=self.cfg.device,
        )
        return text


def build_pixelshap(
    vlm_cfg: VLMConfig,
    segmentation_model: Any,
    manipulator: Any,
    vectorizer: Optional[Any] = None,
    temp_dir: str = "pixelshap_tmp",
    debug: bool = False,
):
    """
    Construct a PixelSHAP object for your VLM.

    This function lazily imports PixelSHAP so that the rest of the
    project can run without the TokenSHAP repo being installed.
    """
    try:
        # PixelSHAP is defined inside the token_shap package (TokenSHAP repo)
        from token_shap.pixel_shap import PixelSHAP
    except ImportError as e:
        raise ImportError(
            "PixelSHAP (from the TokenSHAP repository) is not available. "
            "Please make sure the TokenSHAP repo is installed "
            "(e.g. `pip install -e TokenSHAP`)."
        ) from e

    vlm_wrapper = VLMWrapper(vlm_cfg)

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
) -> str:
    """
    Run PixelSHAP for a specific explanation token and save an overlay image.

    Returns:
        Path to the saved overlay image.
    """
    os.makedirs(out_dir, exist_ok=True)

    prompt = (
        base_prompt
        + f"\n\nExplain the image and especially the concept '{token}'."
    )

    # API: adapt to your PixelSHAP version if needed
    results_df, shapley_values = pixel_shap.analyze(
        image_path=image_path,
        prompt=prompt,
        sampling_ratio=sampling_ratio,
        max_combinations=max_combinations,
        cleanup_temp_files=True,
    )

    out_path = os.path.join(
        out_dir,
        f"{os.path.splitext(os.path.basename(image_path))[0]}__token_{token}.png",
    )

    # Again, adapt visualize() call if your PixelSHAP version differs
    pixel_shap.visualize(
        background_opacity=0.5,
        show_original_side_by_side=True,
        show_labels=False,
        show_model_output=True,
        output_path=out_path,
    )

    return out_path