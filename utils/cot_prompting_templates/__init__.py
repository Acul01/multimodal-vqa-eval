"""
CoT Prompting Templates

This package contains different variants of Chain-of-Thought (CoT) prompting templates.
Each variant is implemented in a separate module.

Available variants:
- cot_2stage: Question is NOT provided in Stage 1 (default)
- cot_2stage_with_question: Question IS provided in Stage 1

To use a specific variant, import it directly:
    from utils.cot_prompting_templates.cot_2stage import (
        prompt_image_description_cot,
        prompt_question_from_description_cot_vqax,
        ...
    )
    
    from utils.cot_prompting_templates.cot_2stage_with_question import (
        prompt_image_description_cot,
        prompt_question_from_description_cot_vqax,
        ...
    )

Or use the default variant (currently cot_2stage):
    from utils.cot_prompting_templates import (
        prompt_image_description_cot,
        prompt_question_from_description_cot_vqax,
        ...
    )
"""

# Import default variant (cot_2stage) - Variant 1: Question NOT in Stage 1
from .cot_2stage import (
    prompt_image_description_cot,
    prompt_question_from_description_cot_vqax,
    prompt_question_from_description_cot_actx,
    prompt_question_from_description_cot_esnlive,
    prompt_question_from_description_cot_vcr,
)

# Also export Variant 2 functions with different names for direct access
from .cot_2stage_with_question import (
    prompt_image_description_cot as prompt_image_description_cot_with_question,
    prompt_question_from_description_cot_vqax as prompt_question_from_description_cot_vqax_with_question,
    prompt_question_from_description_cot_actx as prompt_question_from_description_cot_actx_with_question,
    prompt_question_from_description_cot_esnlive as prompt_question_from_description_cot_esnlive_with_question,
    prompt_question_from_description_cot_vcr as prompt_question_from_description_cot_vcr_with_question,
)

__all__ = [
    # Default variant (cot_2stage)
    "prompt_image_description_cot",
    "prompt_question_from_description_cot_vqax",
    "prompt_question_from_description_cot_actx",
    "prompt_question_from_description_cot_esnlive",
    "prompt_question_from_description_cot_vcr",
    # Variant 2 (with question in stage 1)
    "prompt_image_description_cot_with_question",
    "prompt_question_from_description_cot_vqax_with_question",
    "prompt_question_from_description_cot_actx_with_question",
    "prompt_question_from_description_cot_esnlive_with_question",
    "prompt_question_from_description_cot_vcr_with_question",
]

