# CoT Prompting Templates

This directory contains different variants of Chain-of-Thought (CoT) prompting templates for multimodal VQA tasks.

## Current Variants

### `cot_2stage.py` (Variant 1 - Default)
Two-stage approach:
- **Stage 1**: Generate image description (with image, **without question**)
- **Stage 2**: Answer question based on description (text only, no image)

**Input Flow:**
- Stage 1: `input=image, prompt` → `output=description`
- Stage 2: `input=description, question, prompt` → `output=answer`

### `cot_2stage_with_question.py` (Variant 2)
Two-stage approach:
- **Stage 1**: Generate image description (with image **AND question**)
- **Stage 2**: Answer question based on description (text only, no image)

**Input Flow:**
- Stage 1: `input=image, question, prompt` → `output=description`
- Stage 2: `input=description, question, prompt` → `output=answer`

**Difference:** In Variant 2, the question is provided in Stage 1, allowing the model to generate a more focused description that is relevant to the question.

## Adding a New Variant

To add a new CoT prompting variant:

1. **Create a new Python file** (e.g., `cot_v2.py`) in this directory

2. **Implement the required functions**:
   - `prompt_image_description_cot(prompt_mode, resolve_shot_count, add_fewshot_examples)`: Stage 1 - image description
   - `prompt_question_from_description_cot_vqax(description, question)`: Stage 2 for VQA-X
   - `prompt_question_from_description_cot_actx(description)`: Stage 2 for ACT-X
   - `prompt_question_from_description_cot_esnlive(description, hypothesis)`: Stage 2 for ESNLI-VE
   - `prompt_question_from_description_cot_vcr(description, question, choices)`: Stage 2 for VCR

3. **Function signatures**:
   ```python
   def prompt_image_description_cot(prompt_mode: str = "zero", 
                                     resolve_shot_count=None, 
                                     add_fewshot_examples=None) -> List[Dict]:
       """
       Stage 1: Generate image description.
       Returns a conversation list with image description prompt.
       """
       # Your implementation
       pass
   
   def prompt_question_from_description_cot_vqax(description: str, question: str) -> Dict:
       """
       Stage 2 for VQA-X: Answer question based on description.
       Returns a conversation item (dict) to append.
       """
       # Your implementation
       pass
   ```

4. **Update `__init__.py`** to export your new variant (or make it the default):
   ```python
   from .cot_v2 import (
       prompt_image_description_cot,
       prompt_question_from_description_cot_vqax,
       ...
   )
   ```

5. **Test your variant** by running the evaluation with `--generation_mode cot`

## Function Requirements

### `prompt_image_description_cot`
- **Input**: `prompt_mode` (str), helper functions for shot counting and few-shot injection
- **Output**: List[Dict] - conversation format with image description prompt
- **Must include**: Image in the user content (for Stage 1)

### `prompt_question_from_description_cot_*`
- **Input**: `description` (str) from Stage 1, task-specific parameters
- **Output**: Dict - single conversation item to append (text only, no image)
- **Must include**: Instructions that reference the description

## Example: Creating a New Variant

See `cot_2stage.py` as a reference implementation. You can copy it and modify the prompts, examples, or structure as needed.

