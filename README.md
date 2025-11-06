# Multimodal VQA Evaluation Framework

A comprehensive evaluation framework for multimodal vision-language models on visual question answering (VQA) tasks with natural language explanations. This project supports evaluation of **LLaVA** and **Qwen3-VL** models on four benchmark datasets: **VQA-X**, **ACT-X**, **e-SNLI-VE**, and **VCR**.

## Features

- **Multi-Task Support**: Evaluate models on 4 different VQA tasks with explanations
  - **VQA-X**: Visual Question Answering with Explanations
  - **ACT-X**: Activity Recognition with Explanations
  - **e-SNLI-VE**: Visual Entailment with Explanations
  - **VCR**: Visual Commonsense Reasoning

- **Multi-Model Support**: 
  - LLaVA-1.5-7B (`https://huggingface.co/llava-hf/llava-1.5-7b-hf`)
  - Qwen3-VL-8B (`https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`)

- **Automatic Evaluation**: 
  - Accuracy computation with normalized answer matching
  - Detailed CSV exports with predictions, ground truth, and explanations
  - Summary grid tracking best results across tasks, models, and splits

- **Flexible Configuration**:
  - Run on train/val/test splits
  - Limit number of samples for quick testing

## Data Structure

The project expects the following directory structure:

```
multimodal-vqa-eval/
├── images/              # Image datasets
│   ├── train2014/      # COCO train images (for VQA-X)
│   ├── val2014/        # COCO val images (for VQA-X)
│   ├── mpi/            # MPI images (for ACT-X)
│   ├── flickr30k/      # Flickr30k images (for e-SNLI-VE)
│   └── vcr1images/     # VCR images
├── nle_data/           # Natural Language Explanation annotations
│   ├── VQA-X/
│   │   ├── vqaX_train.json
│   │   ├── vqaX_val.json
│   │   └── vqaX_test.json
│   ├── ACT-X/
│   │   ├── actX_train.json
│   │   └── actX_test.json
│   ├── eSNLI-VE/
│   │   ├── esnlive_train.json
│   │   └── esnlive_test.json
│   └── VCR/
│       ├── vcr_train.json
│       ├── vcr_val.json
│       └── vcr_test.json
└── results/            # Output directory (auto-created)
    ├── llava_results/  # LLaVA evaluation results
    ├── qwen_results/   # Qwen evaluation results
    └── accuracy_grid.csv  # Summary grid of best accuracies
```

## Usage

### Basic Usage

Run evaluation with default settings (VQA-X, LLaVA, val split, 10 samples):

**Arguments:**
- `--task`: Task to evaluate on (default: `vqax`)
  - `vqax`: VQA-X
  - `actx`: ACT-X
  - `esnlive`: e-SNLI-VE
  - `vcr`: VCR
- `--model`: Model to use (default: `llava`)
  - `llava`: LLaVA-1.5-7B
  - `qwen`: Qwen3-VL-8B
- `--split`: Dataset split (default: `val`)
  - `train`, `val`, `test` (availability depends on task)
- `--n_samples`: Number of samples to evaluate (default: `10`)
  - Use `None` or omit to evaluate on full dataset

### Examples

**Evaluate LLaVA on VQA-X validation set (50 samples):**
```bash
python main.py --task vqax --model llava --split val --n_samples 50
```

**Evaluate Qwen on ACT-X test set (full dataset):**
```bash
python main.py --task actx --model qwen --split test --n_samples 1000
```

**Evaluate LLaVA on e-SNLI-VE test set:**
```bash
python main.py --task esnlive --model llava --split test --n_samples 100
```

**Evaluate Qwen on VCR validation set:**
```bash
python main.py --task vcr --model qwen --split val --n_samples 200
```

## Output Format

### Per-Run CSV Files

Each evaluation run generates a detailed CSV file in `results/{model}_results/`:

**VQA-X** (`vqax_{split}_eval.csv`):
- `image_id`, `question`, `gt_answer`, `generated_answer`, `gt_explanation`, `generated_explanation`, `correct`

**ACT-X** (`actx_{split}_eval.csv`):
- `image_id`, `gt_label`, `generated_label`, `gt_explanation`, `generated_explanation`, `correct`

**e-SNLI-VE** (`esnlive_{split}_eval.csv`):
- `image_name`, `image_numeric_id`, `hypothesis`, `gt_label`, `generated_label`, `gt_explanation`, `generated_explanation`, `correct`

**VCR** (`vcr_{split}_eval.csv`):
- `image_name`, `question`, `gt_answer`, `generated_answer`, `gt_explanation`, `generated_explanation`, `correct`

### Summary Grid

The `results/accuracy_grid.csv` file maintains a summary of best accuracies across all configurations:

| task | llava_train | llava_test | llava_val | qwen_train | qwen_test | qwen_val |
|------|-------------|------------|-----------|------------|-----------|----------|
| VQA-X | 0.750 (100) | 0.720 (50) | 0.735 (100) | ... | ... | ... |
| ACT-X | ... | ... | ... | ... | ... | ... |

Each cell shows: `accuracy (n_samples)`. The grid automatically updates to keep only the best accuracy for each task-model-split combination.

## Task-Specific Details

### VQA-X
- Format: `<answer> because <explanation>`
- Evaluation: Exact match on normalized answer (ignoring articles)
- Ground truth: Majority vote from multiple annotators

### ACT-X
- Format: `<activity> because <explanation>`
- Evaluation: Exact match on normalized activity label
- Note: No validation split; use `test` for validation

### e-SNLI-VE
- Format: `<entailment|contradiction|neutral> because <explanation>`
- Evaluation: Label classification accuracy
- Uses few-shot prompting with examples
- Note: No validation split; use `test` for validation

### VCR
- Format: `<answer> because <explanation>`
- Evaluation: Exact match on normalized answer
- Visual commonsense reasoning task





```



