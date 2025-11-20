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
  - Summary grid tracking best results across tasks, models, and splits (CSV and Excel formats)
  - Token-level entropy extraction for explanation analysis

- **Prompting Strategies**:
  - Zero-shot, 1-shot, and 3-shot prompting modes
  - Flexible prompting templates for each task

- **Explanation Analysis**:
  - Token-level entropy computation for generated explanations
  - Explanation quality tracking (valid vs. placeholder explanations)
  - Separate tracking for runs with and without explanations

- **Flexible Configuration**:
  - Run on train/val/test splits
  - Limit number of samples for quick testing
  - Toggle between explanation and answer-only modes

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
└── results/                      # Output directory (auto-created)
    ├── with_explanation/         # Results when generating explanations
    │   ├── llava_results/        # LLaVA evaluation results
    │   ├── qwen_results/         # Qwen evaluation results
    │   ├── accuracy_grid.csv     # Summary grid (CSV)
    │   ├── accuracy_grid.xlsx    # Summary grid (Excel)
    │   └── explanation_stats.csv # Explanation quality statistics
    └── without_explanation/      # Results for answer-only mode
        ├── llava_results/        # LLaVA evaluation results
        ├── qwen_results/         # Qwen evaluation results
        ├── accuracy_grid.csv     # Summary grid (CSV)
        └── accuracy_grid.xlsx    # Summary grid (Excel)
```

## Usage

### Basic Usage

Run evaluation with default settings (VQA-X, LLaVA, val split, full dataset):

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
- `--n_samples`: Number of samples to evaluate (default: `None` for full dataset)
  - Specify integer to limit number of samples
  - Use `None` or omit to evaluate on full dataset
- `--generate_explanations`: Whether to generate explanations (default: `true`)
  - `true`: Generate answers with explanations (saved in `results/with_explanation/`)
  - `false`: Generate answers only (saved in `results/without_explanation/`)
- `--prompt_mode`: Prompting style when `generate_explanations=true` (default: `zero`)
  - `zero`: Zero-shot prompting
  - `1shot`: 1-shot few-shot prompting
  - `3shot`: 3-shot few-shot prompting

### Examples

**Evaluate LLaVA on VQA-X validation set with explanations (50 samples, zero-shot):**
```bash
python main.py --task vqax --model llava --split val --n_samples 50 --generate_explanations true --prompt_mode zero
```

**Evaluate Qwen on ACT-X test set with 3-shot prompting (full dataset):**
```bash
python main.py --task actx --model qwen --split test --generate_explanations true --prompt_mode 3shot
```

**Evaluate LLaVA on e-SNLI-VE test set with 1-shot prompting:**
```bash
python main.py --task esnlive --model llava --split test --n_samples 100 --generate_explanations true --prompt_mode 1shot
```

**Evaluate Qwen on VCR validation set (answers only, no explanations):**
```bash
python main.py --task vcr --model qwen --split val --n_samples 200 --generate_explanations false
```

## Output Format

### Per-Run CSV Files

Each evaluation run generates a detailed CSV file in `results/{with_explanation|without_explanation}/{model}_results/`:

**VQA-X** (`vqax_{split}_eval.csv`):
- `image_id`, `question`, `gt_answer`, `generated_answer`, `gt_explanation`, `generated_explanation`, `correct`, `token_entropy`
  - `token_entropy`: JSON string of dictionary mapping explanation tokens to their entropy values (only when `generate_explanations=true`)

**ACT-X** (`actx_{split}_eval.csv`):
- `image_id`, `gt_label`, `generated_label`, `gt_explanation`, `generated_explanation`, `correct`, `token_entropy`

**e-SNLI-VE** (`esnlive_{split}_eval.csv`):
- `image_name`, `image_numeric_id`, `hypothesis`, `gt_label`, `generated_label`, `gt_explanation`, `generated_explanation`, `correct`, `token_entropy`

**VCR** (`vcr_{split}_eval.csv`):
- `image_name`, `question`, `gt_answer`, `generated_answer`, `gt_explanation`, `generated_explanation`, `correct`, `token_entropy`

**Note:** The `token_entropy` column contains token-level entropy values (rounded to 3 decimal places) only for tokens in the explanation part of the generated text. Entropy is computed as the Shannon entropy over the model's probability distribution for each generated token.

### Summary Grid

The `results/{with_explanation|without_explanation}/accuracy_grid.csv` and `accuracy_grid.xlsx` files maintain a summary of best accuracies across all configurations:

| task | llava_train | llava_test | llava_val | qwen_train | qwen_test | qwen_val |
|------|-------------|------------|-----------|------------|-----------|----------|
| VQA-X | 0.750 (100) | 0.720 (50) | 0.735 (100) | ... | ... | ... |
| ACT-X | ... | ... | ... | ... | ... | ... |

Each cell shows: `accuracy (n_samples)`. The grid automatically updates to keep only the best accuracy for each task-model-split combination. The Excel format (`.xlsx`) requires the `openpyxl` package (`pip install openpyxl`).

### Token-Level Entropy

The framework computes token-level entropy for each generated token during explanation generation. Entropy is calculated as:

```
H = -Σ p(x) * log(p(x))
```

Where `p(x)` is the probability distribution over the vocabulary for each token generation step.

**Features:**
- Entropy values are rounded to 3 decimal places
- Only tokens in the explanation part (after "because") are included
- Stopwords (articles, common words) are filtered out
- Entropies are stored as JSON strings in the `token_entropy` column: `{"token1": 2.345, "token2": 1.890, ...}`
- If a token appears multiple times, the maximum entropy value is kept

This enables analysis of model uncertainty at the token level for explanation generation.

### Explanation Statistics

When `generate_explanations=true`, the framework also maintains `results/with_explanation/explanation_stats.csv` tracking explanation quality metrics:

- `model`, `task`, `split`, `prompt_mode`: Run configuration
- `accuracy`: Task accuracy
- `n_samples`: Number of evaluated samples
- `valid_expl_pct`: Percentage of samples with valid (non-placeholder) explanations
- `valid_expl_count`: Count of samples with valid explanations
- `total_samples`: Total number of samples

This allows tracking explanation quality across different prompting strategies and model configurations.

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



