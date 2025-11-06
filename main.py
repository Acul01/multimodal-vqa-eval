import argparse
import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration

from utils.run_vqa_tasks import run_vqa_task
from utils.eval import (
    evaluate_vqax_to_csv,
    evaluate_actx_to_csv,
    evaluate_esnlive_to_csv,
    evaluate_vcr_to_csv,
)
from utils.summary_grid import init_grid, update_best


# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal tasks with LLaVA or Qwen3-VL.")
    parser.add_argument("--task", type=str, default="vqax", choices=["vqax", "actx", "esnlive", "vcr"])
    parser.add_argument("--model", type=str, default="llava", choices=["llava", "qwen"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--n_samples", type=int, default=10)
    return parser.parse_args()


# Map short task keys to canonical names used internally
TASK_CANON = {"vqax": "VQA-X", "actx": "ACT-X", "esnlive": "ESNLI-VE", "vcr": "VCR"}


def simple_accuracy(results):
    """Compute accuracy from run_vqa_task(...) results list."""
    valid = [r.get("correct") for r in results if r.get("correct") is not None]
    if not valid:
        return None
    return float(sum(valid)) / float(len(valid))


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    print(
        f"\nRunning task={args.task.upper()} | model={args.model.upper()} "
        f"| split={args.split} | n_samples={args.n_samples}"
    )

    # --- Project root (relative paths) ---
    project_root = os.path.dirname(os.path.abspath(__file__))

    # --- Initialize grid only ---
    grid_path = init_grid(project_root, force=False)
    print(f"Summary grid: {os.path.relpath(grid_path, project_root)}")

    # --- Relative data/output dirs ---
    images_root = os.path.join(project_root, "images")
    nle_root = os.path.join(project_root, "nle_data")
    base_results_dir = os.path.join(project_root, "results")

    # -------------------------
    # Load model + processor
    # -------------------------
    model_name = args.model.lower()
    if model_name == "llava":
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to("cuda")
        results_dir = os.path.join(base_results_dir, "llava_results")

    elif model_name == "qwen":
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)
        # Load without accelerate; move to single GPU
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16  # use torch.float16 if bf16 not supported
        ).to("cuda")
        results_dir = os.path.join(base_results_dir, "qwen_results")

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    os.makedirs(results_dir, exist_ok=True)

    # -------------------------
    # Run selected task
    # -------------------------
    task_key = TASK_CANON[args.task.lower()]
    results = run_vqa_task(
        task_key, model, processor, images_root, nle_root, split=args.split, n_samples=args.n_samples
    )

    # -------------------------
    # Evaluate + write per-run CSV
    # -------------------------
    if task_key == "VQA-X":
        out_csv = evaluate_vqax_to_csv(results, images_root, os.path.join(nle_root, "VQA-X"),
                                       split=args.split, save_dir=results_dir)
    elif task_key == "ACT-X":
        out_csv = evaluate_actx_to_csv(results, images_root, os.path.join(nle_root, "ACT-X"),
                                       split=args.split, save_dir=results_dir)
    elif task_key == "ESNLI-VE":
        out_csv = evaluate_esnlive_to_csv(results, images_root, os.path.join(nle_root, "eSNLI-VE"),
                                          split=args.split, save_dir=results_dir)
    elif task_key == "VCR":
        out_csv = evaluate_vcr_to_csv(results, images_root, os.path.join(nle_root, "VCR"),
                                      split=args.split, save_dir=results_dir)
    else:
        out_csv = None

    if out_csv:
        print(f"\nSaved results: {os.path.relpath(out_csv, project_root)}")

    # -------------------------
    # Update accuracy grid 
    # -------------------------
    acc = simple_accuracy(results)
    update_best(
        project_root=project_root,
        task=task_key,
        model=args.model,
        split=args.split,
        n_samples=args.n_samples,
        accuracy=acc,
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()