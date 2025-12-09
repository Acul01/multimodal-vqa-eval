#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration

from utils.summary_grid import init_grid, update_best
from utils.eval import (
    evaluate_vqax_to_csv,
    evaluate_actx_to_csv,
    evaluate_esnlive_to_csv,
    evaluate_vcr_to_csv,
    evaluate_vqax_answers_only_to_csv,
    evaluate_actx_answers_only_to_csv,
    evaluate_esnlive_answers_only_to_csv,
    evaluate_vcr_answers_only_to_csv,
)

# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multimodal NLE/VQA tasks with LLaVA or Qwen3-VL."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="vqax",
        choices=["vqax", "actx", "esnlive", "vcr"],
        help="Which dataset/task to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava",
        choices=["llava", "qwen"],
        help="Which VLM to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Data split to evaluate.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to run (None = full split).",
    )
    parser.add_argument(
        "--generate_explanations",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to generate explanations (true) or answers only (false).",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="zero",
        choices=["zero", "1shot", "3shot", "6shot"],
        help="Prompting style when generate_explanations=true: zero / 1shot / 3shot / 6shot.",
    )
    return parser.parse_args()


TASK_CANON = {"vqax": "VQA-X", "actx": "ACT-X", "esnlive": "ESNLI-VE", "vcr": "VCR"}


def accuracy_from_results(results):
    """Compute mean of the boolean 'correct' field in results list."""
    hits = [r.get("correct") for r in results if r.get("correct") is not None]
    if not hits:
        return None
    return sum(hits) / len(hits)


def compute_explanation_stats(results):
    """
    results: Liste von Dicts mit Schlüssel 'prediction' im Format
             '<answer> because <explanation>'.

    Wir zählen eine Explanation als *verwertbar*, wenn:
      - es einen 'because'-Teil gibt
      - der Explanation-Text NICHT einer Platzhalter-Phrase entspricht
        wie 'no further details' oder 'explanation missing'.
    """
    placeholder_expls = {
        "no further details",
        "explanation missing",
    }

    total = len(results)
    if total == 0:
        return 0, 0.0  # (valid_count, valid_pct)

    valid_count = 0

    for r in results:
        pred = (r.get("prediction") or "").strip().lower()
        if " because " not in pred:
            continue
        _, _, expl = pred.partition(" because ")
        expl = expl.strip()
        if not expl:
            continue
        if expl in placeholder_expls:
            continue
        # Alles was nicht Placeholder ist und nicht leer → verwertbar
        valid_count += 1

    valid_pct = 100.0 * valid_count / total
    return valid_count, valid_pct


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    gen_expl = args.generate_explanations.lower() == "true"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(
        f"\nRunning task={args.task.upper()} | model={args.model.upper()} "
        f"| split={args.split} | n_samples={args.n_samples} "
        f"| explanations={gen_expl} | prompt_mode={args.prompt_mode}"
    )

    project_root = os.path.dirname(os.path.abspath(__file__))

    # -------------------------
    # Paths
    # -------------------------
    images_root = os.path.join(project_root, "images")
    nle_root = os.path.join(project_root, "nle_data")

    # explanation mode → results root
    results_root = os.path.join(
        project_root,
        "results",
        "with_explanation" if gen_expl else "without_explanation",
    )

    # model-specific subdir for CSVs
    model_name = args.model.lower()
    model_results_dir = os.path.join(results_root, f"{model_name}_results")
    os.makedirs(model_results_dir, exist_ok=True)

    # init / migrate accuracy grid for THIS mode (with/without)
    grid_path = init_grid(results_root, force=False)
    print(f"Accuracy grid: {os.path.relpath(grid_path, project_root)}")

    # -------------------------
    # Load model + processor
    # -------------------------
    if model_name == "llava":
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)

        model_dtype = torch.float16 if device == "cuda" else torch.float32
        
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    elif model_name == "qwen":
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)

        model_dtype = torch.float16 if device == "cuda" else torch.float32

        if device == "cuda":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=model_dtype, 
                low_cpu_mem_usage=True,
                device_map="auto",
            ).to(device)
            
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=model_dtype, 
                low_cpu_mem_usage=True,
            ).to("cpu")

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # -------------------------
    # Select runner (with vs. without explanations)
    # -------------------------
    if gen_expl:
        from utils.run_vqa_tasks import run_vqa_task
    else:
        from utils.run_vqa_tasks_noEx import run_vqa_task

    task_key = TASK_CANON[args.task.lower()]

    # -------------------------
    # Run task
    # -------------------------
    if gen_expl:
        # mit Erklärungen → run_vqa_tasks.py erwartet prompt_mode
        results = run_vqa_task(
            task_key,
            model,
            processor,
            images_root,
            nle_root,
            split=args.split,
            n_samples=args.n_samples,
            prompt_mode=args.prompt_mode,
            )
    else:
        # answers only → aktuelle noEx-Version kennt kein prompt_mode
        results = run_vqa_task(
            task_key,
            model,
            processor,
            images_root,
            nle_root,
            split=args.split,
            n_samples=args.n_samples,
        )

    # -------------------------
    # Evaluate + CSV export
    # -------------------------
    if gen_expl:
        # full answer + explanation CSVs
        if task_key == "VQA-X":
            out_csv = evaluate_vqax_to_csv(
                results,
                images_root,
                os.path.join(nle_root, "VQA-X"),
                split=args.split,
                save_dir=model_results_dir,
                prompt_mode=args.prompt_mode,
            )
        elif task_key == "ACT-X":
            out_csv = evaluate_actx_to_csv(
                results,
                images_root,
                os.path.join(nle_root, "ACT-X"),
                split=args.split,
                save_dir=model_results_dir,
                prompt_mode=args.prompt_mode,
            )
        elif task_key == "ESNLI-VE":
            out_csv = evaluate_esnlive_to_csv(
                results,
                images_root,
                os.path.join(nle_root, "eSNLI-VE"),
                split=args.split,
                save_dir=model_results_dir,
                prompt_mode=args.prompt_mode,
            )
        elif task_key == "VCR":
            out_csv = evaluate_vcr_to_csv(
                results,
                images_root,
                os.path.join(nle_root, "VCR"),
                split=args.split,
                save_dir=model_results_dir,
                prompt_mode=args.prompt_mode,
            )
        else:
            out_csv = None
    else:
        # answers-only CSVs (Signaturen: results, split, save_dir, save_name=None)
        if task_key == "VQA-X":
            out_csv = evaluate_vqax_answers_only_to_csv(
                results,
                images_root,
                os.path.join(nle_root, "VQA-X"),
                split=args.split,
                save_dir=model_results_dir,
            )
        elif task_key == "ACT-X":
            out_csv = evaluate_actx_answers_only_to_csv(
                results=results,
                split=args.split,
                save_dir=model_results_dir,
            )
        elif task_key == "ESNLI-VE":
            out_csv = evaluate_esnlive_answers_only_to_csv(
                results=results,
                split=args.split,
                save_dir=model_results_dir,
            )
        elif task_key == "VCR":
            out_csv = evaluate_vcr_answers_only_to_csv(
                results,
                images_root,
                os.path.join(nle_root, "VCR"),
                split=args.split,
                save_dir=model_results_dir,
            )
        else:
            out_csv = None

    if out_csv:
        print(f"\nSaved results CSV: {os.path.relpath(out_csv, project_root)}")

    # -------------------------
    # Update accuracy grid for this mode (with/without)
    # -------------------------
        acc = accuracy_from_results(results)

    print(f"[DEBUG]: len_results: {len(results)}")

    # effektive Anzahl Samples im Lauf (falls n_samples=None -> volle Länge)
    effective_n = args.n_samples if args.n_samples is not None else len(results)

    if acc is not None:
        grid_out = update_best(
            results_root=results_root,
            task=task_key,
            model=model_name,
            split=args.split,
            n_samples=effective_n,
            accuracy=acc,
        )
        print(f"Updated grid: {os.path.relpath(grid_out, project_root)}")
    else:
        print("No valid accuracy could be computed (no 'correct' entries).")

    # -------------------------------------------------
    # NEU: Explanation-Stats nur für generate_explanations = True
    # -------------------------------------------------
    if gen_expl:
        from utils.expl_summary import init_expl_table, append_expl_run

        init_expl_table(results_root)
        valid_count, valid_pct = compute_explanation_stats(results)

        expl_path = append_expl_run(
            results_root=results_root,
            model=model_name,
            task=task_key,
            split=args.split,
            prompt_mode=args.prompt_mode,
            accuracy=acc,
            n_samples=effective_n,
            valid_expl_pct=valid_pct,
            valid_expl_count=valid_count,
            total_samples=len(results),
        )
        print(
            f"Explanation stats updated: {os.path.relpath(expl_path, project_root)} "
            f"(valid explanations: {valid_count}/{len(results)} = {valid_pct:.1f}%)"
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()