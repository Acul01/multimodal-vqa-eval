#!/usr/bin/env python3
"""
Script to calculate ROUGE scores between gt_explanation and generated_explanation
from a results CSV file.

Usage:
    python calculate_rouge_scores.py
    
The script will prompt you to enter the path to the CSV file.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Error: rouge-score package not found!")
    print("Install it with: pip install rouge-score")
    exit(1)


def resolve_csv_path(csv_path: str) -> str:
    """
    Resolve CSV path by trying multiple possible locations.
    
    Tries:
    1. Direct path (absolute or relative to current working directory)
    2. Relative to current working directory
    3. Remove leading "project_scripts/multimodal_vqa_eval/" if present (when running from repo root)
    4. Relative to common workspace roots (if running on cluster)
    
    Args:
        csv_path: Input path (may be relative or absolute)
    
    Returns:
        Resolved absolute path
    
    Raises:
        FileNotFoundError: If file cannot be found
    """
    # Try direct path first (if absolute)
    if os.path.isabs(csv_path) and os.path.exists(csv_path):
        return csv_path
    
    # Try relative to current working directory
    cwd_path = os.path.abspath(os.path.join(os.getcwd(), csv_path))
    if os.path.exists(cwd_path):
        return cwd_path
    
    # If path starts with "project_scripts/multimodal_vqa_eval/", try removing that prefix
    # (user might have copied full path but script is already in that directory)
    normalized_path = csv_path
    prefixes_to_remove = [
        "project_scripts/multimodal_vqa_eval/",
        "project_scripts/multimodal_vqa_eval",
    ]
    
    for prefix in prefixes_to_remove:
        if normalized_path.startswith(prefix):
            normalized_path = normalized_path[len(prefix):].lstrip("/")
            # Try relative to current directory
            test_path = os.path.abspath(os.path.join(os.getcwd(), normalized_path))
            if os.path.exists(test_path):
                return test_path
            break
    
    # Try relative to common cluster paths
    possible_roots = [
        "/netscratch/lrippe/",
        "/netscratch/lrippe/project_scripts/",
        os.path.expanduser("~"),
    ]
    
    # Try original path first
    for root in possible_roots:
        if os.path.exists(root):
            test_path = os.path.abspath(os.path.join(root, csv_path))
            if os.path.exists(test_path):
                return test_path
    
    # Try normalized path (with prefix removed)
    if normalized_path != csv_path:
        for root in possible_roots:
            if os.path.exists(root):
                test_path = os.path.abspath(os.path.join(root, normalized_path))
                if os.path.exists(test_path):
                    return test_path
    
    # Last resort: return the path as-is (will raise error if not found)
    return os.path.abspath(csv_path)


def calculate_rouge_scores(
    csv_path: str,
    gt_column: str = "gt_explanation",
    gen_column: str = "generated_explanation",
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame, str]:
    """
    Calculate ROUGE scores between ground truth and generated explanations.
    
    Args:
        csv_path: Path to the CSV file
        gt_column: Name of the ground truth explanation column
        gen_column: Name of the generated explanation column
    
    Returns:
        Tuple of:
        - Dictionary with average ROUGE-1, ROUGE-2, and ROUGE-L scores (precision, recall, f1)
        - DataFrame with original data + ROUGE score columns
        - Path to saved CSV file with ROUGE scores
    """
    # Resolve CSV path
    resolved_path = resolve_csv_path(csv_path)
    
    # Load CSV
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Resolved to: {resolved_path}\n"
            f"Current working directory: {os.getcwd()}"
        )
    
    print(f"DEBUG: Using resolved path: {resolved_path}")
    df = pd.read_csv(resolved_path)
    
    # Check if columns exist
    if gt_column not in df.columns:
        raise ValueError(f"Column '{gt_column}' not found in CSV. Available columns: {list(df.columns)}")
    if gen_column not in df.columns:
        raise ValueError(f"Column '{gen_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize lists for per-sample scores
    rouge1_f1_list = []
    rouge2_f1_list = []
    rougeL_f1_list = []
    
    # Lists for average calculation
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    valid_pairs = 0
    skipped = 0
    
    # Calculate ROUGE scores for each row
    for idx, row in df.iterrows():
        gt_text = str(row[gt_column]) if pd.notna(row[gt_column]) else ""
        gen_text = str(row[gen_column]) if pd.notna(row[gen_column]) else ""
        
        # Skip if either text is empty
        if not gt_text.strip() or not gen_text.strip():
            rouge1_f1_list.append(np.nan)
            rouge2_f1_list.append(np.nan)
            rougeL_f1_list.append(np.nan)
            skipped += 1
            continue
        
        # Calculate ROUGE scores
        scores = scorer.score(gt_text, gen_text)
        
        # Extract F1 scores for this sample
        rouge1_f1 = scores['rouge1'].fmeasure
        rouge2_f1 = scores['rouge2'].fmeasure
        rougeL_f1 = scores['rougeL'].fmeasure
        
        rouge1_f1_list.append(rouge1_f1)
        rouge2_f1_list.append(rouge2_f1)
        rougeL_f1_list.append(rougeL_f1)
        
        # Store for average calculation
        rouge1_scores.append({
            'precision': scores['rouge1'].precision,
            'recall': scores['rouge1'].recall,
            'f1': rouge1_f1,
        })
        rouge2_scores.append({
            'precision': scores['rouge2'].precision,
            'recall': scores['rouge2'].recall,
            'f1': rouge2_f1,
        })
        rougeL_scores.append({
            'precision': scores['rougeL'].precision,
            'recall': scores['rougeL'].recall,
            'f1': rougeL_f1,
        })
        
        valid_pairs += 1
    
    # Add ROUGE score columns to DataFrame
    df['Rouge-1'] = rouge1_f1_list
    df['Rouge-2'] = rouge2_f1_list
    df['Rouge-L'] = rougeL_f1_list
    
    # Calculate average scores
    if valid_pairs == 0:
        print(f"Warning: No valid pairs found! Skipped {skipped} rows with empty explanations.")
        avg_results = {
            'rouge1': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rouge2': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rougeL': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
        }
    else:
        avg_results = {
            'rouge1': {
                'precision': np.mean([s['precision'] for s in rouge1_scores]),
                'recall': np.mean([s['recall'] for s in rouge1_scores]),
                'f1': np.mean([s['f1'] for s in rouge1_scores]),
            },
            'rouge2': {
                'precision': np.mean([s['precision'] for s in rouge2_scores]),
                'recall': np.mean([s['recall'] for s in rouge2_scores]),
                'f1': np.mean([s['f1'] for s in rouge2_scores]),
            },
            'rougeL': {
                'precision': np.mean([s['precision'] for s in rougeL_scores]),
                'recall': np.mean([s['recall'] for s in rougeL_scores]),
                'f1': np.mean([s['f1'] for s in rougeL_scores]),
            },
        }
    
    print(f"\nProcessed {valid_pairs} valid pairs (skipped {skipped} rows with empty explanations)")
    
    # Save extended CSV in rouge_scores subdirectory
    input_dir = os.path.dirname(resolved_path)
    input_filename = os.path.basename(resolved_path)
    input_name_without_ext = os.path.splitext(input_filename)[0]
    
    # Create or use rouge_scores subdirectory
    rouge_scores_dir = os.path.join(input_dir, "rouge_scores")
    os.makedirs(rouge_scores_dir, exist_ok=True)
    
    # Output filename: [original_filename]_rouge_scores.csv
    output_filename = f"{input_name_without_ext}_rouge_scores.csv"
    output_path = os.path.join(rouge_scores_dir, output_filename)
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved extended CSV with ROUGE scores to: {output_path}")
    
    return avg_results, df, output_path


def print_rouge_results(results: Dict[str, Dict[str, float]]):
    """Print ROUGE results in a formatted table."""
    print("\n" + "="*70)
    print("ROUGE Scores")
    print("="*70)
    print(f"{'Metric':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-"*70)
    
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        scores = results[metric]
        print(f"{metric.upper():<15} {scores['precision']:<15.4f} {scores['recall']:<15.4f} {scores['f1']:<15.4f}")
    
    print("="*70)


def main():
    # Ask user for CSV file path
    print("="*70)
    print("ROUGE Score Calculator")
    print("="*70)
    print("\nThis script calculates ROUGE scores between 'gt_explanation' and 'generated_explanation' columns.")
    print("\nPlease enter the path to the CSV file:")
    
    csv_path = input("CSV file path: ").strip()
    
    # Remove quotes if user added them
    if csv_path.startswith('"') and csv_path.endswith('"'):
        csv_path = csv_path[1:-1]
    elif csv_path.startswith("'") and csv_path.endswith("'"):
        csv_path = csv_path[1:-1]
    
    if not csv_path:
        print("Error: No path provided!")
        return
    
    print(f"\nLoading CSV file: {csv_path}")
    print(f"Calculating ROUGE scores between 'gt_explanation' and 'generated_explanation'...")
    
    # Debug: print current working directory
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG PATH: {csv_path}")
    
    try:
        avg_results, df_extended, output_path = calculate_rouge_scores(csv_path)
        print_rouge_results(avg_results)
        
        # Also print summary
        print(f"\nSummary:")
        print(f"  ROUGE-1 F1: {avg_results['rouge1']['f1']:.4f}")
        print(f"  ROUGE-2 F1: {avg_results['rouge2']['f1']:.4f}")
        print(f"  ROUGE-L F1: {avg_results['rougeL']['f1']:.4f}")
        print(f"\nExtended CSV saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

