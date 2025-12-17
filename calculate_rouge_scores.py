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
from typing import Dict, List
import os

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Error: rouge-score package not found!")
    print("Install it with: pip install rouge-score")
    exit(1)


def calculate_rouge_scores(
    csv_path: str,
    gt_column: str = "gt_explanation",
    gen_column: str = "generated_explanation",
) -> Dict[str, Dict[str, float]]:
    """
    Calculate ROUGE scores between ground truth and generated explanations.
    
    Args:
        csv_path: Path to the CSV file
        gt_column: Name of the ground truth explanation column
        gen_column: Name of the generated explanation column
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores (precision, recall, f1)
    """
    # Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check if columns exist
    if gt_column not in df.columns:
        raise ValueError(f"Column '{gt_column}' not found in CSV. Available columns: {list(df.columns)}")
    if gen_column not in df.columns:
        raise ValueError(f"Column '{gen_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores for each pair
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    valid_pairs = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        gt_text = str(row[gt_column]) if pd.notna(row[gt_column]) else ""
        gen_text = str(row[gen_column]) if pd.notna(row[gen_column]) else ""
        
        # Skip if either text is empty
        if not gt_text.strip() or not gen_text.strip():
            skipped += 1
            continue
        
        # Calculate ROUGE scores
        scores = scorer.score(gt_text, gen_text)
        
        rouge1_scores.append({
            'precision': scores['rouge1'].precision,
            'recall': scores['rouge1'].recall,
            'f1': scores['rouge1'].fmeasure,
        })
        rouge2_scores.append({
            'precision': scores['rouge2'].precision,
            'recall': scores['rouge2'].recall,
            'f1': scores['rouge2'].fmeasure,
        })
        rougeL_scores.append({
            'precision': scores['rougeL'].precision,
            'recall': scores['rougeL'].recall,
            'f1': scores['rougeL'].fmeasure,
        })
        
        valid_pairs += 1
    
    # Calculate average scores
    if valid_pairs == 0:
        print(f"Warning: No valid pairs found! Skipped {skipped} rows with empty explanations.")
        return {
            'rouge1': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rouge2': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rougeL': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
        }
    
    results = {
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
    
    return results


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
    
    try:
        results = calculate_rouge_scores(csv_path)
        print_rouge_results(results)
        
        # Also print summary
        print(f"\nSummary:")
        print(f"  ROUGE-1 F1: {results['rouge1']['f1']:.4f}")
        print(f"  ROUGE-2 F1: {results['rouge2']['f1']:.4f}")
        print(f"  ROUGE-L F1: {results['rougeL']['f1']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

