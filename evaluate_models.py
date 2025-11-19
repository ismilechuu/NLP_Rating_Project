#!/usr/bin/env python3
# evaluate_models.py
# Evaluate and compare all 4 models on a test set with ground truth labels

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score
)
import sys
from pathlib import Path


LABELS = ['profanity', 'sexual', 'violence', 'hate']


def load_predictions(csv_path, label_suffix=''):
    """
    Load predictions from CSV file
    Returns dataframe with predictions for each label
    """
    df = pd.read_csv(csv_path)
    
    # Extract predictions for each label
    predictions = {}
    for label in LABELS:
        col_name = f"{label}{label_suffix}" if label_suffix else label
        if col_name in df.columns:
            predictions[label] = df[col_name].values
        else:
            print(f"⚠️  Warning: Column '{col_name}' not found in {csv_path}")
            predictions[label] = np.zeros(len(df))
    
    return predictions


def calculate_metrics(y_true, y_pred, label_name):
    """
    Calculate precision, recall, F1, and accuracy for a single label
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate support (number of positive samples)
    n_positive = int(y_true.sum())

    return {
        'label': label_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'support': n_positive
    }


def evaluate_model(ground_truth, predictions, model_name):
    """
    Evaluate a single model against ground truth
    Returns list of metrics for each label
    """
    results = []
    
    for label in LABELS:
        y_true = ground_truth[label]
        y_pred = predictions[label]
        
        metrics = calculate_metrics(y_true, y_pred, label)
        metrics['model'] = model_name
        results.append(metrics)
    
    return results


def print_results_table(results_df):
    """
    Print results in a formatted table
    """
    print("\n" + "=" * 100)
    print("📊 EVALUATION RESULTS")
    print("=" * 100)
    
    # Group by model
    for model_name in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model_name]
        
        print(f"\n🤖 Model: {model_name}")
        print("-" * 100)
        print(f"{'Label':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-" * 100)
        
        for _, row in model_results.iterrows():
            print(f"{row['label']:<12} {row['precision']:<12.4f} {row['recall']:<12.4f} "
                  f"{row['f1']:<12.4f} {row['accuracy']:<12.4f} "
                  f"{row['tp']:<8} {row['fp']:<8} {row['fn']:<8}")
        
        # Calculate average F1
        avg_f1 = model_results['f1'].mean()
        print("-" * 100)
        print(f"{'AVERAGE':<12} {'':<12} {'':<12} {avg_f1:<12.4f}")
        print()


def print_comparison_table(results_df):
    """
    Print comparison table across all models
    """
    print("\n" + "=" * 100)
    print("📊 MODEL COMPARISON (F1 Scores)")
    print("=" * 100)
    
    # Pivot table: labels as rows, models as columns
    pivot = results_df.pivot(index='label', columns='model', values='f1')
    
    print(pivot.to_string())
    
    # Add average row
    print("-" * 100)
    avg_row = pivot.mean()
    print(f"{'AVERAGE':<12}", end='')
    for model in pivot.columns:
        print(f" {avg_row[model]:>12.4f}", end='')
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare toxicity detection models"
    )
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="CSV file with ground truth labels (profanity, sexual, violence, hate columns)"
    )
    parser.add_argument(
        "--rule_based",
        help="CSV file with rule-based model predictions"
    )
    parser.add_argument(
        "--detoxify",
        help="CSV file with Detoxify model predictions"
    )
    parser.add_argument(
        "--finetuned",
        help="CSV file with fine-tuned model predictions"
    )
    parser.add_argument(
        "--hybrid",
        help="CSV file with hybrid model predictions"
    )
    parser.add_argument(
        "--out",
        help="Output CSV file to save detailed results (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("🔍 MODEL EVALUATION")
    print("=" * 100)
    
    # Load ground truth
    print(f"\n📄 Loading ground truth: {args.ground_truth}")
    ground_truth = load_predictions(args.ground_truth)
    n_samples = len(ground_truth['profanity'])
    print(f"✅ Loaded {n_samples} samples")
    
    # Print ground truth statistics
    print("\n📊 Ground Truth Statistics:")
    for label in LABELS:
        n_positive = ground_truth[label].sum()
        pct = (n_positive / n_samples) * 100
        print(f"   {label:<12}: {n_positive:>5} / {n_samples} ({pct:>5.2f}%)")
    
    # Evaluate each model
    all_results = []
    
    if args.rule_based:
        print(f"\n📄 Evaluating Rule-based model: {args.rule_based}")
        predictions = load_predictions(args.rule_based)
        results = evaluate_model(ground_truth, predictions, "Rule-based")
        all_results.extend(results)
        print("✅ Done")
    
    if args.detoxify:
        print(f"\n📄 Evaluating Detoxify model: {args.detoxify}")
        predictions = load_predictions(args.detoxify)
        results = evaluate_model(ground_truth, predictions, "Detoxify")
        all_results.extend(results)
        print("✅ Done")
    
    if args.finetuned:
        print(f"\n📄 Evaluating Fine-tuned model: {args.finetuned}")
        predictions = load_predictions(args.finetuned)
        results = evaluate_model(ground_truth, predictions, "Fine-tuned")
        all_results.extend(results)
        print("✅ Done")
    
    if args.hybrid:
        print(f"\n📄 Evaluating Hybrid model: {args.hybrid}")
        predictions = load_predictions(args.hybrid, label_suffix='_hybrid')
        results = evaluate_model(ground_truth, predictions, "Hybrid")
        all_results.extend(results)
        print("✅ Done")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Print results
    print_results_table(results_df)
    print_comparison_table(results_df)
    
    # Save results if output file specified
    if args.out:
        results_df.to_csv(args.out, index=False)
        print(f"💾 Saved detailed results to: {args.out}")
    
    print("=" * 100)
    print("✅ Evaluation complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()

