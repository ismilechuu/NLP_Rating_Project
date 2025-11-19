#!/usr/bin/env python3
"""
Compare predictions from 4 different toxicity detection models.
Shows agreement/disagreement between models without requiring ground truth.
"""

import pandas as pd
import argparse
from pathlib import Path


def load_predictions(csv_path, model_name):
    """Load predictions from a CSV file."""
    df = pd.read_csv(csv_path)

    # Handle duplicate columns - pandas renames them to .1, .2, etc.
    # We want the renamed columns (the actual predictions)
    def get_column(base_name):
        """Get column value, trying .1 suffix first (for duplicate columns)."""
        if f'{base_name}.1' in df.columns:
            return df[f'{base_name}.1'].tolist()
        elif base_name in df.columns:
            return df[base_name].tolist()
        else:
            return [0] * len(df)

    # Extract binary predictions for each category
    predictions = {
        'text': df['text'].tolist(),
        'profanity': get_column('profanity'),
        'sexual': get_column('sexual'),
        'violence': get_column('violence'),
        'hate': get_column('hate'),
    }

    return predictions


def calculate_agreement(pred1, pred2):
    """Calculate agreement percentage between two prediction lists."""
    assert len(pred1) == len(pred2), "Prediction lists must have same length"
    
    agreements = sum(1 for p1, p2 in zip(pred1, pred2) if p1 == p2)
    total = len(pred1)
    
    return agreements / total * 100 if total > 0 else 0


def compare_models(rule_based_csv, detoxify_csv, finetuned_csv, hybrid_csv, output_csv=None):
    """Compare predictions from all 4 models."""
    
    print("=" * 60)
    print("🔍 Model Comparison (Without Ground Truth)")
    print("=" * 60)
    
    # Load predictions
    print("\n📥 Loading predictions...")
    rule_based = load_predictions(rule_based_csv, "Rule-based")
    detoxify = load_predictions(detoxify_csv, "Detoxify")
    finetuned = load_predictions(finetuned_csv, "Fine-tuned")
    hybrid = load_predictions(hybrid_csv, "Hybrid")
    
    n_samples = len(rule_based['text'])
    print(f"✅ Loaded {n_samples} samples from each model")
    
    # Calculate detection rates
    print("\n" + "=" * 60)
    print("📊 Detection Rates (% of samples flagged as toxic)")
    print("=" * 60)
    
    models = {
        'Rule-based': rule_based,
        'Detoxify': detoxify,
        'Fine-tuned': finetuned,
        'Hybrid': hybrid
    }
    
    categories = ['profanity', 'sexual', 'violence', 'hate']
    
    # Print header
    print(f"\n{'Model':<15} {'Profanity':<12} {'Sexual':<12} {'Violence':<12} {'Hate':<12}")
    print("-" * 60)
    
    detection_rates = {}
    for model_name, preds in models.items():
        rates = {}
        for cat in categories:
            count = sum(preds[cat])
            rate = count / n_samples * 100
            rates[cat] = rate
        detection_rates[model_name] = rates
        
        print(f"{model_name:<15} {rates['profanity']:>10.1f}% {rates['sexual']:>10.1f}% "
              f"{rates['violence']:>10.1f}% {rates['hate']:>10.1f}%")
    
    # Calculate pairwise agreement
    print("\n" + "=" * 60)
    print("🤝 Pairwise Agreement (% of samples with same prediction)")
    print("=" * 60)
    
    model_names = list(models.keys())
    
    for cat in categories:
        print(f"\n📌 {cat.upper()}:")
        print(f"{'Model Pair':<35} {'Agreement':<12}")
        print("-" * 50)
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                agreement = calculate_agreement(
                    models[m1][cat],
                    models[m2][cat]
                )
                print(f"{m1} vs {m2:<20} {agreement:>10.1f}%")
    
    # Find disagreement cases
    print("\n" + "=" * 60)
    print("⚠️  Disagreement Analysis")
    print("=" * 60)
    
    disagreements = []
    
    for idx in range(n_samples):
        text = rule_based['text'][idx]
        
        for cat in categories:
            predictions_for_cat = [
                models[m][cat][idx] for m in model_names
            ]
            
            # Check if there's disagreement
            if len(set(predictions_for_cat)) > 1:
                disagreements.append({
                    'text': text[:80] + '...' if len(text) > 80 else text,
                    'category': cat,
                    'rule_based': predictions_for_cat[0],
                    'detoxify': predictions_for_cat[1],
                    'finetuned': predictions_for_cat[2],
                    'hybrid': predictions_for_cat[3],
                })
    
    print(f"\n📊 Total disagreements: {len(disagreements)} (out of {n_samples * 4} predictions)")
    
    if disagreements:
        print(f"\n🔍 Sample disagreements (first 10):")
        print("-" * 100)
        
        for i, d in enumerate(disagreements[:10]):
            print(f"\n{i+1}. Category: {d['category'].upper()}")
            print(f"   Text: {d['text']}")
            print(f"   Rule-based: {d['rule_based']} | Detoxify: {d['detoxify']} | "
                  f"Fine-tuned: {d['finetuned']} | Hybrid: {d['hybrid']}")
    
    # Save disagreements to CSV if requested
    if output_csv:
        df_disagreements = pd.DataFrame(disagreements)
        df_disagreements.to_csv(output_csv, index=False)
        print(f"\n✅ Saved all disagreements to: {output_csv}")
    
    print("\n" + "=" * 60)
    print("✅ Comparison complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare predictions from 4 toxicity detection models"
    )
    parser.add_argument(
        "--rule_based",
        required=True,
        help="CSV file with rule-based predictions"
    )
    parser.add_argument(
        "--detoxify",
        required=True,
        help="CSV file with Detoxify predictions"
    )
    parser.add_argument(
        "--finetuned",
        required=True,
        help="CSV file with fine-tuned model predictions"
    )
    parser.add_argument(
        "--hybrid",
        required=True,
        help="CSV file with hybrid model predictions"
    )
    parser.add_argument(
        "--out",
        help="Output CSV file for disagreements (optional)"
    )
    
    args = parser.parse_args()
    
    compare_models(
        args.rule_based,
        args.detoxify,
        args.finetuned,
        args.hybrid,
        args.out
    )


if __name__ == "__main__":
    main()

