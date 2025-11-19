#!/bin/bash

# run_evaluation.sh
# Complete evaluation pipeline for all 4 models

set -e  # Exit on error

echo "======================================================================================================"
echo "🚀 STARTING COMPLETE MODEL EVALUATION PIPELINE"
echo "======================================================================================================"

# Step 1: Prepare ground truth (if needed)
if [ ! -f "ground_truth.csv" ]; then
    echo ""
    echo "📋 Step 1: Preparing ground truth..."
    python prepare_ground_truth.py
else
    echo ""
    echo "✅ Ground truth already exists: ground_truth.csv"
fi

# Step 2: Run Rule-based model
echo ""
echo "======================================================================================================"
echo "📊 Step 2: Running Rule-based model..."
echo "======================================================================================================"
python rule_based_detector.py \
    --csv ground_truth.csv \
    --out outputs/ground_truth_rule_based.csv

# Step 3: Run Detoxify model
echo ""
echo "======================================================================================================"
echo "📊 Step 3: Running Detoxify model..."
echo "======================================================================================================"
python detoxify_detector.py \
    --csv ground_truth.csv \
    --out outputs/ground_truth_detoxify.csv \
    --model unbiased

# Step 4: Run Fine-tuned model
echo ""
echo "======================================================================================================"
echo "📊 Step 4: Running Fine-tuned model..."
echo "======================================================================================================"
python model/infer.py \
    --model_dir tox_ft/best_model \
    --csv ground_truth.csv \
    --out outputs/ground_truth_finetuned.csv \
    --flag_out outputs/ground_truth_finetuned_flagged.csv \
    --context_window 0

# Step 5: Run Hybrid model
echo ""
echo "======================================================================================================"
echo "📊 Step 5: Running Hybrid model..."
echo "======================================================================================================"
python hybrid_detector.py \
    --csv_in outputs/ground_truth_finetuned.csv \
    --csv_out outputs/ground_truth_hybrid.csv

# Step 6: Evaluate all models
echo ""
echo "======================================================================================================"
echo "📊 Step 6: Evaluating all models..."
echo "======================================================================================================"
python evaluate_models.py \
    --ground_truth ground_truth.csv \
    --rule_based outputs/ground_truth_rule_based.csv \
    --detoxify outputs/ground_truth_detoxify.csv \
    --finetuned outputs/ground_truth_finetuned.csv \
    --hybrid outputs/ground_truth_hybrid.csv \
    --out outputs/evaluation_results.csv

echo ""
echo "======================================================================================================"
echo "✅ EVALUATION COMPLETE!"
echo "======================================================================================================"
echo ""
echo "📁 Output files:"
echo "   - outputs/ground_truth_rule_based.csv"
echo "   - outputs/ground_truth_detoxify.csv"
echo "   - outputs/ground_truth_finetuned.csv"
echo "   - outputs/ground_truth_hybrid.csv"
echo "   - outputs/evaluation_results.csv"
echo ""
echo "======================================================================================================"

