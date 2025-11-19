# Model Evaluation Guide

## Overview

This guide explains how to use the updated Full Pipeline scripts for model evaluation.

## Updated Scripts

### 1. `rule_based_detector.py` - Rule-based Detection

**Two modes of operation:**

#### Mode 1: Full Pipeline (Production)
```bash
python rule_based_detector.py --input transcript.txt
```
- Input: `.txt` transcript file
- Preprocessing: ✅ Automatic (calls `preprocessing.py`)
- Rating: ✅ Calculates content rating (General/15+/18+/20+)
- Output: Rating text file + CSV with all segments

#### Mode 2: Inference Only (Evaluation)
```bash
python rule_based_detector.py --csv input.csv --out output.csv
```
- Input: `.csv` file with `text` column
- Preprocessing: ❌ Skipped (assumes text is already clean)
- Rating: ❌ Skipped
- Output: CSV with binary predictions (0/1) for each label

---

### 2. `detoxify_detector.py` - Detoxify Model

**Two modes of operation:**

#### Mode 1: Full Pipeline (Production)
```bash
python detoxify_detector.py --input transcript.txt
```
- Input: `.txt` transcript file
- Preprocessing: ✅ Automatic (calls `preprocessing.py`)
- Rating: ✅ Calculates content rating
- Output: Rating text file + CSV with all segments

#### Mode 2: Inference Only (Evaluation)
```bash
python detoxify_detector.py --csv input.csv --out output.csv --model unbiased
```
- Input: `.csv` file with `text` column
- Preprocessing: ❌ Skipped
- Rating: ❌ Skipped
- Output: CSV with binary predictions (0/1) for each label
- Options:
  - `--model`: Choose model variant (`original`, `unbiased`, `multilingual`)
  - `--threshold`: Classification threshold (default: 0.5)

---

## Complete Evaluation Pipeline

### Step 1: Prepare Ground Truth
```bash
python prepare_ground_truth.py
```
- Converts `label_complete.csv` to `ground_truth.csv`

### Step 2: Run All Models

#### Rule-based
```bash
python rule_based_detector.py \
  --csv ground_truth.csv \
  --out outputs/ground_truth_rule_based.csv
```

#### Detoxify
```bash
python detoxify_detector.py \
  --csv ground_truth.csv \
  --out outputs/ground_truth_detoxify.csv \
  --model unbiased
```

#### Fine-tuned
```bash
python model/infer.py \
  --model_dir tox_ft/best_model \
  --csv ground_truth.csv \
  --out outputs/ground_truth_finetuned.csv \
  --flag_out outputs/ground_truth_finetuned_flagged.csv \
  --context_window 0
```

#### Hybrid
```bash
python hybrid_detector.py \
  --csv_in outputs/ground_truth_finetuned.csv \
  --csv_out outputs/ground_truth_hybrid.csv
```

### Step 3: Evaluate All Models
```bash
python evaluate_models.py \
  --ground_truth ground_truth.csv \
  --rule_based outputs/ground_truth_rule_based.csv \
  --detoxify outputs/ground_truth_detoxify.csv \
  --finetuned outputs/ground_truth_finetuned.csv \
  --hybrid outputs/ground_truth_hybrid.csv \
  --out outputs/evaluation_results.csv
```

---

## Output Format

All models output CSV files with the following columns:

### Required Columns (for evaluation)
- `text`: Input text
- `profanity`: Binary prediction (0/1)
- `sexual`: Binary prediction (0/1)
- `violence`: Binary prediction (0/1)
- `hate`: Binary prediction (0/1)

### Additional Columns
- `*_prob`: Probability scores
- `*_hybrid`: Hybrid predictions (for hybrid model)
- Raw Detoxify scores (for Detoxify model)

---

## Key Differences

| Feature | Full Pipeline | Inference Only |
|---------|--------------|----------------|
| **Input** | `.txt` file | `.csv` file |
| **Preprocessing** | ✅ Automatic | ❌ Skipped |
| **Rating** | ✅ Calculated | ❌ Skipped |
| **Use Case** | Production | Research/Evaluation |

---

## Notes

- The Full Pipeline scripts (`rule_based_detector.py`, `detoxify_detector.py`) now support both modes
- No need for separate inference scripts (`model/rule_based_infer.py`, `model/detoxify_infer.py`)
- Original functionality is preserved - existing production workflows continue to work

