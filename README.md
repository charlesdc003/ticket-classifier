# Ticket Classifier

![CI](https://github.com/charlesdc003/ticket-classifier/actions/workflows/ci.yml/badge.svg)

A text classification model that categorizes support tickets into billing, auth, bug, feature_request, and general. Built to demonstrate proper ML methodology: baseline comparison, honest metrics, and error analysis.

## Methodology

1. 100 labeled tickets split 72/8/20 train/val/test with stratification
2. Baseline 1: TF-IDF + Logistic Regression
3. Baseline 2: TF-IDF + Naive Bayes
4. Metrics reported on held-out test set only

## Results

| Model | Accuracy | F1 (weighted) | Precision | Recall |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 70% | 0.647 | 0.673 | 0.700 |
| TF-IDF + Naive Bayes | 65% | 0.604 | 0.654 | 0.650 |

Logistic Regression is the stronger model across all metrics.

## Per-class analysis

| Category | LR F1 | Notes |
|---|---|---|
| auth | 0.78 | High recall (1.0) — distinctive keywords |
| bug | 1.00 | Perfectly classified |
| billing | 0.50 | Low recall — confused with general |
| feature_request | 0.00 | Completely missed — overlaps with general |
| general | 0.50 | Catchall category absorbs misclassifications |

## Key failure modes

- `feature_request` is never correctly predicted — the language overlaps too much with general inquiries
- `billing` recall is low — refund and payment language is sometimes classified as general
- Small dataset (100 tickets) limits generalization — more data would significantly improve results

## What would improve accuracy

- More training data — 100 tickets is too small for 5 classes
- Few-shot examples or fine-tuning a sentence transformer
- Merging `general` and `feature_request` into fewer classes
- Adding ticket metadata (customer tier, time of day) as features

## Run locally

```bash
uv sync
uv run python scripts/train.py
uv run pytest tests/ -v
```

## Stack

- scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
- pandas + numpy
- pytest + GitHub Actions CI