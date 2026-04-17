# Week 08 - Wednesday Assignment

## Overview
This project implements:
- CNN on MNIST
- Hate speech classifier
- Semantic search using embeddings
- Two-stage moderation pipeline

---

## Dataset
- Reddit dataset (clean_comment, category)

---

## Requirements

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn sentence-transformers tensorflow

---

## How to Run

1. Place dataset:
   Reddit_Data.csv

2. Run:
   python assignment.py

---

## Output
- Classification report
- Semantic search results
- CNN accuracy
- Filter visualizations
- Moderation pipeline output

---

## Key Results

- Classifier Recall: 0.81
- CNN Accuracy: ~99%
- Semantic search working correctly

---

## Folder Structure

week-08/
  wednesday/
    assignment.py
    Reddit_Data.csv
    README.md
    eda_report.md
    model_report.md
    pipeline_evaluation.md

---

## Notes

- Class imbalance handled using class_weight
- Sentence-BERT used for semantic similarity
- Pipeline improves harmful content detection

---

## Author
Avishka Jindal