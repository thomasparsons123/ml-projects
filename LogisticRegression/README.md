# Logistic Regression from Scratch

An implementation of logistic regression built entirely from scratch using only
NumPy — no scikit-learn. This project implements the full algorithm by hand:
the sigmoid function, gradient descent, log loss, and prediction thresholding.
The goal was to understand what actually happens inside a classifier, not just
to call `.fit()`.

## What it does

Predicts whether a student passes an exam based on hours studied — a binary
classification problem (pass / fail).

## The algorithm, implemented by hand

1. **Model:** compute a linear score `z = w·x + b`, then squash it into a
   probability (0–1) with the **sigmoid** function `σ(z) = 1 / (1 + e^(-z))`.
2. **Loss:** measure error with **log loss** (binary cross-entropy), which
   heavily penalizes confident wrong predictions.
3. **Training:** learn the weights `w` and `b` via **gradient descent**,
   deriving the gradients by hand (`(pred - y)·x`), running for 10,000 steps.
4. **Prediction:** convert probabilities to 0/1 labels by thresholding at 0.5.

## Result

The model reaches ~80% accuracy on the sample data. The two "misclassified"
points are contradictory noise in the data (a student who studied 5 hours passed,
while one who studied 6 hours failed). The model correctly learns the overall
trend rather than overfitting to that noise — which is the desired behavior.

## Key concepts demonstrated

- The sigmoid function and why it's needed for classification
- Gradient descent implemented from scratch
- Log loss (binary cross-entropy)
- Converting probabilities into class labels via a decision threshold
- Recognizing healthy generalization vs. overfitting noise

## Tech stack

- Python 3.12
- NumPy

## Running it

```bash
conda activate ml
python logistic_regression.py
```

## Notes

This is the same core algorithm that powers scikit-learn's `LogisticRegression`.
Building it from scratch makes the internals — the sigmoid, the gradient
derivation, and the training loop — fully transparent.