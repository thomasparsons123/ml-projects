# Linear Regression from Scratch

An implementation of linear regression built entirely from scratch with NumPy —
no scikit-learn. Implements gradient descent by hand to learn the best-fit line
through data, demonstrating the core mechanics that underlie all supervised learning.

## What it does

Predicts house prices from house size, learning the relationship
`price = w · size + b` directly from data via gradient descent.

## The algorithm, implemented by hand

1. **Model:** `pred = w · size + b`
2. **Loss:** Mean Squared Error — the average of the squared gaps between
   predictions and true prices.
3. **Gradients:** derived by hand using calculus — `grad_w = (2/N)·Σ(pred−y)·size`
   and `grad_b = (2/N)·Σ(pred−y)`.
4. **Training:** gradient descent — repeatedly step the weights *downhill*
   (`w = w − learning_rate · gradient`) until the loss is minimized.

## Result

The model recovers the true relationship `price = 160 · size + 40`, with the loss
converging to ~0 on this clean data.

## Key concepts demonstrated

- Gradient descent implemented from scratch
- Deriving loss gradients with calculus (the power rule and the role of each weight)
- Why the update *subtracts* the gradient (stepping downhill, not up)
- The core training loop that underlies all of machine learning

## Tech stack

- Python 3.12
- NumPy

## Running it

```bash
source .venv/bin/activate
python linear_regression.py
```

## Note

This is part of a "from scratch" series building the fundamentals of ML by hand:
**linear regression → logistic regression → neural networks** — implementing each
algorithm's math directly to understand what libraries like scikit-learn and PyTorch
do under the hood.