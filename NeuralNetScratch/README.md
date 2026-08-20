# Neural Network from Scratch (with Backpropagation)

A neural network implemented entirely from scratch in NumPy — including the
**backpropagation** algorithm coded by hand. No PyTorch, no autograd. This
implements what `loss.backward()` does automatically, deriving and coding every
gradient manually via the chain rule.

## What it does

Trains a 2-layer neural network to solve the **XOR problem** — a pattern that a
single neuron cannot solve, requiring a hidden layer to bend the decision boundary.

## Architecture

**2 inputs → 2 hidden neurons (sigmoid) → 1 output (sigmoid)**

## Implemented by hand

- **Forward pass:** `X → (W1, b1) → sigmoid → (W2, b2) → sigmoid → prediction`
- **Backward pass (backpropagation):** the error signal is propagated backward
  through the network using the chain rule:
  - output error: `dz2 = a2 - y`
  - output gradients: `dW2 = a1ᵀ · dz2`, `db2 = Σ dz2`
  - propagate back through W2 and the sigmoid: `dz1 = (dz2 · W2ᵀ) · a1·(1-a1)`
  - hidden gradients: `dW1 = Xᵀ · dz1`, `db1 = Σ dz1`
- **Weight updates:** gradient descent on all four parameter sets.

## Result

Loss converges to ~0 and the network classifies all four XOR cases correctly.

## Key concepts demonstrated

- Backpropagation derived and implemented by hand (the chain rule through layers)
- Why hidden layers enable non-linear decision boundaries
- The sigmoid derivative `a·(1-a)` and its role in backprop
- Matrix shapes across layers, and why gradients match their weights' shapes

## Tech stack

- Python 3.12
- NumPy

## Running it

```bash
source .venv/bin/activate
python neural_net_scratch.py
```

## Note

The finale of a "from scratch" series — **linear regression → logistic regression
→ neural network** — implementing each algorithm's math by hand to understand what
frameworks like PyTorch do under the hood.