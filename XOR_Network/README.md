# Neural Network from Scratch — Solving XOR (PyTorch)

A neural network built and trained in PyTorch to solve the XOR problem — a classic
pattern that a single neuron (linear model) provably *cannot* solve, because the
classes aren't linearly separable. This project demonstrates why hidden layers
matter and shows the full training loop, with an understanding of every component
from the underlying math up.

## The problem

XOR outputs 1 when the two inputs differ, 0 when they match:

| Input | Output |
|-------|--------|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

No single straight line can separate the 1s from the 0s, so a single neuron
(logistic regression) fails on it. A neural network with a hidden layer can bend
the decision boundary and solve it.

## The network

A small feed-forward network: **2 inputs → 4 hidden neurons (sigmoid) → 1 output (sigmoid)**.

Built with `nn.Sequential`, trained with binary cross-entropy loss and stochastic
gradient descent.

## The training loop

The standard 5-step loop, run for 5,000 epochs:

1. **Forward pass** — `model(X)` produces predictions
2. **Compute loss** — binary cross-entropy (log loss)
3. **Zero gradients** — `optimizer.zero_grad()`
4. **Backpropagation** — `loss.backward()` (autograd computes all gradients)
5. **Update weights** — `optimizer.step()` (gradient descent)

## Result

Loss falls from ~0.71 to ~0.004, and the trained network classifies all four
XOR cases correctly:
input [0.0, 0.0] -> predicted 0 (actual 0)
input [0.0, 1.0] -> predicted 1 (actual 1)
input [1.0, 0.0] -> predicted 1 (actual 1)
input [1.0, 1.0] -> predicted 0 (actual 0)

## Key concepts demonstrated

- Why a single neuron can't solve non-linearly-separable problems
- How hidden layers transform data so the boundary can bend
- The forward pass, backpropagation, and gradient descent
- PyTorch fundamentals: tensors, `nn.Sequential`, autograd, optimizers

## Tech stack

- Python 3.12
- PyTorch

## Running it

```bash
source .venv/bin/activate
python XOR_Network.py
```