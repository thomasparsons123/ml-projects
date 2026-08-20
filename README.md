# ML Projects

A collection of machine learning projects built while learning ML from the ground
up — implementing algorithms by hand in NumPy first, then using PyTorch and
scikit-learn once the internals are clear.

## Projects

| Project | What it does | Stack |
|---|---|---|
| **[Titanic Survival](./Titanic)** | Logistic regression predicting passenger survival, ~81% test accuracy. Full pipeline: data cleaning, train/test split, training, evaluation. | scikit-learn, pandas |
| **[Transfer Learning: CIFAR-10](./TransferLearning_CIFAR10)** | A pretrained ResNet18 adapted to classify CIFAR-10 across 10 categories, ~80% accuracy while training <0.05% of the network's parameters. | PyTorch, torchvision |
| **[XOR Neural Network](./XOR_Network)** | A feed-forward network in PyTorch that solves XOR. Hidden layers, backpropagation, and the training loop using library primitives. | PyTorch |
| **[Neural Network from Scratch](./NeuralNetScratch)** | The same XOR problem, but with a 2-layer network and backpropagation coded by hand — no autograd. The counterpart to the PyTorch version above. | NumPy |
| **[Logistic Regression from Scratch](./LogisticRegression)** | Sigmoid activation, gradient descent, and log loss implemented by hand. Shows what a `.fit()` call actually does underneath. | NumPy |
| **[Linear Regression from Scratch](./LinearRegression)** | Gradient descent implemented by hand to fit a line to data. | NumPy |

The two XOR projects are a deliberate pair: one solves the problem with PyTorch's
autograd, the other implements the same forward and backward passes manually so the
gradient math is explicit.

## Setup

```bash
git clone https://github.com/thomasparsons123/ml-projects.git
cd ml-projects
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run any project directly, for example:

```bash
python Titanic/titanic_survival.py
```

CIFAR-10 (~170 MB) downloads automatically on first run of the transfer learning
project into a local `data/` folder, which is git-ignored.

## Notes

Each project has its own README explaining the approach, the result, and the
concepts it demonstrates. Accuracy figures are measured on held-out test sets, not
training data.

*More projects coming as I work through deep learning and computer vision.*
