# Image Classifier with Transfer Learning

An image classifier built with **transfer learning** — adapting a pretrained
ResNet18 (trained on 1.2M ImageNet images) to classify CIFAR-10 images across
10 categories. Reaches ~80% test accuracy in minutes by reusing pretrained
visual features instead of training from scratch.

## How transfer learning works here

1. **Load** a ResNet18 pretrained on ImageNet — it already "knows how to see"
   (edges, textures, shapes, objects).
2. **Freeze** all pretrained layers (`requires_grad = False`) so their learned
   knowledge stays intact.
3. **Replace** the final layer with a fresh one sized for CIFAR-10's 10 classes.
4. **Train only the new final layer** (~5,000 parameters) while reusing ~11 million
   frozen pretrained parameters.

## Result

**~80% test accuracy** on 10,000 held-out images — 8x better than random guessing
(10%) — achieved in ~5 minutes of training on an Apple Silicon GPU (MPS), training
less than 0.05% of the network's parameters.

## Key concepts demonstrated

- Transfer learning: reusing a pretrained model for a new task
- Freezing layers vs. training a new classification head
- Multi-class classification (CrossEntropyLoss, argmax predictions)
- GPU acceleration (Apple MPS)
- Honest evaluation on a held-out test set

## Tech stack

- Python 3.12
- PyTorch + torchvision

## Running it

```bash
source .venv/bin/activate
python transfer_learning.py
```

Note: CIFAR-10 (~170MB) downloads automatically on first run into a local `data/`
folder (git-ignored).