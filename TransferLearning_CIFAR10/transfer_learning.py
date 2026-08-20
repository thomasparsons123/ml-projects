import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Use the M5 GPU (MPS) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# --- image preprocessing ---
# ResNet expects 224x224 images, converted to tensors and normalized
# the normalize numbers are the standard ImageNet values ResNet was trained with
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- download CIFAR-10 (reliable, built-in) ---
train_data = datasets.CIFAR10(
    root="data", train=True, transform=transform, download=True)

test_data = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True)

# 10 categories in CIFAR-10
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print("training images:", len(train_data))
print("test images:", len(test_data))
print("classes:", classes)

model = models.resnet18(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} out of {total:,}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

for step in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"step {step+1} done — last batch loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        predicted = pred.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test accuracy: {100 * correct / total:.2f}%")