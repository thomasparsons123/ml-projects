import torch
import torch.nn as nn

X = torch.tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

print(X)
print(y)

model = nn.Sequential(
        nn.Linear(2,4),
        nn.Sigmoid(),
        nn.Linear(4,1),
        nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(5000):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

print("\nfinal predictions:")
preds = model(X).detach().round()
for i in range(4):
    print(f"  input {X[i].tolist()} -> predicted {int(preds[i].item())}  (actual {int(y[i].item())})")