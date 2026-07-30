import numpy as np

size  = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
price = np.array([200, 280, 360, 440, 520])

N = len(size)

w = 0.0
b = 0.0

learning_rate = 0.1

for step in range(10000):
    pred = w * size + b

    w_grad = (2/N) * np.sum((pred - price) * size)
    b_grad = (2/N) * np.sum(pred - price)

    w = w - (w_grad * learning_rate)
    b = b - (b_grad * learning_rate)

    loss = np.mean((price - pred) ** 2)

    if step % 1000 == 0:
        print(f"step {step}: w={w:.2f} b={b:.2f} loss={loss:.2f}")

print(f"\nLearned model: price = {w:.2f} * size + {b:.2f}")

final_pred = w * size + b
print("predicted:", final_pred)
print("actual   :", price)

new_size = 1.75
print(f"\nPrediction for a {new_size*1000:.0f} sqft house: ${w*new_size + b:.1f}k")

