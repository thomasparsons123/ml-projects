import numpy as np

#Data used
hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
passed = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=float)

#Sigmoid function
def sigmoid(z):
    return (1/(1 + np.exp(-z)))

#Variables for logistic regression
N = len(hours)

w = 0.0
b = 0.0

learning_rate = 0.1
steps = 10000

#Training loop
for step in range(steps):

    z = (w * hours + b)
    pred = sigmoid(z)

    grad_w = np.sum(((pred - passed) * hours)) * (1/N)
    grad_b = np.sum(pred - passed) * (1/N)

    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    if step % 1000 == 0:
        loss = -np.mean(passed * np.log(pred) + (1 - passed) * np.log(1 - pred))
        print(f"step {step}: loss = {loss:.4f}")

#Calculate probability of passing and turn them into 0/1
probs = sigmoid(w * hours + b)
labels = (probs >= 0.5).astype(int)

#Print the models findings
print("predicted:", labels)
print("actual   :", passed.astype(int))
print("accuracy :", (labels == passed).mean())


