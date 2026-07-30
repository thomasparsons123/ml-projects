import numpy as np

np.random.seed(42)

def sigmoid(z):
    return (1/(1 + np.exp(-z)))

X = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = np.array([[0.], [1.], [1.], [0.]])

W1 = np.random.randn(2,2)
b1 = np.random.randn(1,2)

W2 = np.random.randn(2,1)
b2 = np.random.randn(1,1)

learning_rate  = 1.0

for step in range(10000):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)


    dz2 = a2 - y

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * a1 * (1 - a1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if step % 1000 == 0:
        loss = np.mean((a2 - y)**2)
        print(f"step {step}: loss = {loss:.4f}")

print("\nfinal predictions:")
print(np.round(a2))
print("actual:")
print(y)