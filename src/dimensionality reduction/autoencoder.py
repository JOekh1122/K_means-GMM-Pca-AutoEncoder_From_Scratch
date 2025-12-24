import numpy as np
def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_deriv(z): 
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z): return np.tanh(z)
def tanh_deriv(z): return 1 - np.tanh(z)**2


class AutoEncoder:
    def __init__(self, layer_sizes, activations, lr=0.01, l2=0.0):
        self.lr = lr
        self.l2 = l2
        self.L = len(layer_sizes) - 1

        self.W = []
        self.b = []
        self.activations = activations

        for i in range(self.L):
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.b.append(np.zeros((1, layer_sizes[i+1])))

    def forward(self, X):
        A = X
        self.cache = [(X, None)]

        for i in range(self.L):
            Z = A @ self.W[i] + self.b[i]
            A = self.activations[i][0](Z)
            self.cache.append((A, Z))

        return A

    def backward(self, X, X_hat):
        m = X.shape[0]
        dA = 2 * (X_hat - X) / m

        for i in reversed(range(self.L)):
            A_prev, _ = self.cache[i]
            _, Z = self.cache[i+1]

            dZ = dA * self.activations[i][1](Z)
            dW = A_prev.T @ dZ + self.l2 * self.W[i]
            db = np.sum(dZ, axis=0, keepdims=True)

            dA = dZ @ self.W[i].T

            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def train(self, X, epochs=100, batch_size=32, step_size=30, gamma=0.7):
        for epoch in range(epochs):
            if epoch > 0 and epoch % step_size == 0:
                self.lr *= gamma

            perm = np.random.permutation(len(X))
            X = X[perm]

            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                X_hat = self.forward(batch)
                self.backward(batch, X_hat)




# 1. Define your architecture
# Features -> Hidden 1 -> Hidden 2 -> Bottleneck -> Hidden 2 -> Hidden 1 -> Features
#my_layers = [784, 256, 128, 64, 128, 256, 784]

# 2. Define activations (must match the number of layers - 1)
# We use 6 activations because there are 6 transitions between 7 layers
#my_activations = [(relu, relu_deriv)] * 5 + [(sigmoid, sigmoid_deriv)]

# 3. Initialize
#model = AutoEncoder(layer_sizes=my_layers, activations=my_activations)