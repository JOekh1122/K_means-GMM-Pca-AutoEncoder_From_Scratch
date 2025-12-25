import numpy as np
import matplotlib.pyplot as plt
def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)  # 1 where z > 0, else 0 

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
        self.losses = []  # store training loss


        for i in range(self.L):
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.b.append(np.zeros((1, layer_sizes[i+1])))


#Z¹ = A⁰ W¹ + b¹
#A¹ = a¹(Z¹)

    def forward(self, X):
        A = X  # A0
        self.cache = [(X, None)]   #  [(A⁰, None), (A¹, Z¹), (A², Z²), ...] 
# a(wx + b) 
        for i in range(self.L):
            Z = A @ self.W[i] + self.b[i]  # a prev @ W + b
            A = self.activations[i][0](Z)  # a now 
            self.cache.append((A, Z))

        return A

    def backward(self, X, X_hat):
        m = X.shape[0]  # number of examples    
        dA = 2 * (X_hat - X) / m # mean squared error derivative

        for i in reversed(range(self.L)):
            A_prev, _ = self.cache[i]
            _, Z = self.cache[i+1]

           #  cache[i]     = (A^(i),   none or Z)
           # cache[i + 1] = (A^(i+1), Z^(i+1))


            dZ = dA * self.activations[i][1](Z)
            dW = A_prev.T @ dZ + self.l2 * self.W[i] # λW
            db = np.sum(dZ, axis=0, keepdims=True)

            dA = dZ @ self.W[i].T

            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def train(self, X, epochs=100, batch_size=32, step_size=50, gamma=0.8):
        for epoch in range(epochs):
            if epoch > 0 and epoch % step_size == 0:
                self.lr *= gamma

            perm = np.random.permutation(len(X))  # shuffle data
            X = X[perm]

            epoch_loss = 0.0


            # Shuffling ensures:
                 #Each batch is different every epoch
                 # Better generalization

            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                X_hat = self.forward(batch)
                batch_loss = np.mean((batch - X_hat) ** 2)
                epoch_loss += batch_loss * len(batch)
                self.backward(batch, X_hat)
            epoch_loss /= len(X)
            self.losses.append(epoch_loss)


    
    def plot_training_loss(self):
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss (MSE)")
        plt.title("AutoEncoder Training Loss Curve")
        plt.grid(True)
        plt.show() 




# 1. Define your architecture
# Features -> Hidden 1 -> Hidden 2 -> Bottleneck -> Hidden 2 -> Hidden 1 -> Features
#my_layers = [784, 256, 128, 64, 128, 256, 784]

# 2. Define activations (must match the number of layers - 1)
# We use 6 activations because there are 6 transitions between 7 layers
#my_activations = [(relu, relu_deriv)] * 5 + [(sigmoid, sigmoid_deriv)]

# 3. Initialize
#model = AutoEncoder(layer_sizes=my_layers, activations=my_activations)




# Forward  : A_prev -> Z -> A
# Backward : dA -> dZ -> dW, db -> dA_prev
