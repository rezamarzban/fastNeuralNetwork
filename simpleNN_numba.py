
from numba import jit
import numpy as np
import math

# Helper functions
@jit(nopython=True)
def relu(x):
    return np.maximum(0, x)

@jit(nopython=True)
def relu_derivative(x):
    return (x > 0).astype(np.float64)

@jit(nopython=True)
def softmax(x):
    num_samples = x.shape[0]
    num_classes = x.shape[1]
    result = np.zeros_like(x)
    for i in range(num_samples):
        max_x = np.max(x[i])
        exp_x = np.exp(x[i] - max_x)
        sum_exp_x = np.sum(exp_x)
        result[i] = exp_x / sum_exp_x
    return result

@jit(nopython=True)
def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    loss = 0.0
    for i in range(n_samples):
        loss -= math.log(y_pred[i, y_true[i]])
    return loss / n_samples

@jit(nopython=True)
def cross_entropy_loss_derivative(y_true, y_pred):
    n_samples = y_true.shape[0]
    num_classes = y_pred.shape[1]
    grad = np.copy(y_pred)
    for i in range(n_samples):
        grad[i, y_true[i]] -= 1
    return grad / n_samples

@jit(nopython=True)
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros(output_size)
    return W1, b1, W2, b2

@jit(nopython=True)
def forward_pass(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

@jit(nopython=True)
def backpropagation(X, y, z1, a1, z2, a2, W2):
    grad_z2 = cross_entropy_loss_derivative(y, a2)
    grad_W2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)
    grad_a1 = np.dot(grad_z2, W2.T)
    grad_z1 = grad_a1 * relu_derivative(z1)
    grad_W1 = np.dot(X.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0)
    return grad_W1, grad_b1, grad_W2, grad_b2

@jit(nopython=True)
def update_parameters(W1, b1, W2, b2, grad_W1, grad_b1, grad_W2, grad_b2, learning_rate):
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
    return W1, b1, W2, b2

@jit(nopython=True)
def predict(X, W1, b1, W2, b2):
    _, _, _, a2 = forward_pass(X, W1, b1, W2, b2)
    return np.argmax(a2, axis=1)

def main():
    # Data
    np.random.seed(0)
    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)

    # Network parameters
    input_size = 5
    hidden_size = 3
    output_size = 2
    learning_rate = 0.01
    epochs = 1000

    # Initialize weights and biases
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    # Training loop
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward_pass(X, W1, b1, W2, b2)
        loss = cross_entropy_loss(y, a2)
        grad_W1, grad_b1, grad_W2, grad_b2 = backpropagation(X, y, z1, a1, z2, a2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, grad_W1, grad_b1, grad_W2, grad_b2, learning_rate)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Predictions
    predictions = predict(X, W1, b1, W2, b2)
    accuracy = np.mean(predictions == y)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
