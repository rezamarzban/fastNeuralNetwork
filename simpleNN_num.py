import numpy as np

# Helper functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    logp = -np.log(y_pred[range(n_samples), y_true])
    loss = np.sum(logp) / n_samples
    return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    grad = y_pred.copy()
    n_samples = y_true.shape[0]
    grad[range(n_samples), y_true] -= 1
    grad = grad / n_samples
    return grad

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
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    
    # Compute loss
    loss = cross_entropy_loss(y, a2)
    
    # Backpropagation
    grad_z2 = cross_entropy_loss_derivative(y, a2)
    grad_W2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
    
    grad_a1 = np.dot(grad_z2, W2.T)
    grad_z1 = grad_a1 * relu_derivative(z1)
    grad_W1 = np.dot(X.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
    
    # Update weights and biases
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Evaluate the model
def predict(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return np.argmax(a2, axis=1)

# Predictions
predictions = predict(X)
accuracy = np.mean(predictions == y)
print(f'Accuracy: {accuracy}')
