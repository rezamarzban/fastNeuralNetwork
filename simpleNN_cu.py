import cupy as cp

# Helper functions
def relu(x):
    return cp.maximum(0, x)

def relu_derivative(x):
    return cp.where(x > 0, 1, 0)

def softmax(x):
    exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    logp = -cp.log(y_pred[cp.arange(n_samples), y_true])
    loss = cp.sum(logp) / n_samples
    return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    grad = y_pred.copy()
    n_samples = y_true.shape[0]
    grad[cp.arange(n_samples), y_true] -= 1
    grad = grad / n_samples
    return grad

# Data
cp.random.seed(0)
X = cp.random.rand(100, 5)
y = cp.random.randint(2, size=100)

# Network parameters
input_size = 5
hidden_size = 3
output_size = 2
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases
W1 = cp.random.randn(input_size, hidden_size)
b1 = cp.zeros((1, hidden_size))
W2 = cp.random.randn(hidden_size, output_size)
b2 = cp.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = cp.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = cp.dot(a1, W2) + b2
    a2 = softmax(z2)
    
    # Compute loss
    loss = cross_entropy_loss(y, a2)
    
    # Backpropagation
    grad_z2 = cross_entropy_loss_derivative(y, a2)
    grad_W2 = cp.dot(a1.T, grad_z2)
    grad_b2 = cp.sum(grad_z2, axis=0, keepdims=True)
    
    grad_a1 = cp.dot(grad_z2, W2.T)
    grad_z1 = grad_a1 * relu_derivative(z1)
    grad_W1 = cp.dot(X.T, grad_z1)
    grad_b1 = cp.sum(grad_z1, axis=0, keepdims=True)
    
    # Update weights and biases
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Evaluate the model
def predict(X):
    z1 = cp.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = cp.dot(a1, W2) + b2
    a2 = softmax(z2)
    return cp.argmax(a2, axis=1)

# Predictions
predictions = predict(X)
accuracy = cp.mean(predictions == y)
print(f'Accuracy: {accuracy}')
