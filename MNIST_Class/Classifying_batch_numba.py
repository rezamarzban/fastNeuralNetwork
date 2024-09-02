
import numpy as np
from numba import njit

# Load the MNIST dataset from local mnist.npz file
with np.load('mnist.npz') as data:
    train_images = data['x_train']
    train_labels = data['y_train']
    test_images = data['x_test']
    test_labels = data['y_test']

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# One-hot encode the labels
train_labels_one_hot = np.eye(10)[train_labels]
test_labels_one_hot = np.eye(10)[test_labels]

# Xavier initialization function
def xavier_init(size):
    return np.random.randn(*size) * np.sqrt(1 / size[0])

# Network architecture
input_size = 784  # 28x28
hidden_sizes = [128, 64]  # Hidden layer sizes
output_size = 10

layer_sizes = [input_size] + hidden_sizes + [output_size]

weights = []
biases = []

# Initialize weights and biases
for i in range(len(layer_sizes) - 1):
    weights.append(xavier_init((layer_sizes[i], layer_sizes[i + 1])))
    biases.append(np.zeros((1, layer_sizes[i + 1])))

# Activation functions
@njit
def relu(x):
    return np.maximum(0, x)

@njit
def relu_derivative(x):
    return (x > 0).astype(np.float32)

@njit
def softmax(x):
    # Implement softmax manually without axis argument
    max_x = np.empty((x.shape[0], 1), dtype=x.dtype)
    for i in range(x.shape[0]):
        max_x[i, 0] = np.max(x[i])
    
    exp_x = np.exp(x - max_x)
    
    sum_exp_x = np.empty((x.shape[0], 1), dtype=x.dtype)
    for i in range(x.shape[0]):
        sum_exp_x[i, 0] = np.sum(exp_x[i])
    
    return exp_x / sum_exp_x

@njit
def forward_pass(X_batch, weights, biases):
    activations = [X_batch]
    pre_activations = []

    for j in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[j]) + biases[j]
        pre_activations.append(z)
        a = relu(z)
        activations.append(a)

    z = np.dot(activations[-1], weights[-1]) + biases[-1]
    pre_activations.append(z)
    a = softmax(z)
    activations.append(a)
    
    return activations, pre_activations

@njit
def compute_loss(a, y_batch):
    return -np.mean(np.sum(y_batch * np.log(a + 1e-8), axis=1))

@njit
def backward_pass(activations, pre_activations, y_batch, weights):
    deltas = [activations[-1] - y_batch]

    for j in range(len(weights) - 1, 0, -1):
        delta = np.dot(deltas[-1], weights[j].T) * relu_derivative(pre_activations[j-1])
        deltas.append(delta)

    deltas.reverse()
    
    return deltas

@njit
def update_parameters(weights, biases, activations, deltas, learning_rate, batch_size):
    for j in range(len(weights)):
        dW = np.dot(activations[j].T, deltas[j]) / batch_size
        db = np.empty_like(biases[j])
        for i in range(deltas[j].shape[1]):
            db[0, i] = np.sum(deltas[j][:, i]) / batch_size

        weights[j] -= learning_rate * dW
        biases[j] -= learning_rate * db

# Training parameters
epochs = 10
learning_rate = 0.01
batch_size = 64

# Training loop
for epoch in range(epochs):
    permutation = np.random.permutation(train_images.shape[0])
    train_images_shuffled = train_images[permutation]
    train_labels_one_hot_shuffled = train_labels_one_hot[permutation]

    for i in range(0, train_images.shape[0], batch_size):
        X_batch = train_images_shuffled[i:i+batch_size]
        y_batch = train_labels_one_hot_shuffled[i:i+batch_size]
        
        # Forward pass
        activations, pre_activations = forward_pass(X_batch, weights, biases)
        
        # Compute the loss (cross-entropy)
        loss = compute_loss(activations[-1], y_batch)
        
        # Backward pass
        deltas = backward_pass(activations, pre_activations, y_batch, weights)
        
        # Update parameters
        update_parameters(weights, biases, activations, deltas, learning_rate, batch_size)

    print(f'Epoch {epoch + 1}, Loss: {loss}')

# Evaluation
activations, _ = forward_pass(test_images, weights, biases)
predictions = np.argmax(activations[-1], axis=1)
accuracy = np.mean(predictions == test_labels)
print(f'Test accuracy: {accuracy}')
