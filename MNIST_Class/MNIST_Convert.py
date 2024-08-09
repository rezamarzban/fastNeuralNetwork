import numpy as np

# Load MNIST dataset
data = np.load('mnist.npz')
X_train, y_train = data['x_train'], data['y_train']
X_test, y_test = data['x_test'], data['y_test']

# Normalize the data to [0, 1] range
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Flatten the images (28x28 -> 784)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Write the training data to a binary file
with open('mnist_train.bin', 'wb') as f:
    # Write number of samples, input size, and number of classes (10 for MNIST)
    f.write(np.array([X_train.shape[0], X_train.shape[1], 10], dtype=np.int32).tobytes())
    
    # Write images and labels
    for img, label in zip(X_train, y_train):
        f.write(img.tobytes())              # Write image data
        f.write(np.eye(10)[label].astype(np.float32).tobytes())  # One-hot encode the label and write

# Write the testing data to a binary file
with open('mnist_test.bin', 'wb') as f:
    # Write number of samples, input size, and number of classes
    f.write(np.array([X_test.shape[0], X_test.shape[1], 10], dtype=np.int32).tobytes())
    
    # Write images and labels
    for img, label in zip(X_test, y_test):
        f.write(img.tobytes())              # Write image data
        f.write(np.eye(10)[label].astype(np.float32).tobytes())  # One-hot encode the label and write
