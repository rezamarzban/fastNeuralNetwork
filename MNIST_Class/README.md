### image classifying MLP neural network using MNIST datasets only with numpy, numba, cupy, C++ and etc

AI, image classifying MLP neural network and handwriting digits recognition using MNIST datasets only with numpy, numba, cupy, C++ and etc without TensorFlow and PyTorch

In the every Python code, `hidden_sizes = [128, 64]`, means first hidden layer size is 128 neurons and second hidden layer size is 64 neurons. 

For example if want to have 10 hidden layers with sizes: 1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000 neurons at each hidden layer, simply write in the Python code: `hidden_sizes = [1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000]`.

The number of epochs can be set to 1,000 for NumPy and Numba, and 10,000 for CuPy due to their higher speed.

Please pay attention that `mnist.npz` file should be downloaded to working directory, It's download link is provided at `MNIST_link`. For C++ code, Convert `mnist.npz` file to `mnist_test.bin` and  `mnist_train.bin` files with running `MNIST_Convert.py` code.

The difference between batch training and entire dataset (full-batch) training lies primarily in how the training data is processed during the optimization of a neural network. Here are the key differences:

#### 1. **Definition**:
   - **Batch Training (Mini-Batch Gradient Descent)**: The training data is divided into smaller subsets called batches. The model is trained on each batch one at a time, and the weights are updated after processing each batch.
   - **Entire Dataset Training (Full-Batch Gradient Descent)**: The model is trained on the entire dataset at once, and the weights are updated after processing the entire dataset.

#### 2. **Memory Usage**:
   - **Batch Training**: Uses less memory because only a portion of the dataset (a batch) is loaded into memory at a time. This is useful for large datasets that may not fit entirely into memory.
   - **Entire Dataset Training**: Requires the entire dataset to be loaded into memory at once, which can be memory-intensive and may not be feasible for very large datasets.

#### 3. **Computation Speed**:
   - **Batch Training**: Typically faster per iteration since it processes only a portion of the data. However, it might require more iterations to converge since each update is based on a small subset of data, which may be noisy.
   - **Entire Dataset Training**: Generally slower per iteration because it processes the entire dataset at once. However, each update is more accurate because it considers the entire dataset, which may lead to faster convergence in terms of the number of iterations.

#### 4. **Convergence and Stability**:
   - **Batch Training**: Updates are more frequent but based on smaller samples, which can lead to noisier updates. This noise can sometimes help escape local minima, but it can also make convergence less stable.
   - **Entire Dataset Training**: Updates are smoother and more stable since they are based on the entire dataset. However, the process might get stuck in local minima due to the lack of noise in updates.

#### 5. **Flexibility**:
   - **Batch Training**: Allows the use of various batch sizes, including very small sizes (stochastic gradient descent with batch size = 1) or larger sizes. This flexibility can be used to fine-tune the training process.
   - **Entire Dataset Training**: Has no flexibility in terms of batch size, as it always uses the full dataset.

#### 6. **Parallelism and GPU Utilization**:
   - **Batch Training**: Often more efficient on GPUs, as the smaller batches can be processed in parallel more easily.
   - **Entire Dataset Training**: Can also benefit from parallel processing but might not fully utilize GPUs if the dataset is small.

#### 7. **Use Cases**:
   - **Batch Training**: Preferred for very large datasets, where processing the entire dataset in one go is not feasible. Also used in scenarios where the noise in updates (due to smaller batches) might be beneficial.
   - **Entire Dataset Training**: More common with smaller datasets or when the model needs to learn from the entire dataset at once to avoid noise.

#### 8. **Error Landscape**:
   - **Batch Training**: The error landscape can vary for each batch, leading to a more rugged optimization path.
   - **Entire Dataset Training**: The error landscape is smoother, and the optimization path is more direct but might miss potential beneficial noise.

In summary, **batch training** offers more flexibility, lower memory usage, and faster per-iteration updates at the cost of potentially noisier updates. **Entire dataset training** provides stable and accurate updates but requires more memory and might be computationally more expensive for each iteration. The choice between them often depends on the size of the dataset and the specific requirements of the training process.
