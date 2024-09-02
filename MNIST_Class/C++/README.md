
AI, image classifying MLP neural network and handwriting digits recognition using MNIST datasets only with C++, Without TensorFlow and PyTorch

In the C++ code, `std::vector<int> hidden_sizes = {128, 64};`, means first hidden layer size is 128 neurons and second hidden layer size is 64 neurons. 

For example if want to have 10 hidden layers with sizes: 1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000 neurons at each hidden layer, simply write in the C++ code: `std::vector<int> hidden_sizes = {1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000};`

Please pay attention that `mnist.npz` file should be downloaded to working directory, It's download link is provided at `MNIST_link`. For C++ code, Convert `mnist.npz` file to `mnist_test.bin` and  `mnist_train.bin` files with running `MNIST_Convert.py` code.

Why the Python code is faster than the C++ code and how to improve the C++ implementation:

#### Reasons for Python's Speed

1. **Optimized Libraries**: 
   - Python's NumPy library is highly optimized and relies on underlying C and Fortran libraries (such as BLAS and LAPACK) for fast numerical operations.

2. **Vectorized Operations**: 
   - NumPy performs operations on entire arrays at once rather than using explicit loops, leveraging highly optimized low-level code.

3. **Efficient Memory Management**: 
   - NumPy arrays are contiguous in memory and benefit from efficient memory access patterns and pre-allocated storage.

4. **High-Level Abstractions**: 
   - Python code uses high-level abstractions for operations like matrix multiplication and activation functions, which are implemented efficiently in NumPy.

#### Improving C++ Performance

1. **Use Optimized Libraries**:
   - Integrate optimized libraries like Eigen or Intel MKL for matrix operations to benefit from highly efficient implementations.

2. **Optimize Memory Management**:
   - Ensure that memory is allocated efficiently and avoid unnecessary allocations and deallocations.

3. **Leverage Parallelism and Vectorization**:
   - Utilize parallel processing techniques and SIMD instructions to speed up matrix operations.

4. **Profile and Optimize**:
   - Use profiling tools to identify performance bottlenecks and apply optimizations where needed.
   - 
