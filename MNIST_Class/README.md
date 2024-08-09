### image classifying multilayer neural network using MNIST datasets only with numpy, numba, cupy, C++ and etc

AI, image classifying multilayer neural network and handwriting digits recognition using MNIST datasets only with numpy, numba, cupy, C++ and etc without TensorFlow and PyTorch

In the every Python code, `hidden_sizes = [128, 64]` and in the C++ code, `std::vector<int> hidden_sizes = {128, 64};`, means first hidden layer size is 128 neurons and second hidden layer size is 64 neurons. 

For example if want to have 10 hidden layers with sizes: 1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000 neurons simply write in the Python code: `hidden_sizes = [1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000]`, Or write in the C++ code: `std::vector<int> hidden_sizes = {1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000};`

Please pay attention that `mnist.npz` file should be downloaded to working directory, It's download link is provided at `MNIST_link`. For C++ code, Convert `mnist.npz` file to `mnist_test.bin` and  `mnist_train.bin` files with running `MNIST_Convert.py` code.
