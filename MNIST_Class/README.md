#### image classifying multilayer neural network using MNIST datasets only with numpy, numba, cupy, C++ and etc

AI, image classifying multilayer neural network and handwriting digits recognition using MNIST datasets only with numpy, numba, cupy, C++ and etc without TensorFlow and PyTorch

In the every code, `hidden_sizes = [128, 64]` means first hidden layer size is 128 neurons and second hidden layer size is 64 neurons. 

For example if want to have 10 hidden layers with sizes: 1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000 neurons simply write in the code: `hidden_sizes = [1000, 500, 300, 200, 400, 600, 800, 100, 50, 2000]`

Please pay attention that doesn't matter to set epochs equal to 10000 or 100000 or more for best accuracy in the high speed codes such as numba or cupy.
