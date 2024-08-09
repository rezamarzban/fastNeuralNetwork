#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>

// Sigmoid activation function and its derivative
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1.0 - x);
}

// Function to generate random weights and biases
float random_weight() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// Softmax function for output layer
std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    float max = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;

    for (int i = 0; i < x.size(); ++i) {
        result[i] = exp(x[i] - max);
        sum += result[i];
    }

    for (int i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }

    return result;
}

// Neural network class definition
class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, const std::vector<int>& hiddenSizes, int outputSize);
    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& input, const std::vector<float>& target, float learningRate);
    float calculate_loss(const std::vector<float>& target);
    
private:
    std::vector<std::vector<std::vector<float>>> weights;  // Weight matrices for each layer
    std::vector<std::vector<float>> biases;               // Bias vectors for each layer
    std::vector<std::vector<float>> layers;               // Store activations of each layer
    int num_layers;                                       // Total number of layers including output
};

NeuralNetwork::NeuralNetwork(int inputSize, const std::vector<int>& hiddenSizes, int outputSize) {
    srand(time(0));

    // Initialize the layers sizes including input, hidden, and output
    std::vector<int> layer_sizes = {inputSize};
    layer_sizes.insert(layer_sizes.end(), hiddenSizes.begin(), hiddenSizes.end());
    layer_sizes.push_back(outputSize);

    num_layers = layer_sizes.size();

    // Initialize weights and biases for each layer
    weights.resize(num_layers - 1);
    biases.resize(num_layers - 1);
    layers.resize(num_layers);

    for (int i = 0; i < num_layers - 1; ++i) {
        int rows = layer_sizes[i];
        int cols = layer_sizes[i + 1];
        weights[i].resize(rows, std::vector<float>(cols));
        biases[i].resize(cols);
        layers[i + 1].resize(cols);

        for (int j = 0; j < rows; ++j) {
            for (int k = 0; k < cols; ++k) {
                weights[i][j][k] = random_weight();
            }
        }

        for (int j = 0; j < cols; ++j) {
            biases[i][j] = random_weight();
        }
    }
    layers[0].resize(inputSize);  // Initialize input layer
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    layers[0] = input;  // Set input layer

    for (int i = 0; i < num_layers - 2; ++i) {  // Process hidden layers
        for (int j = 0; j < layers[i + 1].size(); ++j) {
            layers[i + 1][j] = biases[i][j];
            for (int k = 0; k < layers[i].size(); ++k) {
                layers[i + 1][j] += layers[i][k] * weights[i][k][j];
            }
            layers[i + 1][j] = sigmoid(layers[i + 1][j]);
        }
    }

    // Process output layer
    for (int i = 0; i < layers[num_layers - 1].size(); ++i) {
        layers[num_layers - 1][i] = biases[num_layers - 2][i];
        for (int j = 0; j < layers[num_layers - 2].size(); ++j) {
            layers[num_layers - 1][i] += layers[num_layers - 2][j] * weights[num_layers - 2][j][i];
        }
    }

    // Apply softmax to output layer
    layers[num_layers - 1] = softmax(layers[num_layers - 1]);

    return layers[num_layers - 1];
}

void NeuralNetwork::backward(const std::vector<float>& input, const std::vector<float>& target, float learningRate) {
    std::vector<std::vector<float>> output_errors(num_layers - 1);

    // Calculate errors for output layer
    output_errors[num_layers - 2].resize(layers[num_layers - 1].size());
    for (int i = 0; i < layers[num_layers - 1].size(); ++i) {
        output_errors[num_layers - 2][i] = target[i] - layers[num_layers - 1][i];
    }

    // Backpropagate errors through hidden layers
    for (int i = num_layers - 3; i >= 0; --i) {
        output_errors[i].resize(layers[i + 1].size(), 0.0f);
        for (int j = 0; j < layers[i + 1].size(); ++j) {
            for (int k = 0; k < layers[i + 2].size(); ++k) {
                output_errors[i][j] += output_errors[i + 1][k] * weights[i + 1][j][k];
            }
            output_errors[i][j] *= sigmoid_derivative(layers[i + 1][j]);
        }
    }

    // Update weights and biases for output and hidden layers
    for (int i = 0; i < num_layers - 2; ++i) {
        for (int j = 0; j < weights[i].size(); ++j) {
            for (int k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] += learningRate * output_errors[i][k] * layers[i][j];
            }
        }
        for (int j = 0; j < biases[i].size(); ++j) {
            biases[i][j] += learningRate * output_errors[i][j];
        }
    }

    // Update weights and biases for the output layer
    for (int i = 0; i < weights[num_layers - 2].size(); ++i) {
        for (int j = 0; j < weights[num_layers - 2][i].size(); ++j) {
            weights[num_layers - 2][i][j] += learningRate * output_errors[num_layers - 2][j] * layers[num_layers - 2][i];
        }
    }
    for (int i = 0; i < biases[num_layers - 2].size(); ++i) {
        biases[num_layers - 2][i] += learningRate * output_errors[num_layers - 2][i];
    }
}

float NeuralNetwork::calculate_loss(const std::vector<float>& target) {
    float loss = 0.0f;
    for (int i = 0; i < layers[num_layers - 1].size(); ++i) {
        loss += -target[i] * log(layers[num_layers - 1][i]);
    }
    return loss;
}

// Function to load data from a binary file
void load_data(const std::string& filename, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    int num_samples, input_size, num_classes;
    file.read(reinterpret_cast<char*>(&num_samples), sizeof(int));
    file.read(reinterpret_cast<char*>(&input_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_classes), sizeof(int));

    inputs.resize(num_samples, std::vector<float>(input_size));
    outputs.resize(num_samples, std::vector<float>(num_classes));

    for (int i = 0; i < num_samples; ++i) {
        file.read(reinterpret_cast<char*>(inputs[i].data()), input_size * sizeof(float));
        file.read(reinterpret_cast<char*>(outputs[i].data()), num_classes * sizeof(float));
    }

    file.close();
}

int main() {
    // Network parameters
    int input_size = 784;  // MNIST images are 28x28 pixels
    std::vector<int> hidden_sizes = {128, 64};  // Two hidden layers with sizes 128 and 64
    int output_size = 10;  // 10 classes for MNIST digits
    float learning_rate = 0.01;
    int epochs = 10;

    // Load training and testing data
    std::vector<std::vector<float>> train_inputs, train_outputs;
    std::vector<std::vector<float>> test_inputs, test_outputs;

    load_data("mnist_train.bin", train_inputs, train_outputs);
    load_data("mnist_test.bin", test_inputs, test_outputs);

    // Initialize the neural network
    NeuralNetwork nn(input_size, hidden_sizes, output_size);

    // Training the network and printing loss
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (int i = 0; i < train_inputs.size(); ++i) {
            std::vector<float> prediction = nn.forward(train_inputs[i]);
            nn.backward(train_inputs[i], train_outputs[i], learning_rate);
            total_loss += nn.calculate_loss(train_outputs[i]);
        }
        if (epoch % 2 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / train_inputs.size() << std::endl;
        }
    }

    // Evaluate accuracy on test data
    int correct_predictions = 0;
    for (int i = 0; i < test_inputs.size(); ++i) {
        std::vector<float> prediction = nn.forward(test_inputs[i]);
        int predicted_label = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
        int actual_label = std::max_element(test_outputs[i].begin(), test_outputs[i].end()) - test_outputs[i].begin();
        if (predicted_label == actual_label) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / test_inputs.size() * 100.0f;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
