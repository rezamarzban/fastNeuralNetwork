#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>

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

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);
    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& input, const std::vector<float>& target, float learningRate);
    float calculate_loss(const std::vector<float>& target);
    
private:
    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;
    std::vector<float> hidden_layer;
    std::vector<float> output_layer;
    std::vector<float> bias_hidden;   // Biases for hidden layer
    std::vector<float> bias_output;   // Biases for output layer
};

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
    srand(time(0));
    weights_input_hidden.resize(inputSize, std::vector<float>(hiddenSize));
    weights_hidden_output.resize(hiddenSize, std::vector<float>(outputSize));
    bias_hidden.resize(hiddenSize);
    bias_output.resize(outputSize);
    
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weights_input_hidden[i][j] = random_weight();
        }
    }
    
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weights_hidden_output[i][j] = random_weight();
        }
    }
    
    for (int i = 0; i < hiddenSize; ++i) {
        bias_hidden[i] = random_weight();
    }
    
    for (int i = 0; i < outputSize; ++i) {
        bias_output[i] = random_weight();
    }
    
    hidden_layer.resize(hiddenSize);
    output_layer.resize(outputSize);
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    for (int i = 0; i < hidden_layer.size(); ++i) {
        hidden_layer[i] = bias_hidden[i];
        for (int j = 0; j < input.size(); ++j) {
            hidden_layer[i] += input[j] * weights_input_hidden[j][i];
        }
        hidden_layer[i] = sigmoid(hidden_layer[i]);
    }
    
    for (int i = 0; i < output_layer.size(); ++i) {
        output_layer[i] = bias_output[i];
        for (int j = 0; j < hidden_layer.size(); ++j) {
            output_layer[i] += hidden_layer[j] * weights_hidden_output[j][i];
        }
    }
    
    // Apply softmax to output layer
    output_layer = softmax(output_layer);
    
    return output_layer;
}

void NeuralNetwork::backward(const std::vector<float>& input, const std::vector<float>& target, float learningRate) {
    std::vector<float> output_errors(output_layer.size());
    for (int i = 0; i < output_layer.size(); ++i) {
        output_errors[i] = target[i] - output_layer[i];
    }
    
    std::vector<float> hidden_errors(hidden_layer.size(), 0.0f);
    for (int i = 0; i < hidden_layer.size(); ++i) {
        for (int j = 0; j < output_errors.size(); ++j) {
            hidden_errors[i] += output_errors[j] * weights_hidden_output[i][j];
        }
    }
    
    for (int i = 0; i < weights_hidden_output.size(); ++i) {
        for (int j = 0; j < weights_hidden_output[i].size(); ++j) {
            weights_hidden_output[i][j] += learningRate * output_errors[j] * hidden_layer[i];
        }
    }
    
    for (int i = 0; i < weights_input_hidden.size(); ++i) {
        for (int j = 0; j < weights_input_hidden[i].size(); ++j) {
            weights_input_hidden[i][j] += learningRate * hidden_errors[j] * sigmoid_derivative(hidden_layer[j]) * input[i];
        }
    }
    
    // Update biases
    for (int i = 0; i < bias_output.size(); ++i) {
        bias_output[i] += learningRate * output_errors[i];
    }
    
    for (int i = 0; i < bias_hidden.size(); ++i) {
        bias_hidden[i] += learningRate * hidden_errors[i] * sigmoid_derivative(hidden_layer[i]);
    }
}

float NeuralNetwork::calculate_loss(const std::vector<float>& target) {
    float loss = 0.0f;
    for (int i = 0; i < output_layer.size(); ++i) {
        loss += -target[i] * log(output_layer[i]);
    }
    return loss;
}

int main() {
    // Network parameters
    int input_size = 5;
    int hidden_size = 3;
    int output_size = 2;
    float learning_rate = 0.01;
    int epochs = 1000;
    
    // Initialize the network with these parameters
    NeuralNetwork nn(input_size, hidden_size, output_size);
    
    // Training data with larger input values
    std::vector<std::vector<float>> inputs = {
        {0.5, 2.0, 1.0, 0.0, 3.0},
        {4.0, 0.0, 0.0, 1.0, 1.5},
        {3.5, 1.2, 2.3, 1.0, 0.8},
        {2.0, 1.0, 0.0, 0.5, 3.0},
        {4.5, 0.5, 1.2, 0.7, 1.0},
        {1.1, 2.5, 0.0, 1.2, 2.3},
        {0.0, 3.3, 1.1, 2.2, 0.5},
        {1.4, 2.0, 3.0, 0.1, 0.9},
        {3.3, 1.5, 2.1, 1.9, 0.0},
        {0.2, 0.4, 2.0, 3.1, 1.2}
    };
    
    std::vector<std::vector<float>> outputs = {
        {0, 1},
        {1, 0},
        {1, 0},
        {0, 1},
        {1, 0},
        {0, 1},
        {0, 1},
        {0, 1},
        {1, 0},
        {1, 0}
    };
    
    // Training the network and printing loss
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (int i = 0; i < inputs.size(); ++i) {
            std::vector<float> prediction = nn.forward(inputs[i]);
            nn.backward(inputs[i], outputs[i], learning_rate);
            total_loss += nn.calculate_loss(outputs[i]);
        }
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }
    
    // Evaluate accuracy after training
    int correct_predictions = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        std::vector<float> prediction = nn.forward(inputs[i]);
        int predicted_label = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
        int actual_label = std::max_element(outputs[i].begin(), outputs[i].end()) - outputs[i].begin();
        if (predicted_label == actual_label) {
            correct_predictions++;
        }
    }
    
    float accuracy = (float)correct_predictions / inputs.size() * 100.0f;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
