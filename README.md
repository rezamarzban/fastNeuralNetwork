# fastNeuralNetwork
Fast Neural Network with pure numpy, numba, cupy, cpp and etc

Artificial neural network applications are often programmed using TensorFlow and similar modules. When programming with these modules, only high-level commands are visible, and the details of the underlying processes are not apparent. Few people truly understand what a neural network is, as most are only programming with high-level languages.
For many years, the process by which the brain learns remained a mystery. **The primary mechanism of learning in the brain is through the adjustment of synaptic weights at the connections between neurons.** These synaptic weights determine the strength of the signal transmitted from one neuron to another. 

When we learn something new, neural pathways are either strengthened or weakened based on the frequency and intensity of the signals passing through them. This process is known as synaptic plasticity. The more a particular pathway is used, the stronger the connections become, making the signal transmission more efficient. Conversely, pathways that are rarely used may weaken over time.

This mechanism is crucial for various types of learning and memory formation, enabling the brain to adapt and reorganize itself in response to new information, experiences, and even injury. By understanding these processes, researchers can develop more effective artificial neural networks and enhance our comprehension of human cognition and neural disorders.

**The main formula behind the neural connections in the brain that facilitate learning can be expressed as:**

$$
 \text{Output} = w \cdot \text{input} + \text{bias} 
$$


1. **Input**: This represents the signal received by a neuron from other neurons.
2. **Weight (w)**: This is a parameter that adjusts the strength of the input signal. Each connection between neurons has a weight that can increase or decrease the signal's impact.
3. **Bias**: This is an additional parameter that allows the neuron to shift the output function, providing greater flexibility in the learning process.

When a neuron receives inputs, each input is multiplied by its corresponding weight. The sum of these weighted inputs is then added to the bias. The resulting value determines the neuron's output, which can then be passed to other neurons.

In essence, learning occurs through the adjustment of these weights and biases. By modifying them, the neural network can better match inputs to desired outputs, improving its performance over time. This adjustment process is known as training in artificial neural networks and synaptic plasticity in biological brains.

The brain contains approximately 85 billion neurons, which form an immense number of connections, each with its own weights and biases. This complex network allows for highly efficient learning.


1. **Neurons**: The brain's basic functional units are neurons. Each neuron can connect to thousands of other neurons, forming a vast network.
2. **Connections (Synapses)**: These connections, or synapses, are where learning occurs. Each synapse has a weight and a bias.
3. **Weights**: Weights determine the strength or importance of the input signals. Adjusting these weights allows the brain to prioritize certain signals over others.
4. **Biases**: Biases help fine-tune the output of neurons, making the neural network more adaptable and responsive.

When the brain learns, it adjusts the weights and biases of these connections based on experiences and information. This process is dynamic and ongoing, enabling the brain to continuously adapt and optimize its performance. The sheer number of neurons and connections creates a highly sophisticated and flexible system capable of complex thought, learning, and memory formation.

The brain can be conceptualized as a highly complex function that takes multiple inputs (x1, x2, ..., xn) and produces multiple outputs (y1, y2, ..., ym). This function, F, involves a vast network of interconnected neurons, each with its own weights and biases. These weights and biases are adjusted through learning processes to map inputs to desired outputs.

Formally, this can be described as:

$$ (y1, y2, ..., ym) = F(x1, x2, ..., xn) $$

The function \( F \) represents the combined effect of many neurons, each applying a weighted sum of inputs plus a bias, followed by an activation function. This process can be incredibly complex, **involving billions of parameters (weights and biases), making the brain's mapping function analogous to a polynomial of a very high order, possibly in the range of billions.**

In neural network terms, this can be broken down as follows:

1. **Inputs (x1, x2, ..., xn):** Sensory or other types of data received by the brain.
2. **Weights and biases:** Parameters that are adjusted through learning to improve the accuracy of the function \( F \).
3. **Activation function:** A nonlinear function applied to the weighted sum of inputs to introduce nonlinearity into the model, allowing it to capture more complex patterns.
4. **Outputs (y1, y2, ..., ym):** The resulting actions, thoughts, or other responses generated by the brain.

This intricate process allows the brain to perform a wide array of tasks, from simple reflexes to complex problem-solving and decision-making.

Let's break down the brain's function with a simplified neural network example, using the given inputs, neurons, and outputs. We'll consider a neural network with one hidden layer for simplicity.

#### Inputs and Neurons

- **Inputs (x):** \( x1, x2, x3, x4, x5 \)
- **Neurons in Hidden Layer 1 (h):** 3 neurons
- **Outputs (y):** \( y1, y2 \)

#### Structure of the Neural Network

1. **Input Layer:** 5 inputs.
2. **Hidden Layer:** 3 neurons.
3. **Output Layer:** 2 outputs.

#### Weights and Biases

- **Weights between Input Layer and Hidden Layer:** 

$$ W_{1} = \begin{bmatrix}
  w_{11} & w_{12} & w_{13} \\
  w_{21} & w_{22} & w_{23} \\
  w_{31} & w_{32} & w_{33} \\
  w_{41} & w_{42} & w_{43} \\
  w_{51} & w_{52} & w_{53}
  \end{bmatrix} $$
  
  Here, \( W1 \) is a 5x3 matrix (5 inputs, 3 neurons in the hidden layer).

- **Biases for Hidden Layer:**

$$ b_{1} = \begin{bmatrix}
  b_{11} \\
  b_{12} \\
  b_{13}
  \end{bmatrix} $$
  
  Here, \( b1 \) is a 3x1 vector (one bias per hidden neuron).

- **Weights between Hidden Layer and Output Layer:**

$$ W_{2} = \begin{bmatrix}
  w_{h1y1} & w_{h1y2} \\
  w_{h2y1} & w_{h2y2} \\
  w_{h3y1} & w_{h3y2}
  \end{bmatrix} $$
  
  Here, \( W2 \) is a 3x2 matrix (3 neurons in the hidden layer, 2 outputs).

- **Biases for Output Layer:**

$$ b_{2} = \begin{bmatrix}
  b_{21} \\
  b_{22}
  \end{bmatrix} $$
  
  Here, \( b2 \) is a 2x1 vector (one bias per output neuron).

#### Calculating the Output

1. **Hidden Layer Activation:**

$$ h = f(W_{1} \cdot x + b_{1}) $$

   Where \( x \) is the input vector:
   
   $$ \begin{bmatrix} x_1 & x_2 & x_3 & x_4 & x_5 \end{bmatrix}^T $$
   
   and \( f \) is the activation function (e.g., ReLU, sigmoid).

3. **Output Layer Activation:**

$$ y = g(W_{2} \cdot h + b_{2}) $$

   Where \( h \) is the hidden layer output vector and \( g \) is the activation function for the output layer (often softmax for classification or linear for regression).

#### Number of Weights and Biases

- **Weights between Input and Hidden Layer:**

 $$ 5 \text{ inputs} \times 3 \text{ neurons} = 15 \text{ weights} $$
  
- **Biases for Hidden Layer:**

 $$ 3 \text{ biases} $$
 
- **Weights between Hidden and Output Layer:**

 $$ 3 \text{ neurons} \times 2 \text{ outputs} = 6 \text{ weights} $$
 
- **Biases for Output Layer:**

 $$ 2 \text{ biases} $$

**Total Weights and Biases:**

$$ 15 + 3 + 6 + 2 = 26 \text{ parameters} $$

Here’s the code with TensorFlow module to define, compile, and summarize the neural network model as described `simpleNN_TF.ipynb`:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential([
    Dense(3, input_shape=(5,), activation='relu', name='hidden_layer'),  # Hidden layer with 3 neurons
    Dense(2, activation='linear', name='output_layer')  # Output layer with 2 neurons
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model.summary()
```

- **Model Definition:** A `Sequential` model is used with two `Dense` layers:
  - The first `Dense` layer has 3 neurons, uses ReLU activation, and takes an input shape of 5 (for the 5 input features).
  - The second `Dense` layer has 2 neurons with a linear activation function for the output.
  
- **Compile:** The model is compiled with the Adam optimizer and mean squared error loss function.

- **Summary:** `model.summary()` prints the model architecture, including the number of parameters and layer details.

Parameters saved in a neural network model, such as weights and biases, are analogous to the synaptic strengths in the brain. These parameters define the mapping function \( F \) that transforms the inputs \( (x1, x2, x3, x4, x5) \) to the outputs \( (y1, y2) \). This transformation can be thought of as a complex polynomial function.

In summary:

The neural network model parameters (weights and biases) function similarly to the synapses in the brain. They define a mapping function \( F \) such that:

$$ (y1, y2) = F(x1, x2, x3, x4, x5) $$

This function \( F \) represents the intricate polynomial-like transformations performed by the network to produce the desired outputs from given inputs.

Let's break down the output calculation from the neural network into a form that explicitly shows the linear combination of inputs with their respective weights and biases. For simplicity, I'll provide a direct calculation for \( y1 \) and \( y2 \) based on the hidden layer activations and weights.

#### Given:
1. **Inputs (x):**

$$ \( x_1, x_2, x_3, x_4, x_5 \) $$

3. **Weights and biases for the hidden layer (from the input to hidden layer):**

$$ W_1 = \begin{bmatrix}
   w_{11} & w_{12} & w_{13} \\
   w_{21} & w_{22} & w_{23} \\
   w_{31} & w_{32} & w_{33} \\
   w_{41} & w_{42} & w_{43} \\
   w_{51} & w_{52} & w_{53}
   \end{bmatrix} $$
   
$$ b_1 = \begin{bmatrix}
   b_{11} \\
   b_{12} \\
   b_{13}
   \end{bmatrix} $$

4. **Weights and biases for the output layer (from the hidden layer to the output layer):**

$$ W_2 = \begin{bmatrix}
   w_{h1y1} & w_{h1y2} \\
   w_{h2y1} & w_{h2y2} \\
   w_{h3y1} & w_{h3y2}
   \end{bmatrix} $$

$$ b_2 = \begin{bmatrix}
   b_{21} \\
   b_{22}
   \end{bmatrix} $$

#### Hidden Layer Activations:
Let \( h_1, h_2, h_3 \) be the activations of the hidden layer neurons. These are calculated as:

$$ h_1 = f(w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + w_{41}x_4 + w_{51}x_5 + b_{11}) $$

$$ h_2 = f(w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + w_{42}x_4 + w_{52}x_5 + b_{12}) $$

$$ h_3 = f(w_{13}x_1 + w_{23}x_2 + w_{33}x_3 + w_{43}x_4 + w_{53}x_5 + b_{13}) $$

where \( f \) is the activation function (e.g., ReLU).

#### Output Layer Calculations:
Using the hidden layer activations, the outputs \( y_1 \) and \( y_2 \) are computed as:

$$ y_1 = w_{h1y1}h_1 + w_{h2y1}h_2 + w_{h3y1}h_3 + b_{21} $$

$$ y_2 = w_{h1y2}h_1 + w_{h2y2}h_2 + w_{h3y2}h_3 + b_{22} $$

Substituting the hidden layer activations into these equations:

$$ y_1 = w_{h1y1} \cdot f(w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + w_{41}x_4 + w_{51}x_5 + b_{11}) + w_{h2y1} \cdot f(w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + w_{42}x_4 + w_{52}x_5 + b_{12}) + w_{h3y1} \cdot f(w_{13}x_1 + w_{23}x_2 + w_{33}x_3 + w_{43}x_4 + w_{53}x_5 + b_{13}) + b_{21} $$

$$ y_2 = w_{h1y2} \cdot f(w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + w_{41}x_4 + w_{51}x_5 + b_{11}) + w_{h2y2} \cdot f(w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + w_{42}x_4 + w_{52}x_5 + b_{12}) + w_{h3y2} \cdot f(w_{13}x_1 + w_{23}x_2 + w_{33}x_3 + w_{43}x_4 + w_{53}x_5 + b_{13}) + b_{22} $$

#### Simplified Form:
For clarity, if we assume \( f \) is an identity function (no activation), the equations reduce to:

$$ y_1 = w_{h1y1}(w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + w_{41}x_4 + w_{51}x_5 + b_{11}) + w_{h2y1}(w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + w_{42}x_4 + w_{52}x_5 + b_{12}) + w_{h3y1}(w_{13}x_1 + w_{23}x_2 + w_{33}x_3 + w_{43}x_4 + w_{53}x_5 + b_{13}) + b_{21} $$

$$ y_2 = w_{h1y2}(w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + w_{41}x_4 + w_{51}x_5 + b_{11}) + w_{h2y2}(w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + w_{42}x_4 + w_{52}x_5 + b_{12}) + w_{h3y2}(w_{13}x_1 + w_{23}x_2 + w_{33}x_3 + w_{43}x_4 + w_{53}x_5 + b_{13}) + b_{22} $$

These equations show how the outputs \( y_1 \) and \( y_2 \) are calculated as linear combinations of the inputs, with weights and biases playing the role of parameters in this complex mapping function.
