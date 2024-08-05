# fastNeuralNetwork
Fast Neural Network with pure numpy, numba, cupy, cpp and etc

Artificial neural network applications are often programmed using TensorFlow and similar modules. When programming with these modules, only high-level commands are visible, and the details of the underlying processes are not apparent. Few people truly understand what a neural network is, as most are only programming with high-level languages.
For many years, the process by which the brain learns remained a mystery. The primary mechanism of learning in the brain is through the adjustment of synaptic weights at the connections between neurons. These synaptic weights determine the strength of the signal transmitted from one neuron to another. 

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

The brain operates with an extraordinarily complex system of billions of neurons, where the function can be approximated as:

$$
 \text{Output} = F(\text{weights} \cdot \text{inputs} + \text{biases}) = A x^N + B x^{N-1} + C x^{N-2} + \ldots + W x^2 + Y x + Z 
$$

1. **Weights and Inputs**: Each neuron receives multiple inputs, each associated with a weight. The weighted sum of these inputs, plus a bias, determines the neuron's output.
   
2. **Function \(F\)**: This represents the activation function, which processes the weighted sum of inputs and biases to produce the neuron's final output. The activation function could be nonlinear, allowing for complex and adaptive behavior.

3. **Polynomial Representation**: The polynomial representation

$$ \(A x^N + B x^{N-1} + C x^{N-2} + \ldots + W x^2 + Y x + Z\) $$

is a simplified way to illustrate how different weights and biases contribute to the final output. In this polynomial, \(x\) represents the input, and the coefficients \(A, B, C, \ldots, Z\) represent the weights and biases.

The exact order of terms (e.g., \(x^{1,000,000,000}\)) is a metaphorical representation to convey the vast number of connections and their complexity, rather than literal polynomial terms. In reality, the actual structure of the brain’s processing is not literally polynomial but involves highly intricate interactions among billions of neurons and their synaptic connections.
