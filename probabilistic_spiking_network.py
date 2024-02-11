import numpy as np

class ProbabilisticSpikingNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, inputs, time_steps):
        hidden_activations = np.zeros((time_steps, self.hidden_size))
        output_activations = np.zeros((time_steps, self.output_size))

        for t in range(time_steps):
            # Input to hidden layer
            hidden_activations[t] = self.sigmoid(np.dot(inputs, self.weights_input_hidden))

            # Hidden to output layer
            output_activations[t] = self.sigmoid(np.dot(hidden_activations[t], self.weights_hidden_output))

        return output_activations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def main():
    input_size = 2
    hidden_size = 3
    output_size = 1
    time_steps = 100

    # XOR dataset
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Initialize and run the network
    pnn = ProbabilisticSpikingNeuralNetwork(input_size, hidden_size, output_size)
    outputs = pnn.forward(inputs, time_steps)

    # Print output probabilities
    print("Output probabilities:")
    for output in outputs:
        print(output[-1])

if __name__ == "__main__":
    main()
