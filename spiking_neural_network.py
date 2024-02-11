import numpy as np

class SpikingNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, inputs):
        # Input to hidden layer
        hidden_potentials = np.dot(inputs, self.weights_input_hidden)
        hidden_spikes = self.spike_activation(hidden_potentials)

        # Hidden to output layer
        output_potentials = np.dot(hidden_spikes, self.weights_hidden_output)
        output_spikes = self.spike_activation(output_potentials)

        return output_spikes

    def spike_activation(self, potentials, threshold=0):
        return (potentials > threshold).astype(int)

    def train(self, inputs, targets, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for input_sample, target_sample in zip(inputs, targets):
                # Forward pass
                hidden_potentials = np.dot(input_sample, self.weights_input_hidden)
                hidden_spikes = self.spike_activation(hidden_potentials)

                output_potentials = np.dot(hidden_spikes, self.weights_hidden_output)
                output_spikes = self.spike_activation(output_potentials)

                # Backpropagation using SpikeProp
                output_error = target_sample - output_spikes
                output_delta = output_error

                hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
                hidden_delta = hidden_error * (hidden_spikes * (1 - hidden_spikes))

                # Update weights
                self.weights_hidden_output += learning_rate * np.outer(hidden_spikes, output_delta)
                self.weights_input_hidden += learning_rate * np.outer(input_sample, hidden_delta)

            if epoch % 10 == 0:
                loss = np.mean(np.abs(output_error))
                print(f"Epoch {epoch}, Loss: {loss}")

def main():
    input_size = 2
    hidden_size = 3
    output_size = 1

    # XOR dataset
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Initialize and train the network
    snn = SpikingNeuralNetwork(input_size, hidden_size, output_size)
    snn.train(inputs, targets, epochs=1000)

    # Test the trained network
    for input_data in inputs:
        output = snn.forward(input_data)
        print(f"Input: {input_data}, Output: {output}")

if __name__ == "__main__":
    main()
