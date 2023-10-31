import numpy as np

# Implementación simple de una red neuronal multicapa
class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.weights_input_hidden = np.random.rand(num_inputs, num_hidden)
        self.weights_hidden_output = np.random.rand(num_hidden, num_outputs)
        self.hidden_layer = np.zeros(num_hidden)
        self.output_layer = np.zeros(num_outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, input_data):
        self.hidden_layer = self.sigmoid(np.dot(input_data, self.weights_input_hidden))
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.output_layer
