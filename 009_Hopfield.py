import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    for pattern in patterns:
                        self.weights[i][j] += pattern[i] * pattern[j]

                    self.weights[i][j] /= self.num_neurons

    def activate(self, pattern):
        for _ in range(10):  # Número máximo de iteraciones
            for i in range(self.num_neurons):
                weighted_sum = np.dot(self.weights[i], pattern)
                if weighted_sum > 0:
                    pattern[i] = 1
                else:
                    pattern[i] = -1
        return pattern

# Ejemplo de uso
if __name__ == "__main__":
    # Definir patrones de entrada (matrices binarias)
    pattern1 = np.array([1, -1, 1, -1, 1])
    pattern2 = np.array([1, 1, 1, -1, -1])
    pattern3 = np.array([-1, -1, -1, 1, 1])

    patterns = [pattern1, pattern2, pattern3]

    # Crear una red de Hopfield con el número de neuronas igual al tamaño de los patrones
    num_neurons = len(pattern1)
    hopfield_net = HopfieldNetwork(num_neurons)

    # Entrenar la red de Hopfield con los patrones
    hopfield_net.train(patterns)

    # Definir un patrón de entrada (ruido en el patrón original)
    noisy_pattern = np.array([1, -1, 1, -1, -1])

    # Activar la red para recuperar el patrón original
    retrieved_pattern = hopfield_net.activate(noisy_pattern)

    print("Patrón original:", pattern1)
    print("Patrón con ruido:", noisy_pattern)
    print("Patrón recuperado:", retrieved_pattern)
