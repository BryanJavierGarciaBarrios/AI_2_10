import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.cost = []

        for _ in range(self.n_iterations):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# Datos de entrenamiento
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y = np.array([1, -1, -1, -1])

# Entrenar el Perceptrón Adaline
model = Adaline(learning_rate=0.1, n_iterations=100)
model.fit(X, y)

# Realizar predicciones
predictions = model.predict(X)
print("Predicciones:", predictions)
