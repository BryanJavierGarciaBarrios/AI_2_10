import numpy as np

# Datos de ejemplo: Patrones de entrada (X) y salidas deseadas (Y)
X = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
Y = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]])

# Inicialización de los pesos sinápticos con ceros
num_inputs = X.shape[1]
num_outputs = Y.shape[1]
weights = np.zeros((num_inputs, num_outputs))

# Parámetro de aprendizaje (tasa de aprendizaje)
learning_rate = 0.1

# Entrenamiento con la regla de Hebb
for i in range(X.shape[0]):
    input_pattern = X[i, :].reshape(1, -1)
    output_pattern = Y[i, :].reshape(1, -1)
    weights += learning_rate * np.dot(input_pattern.T, output_pattern)

# Función para aplicar los pesos aprendidos
def apply_hebb_weights(input_pattern, weights):
    return np.dot(input_pattern, weights)

# Ejemplo de aplicación de los pesos aprendidos
new_input = np.array([[1, -1, -1]])  # Nuevo patrón de entrada
output = apply_hebb_weights(new_input, weights)
print("Resultado de la aplicación de los pesos aprendidos:", output)
