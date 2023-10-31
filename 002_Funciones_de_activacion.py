import numpy as np

# Función de activación Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función de activación ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# Función de activación Tangente hiperbólica (Tanh)
def tanh(x):
    return np.tanh(x)

# Función de activación Softmax (para capas de salida en clasificación multiclase)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Mejora la estabilidad numérica
    return exp_x / exp_x.sum()

# Ejemplo de uso de las funciones de activación
x = np.array([-2, -1, 0, 1, 2])

print("Sigmoid:", sigmoid(x))
print("ReLU:", relu(x))
print("Tanh:", tanh(x))
print("Softmax:", softmax(x))
