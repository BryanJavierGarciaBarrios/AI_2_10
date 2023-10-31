import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Datos de ejemplo
data = np.random.rand(100, 2)

# Crear e inicializar el SOM
som = MiniSom(10, 10, 2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)

# Entrenar el SOM
som.train(data, 100)

# Obtiene las coordenadas de los pesos del SOM
x, y = zip(*som.get_weights().reshape(-1, 2).T)

# Gráfica de los resultados
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()

# Marca los nodos del SOM
for i, j in enumerate(data):
    w = som.winner(j)
    plt.text(w[0]+0.5, w[1]+0.5, str(i), ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))

plt.show()
