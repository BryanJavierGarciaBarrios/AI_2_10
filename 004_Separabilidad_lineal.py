import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Crear datos de ejemplo
np.random.seed(0)
X1 = np.random.randn(100, 2) + np.array([2, 2])
X2 = np.random.randn(100, 2) + np.array([-2, -2])
X = np.vstack([X1, X2])
y = np.array([0] * 100 + [1] * 100)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una red neuronal simple
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_dim=2)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=0)

# Evaluar el modelo en los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")
print(f"Precisión en el conjunto de prueba: {accuracy}")

# Predicciones en nuevos datos
new_data = np.array([[3, 3], [-3, -3]])
predictions = model.predict(new_data)

for i, p in enumerate(predictions):
    print(f"Predicción para {new_data[i]}: {p[0]} (Clase {int(round(p[0]))})")
