pip install tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Cargar el conjunto de datos MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar las imágenes y las etiquetas
train_images, test_images = train_images / 255.0, test_images / 255.0

# Crear el modelo de red neuronal
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Capa de entrada (aplanamiento de imágenes 28x28)
    layers.Dense(128, activation='relu'),  # Capa oculta con activación ReLU
    layers.Dropout(0.2),  # Regularización con dropout
    layers.Dense(10, activation='softmax')  # Capa de salida con 10 clases y activación softmax
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=5)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Pérdida en el conjunto de prueba: {test_loss}")
print(f"Precisión en el conjunto de prueba: {test_accuracy}")

# Realizar predicciones en nuevas imágenes
sample_image = np.expand_dims(test_images[0], axis=0)  # Tomar una imagen de prueba
predictions = model.predict(sample_image)
predicted_label = np.argmax(predictions)
print(f"Predicción para la imagen de prueba: {predicted_label}")
