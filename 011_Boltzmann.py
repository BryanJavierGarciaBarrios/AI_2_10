import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocesamiento de datos
train_images = train_images.astype(np.float32) / 255.0
train_images = train_images.reshape(train_images.shape[0], 784)

# Parámetros de la RBM
num_hidden = 128
num_visible = 784
learning_rate = 0.1
batch_size = 64
k = 1  # Número de pasos de Gibbs para el entrenamiento

# Variables y pesos
weights = tf.Variable(tf.random.normal([num_visible, num_hidden], 0.01), dtype=tf.float32)
visible_bias = tf.Variable(tf.zeros([num_visible]), dtype=tf.float32)
hidden_bias = tf.Variable(tf.zeros([num_hidden]), dtype=tf.float32)

# Función de sigmoide binaria
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

# Paso de muestreo Gibbs
def gibbs_sampling(hidden_prob):
    hidden_state = tf.nn.relu(tf.sign(hidden_prob - tf.random.uniform(tf.shape(hidden_prob)))
    return hidden_state

# Entrenamiento de la RBM
def train_rbm(visible_data):
    with tf.GradientTape() as tape:
        hidden_prob_0 = sigmoid(tf.matmul(visible_data, weights) + hidden_bias)
        hidden_state_0 = gibbs_sampling(hidden_prob_0)

        hidden_prob_k = hidden_prob_0
        hidden_state_k = hidden_state_0

        for _ in range(k):
            visible_prob_k = sigmoid(tf.matmul(hidden_state_k, tf.transpose(weights)) + visible_bias)
            visible_state_k = gibbs_sampling(visible_prob_k)

            hidden_prob_k = sigmoid(tf.matmul(visible_state_k, weights) + hidden_bias)
            hidden_state_k = gibbs_sampling(hidden_prob_k)

        positive_grad = tf.matmul(tf.transpose(visible_data), hidden_prob_0)
        negative_grad = tf.matmul(tf.transpose(visible_state_k), hidden_prob_k)

        delta_w = learning_rate * (positive_grad - negative_grad) / tf.dtypes.cast(tf.shape(visible_data)[0], tf.float32)
        delta_visible_bias = learning_rate * tf.reduce_mean(visible_data - visible_state_k, 0)
        delta_hidden_bias = learning_rate * tf.reduce_mean(hidden_prob_0 - hidden_prob_k, 0)

        # Actualizar los pesos y sesgos
        weights.assign_add(delta_w)
        visible_bias.assign_add(delta_visible_bias)
        hidden_bias.assign_add(delta_hidden_bias)

    return delta_w, delta_visible_bias, delta_hidden_bias

# Entrenamiento de la RBM
num_epochs = 10
for epoch in range(num_epochs):
    for i in range(0, train_images.shape[0], batch_size):
        batch = train_images[i:i + batch_size]
        delta_w, delta_visible_bias, delta_hidden_bias = train_rbm(batch)

        if i % 1000 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Delta W: {tf.reduce_mean(delta_w):.4f}")

# Visualización de las características aprendidas
def plot_features(features, n, title):
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(10, 10, i+1)
        plt.imshow(features[:, i].numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

num_features_to_show = 100
features = tf.transpose(weights)
plot_features(features, num_features_to_show, "Características aprendidas por la RBM")

