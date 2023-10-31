from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Crear un conjunto de datos de ejemplo
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de red neuronal con retropropagación del error
modelo = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predicciones = modelo.predict(X_test)

# Calcular la precisión
precisión = accuracy_score(y_test, predicciones)
print("Precisión del modelo:", precisión)
