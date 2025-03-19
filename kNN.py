import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from collections import Counter

class KNN:
    def __init__(self, distance: str = 'euclidean', K: int = 3):
        self.distance = distance
        self.K = K

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Stocker les résultats
        predictions = np.zeros(len(X))

        for i, x in enumerate(X):
            # Calculer les distances entre x et tous les points d'entraînement
            distances = cdist(x.reshape(1, -1), self.X, metric=self.distance).flatten()
            
            # Trouver les indices des K plus proches voisins
            idxes = np.argsort(distances)[:self.K]
            
            # Trouver la classe majoritaire parmi les voisins
            k_nearest_labels = self.y[idxes]
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            
            predictions[i] = most_common_label

        return predictions

# Chargement des données
df = pd.read_csv("pokemon.csv")  # Assurez-vous d'avoir le fichier
X = df.drop(columns=['Legendary'])  # Features
y = df['Legendary']  # Cible

# Encodage des variables catégoriques
X = pd.get_dummies(X)

# Séparation des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conversion en numpy array
X_train = X_train.values.astype(float)
X_test = X_test.values.astype(float)
y_train = y_train.values.astype(int)  # Assurez-vous que y est de type int
y_test = y_test.values.astype(int)

# Création et entraînement du modèle
knn = KNN(distance='euclidean', K=3)
knn.train(X_train, y_train)

# Prédiction
y_pred = knn.predict(X_test)

# Calcul de la précision
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy}')

# Affichage des prédictions et valeurs réelles
print("Predictions:", y_pred)
print("Réelles:", y_test)
