import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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

# Convertir y de booléen à entier (0 pour False, 1 pour True)
X = X.astype(int)
# Convertir y de booléen à entier (0 pour False, 1 pour True)
y = y.astype(int)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convertir les DataFrames en numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Création et entraînement du modèle
    knn = KNN(distance='euclidean', K=3)
    knn.train(X_train, y_train)

    # Prédire et évaluer
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

print(f'Scores pour chaque fold: {scores}')
print(f'Moyenne des scores: {np.mean(scores):.4f}')

