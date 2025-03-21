import numpy as np
from scipy.spatial.distance import cdist
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