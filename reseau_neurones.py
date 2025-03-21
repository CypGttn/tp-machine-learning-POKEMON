from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from traitement_donnees import *

#print(df_clean.head())
df = df_clean.dropna()

# Séparer les features (X) et la cible (y)
X = df.drop(columns=['Legendary'])  # Features
y = df['Legendary']  # Cible

# Convertir y de booléen à entier (0 pour False, 1 pour True)
y = y.astype(int)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Créer et entraîner le modèle
    model = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
    model.fit(X_train, y_train.values.ravel())  # .ravel() pour convertir y en 1D

    # Prédire et évaluer
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

print(f'Scores pour chaque fold: {scores}')
print(f'Moyenne des scores: {np.mean(scores):.4f}')
