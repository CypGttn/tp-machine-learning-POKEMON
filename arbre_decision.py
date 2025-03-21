from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
import pandas as pd
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
    model = DecisionTreeClassifier(criterion='gini', max_depth=4)
    model.fit(X_train, y_train)

    # Prédire et évaluer
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

print(f'Scores pour chaque fold: {scores}')
print(f'Moyenne des scores: {np.mean(scores):.4f}')


"""
plt.figure(figsize=(16, 8))
tree.plot_tree(clf, 
          feature_names=X.columns,  # Affiche les noms des variables
          filled=True,  # Coloration des noeuds
          rounded=True,  # Coins arrondis pour une meilleure lisibilité
          fontsize=10)  # Ajuster la taille du texte si nécessaire
plt.show()
"""
