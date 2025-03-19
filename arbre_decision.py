from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

from traitement_donnees import *

#print(df_clean.head())
df = df_clean.dropna()

# Séparer les features (X) et la cible (y)
X = df.drop(columns=['Legendary'])  # Features
y = df['Legendary']  # Cible

#print(f'X:\n{X.head()}')
print(f'y:\n{y.head()}')

# Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

################
# Modélisation #
################

# Créer et entraîner le modèle DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=4)
clf.fit(X_train, y_train)

# Prédire et évaluer le modèle
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

plt.figure(figsize=(16, 8))
tree.plot_tree(clf, 
          feature_names=X.columns,  # Affiche les noms des variables
          filled=True,  # Coloration des noeuds
          rounded=True,  # Coins arrondis pour une meilleure lisibilité
          fontsize=10)  # Ajuster la taille du texte si nécessaire
plt.show()

print(f"Précision du modèle : {accuracy}")