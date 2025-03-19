from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy = clf.score(X_test, y_test)
print(accuracy)