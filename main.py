import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from kNN import KNN

csv_name = 'Pokemon.csv'
df = pd.read_csv(csv_name)

#print(df.head())

# Choix des variables inutiles 
useless_var = ['#', 'Name', 'Type 1', 'Type 2', 'Generation'] # Peut être sujet à modification
df_clean = df.drop(useless_var, axis=1)

df = df_clean.dropna()

# Séparer les features (X) et la cible (y)
X = df.drop(columns=['Legendary'])  # Features
y = df['Legendary']  # Cible

# Convertir X et y de booléen à entier (0 pour False, 1 pour True)
X = X.astype(int)
y = y.astype(int)


def algo_kfolding(model) :

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if str(type(model)) == "<class 'kNN.KNN'>":
            # Convertir les DataFrames en numpy arrays
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            # Entraîner le modèle
            model.train(X_train, y_train)

        else :
            # Entraîner le modèle
            model.fit(X_train, y_train)

        # Prédire et évaluer
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    print(f'type : {type(model)}')
    #print(f'Scores pour chaque fold: {scores}')
    print(f'Moyenne des scores: {np.mean(scores):.4f}')
    print(f'\n')

# Choix du modèle
model_arbre_decision = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
model_reseau_neurones = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
model_knn = KNN(distance='euclidean', K=3)

algo_kfolding(model_arbre_decision)
algo_kfolding(model_reseau_neurones)
algo_kfolding(model_knn)