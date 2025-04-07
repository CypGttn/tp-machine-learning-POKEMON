import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from kNN import KNN
from normalisation import *

#csv_name possibles (classé du moins complet au plus complet) : 'Pokemon.csv', 'Pokemon2.csv', 'Pokemo_data.csv'
csv_name = 'Pokemon_data.csv'
df = pd.read_csv(csv_name)

#print(df.head())

# Choix des variables inutiles 
#useless_var_pokemon1 = ['#', 'Name', 'Type 1', 'Type 2', 'Generation'] # Peut être sujet à modification
useless_var = ["name","generation","classification","abilities","type1","type2","against_bug","against_dark","against_dragon","against_electric","against_fairy","against_fighting","against_fire","against_flying","against_ghost","against_grass","against_ground","against_ice","against_normal","against_poison","against_psychic","against_rock","against_steel","against_water","is_mythical"]
df_clean = df.drop(useless_var, axis=1, errors='ignore')
df = df_clean.dropna()
df = pd.get_dummies(df)

# Séparer les features (X) et la cible (y)
X = df.drop(columns=['is_legendary'])  # Features
y = df['is_legendary']  # Cible

# Convertir X et y de booléen à entier (0 pour False, 1 pour True)
X = X.astype(int)
y = y.astype(int)


def algo_kfolding(X, y, model) :

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
model_arbre_decision = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model_reseau_neurones = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
model_knn = KNN(distance='euclidean', K=2)


#algo_kfolding(X, y, model_arbre_decision)
#algo_kfolding(X, y, model_reseau_neurones)
#algo_kfolding(X, y, model_knn)


param_grid_arbre_decision = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
}

param_grid_reseau_neurones = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "lbfgs"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.01, 0.001, 0.0001],
}

# Définition des métriques de scoring
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, average="weighted"),
    "recall": make_scorer(recall_score, average="weighted"),
    "f1": make_scorer(f1_score, average="weighted")
}

def algo_kfold_and_gridsearch(X, y, type_model : str = "arbre_decision", param_grid : list = param_grid_arbre_decision, scoring =scoring, kfold_n_split : int = 5, random_state : int = 42):
    # Initialisation du modèle
    if type_model == "arbre_decision":
        model = DecisionTreeClassifier()
    elif type_model == "reseau_neurones":
        model = MLPClassifier()

    # Définition du K-Fold
    kf = KFold(n_splits=kfold_n_split, shuffle=True, random_state=random_state)

    # GridSearchCV avec K-Fold
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=scoring, refit="f1", n_jobs=-1)
    grid_search.fit(X, y)

    # Affichage des meilleurs paramètres et du score
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleure précision moyenne :", grid_search.best_score_)

#algo_kfold_and_gridsearch(X, y, "arbre_decision", param_grid_arbre_decision)
"""Meilleurs paramètres : {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5}
Meilleure précision moyenne : 0.9805078386870774"""
#algo_kfold_and_gridsearch(X, y, "reseau_neurones", param_grid_reseau_neurones)
#algo_kfolding(X, y, model_knn)


#"########################" partie normalisation
# Séparer les features (X) et la cible (y)
X_norm = df_norm.drop(columns=['is_legendary'])  # Features
y_norm = df_norm['is_legendary']  # Cible

# Convertir X et y de booléen à entier (0 pour False, 1 pour True)
X_norm = X_norm.astype(int)
y_norm = y_norm.astype(int)

#algo_kfolding(X_norm, y_norm, model_arbre_decision)
#algo_kfolding(X_norm, y_norm, model_reseau_neurones)
#algo_kfolding(X_norm, y_norm, model_knn)

#algo_kfold_and_gridsearch(X_norm, y_norm, "arbre_decision", param_grid_arbre_decision)
#algo_kfold_and_gridsearch(X_norm, y_norm, "reseau_neurones", param_grid_reseau_neurones)

def test(type_model = "arbre_decision", opti = True, norm = False):
    print(f"Modèle choisi : {type_model}\nopti : {opti}\ndonnées normalisé : {norm}")
    # Initialisation du modèle
    grideable = False
    if type_model == "arbre_decision":
        model = DecisionTreeClassifier()
        param_grid = param_grid_arbre_decision
        grideable = True

    elif type_model == "reseau_neurones":
        model = MLPClassifier()
        param_grid = param_grid_reseau_neurones
        grideable = True

    elif type_model == "knn":
        model = KNN(distance='euclidean', K=2)

    if grideable:
        if opti:
            if norm:
                algo_kfold_and_gridsearch(X_norm, y_norm, type_model, param_grid)
            else:
                algo_kfold_and_gridsearch(X, y, type_model, param_grid)
        else:
            if norm:
                algo_kfolding(X_norm, y_norm, model)
            else:
                algo_kfolding(X, y, model)
    else:
        if norm:
            algo_kfolding(X_norm, y_norm, model_knn)
        else:
            algo_kfolding(X, y, model_knn)
       
test("arbre_decision",opti=False, norm=False)

        
