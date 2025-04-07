# Utilisation du programme Python 
Ligne 14, ```csv_name = 'Pokemon2.csv'```
quel dataset utiliser ('Pokemon.csv', 'Pokemon2.csv', 'Pokemo_data.csv')
Ligne 176, ```test("arbre_decision",opti=False, norm=False)```

# Objectifs pédagogiques
Ce projet a pour objectif d’amener les étudiants à mettre en pratique les concepts
fondamentaux de l’apprentissage supervisé, en appliquant différentes méthodologies
sur un problème de classification. Ils devront :

— Comprendre et appliquer un workflow rigoureux en machine learning.

— Comparer l’efficacité de plusieurs algorithmes de classification : kNN, arbres de
décision et réseaux de neurones.

— Mettre en œuvre une validation croisée avec k-fold.

— Optimiser les hyperparamètres via une recherche par grille.

— Évaluer les performances des modèles avec des métriques adaptées.

— Rédiger un rapport structuré et analytique des résultats obtenus.

# Classification des Pokémon légendaires

Contexte : Les Pokémon sont des créatures aux caractéristiques variées, certaines étant
classées comme "légendaires" en raison de leur rareté et de leur puissance. L’objectif est
d’entraîner un modèle permettant de prédire si un Pokémon est légendaire ou non à
partir de ses statistiques.

Description des données : Le jeu de données comprend des informations sur 800
Pokémon, incluant des caractéristiques comme les points de vie (HP), l’attaque, la
défense, la vitesse, ainsi que des attributs catégoriels (type, génération, etc.).

Pistes à explorer :

— Sélection des meilleures caractéristiques pour la classification.

— Comparaison des performances des modèles (arbres de décision, kNN, réseaux de
neurones).

— Impact de la normalisation des données sur les résultats.

# Description des variables

#: l'ID de chaque Pokémon.

Name: le nom de chaque Pokémon.

Type 1: chaque Pokémon a un type, il détermine sa force/résistance aux attaques.

Type 2: Un Pokémon doté d'un double type est un Pokémon possédant deux types.

Total: la somme de toutes les statistiques qui viennent après cette variable, un indicateur global de la force d'un Pokémon.

HP: déterminent le nombre de dégâts qu'un Pokémon peut recevoir avant le K.O.

Attack: détermine partiellement la quantité de dégâts qu'un Pokémon provoque lorsqu'il utilise une capacité physique.

Defense: détermine partiellement la quantité de dégâts qu'un Pokémon subit lorsqu'il reçoit une capacité physique.

SP Atk: détermine partiellement la quantité de dégâts qu'un Pokémon provoque lorsqu'il utilise une capacité de catégorie spéciale.

SP Def: détermine partiellement la quantité de dégâts qu'un Pokémon subit lorsqu'il reçoit une capacité de catégorie spéciale.

Speed: détermine quel Pokémon lance la première attaque au début d'un tour.
