import pandas as pd

csv_name = 'Pokemon_data.csv'
df = pd.read_csv(csv_name)

# Choix des variables inutiles 
#useless_var_pokemon1 = ['#', 'Name', 'Type 1', 'Type 2', 'Generation'] # Peut être sujet à modification
useless_var = ["name","generation","classification","abilities","type1","type2","against_bug","against_dark","against_dragon","against_electric","against_fairy","against_fighting","against_fire","against_flying","against_ghost","against_grass","against_ground","against_ice","against_normal","against_poison","against_psychic","against_rock","against_steel","against_water","is_mythical"]

df_clean = df.drop(useless_var, axis=1)

# Convert boolean columns to integers
boolean_columns = df_clean.select_dtypes(include=['bool']).columns
df_clean[boolean_columns] = df_clean[boolean_columns].astype(int)

# Convert all object columns to categorical and then to numerical
object_columns = df_clean.select_dtypes(include=['object']).columns
df_clean[object_columns] = df_clean[object_columns].apply(lambda x: pd.factorize(x)[0])

# Normalisation des données
df_clean = (df_clean - df_clean.min()) / (df_clean.max() - df_clean.min())
df_norm = df_clean.dropna()

