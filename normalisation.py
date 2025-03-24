import pandas as pd

csv_name = 'Pokemon.csv'
df = pd.read_csv(csv_name)

# Choix des variables inutiles 
useless_var = ['#', 'Name', 'Type 1', 'Type 2', 'Generation'] # Peut être sujet à modification
df_clean = df.drop(useless_var, axis=1)

# Convert boolean columns to integers
boolean_columns = df_clean.select_dtypes(include=['bool']).columns
df_clean[boolean_columns] = df_clean[boolean_columns].astype(int)

# Convert all object columns to categorical and then to numerical
object_columns = df_clean.select_dtypes(include=['object']).columns
df_clean[object_columns] = df_clean[object_columns].apply(lambda x: pd.factorize(x)[0])

# Normalisation des données
df_clean = (df_clean - df_clean.min()) / (df_clean.max() - df_clean.min())
df = df_clean.dropna()

print(df.head())