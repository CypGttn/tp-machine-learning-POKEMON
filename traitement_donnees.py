import pandas as pd


csv_name = 'Pokemon.csv'
df = pd.read_csv(csv_name)

# Choix des variables inutiles 
useless_var = ['#', 'Name', 'Type 1', 'Type 2', 'Generation'] # Peut être sujet à modification
df_clean = df.drop(useless_var, axis=1)
