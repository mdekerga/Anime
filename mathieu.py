import pandas as pd

file = "popular_anime.csv"

df = pd.read_csv(file)

#Nombre de valeurs sans colonnes
print("\n Missing Values (per column):")
print(df.isnull().sum())


print(df['episodes'].median())

df['episodes'] = df['episodes'].fillna(df['episodes'].median())




#Nettoyage de données

df["genres"] =df["genres"].fillna("Other")

