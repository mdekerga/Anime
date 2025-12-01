import pandas as pd
import numpy as np



file = "popular_anime.csv"

df = pd.read_csv(file)

#Nombre de lignes avant le nettoyage
print(df.shape)

#Nombre de valeurs sans colonnes
print("\n Missing Values (per column):")
print(df.isnull().sum())

#Remplissage des colonnes vides
df["genres"] = df["genres"].fillna("Other")
df["episodes"] = df["episodes"].fillna(1)

#suppression des colonnes inutiles
colonne_a_drop = ['synopsis','trailer','producers']
df = df.drop(columns=colonne_a_drop)

#suppression de ligne 
df = df.dropna(subset=['score', 'episodes', 'scored_by', 'rank',"rating","studios"])

#Suppression des duplications
df = df.drop_duplicates()

#réinitilise l'index 
df.reset_index(drop=True, inplace=True)

df["aired_from"] = df["aired_from"].fillna(pd.NaT)
df['aired_to'] = df['aired_to'].fillna(pd.NaT)

#Création de nouvelle colonne
def get_season_anime(date):
    if(pd.isna(date)):
        return np.nan

    date = pd.to_datetime(date)

    month = date.month

    season_index = (month - 1) // 3 
        
    seasons = ["Winter", "Spring", "Summer", "Fall"]
        
    return seasons[season_index]

print(get_season_anime("2009-04-05T00:00:00+00:00"))

df["season_aired"] = df["aired_from"].apply(get_season_anime)

#Nombre de lignes après le nettoyage
print(df.isnull().sum())
print(df.shape)



#recherche de corrélation entre les variables
#print(df.corr())