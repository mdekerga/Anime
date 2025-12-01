import pandas as pd
import numpy as np

# ====================================================================
# --- Cleaning ---
# ====================================================================

df = pd.read_csv("C:/Users/babou/Downloads/AnimeProject/Anime/popular_anime.csv")
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

# ====================================================================
# --- Création du Subset 10% ---
# ====================================================================

# Pour analyse des 10%, création d'un subset avec uniquement les lignes contenant des valeurs de score.
score_anime = df.dropna(subset=['score', 'scored_by']).copy()
# Permet d'afficher plus que d'habitude
# pd.set_option('display.max_rows', None)
# Subset des 10% animés avec le plus haut score
top10pourcent = score_anime.head(int(len(score_anime) * 0.10)).copy()
print(top10pourcent)
# Détermination du seuil pour etre dans les 10 pourcents
seuil = top10pourcent['score'].min()
print(seuil)

# Créer une colonne is_top_10
df['is_top_10'] = df.index.isin(top10pourcent.index).astype(int)
print(df)

#Nombre de valeurs sans colonnes
print("\n Missing Values (per column):")
print(df.isnull().sum())
