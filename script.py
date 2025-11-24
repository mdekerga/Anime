# IMPORT
import pandas as pd
# READ CSV
anime = pd.read_csv("C:/Users/babou/Downloads/AnimeProject/Anime/popular_anime.csv")#
# Replace all lines with no value in genres by Others
anime["genres"].fillna("Others",inplace=True)
# Replace all lines with no value in synopsis by Pas de synopsis
anime["synopsis"].fillna("Pas de synopsis",inplace=True)
# Show the percent of missing values by variable
print(((anime.isnull().sum() / anime.shape[0]) * 100).round(2))
# Print the genres
print(anime['genres'])
# Pour analyse des 10%, créer un subset avec uniquement les lignes contenant des valeurs de score.
score_anime = anime.dropna(subset=['score', 'scored_by']).copy()
# Permet d'afficher plus que d'habitude
pd.set_option('display.max_rows', None)
# Juste les 1000 premiers scores
print(score_anime['score'].head(1000))
