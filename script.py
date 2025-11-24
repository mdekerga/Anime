import pandas as pd
anime = pd.read_csv("C:/Users/babou/Downloads/AnimeProject/Anime/popular_anime.csv")
anime["genres"].fillna("Others",inplace=True)
anime["synopsis"].fillna("Pas de synopsis",inplace=True)
print(((anime.isnull().sum() / anime.shape[0]) * 100).round(2))
print(anime['genres'])
