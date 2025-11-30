import pandas as pd



file = "popular_anime.csv"

df = pd.read_csv(file)

#Nombre de valeurs sans colonnes
print("\n Missing Values (per column):")
print(df.isnull().sum())

#colonnes avec valeurs manquantes : genres, type, episodes, aired_from, aired_to, score, scored_by, rank, rating, studios, producers, trailer, synopsis


#Remplissage des colonnes vides

df["genres"] = df["genres"].fillna("Other")
df["episodes"] = df["episodes"].fillna(1)



#suppression des colonnes inutiles

colonne_a_drop = ['synopsis','trailer','producers']
df.drop(columns=colonne_a_drop)


print(df["aired_from"])


duplicate = df[df.duplicated()]



print("Duplicate Rows :")

# Print the resultant Dataframe
print(duplicate["name"])

#Nombre de lignes après le nettoyage
print(df.isnull().sum())
print(df.shape)


df["aired_from"] = df["aired_from"].fillna("Unknown")

#Création de nouvelle colonne
def get_season_anime(date):
    if(date != "Unknown"):
        date = pd.to_datetime(date)

        month = date.month

        season_index = (month - 1) // 3 
        
        seasons = ["Winter", "Spring", "Summer", "Fall"]
        
        return seasons[season_index]

print(get_season_anime("2009-04-05T00:00:00+00:00"))

df["season_aired"] = df["aired_from"].apply(get_season_anime)

#recherche de corrélation entre les variables
#print(df.corr())