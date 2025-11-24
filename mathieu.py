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

#création de nouvelles colonnes : pe

#suppression des colonnes inutiles

colonne_a_drop = ['score','synopsis','trailer']
df.drop(columns=colonne_a_drop)


#Nombre de lignes après le nettoyage
print(df.isnull().sum())
print(df.shape)


duplicate = df[df.duplicated()]

print("Duplicate Rows :")

# Print the resultant Dataframe
print(duplicate["name"])


#recherche de corrélation entre les variables
#print(df.corr())