# ====================================================================
# ÉTAPE 0 : IMPORTS
# ====================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

# Ignorer les avertissements (souvent causés par des problèmes de singularité)
warnings.filterwarnings('ignore')

# ====================================================================
# ÉTAPE 1 : NETTOYAGE ET PRÉPARATION DE LA VARIABLE CIBLE (Y)
# ====================================================================

# NOTE IMPORTANTE : Ajustez ce chemin de fichier
try:
    df = pd.read_csv("C:/Users/babou/Downloads/AnimeProject/Anime/popular_anime.csv")
except FileNotFoundError:
    print("ERREUR FATALE: Fichier non trouvé. Veuillez vérifier et ajuster le chemin du fichier CSV.")
    exit()

# Remplissage des valeurs manquantes initiales
df["genres"].fillna("Other", inplace=True)
df["episodes"].fillna(1, inplace=True)

# Suppression des colonnes inutiles
colonne_a_drop = ['synopsis','trailer','producers']
df = df.drop(columns=colonne_a_drop, errors='ignore')

# Assurer le type numérique et nettoyer les lignes manquantes
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['scored_by'] = pd.to_numeric(df['scored_by'], errors='coerce')
df = df.dropna(subset=['score', 'episodes', 'scored_by', 'rank',"rating","studios"])
df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)

# Nettoyage des anomalies de score (> 10.0)
df = df[df['score'] <= 10.0].copy()

# Gestion des dates et création de la saison
def get_season_anime(date):
    if pd.isna(date): return np.nan
    try:
        date = pd.to_datetime(date)
    except ValueError: return np.nan
    month = date.month
    season_index = (month - 1) // 3  
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    return seasons[season_index]

df["season_aired"] = df["aired_from"].apply(get_season_anime)
df.dropna(subset=['season_aired'], inplace=True)

# --- CRÉATION DE LA VARIABLE CIBLE (Y) ---
# 1. Tri par score pour définir le Top 10%
anime_classe_simple = df.sort_values(by='score', ascending=False)
top_10_count = int(len(anime_classe_simple) * 0.10)
seuil_score_top_10 = anime_classe_simple.iloc[top_10_count-1]['score']

# 2. Création de la variable cible Y : is_top_10
df['is_top_10'] = (df['score'] >= seuil_score_top_10).astype(int)
df.reset_index(drop=True, inplace=True)


# ====================================================================
# ÉTAPE 2 : ENCODAGE DES VARIABLES EXPLICATIVES (X)
# ====================================================================

# 1. Encodage Genres (Multi-étiquettes)
df['genres_list'] = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(df['genres_list']),
                             columns=['Genre_' + genre for genre in mlb.classes_],
                             index=df.index)
df_encoded = pd.concat([df.reset_index(drop=True), genre_dummies], axis=1)

# 2. Encodage variables simples (type, rating, season_aired, first_studio)
df_encoded['first_studio'] = df_encoded['studios'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else np.nan)
df_encoded.dropna(subset=['first_studio', 'season_aired'], inplace=True) 

cols_to_encode = ['type', 'rating', 'season_aired', 'first_studio']

df_encoded = pd.get_dummies(
    df_encoded, 
    columns=cols_to_encode, 
    prefix=cols_to_encode, 
    drop_first=True 
)


# ====================================================================
# ÉTAPE 3 : MODÉLISATION ET ENTRAÎNEMENT
# ====================================================================

# Définition de X et Y
# Solution robuste: Sélectionner UNIQUEMENT les colonnes numériques
X = df_encoded.select_dtypes(include=[np.number])

# Exclure les colonnes qui ne sont pas des features binaires (Y, score, etc.)
cols_a_exclure = ['score', 'scored_by', 'rank', 'episodes', 'is_top_10']
X = X.drop(columns=cols_a_exclure, errors='ignore')

# Supprimer les colonnes peu variables et ajouter la constante
X = X.loc[:, X.nunique() > 1] 
X = sm.add_constant(X, prepend=False)
Y = df_encoded['is_top_10']

# Entraînement du modèle (GLM Binomial)
logit_model = sm.GLM(Y, X, family=sm.families.Binomial())
result = logit_model.fit()
model_features = X.columns # Stocker les noms des features pour la prédiction

print("\n" + "="*70)
print("RÉSUMÉ DU MODÈLE DE RÉGRESSION LOGISTIQUE")
print("="*70)
print(result.summary())


# ====================================================================
# ÉTAPE 4 : ANALYSE ET PRÉDICTION FINALE
# ====================================================================

# --- A. Analyse des Odds Ratios (KPIs) ---

# Calcul des Odds Ratios (OR) pour l'analyse décisionnelle
odds_ratios = pd.DataFrame({
    'Odds Ratio (OR)': np.exp(result.params).round(3),
    'P-Value': result.pvalues.round(4)
})
odds_ratios = odds_ratios.drop('const', errors='ignore')
odds_ratios['Significatif'] = odds_ratios['P-Value'] < 0.05
profil_ideal = odds_ratios[
    (odds_ratios['Significatif'] == True) & 
    (odds_ratios['Odds Ratio (OR)'] > 1)
].sort_values(by='Odds Ratio (OR)', ascending=False)

print("\n" + "="*70)
print("TABLEAU 1 : ODDS RATIOS DU PROFIL IDÉAL (KPIs MAJEURS)")
print("Ces facteurs ont un Odds Ratio > 1 et sont statistiquement significatifs.")
print("="*70)
print(profil_ideal.head(10).to_markdown())


# --- B. Fonction de Prédiction (Interface simple) ---

def predire_probabilite_top10(modele_resultat, caracteristiques_du_profil, model_features):
    """Calcule la probabilité qu'un anime avec un profil donné atteigne le Top 10%."""
    
    # 1. Créer le vecteur d'entrée X_pred
    X_pred = pd.Series(0, index=model_features)
    X_pred['const'] = 1 

    # 2. Activer les caractéristiques du profil (X=1)
    for feature in caracteristiques_du_profil:
        if feature in X_pred.index:
            X_pred[feature] = 1
        else:
            print(f"[Avertissement] : La caractéristique '{feature}' n'a pas été trouvée dans le modèle.")

    # 3. Calcul de la probabilité (Fonction Sigmoïde)
    z = (X_pred * modele_resultat.params).sum()
    probabilite = 1 / (1 + np.exp(-z))
    
    return probabilite

# --- EXEMPLE DE TEST (Entrée Utilisateur) ---

# Utilisez les noms des colonnes encodées trouvées dans 'profil_ideal' pour tester l'impact
profil_anime_invente = [
    'Genre_Sci-Fi', 
    'Genre_Action', 
    'type_Movie', 
    'first_studio_MAPPA', 
    'season_aired_Spring', 
    'rating_R - 17+ (violence & profanity)'
]

prob_finale = predire_probabilite_top10(result, profil_anime_invente, model_features)

print("\n" + "="*50)
print("RÉSULTAT DE LA PRÉDICTION POUR VOTRE PROFIL")
print("="*50)
print(f"Caractéristiques testées : {profil_anime_invente}")
print(f"Probabilité que cet anime atteigne le Top 10% : {prob_finale * 100:.2f}%")
# --- EXEMPLE D'UTILISATION (Tester le Profil Optimisé) ---

profil_optimise = [
    'Genre_Drama', 
    'Genre_Psychological', 
    'type_TV', 
    'season_aired_Fall', 
    'first_studio_Madhouse',
    # Optionnel : Ajoutez ici le rating qui a le meilleur OR dans votre modèle.
]

# Assurez-vous d'avoir exécuté la partie entraînement (Étape 3) avant d'appeler cette fonction
# et que 'result' et 'model_features' sont disponibles.

prob_optimisee = predire_probabilite_top10(result, profil_optimise, model_features)

print("\n" + "="*50)
print("PRÉDICTION POUR LE PROFIL OPTIMISÉ (Haute Probabilité)")
print("="*50)
print(f"Caractéristiques testées : {profil_optimise}")
print(f"Probabilité que cet anime atteigne le Top 10% : {prob_optimisee * 100:.2f}%")