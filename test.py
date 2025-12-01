# ====================================================================
# ÉTAPE 0 : IMPORTS
# ====================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')

# ====================================================================
# ÉTAPE 1 : NETTOYAGE ET PRÉPARATION DES DONNÉES
# ====================================================================

# NOTE IMPORTANTE : Ajustez ce chemin de fichier
try:
    df = pd.read_csv("C:/Users/babou/Downloads/AnimeProject/Anime/popular_anime.csv")
except FileNotFoundError:
    print("ERREUR FATALE: Fichier non trouvé. Veuillez vérifier et ajuster le chemin du fichier CSV.")
    exit()

# [Clean Code - Identical to previous steps]
df["genres"].fillna("Other", inplace=True)
df["episodes"].fillna(1, inplace=True)
colonne_a_drop = ['synopsis','trailer','producers']
df = df.drop(columns=colonne_a_drop, errors='ignore')
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['scored_by'] = pd.to_numeric(df['scored_by'], errors='coerce')
df = df.dropna(subset=['score', 'episodes', 'scored_by', 'rank',"rating","studios"])
df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)
df = df[df['score'] <= 10.0].copy()

def get_season_anime(date):
    if pd.isna(date): return np.nan
    try: date = pd.to_datetime(date)
    except ValueError: return np.nan
    month = date.month
    season_index = (month - 1) // 3  
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    return seasons[season_index]
df["season_aired"] = df["aired_from"].apply(get_season_anime)
df.dropna(subset=['season_aired'], inplace=True)

# [Calcul du Seuil Top 10%]
anime_classe_simple = df.sort_values(by='score', ascending=False)
top_10_count = int(len(anime_classe_simple) * 0.10)
SEUIL_SCORE_TOP_10 = anime_classe_simple.iloc[top_10_count-1]['score']

# ====================================================================
# ÉTAPE 2 : ENCODAGE ET PRÉPARATION DES MATRICES X ET Y
# ====================================================================

# 1. Encodage Genres (Multi-étiquettes)
df['genres_list'] = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(df['genres_list']),
                             columns=['Genre_' + genre for genre in mlb.classes_],
                             index=df.index)
df_encoded = pd.concat([df.reset_index(drop=True), genre_dummies], axis=1)

# 2. Encodage variables simples
df_encoded['first_studio'] = df_encoded['studios'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else np.nan)
df_encoded.dropna(subset=['first_studio', 'season_aired'], inplace=True) 

cols_to_encode = ['type', 'rating', 'season_aired', 'first_studio']

df_encoded = pd.get_dummies(
    df_encoded, 
    columns=cols_to_encode, 
    prefix=cols_to_encode, 
    drop_first=True 
)

# 3. Définition et Nettoyage Final des Matrices X et Y
Y = df_encoded['score']
X = df_encoded.select_dtypes(include=[np.number])
cols_a_exclure = ['score', 'scored_by', 'rank', 'episodes'] 
X = X.drop(columns=cols_a_exclure, errors='ignore')

data_final = pd.concat([X, Y], axis=1)
data_final.dropna(inplace=True)

X = data_final.drop(columns=['score'], errors='ignore') 
Y = data_final['score']

X = X.loc[:, X.nunique() > 1] 
X = sm.add_constant(X, prepend=False)

# ====================================================================
# ÉTAPE 3 : MODÉLISATION ET ENTRAÎNEMENT (OLS)
# ====================================================================

linear_model = sm.OLS(Y, X)
result = linear_model.fit()
MODEL_FEATURES = X.columns # Noms des colonnes finales du modèle
COEFFICIENTS = result.params # Coefficients Beta pour l'analyse

print("\n" + "="*70)
print("MODÈLE ENTRAÎNÉ AVEC SUCCÈS")
print(f"R-carré du modèle : {result.rsquared:.4f}")
print("="*70)

# ====================================================================
# ÉTAPE 4 : FONCTION DE PRÉDICTION ET EXPLICATION
# ====================================================================

def predire_et_expliquer(modele_resultat, caracteristiques_du_profil, model_features, seuil_top_10):
    """Prédit le score, détermine le Top 10% et explique la note."""
    
    X_pred = pd.Series(0, index=model_features)
    X_pred['const'] = 1 
    
    contributions = {}
    
    # Activer les caractéristiques du profil (X=1) et enregistrer les contributions
    for feature in caracteristiques_du_profil:
        if feature in X_pred.index:
            X_pred[feature] = 1
            # Contribution = Coefficient * 1 (car X=1)
            contributions[feature] = modele_resultat.params[feature]
        else:
            contributions[f"Avertissement: {feature}"] = 0
            
    # Ajouter la contribution de la constante (Note de base)
    contributions['Base Score (Intercept)'] = modele_resultat.params['const']
    
    # Calcul du score prédit (Score = sum(Beta_i * X_i))
    score_predit = (X_pred * modele_resultat.params).sum()
    score_predit = np.clip(score_predit, 1.0, 10.0) # Limite 0-10
    
    # Décision Top 10%
    est_top_10 = score_predit >= seuil_top_10
    decision = "**OUI**" if est_top_10 else "**NON**"
    
    return score_predit, decision, contributions

# ====================================================================
# ÉTAPE 5 : INTERFACE UTILISATEUR ET DÉMONSTRATION
# ====================================================================

# Liste des genres populaires (pour guider l'utilisateur)
genres_possibles = [col.replace('Genre_', '') for col in MODEL_FEATURES if col.startswith('Genre_')][:10]
studios_possibles = [col.replace('first_studio_', '') for col in MODEL_FEATURES if col.startswith('first_studio_')][:5]

print("\n\n" + "#"*70)
print("INTERFACE DE PRÉDICTION DE POPULARITÉ DES ANIMES")
print("#"*70)
print(f"Seuil du Top 10% (Score minimum requis) : {SEUIL_SCORE_TOP_10:.2f}")

# --- 1. Saisie des Caractéristiques ---

# Définition des options par défaut et saisie utilisateur (simulée)
print("\n--- Saisir le profil du nouvel anime ---")
print(f"Exemples de Genres (saisir les noms exacts) : {genres_possibles}")
print(f"Exemples de Studios : {studios_possibles}")

# Simulation d'une entrée utilisateur (pour cet exemple, nous utilisons un profil fixe)
# UTILISATEUR : Vous pouvez remplacer ceci par input() ou par une liste de test
profil_test_user = [
    'Genre_Thriller', 
    'Genre_Mystery', 
    'type_Movie', 
    'first_studio_Wit Studio', 
    'season_aired_Fall',
    'rating_R - 17+ (violence & profanity)'
]
print(f"\nProfil choisi pour la démonstration : {profil_test_user}")


# --- 2. Exécution du Modèle ---

score_predit, decision_top10, contributions = predire_et_expliquer(
    result, 
    profil_test_user, 
    MODEL_FEATURES, 
    SEUIL_SCORE_TOP_10
)


# --- 3. Affichage des Résultats et de l'Explication ---

print("\n" + "="*50)
print("RÉSULTAT DE LA PRÉDICTION")
print("="*50)
print(f"SCORE PRÉDIT (0-10) : **{score_predit:.2f}**")
print(f"RÉPONSE À LA PROBLÉMATIQUE (Top 10%) : {decision_top10}")
print("="*50)

print("\n### Explication du Score (Analyse des Contributions)")
print("La note prédite est la somme des contributions suivantes :")

# Afficher les contributions triées (Positives, Négatives, Base)
contributions_df = pd.Series(contributions).sort_values(ascending=False).to_frame(name='Contribution au Score')

# Mise en forme pour l'affichage
def format_contribution(value):
    if value > 0: return f"+{value:.4f}"
    return f"{value:.4f}"

contributions_df['Contribution au Score'] = contributions_df['Contribution au Score'].apply(format_contribution)

print(contributions_df.to_markdown())

# Conclusion de l'explication
print("\n**Interprétation :**")
print("Chaque ligne montre comment une caractéristique active (ou le score de base 'Intercept') tire la note vers le haut (+) ou vers le bas (-).")
print(f"Le score de {score_predit:.2f} est la somme de l'Intercept ({contributions.get('Base Score (Intercept)', 0):.2f}) et des contributions de chaque Genre/Studio, etc.")