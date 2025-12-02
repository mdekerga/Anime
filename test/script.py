import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CHARGEMENT ET PRÉPARATION ---
# Assurez-vous que le fichier est bien dans le dossier
df = pd.read_csv('anime-dataset-2023.csv')

# Extraction de la saison depuis la colonne 'Premiered'
def extract_season(premiered_str):
    if isinstance(premiered_str, str) and premiered_str != 'UNKNOWN':
        return premiered_str.split(' ')[0].capitalize()
    return 'Unknown'

df['season_cleaned'] = df['Premiered'].apply(extract_season)

df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# Nettoyage des chaînes de caractères
df['Studios'] = df['Studios'].astype(str).str.strip()
df['Type'] = df['Type'].astype(str).str.strip()
df['Source'] = df['Source'].astype(str).str.strip()
# AJOUT : Nettoyage de la colonne Rating
df['Rating'] = df['Rating'].astype(str).str.strip()

# Calcul de la moyenne globale
base_score = df['Score'].mean()

# --- 2. CRÉATION DU MODÈLE ---

def get_adjustments(column_name):
    means = df.groupby(column_name)['Score'].mean()
    # On garde seulement les catégories avec au moins 2 animes pour éviter les biais
    counts = df[column_name].value_counts()
    valid_cats = counts[counts >= 2].index
    
    adjustments = {}
    for cat in means.index:
        if cat in valid_cats:
            adjustments[cat] = means[cat] - base_score
        else:
            adjustments[cat] = 0.0
    return adjustments

# Calcul des ajustements
adj_studio = get_adjustments('Studios')
adj_season = get_adjustments('season_cleaned')
adj_type = get_adjustments('Type')
adj_source = get_adjustments('Source')
# AJOUT : Calcul des ajustements pour le Rating
adj_rating = get_adjustments('Rating')

# Calcul des ajustements pour les Genres
all_genres = set()
for g_str in df['Genres']:
    if isinstance(g_str, str):
        all_genres.update([g.strip() for g in g_str.split(',')])

adj_genre = {}
for genre in all_genres:
    mask = df['Genres'].astype(str).apply(lambda x: genre in x)
    if mask.sum() >= 3: # Au moins 3 animes pour valider le genre
        adj_genre[genre] = df[mask]['Score'].mean() - base_score
    else:
        adj_genre[genre] = 0.0

# --- 3. TEST ET ÉVALUATION ---

def predict_row(row):
    pred = base_score
    pred += adj_studio.get(row['Studios'], 0)
    pred += adj_season.get(row['season_cleaned'], 0)
    pred += adj_type.get(row['Type'], 0)
    pred += adj_source.get(row['Source'], 0)
    # AJOUT : Prise en compte du Rating
    pred += adj_rating.get(row['Rating'], 0)
    
    genres_list = [g.strip() for g in str(row['Genres']).split(',')]
    valid_adjs = [adj_genre[g] for g in genres_list if g in adj_genre]
    if valid_adjs:
        pred += sum(valid_adjs) / len(valid_adjs)
        
    return max(1.0, min(10.0, pred))

df['prediction'] = df.apply(predict_row, axis=1)
df['erreur_absolue'] = abs(df['Score'] - df['prediction'])
mae = df['erreur_absolue'].mean()

print(f"--- RÉSULTATS DU MODÈLE ---")
print(f"Note moyenne globale : {base_score:.2f}")
print(f"Erreur Moyenne Absolue (MAE) : {mae:.3f} points")

# --- 4. VISUALISATION CORRIGÉE (Avec Rating) ---

plt.figure(figsize=(18, 12))

# Graphique 1 : Impact de la Source
plt.subplot(2, 3, 1)
src_filtered = {k: v for k, v in adj_source.items() if v != 0}
x_vals = list(src_filtered.keys())
y_vals = list(src_filtered.values())
sns.barplot(x=x_vals, y=y_vals, hue=x_vals, palette="magma", dodge=False)
plt.legend([],[], frameon=False)
plt.title("Impact de la Source")
plt.xticks(rotation=45)
plt.axhline(0, color='black', linewidth=0.8)

# Graphique 2 : Impact du Rating (NOUVEAU)
plt.subplot(2, 3, 2)
rating_filtered = {k: v for k, v in adj_rating.items() if v != 0}
x_vals = list(rating_filtered.keys())
y_vals = list(rating_filtered.values())
# Raccourcir les labels pour l'affichage (ex: "R - 17+..." -> "R - 17+")
x_labels_short = [x.split(' - ')[0] if '-' in x else x for x in x_vals]
sns.barplot(x=x_vals, y=y_vals, hue=x_vals, palette="plasma", dodge=False)
plt.legend([],[], frameon=False)
plt.title("Impact de la Classification (Rating)")
plt.xticks(ticks=range(len(x_vals)), labels=x_labels_short, rotation=45)
plt.axhline(0, color='black', linewidth=0.8)

# Graphique 3 : Impact de la Saison
plt.subplot(2, 3, 3)
seas_filtered = {k: v for k, v in adj_season.items() if k != 'Unknown' and v != 0}
x_vals = list(seas_filtered.keys())
y_vals = list(seas_filtered.values())
sns.barplot(x=x_vals, y=y_vals, hue=x_vals, palette="coolwarm", dodge=False)
plt.legend([],[], frameon=False)
plt.title("Impact de la Saison")
plt.axhline(0, color='black', linewidth=0.8)

# Graphique 4 : Top 5 Genres
sorted_genres = sorted(adj_genre.items(), key=lambda x: x[1], reverse=True)
top_genres = dict(sorted_genres[:5])
x_vals = list(top_genres.values())
y_vals = list(top_genres.keys())
plt.subplot(2, 3, 4)
sns.barplot(x=x_vals, y=y_vals, hue=y_vals, palette="Greens_r", dodge=False)
plt.legend([],[], frameon=False)
plt.title("Top 5 Genres (Bonus)")

# Graphique 5 : Impact du Type
plt.subplot(2, 3, 5)
type_filtered = {k: v for k, v in adj_type.items() if v != 0}
x_vals = list(type_filtered.keys())
y_vals = list(type_filtered.values())
sns.barplot(x=x_vals, y=y_vals, hue=x_vals, palette="viridis", dodge=False)
plt.legend([],[], frameon=False)
plt.title("Impact du Format (Type)")
plt.axhline(0, color='black', linewidth=0.8)

# Graphique 6 : Précision
plt.subplot(2, 3, 6)
plt.scatter(df['Score'], df['prediction'], alpha=0.5, color='purple')
plt.plot([min(df['Score']), 10], [min(df['Score']), 10], color='red', linestyle='--', label="Parfait")
plt.xlabel("Note Réelle")
plt.ylabel("Note Prédite")
plt.title(f"Précision (MAE: {mae:.2f})")
plt.legend()

plt.tight_layout()
plt.show()

# --- 5. SIMULATION AVEC RATING ---
print("\n--- SIMULATION ---")
# Exemple : Anime 'Original' par 'Bones' en 'TV', Rated R
test_studio = "Bones"
test_source = "Manga"
test_type = "TV"
test_genre = "Sci-Fi"
test_saison = "Winter"
test_rating = "R - 17+ (violence & profanity)" # Exemple de rating

score_simu = base_score
score_simu += adj_studio.get(test_studio, 0)
score_simu += adj_source.get(test_source, 0)
score_simu += adj_type.get(test_type, 0)
score_simu += adj_season.get(test_saison, 0)
score_simu += adj_rating.get(test_rating, 0) # Ajout
score_simu += adj_genre.get(test_genre, 0)

print(f"Scénario : {test_studio} | {test_source} | {test_type} | {test_rating}")
print(f"Note prédite : {score_simu:.2f}/10")