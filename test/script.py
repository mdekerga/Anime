import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION
# ==========================================

def charger_et_preparer_donnees(chemin_csv):
    try:
        try:
            df = pd.read_csv(chemin_csv)
        except FileNotFoundError:
            print(" Fichier introuvable.")

        # Nettoyage
        def extract_season(premiered_str):
            if isinstance(premiered_str, str) and premiered_str != 'UNKNOWN':
                return premiered_str.split(' ')[0].capitalize()
            return 'Unknown'

        df['season_cleaned'] = df['Premiered'].apply(extract_season)
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        
        cols_text = ['Studios', 'Type', 'Source', 'Rating', 'Genres']
        for col in cols_text:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = "Unknown"
            
        return df.dropna(subset=['Score'])
    except Exception as e:
        messagebox.showerror("Erreur Fatale", f"Impossible de traiter les données : {e}")
        return None

# ==========================================
# 2. ENTRAÎNEMENT DU MODÈLE
# ==========================================

def entrainer_modele(df):  
    base_score = df['Score'].mean()
    
    def get_adjustments(column_name, min_count=2):
        means = df.groupby(column_name)['Score'].mean()
        counts = df[column_name].value_counts()
        valid_cats = counts[counts >= min_count].index
        
        adjustments = {}
        for cat in means.index:
            if cat in valid_cats:
                adjustments[cat] = means[cat] - base_score
            else:
                adjustments[cat] = 0.0
        return adjustments

    model = {
        'base_score': base_score,
        'adj_studio': get_adjustments('Studios'),
        'adj_season': get_adjustments('season_cleaned'),
        'adj_type': get_adjustments('Type'),
        'adj_source': get_adjustments('Source'),
        'adj_rating': get_adjustments('Rating'),
        'adj_genre': {}
    }
    
    # Gestion des genres (FILTRE Award Winning + Unknown)
    all_genres = set()
    for g_str in df['Genres']:
        all_genres.update([g.strip() for g in g_str.split(',')])

    genres_a_ignorer = ["Award Winning", "Unknown"]

    for genre in all_genres:
        if genre in genres_a_ignorer:
            continue
        mask = df['Genres'].str.contains(genre, regex=False)
        if mask.sum() >= 1: 
            model['adj_genre'][genre] = df[mask]['Score'].mean() - base_score
            
    return model

# ==========================================
# 3. MOTEUR DE PRÉDICTION
# ==========================================

def predire_note(model, studio, source, type_anime, rating, genre, saison):
    score = model['base_score']
    details = []
    
    adjustments = [
        (model['adj_studio'].get(studio, 0), f"Studio: {studio}"),
        (model['adj_source'].get(source, 0), f"Source: {source}"),
        (model['adj_type'].get(type_anime, 0), f"Format: {type_anime}"),
        (model['adj_rating'].get(rating, 0), f"Rating: {rating}"),
        (model['adj_season'].get(saison, 0), f"Saison: {saison}"),
        (model['adj_genre'].get(genre, 0), f"Genre: {genre}")
    ]

    for val, label in adjustments:
        score += val
        signe = "+" if val >= 0 else ""
        details.append(f"{label} : {signe}{val:.2f}")
        
    final_score = max(1.0, min(10.0, score))
    return final_score, details

# ==========================================
# 4. DASHBOARD (Visualisation filtrée)
# ==========================================

def afficher_dashboard(model, df):
    # Calcul des prédictions
    def predict_row(row):
        g = row['Genres'].split(',')[0].strip()
        if g == "Award Winning" and "," in row['Genres']:
             g = row['Genres'].split(',')[1].strip()
        p, _ = predire_note(model, row['Studios'], row['Source'], row['Type'], 
                            row['Rating'], g, row['season_cleaned'])
        return p

    df_viz = df.copy()
    df_viz['prediction'] = df_viz.apply(predict_row, axis=1)

    # Style Matplotlib Clair
    plt.style.use('default') 
    plt.figure(figsize=(14, 10))
    plt.suptitle("Tableau de Bord - Analyse des Facteurs de Succès", fontsize=16)

    # 1. Saisons
    plt.subplot(2, 2, 1)
    saisons_data = {k: v for k, v in model['adj_season'].items() if k != 'Unknown'}
    saisons = list(saisons_data.keys())
    valeurs = list(saisons_data.values())
    if saisons:
        sns.barplot(x=saisons, y=valeurs, hue=saisons, palette="coolwarm", legend=False)
    plt.title("Impact de la Saison")
    plt.ylabel("Bonus/Malus")
    plt.axhline(0, color='black', linewidth=0.8)

    # 2. Formats
    plt.subplot(2, 2, 2)
    types_data = {k: v for k, v in model['adj_type'].items() if k != 'Unknown'}
    types = list(types_data.keys())
    valeurs_type = list(types_data.values())
    if types:
        sns.barplot(x=types, y=valeurs_type, hue=types, palette="viridis", legend=False)
    plt.title("Impact du Format")
    plt.axhline(0, color='black', linewidth=0.8)

    # 3. Top Genres
    sorted_genres = sorted(model['adj_genre'].items(), key=lambda x: x[1], reverse=True)
    top_genres = dict(sorted_genres[:5])
    plt.subplot(2, 2, 3)
    if top_genres:
        sns.barplot(x=list(top_genres.values()), y=list(top_genres.keys()), hue=list(top_genres.keys()), palette="Greens_r", legend=False)
    plt.title("Top 5 Genres (Bonus)")

    # 4. Précision
    plt.subplot(2, 2, 4)
    plt.scatter(df_viz['Score'], df_viz['prediction'], alpha=0.6, color='purple')
    if not df_viz.empty:
        min_val = min(df_viz['Score'].min(), df_viz['prediction'].min())
        plt.plot([min_val, 10], [min_val, 10], color='red', linestyle='--', label="Idéal")
    plt.xlabel("Note Réelle")
    plt.ylabel("Note Prédite")
    plt.title("Précision du Modèle")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. INTERFACE GRAPHIQUE 
# ==========================================

class AnimePredictorApp:
    def __init__(self, root, model, df):
        self.model = model
        self.df = df
        self.root = root
        self.root.title("🔮 Anime Predictor - Version Claire")
        self.root.geometry("650x750")
        
        # --- COULEURS CLAIRES (Lisibilité maximale) ---
        BG_COLOR = "#f5f6fa"       # Fond très clair (quasi blanc)
        FG_COLOR = "#2c3e50"       # Texte gris foncé
        ACCENT_COLOR = "#2980b9"   # Bleu professionnel
        RESULT_BG = "#ffffff"      # Blanc pur pour les résultats

        self.root.configure(bg=BG_COLOR)

        # --- STYLE ---
        style = ttk.Style()
        style.theme_use('clam') # Theme stable
        
        # Configuration générique
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), foreground=ACCENT_COLOR)
        
        # Boutons
        style.configure("TButton", font=("Segoe UI", 11, "bold"), background=ACCENT_COLOR, foreground="white", borderwidth=0)
        style.map("TButton", background=[('active', '#3498db')])
        
        # Listes déroulantes (Combobox) - Style standard système pour éviter les bugs
        style.configure("TCombobox", fieldbackground="white", background="white", foreground="black")

        # --- CONTENU ---
        header = ttk.Label(root, text="ANIME SUCCESS PREDICTOR", style="Header.TLabel")
        header.pack(pady=25)

        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.studios = sorted([k for k in model['adj_studio'].keys()])
        self.sources = sorted([k for k in model['adj_source'].keys()])
        self.types = sorted([k for k in model['adj_type'].keys()])
        self.ratings = sorted([k for k in model['adj_rating'].keys()])
        self.genres = sorted([k for k in model['adj_genre'].keys()])
        self.saisons = ["Winter", "Spring", "Summer", "Fall"]

        self.vars = {}
        row = 0
        self.create_dropdown(main_frame, "🎬 Studio d'Animation", self.studios, "studio", row); row+=1
        self.create_dropdown(main_frame, "📖 Source Originale", self.sources, "source", row); row+=1
        self.create_dropdown(main_frame, "📺 Format", self.types, "type", row); row+=1
        self.create_dropdown(main_frame, "🎭 Genre Principal", self.genres, "genre", row); row+=1
        self.create_dropdown(main_frame, "🌤️ Saison", self.saisons, "saison", row); row+=1
        self.create_dropdown(main_frame, "🔞 Classification", self.ratings, "rating", row); row+=1

        btn_frame = ttk.Frame(root, padding="10")
        btn_frame.pack(fill=tk.X)
        
        predict_btn = ttk.Button(btn_frame, text="LANCER LA SIMULATION", command=self.lancer_calcul)
        predict_btn.pack(fill=tk.X, pady=(10, 5), padx=40, ipady=5)

        viz_btn = ttk.Button(btn_frame, text="📊 VOIR LES STATISTIQUES", command=self.ouvrir_viz)
        viz_btn.pack(fill=tk.X, pady=5, padx=40)

        # Zone Résultat
        self.result_frame = ttk.LabelFrame(root, text=" Analyse ", padding="15")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Style spécifique pour le cadre de résultat
        style.configure("TLabelframe", background=BG_COLOR, bordercolor="#ccc")
        style.configure("TLabelframe.Label", background=BG_COLOR, foreground="#555")
        
        self.score_label = ttk.Label(self.result_frame, text="-- / 10", font=("Segoe UI", 28, "bold"), foreground="#7f8c8d")
        self.score_label.pack(pady=5)
        
        # Zone de texte standard (noir sur blanc)
        self.details_text = tk.Text(self.result_frame, height=8, bg=RESULT_BG, fg="black", relief="flat", font=("Consolas", 10))
        self.details_text.pack(fill=tk.BOTH, expand=True)

    def create_dropdown(self, parent, label_text, values, var_name, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=8)
        var = tk.StringVar()
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=32)
        combo.grid(row=row, column=1, sticky="e", pady=8, padx=10)
        if values: combo.current(0)
        else: combo.set("Unknown")
        self.vars[var_name] = var

    def lancer_calcul(self):
        inputs = {k: v.get() for k, v in self.vars.items()}
        note, details = predire_note(self.model, inputs['studio'], inputs['source'], inputs['type'], 
                                     inputs['rating'], inputs['genre'], inputs['saison'])
        
        # Couleurs adaptées au fond blanc
        color = "#c0392b" if note < 6.5 else "#f39c12" if note < 8 else "#27ae60"
        self.score_label.config(text=f"{note:.2f} / 10", foreground=color)
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, f"BASE SCORE : {self.model['base_score']:.2f}\n" + "─"*30 + "\n")
        for line in details:
            self.details_text.insert(tk.END, f" {line}\n")

    def ouvrir_viz(self):
        afficher_dashboard(self.model, self.df)

# ==========================================
# 6. EXÉCUTION
# ==========================================
if __name__ == "__main__":
    df = charger_et_preparer_donnees('anime-dataset-2023.csv') 
    
    if df is not None:
        modele = entrainer_modele(df)
        root = tk.Tk()
        app = AnimePredictorApp(root, modele, df)
        root.mainloop()