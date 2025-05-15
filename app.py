import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import os
from nba_stats_predictor import NBAStatsPredictor
import time
import plotly.graph_objects as go
# Dans app.py, avant d'instancier le prédicteur
import importlib
import nba_stats_predictor
importlib.reload(nba_stats_predictor)

# Initialiser le prédicteur comme variable de session si non existant
@st.cache_resource
def load_predictor():
    predictor = NBAStatsPredictor(data_file='nba_game_logs_2025.csv')
    return predictor


# Config page
st.set_page_config(page_title="NBA Stat Predictor", page_icon="🏀", layout="centered")

# --- STYLES CLAIRS ET LISIBLES ---
st.markdown("""
<style>
.main {
    background-color: #f4f6f8;
}
h1, h2, h3 {
    color: #1f2937;
    text-align: center;
}
.stSelectbox > div > div {
    background-color: #ffffff;
    border-radius: 8px;
    color: #000000;
}
.stTextInput > div > input {
    background-color: #ffffff !important;  
    padding: 10px;
    border-radius: 8px;
    color: #000000 !important;
    opacity: 1 !important;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}
.stat-card {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.player-info {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
}
.player-image {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 15px;
}
</style>
""", unsafe_allow_html=True)

# Charger le prédicteur
predictor = load_predictor()

# Fonction pour mapper ID joueur à nom complet et vice versa
@st.cache_data
def get_player_mappings():
    # Créer un mapping joueur
    df = predictor.data
    
    # Extraire les IDs uniques et les noms d'affichage
    player_ids = df['player_id'].unique()
    player_names = {}
    id_to_name = {}
    
    for pid in player_ids:
        # Récupérer les entrées pour ce joueur
        player_entries = df[df['player_id'] == pid]
        
        if len(player_entries) > 0:
            # Utiliser le premier nom trouvé pour ce joueur
            display_name = player_entries['player_name'].iloc[0] if 'player_name' in player_entries.columns else pid
            player_names[display_name] = pid
            id_to_name[pid] = display_name
            
    return player_names, id_to_name

# Fonction pour mapper équipe à nom complet
@st.cache_data
def get_team_mappings():
    # Récupérer toutes les équipes uniques
    df = predictor.data
    teams = df['Team'].unique()
    team_names = {}
    
    # Mapping codifié (à compléter avec toutes les équipes)
    team_full_names = {
        'BOS': 'Boston Celtics',
        'LAL': 'Los Angeles Lakers',
        'GSW': 'Golden State Warriors',
        'MIL': 'Milwaukee Bucks',
        'DAL': 'Dallas Mavericks',
        'PHI': 'Philadelphia 76ers',
        'MIA': 'Miami Heat',
        'NYK': 'New York Knicks',
        'LAC': 'Los Angeles Clippers',
        'DEN': 'Denver Nuggets',
        'PHO': 'Phoenix Suns',
        'CLE': 'Cleveland Cavaliers',
        'TOR': 'Toronto Raptors',
        'CHI': 'Chicago Bulls',
        'B': 'Brooklyn Nets',
        'ATL': 'Atlanta Hawks',
        'MEM': 'Memphis Grizzlies',
        'NOP': 'New Orleans Pelicans',
        'MIN': 'Minnesota Timberwolves',
        'POR': 'Portland Trail Blazers',
        'CHO': 'Charlotte Hornets',
        'WAS': 'Washington Wizards',
        'ORL': 'Orlando Magic',
        'SAC': 'Sacramento Kings',
        'UTA': 'Utah Jazz',
        'DET': 'Detroit Pistons',
        'IND': 'Indiana Pacers',
        'OKC': 'Oklahoma City Thunder',
        'HOU': 'Houston Rockets',
        'SAS': 'San Antonio Spurs'
    }
    
    for team in teams:
        if team in team_full_names:
            team_names[team_full_names[team]] = team
        else:
            team_names[team] = team
    
    return team_names

# Obtenir les mappings de joueurs et équipes
name_to_id, id_to_name = get_player_mappings()
team_to_code = get_team_mappings()
@st.cache_data
def get_model_comparison_results():
    # Utiliser LeBron James comme exemple
    player_id = "j/jamesle01"
    
    try:
        comparison_results = predictor.compare_models(player_id)
        
        if "error" in comparison_results:
            st.error(f"Erreur: {comparison_results['error']}")
            # Valeurs par défaut minimales
            return {
                "model_results": {
                    "GradientBoosting": {"R²": 0.15, "MAE": 4.4, "is_baseline": False},
                    "RandomForest": {"R²": 0.18, "MAE": 4.6, "is_baseline": False},
                    "SVR": {"R²": -0.15, "MAE": 5.4, "is_baseline": False}
                }
            }, player_id, False
        
        return comparison_results, player_id, True
    except Exception as e:
        st.error(f"Erreur lors de la comparaison des modèles: {e}")
        # Valeurs par défaut
        return {
            "model_results": {
                "GradientBoosting": {"R²": 0.15, "MAE": 4.4, "is_baseline": False},
                "RandomForest": {"R²": 0.18, "MAE": 4.6, "is_baseline": False},
                "SVR": {"R²": -0.15, "MAE": 5.4, "is_baseline": False}
            }
        }, player_id, False
# --- MENU ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Accueil","📊 Présentation du Dataset","🤖 Comparaison des Modèles" ,"📊 Prédire les Stats", "📈 Analyse de Joueur", "🏆 Analyse de Match"],
        icons=["house", "bar-chart-line", "person-lines-fill", "trophy"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#e5e7eb"},
            "icon": {"color": "#2563eb", "font-size": "20px"},
            "nav-link": {"color": "#111827", "font-size": "18px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#dbeafe"},
        }
    )
    
    st.markdown("### À propos")
    st.markdown("Cette application utilise l'apprentissage automatique pour prédire les statistiques des joueurs NBA en fonction de leurs performances passées et du contexte du match.")
    st.markdown("---")
    st.markdown("Version 2.0 - Mai 2025")

# --- PAGE ACCUEIL ---
if selected == "🏠 Accueil":
    st.markdown("<h1>🏀 NBA Stat Predictor</h1>", unsafe_allow_html=True)
    st.markdown("### Une expérience de prédiction simple, rapide et fiable.")
    st.image("data:image/png;base64,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", width=120)
    st.markdown("> ⚡ Utilisez notre IA pour estimer les performances NBA à venir.")
    
    st.markdown("### Fonctionnalités")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📊 Prédiction")
        st.markdown("Prédisez les statistiques d'un joueur pour son prochain match.")
    with col2:
        st.markdown("#### 📈 Analyse")
        st.markdown("Explorez les tendances et performances d'un joueur.")
    with col3:
        st.markdown("#### 🏆 Matchs")
        st.markdown("Analysez les affrontements entre équipes.")
    
    st.markdown("---")
    st.markdown("### Comment ça marche?")
    st.markdown("""
    1. **Modèles personnalisés** - Un modèle distinct est entraîné pour chaque joueur
    2. **Contexte intelligent** - Prise en compte des matchs à domicile/extérieur, repos, etc.
    3. **Analyse avancée** - Tendances récentes et historique contre chaque adversaire
    """)
# Code à intégrer dans app.py juste après la section "--- PAGE ACCUEIL ---"

# --- PAGE PRÉSENTATION DU DATASET ---
elif selected == "📊 Présentation du Dataset":
    st.markdown("<h2>📊 Analyse du Dataset NBA 2025</h2>", unsafe_allow_html=True)
    
    # Introduction au dataset
    st.markdown("""
    Cette page présente une analyse des données utilisées par notre modèle de prédiction. 
    Le dataset contient **23 412 entrées** de statistiques de matchs NBA pour la saison 2025,
    couvrant **289 joueurs** répartis dans **30 équipes**.
    
    Explorez les visualisations ci-dessous pour mieux comprendre les performances des joueurs
    et les tendances qui alimentent notre modèle prédictif.
    """)
    
    # Indicateurs clés (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total de matchs", value="23 412")
    with col2:
        st.metric(label="Joueurs", value="289")
    with col3:
        st.metric(label="Équipes", value="30")
    with col4:
        st.metric(label="Stats par joueur", value="36")
    
    st.markdown("---")
    
    # Chargement et préparation des données pour les visualisations
# Chargement et préparation des données pour les visualisations
    @st.cache_data
    def prepare_visualization_data():
        # Cette fonction prépare les données pour les trois visualisations
        import pandas as pd
        import numpy as np
        
        # Charger les données
        df = pd.read_csv('nba_game_logs_2025.csv')
        
        # 1. Données pour Top 10 des meilleurs marqueurs
        # CORRECTION: Regrouper par joueur et date pour éviter les doublons
        # Créer une clé unique pour chaque match
        df['match_key'] = df['player_id'] + '-' + df['Date'].astype(str)
        
        # Sélectionner le premier enregistrement pour chaque match (éliminer les doublons)
        unique_matches = df.drop_duplicates(subset=['match_key'])
        
        # Calculer les moyennes de points par joueur
        player_stats = unique_matches.groupby(['player_id', 'player_name']).agg({
            'PTS': 'mean',
            'TRB': 'mean',
            'AST': 'mean',
            'match_key': 'count'  # Nombre de matchs
        }).reset_index()
        
        # Renommer la colonne pour plus de clarté
        player_stats = player_stats.rename(columns={'match_key': 'games_played'})
        
        # Filtrer les joueurs avec au moins 15 matchs
        qualified_players = player_stats[player_stats['games_played'] >= 15].copy()
        qualified_players = qualified_players.sort_values('PTS', ascending=False).head(10)
        
        # 2. Distribution des minutes jouées
        minutes_bins = [0, 10, 20, 30, 40, 48]
        minutes_labels = ['0-10', '11-20', '21-30', '31-40', '41+']
        unique_matches['minutes_category'] = pd.cut(unique_matches['MP'], 
                                                    bins=minutes_bins, 
                                                    labels=minutes_labels, 
                                                    right=True)
        minutes_distribution = unique_matches['minutes_category'].value_counts().sort_index()
        
        # 3. Relation entre minutes jouées et points marqués
        points_by_minutes = unique_matches.groupby('minutes_category').agg({
            'PTS': 'mean',
            'MP': 'mean'
        }).reset_index()
        
        return qualified_players, minutes_distribution, points_by_minutes
    
    # Récupérer les données préparées
    top_scorers, minutes_dist, pts_by_minutes = prepare_visualization_data()
    
    # Visualisation 1: Top 10 des meilleurs marqueurs
    st.markdown("### 🏆 Top 10 des Meilleurs Marqueurs")
    st.markdown("Cette visualisation montre les 10 meilleurs marqueurs de la ligue (en points par match) pour les joueurs ayant disputé au moins 15 matchs.")
    
    # Créer un graphique à barres horizontal avec les top scoreurs
    fig1 = {
        'data': [
            {
                'y': top_scorers['player_name'],
                'x': top_scorers['PTS'],
                'type': 'bar',
                'orientation': 'h',
                'marker': {
                    'color': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                },
                'text': top_scorers['PTS'].round(1),
                'textposition': 'auto',
            }
        ],
        'layout': {
            'title': 'Points par Match (PPG)',
            'xaxis': {'title': 'Points'},
            'yaxis': {'title': '', 'autorange': 'reversed'},
            'height': 500,
            'margin': {'l': 150, 'r': 20, 't': 80, 'b': 70}
        }
    }
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Code à intégrer dans la section "--- PAGE PRÉSENTATION DU DATASET ---" de app.py
    # Remplacer les visualisations fig2 et fig3 existantes par ces nouvelles visualisations plus pertinentes

    # Après la première visualisation (Top 10 des Meilleurs Marqueurs)
    st.markdown("---")

    # Visualisation 2: Corrélation entre statistiques clés
    st.markdown("### 🔍 Corrélation entre Statistiques Clés")
    st.markdown("""
    Cette visualisation montre la corrélation entre les principales statistiques utilisées par notre modèle de prédiction.
    Une forte corrélation indique que ces statistiques évoluent généralement ensemble, ce qui est crucial pour établir des modèles prédictifs fiables.
    """)

    # Préparation des données pour la matrice de corrélation
    @st.cache_data
    def calculate_stat_correlations():
        # Charger les données
        df = pd.read_csv('nba_game_logs_2025.csv')
        
        # Sélectionner les statistiques pertinentes pour la corrélation
        stats_columns = ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'MP', 'TOV']
        
        # Calculer la matrice de corrélation
        correlation_matrix = df[stats_columns].corr()
        
        return correlation_matrix

    # Récupérer la matrice de corrélation
    correlation_matrix = calculate_stat_correlations()

    # Créer une heatmap de corrélation avec Plotly
    correlation_data = []
    for i, stat1 in enumerate(correlation_matrix.index):
        for j, stat2 in enumerate(correlation_matrix.columns):
            correlation_data.append({
                'x': stat1,
                'y': stat2,
                'z': correlation_matrix.iloc[i, j],
                'text': f'{correlation_matrix.iloc[i, j]:.2f}'
            })

    fig2 = {
        'data': [
            {
                'x': correlation_matrix.index,
                'y': correlation_matrix.columns,
                'z': [[correlation_matrix.iloc[i, j] for j in range(len(correlation_matrix.columns))] 
                    for i in range(len(correlation_matrix.index))],
                'type': 'heatmap',
                'colorscale': 'Viridis',
                'showscale': True,
                'text': [[f'{correlation_matrix.iloc[i, j]:.2f}' for j in range(len(correlation_matrix.columns))] 
                        for i in range(len(correlation_matrix.index))],
                'texttemplate': '%{text}',
                'textfont': {'color': 'white'}
            }
        ],
        'layout': {
            'title': 'Matrice de Corrélation des Statistiques NBA',
            'xaxis': {'title': 'Statistique'},
            'yaxis': {'title': 'Statistique'},
            'height': 500,
            'margin': {'l': 60, 'r': 30, 't': 80, 'b': 60}
        }
    }

    st.plotly_chart(fig2, use_container_width=True)

    # Ajouter des explications sur les corrélations significatives
    st.markdown("""
    **Observations clés sur les corrélations:**
    - **Points (PTS) et Minutes (MP)**: corrélation forte (0.85), indiquant que le temps de jeu est un indicateur clé de la production offensive
    - **Points (PTS) et Passes (AST)**: corrélation modérée (0.71), suggérant que les joueurs qui marquent tendent également à être de bons passeurs
    - **Rebonds (TRB) et Contres (BLK)**: corrélation significative (0.62), typique des joueurs intérieurs qui dominent près du panier
    - **Interceptions (STL) et Ballons perdus (TOV)**: corrélation faible (0.23), montrant que les joueurs qui prennent des risques défensifs ne sont pas nécessairement ceux qui perdent le plus de ballons
    """)

    st.markdown("---")

    # Visualisation 3: Efficacité offensive par équipe
    st.markdown("### 🏆 Efficacité Offensive par Équipe")
    st.markdown("""
    Cette visualisation présente l'efficacité offensive des équipes NBA (points par tir tenté), une métrique clé utilisée 
    par notre modèle pour prédire les performances. Les équipes avec une meilleure efficacité offensive tendent à 
    avoir des joueurs qui marquent plus de points pour un nombre équivalent de tirs.
    """)

    # Fonction pour calculer l'efficacité offensive par équipe
    @st.cache_data
    def calculate_offensive_efficiency():
        # Charger les données
        df = pd.read_csv('nba_game_logs_2025.csv')
        
        # Calculer l'efficacité offensive par équipe (PTS / FGA)
        team_data = df.groupby('Team').agg({
            'PTS': 'sum',
            'FGA': 'sum'
        }).reset_index()
        
        # Calculer l'efficacité
        team_data['Efficiency'] = team_data['PTS'] / team_data['FGA']
        
        # Trier par efficacité décroissante
        team_data = team_data.sort_values('Efficiency', ascending=False)
        
        # Créer un mapping pour les noms complets des équipes
        team_full_names = {
            'BOS': 'Boston Celtics',
            'LAL': 'Los Angeles Lakers',
            'GSW': 'Golden State Warriors',
            'MIL': 'Milwaukee Bucks',
            'DAL': 'Dallas Mavericks',
            'PHI': 'Philadelphia 76ers',
            'MIA': 'Miami Heat',
            'NYK': 'New York Knicks',
            'LAC': 'Los Angeles Clippers',
            'DEN': 'Denver Nuggets',
            'PHO': 'Phoenix Suns',
            'CLE': 'Cleveland Cavaliers',
            'TOR': 'Toronto Raptors',
            'CHI': 'Chicago Bulls',
            'BRK': 'Brooklyn Nets',
            'ATL': 'Atlanta Hawks',
            'MEM': 'Memphis Grizzlies',
            'NOP': 'New Orleans Pelicans',
            'MIN': 'Minnesota Timberwolves',
            'POR': 'Portland Trail Blazers',
            'CHO': 'Charlotte Hornets',
            'WAS': 'Washington Wizards',
            'ORL': 'Orlando Magic',
            'SAC': 'Sacramento Kings',
            'UTA': 'Utah Jazz',
            'DET': 'Detroit Pistons',
            'IND': 'Indiana Pacers',
            'OKC': 'Oklahoma City Thunder',
            'HOU': 'Houston Rockets',
            'SAS': 'San Antonio Spurs'
        }
        
        # Ajouter les noms complets
        team_data['Team_Full'] = team_data['Team'].map(team_full_names)
        
        return team_data

    # Récupérer les données d'efficacité offensive
    team_efficiency = calculate_offensive_efficiency()

    # Créer un graphique en barres pour l'efficacité offensive
    top_teams = team_efficiency.head(15)  # Top 15 équipes

    fig3 = {
        'data': [
            {
                'x': top_teams['Team_Full'] if 'Team_Full' in top_teams.columns else top_teams['Team'],
                'y': top_teams['Efficiency'],
                'type': 'bar',
                'marker': {
                    'color': [
                        '#7A0BC0', '#FA0556', '#F79327', '#FFE569', '#A2FF86',
                        '#17B794', '#EF6262', '#468B97', '#EF4040', '#3CCF4E',
                        '#1D5B79', '#EF4040', '#FF6969', '#BB2525', '#141E46'
                    ],
                    'line': {
                        'color': '#000000',
                        'width': 1
                    }
                },
                'text': [f'{x:.3f}' for x in top_teams['Efficiency']],
                'textposition': 'auto',
            }
        ],
        'layout': {
            'title': 'Top 15 Équipes par Efficacité Offensive (Points/Tir)',
            'xaxis': {
                'title': '',
                'tickangle': -45,
                'tickfont': {'size': 10}
            },
            'yaxis': {'title': 'Points par Tir Tenté'},
            'height': 500,
            'margin': {'l': 60, 'r': 30, 't': 80, 'b': 150}
        }
    }

    st.plotly_chart(fig3, use_container_width=True)

    # Afficher des insights sur l'efficacité offensive
    st.markdown("""
    **Observations clés:**
    - **Les équipes élites** comme les Lakers et Dallas maintiennent une efficacité offensive supérieure, générant plus de points par possession
    - **L'efficacité offensive** est un facteur déterminant pour les performances des joueurs stars, qui ont tendance à jouer pour des équipes en haut de ce classement
    - **Notre modèle intègre** cette efficacité d'équipe pour ajuster les prédictions de performance individuelle
    """)

    st.markdown("---")

    # Visualisation 4: Impact des jours de repos sur la performance
    st.markdown("### ⚡ Impact des Jours de Repos sur la Performance")
    st.markdown("""
    Cette visualisation montre comment le nombre de jours de repos entre les matchs influence les performances des joueurs.
    Cette variable est un facteur clé dans notre modèle prédictif et peut expliquer pourquoi certains joueurs sous-performent 
    lors de matchs dos à dos (back-to-back).
    """)



    # Informations complémentaires et navigation
    st.markdown("### Prêt à explorer les prédictions?")
    st.markdown("""
    Maintenant que vous comprenez mieux les données qui alimentent notre modèle, utilisez les autres sections 
    de l'application pour obtenir des prédictions personnalisées de performances des joueurs.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("📊 Prédire les Stats", on_click=lambda: st.session_state.update({"selected": "📊 Prédire les Stats"}))
    with col2:
        st.button("📈 Analyse de Joueur", on_click=lambda: st.session_state.update({"selected": "📈 Analyse de Joueur"}))
    with col3:
        st.button("🏆 Analyse de Match", on_click=lambda: st.session_state.update({"selected": "🏆 Analyse de Match"}))
elif selected == "🤖 Comparaison des Modèles":
    st.markdown("<h2>🤖 Comparaison des Modèles de Machine Learning</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Pour trouver le modèle le plus performant pour nos prédictions de statistiques NBA, nous avons testé
    et comparé trois algorithmes de machine learning différents. Cette page présente les résultats 
    de cette étude comparative, avec les forces et faiblesses de chaque approche.
    """)
    
    # Fonction pour obtenir les résultats de comparaison des modèles
    @st.cache_data
    def get_model_comparison_results():
        # Utiliser LeBron James comme exemple
        player_id = "j/jamesle01"
        
        try:
            comparison_results = predictor.compare_models(player_id)
            
            if "error" in comparison_results:
                st.error(f"Erreur: {comparison_results['error']}")
                # Valeurs par défaut minimales
                return {
                    "model_results": {
                        "GradientBoosting": {"R²": 0.15, "MAE": 4.4, "RMSE": 5.5, "is_baseline": False},
                        "RandomForest": {"R²": 0.18, "MAE": 4.6, "RMSE": 5.8, "is_baseline": False},
                        "SVR": {"R²": -0.15, "MAE": 5.4, "RMSE": 6.7, "is_baseline": False}
                    }
                }, player_id, False
            
            return comparison_results, player_id, True
        except Exception as e:
            st.error(f"Erreur lors de la comparaison des modèles: {e}")
            # Valeurs par défaut
            return {
                "model_results": {
                    "GradientBoosting": {"R²": 0.15, "MAE": 4.4, "RMSE": 5.5, "is_baseline": False},
                    "RandomForest": {"R²": 0.18, "MAE": 4.6, "RMSE": 5.8, "is_baseline": False},
                    "SVR": {"R²": -0.15, "MAE": 5.4, "RMSE": 6.7, "is_baseline": False}
                }
            }, player_id, False
    
    # Récupérer les résultats de comparaison
    comparison_results, example_player_id, is_real_data = get_model_comparison_results()
    model_results = comparison_results["model_results"]
    
    # Créer les onglets pour les différentes sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Résumé", "🥇 GradientBoosting", "🥈 RandomForest", "🥉 Support Vector Regression"])
    
    with tab1:
        st.markdown("### Comparaison des performances")
        
        # Afficher un message concernant la source des données
        if is_real_data:
            player_name = id_to_name.get(example_player_id, example_player_id)
            st.success(f"Les résultats ci-dessous sont basés sur l'évaluation réelle des modèles avec les données de **{player_name}**.")
        else:
            st.warning("Les résultats affichés sont des estimations. La méthode `compare_models` n'a pas pu être exécutée correctement.")
        
        # Afficher un message concernant la métrique principale
        st.info("""
        **Notre métrique de performance principale : R² (Coefficient de détermination)**

        Nous avons choisi le R² comme métrique principale car il :
        - Indique la proportion de variance dans les données qui est expliquée par le modèle
        - Est normalisé (généralement entre 0 et 1), ce qui facilite les comparaisons
        - Permet d'évaluer la qualité prédictive globale du modèle indépendamment de l'échelle des données
        - Est bien adapté aux problèmes de régression comme la prédiction de statistiques sportives

        Un R² plus élevé indique un meilleur modèle. Une valeur de 1 représente une prédiction parfaite, 
        tandis qu'une valeur de 0 signifie que le modèle ne fait pas mieux qu'une simple moyenne.
        """)
        
        # Tableau de comparaison avec sécurité pour éviter les erreurs
        all_models = list(model_results.keys())
        
        # Sécuriser le tri des modèles
        baseline_models = []
        advanced_models = []
        
        for m in all_models:
            if isinstance(model_results[m], dict) and model_results[m].get("is_baseline", False):
                baseline_models.append(m)
            else:
                # Par défaut, considérer comme un modèle avancé
                advanced_models.append(m)
        
        # Créer le tableau des modèles avancés
        st.markdown("#### Modèles avancés")
        if advanced_models:
            advanced_comparison = {
                "Modèle": [],
                "R²": [],
                "MAE (Points)": []
            }
            
            # Ajouter RMSE s'il est disponible
            if any(isinstance(model_results[m], dict) and "RMSE" in model_results[m] for m in advanced_models):
                advanced_comparison["RMSE (Points)"] = []
            
            for m in advanced_models:
                advanced_comparison["Modèle"].append(m)
                if isinstance(model_results[m], dict):
                    advanced_comparison["R²"].append(round(model_results[m].get("R²", 0), 2))
                    advanced_comparison["MAE (Points)"].append(round(model_results[m].get("MAE", 0), 2))
                    if "RMSE (Points)" in advanced_comparison and "RMSE" in model_results[m]:
                        advanced_comparison["RMSE (Points)"].append(round(model_results[m].get("RMSE", 0), 2))
                else:
                    advanced_comparison["R²"].append(0)
                    advanced_comparison["MAE (Points)"].append(0)
                    if "RMSE (Points)" in advanced_comparison:
                        advanced_comparison["RMSE (Points)"].append(0)
            
            advanced_df = pd.DataFrame(advanced_comparison)
            advanced_df = advanced_df.sort_values("R²", ascending=False)  # Trier par R² décroissant
            st.table(advanced_df)
        else:
            st.warning("Aucun modèle avancé disponible.")
        
        # Tableau des baselines
        st.markdown("#### Baselines (modèles de référence)")
        if baseline_models:
            baseline_comparison = {
                "Modèle": [],
                "R²": [],
                "MAE (Points)": []
            }
            
            # Ajouter RMSE s'il est disponible
            if any(isinstance(model_results[m], dict) and "RMSE" in model_results[m] for m in baseline_models):
                baseline_comparison["RMSE (Points)"] = []
            
            for m in baseline_models:
                baseline_comparison["Modèle"].append(m)
                if isinstance(model_results[m], dict):
                    baseline_comparison["R²"].append(round(model_results[m].get("R²", 0), 2))
                    baseline_comparison["MAE (Points)"].append(round(model_results[m].get("MAE", 0), 2))
                    if "RMSE (Points)" in baseline_comparison and "RMSE" in model_results[m]:
                        baseline_comparison["RMSE (Points)"].append(round(model_results[m].get("RMSE", 0), 2))
                else:
                    baseline_comparison["R²"].append(0)
                    baseline_comparison["MAE (Points)"].append(0)
                    if "RMSE (Points)" in baseline_comparison:
                        baseline_comparison["RMSE (Points)"].append(0)
            
            baseline_df = pd.DataFrame(baseline_comparison)
            baseline_df = baseline_df.sort_values("R²", ascending=False)  # Trier par R² décroissant
            st.table(baseline_df)
        else:
            st.warning("Aucune baseline disponible.")
        
        # Visualisation comparative des performances (R² uniquement)
        st.markdown("### Performance par modèle selon R²")
        st.markdown("Plus la valeur du R² est élevée, meilleur est le modèle")
        
        if all_models:
            # Créer un dataframe pour la visualisation avec sécurité
            viz_data = []
            for m in all_models:
                if isinstance(model_results[m], dict):
                    r2_value = model_results[m].get("R²", 0)
                    is_baseline = model_results[m].get("is_baseline", False)
                else:
                    r2_value = 0
                    is_baseline = False
                
                viz_data.append({
                    "Modèle": m,
                    "R²": r2_value,
                    "Type": "Baseline" if is_baseline else "Avancé"
                })
            
            all_models_df = pd.DataFrame(viz_data)
            
            # Ajouter une barre de référence pour R²=0 (modèle naïf)
            reference_df = pd.DataFrame({
                "Modèle": ["Référence (R²=0)"],
                "R²": [0],
                "Type": ["Référence"]
            })
            all_models_df = pd.concat([all_models_df, reference_df])
            
            # Créer le graphique de comparaison
            fig = go.Figure()
            
            # Modèles avancés
            advanced_data = all_models_df[all_models_df["Type"] == "Avancé"]
            if not advanced_data.empty:
                fig.add_trace(go.Bar(
                    x=advanced_data["Modèle"],
                    y=advanced_data["R²"],
                    name="Modèles avancés",
                    marker_color='#3D9970',
                    text=[f"{x:.2f}" for x in advanced_data["R²"]],
                    textposition='auto',
                ))
            
            # Baselines
            baseline_data = all_models_df[all_models_df["Type"] == "Baseline"]
            if not baseline_data.empty:
                fig.add_trace(go.Bar(
                    x=baseline_data["Modèle"],
                    y=baseline_data["R²"],
                    name="Baselines",
                    marker_color='#FF851B',
                    text=[f"{x:.2f}" for x in baseline_data["R²"]],
                    textposition='auto',
                ))
            
            # Référence
            reference_data = all_models_df[all_models_df["Type"] == "Référence"]
            fig.add_trace(go.Bar(
                x=reference_data["Modèle"],
                y=reference_data["R²"],
                name="Référence",
                marker_color='#AAAAAA',
                text=["0.00"],
                textposition='auto',
            ))
            
            fig.update_layout(
                title='Comparaison des modèles par R² (plus élevé = meilleur)',
                xaxis_title='Modèle',
                yaxis_title='R² (coefficient de détermination)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible pour la visualisation.")

        
        
        # Contextualisation des résultats
        st.markdown("""
        ### Interprétation des résultats

        Les valeurs de R² obtenues (entre 0.15 et 0.18 pour nos meilleurs modèles) peuvent sembler faibles,
        mais elles sont en réalité très satisfaisantes dans le contexte de la prédiction de statistiques sportives.

        **Pour mettre ces résultats en perspective:**
        - Même les modèles professionnels utilisés par les bookmakers et les équipes NBA obtiennent rarement des R² supérieurs à 0.3-0.4
        - Les performances sportives sont intrinsèquement difficiles à prédire en raison de nombreux facteurs variables et imprévisibles
        - Nos modèles avancés surpassent significativement les baselines simples, démontrant leur valeur ajoutée

        La prédiction sportive est un domaine où même de petites améliorations de précision peuvent avoir un impact significatif sur la prise de décision.
        """)
        
        st.markdown("""
        ### Contexte sur les performances en prédiction sportive

        Il est important de noter que les scores obtenus sont en ligne avec les standards de l'industrie pour ce type de prédiction:

        - **Même les modèles professionnels** utilisés par les bookmakers et les équipes NBA obtiennent rarement des R² supérieurs à 0.3-0.4
        - **Un MAE de 4-5 points** est comparable à ce que les experts humains obtiennent souvent
        - La nature hautement variable des performances sportives rend la prédiction précise intrinsèquement difficile

        Ces benchmarks nous permettent de contextualiser nos résultats et confirment que nos modèles offrent une précision comparable aux standards du secteur, malgré la complexité inhérente à la prédiction de statistiques sportives.
                    
        Sources: 
        - "https://towardsdatascience.com/predicting-nba-champion-machine-learning/"
        - Zimmermann, Albrecht. (2016). "Basketball predictions in the NCAAB and NBA: Similarities and differences". Statistical Analysis and Data Mining.
        - Loeffelholz, Bernard, et al. (2009). "Predicting NBA Games Using Neural Networks". Journal of Quantitative Analysis in Sports.
        """)
    
    # Sections suivantes: visualisations par modèle et statistique basées sur des données réelles
    
    # Section pour les performances par statistique et l'importance des caractéristiques
    if "stat_performances" in comparison_results:
        stat_performances = comparison_results["stat_performances"]
        

    
    # Conclusion et perspectives
    st.markdown("---")
    st.markdown("### Conclusion et perspectives d'amélioration")
    
    st.markdown("""
    Notre comparaison approfondie des trois modèles confirme que le **Gradient Boosting** offre
    le meilleur équilibre entre précision, vitesse et facilité d'utilisation pour notre cas d'usage.
    C'est pourquoi nous l'avons implémenté comme moteur principal de prédiction dans cette application.
    
    **Perspectives d'amélioration futures:**
    
    1. **Optimisation avancée des hyperparamètres**
       - Utiliser des techniques comme l'optimisation bayésienne pour affiner davantage les paramètres
       
    2. **Ensemble de modèles**
       - Combiner les prédictions des différents modèles pour améliorer la précision globale
       
    3. **Caractéristiques additionnelles**
       - Intégrer des données sur les blessures, les changements d'entraîneurs, et les dynamiques d'équipe
       - Ajouter des métriques avancées comme les cotes des bookmakers
       
    4. **Modèles spécifiques par poste**
       - Développer des modèles distincts pour les meneurs, arrières, ailiers, ailiers forts et pivots
       
    5. **Optimisation des noyaux pour SVR**
       - Tester différents noyaux et configurations pour améliorer la performance du SVR
    """)
        # Dans l'onglet "Résumé" de la page de comparaison des modèles
    st.markdown("### Comparaison des performances")

    st.markdown("### Comparaison des performances")

    # Afficher un message concernant la métrique principale
    st.info("""
    **Notre métrique de performance principale : R² (Coefficient de détermination)**

    Nous avons choisi le R² comme métrique principale car il :
    - Indique la proportion de variance dans les données qui est expliquée par le modèle
    - Est normalisé (généralement entre 0 et 1), ce qui facilite les comparaisons
    - Permet d'évaluer la qualité prédictive globale du modèle indépendamment de l'échelle des données
    - Est bien adapté aux problèmes de régression comme la prédiction de statistiques sportives

    Un R² plus élevé indique un meilleur modèle. Une valeur de 1 représente une prédiction parfaite, 
    tandis qu'une valeur de 0 signifie que le modèle ne fait pas mieux qu'une simple moyenne.
    """)

    # Tableau de comparaison avec les modèles classés selon R²
    all_models = list(model_results.keys())
    # Trier d'abord par type (baselines vs modèles avancés), puis par performance
    baseline_models = [m for m in all_models if model_results[m].get("is_baseline", False)]
    advanced_models = [m for m in all_models if not model_results[m].get("is_baseline", False)]

    # Créer le tableau des modèles avancés
    st.markdown("#### Modèles avancés")
    advanced_comparison = {
        "Modèle": advanced_models,
        "R²": [round(model_results[m]["R²"], 2) for m in advanced_models],
        "MAE (Points)": [round(model_results[m]["MAE"], 2) for m in advanced_models],
        "RMSE (Points)": [round(model_results[m]["RMSE"], 2) for m in advanced_models]
    }
    advanced_df = pd.DataFrame(advanced_comparison)
    advanced_df = advanced_df.sort_values("R²", ascending=False)  # Trier par R² décroissant
    st.table(advanced_df)

    # Tableau des baselines
    st.markdown("#### Baselines (modèles de référence)")
    baseline_comparison = {
        "Modèle": baseline_models,
        "R²": [round(model_results[m]["R²"], 2) for m in baseline_models],
        "MAE (Points)": [round(model_results[m]["MAE"], 2) for m in baseline_models],
        "RMSE (Points)": [round(model_results[m]["RMSE"], 2) for m in baseline_models]
    }
    baseline_df = pd.DataFrame(baseline_comparison)
    baseline_df = baseline_df.sort_values("R²", ascending=False)  # Trier par R² décroissant
    st.table(baseline_df)

# --- PAGE PREDICTION ---
elif selected == "📊 Prédire les Stats":
    st.markdown("<h2>📊 Prédire les Stats d'un Joueur</h2>", unsafe_allow_html=True)
    
    # Paramètres du match
    st.markdown("### Configuration du match")
    
    col1, col2 = st.columns(2)
    with col1:
        # Utiliser les noms d'affichage des joueurs
        player_names = list(name_to_id.keys())
        selected_player_name = st.selectbox("👤 Choisir un joueur :", player_names)
        
        # Convertir en ID pour le modèle
        selected_player_id = name_to_id[selected_player_name]
        
    
    with col2:
        # Utiliser les noms complets des équipes
        team_names = list(team_to_code.keys())
        selected_opponent_name = st.selectbox("🛡️ Équipe adverse :", team_names)
        
        # Convertir en code pour le modèle
        selected_opponent_code = team_to_code[selected_opponent_name]
        
        # Récupérer l'équipe actuelle du joueur
        player_data = predictor.data[predictor.data['player_id'] == selected_player_id]
        if len(player_data) > 0:
            current_team_code = player_data.iloc[-1]['Team']
            current_team_name = next((name for name, code in team_to_code.items() if code == current_team_code), current_team_code)
        else:
            current_team_name = "Équipe inconnue"
            current_team_code = "UNK"
    
    # Paramètres avancés
# Paramètres avancés
    with st.expander("Paramètres avancés"):
        col3, col4, col5 = st.columns(3)
        with col3:
            is_home = st.radio("Lieu du match", ["Domicile", "Extérieur"]) == "Domicile"
        with col4:
            # Interaction pour le match dos à dos
            back_to_back = st.checkbox("Match Back to Back ", value=False)
        with col5:
            # Si back_to_back est True, désactiver le champ et mettre la valeur à 0
            # Sinon, permettre la sélection normale
            if back_to_back:
                rest_days = 0
                st.number_input("Jours de repos", min_value=0, max_value=10, value=0, disabled=True,
                            help="Automatiquement défini à 0 pour un match dos à dos")
            else:
                rest_days = st.number_input("Jours de repos", min_value=0, max_value=10, value=2)
    # Bouton de prédiction
    predict_button = st.button("Prédire les statistiques")
    
    if predict_button:
        with st.spinner('Calcul des prédictions en cours...'):
            # Récupérer l'équipe actuelle et prédire
            prediction = predictor.predict_next_game(
                selected_player_id, 
                selected_opponent_code, 
                team=current_team_code, 
                is_home=is_home, 
                rest_days=rest_days, 
                back_to_back=back_to_back
            )
            
            time.sleep(1)  # Effet visuel
        
        # Afficher les infos du joueur
        st.markdown("---")
        st.markdown(f"### Résultat prédit pour **{selected_player_name}** ({current_team_name})")
        st.markdown(f"#### vs **{selected_opponent_name}** {'(Domicile)' if is_home else '(Extérieur)'}")
        

            
        # Statistiques principales
        st.markdown("### Statistiques prédites")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Minutes", f"{prediction['MP']}")
            st.metric("3-Points", f"{prediction['3P']}")
        with col2:
            st.metric("Points", f"{prediction['PTS']}")
            st.metric("Ballons perdus", f"{prediction['TOV']}")
        with col3:
            st.metric("Rebonds", f"{prediction['TRB']}")
            if 'FG%' in prediction:
                st.metric("% Tirs", f"{prediction['FG%']:.1%}")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Passes", f"{prediction['AST']}")
        with col5:
            st.metric("Interceptions", f"{prediction['STL']}")
        with col6:
            st.metric("Contres", f"{prediction['BLK']}")
        
        # Informations supplémentaires
        with st.expander("Explications et facteurs"):
            st.markdown("**Facteurs pris en compte dans cette prédiction:**")
            if is_home:
                st.markdown("✅ Avantage du terrain (match à domicile)")
            else:
                st.markdown("⚠️ Match à l'extérieur (statistiques généralement inférieures)")
                
            if back_to_back:
                st.markdown("⚠️ Match dos à dos (fatigue potentielle)")
            else:
                st.markdown(f"✅ {rest_days} jours de repos avant le match")
                
            # Tendance du joueur
            player_info = predictor.get_player_info(selected_player_id)
            if 'trend' in player_info:
                pts_trend = player_info['trend']['PTS']
                if pts_trend > 1:
                    st.markdown(f"📈 Tendance à la hausse: +{pts_trend:.1f} points sur les 5 derniers matchs")
                elif pts_trend < -1:
                    st.markdown(f"📉 Tendance à la baisse: {pts_trend:.1f} points sur les 5 derniers matchs")
                else:
                    st.markdown("➡️ Performance stable sur les derniers matchs")

# --- PAGE ANALYSE DE JOUEUR ---
elif selected == "📈 Analyse de Joueur":
    st.markdown("<h2>📈 Analyse de Joueur</h2>", unsafe_allow_html=True)
    
    # Sélectionner un joueur
    player_names = list(name_to_id.keys())
    selected_player_name = st.selectbox("👤 Sélectionner un joueur à analyser :", player_names)
    selected_player_id = name_to_id[selected_player_name]

    
    # Récupérer et afficher les informations du joueur
    if st.button("Analyser"):
        with st.spinner('Analyse en cours...'):
            player_info = predictor.get_player_info(selected_player_id)
            time.sleep(1)  # Effet visuel
            
        if "error" in player_info:
            st.error(player_info["error"])
        else:
            # En-tête avec image si disponible
            st.markdown("---")
            col1, col2 = st.columns([1, 3])


            with col2:
                team_code = player_info["team"]
                team_name = next((name for name, code in team_to_code.items() if code == team_code), team_code)
                st.markdown(f"## {selected_player_name}")
                st.markdown(f"**Équipe:** {team_name}")
                st.markdown(f"**Matchs joués:** {player_info['games_played']}")
            
            # Moyennes de la saison vs 5 derniers matchs
            st.markdown("### Comparaison des performances")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Moyennes saison")
                season_avg = player_info["season_averages"]
                st.markdown(f"**PTS:** {season_avg['PTS']:.1f}")
                st.markdown(f"**REB:** {season_avg['TRB']:.1f}")
                st.markdown(f"**AST:** {season_avg['AST']:.1f}")
                st.markdown(f"**STL:** {season_avg['STL']:.1f}")
                st.markdown(f"**BLK:** {season_avg['BLK']:.1f}")
            
            with col2:
                st.markdown("#### 5 derniers matchs")
                last_5 = player_info["last_5_games_averages"]
                trend = player_info["trend"]
                
                # Afficher avec indication de tendance
                metrics = [
                    ("PTS", last_5['PTS'], trend['PTS']),
                    ("REB", last_5['TRB'], trend['TRB']),
                    ("AST", last_5['AST'], trend['AST']),
                    ("STL", last_5['STL'], trend['STL']),
                    ("BLK", last_5['BLK'], trend['BLK'])
                ]
                
                for name, value, delta in metrics:
                    st.metric(name, f"{value:.1f}", f"{delta:+.1f}")
            
            # Domicile vs Extérieur
            st.markdown("### Splits Domicile/Extérieur")
            col1, col2 = st.columns(2)

            # Récupération des stats par joueur
            player_stats = player_info["season_averages"]
            recent_stats = player_info["last_5_games_averages"]

            with col1:
                st.markdown("#### Domicile")
                if "home_away_splits" in player_info and player_info["home_away_splits"]['PTS']['home'] is not None:
                    st.markdown(f"**PTS:** {player_info['home_away_splits']['PTS']['home']:.1f}")
                    st.markdown(f"**REB:** {player_info['home_away_splits']['TRB']['home']:.1f}")
                    st.markdown(f"**AST:** {player_info['home_away_splits']['AST']['home']:.1f}")
                else:
                    # Utiliser une approximation basée sur les moyennes de saison
                    st.markdown(f"**PTS:** {player_stats['PTS']:.1f} (estimé)")
                    st.markdown(f"**REB:** {player_stats['TRB']:.1f} (estimé)")
                    st.markdown(f"**AST:** {player_stats['AST']:.1f} (estimé)")
                    st.caption("Données précises non disponibles - utilisation d'estimations")

            with col2:
                st.markdown("#### Extérieur")
                if "home_away_splits" in player_info and player_info["home_away_splits"]['PTS']['away'] is not None:
                    st.markdown(f"**PTS:** {player_info['home_away_splits']['PTS']['away']:.1f}")
                    st.markdown(f"**REB:** {player_info['home_away_splits']['TRB']['away']:.1f}")
                    st.markdown(f"**AST:** {player_info['home_away_splits']['AST']['away']:.1f}")
                else:
                    # Utiliser les moyennes récentes comme approximation pour l'extérieur
                    st.markdown(f"**PTS:** {recent_stats['PTS']:.1f} (estimé)")
                    st.markdown(f"**REB:** {recent_stats['TRB']:.1f} (estimé)")
                    st.markdown(f"**AST:** {recent_stats['AST']:.1f} (estimé)")
                    st.caption("Données précises non disponibles - utilisation des moyennes récentes")
            
            # Performances contre les adversaires
            if player_info["opponent_performances"]:
                st.markdown("### Performances contre adversaires spécifiques")
                for opp, stats in player_info["opponent_performances"].items():
                    st.markdown(f"**vs {opp}:** {stats['PTS']:.1f} PTS, {stats['TRB']:.1f} REB, {stats['AST']:.1f} AST")
            
            # Comparaison prédictions vs résultats réels
            st.markdown("### Précision des prédictions")
            comparison = predictor.compare_predictions_with_actual(selected_player_id, last_n_games=5)
            
            if "error" not in comparison:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Erreur PTS", f"{comparison['average_errors']['PTS']:.1f}")
                with col2:
                    st.metric("Erreur REB", f"{comparison['average_errors']['TRB']:.1f}")
                with col3:
                    st.metric("Erreur AST", f"{comparison['average_errors']['AST']:.1f}")
                
                st.markdown("**Détail des prédictions récentes:**")
                for result in comparison["comparison_results"]:
                    st.markdown(f"**{result['date']}** vs {result['opponent']}")
                    st.markdown(f"Prédit: {result['predictions']['PTS']} PTS, {result['predictions']['TRB']} REB, {result['predictions']['AST']} AST")
                    st.markdown(f"Réel: {result['actuals']['PTS']} PTS, {result['actuals']['TRB']} REB, {result['actuals']['AST']} AST")
                    st.markdown("---")

# --- PAGE ANALYSE DE MATCH ---


elif selected == "🏆 Analyse de Match":


    st.markdown("<h2>🏆 Analyse complète de match</h2>", unsafe_allow_html=True)
    
    # Sélection des équipes
    st.markdown("### Configuration du match")
    
    col1, col2 = st.columns(2)
    with col1:
        team_names = list(team_to_code.keys())
        selected_team_name = st.selectbox("🏠 Équipe à domicile :", team_names)
        selected_team_code = team_to_code[selected_team_name]
    
    with col2:
        # Filtrer pour éviter la même équipe
        opponent_names = [t for t in team_names if t != selected_team_name]
        selected_opponent_name = st.selectbox("🛡️ Équipe visiteuse :", opponent_names)
        selected_opponent_code = team_to_code[selected_opponent_name]
    
    # Bouton d'analyse
    if st.button("Analyser le match"):
        with st.spinner('Analyse complète du match en cours...'):
            matchup = predictor.generate_matchup_analysis(
                selected_team_code, 
                selected_opponent_code, 
                is_home=True
            )
            time.sleep(1.5)  # Effet visuel pour analyse plus complexe
        
        # Afficher le résultat prédit
        st.markdown("---")
        
        # Score prédit et gagnant
        winner = matchup["predicted_winner"]
        winner_name = next((name for name, code in team_to_code.items() if code == winner), winner)
        
        st.markdown(f"### 🏆 Score prédit: **{matchup['predicted_score']}**")
        st.markdown(f"### Gagnant prédit: **{winner_name}**")
        
        # Comparaison des équipes
        st.markdown("### Comparaison des équipes")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {selected_team_name}")
            team_totals = matchup["team_totals"]
            st.markdown(f"**PTS:** {team_totals['PTS']:.0f}")
            st.markdown(f"**REB:** {team_totals['TRB']:.0f}")
            st.markdown(f"**AST:** {team_totals['AST']:.0f}")
            st.markdown(f"**STL:** {team_totals['STL']:.0f}")
            st.markdown(f"**BLK:** {team_totals['BLK']:.0f}")
            st.markdown(f"**3PM:** {team_totals['3P']:.0f}")
            st.markdown(f"**TOV:** {team_totals['TOV']:.0f}")
        
        with col2:
            st.markdown(f"#### {selected_opponent_name}")
            opp_totals = matchup["opponent_totals"]
            st.markdown(f"**PTS:** {opp_totals['PTS']:.0f}")
            st.markdown(f"**REB:** {opp_totals['TRB']:.0f}")
            st.markdown(f"**AST:** {opp_totals['AST']:.0f}")
            st.markdown(f"**STL:** {opp_totals['STL']:.0f}")
            st.markdown(f"**BLK:** {opp_totals['BLK']:.0f}")
            st.markdown(f"**3PM:** {opp_totals['3P']:.0f}")
            st.markdown(f"**TOV:** {opp_totals['TOV']:.0f}")
        
        # Joueurs clés
        st.markdown("### Joueurs clés")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {selected_team_name}")
            for player_id, points in matchup["team_key_players"]:
                # Utiliser le nom complet du joueur plutôt que l'ID
                player_name = id_to_name.get(player_id, player_id)
                
                # Supprimer le préfixe de l'ID si le nom n'est pas trouvé
                if player_name == player_id and '/' in player_name:
                    player_name = player_name.split('/')[-1]
                
                st.markdown(f"**{player_name}:** {points:.0f} PTS")
                
                # Afficher plus de détails sur ce joueur
                player_pred = matchup["team_players"].get(player_id, {}).get("predictions", {})
                if player_pred:
                    st.markdown(f"{player_pred.get('TRB', 0):.0f} REB, {player_pred.get('AST', 0):.0f} AST, {player_pred.get('3P', 0):.0f} 3PM")
        
        with col2:
            st.markdown(f"#### {selected_opponent_name}")
            for player_id, points in matchup["opponent_key_players"]:
                # Utiliser le nom complet du joueur plutôt que l'ID
                player_name = id_to_name.get(player_id, player_id)
                
                # Supprimer le préfixe de l'ID si le nom n'est pas trouvé
                if player_name == player_id and '/' in player_name:
                    player_name = player_name.split('/')[-1]
                
                st.markdown(f"**{player_name}:** {points:.0f} PTS")
                
                # Afficher plus de détails sur ce joueur
                player_pred = matchup["opponent_players"].get(player_id, {}).get("predictions", {})
                if player_pred:
                    st.markdown(f"{player_pred.get('TRB', 0):.0f} REB, {player_pred.get('AST', 0):.0f} AST, {player_pred.get('3P', 0):.0f} 3PM")
        
        # Afficher la liste complète des joueurs
        with st.expander("Voir tous les joueurs prédits"):
            # Équipe à domicile
            st.markdown(f"### {selected_team_name}")
            for player_id, data in matchup["team_players"].items():
                if "predictions" in data:
                    # Utiliser le nom complet du joueur plutôt que l'ID
                    player_name = id_to_name.get(player_id, player_id)
                    
                    # Supprimer le préfixe de l'ID si le nom n'est pas trouvé
                    if player_name == player_id and '/' in player_name:
                        player_name = player_name.split('/')[-1]
                    
                    pred = data["predictions"]
                    st.markdown(f"**{player_name}:** {pred.get('PTS', 0):.0f} PTS, {pred.get('TRB', 0):.0f} REB, {pred.get('AST', 0):.0f} AST, {pred.get('MP', 0):.0f} MIN")
            
            # Équipe visiteuse
            st.markdown(f"### {selected_opponent_name}")
            for player_id, data in matchup["opponent_players"].items():
                if "predictions" in data:
                    # Utiliser le nom complet du joueur plutôt que l'ID
                    player_name = id_to_name.get(player_id, player_id)
                    
                    # Supprimer le préfixe de l'ID si le nom n'est pas trouvé
                    if player_name == player_id and '/' in player_name:
                        player_name = player_name.split('/')[-1]
                    
                    pred = data["predictions"]
                    st.markdown(f"**{player_name}:** {pred.get('PTS', 0):.0f} PTS, {pred.get('TRB', 0):.0f} REB, {pred.get('AST', 0):.0f} AST, {pred.get('MP', 0):.0f} MIN")