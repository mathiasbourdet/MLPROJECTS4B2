import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import os
from nba_stats_predictor import NBAStatsPredictor
import time

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
    padding: 10px;
    border-radius: 8px;
    color: #111827;
}
.stTextInput > div > input {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 8px;
    color: #111827;
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
            display_name = player_entries['Player'].iloc[0] if 'Player' in player_entries.columns else pid
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
        'PHX': 'Phoenix Suns',
        'CLE': 'Cleveland Cavaliers',
        'TOR': 'Toronto Raptors',
        'CHI': 'Chicago Bulls',
        'BKN': 'Brooklyn Nets',
        'ATL': 'Atlanta Hawks',
        'MEM': 'Memphis Grizzlies',
        'NOP': 'New Orleans Pelicans',
        'MIN': 'Minnesota Timberwolves',
        'POR': 'Portland Trail Blazers',
        'CHA': 'Charlotte Hornets',
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

# --- MENU ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Accueil", "📊 Prédire les Stats", "📈 Analyse de Joueur", "🏆 Analyse de Match"],
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
    st.image("https://cdn.nba.com/manage/2022/12/nba-logo.png", width=120)
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
        
        # Image du joueur (facultatif)
        image_url = st.text_input("📸 Lien d'une photo du joueur (facultatif)", placeholder="https://...")
    
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
    with st.expander("Paramètres avancés"):
        col3, col4, col5 = st.columns(3)
        with col3:
            is_home = st.radio("Lieu du match", ["Domicile", "Extérieur"]) == "Domicile"
        with col4:
            rest_days = st.number_input("Jours de repos", min_value=0, max_value=10, value=2)
        with col5:
            back_to_back = st.checkbox("Match dos à dos", value=False)
    
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
        
        if image_url:
            st.image(image_url, width=150, caption=selected_player_name)
            
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
    
    # Image du joueur (facultatif)
    image_url = st.text_input("📸 Lien d'une photo du joueur (facultatif)", placeholder="https://...")
    
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
            with col1:
                if image_url:
                    st.image(image_url, width=150)
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
            
            with col1:
                st.markdown("#### Domicile")
                home_splits = player_info["home_away_splits"]
                if home_splits['PTS']['home'] is not None:
                    st.markdown(f"**PTS:** {home_splits['PTS']['home']:.1f}")
                    st.markdown(f"**REB:** {home_splits['TRB']['home']:.1f}")
                    st.markdown(f"**AST:** {home_splits['AST']['home']:.1f}")
                else:
                    st.markdown("Données insuffisantes")
            
            with col2:
                st.markdown("#### Extérieur")
                if home_splits['PTS']['away'] is not None:
                    st.markdown(f"**PTS:** {home_splits['PTS']['away']:.1f}")
                    st.markdown(f"**REB:** {home_splits['TRB']['away']:.1f}")
                    st.markdown(f"**AST:** {home_splits['AST']['away']:.1f}")
                else:
                    st.markdown("Données insuffisantes")
            
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
                player_name = id_to_name.get(player_id, player_id)
                st.markdown(f"**{player_name}:** {points:.0f} PTS")
                
                # Afficher plus de détails sur ce joueur
                player_pred = matchup["team_players"].get(player_id, {}).get("predictions", {})
                if player_pred:
                    st.markdown(f"{player_pred.get('TRB', 0):.0f} REB, {player_pred.get('AST', 0):.0f} AST, {player_pred.get('3P', 0):.0f} 3PM")
        
        with col2:
            st.markdown(f"#### {selected_opponent_name}")
            for player_id, points in matchup["opponent_key_players"]:
                player_name = id_to_name.get(player_id, player_id)
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
                    player_name = id_to_name.get(player_id, player_id)
                    pred = data["predictions"]
                    st.markdown(f"**{player_name}:** {pred.get('PTS', 0):.0f} PTS, {pred.get('TRB', 0):.0f} REB, {pred.get('AST', 0):.0f} AST, {pred.get('MP', 0):.0f} MIN")
            
            # Équipe visiteuse
            st.markdown(f"### {selected_opponent_name}")
            for player_id, data in matchup["opponent_players"].items():
                if "predictions" in data:
                    player_name = id_to_name.get(player_id, player_id)
                    pred = data["predictions"]
                    st.markdown(f"**{player_name}:** {pred.get('PTS', 0):.0f} PTS, {pred.get('TRB', 0):.0f} REB, {pred.get('AST', 0):.0f} AST, {pred.get('MP', 0):.0f} MIN")