import streamlit as st
from streamlit_option_menu import option_menu

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
    </style>
""", unsafe_allow_html=True)

# --- MENU ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Accueil", "📊 Prédire les Stats"],
        icons=["house", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#e5e7eb"},
            "icon": {"color": "#2563eb", "font-size": "20px"},
            "nav-link": {"color": "#111827", "font-size": "18px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#dbeafe"},
        }
    )

# --- PAGE ACCUEIL ---
if selected == "🏠 Accueil":
    st.markdown("<h1>🏀 NBA Stat Predictor</h1>", unsafe_allow_html=True)
    st.markdown("### Une expérience de prédiction simple, rapide et fiable.")
    st.image("https://cdn.nba.com/manage/2022/12/nba-logo.png", width=120)
    st.markdown("> ⚡ Utilise notre IA pour estimer les performances NBA à venir.")
    st.markdown("**Commence en cliquant sur ‘📊 Prédire les Stats’ dans le menu.**")

# --- PAGE PREDICTION ---
elif selected == "📊 Prédire les Stats":
    st.markdown("<h2>📊 Prédire les Stats d’un Joueur</h2>", unsafe_allow_html=True)

    # Saisie lien image
    image_url = st.text_input("📸 Colle ici le lien d'une photo du joueur (facultatif)", placeholder="https://...")

    if image_url:
        st.image(image_url, width=200, caption="Image du joueur")

    # Sélections joueur + équipe
    players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo", "Luka Dončić", "Jayson Tatum"]
    teams = ["Boston Celtics", "Los Angeles Lakers", "Golden State Warriors", "Milwaukee Bucks", "Dallas Mavericks"]

    col1, col2 = st.columns(2)
    with col1:
        selected_player = st.selectbox("👤 Choisir un joueur :", players)
    with col2:
        selected_team = st.selectbox("🛡️ Équipe adverse :", teams)

    st.markdown("------")

    # Résultats fictifs (mock prédiction)
    st.markdown(f"### Résultat prédit pour **{selected_player}** face aux **{selected_team}**")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Minutes", "34.5")
    with col4:
        st.metric("Points", "27.8")
    with col5:
        st.metric("Rebonds", "7.2")

    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric("Passes", "6.9")
    with col7:
        st.metric("Interceptions", "1.4")
    with col8:
        st.metric("Contres", "0.9")

    st.markdown("<br><hr><center style='color:gray'>✨ Modèle IA en cours d'intégration — version front-end uniquement</center>", unsafe_allow_html=True)
