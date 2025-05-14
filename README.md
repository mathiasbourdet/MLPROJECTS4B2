NBA Player Game Logs Dataset
Description
Ce projet consiste en la création d'un dataset contenant les statistiques de matchs (game logs) des principaux joueurs de la NBA pour la saison 2024-2025. Les données sont obtenues par web scraping du site Basketball Reference.
Contenu du Dataset
Le dataset contient les statistiques de match par match pour des joueurs NBA, incluant :

Points, rebonds, passes décisives par match
Pourcentages de tir (FG%, 3P%, FT%)
Minutes jouées
Autres statistiques avancées

Chaque ligne correspond à un match joué par un joueur, avec un identifiant unique pour chaque joueur.
[Lien vers le Dataset](https://drive.google.com/drive/folders/1u1KTRxpqRyB3bzIk1aUvmnHu7-pNRy63?usp=share_link)
Accéder au dataset sur Google Drive
Script de Génération
Le script nba_scraper.py contient le code utilisé pour générer ce dataset. Il utilise les bibliothèques suivantes :

requests pour les requêtes HTTP
pandas pour la manipulation des données
time et random pour gérer les délais entre les requêtes (afin d'éviter le rate limiting)

Caractéristiques du Script

Gestion des erreurs 429 (Too Many Requests) avec système de retry
Délais aléatoires entre les requêtes pour éviter le rate limiting
Sauvegarde de points de contrôle pendant le scraping
Support pour différents formats d'ID de joueurs
Nettoyage et normalisation des données (conversion des minutes, gestion des colonnes, etc.)

Comment Utiliser

# Guide d'Installation des Dépendances avec UV

## Qu'est-ce que UV?

UV est un installateur de paquets Python ultra-rapide, écrit en Rust, qui peut remplacer pip. Il est significativement plus rapide que pip et offre une meilleure gestion des dépendances.

## Installation de UV

### Sur macOS
```bash
brew install uv
```

### Sur Linux/WSL
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Sur Windows
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Création d'un Environnement Virtuel avec UV

1. Créez un nouvel environnement virtuel :
```bash
python -m venv .venv
```

2. Activez l'environnement virtuel :

   - Sur macOS/Linux :
   ```bash
   source .venv/bin/activate
   ```
   
   - Sur Windows :
   ```bash
   .venv\Scripts\activate
   ```

## Installation des Dépendances

Une fois l'environnement virtuel activé, installez les dépendances avec UV :

```bash
uv pip install -r requirements.txt
```

### Avantages d'utiliser UV

- 10-100x plus rapide que pip
- Cache intelligent des paquets
- Résolution déterministe des dépendances
- Compatible avec pip et les requirements.txt standards

### Commandes UV Utiles

- Installer un paquet spécifique :
```bash
uv pip install numpy
```

- Mettre à jour un paquet :
```bash
uv pip install --upgrade pandas
```

- Voir les paquets installés :
```bash
uv pip freeze
```

## Dépendances du Projet

Ce projet utilise les bibliothèques Python suivantes :

- numpy (≥1.24.0) : Calculs numériques
- pandas (≥2.0.0) : Manipulation de données
- scikit-learn (≥1.2.0) : Machine learning
- matplotlib (≥3.7.0) : Visualisation de données
- seaborn (≥0.12.0) : Visualisation statistique
- jupyter (≥1.0.0) : Notebooks Jupyter
- python-dotenv (≥1.0.0) : Gestion des variables d'environnement

## Résolution des Problèmes Courants

1. Si UV n'est pas reconnu comme une commande :
   - Vérifiez que UV est bien installé : `which uv`
   - Redémarrez votre terminal
   - Assurez-vous que le chemin de UV est dans votre PATH

2. En cas d'erreur de dépendances :
   ```bash
   uv pip install --upgrade -r requirements.txt
   ```

3. Pour nettoyer le cache de UV :
   ```bash
   uv cache clean
   ```


Exécuter le script :

python nba_scraper.py

Les résultats seront sauvegardés dans un fichier CSV (nba_game_logs_2024.csv).

Structure des Données
Le dataset final inclut les colonnes suivantes :

Date: Date du match
Team: Équipe du joueur
Opp: Équipe adverse
Result: Résultat du match
MP: Minutes jouées
FG: Field Goals (tirs réussis)
FGA: Field Goal Attempts (tentatives de tirs)
FG%: Field Goal Percentage (pourcentage de réussite aux tirs)
3P: Three Points (tirs à 3 points réussis)
3PA: Three Point Attempts (tentatives de tirs à 3 points)
3P%: Three Point Percentage (pourcentage de réussite aux tirs à 3 points)
FT: Free Throws (lancers francs réussis)
FTA: Free Throw Attempts (tentatives de lancers francs)
FT%: Free Throw Percentage (pourcentage de réussite aux lancers francs)
TRB: Total Rebounds (rebonds totaux)
AST: Assists (passes décisives)
STL: Steals (interceptions)
BLK: Blocks (contres)
TOV: Turnovers (pertes de balle)
PF: Personal Fouls (fautes personnelles)
PTS: Points
player_id: Identifiant du joueur

Auteur
Mathias Kuzmak Bourdet
Mael Zapata
Thomas Foussard Le-Meur
Celeste Rein
Charles Defosseux

Note
Le site restreint les requetes (Code: 429), le dataset devra donc encore evoluer dans les prochains jours.
