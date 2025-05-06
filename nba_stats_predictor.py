import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class NBAStatsPredictor:
    """
    Prédicteur de statistiques pour joueurs NBA basé sur leurs performances passées.
    Utilise un modèle par joueur pour des prédictions personnalisées.
    Version améliorée avec plus de contexte et moins de dépendance aux moyennes.
    """
    
    def __init__(self, data_file='nba_game_logs_2025.csv'):
        """
        Initialise le prédicteur avec les données existantes
        
        Args:
            data_file (str): Chemin vers le fichier CSV contenant les données
        """
        print(f"Initialisation du prédicteur avec le fichier {data_file}")
        self.data_file = data_file
        self.data = None
        self.models = {}
        self.stats_to_predict = ['MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'FG%', 'TOV']
        self.features = ['MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'FG%', 'TOV', 'Team', 'Opp']
        
        # Charger les données
        self.load_data()
        
        # Dictionnaires pour stocker les forces des équipes
        self.team_defensive_ratings = {}
        self.team_pace_factors = {}
        
    def load_data(self):
        """Charge et prépare les données"""
        try:
            self.data = pd.read_csv(self.data_file)
            print(f"Données chargées avec succès! {len(self.data)} matchs trouvés.")
            
            # Convertir les colonnes de date en datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            
            # Supprimer les lignes avec des valeurs manquantes dans les colonnes clés
            key_columns = ['player_id', 'Date', 'PTS', 'MP']
            self.data = self.data.dropna(subset=key_columns)
            
            # Trier par joueur et date
            self.data = self.data.sort_values(['player_id', 'Date'])
            
            # Créer des caractéristiques de tendance (moyennes mobiles et plus)
            self._create_enhanced_features()
            
            # Calculer les forces des équipes
            self._calculate_team_strength()
            
            print(f"Préparation des données terminée. {len(self.data)} matchs après nettoyage.")
            
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
    
    def _calculate_team_strength(self):
        """Calcule des métriques de force pour chaque équipe"""
        # Calcul de la cote défensive de chaque équipe (points encaissés/100 possessions)
        team_games = self.data.groupby(['Date', 'Opp']).agg({'PTS': 'sum'}).reset_index()
        team_games.columns = ['Date', 'Team', 'PTS_Against']
        
        # Moyenne des points encaissés par match pour chaque équipe
        team_def = team_games.groupby('Team').agg({'PTS_Against': 'mean'})
        
        # Normalisation (100 = moyenne de la ligue)
        avg_pts = team_def['PTS_Against'].mean()
        for team in team_def.index:
            self.team_defensive_ratings[team] = 100 * (avg_pts / team_def.loc[team, 'PTS_Against'])

        # Calcul du rythme de jeu de chaque équipe (estimé par les possessions)
        team_pace = self.data.groupby('Team').agg({'FGA': 'mean', 'TOV': 'mean'})
        avg_pace = team_pace.mean().sum()
        
        for team in team_pace.index:
            pace_value = (team_pace.loc[team, 'FGA'] + team_pace.loc[team, 'TOV']) / avg_pace
            self.team_pace_factors[team] = pace_value
            
    def _create_enhanced_features(self):
        """Créer des caractéristiques avancées pour mieux capturer les variations match par match"""
        # Grouper par joueur
        grouped = self.data.groupby('player_id')
        
        # Moyennes mobiles et écarts-types sur différentes fenêtres
        for stat in self.stats_to_predict:
            # Moyennes mobiles standard (3, 5, 10 matchs)
            self.data[f'{stat}_avg_3'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
            
            self.data[f'{stat}_avg_5'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
            
            self.data[f'{stat}_avg_10'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
            
            # Écart-type sur les derniers matchs (mesure de consistance)
            self.data[f'{stat}_std_5'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=2).std())
            
            # Tendance linéaire (pente sur les 5 derniers matchs)
            self.data[f'{stat}_trend'] = grouped[stat].transform(
                lambda x: (x.shift(1) - x.shift(5)) / 4 if len(x) >= 5 else 0)
            
            # Performances pondérées par récence (plus de poids aux matchs récents)
            weights = np.array([0.5, 0.3, 0.2])  # Plus récent → plus important
            self.data[f'{stat}_weighted'] = grouped.apply(
                lambda g: g[stat].shift(1).rolling(window=3, min_periods=1).apply(
                    lambda x: np.sum(weights[:len(x)] * x[::-1]) / np.sum(weights[:len(x)]) 
                    if len(x) > 0 else np.nan)
            ).reset_index(level=0, drop=True)
            
            # Performance contre l'équipe spécifique (historique vs cette équipe)
            # Cette fonction sera complexe car nécessite de croiser les données par adversaire
            # Simplifié ici
            
        # Ajouter d'autres caractéristiques contextuelles
        # Jours de repos
        self.data['rest_days'] = grouped['Date'].transform(
            lambda x: (x - x.shift(1)).dt.days)
        
        # Indicateur domicile/extérieur
        self.data['is_home'] = ~self.data['Opp'].str.contains('@', na=False).astype(int)
        
        # Ajouter indicateur de back-to-back (match dos à dos)
        self.data['is_back_to_back'] = (self.data['rest_days'] <= 1).astype(int)
        
        # Ajouter indicateur de road trip (série de matchs à l'extérieur)
        # Détecte 3+ matchs consécutifs à l'extérieur
        self.data['road_trip_game'] = 0
        
        # Calculer l'utilisation du joueur (% des possessions terminées par le joueur)
        # Formule simplifiée: (FGA + 0.44*FTA + TOV) / MP
        self.data['usage_rate'] = (self.data['FGA'] + 0.44*self.data['FTA'] + self.data['TOV']) / self.data['MP']
        
        # Efficacité offensive (points par tir)
        self.data['pts_per_shot'] = self.data['PTS'] / (self.data['FGA'] + 0.44*self.data['FTA'])
        
        print("Caractéristiques avancées créées avec succès")
            
    def train_models(self, retrain=False):
        """
        Entraîne un modèle séparé pour chaque joueur et chaque statistique
        
        Args:
            retrain (bool): Si True, réentraîne même si des modèles existent déjà
        """
        os.makedirs('models', exist_ok=True)
        
        # Obtenir la liste des joueurs avec suffisamment de matchs
        player_counts = self.data['player_id'].value_counts()
        qualified_players = player_counts[player_counts >= 15].index.tolist()  # Minimum 15 matchs
        
        print(f"Entraînement des modèles pour {len(qualified_players)} joueurs qualifiés")
        
        for player_id in qualified_players:
            player_data = self.data[self.data['player_id'] == player_id].copy()
            
            print(f"\nEntraînement des modèles pour {player_id} ({len(player_data)} matchs)")
            
            # Pour chaque statistique à prédire
            for stat in self.stats_to_predict:
                model_filename = f"models/{player_id}_{stat}_model.pkl"
                
                # Vérifier si le modèle existe déjà et si on ne force pas le réentraînement
                if os.path.exists(model_filename) and not retrain:
                    continue
                
                # Définir les caractéristiques personnalisées pour cette statistique
                if stat in ['PTS', 'FG%', '3P']:
                    # Stats offensives - utiliser caractéristiques spécifiques à l'attaque
                    features = [
                        f'{stat}_avg_3', f'{stat}_avg_5', f'{stat}_avg_10', 
                        f'{stat}_std_5', f'{stat}_trend', f'{stat}_weighted',
                        'usage_rate', 'pts_per_shot', 'rest_days', 
                        'is_home', 'is_back_to_back', 'road_trip_game',
                        'Team', 'Opp'
                    ]
                elif stat in ['TRB', 'BLK']:
                    # Stats défensives
                    features = [
                        f'{stat}_avg_3', f'{stat}_avg_5', f'{stat}_avg_10', 
                        f'{stat}_std_5', f'{stat}_trend', f'{stat}_weighted',
                        'rest_days', 'is_home', 'is_back_to_back',
                        'Team', 'Opp'
                    ]
                else:
                    # Stats mixtes
                    features = [
                        f'{stat}_avg_3', f'{stat}_avg_5', f'{stat}_avg_10', 
                        f'{stat}_std_5', f'{stat}_trend', f'{stat}_weighted',
                        'usage_rate', 'rest_days', 'is_home', 
                        'is_back_to_back', 'Team', 'Opp'
                    ]
                
                # Éliminer les lignes avec des NaN dans les caractéristiques
                valid_data = player_data.dropna(subset=features + [stat])
                
                if len(valid_data) < 15:
                    print(f"  Pas assez de données pour {stat}, ignoré")
                    continue
                
                # Diviser en ensemble d'entraînement et de validation
                X = valid_data[features]
                y = valid_data[stat]
                
                if len(valid_data) > 30:  # Assez de données pour faire un split
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                else:
                    X_train, y_train = X, y
                
                # Définir les préprocesseurs pour les colonnes catégorielles
                categorical_features = ['Team', 'Opp']
                categorical_transformer = OneHotEncoder(handle_unknown='ignore')
                
                # Préprocesseur pour les colonnes numériques
                numerical_features = [f for f in features if f not in categorical_features]
                numerical_transformer = StandardScaler()
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', categorical_transformer, categorical_features),
                        ('num', numerical_transformer, numerical_features)
                    ]
                )
                
                # Utiliser GradientBoostingRegressor pour plus de précision
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=150, 
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42
                    ))
                ])
                
                # Entraîner le modèle
                try:
                    pipeline.fit(X_train, y_train)
                    
                    # Évaluer sur l'ensemble de validation si disponible
                    if len(valid_data) > 30:
                        train_score = pipeline.score(X_train, y_train)
                        val_score = pipeline.score(X_val, y_val)
                        print(f"  Modèle pour {stat}: R² train={train_score:.3f}, val={val_score:.3f}")
                    
                    # Sauvegarder le modèle
                    joblib.dump(pipeline, model_filename)
                    print(f"  Modèle pour {stat} entraîné et sauvegardé")
                    
                except Exception as e:
                    print(f"  Erreur lors de l'entraînement pour {stat}: {e}")
        
        print("\nEntraînement des modèles terminé!")
                
    def predict_next_game(self, player_id, opponent, team=None, is_home=True, rest_days=2, back_to_back=False):
        """
        Prédit les statistiques pour le prochain match d'un joueur avec plus de contexte
        
        Args:
            player_id (str): ID du joueur
            opponent (str): Équipe adverse
            team (str): Équipe du joueur (si None, utilise la dernière connue)
            is_home (bool): Si True, match à domicile, sinon à l'extérieur
            rest_days (int): Nombre de jours de repos depuis le dernier match
            back_to_back (bool): Si True, il s'agit d'un match dos à dos
            
        Returns:
            dict: Dictionnaire des statistiques prédites (arrondies en nombres entiers)
        """
        # Vérifier si des données existent pour ce joueur
        player_data = self.data[self.data['player_id'] == player_id]
        
        if len(player_data) == 0:
            return {"error": f"Aucune donnée trouvée pour le joueur {player_id}"}
        
        # Si l'équipe n'est pas spécifiée, utiliser la dernière équipe connue
        if team is None:
            team = player_data.iloc[-1]['Team']
            
        # Déterminer le format de l'adversaire basé sur is_home
        opp_format = opponent if is_home else f"@ {opponent}"
        
        # Récupérer les forces des équipes
        def_rating = self.team_defensive_ratings.get(opponent, 100)
        pace_factor = self.team_pace_factors.get(opponent, 1.0)
        
        # Préparer les prédictions
        predictions = {}
        raw_predictions = {}
        
        for stat in self.stats_to_predict:
            model_filename = f"models/{player_id}_{stat}_model.pkl"
            
            # Vérifier si le modèle existe
            if not os.path.exists(model_filename):
                # Pas de modèle, utiliser une méthode plus sophistiquée qu'une simple moyenne
                # Utiliser la moyenne pondérée des 5 derniers matchs
                last_values = player_data[stat].tail(5).values
                if len(last_values) > 0:
                    weights = np.array([0.35, 0.25, 0.2, 0.15, 0.05])[:len(last_values)]
                    weights = weights / weights.sum()
                    predictions[stat] = np.round(np.sum(last_values * weights))
                else:
                    predictions[stat] = np.round(player_data[stat].mean())
                continue
                
            # Charger le modèle
            model = joblib.load(model_filename)
            
            # Préparer les caractéristiques avancées
            X_pred = {}
            
            # Moyennes mobiles
            X_pred[f'{stat}_avg_3'] = player_data[stat].tail(3).mean()
            X_pred[f'{stat}_avg_5'] = player_data[stat].tail(5).mean()
            X_pred[f'{stat}_avg_10'] = player_data[stat].tail(10).mean()
            
            # Écart-type (mesure de consistance)
            X_pred[f'{stat}_std_5'] = player_data[stat].tail(5).std() if len(player_data) >= 5 else 0
            
            # Tendance (pente sur les 5 derniers matchs)
            if len(player_data) >= 5:
                last_5 = player_data[stat].tail(5).values
                X_pred[f'{stat}_trend'] = (last_5[-1] - last_5[0]) / 4
            else:
                X_pred[f'{stat}_trend'] = 0
            
            # Moyenne pondérée par récence
            if len(player_data) >= 3:
                last_3 = player_data[stat].tail(3).values
                weights = np.array([0.5, 0.3, 0.2])[:len(last_3)]
                weights = weights / weights.sum()
                X_pred[f'{stat}_weighted'] = np.sum(last_3[::-1] * weights)
            else:
                X_pred[f'{stat}_weighted'] = player_data[stat].mean()
            
            # Caractéristiques d'utilisation du joueur
            if 'usage_rate' in player_data.columns:
                X_pred['usage_rate'] = player_data['usage_rate'].tail(5).mean()
            else:
                X_pred['usage_rate'] = 0.2  # Valeur par défaut
                
            if 'pts_per_shot' in player_data.columns:
                X_pred['pts_per_shot'] = player_data['pts_per_shot'].tail(5).mean()
            else:
                X_pred['pts_per_shot'] = 1.0  # Valeur par défaut
            
            # Variables contextuelles du match
            X_pred['rest_days'] = rest_days
            X_pred['is_home'] = 1 if is_home else 0
            X_pred['is_back_to_back'] = 1 if back_to_back else 0
            X_pred['road_trip_game'] = 0  # Simplifié
            X_pred['Team'] = team
            X_pred['Opp'] = opp_format
            
            # Créer le DataFrame pour la prédiction
            X_df = pd.DataFrame({k: [v] for k, v in X_pred.items()})
            
            # Faire la prédiction
            try:
                # Prédiction brute du modèle
                raw_pred = model.predict(X_df)[0]
                raw_predictions[stat] = raw_pred
                
                # Ajuster la prédiction en fonction du contexte
                adjusted_pred = raw_pred
                
                # Ajuster en fonction de la force défensive de l'adversaire
                if stat in ['PTS', '3P', 'AST']:
                    # Plus la cote défensive est élevée, plus c'est difficile de marquer
                    defensive_factor = 2 - (def_rating / 100)  # Normaliser autour de 1
                    adjusted_pred *= defensive_factor
                
                # Ajuster en fonction du rythme de jeu
                if stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']:
                    # Plus le rythme est élevé, plus il y a d'opportunités statistiques
                    adjusted_pred *= pace_factor
                
                # Ajuster pour les matchs à domicile/extérieur
                if not is_home and stat in ['PTS', '3P', 'FG%']:
                    # Performance légèrement réduite à l'extérieur
                    adjusted_pred *= 0.95
                
                # Ajuster pour les back-to-backs
                if back_to_back and stat in ['PTS', 'MP']:
                    # Performance et minutes réduites en back-to-back
                    adjusted_pred *= 0.9
                
                # Appliquer des contraintes spécifiques à chaque statistique et arrondir
                if stat in ['FG%', '3P%', 'FT%']:
                    # Pourcentages entre 0 et 1
                    adjusted_pred = max(0, min(1, adjusted_pred))
                    # Pas d'arrondi pour les pourcentages
                    predictions[stat] = adjusted_pred
                elif stat == 'MP':
                    # Minutes de jeu (arrondir à l'entier le plus proche)
                    adjusted_pred = max(0, adjusted_pred)
                    predictions[stat] = round(adjusted_pred)
                elif stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'TOV']:
                    # Stats de comptage (entiers positifs)
                    adjusted_pred = max(0, adjusted_pred)
                    predictions[stat] = round(adjusted_pred)  # Arrondir à l'entier
                else:
                    # Autres statistiques
                    predictions[stat] = round(adjusted_pred)
                
            except Exception as e:
                print(f"Erreur lors de la prédiction pour {stat}: {e}")
                # Utiliser une moyenne pondérée en cas d'erreur
                last_values = player_data[stat].tail(5).values
                if len(last_values) > 0:
                    weights = np.array([0.35, 0.25, 0.2, 0.15, 0.05])[:len(last_values)]
                    weights = weights / weights.sum()
                    predictions[stat] = round(np.sum(last_values * weights))
                else:
                    predictions[stat] = round(player_data[stat].mean())
        
        return predictions
    
    def get_player_info(self, player_id):
        """
        Récupère les informations et statistiques d'un joueur avec analyse améliorée
        
        Args:
            player_id (str): ID du joueur
            
        Returns:
            dict: Informations détaillées sur le joueur
        """
        player_data = self.data[self.data['player_id'] == player_id]
        
        if len(player_data) == 0:
            return {"error": f"Joueur {player_id} non trouvé"}
        
        # Obtenir la dernière équipe connue
        latest_team = player_data.iloc[-1]['Team']
        
        # Calculer les moyennes de la saison
        season_avgs = {
            stat: player_data[stat].mean() for stat in self.stats_to_predict
        }
        
        # Calculer les moyennes des 5 derniers matchs
        last_5_avgs = {
            stat: player_data[stat].tail(5).mean() for stat in self.stats_to_predict
        }
        
        # Calculer la tendance (différence entre moy 5 derniers et moy saison)
        trend = {
            stat: last_5_avgs[stat] - season_avgs[stat] for stat in self.stats_to_predict
        }
        
        # Calculer les écarts-types (consistance)
        consistency = {
            stat: player_data[stat].std() for stat in self.stats_to_predict
        }
        
        # Statistiques à domicile vs extérieur
        home_data = player_data[~player_data['Opp'].str.contains('@', na=False)]
        away_data = player_data[player_data['Opp'].str.contains('@', na=False)]
        
        home_away_splits = {}
        for stat in self.stats_to_predict:
            home_away_splits[stat] = {
                'home': home_data[stat].mean() if len(home_data) > 0 else None,
                'away': away_data[stat].mean() if len(away_data) > 0 else None
            }
        
        # Obtenir le dernier match
        last_game = player_data.iloc[-1].to_dict()
        
        # Informations sur les performances contre chaque adversaire
        opp_performances = {}
        for opp in player_data['Opp'].str.replace('@ ', '').unique():
            opp_data = player_data[player_data['Opp'].str.contains(opp)]
            if len(opp_data) >= 2:  # Au moins 2 matchs contre cet adversaire
                opp_performances[opp] = {
                    stat: opp_data[stat].mean() for stat in ['PTS', 'TRB', 'AST']
                }
        
        return {
            "player_id": player_id,
            "team": latest_team,
            "games_played": len(player_data),
            "season_averages": season_avgs,
            "last_5_games_averages": last_5_avgs,
            "trend": trend,
            "consistency": consistency,
            "home_away_splits": home_away_splits,
            "opponent_performances": opp_performances,
            "last_game": last_game
        }
        
    def list_all_players(self):
        """
        Liste tous les joueurs dans le dataset avec leur nombre de matchs
        
        Returns:
            dict: Dictionnaire avec les IDs des joueurs et leur nombre de matchs
        """
        player_counts = self.data['player_id'].value_counts().to_dict()
        return player_counts
        
    def predict_for_all_players(self, opponent, team, is_home=True, rest_days=2, back_to_back=False):
        """
        Prédit les statistiques pour tous les joueurs d'une équipe contre un adversaire donné
        
        Args:
            opponent (str): Équipe adverse
            team (str): Équipe des joueurs
            is_home (bool): Si True, match à domicile
            rest_days (int): Nombre de jours de repos
            back_to_back (bool): Si True, match dos à dos
            
        Returns:
            dict: Dictionnaire des prédictions par joueur
        """
        all_predictions = {}
        
        # Obtenir la liste des joueurs de l'équipe
        team_players = self.data[self.data['Team'] == team]['player_id'].unique()
        
        for player_id in team_players:
            # Vérifier si le joueur a suffisamment de données
            player_data = self.data[self.data['player_id'] == player_id]
            if len(player_data) < 10:
                continue
                
            # Prédire les stats du joueur
            prediction = self.predict_next_game(
                player_id, opponent, team, is_home, rest_days, back_to_back
            )
            
            # Calculer le temps de jeu moyen récent
            recent_mp = player_data['MP'].tail(10).mean()
            
            # Ne conserver que les joueurs avec un temps de jeu significatif
            if recent_mp >= 10:
                all_predictions[player_id] = {
                    "team": team,
                    "predictions": prediction,
                    "avg_minutes": recent_mp
                }
        
        return all_predictions
        
    def compare_predictions_with_actual(self, player_id, last_n_games=5):
        """
        Compare les prédictions avec les résultats réels pour les derniers matchs
        
        Args:
            player_id (str): ID du joueur
            last_n_games (int): Nombre de derniers matchs à comparer
            
        Returns:
            dict: Comparaison détaillée des prédictions vs résultats réels
        """
        player_data = self.data[self.data['player_id'] == player_id].copy()
        
        if len(player_data) <= last_n_games:
            return {"error": f"Pas assez de données pour {player_id}"}
            
        # Prendre les n derniers matchs
        test_games = player_data.tail(last_n_games).copy()
        
        results = []
        
        for i, (_, game) in enumerate(test_games.iterrows()):
            # Récupérer les données jusqu'à ce match (exclu)
            historical_data = player_data.iloc[:-last_n_games+i]
            
            # Déterminer si c'est un match à domicile
            is_home = 1 if '@' not in str(game['Opp']) else 0
            
            # Déterminer l'adversaire
            opponent = game['Opp'].replace('@ ', '') if '@' in str(game['Opp']) else game['Opp']
            
            # Déterminer les jours de repos
            rest_days = game['rest_days'] if 'rest_days' in game else 2
            
            # Déterminer si c'est un back-to-back
            is_back_to_back = game['is_back_to_back'] if 'is_back_to_back' in game else False
            
            # Prédire pour chaque statistique
            predictions = {}
            for stat in self.stats_to_predict:
                model_filename = f"models/{player_id}_{stat}_model.pkl"
                
                if not os.path.exists(model_filename):
                    # Pas de modèle, utiliser une moyenne pondérée des données historiques
                    if len(historical_data) >= 5:
                        weights = np.array([0.35, 0.25, 0.2, 0.15, 0.05])
                        last_values = historical_data[stat].tail(5).values
                        predictions[stat] = np.round(np.sum(weights[:len(last_values)] * last_values[::-1]))
                    else:
                        predictions[stat] = np.round(historical_data[stat].mean())
                    continue
                    
                try:
                    model = joblib.load(model_filename)
                    
                    # Préparer les caractéristiques pour la prédiction
                    X_pred = {}
                    
                    # Moyennes mobiles
                    X_pred[f'{stat}_avg_3'] = historical_data[stat].tail(3).mean() if len(historical_data) >= 3 else historical_data[stat].mean()
                    X_pred[f'{stat}_avg_5'] = historical_data[stat].tail(5).mean() if len(historical_data) >= 5 else historical_data[stat].mean()
                    X_pred[f'{stat}_avg_10'] = historical_data[stat].tail(10).mean() if len(historical_data) >= 10 else historical_data[stat].mean()
                    
                    # Écart-type
                    X_pred[f'{stat}_std_5'] = historical_data[stat].tail(5).std() if len(historical_data) >= 5 else 0
                    
                    # Tendance
                    if len(historical_data) >= 5:
                        last_5 = historical_data[stat].tail(5).values
                        X_pred[f'{stat}_trend'] = (last_5[-1] - last_5[0]) / 4
                    else:
                        X_pred[f'{stat}_trend'] = 0
                    
                    # Moyenne pondérée
                    if len(historical_data) >= 3:
                        last_3 = historical_data[stat].tail(3).values
                        weights = np.array([0.5, 0.3, 0.2])[:len(last_3)]
                        weights = weights / weights.sum()
                        X_pred[f'{stat}_weighted'] = np.sum(last_3[::-1] * weights)
                    else:
                        X_pred[f'{stat}_weighted'] = historical_data[stat].mean()
                    
                    # Variables d'utilisation
                    if 'usage_rate' in historical_data.columns:
                        X_pred['usage_rate'] = historical_data['usage_rate'].tail(5).mean() if len(historical_data) >= 5 else 0.2
                    else:
                        X_pred['usage_rate'] = 0.2
                        
                    if 'pts_per_shot' in historical_data.columns:
                        X_pred['pts_per_shot'] = historical_data['pts_per_shot'].tail(5).mean() if len(historical_data) >= 5 else 1.0
                    else:
                        X_pred['pts_per_shot'] = 1.0
                    
                    # Variables contextuelles
                    X_pred['rest_days'] = rest_days
                    X_pred['is_home'] = is_home
                    X_pred['is_back_to_back'] = 1 if is_back_to_back else 0
                    X_pred['road_trip_game'] = 0  # Simplifié
                    X_pred['Team'] = game['Team']
                    X_pred['Opp'] = game['Opp']
                    
                    # Prédiction
                    X_df = pd.DataFrame({k: [v] for k, v in X_pred.items()})
                    pred = model.predict(X_df)[0]
                    
                    # Arrondir à l'entier pour les statistiques de comptage
                    if stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'TOV', 'MP']:
                        predictions[stat] = round(max(0, pred))
                    else:
                        predictions[stat] = pred
                        
                except Exception as e:
                    print(f"Erreur lors de la prédiction pour {stat}: {e}")
                    predictions[stat] = round(historical_data[stat].mean())
            
            # Comparer avec les résultats réels
            actuals = {stat: game[stat] for stat in self.stats_to_predict}
            
            # Calculer les erreurs
            errors = {stat: abs(predictions[stat] - actuals[stat]) for stat in self.stats_to_predict}
            
            # Calculer les pourcentages d'erreur pour les statistiques importantes
            pct_errors = {}
            for stat in ['PTS', 'TRB', 'AST']:
                if actuals[stat] > 0:
                    pct_errors[stat] = (errors[stat] / actuals[stat]) * 100
                else:
                    pct_errors[stat] = 0
            
            results.append({
                "date": game['Date'],
                "opponent": game['Opp'],
                "predictions": predictions,
                "actuals": actuals,
                "errors": errors,
                "pct_errors": pct_errors
            })
            
        # Calculer les erreurs moyennes
        avg_errors = {
            stat: sum(r["errors"][stat] for r in results) / len(results) 
            for stat in self.stats_to_predict
        }
        
        # Calculer les pourcentages d'erreur moyens
        avg_pct_errors = {}
        for stat in ['PTS', 'TRB', 'AST']:
            valid_errors = [r["pct_errors"][stat] for r in results if stat in r["pct_errors"]]
            if valid_errors:
                avg_pct_errors[stat] = sum(valid_errors) / len(valid_errors)
            else:
                avg_pct_errors[stat] = None
        
        return {
            "player_id": player_id,
            "comparison_results": results,
            "average_errors": avg_errors,
            "average_pct_errors": avg_pct_errors
        }
    
    def generate_matchup_analysis(self, team, opponent, is_home=True):
        """
        Génère une analyse détaillée d'un match entre deux équipes
        
        Args:
            team (str): Équipe principale
            opponent (str): Équipe adverse
            is_home (bool): Si True, l'équipe principale joue à domicile
            
        Returns:
            dict: Analyse complète du match avec prédictions de tous les joueurs
        """
        # Prédire les stats pour tous les joueurs de l'équipe
        team_predictions = self.predict_for_all_players(opponent, team, is_home)
        
        # Prédire les stats pour tous les joueurs de l'équipe adverse
        opp_predictions = self.predict_for_all_players(team, opponent, not is_home)
        
        # Calculer les totaux d'équipe
        team_totals = {
            'PTS': sum(p['predictions']['PTS'] for p in team_predictions.values()),
            'TRB': sum(p['predictions']['TRB'] for p in team_predictions.values()),
            'AST': sum(p['predictions']['AST'] for p in team_predictions.values()),
            'STL': sum(p['predictions']['STL'] for p in team_predictions.values()),
            'BLK': sum(p['predictions']['BLK'] for p in team_predictions.values()),
            '3P': sum(p['predictions']['3P'] for p in team_predictions.values()),
            'TOV': sum(p['predictions']['TOV'] for p in team_predictions.values())
        }
        
        opp_totals = {
            'PTS': sum(p['predictions']['PTS'] for p in opp_predictions.values()),
            'TRB': sum(p['predictions']['TRB'] for p in opp_predictions.values()),
            'AST': sum(p['predictions']['AST'] for p in opp_predictions.values()),
            'STL': sum(p['predictions']['STL'] for p in opp_predictions.values()),
            'BLK': sum(p['predictions']['BLK'] for p in opp_predictions.values()),
            '3P': sum(p['predictions']['3P'] for p in opp_predictions.values()),
            'TOV': sum(p['predictions']['TOV'] for p in opp_predictions.values())
        }
        
        # Identifier les joueurs clés (top 3 scores prédits)
        team_key_players = sorted(
            [(pid, p['predictions']['PTS']) for pid, p in team_predictions.items()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        opp_key_players = sorted(
            [(pid, p['predictions']['PTS']) for pid, p in opp_predictions.items()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        return {
            "team": team,
            "opponent": opponent,
            "is_home": is_home,
            "team_players": team_predictions,
            "opponent_players": opp_predictions,
            "team_totals": team_totals,
            "opponent_totals": opp_totals,
            "team_key_players": team_key_players, 
            "opponent_key_players": opp_key_players,
            "predicted_winner": team if team_totals['PTS'] > opp_totals['PTS'] else opponent,
            "predicted_score": f"{round(team_totals['PTS'])}-{round(opp_totals['PTS'])}"
        }

def main():
    """
    Fonction principale pour démontrer l'utilisation de l'outil amélioré
    """
    print("=== Prédicteur de Statistiques NBA Amélioré ===")
    
    # Initialiser le prédicteur
    predictor = NBAStatsPredictor()
    
    # Entrainer les modèles
    print("\nEntrainement des modèles...")
    predictor.train_models()
    
    # Exemple: prédire pour un joueur spécifique
    player_id = "j/jamesle01"  # LeBron James
    opponent = "BOS"  # Boston Celtics
    is_home = True
    
    print(f"\nPrédiction pour {player_id} contre {opponent} (à domicile: {is_home})")
    prediction = predictor.predict_next_game(player_id, opponent, is_home=is_home)
    
    # Afficher les résultats (entiers)
    print("\nStatistiques prédites:")
    for stat, value in prediction.items():
        if stat in ['MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', '3P']:
            print(f"  {stat}: {int(value)}")  # Affichage en entiers
        elif stat in ['FG%']:
            print(f"  {stat}: {value:.3f}")
    
    # Comparer les prédictions avec les résultats réels
    print("\nComparaison des prédictions avec les résultats réels:")
    comparison = predictor.compare_predictions_with_actual(player_id, last_n_games=3)
    
    if "error" not in comparison:
        print("  Erreurs moyennes:")
        for stat, error in comparison["average_errors"].items():
            if stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
                print(f"    {stat}: {error:.2f}")
        
        if "average_pct_errors" in comparison:
            print("  Pourcentages d'erreur moyens:")
            for stat, pct_error in comparison["average_pct_errors"].items():
                if pct_error is not None:
                    print(f"    {stat}: {pct_error:.1f}%")
    
    # Générer une analyse de match
    print("\nAnalyse de match:")
    matchup = predictor.generate_matchup_analysis("LAL", "BOS", True)
    
    print(f"  Score prédit: {matchup['predicted_score']} (Gagnant: {matchup['predicted_winner']})")
    print("  Joueurs clés LAL:")
    for player_id, points in matchup['team_key_players']:
        print(f"    {player_id}: {int(points)} PTS")
    
    print("  Joueurs clés BOS:")
    for player_id, points in matchup['opponent_key_players']:
        print(f"    {player_id}: {int(points)} PTS")
    
    print("\nTerminé!")

if __name__ == "__main__":
    main()