import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
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
            
            # Trier par joueur et date - ESSENTIEL pour éviter la fuite de données
            self.data = self.data.sort_values(['player_id', 'Date'])
            
            # Ajouter indicateur domicile/extérieur
            self.data['is_home'] = self.data['Result'].apply(
            lambda x: 0 if isinstance(x, str) and x.split(',')[0] == 'W' and 
                         int(x.split(' ')[1].split('-')[0]) > int(x.split(' ')[1].split('-')[1])
                       else 1  # Par défaut, considérer comme à domicile
        )
            
            # Ajouter jours de repos - ATTENTION: ne pas utiliser diff() directement car peut causer des fuites
            self.data['rest_days'] = self.data.groupby('player_id')['Date'].diff().dt.days.fillna(3)
            
            # Ajouter indicateur de back-to-back (match dos à dos)
            self.data['is_back_to_back'] = (self.data['rest_days'] <= 1).astype(int)
            
            # Calculer l'utilisation du joueur (% des possessions terminées par le joueur)
            # Formule simplifiée: (FGA + 0.44*FTA + TOV) / MP
            self.data['usage_rate'] = (self.data['FGA'] + 0.44*self.data['FTA'] + self.data['TOV']) / self.data['MP']
            
            # Efficacité offensive (points par tir)
            self.data['pts_per_shot'] = self.data['PTS'] / (self.data['FGA'] + 0.44*self.data['FTA'])
            
            print(f"Préparation des données terminée. {len(self.data)} matchs après nettoyage.")
            
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
    
    def _calculate_team_strength(self, train_data_only):
        """
        Calcule des métriques de force pour chaque équipe, UNIQUEMENT avec les données d'entraînement
        
        Args:
            train_data_only (pd.DataFrame): Données d'entraînement uniquement
        """
        # Calcul de la cote défensive de chaque équipe (points encaissés/100 possessions)
        team_games = train_data_only.groupby(['Date', 'Opp']).agg({'PTS': 'sum'}).reset_index()
        team_games.columns = ['Date', 'Team', 'PTS_Against']
        
        # Moyenne des points encaissés par match pour chaque équipe
        team_def = team_games.groupby('Team').agg({'PTS_Against': 'mean'})
        
        # Normalisation (100 = moyenne de la ligue)
        avg_pts = team_def['PTS_Against'].mean()
        for team in team_def.index:
            self.team_defensive_ratings[team] = 100 * (avg_pts / team_def.loc[team, 'PTS_Against'])

        # Calcul du rythme de jeu de chaque équipe (estimé par les possessions)
        team_pace = train_data_only.groupby('Team').agg({'FGA': 'mean', 'TOV': 'mean'})
        avg_pace = team_pace.mean().sum()
        
        for team in team_pace.index:
            pace_value = (team_pace.loc[team, 'FGA'] + team_pace.loc[team, 'TOV']) / avg_pace
            self.team_pace_factors[team] = pace_value
            
    def _create_features_for_player(self, player_data, for_prediction=False):
        """
        Crée des caractéristiques avancées pour un joueur donné, en respectant la temporalité
        
        Args:
            player_data (pd.DataFrame): Données d'un joueur spécifique, triées par date
            for_prediction (bool): Si True, crée les caractéristiques pour la prédiction
                                   (en utilisant toutes les données)
            
        Returns:
            pd.DataFrame: Données avec caractéristiques ajoutées
        """
        result = player_data.copy()
        
        # Pour chaque statistique à prédire
        for stat in self.stats_to_predict:
            # Moyennes mobiles standard (3, 5, 10 matchs)
            # IMPORTANT: Utiliser shift(1) pour éviter la fuite de données
            result[f'{stat}_avg_3'] = result[stat].shift(1).rolling(window=3, min_periods=1).mean()
            result[f'{stat}_avg_5'] = result[stat].shift(1).rolling(window=5, min_periods=1).mean()
            result[f'{stat}_avg_10'] = result[stat].shift(1).rolling(window=10, min_periods=1).mean()
            
            # Écart-type sur les derniers matchs (mesure de consistance)
            result[f'{stat}_std_5'] = result[stat].shift(1).rolling(window=5, min_periods=2).std()
            
            # Tendance linéaire (pente sur les 5 derniers matchs)
            result[f'{stat}_trend'] = (result[stat].shift(1) - result[stat].shift(5)) / 4
            
            # Performances pondérées par récence (plus de poids aux matchs récents)
            def weighted_avg(x):
                if len(x) == 0:
                    return np.nan
                weights = np.array([0.5, 0.3, 0.2])[:len(x)]
                weights = weights / weights.sum()
                return np.sum(weights * x[::-1])
                
            result[f'{stat}_weighted'] = result[stat].shift(1).rolling(window=3, min_periods=1).apply(
                weighted_avg, raw=True)
        
        return result
            
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
                    base_features = [
                        'usage_rate', 'pts_per_shot', 'rest_days', 
                        'is_home', 'is_back_to_back', 'Team', 'Opp'
                    ]
                elif stat in ['TRB', 'BLK']:
                    # Stats défensives
                    base_features = [
                        'rest_days', 'is_home', 'is_back_to_back',
                        'Team', 'Opp'
                    ]
                else:
                    # Stats mixtes
                    base_features = [
                        'usage_rate', 'rest_days', 'is_home', 
                        'is_back_to_back', 'Team', 'Opp'
                    ]
                
                # Utiliser TimeSeriesSplit pour respecter la temporalité des données
                tscv = TimeSeriesSplit(n_splits=5, test_size=3)
                
                best_model = None
                best_score = -np.inf
                
                # IMPORTANT: Pour éviter les fuites de données, on traite chaque fold séparément
                # TimeSeriesSplit garantit que les données d'entrainement précèdent les données de test
                for train_index, test_index in tscv.split(player_data):
                    # Séparer en ensembles d'entraînement et de test
                    train_data = player_data.iloc[train_index].copy()
                    test_data = player_data.iloc[test_index].copy()
                    
                    if len(train_data) < 10:  # Pas assez de données pour entraîner
                        continue
                        
                    # CORRECTION: Créer les caractéristiques séparément pour les données d'entraînement
                    # et les données de test pour éviter la fuite
                    train_data_with_features = self._create_features_for_player(train_data)
                    
                    # Calculer les forces des équipes UNIQUEMENT avec les données d'entraînement
                    self._calculate_team_strength(train_data_with_features)
                    
                    # CORRECTION: Pour les données de test, créons les caractéristiques avec 
                    # seulement les données jusqu'à la date du premier point de test
                    test_start_date = test_data['Date'].min()
                    
                    # Filtrer les données jusqu'à la date du premier point de test
                    historical_data_for_test = player_data[player_data['Date'] < test_start_date].copy()
                    
                    # Ajouter les données de test à la fin
                    combined_data = pd.concat([historical_data_for_test, test_data]).sort_values('Date')
                    
                    # Créer les caractéristiques pour les données combinées
                    combined_data_with_features = self._create_features_for_player(combined_data)
                    
                    # Extraire uniquement les lignes de test
                    test_data_with_features = combined_data_with_features[
                        combined_data_with_features['Date'] >= test_start_date
                    ].copy()
                    
                    # Définir les caractéristiques avancées pour cette statistique
                    full_features = base_features.copy()
                    for f in [f'{stat}_avg_3', f'{stat}_avg_5', f'{stat}_avg_10', 
                              f'{stat}_std_5', f'{stat}_trend', f'{stat}_weighted']:
                        if f in train_data_with_features.columns:
                            full_features.append(f)
                    
                    # Éliminer les lignes avec des NaN dans les caractéristiques
                    valid_train = train_data_with_features.dropna(subset=full_features + [stat])
                    valid_test = test_data_with_features.dropna(subset=full_features + [stat])
                    
                    if len(valid_train) < 10:  # Pas assez de données pour entraîner
                        continue
                    
                    # Diviser en X et y
                    X_train = valid_train[full_features]
                    y_train = valid_train[stat]
                    X_test = valid_test[full_features]
                    y_test = valid_test[stat]
                    
                    # Définir les préprocesseurs pour les colonnes catégorielles
                    categorical_features = [f for f in full_features if f in ['Team', 'Opp']]
                    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
                    
                    # Préprocesseur pour les colonnes numériques
                    numerical_features = [f for f in full_features if f not in categorical_features]
                    numerical_transformer = StandardScaler()
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat', categorical_transformer, categorical_features),
                            ('num', numerical_transformer, numerical_features)
                        ],
                        remainder='passthrough'
                    )
                    
                    # Utiliser GradientBoostingRegressor pour plus de précision
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(
                            n_estimators=150, 
                            learning_rate=0.05,
                            max_depth=4,
                            random_state=42
                        ))
                    ])
                    
                    # Entraîner le modèle
                    try:
                        pipeline.fit(X_train, y_train)
                        
                        # Évaluer sur l'ensemble de test
                        test_score = pipeline.score(X_test, y_test)
                        train_score = pipeline.score(X_train, y_train)
                        
                        print(f"  Fold pour {stat}: R² train={train_score:.3f}, test={test_score:.3f}")
                        
                        # Garder le meilleur modèle
                        if test_score > best_score:
                            best_score = test_score
                            best_model = pipeline
                        
                    except Exception as e:
                        print(f"  Erreur lors de l'entraînement pour {stat}: {e}")
                
                # Sauvegarder le meilleur modèle
                if best_model is not None:
                    joblib.dump(best_model, model_filename)
                    print(f"  Meilleur modèle pour {stat} entraîné et sauvegardé (R²={best_score:.3f})")
                else:
                    print(f"  Pas de modèle valide pour {stat}, ignoré")
        
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
        
        # Créer des caractéristiques à jour pour ce joueur
        player_data_with_features = self._create_features_for_player(player_data, for_prediction=True)
        
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
            
            # Utiliser les caractéristiques déjà calculées
            last_row = player_data_with_features.iloc[-1]
            
            # Moyennes mobiles
            X_pred[f'{stat}_avg_3'] = last_row[f'{stat}_avg_3']
            X_pred[f'{stat}_avg_5'] = last_row[f'{stat}_avg_5']
            X_pred[f'{stat}_avg_10'] = last_row[f'{stat}_avg_10']
            
            # Écart-type (mesure de consistance)
            X_pred[f'{stat}_std_5'] = last_row[f'{stat}_std_5']
            
            # Tendance (pente sur les 5 derniers matchs)
            X_pred[f'{stat}_trend'] = last_row[f'{stat}_trend']
            
            # Moyenne pondérée par récence
            X_pred[f'{stat}_weighted'] = last_row[f'{stat}_weighted']
            
            # Caractéristiques d'utilisation du joueur
            X_pred['usage_rate'] = last_row['usage_rate']
            
            if 'pts_per_shot' in player_data_with_features.columns:
                X_pred['pts_per_shot'] = last_row['pts_per_shot']
            
            # Variables contextuelles du match
            X_pred['rest_days'] = rest_days
            X_pred['is_home'] = 1 if is_home else 0
            X_pred['is_back_to_back'] = 1 if back_to_back else 0
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
        Prédit les statistiques pour tous les joueurs d'une équipe contre un adversaire donné,
        en tenant compte des transferts et de l'équipe actuelle des joueurs
        
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
        
        # Obtenir la date actuelle (celle de la prédiction)
        # Pour une simulation, on peut utiliser la date du dernier match dans les données
        current_date = self.data['Date'].max()
        
        # Filtrer les joueurs actuellement dans l'équipe
        # En prenant l'équipe la plus récente pour chaque joueur
        player_current_teams = {}
        
        # Trouver l'équipe actuelle pour chaque joueur
        for player_id in self.data['player_id'].unique():
            player_games = self.data[self.data['player_id'] == player_id].sort_values('Date')
            if len(player_games) > 0:
                # Prendre l'équipe du dernier match joué
                last_game = player_games.iloc[-1]
                player_current_teams[player_id] = last_game['Team']
        
        # Filtrer seulement les joueurs qui sont actuellement dans l'équipe demandée
        team_players = [pid for pid, current_team in player_current_teams.items() 
                        if current_team == team]
        
        # S'il n'y a pas assez de joueurs trouvés (au moins 8), c'est peut-être un problème de données
        if len(team_players) < 8:
            print(f"Attention: Seulement {len(team_players)} joueurs trouvés pour {team}.")
        
        for player_id in team_players:
            # Vérifier si le joueur a suffisamment de données
            player_data = self.data[self.data['player_id'] == player_id]
            if len(player_data) < 10:
                continue
                
            # Prédire les stats du joueur
            prediction = self.predict_next_game(
                player_id, opponent, team, is_home, rest_days, back_to_back
            )
            
            # Calculer le temps de jeu moyen récent (des 5 derniers matchs avec l'équipe actuelle)
            recent_player_data = player_data[player_data['Team'] == team].tail(5)
            if len(recent_player_data) > 0:
                recent_mp = recent_player_data['MP'].mean()
            else:
                # Si pas de données récentes avec cette équipe, utiliser les 3 derniers matchs peu importe l'équipe
                recent_mp = player_data.tail(3)['MP'].mean()
            
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
            # IMPORTANT: Utiliser seulement les données disponibles jusqu'à ce point dans le temps
            # Pour éviter la fuite de données temporelles
            cutoff_date = game['Date']
            historical_data = player_data[player_data['Date'] < cutoff_date].copy()
            
            # S'assurer que nous avons suffisamment de données historiques
            if len(historical_data) < 10:
                continue
                
            # Déterminer si c'est un match à domicile
            is_home = 1 if '@' not in str(game['Opp']) else 0
            
            # Déterminer l'adversaire
            opponent = game['Opp'].replace('@ ', '') if '@' in str(game['Opp']) else game['Opp']
            
            # Déterminer les jours de repos
            rest_days = game['rest_days'] if 'rest_days' in game else 2
            
            # Déterminer si c'est un back-to-back
            is_back_to_back = game['is_back_to_back'] if 'is_back_to_back' in game else False
            
            # Créer les caractéristiques pour les données historiques
            historical_data_with_features = self._create_features_for_player(historical_data)
            
            # Calculer les forces des équipes avec les données historiques
            self._calculate_team_strength(historical_data_with_features)
            
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
                    
                    # Utiliser les dernières valeurs des données historiques
                    last_row = historical_data_with_features.iloc[-1]
                    
                    # Moyennes mobiles
                    X_pred[f'{stat}_avg_3'] = last_row[f'{stat}_avg_3'] if f'{stat}_avg_3' in last_row else historical_data[stat].tail(3).mean()
                    X_pred[f'{stat}_avg_5'] = last_row[f'{stat}_avg_5'] if f'{stat}_avg_5' in last_row else historical_data[stat].tail(5).mean()
                    X_pred[f'{stat}_avg_10'] = last_row[f'{stat}_avg_10'] if f'{stat}_avg_10' in last_row else historical_data[stat].tail(10).mean()
                    
                    # Écart-type
                    X_pred[f'{stat}_std_5'] = last_row[f'{stat}_std_5'] if f'{stat}_std_5' in last_row else historical_data[stat].tail(5).std()
                    
                    # Tendance
                    X_pred[f'{stat}_trend'] = last_row[f'{stat}_trend'] if f'{stat}_trend' in last_row else 0
                    
                    # Moyenne pondérée
                    X_pred[f'{stat}_weighted'] = last_row[f'{stat}_weighted'] if f'{stat}_weighted' in last_row else historical_data[stat].tail(3).mean()
                    
                    # Variables d'utilisation
                    if 'usage_rate' in historical_data_with_features.columns:
                        X_pred['usage_rate'] = last_row['usage_rate']
                    else:
                        X_pred['usage_rate'] = 0.2
                        
                    if 'pts_per_shot' in historical_data_with_features.columns:
                        X_pred['pts_per_shot'] = last_row['pts_per_shot']
                    else:
                        X_pred['pts_per_shot'] = 1.0
                    
                    # Variables contextuelles
                    X_pred['rest_days'] = rest_days
                    X_pred['is_home'] = is_home
                    X_pred['is_back_to_back'] = 1 if is_back_to_back else 0
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
        Génère une analyse détaillée d'un match entre deux équipes avec des scores plus réalistes
        et des minutes cohérentes
        
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
        
        # Normaliser les minutes pour qu'elles totalisent 240 minutes par équipe (5 joueurs × 48 minutes)
        TOTAL_TEAM_MINUTES = 240
        
        # Normaliser les minutes pour l'équipe principale
        total_team_minutes = sum(p['predictions']['MP'] for p in team_predictions.values())
        if total_team_minutes > 0:  # Éviter la division par zéro
            minutes_scale_factor = TOTAL_TEAM_MINUTES / total_team_minutes
            
            # Appliquer le facteur d'échelle aux minutes et ajuster les autres stats proportionnellement
            for player_id, player_data in team_predictions.items():
                original_minutes = player_data['predictions']['MP']
                scaled_minutes = original_minutes * minutes_scale_factor
                
                # Limiter les minutes à 48 par joueur maximum (cas de joueurs avec trop de minutes)
                if scaled_minutes > 48:
                    scaled_minutes = 48
                    
                # Calculer le facteur d'ajustement pour les autres stats basé sur la modification des minutes
                if original_minutes > 0:  # Éviter la division par zéro
                    stat_scale_factor = scaled_minutes / original_minutes
                    
                    # Appliquer le facteur aux autres stats liées au temps de jeu
                    for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'TOV']:
                        player_data['predictions'][stat] *= stat_scale_factor
                    
                    # Mettre à jour les minutes
                    player_data['predictions']['MP'] = scaled_minutes
        
        # Faire de même pour l'équipe adverse
        total_opp_minutes = sum(p['predictions']['MP'] for p in opp_predictions.values())
        if total_opp_minutes > 0:
            minutes_scale_factor = TOTAL_TEAM_MINUTES / total_opp_minutes
            
            for player_id, player_data in opp_predictions.items():
                original_minutes = player_data['predictions']['MP']
                scaled_minutes = original_minutes * minutes_scale_factor
                
                if scaled_minutes > 48:
                    scaled_minutes = 48
                    
                if original_minutes > 0:
                    stat_scale_factor = scaled_minutes / original_minutes
                    
                    for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', 'TOV']:
                        player_data['predictions'][stat] *= stat_scale_factor
                    
                    player_data['predictions']['MP'] = scaled_minutes
        
        # Calculer les totaux d'équipe ajustés
        team_totals = {
            'PTS': sum(p['predictions']['PTS'] for p in team_predictions.values()),
            'TRB': sum(p['predictions']['TRB'] for p in team_predictions.values()),
            'AST': sum(p['predictions']['AST'] for p in team_predictions.values()),
            'STL': sum(p['predictions']['STL'] for p in team_predictions.values()),
            'BLK': sum(p['predictions']['BLK'] for p in team_predictions.values()),
            '3P': sum(p['predictions']['3P'] for p in team_predictions.values()),
            'TOV': sum(p['predictions']['TOV'] for p in team_predictions.values()),
            'MP': sum(p['predictions']['MP'] for p in team_predictions.values())
        }
        
        opp_totals = {
            'PTS': sum(p['predictions']['PTS'] for p in opp_predictions.values()),
            'TRB': sum(p['predictions']['TRB'] for p in opp_predictions.values()),
            'AST': sum(p['predictions']['AST'] for p in opp_predictions.values()),
            'STL': sum(p['predictions']['STL'] for p in opp_predictions.values()),
            'BLK': sum(p['predictions']['BLK'] for p in opp_predictions.values()),
            '3P': sum(p['predictions']['3P'] for p in opp_predictions.values()),
            'TOV': sum(p['predictions']['TOV'] for p in opp_predictions.values()),
            'MP': sum(p['predictions']['MP'] for p in opp_predictions.values())
        }
        
        # Ajuster les scores pour qu'ils soient réalistes
        # Les équipes NBA marquent généralement entre 85 et 135 points par match
        MIN_SCORE = 85
        MAX_SCORE = 135
        
        # Facteurs d'ajustement pour équilibrer les prédictions 
        # (basé sur l'avantage du terrain et les forces relatives)
        home_advantage = 3.5  # L'équipe à domicile marque en moyenne 3.5 points de plus
        
        # Calculer le facteur d'échelle pour ramener les scores dans une plage réaliste
        team_score = team_totals['PTS']
        opp_score = opp_totals['PTS']
        
        # Ajuster en fonction de l'avantage du terrain
        if is_home:
            team_score += home_advantage
            opp_score -= home_advantage
        else:
            team_score -= home_advantage
            opp_score += home_advantage
        
        # Normaliser les scores si nécessaire
        if team_score < MIN_SCORE:
            team_scale = MIN_SCORE / team_score
            team_score = MIN_SCORE
        elif team_score > MAX_SCORE:
            team_scale = MAX_SCORE / team_score
            team_score = MAX_SCORE
        else:
            team_scale = 1.0
            
        if opp_score < MIN_SCORE:
            opp_scale = MIN_SCORE / opp_score
            opp_score = MIN_SCORE
        elif opp_score > MAX_SCORE:
            opp_scale = MAX_SCORE / opp_score
            opp_score = MAX_SCORE
        else:
            opp_scale = 1.0
        
        # Mettre à jour les points dans les totaux
        team_totals['PTS'] = team_score
        opp_totals['PTS'] = opp_score
        
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
    def compare_models(self, player_id):
        """
        Compare différents modèles de machine learning pour un joueur donné
        
        Args:
            player_id (str): ID du joueur
            
        Returns:
            dict: Résultats de performance des différents modèles
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_absolute_error
        
        # Récupérer les données du joueur
        player_data = self.data[self.data['player_id'] == player_id].copy()
        
        if len(player_data) < 20:  # Minimum de données pour une évaluation fiable
            return {"error": f"Pas assez de données pour {player_id}"}
        
        # Trier par date (crucial pour éviter la fuite de données)
        player_data = player_data.sort_values('Date')
        
        # Définir les indices de séparation (80% entraînement, 20% test)
        train_size = int(len(player_data) * 0.8)
        train_data = player_data.iloc[:train_size]
        test_data = player_data.iloc[train_size:]
        
        # Sélectionner une statistique pour l'évaluation (par exemple, points)
        target_stat = 'PTS'
        
        # Préparer les features (caractéristiques simplifiées pour cette démonstration)
        train_data_with_features = self._create_features_for_player(train_data)
        
        # Créer une version de test avec les caractéristiques
        test_start_date = test_data['Date'].min()
        historical_data_for_test = player_data[player_data['Date'] < test_start_date].copy()
        combined_data = pd.concat([historical_data_for_test, test_data]).sort_values('Date')
        combined_data_with_features = self._create_features_for_player(combined_data)
        test_data_with_features = combined_data_with_features[
            combined_data_with_features['Date'] >= test_start_date
        ].copy()
        
        # Définir les caractéristiques communes pour tous les modèles
        base_features = [
            f'{target_stat}_avg_3', f'{target_stat}_avg_5', f'{target_stat}_avg_10',
            f'{target_stat}_std_5', f'{target_stat}_trend', f'{target_stat}_weighted',
            'is_home', 'is_back_to_back', 'rest_days'
        ]
        
        # Vérifier quelles caractéristiques sont disponibles
        available_features = [f for f in base_features if f in train_data_with_features.columns]
        
        if len(available_features) < 3:  # Pas assez de caractéristiques
            return {"error": "Caractéristiques insuffisantes pour une comparaison de modèles"}
        
        # Préparer les données
        X_train = train_data_with_features[available_features].dropna()
        y_train = train_data_with_features.loc[X_train.index, target_stat]
        
        X_test = test_data_with_features[available_features].dropna()
        y_test = test_data_with_features.loc[X_test.index, target_stat]
        
        if len(X_train) < 10 or len(X_test) < 5:
            return {"error": "Données insuffisantes après traitement"}
        
        # Normaliser les données pour SVR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Définir les modèles à comparer
        models = {
            "GradientBoosting": {
                "model": GradientBoostingRegressor(
                    n_estimators=150, 
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=42
                ),
                "needs_scaling": False
            },
            "RandomForest": {
                "model": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                "needs_scaling": False
            },
            "SVR": {
                "model": SVR(
                    kernel='rbf',
                    C=10.0,
                    epsilon=0.2,
                    gamma='scale'
                ),
                "needs_scaling": True
            }
        }
        
        # Entraîner et évaluer chaque modèle
        results = {}
        for name, model_info in models.items():
            model = model_info["model"]
            
            # Utiliser les données normalisées pour SVR
            if model_info["needs_scaling"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculer les métriques
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                "R²": r2,
                "MAE": mae,
                "predictions": y_pred.tolist(),
                "actual": y_test.tolist()
            }
        
        return {
            "player_id": player_id,
            "model_results": results,
            "test_size": len(y_test),
            "feature_count": len(available_features),
            "features_used": available_features
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