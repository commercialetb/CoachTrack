"""
Machine Learning Models Module
- ML Injury Risk Predictor (Random Forest)
- Performance Predictor Next Game (Gradient Boosting)
- Shot Form Analyzer (Computer Vision placeholder)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime, timedelta

# =================================================================
# ML INJURY RISK PREDICTOR (RANDOM FOREST)
# =================================================================

class MLInjuryPredictor:
    """
    Machine Learning Injury Risk Predictor using Random Forest
    Features: ACWR, asymmetry, fatigue, workload, rest days, age, etc.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'acwr', 'asymmetry_pct', 'fatigue_index', 
            'total_distance_7d', 'high_intensity_distance_7d',
            'rest_days', 'age', 'weight_kg', 'body_fat_pct',
            'avg_speed', 'max_speed', 'acceleration_events'
        ]
    
    def extract_features(self, player_data, physical_data=None, player_age=25):
        """Extract features from player tracking and physical data"""
        
        features = {}
        
        # Calculate ACWR
        if len(player_data) >= 7:
            recent_distance = self._calculate_distance(player_data.tail(int(len(player_data) * 0.15)))
            chronic_distance = self._calculate_distance(player_data) / len(player_data) * (len(player_data) * 0.15)
            features['acwr'] = recent_distance / chronic_distance if chronic_distance > 0 else 1.0
        else:
            features['acwr'] = 1.0
        
        # Asymmetry
        left_moves = len(player_data[player_data['dx'] < -0.5])
        right_moves = len(player_data[player_data['dx'] > 0.5])
        total_lateral = left_moves + right_moves
        features['asymmetry_pct'] = abs(left_moves - right_moves) / total_lateral * 100 if total_lateral > 0 else 0
        
        # Fatigue index
        if 'speed_kmh_calc' in player_data.columns:
            first_q = player_data.head(len(player_data)//4)['speed_kmh_calc'].mean()
            last_q = player_data.tail(len(player_data)//4)['speed_kmh_calc'].mean()
            features['fatigue_index'] = (first_q - last_q) / first_q if first_q > 0 else 0
        else:
            features['fatigue_index'] = 0
        
        # Workload metrics (7 days)
        features['total_distance_7d'] = self._calculate_distance(player_data)
        features['high_intensity_distance_7d'] = len(player_data[player_data.get('speed_kmh_calc', pd.Series([0])) > 15])
        
        # Rest days (simulated)
        features['rest_days'] = np.random.randint(0, 3)
        
        # Physical data
        features['age'] = player_age
        if physical_
            features['weight_kg'] = physical_data.get('weight_kg', 80)
            features['body_fat_pct'] = physical_data.get('body_fat_pct', 12)
        else:
            features['weight_kg'] = 80
            features['body_fat_pct'] = 12
        
        # Speed metrics
        if 'speed_kmh_calc' in player_data.columns:
            features['avg_speed'] = player_data['speed_kmh_calc'].mean()
            features['max_speed'] = player_data['speed_kmh_calc'].max()
        else:
            features['avg_speed'] = 10
            features['max_speed'] = 20
        
        # Acceleration events (high acceleration changes)
        features['acceleration_events'] = len(player_data[abs(player_data['dx']) > 1.0])
        
        return features
    
    def _calculate_distance(self, df):
        """Calculate total distance from tracking data"""
        if len(df) < 2:
            return 0
        dx = np.diff(df['x'].values)
        dy = np.diff(df['y'].values)
        return np.sum(np.sqrt(dx**2 + dy**2))
    
    def train(self, training_data, labels):
        """
        Train the model
        training_ list of feature dicts
        labels: list of injury outcomes (0=no injury, 1=injury)
        """
        X = pd.DataFrame(training_data)[self.feature_names]
        y = np.array(labels)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return self.model.score(X_scaled, y)
    
    def predict(self, features):
        """Predict injury risk for a player"""
        if not self.is_trained:
            # Use synthetic training data if not trained
            self._train_synthetic()
        
        X = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        risk_prob = self.model.predict_proba(X_scaled)[0][1]  # Probability of injury
        risk_class = self.model.predict(X_scaled)[0]
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Risk level
        if risk_prob > 0.7:
            risk_level = "ALTO"
        elif risk_prob > 0.4:
            risk_level = "MEDIO"
        else:
            risk_level = "BASSO"
        
        return {
            'risk_level': risk_level,
            'risk_probability': round(risk_prob * 100, 1),
            'risk_class': int(risk_class),
            'top_risk_factors': top_features,
            'recommendations': self._generate_recommendations(risk_level, features, top_features)
        }
    
    def _train_synthetic(self):
        """Train with synthetic data if no real training data available"""
        np.random.seed(42)
        n_samples = 500
        
        synthetic_data = []
        labels = []
        
        for _ in range(n_samples):
            features = {
                'acwr': np.random.uniform(0.5, 2.0),
                'asymmetry_pct': np.random.uniform(0, 30),
                'fatigue_index': np.random.uniform(0, 0.3),
                'total_distance_7d': np.random.uniform(5000, 15000),
                'high_intensity_distance_7d': np.random.uniform(500, 3000),
                'rest_days': np.random.randint(0, 4),
                'age': np.random.randint(18, 38),
                'weight_kg': np.random.uniform(70, 110),
                'body_fat_pct': np.random.uniform(6, 20),
                'avg_speed': np.random.uniform(8, 18),
                'max_speed': np.random.uniform(15, 28),
                'acceleration_events': np.random.randint(50, 300)
            }
            
            # Simulate injury based on risk factors
            injury_score = 0
            if features['acwr'] > 1.5: injury_score += 3
            if features['asymmetry_pct'] > 20: injury_score += 2
            if features['fatigue_index'] > 0.15: injury_score += 2
            if features['rest_days'] == 0: injury_score += 1
            
            label = 1 if injury_score > 4 or np.random.random() < 0.15 else 0
            
            synthetic_data.append(features)
            labels.append(label)
        
        self.train(synthetic_data, labels)
    
    def _generate_recommendations(self, risk_level, features, top_factors):
        """Generate recommendations based on risk level and factors"""
        recommendations = []
        
        if risk_level == "ALTO":
            recommendations.append("🚨 RIPOSO IMMEDIATO: 48-72h senza attività intensa")
            recommendations.append("👨‍⚕️ Valutazione medica/fisioterapista URGENTE")
        
        # Specific recommendations based on top factors
        for factor, importance in top_factors[:3]:
            if factor == 'acwr' and features['acwr'] > 1.5:
                recommendations.append(f"⚠️ ACWR troppo alto ({features['acwr']:.2f}): ridurre carico allenamento 30-40%")
            elif factor == 'asymmetry_pct' and features['asymmetry_pct'] > 15:
                recommendations.append(f"⚖️ Asimmetria elevata ({features['asymmetry_pct']:.1f}%): esercizi unilaterali e correzione biomeccanica")
            elif factor == 'fatigue_index' and features['fatigue_index'] > 0.1:
                recommendations.append(f"😴 Fatica significativa: aumentare recupero, valutare sonno e nutrizione")
            elif factor == 'rest_days' and features['rest_days'] == 0:
                recommendations.append("📅 Nessun giorno di riposo recente: programmare 1-2 giorni recovery")
        
        if risk_level != "ALTO":
            recommendations.append("✅ Continuare monitoraggio settimanale")
        
        return recommendations

# =================================================================
# PERFORMANCE PREDICTOR (GRADIENT BOOSTING)
# =================================================================

class PerformancePredictor:
    """
    Predict next game performance using Gradient Boosting
    Predicts: points, assists, rebounds, efficiency
    """
    
    def __init__(self):
        self.models = {
            'points': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'assists': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'rebounds': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'efficiency': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        self.scalers = {key: StandardScaler() for key in self.models.keys()}
        self.is_trained = False
        
        self.feature_names = [
            'avg_points_last5', 'avg_assists_last5', 'avg_rebounds_last5',
            'trend_points', 'rest_days', 'opponent_def_rating',
            'home_away', 'minutes_played_last', 'usage_rate',
            'fatigue_score', 'injury_risk_score'
        ]
    
    def extract_features(self, player_stats_history, opponent_info, injury_risk=None):
        """Extract features from player history and game context"""
        
        features = {}
        
        # Recent averages (last 5 games)
        if len(player_stats_history) >= 5:
            last5 = player_stats_history.tail(5)
            features['avg_points_last5'] = last5['points'].mean()
            features['avg_assists_last5'] = last5['assists'].mean()
            features['avg_rebounds_last5'] = last5['rebounds'].mean()
            
            # Trend (linear slope)
            features['trend_points'] = (last5['points'].iloc[-1] - last5['points'].iloc[0]) / 5
        else:
            features['avg_points_last5'] = 15
            features['avg_assists_last5'] = 5
            features['avg_rebounds_last5'] = 6
            features['trend_points'] = 0
        
        # Game context
        features['rest_days'] = opponent_info.get('rest_days', 1)
        features['opponent_def_rating'] = opponent_info.get('def_rating', 110)
        features['home_away'] = 1 if opponent_info.get('location') == 'home' else 0
        
        # Player condition
        features['minutes_played_last'] = player_stats_history.iloc[-1]['minutes'] if len(player_stats_history) > 0 else 30
        features['usage_rate'] = opponent_info.get('usage_rate', 25)
        features['fatigue_score'] = opponent_info.get('fatigue', 0.1)
        features['injury_risk_score'] = injury_risk['risk_probability'] if injury_risk else 20
        
        return features
    
    def predict_next_game(self, features):
        """Predict performance for next game"""
        if not self.is_trained:
            self._train_synthetic()
        
        predictions = {}
        
        X = pd.DataFrame([features])[self.feature_names]
        
        for stat, model in self.models.items():
            X_scaled = self.scalers[stat].transform(X)
            pred = model.predict(X_scaled)[0]
            predictions[stat] = round(max(0, pred), 1)  # Ensure non-negative
        
        # Calculate confidence intervals (simple approach)
        predictions['confidence'] = 'ALTA' if features['injury_risk_score'] < 30 else 'MEDIA' if features['injury_risk_score'] < 60 else 'BASSA'
        
        return predictions
    
    def _train_synthetic(self):
        """Train with synthetic data"""
        np.random.seed(42)
        n_samples = 1000
        
        X_data = []
        y_data = {key: [] for key in self.models.keys()}
        
        for _ in range(n_samples):
            features = {
                'avg_points_last5': np.random.uniform(8, 25),
                'avg_assists_last5': np.random.uniform(2, 10),
                'avg_rebounds_last5': np.random.uniform(3, 12),
                'trend_points': np.random.uniform(-2, 2),
                'rest_days': np.random.randint(0, 4),
                'opponent_def_rating': np.random.uniform(100, 120),
                'home_away': np.random.choice([0, 1]),
                'minutes_played_last': np.random.uniform(20, 40),
                'usage_rate': np.random.uniform(18, 32),
                'fatigue_score': np.random.uniform(0, 0.3),
                'injury_risk_score': np.random.uniform(0, 80)
            }
            
            # Simulate performance with some correlation
            points = features['avg_points_last5'] + features['trend_points'] + np.random.normal(0, 3)
            points -= (features['opponent_def_rating'] - 110) * 0.1
            points += 2 if features['home_away'] == 1 else 0
            points -= features['fatigue_score'] * 10
            
            assists = features['avg_assists_last5'] + np.random.normal(0, 1.5)
            rebounds = features['avg_rebounds_last5'] + np.random.normal(0, 2)
            efficiency = (points + assists + rebounds) / features['minutes_played_last'] * 10
            
            X_data.append(features)
            y_data['points'].append(max(0, points))
            y_data['assists'].append(max(0, assists))
            y_data['rebounds'].append(max(0, rebounds))
            y_data['efficiency'].append(max(0, efficiency))
        
        X_df = pd.DataFrame(X_data)[self.feature_names]
        
        for stat, model in self.models.items():
            X_scaled = self.scalers[stat].fit_transform(X_df)
            model.fit(X_scaled, y_data[stat])
        
        self.is_trained = True

# =================================================================
# SHOT FORM ANALYZER (COMPUTER VISION PLACEHOLDER)
# =================================================================

class ShotFormAnalyzer:
    """
    Shot Form Analysis using Computer Vision
    Placeholder - requires MediaPipe/OpenCV integration for real implementation
    """
    
    def __init__(self):
        self.reference_form = {
            'elbow_angle': 90,
            'release_height': 2.4,
            'release_angle': 52,
            'follow_through': True,
            'knee_bend': 45
        }
    
    def analyze_shot_video(self, video_path):
        """
        Analyze shot form from video
        TODO: Implement with MediaPipe Pose + OpenCV
        """
        # Placeholder - would use MediaPipe for real implementation
        return {
            'status': 'PLACEHOLDER',
            'message': 'Shot form analysis richiede integrazione MediaPipe/OpenCV',
            'next_steps': [
                'Installare mediapipe: pip install mediapipe',
                'Integrare pose estimation',
                'Estrarre angoli chiave (elbow, wrist, knee)',
                'Comparare con reference form'
            ]
        }
    
    def get_recommendations(self, shot_analysis):
        """Get recommendations based on shot form analysis"""
        recommendations = [
            "📹 Feature in sviluppo - Computer Vision integration",
            "🎯 Richiede MediaPipe Pose per analisi video",
            "📊 Analizzerà: angolo gomito, release point, follow-through"
        ]
        return recommendations

# =================================================================
# MODEL PERSISTENCE
# =================================================================

def save_model(model, filepath):
    """Save trained model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load trained model from disk"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
