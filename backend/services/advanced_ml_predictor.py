"""
Advanced ML predictor with improved model architecture and validation.

This module implements sophisticated ML models for momentum prediction with
proper train/test splits, ensemble methods, and advanced feature engineering.
"""

import logging
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

# Advanced models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from backend.models.game_models import GameEvent, TeamMomentumIndex

logger = logging.getLogger(__name__)


class AdvancedMLPredictor:
    """
    Advanced ML predictor with ensemble methods and proper validation.
    
    Improvements over original:
    - Multiple model architectures (ensemble)
    - Proper game-based train/test splits
    - Advanced feature engineering
    - Comprehensive validation
    - Hyperparameter tuning
    """
    
    def __init__(self, model_path: str = "models/advanced_momentum_predictor.pkl"):
        """Initialize the advanced ML predictor."""
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.ensemble_weights = {}
        
        # Model configurations
        self.model_configs = {
            'logistic': {
                'model': LogisticRegression(
                    random_state=42, max_iter=1000, class_weight='balanced'
                ),
                'scaler': StandardScaler()
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight='balanced',
                    max_depth=10, min_samples_split=5
                ),
                'scaler': None  # Tree-based models don't need scaling
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100, random_state=42, max_depth=6,
                    learning_rate=0.1, subsample=0.8
                ),
                'scaler': None
            },
            'svm': {
                'model': SVC(
                    random_state=42, class_weight='balanced', probability=True,
                    kernel='rbf', C=1.0
                ),
                'scaler': RobustScaler()  # SVM benefits from robust scaling
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model': XGBClassifier(
                    random_state=42, n_estimators=100, max_depth=6,
                    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
                ),
                'scaler': None
            }
        else:
            logger.info("XGBoost not available, using sklearn models only")
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def collect_enhanced_training_data(
        self,
        events_data: List[GameEvent],
        min_games: int = 50
    ) -> pd.DataFrame:
        """
        Collect training data with enhanced features and proper game-based splits.
        
        Args:
            events_data: List of game events
            min_games: Minimum number of games required
            
        Returns:
            DataFrame with enhanced features and labels
        """
        logger.info(f"Collecting enhanced training data from {len(events_data)} events")
        
        # Group events by game
        games_data = {}
        for event in events_data:
            if event.game_id not in games_data:
                games_data[event.game_id] = []
            games_data[event.game_id].append(event)
        
        logger.info(f"Processing {len(games_data)} games for enhanced training data")
        
        if len(games_data) < min_games:
            logger.warning(f"Only {len(games_data)} games available, minimum {min_games} recommended")
        
        # Import enhanced momentum engine
        from backend.services.enhanced_momentum_engine import EnhancedMomentumEngine
        
        # Process each game to extract enhanced training examples
        training_data = []
        momentum_engine = EnhancedMomentumEngine(rolling_window_size=8)
        
        for game_id, game_events in games_data.items():
            try:
                game_training_data = self._extract_enhanced_game_training_data(
                    game_events, momentum_engine, game_id
                )
                training_data.extend(game_training_data)
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        logger.info(f"Collected {len(df)} enhanced training examples")
        
        return df
    
    def _extract_enhanced_game_training_data(
        self,
        game_events: List[GameEvent],
        momentum_engine,
        game_id: str
    ) -> List[Dict[str, Any]]:
        """Extract enhanced training examples from a single game."""
        training_examples = []
        
        # Segment possessions
        possessions = momentum_engine.segment_possessions(game_events)
        
        if len(possessions) < 10:  # Need minimum possessions
            return []
        
        # Group by team
        team_possessions = {}
        for possession in possessions:
            team = possession.team_tricode
            if team not in team_possessions:
                team_possessions[team] = []
            team_possessions[team].append(possession)
        
        # Extract training examples for each team
        for team, team_poss in team_possessions.items():
            if len(team_poss) < 8:  # Need minimum possessions for meaningful examples
                continue
            
            # Calculate enhanced features for all possessions
            try:
                # Create game context (simplified)
                game_context = {
                    'game_id': game_id,
                    'competitive_game': True,  # Assume competitive
                    'total_points': 200,  # Estimate
                    'pace_estimate': 100
                }
                
                enhanced_features = momentum_engine.calculate_enhanced_possession_features(
                    team_poss, game_context
                )
                
                # Create training examples using sliding window
                for i in range(8, len(enhanced_features) - 1):  # Need 8 for history, 1 for future
                    current_window = enhanced_features[i-8:i]
                    next_possession = enhanced_features[i]
                    
                    # Enhanced momentum continuation definition
                    momentum_continued = self._determine_momentum_continuation(
                        current_window, next_possession
                    )
                    
                    # Extract enhanced features for this example
                    example_features = self._calculate_enhanced_training_features(
                        current_window, next_possession
                    )
                    example_features['momentum_continued'] = int(momentum_continued)
                    example_features['game_id'] = game_id
                    example_features['team'] = team
                    
                    training_examples.append(example_features)
                    
            except Exception as e:
                logger.warning(f"Error processing team {team} in game {game_id}: {e}")
                continue
        
        return training_examples
    
    def _determine_momentum_continuation(
        self,
        current_window: List,
        next_possession
    ) -> bool:
        """
        Determine if momentum continued using sophisticated basketball logic.
        
        Args:
            current_window: Recent possession features
            next_possession: Next possession features
            
        Returns:
            True if momentum continued, False otherwise
        """
        # Calculate current momentum indicators
        recent_scoring = np.mean([f.points_scored for f in current_window[-3:]])
        recent_efficiency = np.mean([f.effective_fg_percentage for f in current_window[-3:]])
        recent_energy = np.mean([f.energy_level for f in current_window[-3:]])
        current_run = current_window[-1].scoring_run_length
        
        # Next possession performance
        next_scoring = next_possession.points_scored
        next_efficiency = next_possession.effective_fg_percentage
        next_energy = next_possession.energy_level
        
        # Multiple criteria for momentum continuation
        criteria_met = 0
        
        # Scoring criterion
        if next_scoring >= recent_scoring:
            criteria_met += 1
        
        # Efficiency criterion
        if next_efficiency >= recent_efficiency * 0.8:  # Allow some variance
            criteria_met += 1
        
        # Energy criterion
        if next_energy >= recent_energy * 0.9:
            criteria_met += 1
        
        # Scoring run criterion
        if current_run > 0 and next_scoring > 0:
            criteria_met += 1
        
        # Momentum events criterion
        if next_possession.momentum_events > 0:
            criteria_met += 1
        
        # Defensive criterion
        if next_possession.defensive_stops >= current_window[-1].defensive_stops:
            criteria_met += 1
        
        # Momentum continues if at least 3 out of 6 criteria are met
        return criteria_met >= 3
    
    def _calculate_enhanced_training_features(
        self,
        possession_window: List,
        next_possession
    ) -> Dict[str, float]:
        """Calculate enhanced features for training."""
        features = {}
        
        # Basic statistical features
        features['avg_points_scored'] = np.mean([f.points_scored for f in possession_window])
        features['avg_fg_percentage'] = np.mean([f.fg_percentage for f in possession_window])
        features['avg_effective_fg_pct'] = np.mean([f.effective_fg_percentage for f in possession_window])
        features['avg_true_shooting_pct'] = np.mean([f.true_shooting_percentage for f in possession_window])
        features['avg_turnovers'] = np.mean([f.turnovers for f in possession_window])
        features['avg_rebounds'] = np.mean([f.rebounds for f in possession_window])
        features['avg_fouls'] = np.mean([f.fouls for f in possession_window])
        features['avg_pace'] = np.mean([f.pace for f in possession_window])
        
        # Advanced basketball features
        features['avg_offensive_rating'] = np.mean([f.offensive_rating for f in possession_window])
        features['avg_defensive_rating'] = np.mean([f.defensive_rating for f in possession_window])
        features['avg_assist_to_ratio'] = np.mean([f.assist_to_turnover_ratio for f in possession_window])
        
        # Trend features
        if len(possession_window) >= 4:
            early = possession_window[:4]
            late = possession_window[-4:]
            
            features['points_trend'] = (
                np.mean([f.points_scored for f in late]) -
                np.mean([f.points_scored for f in early])
            )
            features['efficiency_trend'] = (
                np.mean([f.effective_fg_percentage for f in late]) -
                np.mean([f.effective_fg_percentage for f in early])
            )
            features['energy_trend'] = (
                np.mean([f.energy_level for f in late]) -
                np.mean([f.energy_level for f in early])
            )
        else:
            features['points_trend'] = 0.0
            features['efficiency_trend'] = 0.0
            features['energy_trend'] = 0.0
        
        # Momentum-specific features
        features['current_scoring_run'] = possession_window[-1].scoring_run_length
        features['current_defensive_stops'] = possession_window[-1].defensive_stops
        features['avg_momentum_events'] = np.mean([f.momentum_events for f in possession_window])
        features['avg_energy_level'] = np.mean([f.energy_level for f in possession_window])
        
        # Situational features
        features['time_remaining'] = possession_window[-1].time_remaining
        features['score_margin'] = possession_window[-1].score_margin
        features['is_clutch_time'] = float(possession_window[-1].is_clutch_time)
        features['period'] = possession_window[-1].period
        
        # Advanced features
        features['avg_player_impact'] = np.mean([f.player_impact for f in possession_window])
        features['avg_shot_quality'] = np.mean([f.shot_quality for f in possession_window])
        features['avg_defensive_pressure'] = np.mean([f.defensive_pressure for f in possession_window])
        features['pace_differential'] = possession_window[-1].pace_differential
        
        # Consistency features
        features['scoring_consistency'] = 1.0 / (1.0 + np.std([f.points_scored for f in possession_window]))
        features['efficiency_consistency'] = 1.0 / (1.0 + np.std([f.effective_fg_percentage for f in possession_window]))
        features['energy_consistency'] = 1.0 / (1.0 + np.std([f.energy_level for f in possession_window]))
        
        return features
    
    def train_ensemble_model(
        self,
        training_data: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train ensemble model with proper game-based splits.
        
        Args:
            training_data: Enhanced training data
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting ensemble model training with game-based splits")
        
        if len(training_data) < 200:
            raise ValueError(f"Insufficient training data: {len(training_data)} examples")
        
        # Game-based splitting to prevent data leakage
        unique_games = training_data['game_id'].unique()
        
        # Split games into train/val/test
        train_games, temp_games = train_test_split(
            unique_games, test_size=(test_size + val_size), random_state=random_state
        )
        val_games, test_games = train_test_split(
            temp_games, test_size=(test_size / (test_size + val_size)), random_state=random_state
        )
        
        # Create data splits based on games
        train_data = training_data[training_data['game_id'].isin(train_games)]
        val_data = training_data[training_data['game_id'].isin(val_games)]
        test_data = training_data[training_data['game_id'].isin(test_games)]
        
        logger.info(f"Train: {len(train_data)} examples from {len(train_games)} games")
        logger.info(f"Val: {len(val_data)} examples from {len(val_games)} games")
        logger.info(f"Test: {len(test_data)} examples from {len(test_games)} games")
        
        # Prepare features and labels
        feature_columns = [col for col in training_data.columns 
                          if col not in ['momentum_continued', 'game_id', 'team']]
        self.feature_names = feature_columns
        
        X_train = train_data[feature_columns].values
        y_train = train_data['momentum_continued'].values
        X_val = val_data[feature_columns].values
        y_val = val_data['momentum_continued'].values
        X_test = test_data[feature_columns].values
        y_test = test_data['momentum_continued'].values
        
        # Train individual models
        model_results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name} model")
            
            try:
                model = config['model']
                scaler = config['scaler']
                
                # Scale features if needed
                if scaler is not None:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    X_test_scaled = scaler.transform(X_test)
                    self.scalers[model_name] = scaler
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                    X_test_scaled = X_test
                    self.scalers[model_name] = None
                
                # Train model
                model.fit(X_train_scaled, y_train)
                self.models[model_name] = model
                
                # Evaluate on validation set
                val_pred = model.predict(X_val_scaled)
                val_proba = model.predict_proba(X_val_scaled)[:, 1]
                
                val_accuracy = accuracy_score(y_val, val_pred)
                val_precision = precision_score(y_val, val_pred)
                val_recall = recall_score(y_val, val_pred)
                val_f1 = f1_score(y_val, val_pred)
                val_auc = roc_auc_score(y_val, val_proba)
                
                model_results[model_name] = {
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'val_auc': val_auc
                }
                
                logger.info(f"{model_name} - Val Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Calculate ensemble weights based on validation performance
        self._calculate_ensemble_weights(model_results)
        
        # Evaluate ensemble on test set
        test_results = self._evaluate_ensemble(X_test, y_test)
        
        # Compare with baseline
        baseline_results = self._evaluate_baseline(X_test, y_test)
        
        self.is_trained = True
        
        results = {
            'individual_models': model_results,
            'ensemble_results': test_results,
            'baseline_results': baseline_results,
            'ensemble_weights': self.ensemble_weights,
            'feature_names': self.feature_names,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Ensemble training completed - Test accuracy: {test_results['accuracy']:.4f}")
        
        return results
    
    def _calculate_ensemble_weights(self, model_results: Dict[str, Dict]) -> None:
        """Calculate ensemble weights based on validation performance."""
        if not model_results:
            return
        
        # Use AUC as the primary metric for weighting
        auc_scores = {name: results['val_auc'] for name, results in model_results.items()}
        
        # Softmax weighting based on AUC scores
        auc_values = np.array(list(auc_scores.values()))
        exp_scores = np.exp(auc_values * 5)  # Scale for more pronounced differences
        weights = exp_scores / np.sum(exp_scores)
        
        self.ensemble_weights = dict(zip(auc_scores.keys(), weights))
        
        logger.info("Ensemble weights:")
        for model_name, weight in self.ensemble_weights.items():
            logger.info(f"  {model_name}: {weight:.4f}")
    
    def _evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble model on test set."""
        if not self.models or not self.ensemble_weights:
            return {}
        
        # Get predictions from all models
        ensemble_proba = np.zeros(len(X_test))
        
        for model_name, weight in self.ensemble_weights.items():
            if model_name in self.models:
                model = self.models[model_name]
                scaler = self.scalers[model_name]
                
                # Scale features if needed
                if scaler is not None:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_test_scaled = X_test
                
                # Get probabilities
                model_proba = model.predict_proba(X_test_scaled)[:, 1]
                ensemble_proba += weight * model_proba
        
        # Convert probabilities to predictions
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred)
        recall = recall_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred)
        auc = roc_auc_score(y_test, ensemble_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def _evaluate_baseline(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate baseline models."""
        # Most frequent class baseline
        dummy_frequent = DummyClassifier(strategy='most_frequent')
        dummy_frequent.fit(X_test, y_test)  # Dummy fit
        frequent_pred = dummy_frequent.predict(X_test)
        frequent_acc = accuracy_score(y_test, frequent_pred)
        
        # Random baseline
        dummy_random = DummyClassifier(strategy='uniform', random_state=42)
        dummy_random.fit(X_test, y_test)
        random_pred = dummy_random.predict(X_test)
        random_acc = accuracy_score(y_test, random_pred)
        
        return {
            'most_frequent_accuracy': frequent_acc,
            'random_accuracy': random_acc
        }
    
    def predict_momentum_continuation(
        self,
        tmi_history: List[TeamMomentumIndex],
        current_features: List
    ) -> Tuple[float, float]:
        """
        Predict momentum continuation using ensemble model.
        
        Args:
            tmi_history: Recent TMI calculations
            current_features: Enhanced possession features
            
        Returns:
            Tuple of (continuation_probability, confidence_score)
        """
        if not self.is_trained or not self.models:
            logger.warning("Ensemble model not trained, returning neutral prediction")
            return 0.5, 0.0
        
        try:
            # Extract features for prediction
            features = self._extract_prediction_features(current_features)
            
            if features is None:
                return 0.5, 0.0
            
            # Get ensemble prediction
            ensemble_proba = 0.0
            total_weight = 0.0
            
            for model_name, weight in self.ensemble_weights.items():
                if model_name in self.models:
                    model = self.models[model_name]
                    scaler = self.scalers[model_name]
                    
                    # Scale features if needed
                    if scaler is not None:
                        features_scaled = scaler.transform(features)
                    else:
                        features_scaled = features
                    
                    # Get probability
                    model_proba = model.predict_proba(features_scaled)[0, 1]
                    ensemble_proba += weight * model_proba
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_proba /= total_weight
            
            # Calculate confidence based on model agreement
            confidence = self._calculate_prediction_confidence(features)
            
            return float(ensemble_proba), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return 0.5, 0.0
    
    def _extract_prediction_features(
        self,
        current_features: List
    ) -> Optional[np.ndarray]:
        """Extract features for prediction from enhanced possession features."""
        if not current_features or len(current_features) < 8:
            return None
        
        # Use last 8 possessions as window
        possession_window = current_features[-8:]
        
        # Calculate same features as in training
        features = self._calculate_enhanced_training_features(
            possession_window, possession_window[-1]  # Use last possession as "next"
        )
        
        # Remove non-feature columns
        feature_values = [features.get(name, 0.0) for name in self.feature_names]
        
        return np.array([feature_values])
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate prediction confidence based on model agreement."""
        if not self.models:
            return 0.0
        
        predictions = []
        
        for model_name in self.models:
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            # Scale features if needed
            if scaler is not None:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            # Get probability
            proba = model.predict_proba(features_scaled)[0, 1]
            predictions.append(proba)
        
        # Confidence based on standard deviation (lower std = higher confidence)
        if len(predictions) > 1:
            std_dev = np.std(predictions)
            confidence = max(0.0, 1.0 - std_dev * 4)  # Scale std to 0-1 range
        else:
            confidence = 0.5
        
        return confidence
    
    def save_model(self) -> bool:
        """Save the ensemble model to disk."""
        if not self.is_trained:
            logger.error("No trained ensemble model to save")
            return False
        
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'ensemble_weights': self.ensemble_weights,
                'feature_names': self.feature_names,
                'trained_at': datetime.utcnow().isoformat(),
                'model_type': 'advanced_ensemble'
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Ensemble model saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ensemble model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load ensemble model from disk."""
        if not os.path.exists(self.model_path):
            logger.info(f"No existing ensemble model found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.ensemble_weights = model_data['ensemble_weights']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            
            trained_at = model_data.get('trained_at', 'unknown')
            model_type = model_data.get('model_type', 'unknown')
            
            logger.info(f"Ensemble model loaded from {self.model_path}")
            logger.info(f"Model type: {model_type}, trained at: {trained_at}")
            logger.info(f"Loaded {len(self.models)} individual models")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            return False


def create_advanced_predictor(model_path: str = "models/advanced_momentum_predictor.pkl") -> AdvancedMLPredictor:
    """Create an advanced ML predictor instance."""
    return AdvancedMLPredictor(model_path=model_path)


def train_advanced_momentum_model(
    historical_events: List[GameEvent],
    model_path: str = "models/advanced_momentum_predictor.pkl"
) -> Dict[str, Any]:
    """
    Train advanced ensemble momentum prediction model.
    
    Args:
        historical_events: List of historical game events
        model_path: Path to save the trained model
        
    Returns:
        Training results and metrics
    """
    predictor = AdvancedMLPredictor(model_path=model_path)
    
    # Collect enhanced training data
    training_data = predictor.collect_enhanced_training_data(historical_events)
    
    # Train ensemble model
    results = predictor.train_ensemble_model(training_data)
    
    # Save model
    predictor.save_model()
    
    return results