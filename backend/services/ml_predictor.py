"""
Machine Learning prediction service for momentum continuation.

This module implements the ML prediction functionality for MomentumML,
including historical data collection, feature extraction, model training,
and prediction services.
"""

import logging
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from backend.models.game_models import GameEvent, Possession, TeamMomentumIndex
from backend.services.momentum_engine import MomentumEngine, PossessionFeatures

# Configure logging
logger = logging.getLogger(__name__)


class MomentumPredictor:
    """
    Machine learning predictor for momentum continuation.
    
    Predicts the probability that current team momentum will continue
    in the next possession using historical play-by-play data.
    """
    
    def __init__(self, model_path: str = "models/momentum_predictor.pkl"):
        """
        Initialize the momentum predictor.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def collect_historical_data(
        self,
        events_data: List[GameEvent],
        min_games: int = 200
    ) -> pd.DataFrame:
        """
        Collect and process historical play-by-play data for model training.
        
        Args:
            events_data: List of historical game events
            min_games: Minimum number of games required for training
            
        Returns:
            DataFrame with features and labels for training
        """
        logger.info(f"Collecting historical data from {len(events_data)} events")
        
        # Group events by game
        games_data = {}
        for event in events_data:
            if event.game_id not in games_data:
                games_data[event.game_id] = []
            games_data[event.game_id].append(event)
        
        logger.info(f"Processing {len(games_data)} games for training data")
        
        if len(games_data) < min_games:
            logger.warning(f"Only {len(games_data)} games available, minimum {min_games} recommended")
        
        # Process each game to extract training examples
        training_data = []
        momentum_engine = MomentumEngine(rolling_window_size=5)
        
        for game_id, game_events in games_data.items():
            try:
                game_training_data = self._extract_game_training_data(
                    game_events, momentum_engine
                )
                training_data.extend(game_training_data)
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        logger.info(f"Collected {len(df)} training examples from historical data")
        
        return df
    
    def extract_prediction_features(
        self,
        tmi_history: List[TeamMomentumIndex],
        current_features: List[PossessionFeatures]
    ) -> np.ndarray:
        """
        Extract features for momentum prediction from current game state.
        
        Args:
            tmi_history: Recent TMI calculations for context
            current_features: Current possession features
            
        Returns:
            Feature array ready for model prediction
        """
        if not current_features:
            # Return neutral features if no data
            return np.zeros((1, len(self.feature_names)))
        
        # Calculate current window statistics
        recent_features = current_features[-5:]  # Last 5 possessions
        
        features = {}
        
        # Basic possession statistics
        features['avg_points_scored'] = np.mean([f.points_scored for f in recent_features])
        features['avg_fg_percentage'] = np.mean([f.fg_percentage for f in recent_features])
        features['avg_turnovers'] = np.mean([f.turnovers for f in recent_features])
        features['avg_rebounds'] = np.mean([f.rebounds for f in recent_features])
        features['avg_fouls'] = np.mean([f.fouls for f in recent_features])
        features['avg_pace'] = np.mean([f.pace for f in recent_features])
        
        # Trend features (change over last few possessions)
        if len(recent_features) >= 3:
            early_features = recent_features[:2]
            late_features = recent_features[-2:]
            
            features['points_trend'] = (
                np.mean([f.points_scored for f in late_features]) -
                np.mean([f.points_scored for f in early_features])
            )
            features['fg_pct_trend'] = (
                np.mean([f.fg_percentage for f in late_features]) -
                np.mean([f.fg_percentage for f in early_features])
            )
            features['turnover_trend'] = (
                np.mean([f.turnovers for f in late_features]) -
                np.mean([f.turnovers for f in early_features])
            )
        else:
            features['points_trend'] = 0.0
            features['fg_pct_trend'] = 0.0
            features['turnover_trend'] = 0.0
        
        # TMI-based features
        if tmi_history:
            recent_tmi = tmi_history[-3:]  # Last 3 TMI calculations
            features['current_tmi'] = recent_tmi[-1].tmi_value
            features['tmi_volatility'] = np.std([t.tmi_value for t in recent_tmi])
            
            if len(recent_tmi) >= 2:
                features['tmi_trend'] = recent_tmi[-1].tmi_value - recent_tmi[0].tmi_value
            else:
                features['tmi_trend'] = 0.0
        else:
            features['current_tmi'] = 0.0
            features['tmi_volatility'] = 0.0
            features['tmi_trend'] = 0.0
        
        # Consistency features
        features['fg_consistency'] = 1.0 / (1.0 + np.std([f.fg_percentage for f in recent_features]))
        features['scoring_consistency'] = 1.0 / (1.0 + np.std([f.points_scored for f in recent_features]))
        
        # Convert to array in correct order
        if not self.feature_names:
            self.feature_names = sorted(features.keys())
        
        feature_array = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        
        return feature_array
    
    def train_model(
        self,
        training_data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the logistic regression model for momentum prediction.
        
        Args:
            training_data: DataFrame with features and labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics and results
        """
        logger.info("Starting model training")
        
        if len(training_data) < 100:
            raise ValueError(f"Insufficient training data: {len(training_data)} examples")
        
        # Separate features and labels
        feature_columns = [col for col in training_data.columns if col != 'momentum_continued']
        X = training_data[feature_columns].values
        y = training_data['momentum_continued'].values
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression model
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Get feature importance (coefficients)
        feature_importance = dict(zip(
            self.feature_names,
            abs(self.model.coef_[0])
        ))
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, test_predictions)
        }
        
        self.is_trained = True
        
        logger.info(f"Model training completed - Test accuracy: {test_accuracy:.3f}")
        logger.info(f"Feature importance: {feature_importance}")
        
        return results
    
    def predict_momentum_continuation(
        self,
        tmi_history: List[TeamMomentumIndex],
        current_features: List[PossessionFeatures]
    ) -> Tuple[float, float]:
        """
        Predict the probability that current momentum will continue.
        
        Args:
            tmi_history: Recent TMI calculations for context
            current_features: Current possession features
            
        Returns:
            Tuple of (continuation_probability, confidence_score)
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, returning neutral prediction")
            return 0.5, 0.0
        
        try:
            # Extract features
            features = self.extract_prediction_features(tmi_history, current_features)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            continuation_prob = probabilities[1]  # Probability of class 1 (momentum continues)
            
            # Calculate confidence based on how far from 0.5 the prediction is
            confidence = abs(continuation_prob - 0.5) * 2
            
            return float(continuation_prob), float(confidence)
        
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.5, 0.0
    
    def save_model(self) -> bool:
        """
        Save the trained model to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained or self.model is None:
            logger.error("No trained model to save")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'trained_at': datetime.utcnow().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {self.model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load a trained model from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.model_path):
            logger.info(f"No existing model found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            
            trained_at = model_data.get('trained_at', 'unknown')
            logger.info(f"Model loaded from {self.model_path} (trained at: {trained_at})")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _extract_game_training_data(
        self,
        game_events: List[GameEvent],
        momentum_engine: MomentumEngine
    ) -> List[Dict[str, Any]]:
        """
        Extract training examples from a single game.
        
        Args:
            game_events: Events from one game
            momentum_engine: Momentum engine for processing
            
        Returns:
            List of training examples
        """
        training_examples = []
        
        # Segment possessions
        possessions = momentum_engine.segment_possessions(game_events)
        
        # Group by team
        team_possessions = {}
        for possession in possessions:
            team = possession.team_tricode
            if team not in team_possessions:
                team_possessions[team] = []
            team_possessions[team].append(possession)
        
        # Extract training examples for each team
        for team, team_poss in team_possessions.items():
            if len(team_poss) < 10:  # Need minimum possessions for meaningful examples
                continue
            
            # Calculate features for all possessions
            features = momentum_engine.calculate_possession_features(team_poss)
            
            # Create training examples using sliding window
            for i in range(5, len(features) - 1):  # Need 5 for history, 1 for future
                # Current window features
                current_window = features[i-5:i]
                next_possession = features[i]
                
                # Calculate if momentum continued (simplified heuristic)
                current_avg_points = np.mean([f.points_scored for f in current_window])
                next_points = next_possession.points_scored
                
                # Momentum continues if next possession performs better than average
                momentum_continued = next_points >= current_avg_points
                
                # Extract features for this example
                example_features = self._calculate_training_features(current_window)
                example_features['momentum_continued'] = int(momentum_continued)
                
                training_examples.append(example_features)
        
        return training_examples
    
    def _calculate_training_features(
        self,
        possession_window: List[PossessionFeatures]
    ) -> Dict[str, float]:
        """
        Calculate features for a training example.
        
        Args:
            possession_window: Window of recent possessions
            
        Returns:
            Dictionary of calculated features
        """
        features = {}
        
        # Basic statistics
        features['avg_points_scored'] = np.mean([f.points_scored for f in possession_window])
        features['avg_fg_percentage'] = np.mean([f.fg_percentage for f in possession_window])
        features['avg_turnovers'] = np.mean([f.turnovers for f in possession_window])
        features['avg_rebounds'] = np.mean([f.rebounds for f in possession_window])
        features['avg_fouls'] = np.mean([f.fouls for f in possession_window])
        features['avg_pace'] = np.mean([f.pace for f in possession_window])
        
        # Trend features
        if len(possession_window) >= 3:
            early = possession_window[:2]
            late = possession_window[-2:]
            
            features['points_trend'] = (
                np.mean([f.points_scored for f in late]) -
                np.mean([f.points_scored for f in early])
            )
            features['fg_pct_trend'] = (
                np.mean([f.fg_percentage for f in late]) -
                np.mean([f.fg_percentage for f in early])
            )
            features['turnover_trend'] = (
                np.mean([f.turnovers for f in late]) -
                np.mean([f.turnovers for f in early])
            )
        else:
            features['points_trend'] = 0.0
            features['fg_pct_trend'] = 0.0
            features['turnover_trend'] = 0.0
        
        # Consistency features
        features['fg_consistency'] = 1.0 / (1.0 + np.std([f.fg_percentage for f in possession_window]))
        features['scoring_consistency'] = 1.0 / (1.0 + np.std([f.points_scored for f in possession_window]))
        
        # Mock TMI features (would be calculated from actual TMI in real scenario)
        features['current_tmi'] = features['avg_points_scored'] * 0.4 + features['avg_fg_percentage'] * 0.3
        features['tmi_volatility'] = np.std([f.points_scored for f in possession_window]) * 0.1
        features['tmi_trend'] = features['points_trend'] * 0.2
        
        return features


# Utility functions
def create_predictor(model_path: str = "models/momentum_predictor.pkl") -> MomentumPredictor:
    """Create a momentum predictor instance."""
    return MomentumPredictor(model_path=model_path)


def train_momentum_model(
    historical_events: List[GameEvent],
    model_path: str = "models/momentum_predictor.pkl"
) -> Dict[str, Any]:
    """
    Train a momentum prediction model with historical data.
    
    Args:
        historical_events: List of historical game events
        model_path: Path to save the trained model
        
    Returns:
        Training results and metrics
    """
    predictor = MomentumPredictor(model_path=model_path)
    
    # Collect training data
    training_data = predictor.collect_historical_data(historical_events)
    
    # Train model
    results = predictor.train_model(training_data)
    
    # Save model
    predictor.save_model()
    
    return results