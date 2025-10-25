"""
Production Momentum Predictor Service

Integrates the trained advanced NBA momentum model into the production system.
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from backend.models.game_models import GameEvent, TeamMomentumIndex

logger = logging.getLogger(__name__)


class ProductionMomentumPredictor:
    """Production-ready momentum predictor using the trained advanced model."""
    
    def __init__(self, model_path: str = None):
        """Initialize the production predictor."""
        if model_path is None:
            # Get the absolute path to the model file
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(project_root, "models", "advanced", "advanced_nba_momentum_random_forest_20251023_172504.pkl")
        
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_loaded = False
        
        # Load the model
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model from disk."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Loaded production model: {model_data.get('model_name', 'unknown')}")
            logger.info(f"Model trained at: {model_data.get('trained_at', 'unknown')}")
            logger.info(f"Features: {len(self.feature_names)}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_momentum_continuation(
        self,
        events_sequence: List[GameEvent],
        team: str
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Predict momentum continuation probability for a team.
        
        Args:
            events_sequence: Recent sequence of game events (minimum 12 events)
            team: Team tricode to predict momentum for
            
        Returns:
            Tuple of (probability, confidence, feature_details)
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, returning neutral prediction")
            return 0.5, 0.0, {}
        
        if len(events_sequence) < 12:
            logger.warning(f"Insufficient events for prediction: {len(events_sequence)} < 12")
            return 0.5, 0.0, {"error": "insufficient_events"}
        
        try:
            # Filter events for the specific team
            team_events = [e for e in events_sequence if e.team_tricode == team]
            
            if len(team_events) < 8:
                logger.warning(f"Insufficient team events: {len(team_events)} < 8")
                return 0.5, 0.0, {"error": "insufficient_team_events"}
            
            # Extract features from the event sequence
            features = self._extract_features_from_sequence(team_events)
            
            if features is None:
                return 0.5, 0.0, {"error": "feature_extraction_failed"}
            
            # Make prediction
            if self.scaler is not None:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Get probability
            probability = self.model.predict_proba(features_scaled)[0, 1]
            
            # Calculate confidence (distance from 0.5)
            confidence = abs(probability - 0.5) * 2
            
            # Feature details for debugging/explanation
            feature_details = {
                name: value for name, value in zip(self.feature_names, features)
            }
            
            return float(probability), float(confidence), feature_details
            
        except Exception as e:
            logger.error(f"Error in momentum prediction: {e}")
            return 0.5, 0.0, {"error": str(e)}
    
    def _extract_features_from_sequence(self, team_events: List[GameEvent]) -> Optional[List[float]]:
        """Extract features from a sequence of team events."""
        try:
            # Use the last 12 events as the window
            window = team_events[-12:]
            
            # Calculate features (same as training)
            features = {}
            
            # === BASIC EVENT COUNTS ===
            event_counts = {}
            for event in window:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            
            # Core basketball events
            features['shots'] = event_counts.get('shot', 0)
            features['made_shots'] = len([e for e in window if e.event_type == 'shot' and getattr(e, 'shot_result', None) == 'Made'])
            features['missed_shots'] = features['shots'] - features['made_shots']
            features['rebounds'] = event_counts.get('rebound', 0)
            features['turnovers'] = event_counts.get('turnover', 0)
            features['steals'] = event_counts.get('steal', 0)
            features['blocks'] = event_counts.get('block', 0)
            features['assists'] = event_counts.get('assist', 0)
            features['fouls'] = event_counts.get('foul', 0)
            
            # === ADVANCED BASKETBALL METRICS ===
            
            # Shooting efficiency
            features['fg_percentage'] = features['made_shots'] / max(features['shots'], 1)
            features['shot_attempts_per_event'] = features['shots'] / len(window)
            
            # Offensive efficiency
            total_possessions = max(features['shots'] + features['turnovers'], 1)
            features['points_per_possession'] = (features['made_shots'] * 2) / total_possessions
            features['turnover_rate'] = features['turnovers'] / total_possessions
            
            # Defensive metrics
            features['steal_rate'] = features['steals'] / len(window)
            features['block_rate'] = features['blocks'] / len(window)
            features['defensive_events'] = features['steals'] + features['blocks']
            
            # === MOMENTUM-SPECIFIC FEATURES ===
            
            # Scoring runs
            features['scoring_run'] = self._calculate_scoring_run(window)
            features['defensive_run'] = self._calculate_defensive_run(window)
            
            # Event clustering
            features['shot_clustering'] = self._calculate_event_clustering(window, 'shot')
            features['turnover_clustering'] = self._calculate_event_clustering(window, 'turnover')
            
            # Momentum swings
            features['momentum_swings'] = self._calculate_momentum_swings(window)
            
            # === TEMPORAL FEATURES ===
            
            # Game context
            if window:
                avg_period = np.mean([e.period for e in window])
                features['avg_period'] = avg_period
                features['late_game'] = float(avg_period >= 4)
                
                # Time remaining (if available)
                if hasattr(window[0], 'time_remaining'):
                    avg_time_remaining = np.mean([getattr(e, 'time_remaining', 24) for e in window])
                    features['avg_time_remaining'] = avg_time_remaining
                    features['clutch_time'] = float(avg_time_remaining <= 5 and avg_period >= 4)
                else:
                    features['avg_time_remaining'] = 24.0
                    features['clutch_time'] = 0.0
            else:
                features['avg_period'] = 2.0
                features['late_game'] = 0.0
                features['avg_time_remaining'] = 24.0
                features['clutch_time'] = 0.0
            
            # === SEQUENCE CONTEXT ===
            
            # Recent performance trends
            window_third = len(window) // 3
            if window_third > 0:
                early_window = window[:window_third]
                late_window = window[-window_third:]
                
                early_made = len([e for e in early_window if e.event_type == 'shot' and getattr(e, 'shot_result', None) == 'Made'])
                late_made = len([e for e in late_window if e.event_type == 'shot' and getattr(e, 'shot_result', None) == 'Made'])
                
                early_shots = len([e for e in early_window if e.event_type == 'shot'])
                late_shots = len([e for e in late_window if e.event_type == 'shot'])
                
                features['shooting_trend'] = (late_made / max(late_shots, 1)) - (early_made / max(early_shots, 1))
                
                early_turnovers = len([e for e in early_window if e.event_type == 'turnover'])
                late_turnovers = len([e for e in late_window if e.event_type == 'turnover'])
                
                features['turnover_trend'] = late_turnovers - early_turnovers
            else:
                features['shooting_trend'] = 0.0
                features['turnover_trend'] = 0.0
            
            # === COMPOSITE MOMENTUM SCORE ===
            
            # Positive momentum events
            positive_events = (
                features['made_shots'] * 2 +
                features['steals'] * 3 +
                features['blocks'] * 2 +
                features['assists'] * 1.5 +
                features['rebounds'] * 0.5
            )
            
            # Negative momentum events
            negative_events = (
                features['missed_shots'] * 1 +
                features['turnovers'] * 3 +
                features['fouls'] * 0.5
            )
            
            features['momentum_score'] = positive_events - negative_events
            features['momentum_ratio'] = positive_events / max(negative_events, 1)
            
            # Convert to feature array in the correct order
            feature_array = [features.get(name, 0.0) for name in self.feature_names]
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _calculate_scoring_run(self, window: List[GameEvent]) -> int:
        """Calculate current scoring run length."""
        run_length = 0
        for event in reversed(window):
            if event.event_type == 'shot' and getattr(event, 'shot_result', None) == 'Made':
                run_length += 2  # Assume 2 points per made shot
            elif event.event_type in ['turnover', 'foul']:
                break
        return run_length
    
    def _calculate_defensive_run(self, window: List[GameEvent]) -> int:
        """Calculate current defensive run (steals + blocks)."""
        run_length = 0
        for event in reversed(window):
            if event.event_type in ['steal', 'block']:
                run_length += 1
            elif event.event_type == 'shot' and getattr(event, 'shot_result', None) == 'Made':
                break
        return run_length
    
    def _calculate_event_clustering(self, window: List[GameEvent], event_type: str) -> float:
        """Calculate clustering of specific event types."""
        if not window:
            return 0.0
        
        # Find consecutive occurrences
        consecutive_counts = []
        current_count = 0
        
        for event in window:
            if event.event_type == event_type:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        # Return max consecutive count
        return max(consecutive_counts) if consecutive_counts else 0
    
    def _calculate_momentum_swings(self, window: List[GameEvent]) -> int:
        """Calculate number of momentum swings in the window."""
        if len(window) < 3:
            return 0
        
        # Define momentum events
        momentum_sequence = []
        for event in window:
            if event.event_type == 'shot':
                if getattr(event, 'shot_result', None) == 'Made':
                    momentum_sequence.append(1)
                else:
                    momentum_sequence.append(-1)
            elif event.event_type in ['steal', 'block', 'assist']:
                momentum_sequence.append(1)
            elif event.event_type in ['turnover', 'foul']:
                momentum_sequence.append(-1)
        
        # Count direction changes
        swings = 0
        for i in range(1, len(momentum_sequence)):
            if momentum_sequence[i] != momentum_sequence[i-1]:
                swings += 1
        
        return swings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "model_type": type(self.model).__name__,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names
        }


# Global instance
_production_predictor = None


def get_production_predictor() -> ProductionMomentumPredictor:
    """Get the global production predictor instance."""
    global _production_predictor
    if _production_predictor is None:
        _production_predictor = ProductionMomentumPredictor()
    return _production_predictor