#!/usr/bin/env python3
"""
Enhanced Momentum Predictor with Game-Level Momentum Analysis

Features:
- Individual team momentum (existing)
- Combined game momentum (new)
- Momentum confidence scoring
- Team highlighting based on momentum
- Uses the new improved V2 model
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedMomentumPredictor:
    """
    Enhanced momentum predictor that provides both individual team momentum
    and overall game momentum analysis.
    """
    
    def __init__(self, model_path: str = "models/improved_v2/improved_v2_momentum_model_20251025_135412.pkl"):
        self.model_path = model_path
        self.model_package = None
        self.load_model()
    
    def load_model(self):
        """Load the improved V2 momentum model."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_package = pickle.load(f)
            
            logger.info(f"Loaded improved V2 momentum model: {self.model_package.get('model_type', 'unknown')}")
            logger.info(f"Best model: {self.model_package.get('best_model_name', 'unknown')}")
            logger.info(f"Features: {len(self.model_package.get('selected_features', []))}")
            
        except Exception as e:
            logger.error(f"Error loading momentum model: {e}")
            self.model_package = None
    
    def extract_features_from_events(self, events: List[Dict]) -> pd.DataFrame:
        """Extract features from game events for prediction."""
        if not events:
            return pd.DataFrame()
        
        # Convert events to DataFrame
        df = pd.DataFrame(events)
        
        # Ensure required columns exist
        required_cols = ['event_type', 'team_tricode', 'timestamp', 'shot_result', 'points_total']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create basic event features
        df['is_shot'] = (df['event_type'] == 'shot').astype(int)
        df['is_made_shot'] = ((df['event_type'] == 'shot') & (df['shot_result'] == 'Made')).astype(int)
        df['is_missed_shot'] = ((df['event_type'] == 'shot') & (df['shot_result'] == 'Missed')).astype(int)
        df['is_rebound'] = (df['event_type'] == 'rebound').astype(int)
        df['is_turnover'] = (df['event_type'] == 'turnover').astype(int)
        df['is_steal'] = (df['event_type'] == 'steal').astype(int)
        df['is_block'] = (df['event_type'] == 'block').astype(int)
        df['is_assist'] = (df['event_type'] == 'assist').astype(int)
        df['is_foul'] = (df['event_type'] == 'foul').astype(int)
        
        # Event values
        df['event_value'] = 0
        df.loc[df['is_made_shot'] == 1, 'event_value'] = df.loc[df['is_made_shot'] == 1, 'points_total'].fillna(2)
        df.loc[df['is_missed_shot'] == 1, 'event_value'] = -1
        df.loc[df['is_steal'] == 1, 'event_value'] = 2
        df.loc[df['is_block'] == 1, 'event_value'] = 1.5
        df.loc[df['is_assist'] == 1, 'event_value'] = 1
        df.loc[df['is_turnover'] == 1, 'event_value'] = -2
        df.loc[df['is_foul'] == 1, 'event_value'] = -0.5
        
        # Add period if not present (estimate from timestamp)
        if 'period' not in df.columns:
            df['period'] = 1  # Default to first quarter
        
        # Game context
        df['early_game'] = (df['period'] <= 2).astype(int)
        df['late_game'] = (df['period'] >= 4).astype(int)
        df['clutch_time'] = (df['period'] >= 4).astype(int)
        
        # Rolling features for each team
        grouped = df.groupby('team_tricode')
        
        # Rolling statistics (last 5, 10, 15 events)
        for window in [5, 10, 15]:
            df[f'shots_last_{window}'] = grouped['is_shot'].rolling(window, min_periods=1).sum().values
            df[f'made_shots_last_{window}'] = grouped['is_made_shot'].rolling(window, min_periods=1).sum().values
            df[f'momentum_last_{window}'] = grouped['event_value'].rolling(window, min_periods=1).sum().values
            df[f'turnovers_last_{window}'] = grouped['is_turnover'].rolling(window, min_periods=1).sum().values
        
        # Shooting efficiency
        df['fg_pct_last_5'] = df['made_shots_last_5'] / (df['shots_last_5'] + 0.01)
        df['fg_pct_last_10'] = df['made_shots_last_10'] / (df['shots_last_10'] + 0.01)
        df['fg_pct_last_15'] = df['made_shots_last_15'] / (df['shots_last_15'] + 0.01)
        
        # Momentum trends
        df['momentum_change'] = grouped['event_value'].diff().values
        df['momentum_acceleration'] = grouped['momentum_change'].diff().values
        
        # Situational features
        df['high_value_event'] = (abs(df['event_value']) >= 2).astype(int)
        df['positive_event'] = (df['event_value'] > 0).astype(int)
        df['negative_event'] = (df['event_value'] < 0).astype(int)
        
        # Interaction features
        df['clutch_momentum'] = df['clutch_time'] * df['event_value']
        df['late_game_momentum'] = df['late_game'] * df['event_value']
        df['early_game_momentum'] = df['early_game'] * df['event_value']
        
        # Streak features
        df['positive_streak'] = (df['event_value'] > 0).astype(int)
        df['positive_streak_length'] = grouped['positive_streak'].rolling(10, min_periods=1).sum().values
        
        return df
    
    def predict_team_momentum(self, team_events: pd.DataFrame) -> Dict:
        """Predict momentum for a specific team."""
        if self.model_package is None or team_events.empty:
            return {
                'momentum_probability': 0.5,
                'momentum_confidence': 'low',
                'momentum_direction': 'neutral',
                'recent_performance': {}
            }
        
        try:
            # Get the latest event for this team
            latest_event = team_events.iloc[-1]
            
            # Extract features for prediction
            selected_features = self.model_package['selected_features']
            
            # Create feature vector
            feature_vector = []
            for feature in selected_features:
                if feature in latest_event:
                    feature_vector.append(latest_event[feature])
                else:
                    feature_vector.append(0)  # Default value for missing features
            
            # Scale features
            scaler = self.model_package['scaler']
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Select features
            feature_selector = self.model_package['feature_selector']
            feature_vector_selected = feature_selector.transform(feature_vector_scaled)
            
            # Predict using best model
            best_model = self.model_package['best_model']
            momentum_probability = best_model.predict_proba(feature_vector_selected)[0][1]
            
            # Determine confidence and direction
            if momentum_probability >= 0.7:
                confidence = 'high'
                direction = 'positive'
            elif momentum_probability >= 0.5:
                confidence = 'medium'
                direction = 'positive'
            elif momentum_probability >= 0.3:
                confidence = 'medium'
                direction = 'neutral'
            else:
                confidence = 'low'
                direction = 'negative'
            
            # Recent performance metrics
            recent_performance = {
                'recent_fg_pct': latest_event.get('fg_pct_last_10', 0),
                'recent_momentum': latest_event.get('momentum_last_10', 0),
                'recent_turnovers': latest_event.get('turnovers_last_10', 0),
                'positive_events': latest_event.get('positive_streak_length', 0)
            }
            
            return {
                'momentum_probability': float(momentum_probability),
                'momentum_confidence': confidence,
                'momentum_direction': direction,
                'recent_performance': recent_performance
            }
            
        except Exception as e:
            logger.error(f"Error predicting team momentum: {e}")
            return {
                'momentum_probability': 0.5,
                'momentum_confidence': 'low',
                'momentum_direction': 'neutral',
                'recent_performance': {}
            }
    
    def analyze_game_momentum(self, events: List[Dict]) -> Dict:
        """
        Analyze overall game momentum and determine which team has momentum.
        
        Returns comprehensive momentum analysis including:
        - Individual team momentum
        - Overall game momentum
        - Which team has momentum advantage
        - Momentum confidence scores
        """
        if not events:
            return self._default_momentum_analysis()
        
        try:
            # Extract features from events
            df = self.extract_features_from_events(events)
            
            if df.empty:
                return self._default_momentum_analysis()
            
            # Get unique teams
            teams = df['team_tricode'].unique()
            
            if len(teams) < 2:
                return self._default_momentum_analysis()
            
            team1, team2 = teams[0], teams[1]
            
            # Analyze momentum for each team
            team1_events = df[df['team_tricode'] == team1]
            team2_events = df[df['team_tricode'] == team2]
            
            team1_momentum = self.predict_team_momentum(team1_events)
            team2_momentum = self.predict_team_momentum(team2_events)
            
            # Calculate overall game momentum
            team1_prob = team1_momentum['momentum_probability']
            team2_prob = team2_momentum['momentum_probability']
            
            # Determine which team has momentum advantage
            momentum_diff = team1_prob - team2_prob
            
            if abs(momentum_diff) < 0.1:
                game_momentum = 'balanced'
                momentum_team = None
                momentum_strength = 'neutral'
            elif momentum_diff > 0:
                game_momentum = team1
                momentum_team = team1
                momentum_strength = 'strong' if abs(momentum_diff) > 0.3 else 'moderate'
            else:
                game_momentum = team2
                momentum_team = team2
                momentum_strength = 'strong' if abs(momentum_diff) > 0.3 else 'moderate'
            
            # Calculate game momentum percentage
            total_momentum = team1_prob + team2_prob
            if total_momentum > 0:
                team1_percentage = (team1_prob / total_momentum) * 100
                team2_percentage = (team2_prob / total_momentum) * 100
            else:
                team1_percentage = team2_percentage = 50
            
            # Recent game flow analysis
            recent_events = df.tail(20)  # Last 20 events
            recent_momentum_by_team = recent_events.groupby('team_tricode')['event_value'].sum()
            
            # Momentum streaks
            team1_recent_momentum = recent_momentum_by_team.get(team1, 0)
            team2_recent_momentum = recent_momentum_by_team.get(team2, 0)
            
            return {
                'game_momentum': {
                    'leading_team': momentum_team,
                    'momentum_direction': game_momentum,
                    'momentum_strength': momentum_strength,
                    'confidence': max(team1_prob, team2_prob)
                },
                'team_momentum': {
                    team1: {
                        **team1_momentum,
                        'momentum_percentage': team1_percentage,
                        'recent_momentum_score': float(team1_recent_momentum)
                    },
                    team2: {
                        **team2_momentum,
                        'momentum_percentage': team2_percentage,
                        'recent_momentum_score': float(team2_recent_momentum)
                    }
                },
                'momentum_comparison': {
                    'difference': abs(momentum_diff),
                    'advantage': momentum_team,
                    'strength': momentum_strength
                },
                'recent_flow': {
                    'last_20_events': {
                        team1: float(team1_recent_momentum),
                        team2: float(team2_recent_momentum)
                    },
                    'trending_team': team1 if team1_recent_momentum > team2_recent_momentum else team2
                },
                'analysis_timestamp': datetime.now().isoformat(),
                'events_analyzed': len(events),
                'model_info': {
                    'model_type': self.model_package.get('model_type', 'unknown'),
                    'confidence_level': 'high' if max(team1_prob, team2_prob) > 0.7 else 'medium'
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing game momentum: {e}")
            return self._default_momentum_analysis()
    
    def _default_momentum_analysis(self) -> Dict:
        """Return default momentum analysis when no data is available."""
        return {
            'game_momentum': {
                'leading_team': None,
                'momentum_direction': 'balanced',
                'momentum_strength': 'neutral',
                'confidence': 0.5
            },
            'team_momentum': {},
            'momentum_comparison': {
                'difference': 0,
                'advantage': None,
                'strength': 'neutral'
            },
            'recent_flow': {
                'last_20_events': {},
                'trending_team': None
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'events_analyzed': 0,
            'model_info': {
                'model_type': 'none',
                'confidence_level': 'low'
            }
        }
    
    def get_momentum_visualization_data(self, events: List[Dict]) -> Dict:
        """
        Get data specifically formatted for frontend momentum visualization.
        """
        analysis = self.analyze_game_momentum(events)
        
        # Format for frontend consumption
        return {
            'momentum_meter': {
                'leading_team': analysis['game_momentum']['leading_team'],
                'strength': analysis['game_momentum']['momentum_strength'],
                'confidence': analysis['game_momentum']['confidence']
            },
            'team_highlights': {
                team: {
                    'has_momentum': team == analysis['game_momentum']['leading_team'],
                    'momentum_level': data['momentum_confidence'],
                    'percentage': data['momentum_percentage'],
                    'recent_score': data['recent_momentum_score']
                }
                for team, data in analysis['team_momentum'].items()
            },
            'momentum_bar': {
                team: data['momentum_percentage']
                for team, data in analysis['team_momentum'].items()
            },
            'recent_trend': analysis['recent_flow']['trending_team'],
            'last_updated': analysis['analysis_timestamp']
        }


# Create global instance
enhanced_predictor = EnhancedMomentumPredictor()


def get_enhanced_momentum_analysis(events: List[Dict]) -> Dict:
    """
    Main function to get enhanced momentum analysis.
    Used by the API endpoints.
    """
    return enhanced_predictor.analyze_game_momentum(events)


def get_momentum_visualization_data(events: List[Dict]) -> Dict:
    """
    Get momentum data formatted for frontend visualization.
    """
    return enhanced_predictor.get_momentum_visualization_data(events)