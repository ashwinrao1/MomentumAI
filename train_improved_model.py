#!/usr/bin/env python3
"""
Improved model training script implementing key recommendations.

This script demonstrates the improvements recommended in the model effectiveness analysis:
1. Better feature engineering
2. Proper train/test splits by game
3. Ensemble methods
4. Comprehensive evaluation
5. Realistic training data
"""

import argparse
import logging
import sys
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from services.historical_data_collector import create_sample_training_data
    from services.momentum_engine import MomentumEngine
    from models.game_models import GameEvent
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedMomentumPredictor:
    """
    Improved momentum predictor implementing key recommendations.
    """
    
    def __init__(self, model_path: str = "models/improved_momentum_predictor.pkl"):
        """Initialize the improved predictor."""
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.ensemble_weights = {}
        self.is_trained = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def create_realistic_training_data(self) -> List[GameEvent]:
        """Create realistic training data with proper basketball patterns."""
        logger.info("Creating realistic training data with basketball patterns")
        
        np.random.seed(42)  # For reproducibility
        sample_events = []
        
        # Create sample events for multiple games with realistic patterns
        teams = ['LAL', 'GSW', 'BOS', 'MIA', 'PHX', 'MIL', 'BKN', 'DEN', 'DAL', 'NYK']
        
        for game_num in range(40):  # 40 sample games
            # Pick two random teams
            home_team, away_team = np.random.choice(teams, 2, replace=False)
            game_id = f"realistic_game_{game_num:03d}"
            
            # Game characteristics
            competitive_game = np.random.random() > 0.2  # 80% competitive games
            
            # Create events for each team with momentum patterns
            for team_idx, team in enumerate([home_team, away_team]):
                # Team strength affects performance
                team_strength = np.random.uniform(0.45, 0.85)
                
                # Momentum tracking
                current_momentum = 0.0
                scoring_run = 0
                defensive_stops = 0
                
                for possession in range(30):  # 30 possessions per team
                    # Momentum affects performance
                    momentum_boost = current_momentum * 0.25
                    performance_level = team_strength + momentum_boost + np.random.normal(0, 0.15)
                    performance_level = max(0.15, min(0.9, performance_level))
                    
                    # Create realistic possession events
                    possession_events = []
                    
                    # Most possessions end with a shot
                    if np.random.random() < 0.88:  # 88% of possessions have shots
                        shot_made = np.random.random() < performance_level
                        
                        # Shot type distribution
                        if shot_made:
                            if np.random.random() < 0.35:  # 35% are 3-pointers
                                shot_desc = f"{team} makes 3pt shot"
                                points = 3
                                momentum_gain = 0.4
                            else:
                                shot_desc = f"{team} makes 2pt shot"
                                points = 2
                                momentum_gain = 0.25
                            
                            scoring_run += points
                            current_momentum = min(1.0, current_momentum + momentum_gain)
                            
                        else:
                            if np.random.random() < 0.35:
                                shot_desc = f"{team} misses 3pt shot"
                            else:
                                shot_desc = f"{team} misses 2pt shot"
                            points = 0
                            
                            # Missed shots can break momentum
                            if scoring_run > 6:  # Long scoring run
                                current_momentum = max(-0.3, current_momentum - 0.3)
                                scoring_run = 0
                            else:
                                current_momentum = max(-0.5, current_momentum - 0.15)
                        
                        # Create shot event
                        shot_event = GameEvent(
                            event_id=f"{game_id}_{team}_{possession}_shot",
                            game_id=game_id,
                            team_tricode=team,
                            player_name=f"Player_{possession % 10}",
                            event_type='shot',
                            clock=f"{11 - (possession // 6)}:{max(0, 60 - possession * 2) % 60:02d}",
                            period=min(4, 1 + (possession // 30)),
                            points_total=possession * 2 + points,
                            shot_result='Made' if shot_made else 'Missed',
                            timestamp=datetime.utcnow(),
                            description=shot_desc
                        )
                        possession_events.append(shot_event)
                    
                    # Rebound events
                    if possession_events and possession_events[-1].shot_result == 'Missed':
                        if np.random.random() < 0.75:  # 75% of misses have rebounds
                            # Offensive rebound chance increases with momentum
                            offensive_rebound_chance = 0.25 + max(0, current_momentum * 0.15)
                            is_offensive = np.random.random() < offensive_rebound_chance
                            
                            rebound_type = "offensive" if is_offensive else "defensive"
                            rebound_event = GameEvent(
                                event_id=f"{game_id}_{team}_{possession}_rebound",
                                game_id=game_id,
                                team_tricode=team,
                                player_name=f"Player_{(possession + 1) % 10}",
                                event_type='rebound',
                                clock=possession_events[-1].clock,
                                period=possession_events[-1].period,
                                points_total=possession_events[-1].points_total,
                                shot_result=None,
                                timestamp=datetime.utcnow(),
                                description=f"{team} {rebound_type} rebound"
                            )
                            possession_events.append(rebound_event)
                            
                            # Offensive rebounds boost momentum
                            if is_offensive:
                                current_momentum = min(1.0, current_momentum + 0.2)
                    
                    # Assist events
                    if possession_events and possession_events[-1].shot_result == 'Made':
                        # Assist chance increases with good ball movement (momentum)
                        assist_chance = 0.55 + max(0, current_momentum * 0.2)
                        if np.random.random() < assist_chance:
                            assist_event = GameEvent(
                                event_id=f"{game_id}_{team}_{possession}_assist",
                                game_id=game_id,
                                team_tricode=team,
                                player_name=f"Player_{(possession + 2) % 10}",
                                event_type='assist',
                                clock=possession_events[-1].clock,
                                period=possession_events[-1].period,
                                points_total=possession_events[-1].points_total,
                                shot_result=None,
                                timestamp=datetime.utcnow(),
                                description=f"{team} assist"
                            )
                            possession_events.append(assist_event)
                    
                    # Turnover events (momentum affects turnover rate)
                    turnover_chance = 0.16 - max(0, current_momentum * 0.08)  # Less turnovers with momentum
                    if not possession_events and np.random.random() < turnover_chance:
                        turnover_event = GameEvent(
                            event_id=f"{game_id}_{team}_{possession}_turnover",
                            game_id=game_id,
                            team_tricode=team,
                            player_name=f"Player_{possession % 10}",
                            event_type='turnover',
                            clock=f"{11 - (possession // 6)}:{max(0, 60 - possession * 2) % 60:02d}",
                            period=min(4, 1 + (possession // 30)),
                            points_total=possession * 2,
                            shot_result=None,
                            timestamp=datetime.utcnow(),
                            description=f"{team} turnover"
                        )
                        possession_events.append(turnover_event)
                        
                        # Turnovers kill momentum
                        current_momentum = max(-1.0, current_momentum - 0.5)
                        scoring_run = 0
                        defensive_stops = 0
                    
                    # Steal events (defensive momentum plays)
                    if np.random.random() < 0.09:  # 9% steal rate
                        steal_event = GameEvent(
                            event_id=f"{game_id}_{team}_{possession}_steal",
                            game_id=game_id,
                            team_tricode=team,
                            player_name=f"Player_{(possession + 3) % 10}",
                            event_type='steal',
                            clock=f"{11 - (possession // 6)}:{max(0, 60 - possession * 2) % 60:02d}",
                            period=min(4, 1 + (possession // 30)),
                            points_total=possession * 2,
                            shot_result=None,
                            timestamp=datetime.utcnow(),
                            description=f"{team} steal"
                        )
                        possession_events.append(steal_event)
                        
                        # Steals create momentum
                        current_momentum = min(1.0, current_momentum + 0.45)
                        defensive_stops += 1
                    
                    # Block events
                    if np.random.random() < 0.05:  # 5% block rate
                        block_event = GameEvent(
                            event_id=f"{game_id}_{team}_{possession}_block",
                            game_id=game_id,
                            team_tricode=team,
                            player_name=f"Player_{(possession + 4) % 10}",
                            event_type='block',
                            clock=f"{11 - (possession // 6)}:{max(0, 60 - possession * 2) % 60:02d}",
                            period=min(4, 1 + (possession // 30)),
                            points_total=possession * 2,
                            shot_result=None,
                            timestamp=datetime.utcnow(),
                            description=f"{team} block"
                        )
                        possession_events.append(block_event)
                        
                        # Blocks create momentum
                        current_momentum = min(1.0, current_momentum + 0.35)
                        defensive_stops += 1
                    
                    # Add all possession events
                    sample_events.extend(possession_events)
                    
                    # Natural momentum decay
                    current_momentum *= 0.96
                    
                    # Reset long scoring runs occasionally
                    if scoring_run > 12 and np.random.random() < 0.3:
                        scoring_run = max(0, scoring_run - 4)
        
        logger.info(f"Created {len(sample_events)} realistic events from 40 games")
        return sample_events
    
    def extract_enhanced_features(self, events_data: List[GameEvent]) -> pd.DataFrame:
        """Extract enhanced features from game events."""
        logger.info(f"Extracting enhanced features from {len(events_data)} events")
        
        # Group events by game
        games_data = {}
        for event in events_data:
            if event.game_id not in games_data:
                games_data[event.game_id] = []
            games_data[event.game_id].append(event)
        
        # Process each game
        training_data = []
        momentum_engine = MomentumEngine(rolling_window_size=8)
        
        for game_id, game_events in games_data.items():
            try:
                game_training_data = self._extract_game_features(
                    game_events, momentum_engine, game_id
                )
                training_data.extend(game_training_data)
            except Exception as e:
                logger.warning(f"Error processing game {game_id}: {e}")
                continue
        
        df = pd.DataFrame(training_data)
        logger.info(f"Extracted {len(df)} training examples with enhanced features")
        
        return df
    
    def _extract_game_features(
        self,
        game_events: List[GameEvent],
        momentum_engine: MomentumEngine,
        game_id: str
    ) -> List[Dict[str, Any]]:
        """Extract enhanced features from a single game."""
        training_examples = []
        
        # Segment possessions
        possessions = momentum_engine.segment_possessions(game_events)
        
        if len(possessions) < 12:
            return []
        
        # Group by team
        team_possessions = {}
        for possession in possessions:
            team = possession.team_tricode
            if team not in team_possessions:
                team_possessions[team] = []
            team_possessions[team].append(possession)
        
        # Extract features for each team
        for team, team_poss in team_possessions.items():
            if len(team_poss) < 10:
                continue
            
            # Calculate basic features
            basic_features = momentum_engine.calculate_possession_features(team_poss)
            
            # Create training examples with sliding window
            for i in range(8, len(basic_features) - 1):
                current_window = basic_features[i-8:i]
                next_possession = basic_features[i]
                
                # Enhanced momentum continuation logic
                momentum_continued = self._determine_enhanced_momentum_continuation(
                    current_window, next_possession, team_poss[i-8:i+1]
                )
                
                # Calculate enhanced features
                enhanced_features = self._calculate_enhanced_features(
                    current_window, team_poss[i-8:i], game_id
                )
                enhanced_features['momentum_continued'] = int(momentum_continued)
                enhanced_features['game_id'] = game_id
                enhanced_features['team'] = team
                
                training_examples.append(enhanced_features)
        
        return training_examples
    
    def _determine_enhanced_momentum_continuation(
        self,
        current_window: List,
        next_possession,
        possession_context: List
    ) -> bool:
        """Determine momentum continuation using enhanced basketball logic."""
        
        # Calculate current momentum indicators
        recent_scoring = np.mean([f.points_scored for f in current_window[-4:]])
        recent_efficiency = np.mean([f.fg_percentage for f in current_window[-4:]])
        recent_turnovers = np.mean([f.turnovers for f in current_window[-4:]])
        
        # Next possession performance
        next_scoring = next_possession.points_scored
        next_efficiency = next_possession.fg_percentage
        next_turnovers = next_possession.turnovers
        
        # Count momentum events in recent possessions
        momentum_events = 0
        for poss in possession_context[-4:]:
            for event in poss.events:
                if event.event_type in ['steal', 'block']:
                    momentum_events += 2
                elif event.event_type == 'shot' and event.shot_result == 'Made':
                    if '3pt' in event.description.lower():
                        momentum_events += 1.5
                    else:
                        momentum_events += 1
                elif event.event_type == 'assist':
                    momentum_events += 0.5
        
        # Multiple criteria for momentum continuation
        criteria_score = 0.0
        
        # Scoring criterion (40% weight)
        if next_scoring >= recent_scoring * 0.8:  # Allow some variance
            criteria_score += 0.4
        
        # Efficiency criterion (25% weight)
        if next_efficiency >= recent_efficiency * 0.7:
            criteria_score += 0.25
        
        # Turnover criterion (15% weight) - fewer turnovers is better
        if next_turnovers <= recent_turnovers * 1.2:
            criteria_score += 0.15
        
        # Momentum events criterion (20% weight)
        if momentum_events >= 2:  # Significant momentum events
            criteria_score += 0.2
        
        # Momentum continues if score >= 0.5 (50% threshold)
        return criteria_score >= 0.5
    
    def _calculate_enhanced_features(
        self,
        possession_window: List,
        possession_context: List,
        game_id: str
    ) -> Dict[str, float]:
        """Calculate enhanced features for training."""
        features = {}
        
        # Basic statistical features
        features['avg_points_scored'] = np.mean([f.points_scored for f in possession_window])
        features['avg_fg_percentage'] = np.mean([f.fg_percentage for f in possession_window])
        features['avg_turnovers'] = np.mean([f.turnovers for f in possession_window])
        features['avg_rebounds'] = np.mean([f.rebounds for f in possession_window])
        features['avg_fouls'] = np.mean([f.fouls for f in possession_window])
        features['avg_pace'] = np.mean([f.pace for f in possession_window])
        
        # Trend features (compare first half vs second half of window)
        if len(possession_window) >= 6:
            early = possession_window[:4]
            late = possession_window[-4:]
            
            features['points_trend'] = (
                np.mean([f.points_scored for f in late]) -
                np.mean([f.points_scored for f in early])
            )
            features['efficiency_trend'] = (
                np.mean([f.fg_percentage for f in late]) -
                np.mean([f.fg_percentage for f in early])
            )
            features['turnover_trend'] = (
                np.mean([f.turnovers for f in late]) -
                np.mean([f.turnovers for f in early])
            )
        else:
            features['points_trend'] = 0.0
            features['efficiency_trend'] = 0.0
            features['turnover_trend'] = 0.0
        
        # Enhanced basketball features
        features['scoring_consistency'] = 1.0 / (1.0 + np.std([f.points_scored for f in possession_window]))
        features['efficiency_consistency'] = 1.0 / (1.0 + np.std([f.fg_percentage for f in possession_window]))
        
        # Momentum-specific features
        features['momentum_events_score'] = self._calculate_momentum_events_score(possession_context)
        features['scoring_run_length'] = self._calculate_scoring_run_length(possession_context)
        features['defensive_stops'] = self._calculate_defensive_stops(possession_context)
        features['energy_level'] = self._calculate_energy_level(possession_context)
        
        # Advanced features
        features['shot_quality'] = self._calculate_shot_quality(possession_context)
        features['ball_movement'] = self._calculate_ball_movement(possession_context)
        features['defensive_pressure'] = self._calculate_defensive_pressure(possession_context)
        
        # Situational features
        if possession_context:
            latest_poss = possession_context[-1]
            features['period'] = latest_poss.events[0].period if latest_poss.events else 1
            features['time_pressure'] = min(1.0, features['period'] / 4.0)  # Increases with period
        else:
            features['period'] = 1
            features['time_pressure'] = 0.25
        
        return features
    
    def _calculate_momentum_events_score(self, possessions: List) -> float:
        """Calculate momentum events score."""
        score = 0.0
        for poss in possessions[-4:]:  # Last 4 possessions
            for event in poss.events:
                if event.event_type == 'steal':
                    score += 3.0
                elif event.event_type == 'block':
                    score += 2.5
                elif event.event_type == 'shot' and event.shot_result == 'Made':
                    if '3pt' in event.description.lower():
                        score += 2.0
                    else:
                        score += 1.0
                elif event.event_type == 'assist':
                    score += 1.0
                elif event.event_type == 'turnover':
                    score -= 2.0
        return score
    
    def _calculate_scoring_run_length(self, possessions: List) -> int:
        """Calculate current scoring run length."""
        run_length = 0
        for poss in reversed(possessions[-6:]):  # Last 6 possessions
            points = sum(
                2 if '2pt' in event.description.lower() else 3 if '3pt' in event.description.lower() else 1
                for event in poss.events
                if event.event_type == 'shot' and event.shot_result == 'Made'
            )
            if points > 0:
                run_length += points
            else:
                break
        return run_length
    
    def _calculate_defensive_stops(self, possessions: List) -> int:
        """Calculate defensive stops."""
        stops = 0
        for poss in possessions[-4:]:
            has_steal_or_block = any(
                event.event_type in ['steal', 'block'] for event in poss.events
            )
            if has_steal_or_block:
                stops += 1
        return stops
    
    def _calculate_energy_level(self, possessions: List) -> float:
        """Calculate team energy level."""
        energy = 0.0
        for poss in possessions[-3:]:
            for event in poss.events:
                if event.event_type in ['steal', 'block']:
                    energy += 2.0
                elif event.event_type == 'shot' and event.shot_result == 'Made':
                    if '3pt' in event.description.lower():
                        energy += 1.5
                    else:
                        energy += 0.5
        return min(10.0, energy)
    
    def _calculate_shot_quality(self, possessions: List) -> float:
        """Calculate shot quality metric."""
        shots = []
        assists = 0
        
        for poss in possessions[-3:]:
            for event in poss.events:
                if event.event_type == 'shot':
                    shots.append(event)
                elif event.event_type == 'assist':
                    assists += 1
        
        if not shots:
            return 0.5
        
        # Quality based on makes and assists
        made_shots = [s for s in shots if s.shot_result == 'Made']
        quality = len(made_shots) / len(shots)
        
        # Bonus for assists (indicates good ball movement)
        assist_bonus = min(0.3, assists * 0.1)
        
        return min(1.0, quality + assist_bonus)
    
    def _calculate_ball_movement(self, possessions: List) -> float:
        """Calculate ball movement quality."""
        total_assists = 0
        total_possessions = len(possessions[-4:])
        
        for poss in possessions[-4:]:
            assists_in_poss = sum(
                1 for event in poss.events if event.event_type == 'assist'
            )
            total_assists += assists_in_poss
        
        return total_assists / max(1, total_possessions)
    
    def _calculate_defensive_pressure(self, possessions: List) -> float:
        """Calculate defensive pressure applied."""
        pressure = 0.0
        for poss in possessions[-3:]:
            for event in poss.events:
                if event.event_type == 'steal':
                    pressure += 3.0
                elif event.event_type == 'block':
                    pressure += 2.5
                elif event.event_type == 'turnover':
                    pressure += 1.0  # Might indicate pressure
        return pressure
    
    def train_ensemble_model(
        self,
        training_data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train ensemble model with proper game-based splits."""
        logger.info("Training ensemble model with game-based splits")
        
        # Game-based splitting to prevent data leakage
        unique_games = training_data['game_id'].unique()
        train_games, test_games = train_test_split(
            unique_games, test_size=test_size, random_state=random_state
        )
        
        # Create data splits
        train_data = training_data[training_data['game_id'].isin(train_games)]
        test_data = training_data[training_data['game_id'].isin(test_games)]
        
        logger.info(f"Train: {len(train_data)} examples from {len(train_games)} games")
        logger.info(f"Test: {len(test_data)} examples from {len(test_games)} games")
        
        # Prepare features
        feature_columns = [col for col in training_data.columns 
                          if col not in ['momentum_continued', 'game_id', 'team']]
        self.feature_names = feature_columns
        
        X_train = train_data[feature_columns].values
        y_train = train_data['momentum_continued'].values
        X_test = test_data[feature_columns].values
        y_test = test_data['momentum_continued'].values
        
        # Model configurations
        model_configs = {
            'logistic': {
                'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                'use_scaler': True
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight='balanced',
                    max_depth=10, min_samples_split=5
                ),
                'use_scaler': False
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100, random_state=42, max_depth=6,
                    learning_rate=0.1, subsample=0.8
                ),
                'use_scaler': False
            }
        }
        
        # Train individual models
        model_results = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name}")
            
            model = config['model']
            
            # Scale features if needed
            if config['use_scaler']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[model_name] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                self.scalers[model_name] = None
            
            # Train model
            model.fit(X_train_scaled, y_train)
            self.models[model_name] = model
            
            # Evaluate
            test_pred = model.predict(X_test_scaled)
            test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred)
            recall = recall_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred)
            auc = roc_auc_score(y_test, test_proba)
            
            model_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Calculate ensemble weights based on AUC
        auc_scores = {name: results['auc'] for name, results in model_results.items()}
        auc_values = np.array(list(auc_scores.values()))
        weights = auc_values / np.sum(auc_values)  # Simple proportional weighting
        self.ensemble_weights = dict(zip(auc_scores.keys(), weights))
        
        # Evaluate ensemble
        ensemble_results = self._evaluate_ensemble(X_test, y_test)
        
        # Baseline comparison
        baseline_results = self._evaluate_baseline(X_test, y_test)
        
        self.is_trained = True
        
        return {
            'individual_models': model_results,
            'ensemble_results': ensemble_results,
            'baseline_results': baseline_results,
            'ensemble_weights': self.ensemble_weights,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def _evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble model."""
        ensemble_proba = np.zeros(len(X_test))
        
        for model_name, weight in self.ensemble_weights.items():
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            model_proba = model.predict_proba(X_test_scaled)[:, 1]
            ensemble_proba += weight * model_proba
        
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred),
            'recall': recall_score(y_test, ensemble_pred),
            'f1': f1_score(y_test, ensemble_pred),
            'auc': roc_auc_score(y_test, ensemble_proba)
        }
    
    def _evaluate_baseline(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate baseline models."""
        dummy_frequent = DummyClassifier(strategy='most_frequent')
        dummy_frequent.fit(X_test, y_test)
        frequent_acc = accuracy_score(y_test, dummy_frequent.predict(X_test))
        
        dummy_random = DummyClassifier(strategy='uniform', random_state=42)
        dummy_random.fit(X_test, y_test)
        random_acc = accuracy_score(y_test, dummy_random.predict(X_test))
        
        return {
            'most_frequent_accuracy': frequent_acc,
            'random_accuracy': random_acc
        }
    
    def save_model(self) -> bool:
        """Save the trained model."""
        if not self.is_trained:
            return False
        
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'ensemble_weights': self.ensemble_weights,
                'feature_names': self.feature_names,
                'trained_at': datetime.utcnow().isoformat(),
                'model_type': 'improved_ensemble'
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Improved MomentumML model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/improved_momentum_predictor.pkl",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Improved MomentumML model training")
    
    try:
        # Create predictor
        predictor = ImprovedMomentumPredictor(args.model_path)
        
        # Create realistic training data
        training_events = predictor.create_realistic_training_data()
        
        # Extract enhanced features
        training_data = predictor.extract_enhanced_features(training_events)
        
        logger.info(f"Training data shape: {training_data.shape}")
        logger.info(f"Class distribution: {training_data['momentum_continued'].value_counts().to_dict()}")
        
        # Train ensemble model
        results = predictor.train_ensemble_model(training_data)
        
        # Save model
        predictor.save_model()
        
        # Print results
        logger.info("=" * 60)
        logger.info("IMPROVED MODEL TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        # Individual model results
        logger.info("Individual Model Performance:")
        for model_name, metrics in results['individual_models'].items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"    AUC: {metrics['auc']:.4f}")
            logger.info(f"    F1: {metrics['f1']:.4f}")
        
        # Ensemble results
        ensemble = results['ensemble_results']
        logger.info("Ensemble Model Performance:")
        logger.info(f"  Test Accuracy: {ensemble['accuracy']:.4f}")
        logger.info(f"  Test Precision: {ensemble['precision']:.4f}")
        logger.info(f"  Test Recall: {ensemble['recall']:.4f}")
        logger.info(f"  Test F1-Score: {ensemble['f1']:.4f}")
        logger.info(f"  Test AUC: {ensemble['auc']:.4f}")
        
        # Baseline comparison
        baseline = results['baseline_results']
        improvement = ensemble['accuracy'] - baseline['most_frequent_accuracy']
        
        logger.info("Baseline Comparison:")
        logger.info(f"  Most Frequent Class: {baseline['most_frequent_accuracy']:.4f}")
        logger.info(f"  Random Prediction: {baseline['random_accuracy']:.4f}")
        logger.info(f"  Ensemble Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
        
        # Performance assessment
        if ensemble['accuracy'] > 0.7:
            logger.info("✅ EXCELLENT: Model shows strong performance!")
        elif ensemble['accuracy'] > 0.6:
            logger.info("✅ GOOD: Model shows decent performance")
        elif ensemble['accuracy'] > 0.55:
            logger.info("⚠️  MODERATE: Model shows some predictive ability")
        else:
            logger.info("❌ POOR: Model needs improvement")
        
        if improvement > 0.1:
            logger.info("✅ Significant improvement over baseline!")
        elif improvement > 0.05:
            logger.info("✅ Moderate improvement over baseline")
        else:
            logger.info("⚠️  Limited improvement over baseline")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()