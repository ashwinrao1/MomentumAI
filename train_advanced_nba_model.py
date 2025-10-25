#!/usr/bin/env python3
"""
Advanced NBA Momentum Model Training

This script implements sophisticated feature engineering and model architectures
specifically designed for NBA momentum prediction using real play-by-play data.
"""

import argparse
import logging
import pickle
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from models.game_models import GameEvent
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedNBAMomentumTrainer:
    """Advanced trainer for NBA momentum prediction with sophisticated features."""
    
    def __init__(self, output_dir: str = "models/advanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.models = self._get_model_configs()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def _get_model_configs(self) -> Dict[str, Any]:
        """Get advanced model configurations."""
        configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=42, max_iter=2000, class_weight='balanced',
                    C=0.1, penalty='l2'
                ),
                'scaler': StandardScaler(),
                'description': 'Regularized Logistic Regression'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight='balanced',
                    max_depth=12, min_samples_split=20, min_samples_leaf=10,
                    max_features='sqrt', n_jobs=-1
                ),
                'scaler': None,
                'description': 'Optimized Random Forest'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=200, random_state=42, max_depth=6,
                    learning_rate=0.05, subsample=0.8, max_features='sqrt'
                ),
                'scaler': None,
                'description': 'Gradient Boosting with regularization'
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=42, n_estimators=300, max_depth=6,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=3.0,
                    eval_metric='logloss', n_jobs=-1
                ),
                'scaler': None,
                'description': 'XGBoost with class balancing'
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            configs['lightgbm'] = {
                'model': lgb.LGBMClassifier(
                    random_state=42, n_estimators=300, max_depth=6,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, class_weight='balanced',
                    verbose=-1, n_jobs=-1
                ),
                'scaler': None,
                'description': 'LightGBM with balanced classes'
            }
        
        return configs
    
    def load_and_preprocess_data(self, dataset_path: str) -> pd.DataFrame:
        """Load and preprocess NBA data with advanced feature engineering."""
        logger.info(f"Loading and preprocessing data from {dataset_path}")
        
        # Load events
        with open(dataset_path, 'rb') as f:
            events = pickle.load(f)
        
        logger.info(f"Loaded {len(events):,} events from {len(set(e.game_id for e in events))} games")
        
        # Advanced feature extraction
        training_data = self._extract_advanced_features(events)
        
        return training_data
    
    def _extract_advanced_features(self, events: List[GameEvent]) -> pd.DataFrame:
        """Extract advanced basketball-specific features."""
        logger.info("Extracting advanced basketball features")
        
        # Group by game and team
        game_team_sequences = {}
        for event in events:
            key = (event.game_id, event.team_tricode)
            if key not in game_team_sequences:
                game_team_sequences[key] = []
            game_team_sequences[key].append(event)
        
        # Sort events within each sequence
        for key in game_team_sequences:
            game_team_sequences[key].sort(key=lambda x: (x.period, x.event_id))
        
        training_examples = []
        
        for (game_id, team), sequence in game_team_sequences.items():
            if len(sequence) < 15:  # Need minimum sequence length
                continue
            
            # Extract features from this sequence
            sequence_features = self._extract_sequence_features(sequence, game_id, team)
            training_examples.extend(sequence_features)
        
        df = pd.DataFrame(training_examples)
        logger.info(f"Extracted {len(df)} training examples with advanced features")
        
        return df
    
    def _extract_sequence_features(
        self, 
        sequence: List[GameEvent], 
        game_id: str, 
        team: str
    ) -> List[Dict[str, Any]]:
        """Extract features from a sequence of events for one team."""
        features_list = []
        window_size = 12  # Larger window for better context
        
        for i in range(window_size, len(sequence) - 2):  # Leave room for future events
            current_window = sequence[i-window_size:i]
            next_events = sequence[i:i+2]  # Look at next 2 events for momentum
            
            # Calculate comprehensive features
            features = self._calculate_advanced_window_features(
                current_window, next_events, sequence, i
            )
            
            # Add metadata
            features['game_id'] = game_id
            features['team'] = team
            features['sequence_position'] = i / len(sequence)  # Position in game
            
            features_list.append(features)
        
        return features_list
    
    def _calculate_advanced_window_features(
        self,
        window: List[GameEvent],
        next_events: List[GameEvent],
        full_sequence: List[GameEvent],
        position: int
    ) -> Dict[str, Any]:
        """Calculate advanced features for a window of events."""
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
        features['points_per_possession'] = (features['made_shots'] * 2) / total_possessions  # Simplified
        features['turnover_rate'] = features['turnovers'] / total_possessions
        
        # Defensive metrics
        features['steal_rate'] = features['steals'] / len(window)
        features['block_rate'] = features['blocks'] / len(window)
        features['defensive_events'] = features['steals'] + features['blocks']
        
        # === MOMENTUM-SPECIFIC FEATURES ===
        
        # Scoring runs
        features['scoring_run'] = self._calculate_scoring_run(window)
        features['defensive_run'] = self._calculate_defensive_run(window)
        
        # Event clustering (consecutive similar events)
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
        
        # Recent performance (last 1/3 of window vs first 1/3)
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
        
        # === TARGET VARIABLE ===
        
        # Advanced momentum continuation logic
        features['momentum_continued'] = self._determine_momentum_continuation(
            window, next_events, features
        )
        
        return features
    
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
        positive_events = {'shot': 1, 'steal': 1, 'block': 1, 'assist': 1}  # if shot is made
        negative_events = {'turnover': -1, 'foul': -1}  # if shot is missed
        
        momentum_sequence = []
        for event in window:
            if event.event_type == 'shot':
                if getattr(event, 'shot_result', None) == 'Made':
                    momentum_sequence.append(1)
                else:
                    momentum_sequence.append(-1)
            elif event.event_type in positive_events:
                momentum_sequence.append(1)
            elif event.event_type in negative_events:
                momentum_sequence.append(-1)
        
        # Count direction changes
        swings = 0
        for i in range(1, len(momentum_sequence)):
            if momentum_sequence[i] != momentum_sequence[i-1]:
                swings += 1
        
        return swings
    
    def _determine_momentum_continuation(
        self,
        window: List[GameEvent],
        next_events: List[GameEvent],
        features: Dict[str, Any]
    ) -> int:
        """Determine if momentum continues using sophisticated logic."""
        
        # Current momentum state
        current_momentum = features['momentum_score']
        current_shooting = features['fg_percentage']
        current_turnovers = features['turnovers']
        
        # Analyze next events
        next_positive = 0
        next_negative = 0
        
        for event in next_events:
            if event.event_type == 'shot' and getattr(event, 'shot_result', None) == 'Made':
                next_positive += 2
            elif event.event_type in ['steal', 'block']:
                next_positive += 2
            elif event.event_type == 'assist':
                next_positive += 1
            elif event.event_type == 'turnover':
                next_negative += 2
            elif event.event_type == 'shot' and getattr(event, 'shot_result', None) == 'Missed':
                next_negative += 1
        
        next_momentum = next_positive - next_negative
        
        # Momentum continues if:
        # 1. Current momentum is positive and next events maintain it
        # 2. Current momentum is negative but next events are strongly positive
        # 3. Shooting percentage is good and continues
        
        if current_momentum > 0:
            # Positive momentum continues if next events are not strongly negative
            return int(next_momentum >= -1)
        elif current_momentum < -2:
            # Strong negative momentum needs strong positive next events to continue as positive
            return int(next_momentum > 2)
        else:
            # Neutral momentum - depends on next events
            return int(next_momentum > 0)
    
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train all models with proper evaluation."""
        logger.info("Training advanced models")
        
        # Prepare data
        feature_columns = [col for col in training_data.columns 
                          if col not in ['momentum_continued', 'game_id', 'team', 'sequence_position']]
        
        X = training_data[feature_columns].values
        y = training_data['momentum_continued'].values
        
        # Game-based split to prevent data leakage
        unique_games = training_data['game_id'].unique()
        train_games, test_games = train_test_split(unique_games, test_size=0.2, random_state=42)
        
        train_mask = training_data['game_id'].isin(train_games)
        test_mask = training_data['game_id'].isin(test_games)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        logger.info(f"Training set: {len(X_train)} examples from {len(train_games)} games")
        logger.info(f"Test set: {len(X_test)} examples from {len(test_games)} games")
        logger.info(f"Features: {len(feature_columns)}")
        
        # Check class distribution
        train_dist = pd.Series(y_train).value_counts()
        test_dist = pd.Series(y_test).value_counts()
        logger.info(f"Train class distribution: {dict(train_dist)}")
        logger.info(f"Test class distribution: {dict(test_dist)}")
        
        results = {}
        
        # Train each model
        for name, config in self.models.items():
            logger.info(f"Training {name}: {config['description']}")
            
            try:
                model = config['model']
                scaler = config['scaler']
                
                # Scale features if needed
                if scaler is not None:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                test_pred = model.predict(X_test_scaled)
                test_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'accuracy': accuracy_score(y_test, test_pred),
                    'precision': precision_score(y_test, test_pred, zero_division=0),
                    'recall': recall_score(y_test, test_pred, zero_division=0),
                    'f1': f1_score(y_test, test_pred, zero_division=0),
                    'auc': roc_auc_score(y_test, test_proba),
                    'predictions': test_pred,
                    'probabilities': test_proba
                }
                
                logger.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, "
                           f"AUC: {results[name]['auc']:.4f}, F1: {results[name]['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Baseline comparison
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        baseline_acc = accuracy_score(y_test, dummy.predict(X_test))
        
        dummy_stratified = DummyClassifier(strategy='stratified', random_state=42)
        dummy_stratified.fit(X_train, y_train)
        stratified_acc = accuracy_score(y_test, dummy_stratified.predict(X_test))
        
        logger.info(f"Baseline (most frequent): {baseline_acc:.4f}")
        logger.info(f"Baseline (stratified): {stratified_acc:.4f}")
        
        # Find best model by F1 score (better for imbalanced data)
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
            self.best_model_name = best_model_name
            self.best_model = results[best_model_name]
            
            logger.info(f"Best model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")
        
        # Store additional info
        results['baseline_accuracy'] = baseline_acc
        results['stratified_baseline'] = stratified_acc
        results['feature_names'] = feature_columns
        results['test_data'] = (X_test, y_test)
        
        self.results = results
        return results
    
    def save_model_and_report(self) -> Tuple[str, str]:
        """Save the best model and evaluation report."""
        if not self.best_model:
            raise ValueError("No best model found")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_data = {
            'model': self.best_model['model'],
            'scaler': self.best_model['scaler'],
            'model_name': self.best_model_name,
            'feature_names': self.results['feature_names'],
            'metrics': {
                'accuracy': self.best_model['accuracy'],
                'precision': self.best_model['precision'],
                'recall': self.best_model['recall'],
                'f1': self.best_model['f1'],
                'auc': self.best_model['auc']
            },
            'trained_at': datetime.now().isoformat(),
            'model_type': 'advanced_nba_momentum_predictor'
        }
        
        model_path = self.output_dir / f"advanced_nba_momentum_{self.best_model_name}_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save evaluation report
        report = {
            'best_model': self.best_model_name,
            'model_performance': {
                name: {
                    'accuracy': res['accuracy'],
                    'precision': res['precision'],
                    'recall': res['recall'],
                    'f1': res['f1'],
                    'auc': res['auc']
                } for name, res in self.results.items() 
                if isinstance(res, dict) and 'accuracy' in res
            },
            'baseline_comparison': {
                'most_frequent': self.results['baseline_accuracy'],
                'stratified': self.results['stratified_baseline'],
                'improvement_over_baseline': self.best_model['accuracy'] - self.results['baseline_accuracy'],
                'f1_improvement': self.best_model['f1']
            },
            'feature_names': self.results['feature_names'],
            'training_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_examples': len(self.results['test_data'][0]) + len(self.results['test_data'][1]),
                'test_examples': len(self.results['test_data'][0]),
                'num_features': len(self.results['feature_names'])
            }
        }
        
        report_path = self.output_dir / f"advanced_evaluation_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Report saved to: {report_path}")
        
        return str(model_path), str(report_path)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train advanced NBA momentum prediction model")
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the NBA dataset pickle file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/advanced",
        help="Output directory for models and reports"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Advanced NBA Momentum Model Training")
    logger.info("=" * 60)
    
    try:
        # Initialize trainer
        trainer = AdvancedNBAMomentumTrainer(output_dir=args.output_dir)
        
        # Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        training_data = trainer.load_and_preprocess_data(args.dataset_path)
        
        if training_data.empty:
            logger.error("No training data extracted")
            return 1
        
        # Check class distribution
        class_dist = training_data['momentum_continued'].value_counts()
        logger.info(f"Class distribution: {dict(class_dist)}")
        
        # Train models
        logger.info("Step 2: Training advanced models...")
        results = trainer.train_models(training_data)
        
        if not results or not trainer.best_model:
            logger.error("No models were successfully trained")
            return 1
        
        # Save results
        logger.info("Step 3: Saving model and report...")
        model_path, report_path = trainer.save_model_and_report()
        
        # Print final results
        logger.info("=" * 60)
        logger.info("ADVANCED TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        best_metrics = trainer.best_model
        
        logger.info(f"Best Model: {trainer.best_model_name}")
        logger.info(f"Test Accuracy:  {best_metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {best_metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {best_metrics['recall']:.4f}")
        logger.info(f"Test F1-Score:  {best_metrics['f1']:.4f}")
        logger.info(f"Test AUC:       {best_metrics['auc']:.4f}")
        
        baseline_acc = results['baseline_accuracy']
        improvement = best_metrics['accuracy'] - baseline_acc
        
        logger.info(f"Baseline:       {baseline_acc:.4f}")
        logger.info(f"Improvement:    {improvement:.4f} ({improvement*100:.1f}%)")
        
        logger.info(f"Model saved:    {model_path}")
        logger.info(f"Report saved:   {report_path}")
        
        # Performance assessment
        if best_metrics['f1'] > 0.6:
            logger.info("✅ EXCELLENT: Strong predictive performance!")
        elif best_metrics['f1'] > 0.4:
            logger.info("✅ GOOD: Decent predictive performance")
        elif best_metrics['f1'] > 0.3:
            logger.info("⚠️  MODERATE: Some predictive ability")
        else:
            logger.info("❌ POOR: Limited predictive ability")
        
        if improvement > 0.05:
            logger.info("✅ Meaningful improvement over baseline!")
        elif improvement > 0:
            logger.info("✅ Some improvement over baseline")
        else:
            logger.info("⚠️  No improvement over baseline")
        
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())