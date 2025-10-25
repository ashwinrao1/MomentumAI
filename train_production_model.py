#!/usr/bin/env python3
"""
Production Model Training Script - Simplified Version

Trains advanced ML models on real NBA data with comprehensive evaluation.
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

# Core ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier

# Advanced ML libraries
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
    from services.enhanced_momentum_engine import EnhancedMomentumEngine
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


def load_nba_dataset(dataset_path: str) -> List[GameEvent]:
    """Load the NBA dataset from pickle file."""
    logger.info(f"Loading NBA dataset from {dataset_path}")
    
    try:
        with open(dataset_path, 'rb') as f:
            events = pickle.load(f)
        
        logger.info(f"Loaded {len(events):,} events")
        
        # Basic validation
        if not events:
            raise ValueError("Dataset is empty")
        
        # Check data quality
        games = set(event.game_id for event in events)
        teams = set(event.team_tricode for event in events)
        
        logger.info(f"Dataset contains {len(games)} games and {len(teams)} teams")
        
        return events
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def extract_simplified_features(events: List[GameEvent]) -> pd.DataFrame:
    """Extract simplified features from NBA events for faster processing."""
    logger.info("Extracting simplified features from NBA events")
    
    # Group events by game and team
    game_team_data = {}
    for event in events:
        key = (event.game_id, event.team_tricode)
        if key not in game_team_data:
            game_team_data[key] = []
        game_team_data[key].append(event)
    
    logger.info(f"Processing {len(game_team_data)} game-team combinations")
    
    training_data = []
    
    for (game_id, team), team_events in game_team_data.items():
        if len(team_events) < 20:  # Need minimum events
            continue
        
        # Sort events by some order (using event_id as proxy)
        team_events.sort(key=lambda x: x.event_id)
        
        # Create sliding windows
        window_size = 10
        for i in range(window_size, len(team_events) - 1):
            window_events = team_events[i-window_size:i]
            next_event = team_events[i]
            
            # Calculate features for this window
            features = calculate_window_features(window_events, next_event)
            features['game_id'] = game_id
            features['team'] = team
            
            training_data.append(features)
    
    df = pd.DataFrame(training_data)
    logger.info(f"Extracted {len(df)} training examples with {df.shape[1]} features")
    
    return df


def calculate_window_features(window_events: List[GameEvent], next_event: GameEvent) -> Dict[str, Any]:
    """Calculate features for a window of events."""
    features = {}
    
    # Basic event counts
    event_types = {}
    for event in window_events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    # Feature extraction
    features['shot_count'] = event_types.get('shot', 0)
    features['rebound_count'] = event_types.get('rebound', 0)
    features['turnover_count'] = event_types.get('turnover', 0)
    features['foul_count'] = event_types.get('foul', 0)
    features['steal_count'] = event_types.get('steal', 0)
    features['block_count'] = event_types.get('block', 0)
    features['assist_count'] = event_types.get('assist', 0)
    features['other_count'] = event_types.get('other', 0)
    
    # Calculate ratios
    total_events = len(window_events)
    features['shot_rate'] = features['shot_count'] / total_events
    features['turnover_rate'] = features['turnover_count'] / total_events
    features['steal_rate'] = features['steal_count'] / total_events
    features['block_rate'] = features['block_count'] / total_events
    
    # Momentum indicators
    positive_events = features['shot_count'] + features['steal_count'] + features['block_count'] + features['assist_count']
    negative_events = features['turnover_count'] + features['foul_count']
    features['momentum_score'] = positive_events - negative_events
    features['momentum_ratio'] = positive_events / max(negative_events, 1)
    
    # Time-based features (if available)
    if hasattr(window_events[0], 'period'):
        features['avg_period'] = np.mean([event.period for event in window_events])
    else:
        features['avg_period'] = 2.0  # Default
    
    # Event value features (if available)
    if hasattr(window_events[0], 'event_value'):
        event_values = [getattr(event, 'event_value', 0) for event in window_events]
        features['avg_event_value'] = np.mean(event_values)
        features['total_event_value'] = np.sum(event_values)
    else:
        features['avg_event_value'] = 0.0
        features['total_event_value'] = 0.0
    
    # Determine momentum continuation (simplified)
    # Momentum continues if next event is positive
    next_event_positive = next_event.event_type in ['shot', 'steal', 'block', 'assist', 'rebound']
    features['momentum_continued'] = int(next_event_positive)
    
    return features


def train_models(training_data: pd.DataFrame) -> Dict[str, Any]:
    """Train multiple models and return results."""
    logger.info("Training multiple models")
    
    # Prepare data
    feature_columns = [col for col in training_data.columns 
                      if col not in ['momentum_continued', 'game_id', 'team']]
    
    X = training_data[feature_columns].values
    y = training_data['momentum_continued'].values
    
    # Game-based split
    unique_games = training_data['game_id'].unique()
    train_games, test_games = train_test_split(unique_games, test_size=0.2, random_state=42)
    
    train_mask = training_data['game_id'].isin(train_games)
    test_mask = training_data['game_id'].isin(test_games)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Train: {len(X_train)} examples, Test: {len(X_test)} examples")
    logger.info(f"Features: {len(feature_columns)}")
    
    # Model configurations
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['xgboost'] = xgb.XGBClassifier(
            random_state=42, n_estimators=100, max_depth=6,
            learning_rate=0.1, eval_metric='logloss'
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = lgb.LGBMClassifier(
            random_state=42, n_estimators=100, max_depth=6,
            learning_rate=0.1, verbose=-1
        )
    
    results = {}
    
    # Train each model
    for name, model in models.items():
        logger.info(f"Training {name}")
        
        try:
            # Scale data for linear models
            if name in ['logistic_regression']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                scaler = None
            
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
                'precision': precision_score(y_test, test_pred),
                'recall': recall_score(y_test, test_pred),
                'f1': f1_score(y_test, test_pred),
                'auc': roc_auc_score(y_test, test_proba),
                'predictions': test_pred,
                'probabilities': test_proba
            }
            
            logger.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, AUC: {results[name]['auc']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            continue
    
    # Baseline comparison
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, dummy.predict(X_test))
    
    logger.info(f"Baseline (most frequent): {baseline_acc:.4f}")
    
    # Find best model
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        logger.info(f"Best model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
        
        results['best_model'] = best_model_name
        results['baseline_accuracy'] = baseline_acc
        results['feature_names'] = feature_columns
        results['test_data'] = (X_test, y_test)
    
    return results


def save_results(results: Dict[str, Any], output_dir: str):
    """Save model results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best model
    if 'best_model' in results:
        best_name = results['best_model']
        best_result = results[best_name]
        
        model_data = {
            'model': best_result['model'],
            'scaler': best_result['scaler'],
            'model_name': best_name,
            'feature_names': results['feature_names'],
            'metrics': {
                'accuracy': best_result['accuracy'],
                'precision': best_result['precision'],
                'recall': best_result['recall'],
                'f1': best_result['f1'],
                'auc': best_result['auc']
            },
            'trained_at': datetime.now().isoformat()
        }
        
        model_path = output_path / f"nba_momentum_model_{best_name}_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Best model saved to: {model_path}")
        
        # Save evaluation report
        report = {
            'best_model': best_name,
            'model_performance': {name: {
                'accuracy': res['accuracy'],
                'precision': res['precision'],
                'recall': res['recall'],
                'f1': res['f1'],
                'auc': res['auc']
            } for name, res in results.items() if isinstance(res, dict) and 'accuracy' in res},
            'baseline_accuracy': results['baseline_accuracy'],
            'improvement_over_baseline': best_result['accuracy'] - results['baseline_accuracy'],
            'feature_names': results['feature_names'],
            'training_timestamp': datetime.now().isoformat()
        }
        
        report_path = output_path / f"evaluation_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        
        return str(model_path), str(report_path)
    
    return None, None


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train NBA momentum prediction model")
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the NBA dataset pickle file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/production",
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
    
    logger.info("Starting NBA Momentum Model Training")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load dataset
        logger.info("Step 1: Loading NBA dataset...")
        events = load_nba_dataset(args.dataset_path)
        
        # Extract features
        logger.info("Step 2: Extracting features...")
        training_data = extract_simplified_features(events)
        
        if training_data.empty:
            logger.error("No training data extracted")
            return 1
        
        # Check class distribution
        class_dist = training_data['momentum_continued'].value_counts()
        logger.info(f"Class distribution: {dict(class_dist)}")
        
        # Train models
        logger.info("Step 3: Training models...")
        results = train_models(training_data)
        
        if not results or 'best_model' not in results:
            logger.error("No models were successfully trained")
            return 1
        
        # Save results
        logger.info("Step 4: Saving results...")
        model_path, report_path = save_results(results, args.output_dir)
        
        # Print final results
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED!")
        logger.info("=" * 50)
        
        best_name = results['best_model']
        best_metrics = results[best_name]
        
        logger.info(f"Best Model: {best_name}")
        logger.info(f"Test Accuracy:  {best_metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {best_metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {best_metrics['recall']:.4f}")
        logger.info(f"Test F1-Score:  {best_metrics['f1']:.4f}")
        logger.info(f"Test AUC:       {best_metrics['auc']:.4f}")
        
        improvement = best_metrics['accuracy'] - results['baseline_accuracy']
        logger.info(f"Baseline:       {results['baseline_accuracy']:.4f}")
        logger.info(f"Improvement:    {improvement:.4f} ({improvement*100:.1f}%)")
        
        if model_path:
            logger.info(f"Model saved:    {model_path}")
        if report_path:
            logger.info(f"Report saved:   {report_path}")
        
        # Performance assessment
        if best_metrics['accuracy'] > 0.7:
            logger.info("✅ EXCELLENT: Strong model performance!")
        elif best_metrics['accuracy'] > 0.6:
            logger.info("✅ GOOD: Decent model performance")
        elif best_metrics['accuracy'] > 0.55:
            logger.info("⚠️  MODERATE: Some predictive ability")
        else:
            logger.info("❌ POOR: Limited predictive ability")
        
        if improvement > 0.1:
            logger.info("✅ Significant improvement over baseline!")
        elif improvement > 0.05:
            logger.info("✅ Moderate improvement over baseline")
        else:
            logger.info("⚠️  Limited improvement over baseline")
        
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())