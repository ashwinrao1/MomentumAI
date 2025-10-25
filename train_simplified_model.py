#!/usr/bin/env python3
"""
Simplified Next-Generation NBA Momentum Model

Focuses on core functionality with 2.3M real NBA events.
No synthetic data, optimized for large datasets.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Progress bars
from tqdm import tqdm

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.feature_selection import SelectKBest, f_classif

# Advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class SimplifiedMomentumPredictor:
    """
    Simplified momentum prediction system optimized for large NBA datasets.
    """
    
    def __init__(self, data_path: str = "data/nba_extracted_dataset.csv"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
    def load_and_sample_data(self, sample_games: int = 1000) -> pd.DataFrame:
        """Load NBA data and sample for manageable training."""
        print("📊 Loading NBA dataset...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"  ✅ Loaded {len(df)} events from {df['game_id'].nunique()} games")
        
        # Sample games for manageable training
        if sample_games < df['game_id'].nunique():
            print(f"  🎯 Sampling {sample_games} games...")
            unique_games = df['game_id'].unique()
            sampled_games = np.random.choice(unique_games, size=sample_games, replace=False)
            df = df[df['game_id'].isin(sampled_games)]
            print(f"  📊 Using {len(df)} events from {df['game_id'].nunique()} games")
        
        return df
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic momentum features without complex rolling calculations."""
        print("🔧 Creating basic momentum features...")
        
        # Sort data
        df = df.sort_values(['game_id', 'team_tricode', 'timestamp'])
        
        # Basic event features
        print("  📊 Basic event encoding...")
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
        print("  💯 Event values...")
        df['event_value'] = 0
        df.loc[df['is_made_shot'] == 1, 'event_value'] = df.loc[df['is_made_shot'] == 1, 'points_total']
        df.loc[df['is_missed_shot'] == 1, 'event_value'] = -1
        df.loc[df['is_steal'] == 1, 'event_value'] = 2
        df.loc[df['is_block'] == 1, 'event_value'] = 1.5
        df.loc[df['is_assist'] == 1, 'event_value'] = 1
        df.loc[df['is_turnover'] == 1, 'event_value'] = -2
        df.loc[df['is_foul'] == 1, 'event_value'] = -0.5
        
        # Game context
        print("  ⏰ Game context...")
        df['early_game'] = (df['period'] <= 2).astype(int)
        df['late_game'] = (df['period'] >= 4).astype(int)
        df['clutch_time'] = (df['period'] >= 4).astype(int)
        
        # Simple aggregations (last 10 events per team per game)
        print("  📈 Simple rolling features...")
        
        # Use simpler rolling calculations
        grouped = df.groupby(['game_id', 'team_tricode'])
        
        df['shots_last_10'] = grouped['is_shot'].rolling(10, min_periods=1).sum().values
        df['made_shots_last_10'] = grouped['is_made_shot'].rolling(10, min_periods=1).sum().values
        df['momentum_last_10'] = grouped['event_value'].rolling(10, min_periods=1).sum().values
        df['avg_momentum_last_10'] = grouped['event_value'].rolling(10, min_periods=1).mean().values
        
        # Derived features
        df['fg_pct_last_10'] = df['made_shots_last_10'] / (df['shots_last_10'] + 0.01)
        df['momentum_change'] = grouped['event_value'].diff().values
        
        # Interaction features
        df['clutch_momentum'] = df['clutch_time'] * df['event_value']
        df['late_game_momentum'] = df['late_game'] * df['event_value']
        
        print(f"  ✅ Created {len(df.columns)} features")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        print("🎯 Preparing training data...")
        
        # Create target: momentum continues (next event has positive value)
        df = df.sort_values(['game_id', 'team_tricode', 'timestamp'])
        df['next_event_value'] = df.groupby(['game_id', 'team_tricode'])['event_value'].shift(-1)
        df['momentum_continues'] = (df['next_event_value'] > 0).astype(int)
        
        # Remove rows without target
        df = df.dropna(subset=['momentum_continues'])
        
        # Select features
        feature_cols = [
            'is_shot', 'is_made_shot', 'is_missed_shot', 'is_rebound', 'is_turnover',
            'is_steal', 'is_block', 'is_assist', 'is_foul', 'event_value',
            'early_game', 'late_game', 'clutch_time', 'period',
            'shots_last_10', 'made_shots_last_10', 'momentum_last_10', 'avg_momentum_last_10',
            'fg_pct_last_10', 'momentum_change', 'clutch_momentum', 'late_game_momentum'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].fillna(0)
        y = df['momentum_continues']
        
        print(f"  📊 Training data: {len(X)} samples, {len(available_features)} features")
        print(f"  ⚖️  Class balance: {y.mean():.3f} positive momentum")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple models."""
        print("🤖 Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        print("  📏 Scaling features...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        print("  🎯 Selecting features...")
        selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"    ✅ Selected {len(selected_features)} features")
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, random_state=42, max_iter=1000
            )
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1
            )
        
        # Train models
        trained_models = {}
        model_scores = {}
        
        for name, model in tqdm(models.items(), desc="Training models"):
            try:
                model.fit(X_train_selected, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_selected)
                y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
                
                score = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                trained_models[name] = model
                model_scores[name] = score
                
                print(f"    ✅ {name}: F1={score['f1']:.4f}, AUC={score['auc']:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error training {name}: {e}")
                continue
        
        # Find best model
        if model_scores:
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['f1'])
            best_model = trained_models[best_model_name]
            
            print(f"  🏆 Best model: {best_model_name} (F1: {model_scores[best_model_name]['f1']:.4f})")
        else:
            best_model_name = None
            best_model = None
        
        return {
            'models': trained_models,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'model_scores': model_scores,
            'selected_features': selected_features,
            'scaler': scaler,
            'feature_selector': selector
        }
    
    def save_model(self, results: Dict) -> str:
        """Save the trained model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = "models/simplified"
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = f"{model_dir}/simplified_momentum_model_{timestamp}.pkl"
        
        model_package = {
            'models': results['models'],
            'best_model': results['best_model'],
            'best_model_name': results['best_model_name'],
            'scaler': results['scaler'],
            'feature_selector': results['feature_selector'],
            'selected_features': results['selected_features'],
            'training_timestamp': timestamp,
            'model_type': 'simplified_ensemble'
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save evaluation report
        report_file = f"{model_dir}/evaluation_report_{timestamp}.json"
        
        evaluation_report = {
            'model_info': {
                'model_type': 'simplified_ensemble',
                'training_timestamp': timestamp,
                'num_features': len(results['selected_features']),
                'models_trained': list(results['models'].keys())
            },
            'model_scores': results['model_scores'],
            'best_model': results['best_model_name'],
            'selected_features': results['selected_features']
        }
        
        with open(report_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print(f"💾 Model saved to: {model_file}")
        print(f"📊 Report saved to: {report_file}")
        
        return model_file


def main():
    """Train simplified momentum model with real NBA data."""
    
    print("🚀 SIMPLIFIED MOMENTUM MODEL TRAINING")
    print("="*60)
    print("Using 2.3M real NBA events - NO synthetic data")
    print("="*60)
    print()
    
    predictor = SimplifiedMomentumPredictor()
    
    try:
        # Load and sample data
        df = predictor.load_and_sample_data(sample_games=1000)
        
        # Create features
        df = predictor.create_basic_features(df)
        
        # Prepare training data
        X, y = predictor.prepare_training_data(df)
        
        # Train models
        results = predictor.train_models(X, y)
        
        # Save model
        model_file = predictor.save_model(results)
        
        print()
        print("="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        
        # Print results
        print(f"📈 Model Performance:")
        for name, score in results['model_scores'].items():
            print(f"   • {name}: F1={score['f1']:.4f}, AUC={score['auc']:.4f}")
        
        if results['best_model_name']:
            print(f"\n🏆 Best Model: {results['best_model_name']}")
            best_score = results['model_scores'][results['best_model_name']]
            print(f"   • F1-Score: {best_score['f1']:.4f}")
            print(f"   • AUC: {best_score['auc']:.4f}")
            print(f"   • Accuracy: {best_score['accuracy']:.4f}")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Training samples: {len(X):,}")
        print(f"   • Features: {len(results['selected_features'])}")
        print(f"   • Games: {df['game_id'].nunique()}")
        
        print(f"\n💾 Model saved: {model_file}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()