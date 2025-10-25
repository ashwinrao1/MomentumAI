#!/usr/bin/env python3
"""
Improved V2 NBA Momentum Model

Key improvements:
1. Better target definition (momentum over next 5 events)
2. Class imbalance handling (SMOTE + class weights)
3. More sophisticated features
4. Better evaluation metrics
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

from tqdm import tqdm

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

# Handle class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("⚠️  Install imbalanced-learn for SMOTE: pip install imbalanced-learn")

# Advanced ML
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


class ImprovedMomentumPredictor:
    """
    Improved momentum prediction with better target definition and class balance handling.
    """
    
    def __init__(self, data_path: str = "data/nba_extracted_dataset.csv"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        
    def load_and_sample_data(self, sample_games: int = 1500) -> pd.DataFrame:
        """Load NBA data with larger sample for better training."""
        print("📊 Loading NBA dataset...")
        
        df = pd.read_csv(self.data_path)
        print(f"  ✅ Loaded {len(df)} events from {df['game_id'].nunique()} games")
        
        # Use more games for better training
        if sample_games < df['game_id'].nunique():
            print(f"  🎯 Sampling {sample_games} games...")
            unique_games = df['game_id'].unique()
            sampled_games = np.random.choice(unique_games, size=sample_games, replace=False)
            df = df[df['game_id'].isin(sampled_games)]
            print(f"  📊 Using {len(df)} events from {df['game_id'].nunique()} games")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create more sophisticated features."""
        print("🔧 Creating advanced momentum features...")
        
        # Sort data properly
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
        
        # Enhanced event values
        print("  💯 Enhanced event values...")
        df['event_value'] = 0
        df.loc[df['is_made_shot'] == 1, 'event_value'] = df.loc[df['is_made_shot'] == 1, 'points_total']
        df.loc[df['is_missed_shot'] == 1, 'event_value'] = -1
        df.loc[df['is_steal'] == 1, 'event_value'] = 2
        df.loc[df['is_block'] == 1, 'event_value'] = 1.5
        df.loc[df['is_assist'] == 1, 'event_value'] = 1
        df.loc[df['is_turnover'] == 1, 'event_value'] = -2
        df.loc[df['is_foul'] == 1, 'event_value'] = -0.5
        
        # Game context features
        print("  ⏰ Game context...")
        df['early_game'] = (df['period'] <= 2).astype(int)
        df['late_game'] = (df['period'] >= 4).astype(int)
        df['clutch_time'] = (df['period'] >= 4).astype(int)
        
        # Team performance features
        print("  🏀 Team performance...")
        grouped = df.groupby(['game_id', 'team_tricode'])
        
        # Rolling statistics (multiple windows)
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
        print("  🎯 Situational features...")
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
        
        print(f"  ✅ Created {len(df.columns)} features")
        return df
    
    def create_better_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a better target variable that's more balanced."""
        print("🎯 Creating improved target variable...")
        
        df = df.sort_values(['game_id', 'team_tricode', 'timestamp'])
        grouped = df.groupby(['game_id', 'team_tricode'])
        
        # Option 1: Momentum in next 5 events (more balanced)
        df['next_5_momentum'] = grouped['event_value'].rolling(5, min_periods=1).sum().shift(-5).values
        df['momentum_continues_5'] = (df['next_5_momentum'] > 0).astype(int)
        
        # Option 2: Strong momentum in next 3 events
        df['next_3_momentum'] = grouped['event_value'].rolling(3, min_periods=1).sum().shift(-3).values
        df['strong_momentum_3'] = (df['next_3_momentum'] >= 2).astype(int)
        
        # Option 3: Any positive event in next 3
        df['next_positive'] = grouped['positive_event'].rolling(3, min_periods=1).max().shift(-3).values
        
        # Check class balance for each target
        targets = ['momentum_continues_5', 'strong_momentum_3', 'next_positive']
        
        for target in targets:
            if target in df.columns:
                balance = df[target].mean()
                print(f"  📊 {target}: {balance:.3f} positive class")
        
        # Choose the most balanced target
        if 'momentum_continues_5' in df.columns:
            return df, 'momentum_continues_5'
        else:
            return df, 'next_positive'
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data with the chosen target."""
        print(f"📋 Preparing training data with target: {target_col}")
        
        # Remove rows without target
        df = df.dropna(subset=[target_col])
        
        # Select features
        feature_cols = [
            # Basic events
            'is_shot', 'is_made_shot', 'is_missed_shot', 'is_rebound', 'is_turnover',
            'is_steal', 'is_block', 'is_assist', 'is_foul', 'event_value',
            
            # Game context
            'early_game', 'late_game', 'clutch_time', 'period',
            
            # Rolling features
            'shots_last_5', 'made_shots_last_5', 'momentum_last_5',
            'shots_last_10', 'made_shots_last_10', 'momentum_last_10',
            'shots_last_15', 'made_shots_last_15', 'momentum_last_15',
            'turnovers_last_5', 'turnovers_last_10', 'turnovers_last_15',
            
            # Efficiency
            'fg_pct_last_5', 'fg_pct_last_10', 'fg_pct_last_15',
            
            # Momentum trends
            'momentum_change', 'momentum_acceleration',
            
            # Situational
            'high_value_event', 'positive_event', 'negative_event',
            
            # Interactions
            'clutch_momentum', 'late_game_momentum', 'early_game_momentum',
            
            # Streaks
            'positive_streak_length'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].fillna(0)
        y = df[target_col]
        
        print(f"  📊 Training data: {len(X)} samples, {len(available_features)} features")
        print(f"  ⚖️  Class balance: {y.mean():.3f} positive class")
        
        return X, y
    
    def train_improved_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train models with class imbalance handling."""
        print("🤖 Training improved models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        print("  📏 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        print("  🎯 Selecting features...")
        selector = SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"    ✅ Selected {len(selected_features)} features")
        
        # Handle class imbalance
        print("  ⚖️  Handling class imbalance...")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"    📊 Class weights: {class_weight_dict}")
        
        # Apply SMOTE if available
        if HAS_IMBALANCED and len(np.unique(y_train)) == 2:
            print("    🔄 Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
            print(f"    ✅ Balanced data: {len(X_train_balanced)} samples")
        else:
            X_train_balanced, y_train_balanced = X_train_selected, y_train
        
        # Define models with class weights
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                class_weight=class_weight_dict, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=8,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, class_weight=class_weight_dict, random_state=42, max_iter=1000
            )
        }
        
        # Add XGBoost with class weights
        if HAS_XGBOOST:
            scale_pos_weight = class_weights[0] / class_weights[1]
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=8,
                scale_pos_weight=scale_pos_weight, random_state=42
            )
        
        # Add LightGBM with class weights
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=8,
                class_weight=class_weight_dict, random_state=42, verbose=-1
            )
        
        # Train models
        trained_models = {}
        model_scores = {}
        
        for name, model in tqdm(models.items(), desc="Training models"):
            try:
                # Use balanced data for tree-based models, original for others
                if name in ['gradient_boosting'] or not HAS_IMBALANCED:
                    model.fit(X_train_selected, y_train)
                else:
                    model.fit(X_train_balanced, y_train_balanced)
                
                # Evaluate on original test set
                y_pred = model.predict(X_test_selected)
                y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
                
                # Comprehensive metrics
                score = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'auc': roc_auc_score(y_test, y_pred_proba),
                    'avg_precision': average_precision_score(y_test, y_pred_proba)
                }
                
                trained_models[name] = model
                model_scores[name] = score
                
                print(f"    ✅ {name}: F1={score['f1']:.4f}, AUC={score['auc']:.4f}, Precision={score['precision']:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error training {name}: {e}")
                continue
        
        # Find best model by F1 score
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
            'feature_selector': selector,
            'class_weights': class_weight_dict
        }
    
    def save_model(self, results: Dict, target_col: str) -> str:
        """Save the improved model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = "models/improved_v2"
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = f"{model_dir}/improved_v2_momentum_model_{timestamp}.pkl"
        
        model_package = {
            'models': results['models'],
            'best_model': results['best_model'],
            'best_model_name': results['best_model_name'],
            'scaler': results['scaler'],
            'feature_selector': results['feature_selector'],
            'selected_features': results['selected_features'],
            'class_weights': results['class_weights'],
            'target_column': target_col,
            'training_timestamp': timestamp,
            'model_type': 'improved_v2_ensemble'
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save evaluation report
        report_file = f"{model_dir}/evaluation_report_{timestamp}.json"
        
        evaluation_report = {
            'model_info': {
                'model_type': 'improved_v2_ensemble',
                'training_timestamp': timestamp,
                'target_column': target_col,
                'num_features': len(results['selected_features']),
                'models_trained': list(results['models'].keys()),
                'class_weights': results['class_weights']
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
    """Train improved V2 momentum model."""
    
    print("🚀 IMPROVED V2 MOMENTUM MODEL TRAINING")
    print("="*60)
    print("Key improvements:")
    print("• Better target definition (momentum over 5 events)")
    print("• Class imbalance handling (SMOTE + class weights)")
    print("• More sophisticated features")
    print("• Larger training dataset (1500 games)")
    print("="*60)
    print()
    
    predictor = ImprovedMomentumPredictor()
    
    try:
        # Load more data
        df = predictor.load_and_sample_data(sample_games=1500)
        
        # Create advanced features
        df = predictor.create_advanced_features(df)
        
        # Create better target
        df, target_col = predictor.create_better_target(df)
        
        # Prepare training data
        X, y = predictor.prepare_training_data(df, target_col)
        
        # Train improved models
        results = predictor.train_improved_models(X, y)
        
        # Save model
        model_file = predictor.save_model(results, target_col)
        
        print()
        print("="*60)
        print("🎉 IMPROVED V2 TRAINING COMPLETE!")
        print("="*60)
        
        # Print results
        print(f"📈 Model Performance:")
        for name, score in results['model_scores'].items():
            print(f"   • {name}:")
            print(f"     - F1: {score['f1']:.4f}")
            print(f"     - AUC: {score['auc']:.4f}")
            print(f"     - Precision: {score['precision']:.4f}")
            print(f"     - Recall: {score['recall']:.4f}")
        
        if results['best_model_name']:
            print(f"\n🏆 Best Model: {results['best_model_name']}")
            best_score = results['model_scores'][results['best_model_name']]
            print(f"   • F1-Score: {best_score['f1']:.4f}")
            print(f"   • AUC: {best_score['auc']:.4f}")
            print(f"   • Precision: {best_score['precision']:.4f}")
            print(f"   • Recall: {best_score['recall']:.4f}")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Training samples: {len(X):,}")
        print(f"   • Features: {len(results['selected_features'])}")
        print(f"   • Games: {df['game_id'].nunique()}")
        print(f"   • Target: {target_col}")
        print(f"   • Class balance: {y.mean():.3f}")
        
        print(f"\n💾 Model saved: {model_file}")
        
    except Exception as e:
        print(f"❌ Error training improved model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()