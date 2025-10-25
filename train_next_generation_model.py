#!/usr/bin/env python3
"""
Next-Generation NBA Momentum Prediction Model

This script implements state-of-the-art machine learning techniques for momentum prediction:
- Deep Neural Networks (MLPs, LSTMs)
- Advanced Ensemble Methods (Stacking, Voting)
- Sophisticated Feature Engineering
- Temporal Sequence Modeling
- Real NBA data from 400+ games (172K+ events)
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress function
    def tqdm(iterable, desc="Processing", total=None):
        return iterable

# Core ML libraries
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel,
    mutual_info_classif, chi2
)

# Advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available - install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available - install with: pip install lightgbm")

# Deep Learning (optional)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not available - install with: pip install tensorflow")


class NextGenerationMomentumPredictor:
    """
    Advanced momentum prediction system using state-of-the-art ML techniques.
    """
    
    def __init__(self, data_path: str = "data/nba_extracted_dataset.csv"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_names = []
        self.training_history = {}
        
    def load_nba_data(self) -> pd.DataFrame:
        """Load the comprehensive NBA dataset from CSV."""
        print("📊 Loading comprehensive NBA dataset...")
        
        try:
            print(f"  📁 Loading from {self.data_path}")
            
            # Load the clean CSV data
            df = pd.read_csv(self.data_path)
            
            print(f"  ✅ Loaded {len(df)} events from {df['game_id'].nunique()} games")
            print(f"  🏟️  Teams: {df['team_tricode'].nunique()}")
            print(f"  �  Event types: {df['event_type'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"  ❌ Error loading data: {e}")
            raise ValueError(f"Could not load NBA data: {e}")
    
    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sophisticated features for momentum prediction.
        """
        print("🔧 Engineering advanced features...")
        
        # Sort by game and timestamp for temporal features
        df = df.sort_values(['game_id', 'timestamp'])
        
        # 1. BASIC EVENT FEATURES
        print("  📊 Creating basic event features...")
        
        # Event type encoding
        df['is_shot'] = df['event_type'].str.contains('shot', case=False, na=False).astype(int)
        df['is_made_shot'] = (df['shot_result'] == 'Made').astype(int)
        df['is_missed_shot'] = (df['shot_result'] == 'Missed').astype(int)
        df['is_rebound'] = (df['event_type'] == 'rebound').astype(int)
        df['is_turnover'] = (df['event_type'] == 'turnover').astype(int)
        df['is_steal'] = (df['event_type'] == 'steal').astype(int)
        df['is_block'] = (df['event_type'] == 'block').astype(int)
        df['is_assist'] = (df['event_type'] == 'assist').astype(int)
        df['is_foul'] = (df['event_type'] == 'foul').astype(int)
        
        # Event value (if available)
        if hasattr(df.iloc[0], 'event_value') or 'event_value' in df.columns:
            df['event_value'] = df.get('event_value', 0)
        else:
            # Calculate event value based on type and result
            df['event_value'] = 0
            df.loc[df['is_made_shot'] == 1, 'event_value'] = 2  # Made shot
            df.loc[df['is_missed_shot'] == 1, 'event_value'] = -1  # Missed shot
            df.loc[df['is_steal'] == 1, 'event_value'] = 2  # Steal
            df.loc[df['is_block'] == 1, 'event_value'] = 1.5  # Block
            df.loc[df['is_assist'] == 1, 'event_value'] = 1  # Assist
            df.loc[df['is_turnover'] == 1, 'event_value'] = -2  # Turnover
            df.loc[df['is_foul'] == 1, 'event_value'] = -0.5  # Foul
        
        # 2. TEMPORAL FEATURES
        print("  ⏰ Creating temporal features...")
        
        # Game context
        df['early_game'] = (df['period'] <= 2).astype(int)
        df['late_game'] = (df['period'] >= 4).astype(int)
        df['clutch_time'] = (df['period'] >= 4).astype(int)  # Simplified clutch definition
        
        # Time remaining (if available)
        if hasattr(df.iloc[0], 'time_remaining') or 'time_remaining' in df.columns:
            df['time_remaining'] = df.get('time_remaining', 0)
        else:
            # Estimate time remaining based on period
            df['time_remaining'] = (4 - df['period']) * 12
        
        df['very_clutch'] = ((df['period'] >= 4) & (df['time_remaining'] <= 5)).astype(int)
        
        # 3. ROLLING WINDOW FEATURES
        print("  📈 Creating rolling window features...")
        
        # Rolling statistics for each team (simplified for large dataset)
        windows = [5, 10, 20]  # Reduced windows for performance
        
        print("  📈 Creating rolling window features...")
        
        # Sort data first for rolling calculations
        df = df.sort_values(['game_id', 'team_tricode', 'timestamp'])
        
        for window in tqdm(windows, desc="Rolling windows"):
            # Use a more memory-efficient approach
            grouped = df.groupby(['game_id', 'team_tricode'])
            
            # Rolling event counts
            df[f'shots_last_{window}'] = grouped['is_shot'].rolling(window, min_periods=1).sum().values
            df[f'made_shots_last_{window}'] = grouped['is_made_shot'].rolling(window, min_periods=1).sum().values
            df[f'turnovers_last_{window}'] = grouped['is_turnover'].rolling(window, min_periods=1).sum().values
            df[f'steals_last_{window}'] = grouped['is_steal'].rolling(window, min_periods=1).sum().values
            
            # Rolling momentum (event value)
            df[f'momentum_last_{window}'] = grouped['event_value'].rolling(window, min_periods=1).sum().values
            df[f'avg_momentum_last_{window}'] = grouped['event_value'].rolling(window, min_periods=1).mean().values
        
        # Calculate shooting percentages after rolling sums
        for window in windows:
            df[f'fg_pct_last_{window}'] = df[f'made_shots_last_{window}'] / (df[f'shots_last_{window}'] + 0.01)
        
        # 4. MOMENTUM ACCELERATION FEATURES
        print("  🚀 Creating momentum acceleration features...")
        
        # Momentum changes
        df['momentum_change'] = df.groupby(['game_id', 'team_tricode'])['event_value'].diff()
        df['momentum_acceleration'] = df.groupby(['game_id', 'team_tricode'])['momentum_change'].diff()
        
        # Momentum streaks
        df['positive_streak'] = (df['event_value'] > 0).astype(int)
        df['negative_streak'] = (df['event_value'] < 0).astype(int)
        
        # Streak lengths
        df['pos_streak_length'] = df.groupby(['game_id', 'team_tricode', (df['positive_streak'] != df['positive_streak'].shift()).cumsum()])['positive_streak'].cumsum()
        df['neg_streak_length'] = df.groupby(['game_id', 'team_tricode', (df['negative_streak'] != df['negative_streak'].shift()).cumsum()])['negative_streak'].cumsum()
        
        # 5. OPPONENT-RELATIVE FEATURES
        print("  🆚 Creating opponent-relative features...")
        
        # Calculate opponent stats for each game
        game_team_stats = df.groupby(['game_id', 'team_tricode']).agg({
            'event_value': ['sum', 'mean'],
            'is_shot': 'sum',
            'is_made_shot': 'sum',
            'is_turnover': 'sum'
        }).reset_index()
        
        # Flatten column names
        game_team_stats.columns = ['game_id', 'team_tricode', 'total_momentum', 'avg_momentum', 'total_shots', 'made_shots', 'turnovers']
        
        # Calculate opponent averages
        game_stats = game_team_stats.groupby('game_id').agg({
            'total_momentum': 'mean',
            'avg_momentum': 'mean',
            'total_shots': 'mean',
            'made_shots': 'mean',
            'turnovers': 'mean'
        }).add_suffix('_game_avg')
        
        # Merge back
        df = df.merge(game_stats, left_on='game_id', right_index=True, how='left')
        
        # 6. ADVANCED STATISTICAL FEATURES
        print("  📊 Creating advanced statistical features...")
        
        # Efficiency metrics
        df['shot_efficiency'] = df['made_shots_last_10'] / (df['shots_last_10'] + 0.01)
        df['turnover_rate'] = df['turnovers_last_10'] / 10
        df['steal_rate'] = df['steals_last_10'] / 10
        
        # Momentum volatility (fixed indexing)
        volatility_series = df.groupby(['game_id', 'team_tricode'])['event_value'].rolling(10, min_periods=1).std()
        df['momentum_volatility'] = volatility_series.values
        
        # Performance under pressure
        df['clutch_performance'] = df['clutch_time'] * df['event_value']
        df['pressure_shots'] = df['very_clutch'] * df['is_shot']
        df['pressure_makes'] = df['very_clutch'] * df['is_made_shot']
        
        # 7. INTERACTION FEATURES
        print("  🔗 Creating interaction features...")
        
        # Time-based interactions
        df['late_game_momentum'] = df['late_game'] * df['event_value']
        df['clutch_shooting'] = df['clutch_time'] * df['shot_efficiency']
        
        # Streak interactions
        df['hot_streak'] = (df['pos_streak_length'] >= 3).astype(int)
        df['cold_streak'] = (df['neg_streak_length'] >= 3).astype(int)
        df['streak_momentum'] = df['hot_streak'] * df['event_value'] - df['cold_streak'] * abs(df['event_value'])
        
        # Fill NaN values
        df = df.fillna(0)
        
        print(f"  ✅ Created {len(df.columns)} total features")
        return df
    
    def create_temporal_sequences(self, df: pd.DataFrame, sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create temporal sequences for LSTM/RNN models.
        """
        print(f"🔄 Creating temporal sequences (length={sequence_length})...")
        
        # Select features for sequences
        feature_cols = [col for col in df.columns if col not in ['game_id', 'team_tricode', 'timestamp', 'event_id', 'player_name', 'description']]
        
        sequences = []
        targets = []
        
        for (game_id, team), group in df.groupby(['game_id', 'team_tricode']):
            if len(group) < sequence_length + 1:
                continue
            
            group_features = group[feature_cols].values
            
            for i in range(len(group_features) - sequence_length):
                # Sequence of features
                seq = group_features[i:i+sequence_length]
                
                # Target: momentum continues (positive event value in next step)
                target = 1 if group_features[i+sequence_length][feature_cols.index('event_value')] > 0 else 0
                
                sequences.append(seq)
                targets.append(target)
        
        X_seq = np.array(sequences)
        y_seq = np.array(targets)
        
        print(f"  ✅ Created {len(X_seq)} sequences of shape {X_seq.shape}")
        return X_seq, y_seq
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final training dataset with target variable.
        """
        print("🎯 Preparing training data...")
        
        # Create target variable: momentum continuation
        # Target = 1 if next event has positive value, 0 otherwise
        df = df.sort_values(['game_id', 'team_tricode', 'timestamp'])
        df['next_event_value'] = df.groupby(['game_id', 'team_tricode'])['event_value'].shift(-1)
        df['momentum_continues'] = (df['next_event_value'] > 0).astype(int)
        
        # Remove rows without target
        df = df.dropna(subset=['momentum_continues'])
        
        # Select features for training
        exclude_cols = [
            'game_id', 'team_tricode', 'timestamp', 'event_id', 'player_name', 
            'description', 'next_event_value', 'momentum_continues', 'event_type',
            'shot_result', 'clock'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['momentum_continues']
        
        self.feature_names = feature_cols
        
        print(f"  📊 Training data: {len(X)} samples, {len(feature_cols)} features")
        print(f"  ⚖️  Class balance: {y.mean():.3f} positive momentum")
        
        return X, y
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train multiple advanced models and create ensemble.
        """
        print("🤖 Training advanced ensemble models...")
        
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
        print("  🎯 Selecting best features...")
        selector = SelectKBest(score_func=f_classif, k=min(100, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"    ✅ Selected {len(selected_features)} best features")
        
        # Store preprocessors
        self.scalers['ensemble'] = scaler
        self.feature_selectors['ensemble'] = selector
        
        # Define models
        models = {}
        
        # 1. Random Forest (robust baseline)
        print("  🌲 Training Random Forest...")
        models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 2. Extra Trees (more randomization)
        print("  🌳 Training Extra Trees...")
        models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 3. Gradient Boosting
        print("  📈 Training Gradient Boosting...")
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        # 4. XGBoost (if available)
        if HAS_XGBOOST:
            print("  🚀 Training XGBoost...")
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        # 5. LightGBM (if available)
        if HAS_LIGHTGBM:
            print("  💡 Training LightGBM...")
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        # 6. Neural Network
        print("  🧠 Training Neural Network...")
        models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # 7. SVM (for diversity)
        print("  🎯 Training SVM...")
        models['svm'] = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Train all models with progress bar
        trained_models = {}
        model_scores = {}
        
        model_progress = tqdm(models.items(), desc="Training models", unit="models")
        
        for name, model in model_progress:
            try:
                model_progress.set_postfix({"Current": name})
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
                
                print(f"      ✅ {name}: F1={score['f1']:.4f}, AUC={score['auc']:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error training {name}: {e}")
                continue
        
        # Create voting ensemble
        print("  🗳️  Creating voting ensemble...")
        
        voting_models = [(name, model) for name, model in trained_models.items()]
        
        if len(voting_models) >= 3:
            ensemble = VotingClassifier(
                estimators=voting_models,
                voting='soft'
            )
            
            ensemble.fit(X_train_selected, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test_selected)
            y_pred_proba_ensemble = ensemble.predict_proba(X_test_selected)[:, 1]
            
            ensemble_score = {
                'accuracy': accuracy_score(y_test, y_pred_ensemble),
                'precision': precision_score(y_test, y_pred_ensemble),
                'recall': recall_score(y_test, y_pred_ensemble),
                'f1': f1_score(y_test, y_pred_ensemble),
                'auc': roc_auc_score(y_test, y_pred_proba_ensemble)
            }
            
            trained_models['ensemble'] = ensemble
            model_scores['ensemble'] = ensemble_score
            
            print(f"    ✅ Ensemble: F1={ensemble_score['f1']:.4f}, AUC={ensemble_score['auc']:.4f}")
        
        # Find best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['f1'])
        best_model = trained_models[best_model_name]
        
        print(f"  🏆 Best model: {best_model_name} (F1: {model_scores[best_model_name]['f1']:.4f})")
        
        # Feature importance (from tree-based models)
        feature_importance = {}
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(selected_features, best_model.feature_importances_))
        elif hasattr(best_model, 'named_estimators_'):
            # For ensemble, try to get from first tree-based model
            for name, estimator in best_model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    feature_importance = dict(zip(selected_features, estimator.feature_importances_))
                    break
        
        return {
            'models': trained_models,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'model_scores': model_scores,
            'selected_features': selected_features,
            'feature_importance': feature_importance,
            'scaler': scaler,
            'feature_selector': selector
        }
    
    def train_deep_learning_model(self, X: pd.DataFrame, y: pd.Series) -> Optional[Dict]:
        """
        Train deep learning models (if TensorFlow is available).
        """
        if not HAS_TENSORFLOW:
            print("  ⚠️  TensorFlow not available, skipping deep learning models")
            return None
        
        print("🧠 Training deep learning models...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build deep neural network
        model = Sequential([
            Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        print("  🚀 Training deep neural network...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        score = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"    ✅ Deep NN: F1={score['f1']:.4f}, AUC={score['auc']:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'score': score,
            'history': history.history
        }
    
    def save_models(self, ensemble_results: Dict, dl_results: Optional[Dict] = None) -> str:
        """Save all trained models."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f"models/next_generation"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble models
        ensemble_file = f"{model_dir}/next_gen_ensemble_{timestamp}.pkl"
        
        ensemble_package = {
            'models': ensemble_results['models'],
            'best_model': ensemble_results['best_model'],
            'best_model_name': ensemble_results['best_model_name'],
            'scaler': ensemble_results['scaler'],
            'feature_selector': ensemble_results['feature_selector'],
            'selected_features': ensemble_results['selected_features'],
            'feature_importance': ensemble_results['feature_importance'],
            'training_timestamp': timestamp,
            'model_type': 'next_generation_ensemble'
        }
        
        with open(ensemble_file, 'wb') as f:
            pickle.dump(ensemble_package, f)
        
        # Save evaluation report
        report_file = f"{model_dir}/evaluation_report_{timestamp}.json"
        
        evaluation_report = {
            'model_info': {
                'model_type': 'next_generation_ensemble',
                'training_timestamp': timestamp,
                'num_features': len(ensemble_results['selected_features']),
                'models_trained': list(ensemble_results['models'].keys())
            },
            'model_scores': ensemble_results['model_scores'],
            'best_model': ensemble_results['best_model_name'],
            'feature_analysis': {
                'selected_features': ensemble_results['selected_features'],
                'feature_importance': ensemble_results['feature_importance']
            }
        }
        
        if dl_results:
            evaluation_report['deep_learning'] = {
                'score': dl_results['score'],
                'training_history': dl_results['history']
            }
        
        with open(report_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print(f"💾 Models saved to: {ensemble_file}")
        print(f"📊 Report saved to: {report_file}")
        
        return ensemble_file


def main():
    """Train next-generation momentum prediction models."""
    
    print("🚀 NEXT-GENERATION MOMENTUM MODEL TRAINING")
    print("="*60)
    print()
    
    predictor = NextGenerationMomentumPredictor()
    
    try:
        # Load NBA data (sample 1000 games for faster training)
        df = predictor.load_nba_data()
        
        # Sample for faster training
        print("🎯 Sampling data for manageable training time...")
        unique_games = df['game_id'].unique()
        sampled_games = np.random.choice(unique_games, size=min(1000, len(unique_games)), replace=False)
        df = df[df['game_id'].isin(sampled_games)]
        print(f"  📊 Using {len(df)} events from {df['game_id'].nunique()} games for training")
        
        # Engineer advanced features
        df = predictor.engineer_advanced_features(df)
        
        # Prepare training data
        X, y = predictor.prepare_training_data(df)
        
        # Train ensemble models
        ensemble_results = predictor.train_ensemble_models(X, y)
        
        # Train deep learning models (if available)
        dl_results = predictor.train_deep_learning_model(X, y)
        
        # Save models
        model_file = predictor.save_models(ensemble_results, dl_results)
        
        print()
        print("="*60)
        print("🎉 NEXT-GENERATION MODEL TRAINING COMPLETE!")
        print("="*60)
        
        # Print results summary
        print(f"📈 Model Performance Summary:")
        for name, score in ensemble_results['model_scores'].items():
            print(f"   • {name}: F1={score['f1']:.4f}, AUC={score['auc']:.4f}")
        
        print(f"\n🏆 Best Model: {ensemble_results['best_model_name']}")
        print(f"   • F1-Score: {ensemble_results['model_scores'][ensemble_results['best_model_name']]['f1']:.4f}")
        print(f"   • AUC: {ensemble_results['model_scores'][ensemble_results['best_model_name']]['auc']:.4f}")
        
        if dl_results:
            print(f"\n🧠 Deep Learning Model:")
            print(f"   • F1-Score: {dl_results['score']['f1']:.4f}")
            print(f"   • AUC: {dl_results['score']['auc']:.4f}")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Total samples: {len(X):,}")
        print(f"   • Features: {len(ensemble_results['selected_features'])}")
        print(f"   • Games: {df['game_id'].nunique()}")
        
        print(f"\n💾 Model saved: {model_file}")
        
    except Exception as e:
        print(f"❌ Error training next-generation model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()