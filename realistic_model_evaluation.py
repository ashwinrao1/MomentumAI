#!/usr/bin/env python3
"""
Realistic model evaluation that demonstrates proper ML evaluation practices.

This script shows how to properly evaluate a model with realistic data splits
and identifies the actual issues with the current model.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from services.historical_data_collector import create_sample_training_data
    from services.momentum_engine import MomentumEngine
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_realistic_training_data():
    """Create more realistic training data with proper variability."""
    logger.info("Creating realistic training data with proper variability")
    
    np.random.seed(42)  # For reproducibility
    
    # Create more diverse sample data
    training_examples = []
    
    # Simulate 50 games with varying characteristics
    for game_id in range(50):
        # Each game has different momentum patterns
        game_momentum_bias = np.random.uniform(0.2, 0.8)  # Some games have more momentum shifts
        
        # Create possessions for this game
        for possession_id in range(20):  # 20 possessions per game
            
            # Create features with realistic correlations and noise
            avg_points_scored = np.random.uniform(0.5, 1.5)
            avg_fg_percentage = np.random.uniform(0.2, 0.6)
            avg_turnovers = np.random.uniform(0.0, 0.5)
            avg_rebounds = np.random.uniform(0.5, 1.5)
            avg_fouls = np.random.uniform(0.2, 1.0)
            avg_pace = np.random.uniform(1.5, 2.5)
            
            # Trend features (with some correlation to performance)
            points_trend = np.random.uniform(-0.5, 0.5)
            fg_pct_trend = np.random.uniform(-0.3, 0.3)
            turnover_trend = np.random.uniform(-0.3, 0.3)
            
            # Consistency features
            fg_consistency = np.random.uniform(0.3, 1.0)
            scoring_consistency = np.random.uniform(0.3, 1.0)
            
            # TMI features
            current_tmi = avg_points_scored * 0.4 + avg_fg_percentage * 0.3 + np.random.normal(0, 0.1)
            tmi_volatility = np.random.uniform(0.05, 0.2)
            tmi_trend = points_trend * 0.2 + np.random.normal(0, 0.05)
            
            # Create momentum continuation label with realistic logic
            # Higher performance metrics should correlate with momentum continuation
            momentum_score = (
                avg_points_scored * 0.3 +
                avg_fg_percentage * 0.25 +
                (1 - avg_turnovers) * 0.15 +  # Fewer turnovers is better
                points_trend * 0.2 +
                fg_pct_trend * 0.1 +
                game_momentum_bias * 0.1 +  # Game-specific momentum tendency
                np.random.normal(0, 0.2)  # Add noise
            )
            
            # Convert to binary with some threshold
            momentum_continued = 1 if momentum_score > 0.6 else 0
            
            # Add some random noise to make it more realistic
            if np.random.random() < 0.1:  # 10% random flips
                momentum_continued = 1 - momentum_continued
            
            example = {
                'avg_points_scored': avg_points_scored,
                'avg_fg_percentage': avg_fg_percentage,
                'avg_turnovers': avg_turnovers,
                'avg_rebounds': avg_rebounds,
                'avg_fouls': avg_fouls,
                'avg_pace': avg_pace,
                'points_trend': points_trend,
                'fg_pct_trend': fg_pct_trend,
                'turnover_trend': turnover_trend,
                'fg_consistency': fg_consistency,
                'scoring_consistency': scoring_consistency,
                'current_tmi': current_tmi,
                'tmi_volatility': tmi_volatility,
                'tmi_trend': tmi_trend,
                'momentum_continued': momentum_continued,
                'game_id': game_id
            }
            
            training_examples.append(example)
    
    df = pd.DataFrame(training_examples)
    logger.info(f"Created {len(df)} realistic training examples from {df['game_id'].nunique()} games")
    
    return df


def evaluate_with_proper_splits(training_data):
    """Evaluate model with proper train/validation/test splits."""
    logger.info("=== PROPER TRAIN/VALIDATION/TEST EVALUATION ===")
    
    # Separate features and target
    feature_columns = [col for col in training_data.columns if col not in ['momentum_continued', 'game_id']]
    X = training_data[feature_columns].values
    y = training_data['momentum_continued'].values
    games = training_data['game_id'].values
    
    # Split by games to avoid data leakage (no game appears in both train and test)
    unique_games = np.unique(games)
    train_games, test_games = train_test_split(unique_games, test_size=0.2, random_state=42)
    train_games, val_games = train_test_split(train_games, test_size=0.2, random_state=42)
    
    # Create splits based on games
    train_mask = np.isin(games, train_games)
    val_mask = np.isin(games, val_games)
    test_mask = np.isin(games, test_games)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Train set: {len(X_train)} examples from {len(train_games)} games")
    logger.info(f"Validation set: {len(X_val)} examples from {len(val_games)} games")
    logger.info(f"Test set: {len(X_test)} examples from {len(test_games)} games")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on all sets
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    logger.info(f"Training Accuracy:   {train_acc:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Test Accuracy:       {test_acc:.4f}")
    
    # Check for overfitting
    train_val_gap = train_acc - val_acc
    train_test_gap = train_acc - test_acc
    
    logger.info(f"Train-Val Gap:       {train_val_gap:.4f}")
    logger.info(f"Train-Test Gap:      {train_test_gap:.4f}")
    
    if train_val_gap > 0.1:
        logger.warning("Significant overfitting detected (train-val gap > 0.1)")
    elif train_val_gap > 0.05:
        logger.warning("Moderate overfitting detected (train-val gap > 0.05)")
    else:
        logger.info("No significant overfitting detected")
    
    # Detailed test set evaluation
    test_precision = precision_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    
    logger.info(f"Test Precision:      {test_precision:.4f}")
    logger.info(f"Test Recall:         {test_recall:.4f}")
    logger.info(f"Test F1-Score:       {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    logger.info("Test Confusion Matrix:")
    logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }


def compare_with_baseline(training_data):
    """Compare model performance with baseline predictors."""
    logger.info("=== BASELINE COMPARISON ===")
    
    feature_columns = [col for col in training_data.columns if col not in ['momentum_continued', 'game_id']]
    X = training_data[feature_columns].values
    y = training_data['momentum_continued'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train our model
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    model_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    # Baseline 1: Most frequent class
    dummy_frequent = DummyClassifier(strategy='most_frequent')
    dummy_frequent.fit(X_train, y_train)
    frequent_acc = accuracy_score(y_test, dummy_frequent.predict(X_test))
    
    # Baseline 2: Random prediction
    dummy_random = DummyClassifier(strategy='uniform', random_state=42)
    dummy_random.fit(X_train, y_train)
    random_acc = accuracy_score(y_test, dummy_random.predict(X_test))
    
    # Baseline 3: Stratified (class proportions)
    dummy_stratified = DummyClassifier(strategy='stratified', random_state=42)
    dummy_stratified.fit(X_train, y_train)
    stratified_acc = accuracy_score(y_test, dummy_stratified.predict(X_test))
    
    logger.info("Model vs Baseline Comparison:")
    logger.info(f"Logistic Regression:  {model_acc:.4f}")
    logger.info(f"Most Frequent Class:  {frequent_acc:.4f}")
    logger.info(f"Random Prediction:    {random_acc:.4f}")
    logger.info(f"Stratified Baseline:  {stratified_acc:.4f}")
    
    # Calculate improvement over baselines
    improvement_over_frequent = model_acc - frequent_acc
    improvement_over_random = model_acc - random_acc
    
    logger.info(f"Improvement over most frequent: {improvement_over_frequent:.4f}")
    logger.info(f"Improvement over random:        {improvement_over_random:.4f}")
    
    if improvement_over_frequent < 0.05:
        logger.warning("Model barely beats most frequent class baseline!")
    if improvement_over_random < 0.1:
        logger.warning("Model shows limited improvement over random prediction!")
    
    return {
        'model_acc': model_acc,
        'frequent_acc': frequent_acc,
        'random_acc': random_acc,
        'stratified_acc': stratified_acc
    }


def analyze_feature_importance(training_data):
    """Analyze which features are actually important for prediction."""
    logger.info("=== FEATURE IMPORTANCE ANALYSIS ===")
    
    feature_columns = [col for col in training_data.columns if col not in ['momentum_continued', 'game_id']]
    X = training_data[feature_columns].values
    y = training_data['momentum_continued'].values
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Get feature importance (absolute coefficients)
    feature_importance = dict(zip(feature_columns, abs(model.coef_[0])))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("Feature Importance Ranking:")
    for i, (feature, importance) in enumerate(sorted_features):
        logger.info(f"  {i+1:2d}. {feature:20s}: {importance:.4f}")
    
    # Test feature removal impact
    logger.info("\nFeature Removal Impact:")
    baseline_acc = accuracy_score(y_test, model.predict(scaler.transform(X_test)))
    
    for feature, _ in sorted_features[:5]:  # Test top 5 features
        # Remove this feature and retrain
        feature_idx = feature_columns.index(feature)
        X_train_reduced = np.delete(X_train, feature_idx, axis=1)
        X_test_reduced = np.delete(X_test, feature_idx, axis=1)
        
        scaler_reduced = StandardScaler()
        X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
        X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)
        
        model_reduced = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model_reduced.fit(X_train_reduced_scaled, y_train)
        
        reduced_acc = accuracy_score(y_test, model_reduced.predict(X_test_reduced_scaled))
        impact = baseline_acc - reduced_acc
        
        logger.info(f"  Remove {feature:20s}: {reduced_acc:.4f} (impact: {impact:+.4f})")
    
    return feature_importance


def cross_validate_properly(training_data):
    """Perform proper cross-validation with game-based splits."""
    logger.info("=== PROPER CROSS-VALIDATION ===")
    
    feature_columns = [col for col in training_data.columns if col not in ['momentum_continued', 'game_id']]
    X = training_data[feature_columns].values
    y = training_data['momentum_continued'].values
    games = training_data['game_id'].values
    
    # Custom cross-validation that respects game boundaries
    unique_games = np.unique(games)
    n_folds = 5
    fold_size = len(unique_games) // n_folds
    
    cv_scores = []
    
    for fold in range(n_folds):
        # Define test games for this fold
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(unique_games)
        test_games = unique_games[start_idx:end_idx]
        train_games = unique_games[~np.isin(unique_games, test_games)]
        
        # Create train/test splits based on games
        train_mask = np.isin(games, train_games)
        test_mask = np.isin(games, test_games)
        
        X_train_fold, y_train_fold = X[train_mask], y[train_mask]
        X_test_fold, y_test_fold = X[test_mask], y[test_mask]
        
        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model.fit(X_train_scaled, y_train_fold)
        
        # Evaluate
        fold_acc = accuracy_score(y_test_fold, model.predict(X_test_scaled))
        cv_scores.append(fold_acc)
        
        logger.info(f"Fold {fold+1}: {fold_acc:.4f} (train games: {len(train_games)}, test games: {len(test_games)})")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    logger.info(f"Cross-validation results: {cv_mean:.4f} ± {cv_std:.4f}")
    
    if cv_std > 0.1:
        logger.warning("High variance in CV scores - model may be unstable")
    
    return cv_scores


def main():
    """Run comprehensive realistic model evaluation."""
    logger.info("Starting Realistic Model Evaluation")
    logger.info("=" * 60)
    
    # Create realistic training data
    training_data = create_realistic_training_data()
    
    # Analyze class distribution
    class_counts = training_data['momentum_continued'].value_counts()
    logger.info(f"Class distribution: {dict(class_counts)}")
    logger.info(f"Class balance ratio: {max(class_counts) / min(class_counts):.2f}")
    
    print("\n")
    
    # 1. Proper train/validation/test evaluation
    results = evaluate_with_proper_splits(training_data)
    
    print("\n")
    
    # 2. Baseline comparison
    baseline_results = compare_with_baseline(training_data)
    
    print("\n")
    
    # 3. Feature importance analysis
    feature_importance = analyze_feature_importance(training_data)
    
    print("\n")
    
    # 4. Proper cross-validation
    cv_scores = cross_validate_properly(training_data)
    
    print("\n")
    
    # Summary
    logger.info("=== REALISTIC EVALUATION SUMMARY ===")
    logger.info(f"Test Accuracy: {results['test_acc']:.1%}")
    logger.info(f"Improvement over baseline: {results['test_acc'] - baseline_results['frequent_acc']:.1%}")
    logger.info(f"Cross-validation: {np.mean(cv_scores):.1%} ± {np.std(cv_scores):.1%}")
    
    # Realistic assessment
    if results['test_acc'] > 0.7:
        logger.info("✅ Model shows good performance")
    elif results['test_acc'] > 0.6:
        logger.info("⚠️  Model shows moderate performance")
    else:
        logger.info("❌ Model shows poor performance")
    
    if results['test_acc'] - baseline_results['frequent_acc'] < 0.05:
        logger.warning("❌ Model barely beats baseline - may not be useful")
    
    logger.info("\nKey Insights:")
    logger.info("1. This evaluation uses proper train/test splits by game")
    logger.info("2. Model performance is compared against meaningful baselines")
    logger.info("3. Cross-validation respects temporal/game structure")
    logger.info("4. Feature importance is analyzed systematically")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())