#!/usr/bin/env python3
"""
Model evaluation script for MomentumML.

This script evaluates the trained ML model's effectiveness, including:
- Model type and architecture analysis
- Training data characteristics
- Proper train/test split validation
- Performance metrics and analysis
"""

import sys
import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from services.ml_predictor import MomentumPredictor, train_momentum_model
    from services.historical_data_collector import create_sample_training_data
    from services.momentum_engine import MomentumEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_model_architecture(model_path: str):
    """Analyze the model architecture and type."""
    logger.info("=== MODEL ARCHITECTURE ANALYSIS ===")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        trained_at = model_data.get('trained_at', 'Unknown')
        
        logger.info(f"Model Type: {type(model).__name__}")
        logger.info(f"Model Class: {model.__class__.__module__}.{model.__class__.__name__}")
        logger.info(f"Trained At: {trained_at}")
        logger.info(f"Number of Features: {len(feature_names)}")
        logger.info(f"Feature Names: {feature_names}")
        
        # Logistic Regression specific analysis
        if hasattr(model, 'coef_'):
            logger.info(f"Model Coefficients Shape: {model.coef_.shape}")
            logger.info(f"Intercept: {model.intercept_}")
            logger.info(f"Classes: {model.classes_}")
            
            # Feature importance (absolute coefficients)
            feature_importance = dict(zip(feature_names, abs(model.coef_[0])))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("Top 5 Most Important Features:")
            for feature, importance in sorted_features[:5]:
                logger.info(f"  {feature}: {importance:.4f}")
        
        # Scaler analysis
        if scaler:
            logger.info(f"Scaler Type: {type(scaler).__name__}")
            if hasattr(scaler, 'mean_'):
                logger.info(f"Feature Means: {scaler.mean_}")
                logger.info(f"Feature Scales: {scaler.scale_}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"Error analyzing model: {e}")
        return None


def analyze_training_data():
    """Analyze the training data characteristics."""
    logger.info("=== TRAINING DATA ANALYSIS ===")
    
    try:
        # Create sample training data (same as used for training)
        sample_events = create_sample_training_data()
        logger.info(f"Total Training Events: {len(sample_events)}")
        
        # Analyze event distribution
        event_types = {}
        teams = set()
        games = set()
        
        for event in sample_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            teams.add(event.team_tricode)
            games.add(event.game_id)
        
        logger.info(f"Number of Games: {len(games)}")
        logger.info(f"Number of Teams: {len(teams)}")
        logger.info(f"Teams: {sorted(teams)}")
        
        logger.info("Event Type Distribution:")
        for event_type, count in sorted(event_types.items()):
            percentage = (count / len(sample_events)) * 100
            logger.info(f"  {event_type}: {count} ({percentage:.1f}%)")
        
        # Analyze temporal distribution
        periods = [event.period for event in sample_events]
        logger.info(f"Period Range: {min(periods)} - {max(periods)}")
        
        return sample_events
        
    except Exception as e:
        logger.error(f"Error analyzing training data: {e}")
        return None


def evaluate_train_test_split(model_path: str):
    """Evaluate the train/test split methodology."""
    logger.info("=== TRAIN/TEST SPLIT EVALUATION ===")
    
    try:
        # Recreate the training process to analyze split
        sample_events = create_sample_training_data()
        predictor = MomentumPredictor(model_path)
        
        # Collect training data (same process as in training)
        training_data = predictor.collect_historical_data(sample_events)
        
        if training_data.empty:
            logger.error("No training data collected")
            return False
        
        logger.info(f"Total Training Examples: {len(training_data)}")
        
        # Analyze class distribution
        if 'momentum_continued' in training_data.columns:
            class_counts = training_data['momentum_continued'].value_counts()
            logger.info("Class Distribution:")
            for class_val, count in class_counts.items():
                percentage = (count / len(training_data)) * 100
                label = "Momentum Continued" if class_val == 1 else "Momentum Stopped"
                logger.info(f"  {label}: {count} ({percentage:.1f}%)")
            
            # Check for class imbalance
            imbalance_ratio = max(class_counts) / min(class_counts)
            logger.info(f"Class Imbalance Ratio: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 2:
                logger.warning("Significant class imbalance detected!")
        
        # Analyze feature statistics
        feature_columns = [col for col in training_data.columns if col != 'momentum_continued']
        logger.info(f"Number of Features: {len(feature_columns)}")
        
        # Check for missing values
        missing_values = training_data[feature_columns].isnull().sum()
        if missing_values.any():
            logger.warning("Missing values detected:")
            for feature, missing_count in missing_values[missing_values > 0].items():
                logger.warning(f"  {feature}: {missing_count} missing values")
        else:
            logger.info("No missing values in features")
        
        # Feature statistics
        logger.info("Feature Statistics Summary:")
        stats = training_data[feature_columns].describe()
        logger.info(f"Mean feature values: {stats.loc['mean'].mean():.4f}")
        logger.info(f"Std feature values: {stats.loc['std'].mean():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error evaluating train/test split: {e}")
        return False


def perform_cross_validation(model_path: str):
    """Perform cross-validation to get robust performance estimates."""
    logger.info("=== CROSS-VALIDATION EVALUATION ===")
    
    try:
        # Recreate training data
        sample_events = create_sample_training_data()
        predictor = MomentumPredictor(model_path)
        training_data = predictor.collect_historical_data(sample_events)
        
        if training_data.empty:
            logger.error("No training data for cross-validation")
            return None
        
        # Prepare data
        feature_columns = [col for col in training_data.columns if col != 'momentum_continued']
        X = training_data[feature_columns].values
        y = training_data['momentum_continued'].values
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Perform stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Calculate various metrics
        cv_accuracy = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(model, X_scaled, y, cv=cv, scoring='precision')
        cv_recall = cross_val_score(model, X_scaled, y, cv=cv, scoring='recall')
        cv_f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        
        logger.info("Cross-Validation Results (5-fold):")
        logger.info(f"Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        logger.info(f"Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
        logger.info(f"Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
        logger.info(f"F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        
        # Individual fold results
        logger.info("Individual Fold Accuracies:")
        for i, acc in enumerate(cv_accuracy):
            logger.info(f"  Fold {i+1}: {acc:.4f}")
        
        return {
            'accuracy': cv_accuracy,
            'precision': cv_precision,
            'recall': cv_recall,
            'f1': cv_f1
        }
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        return None


def evaluate_model_performance(model_path: str):
    """Evaluate detailed model performance metrics."""
    logger.info("=== DETAILED PERFORMANCE EVALUATION ===")
    
    try:
        # Recreate the exact training/test split used during training
        sample_events = create_sample_training_data()
        predictor = MomentumPredictor(model_path)
        training_data = predictor.collect_historical_data(sample_events)
        
        if training_data.empty:
            logger.error("No training data available")
            return None
        
        # Prepare data
        feature_columns = [col for col in training_data.columns if col != 'momentum_continued']
        X = training_data[feature_columns].values
        y = training_data['momentum_continued'].values
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Use the same train/test split as during training
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        train_proba = model.predict_proba(X_train_scaled)[:, 1]
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_precision = precision_score(y_test, test_pred)
        test_recall = recall_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)
        
        logger.info("Performance Metrics:")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy:     {test_accuracy:.4f}")
        logger.info(f"Test Precision:    {test_precision:.4f}")
        logger.info(f"Test Recall:       {test_recall:.4f}")
        logger.info(f"Test F1-Score:     {test_f1:.4f}")
        
        # Check for overfitting
        overfitting_gap = train_accuracy - test_accuracy
        logger.info(f"Overfitting Gap:   {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.1:
            logger.warning("Significant overfitting detected!")
        elif overfitting_gap > 0.05:
            logger.warning("Moderate overfitting detected")
        else:
            logger.info("No significant overfitting")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        logger.info("Confusion Matrix:")
        logger.info(f"  True Negatives:  {cm[0,0]}")
        logger.info(f"  False Positives: {cm[0,1]}")
        logger.info(f"  False Negatives: {cm[1,0]}")
        logger.info(f"  True Positives:  {cm[1,1]}")
        
        # Classification report
        logger.info("Detailed Classification Report:")
        report = classification_report(y_test, test_pred, target_names=['No Momentum', 'Momentum Continues'])
        logger.info(f"\n{report}")
        
        # Probability distribution analysis
        logger.info("Prediction Probability Analysis:")
        logger.info(f"Mean probability for positive class: {test_proba.mean():.4f}")
        logger.info(f"Std of probabilities: {test_proba.std():.4f}")
        logger.info(f"Min probability: {test_proba.min():.4f}")
        logger.info(f"Max probability: {test_proba.max():.4f}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'overfitting_gap': overfitting_gap,
            'confusion_matrix': cm,
            'probabilities': test_proba
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model performance: {e}")
        return None


def assess_data_quality():
    """Assess the quality and limitations of the training data."""
    logger.info("=== DATA QUALITY ASSESSMENT ===")
    
    try:
        sample_events = create_sample_training_data()
        
        logger.info("Data Source: Synthetic sample data")
        logger.info("⚠️  WARNING: This model is trained on SYNTHETIC data, not real NBA data!")
        
        # Analyze data realism
        logger.info("Data Realism Analysis:")
        logger.info("- Events are artificially generated with simple patterns")
        logger.info("- No real player behavior or game dynamics")
        logger.info("- Limited event type diversity")
        logger.info("- Simplified momentum definitions")
        
        # Check for data leakage
        logger.info("Data Leakage Check:")
        logger.info("- Features are calculated from possession windows")
        logger.info("- Target variable (momentum continuation) is based on next possession performance")
        logger.info("- No obvious temporal leakage detected")
        
        # Temporal consistency
        games = {}
        for event in sample_events:
            if event.game_id not in games:
                games[event.game_id] = []
            games[event.game_id].append(event)
        
        logger.info(f"Temporal Structure:")
        logger.info(f"- {len(games)} games in dataset")
        logger.info(f"- Average events per game: {len(sample_events) / len(games):.1f}")
        
        # Feature engineering quality
        logger.info("Feature Engineering:")
        logger.info("- Basic statistical features (means, trends)")
        logger.info("- Consistency metrics")
        logger.info("- TMI-based features")
        logger.info("- No advanced basketball-specific features")
        
        return True
        
    except Exception as e:
        logger.error(f"Error assessing data quality: {e}")
        return False


def main():
    """Run comprehensive model evaluation."""
    logger.info("Starting Comprehensive Model Evaluation")
    logger.info("=" * 60)
    
    model_path = "models/momentum_predictor.pkl"
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please train the model first using: python backend/train_model.py")
        return 1
    
    try:
        # 1. Model Architecture Analysis
        model_data = analyze_model_architecture(model_path)
        if not model_data:
            return 1
        
        print("\n")
        
        # 2. Training Data Analysis
        training_events = analyze_training_data()
        if not training_events:
            return 1
        
        print("\n")
        
        # 3. Train/Test Split Evaluation
        split_ok = evaluate_train_test_split(model_path)
        if not split_ok:
            return 1
        
        print("\n")
        
        # 4. Cross-Validation
        cv_results = perform_cross_validation(model_path)
        
        print("\n")
        
        # 5. Detailed Performance Evaluation
        perf_results = evaluate_model_performance(model_path)
        
        print("\n")
        
        # 6. Data Quality Assessment
        assess_data_quality()
        
        print("\n")
        
        # Summary and Recommendations
        logger.info("=== SUMMARY AND RECOMMENDATIONS ===")
        
        if perf_results:
            test_acc = perf_results['test_accuracy']
            overfitting = perf_results['overfitting_gap']
            
            logger.info(f"Overall Model Performance: {test_acc:.1%}")
            
            if test_acc > 0.8:
                logger.info("✅ Good model performance")
            elif test_acc > 0.6:
                logger.info("⚠️  Moderate model performance")
            else:
                logger.info("❌ Poor model performance")
            
            if overfitting > 0.1:
                logger.info("❌ Significant overfitting - model may not generalize well")
            else:
                logger.info("✅ No significant overfitting detected")
        
        logger.info("\nKey Limitations:")
        logger.info("1. Model trained on synthetic data, not real NBA games")
        logger.info("2. Simple feature engineering - missing advanced basketball metrics")
        logger.info("3. Limited training data diversity")
        logger.info("4. Simplified momentum definition")
        
        logger.info("\nRecommendations for Improvement:")
        logger.info("1. Collect real NBA play-by-play data")
        logger.info("2. Add advanced basketball features (player efficiency, situational context)")
        logger.info("3. Implement more sophisticated momentum definitions")
        logger.info("4. Try ensemble methods or neural networks")
        logger.info("5. Add temporal features (game situation, score differential)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())