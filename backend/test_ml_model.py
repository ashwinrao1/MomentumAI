#!/usr/bin/env python3
"""
Unit tests for ML model functionality in MomentumML.

This module contains comprehensive unit tests for the ML prediction model,
including training, prediction, feature extraction, and model persistence.
"""

import unittest
import tempfile
import os
import shutil
from datetime import datetime
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.services.ml_predictor import MomentumPredictor, train_momentum_model
from backend.services.historical_data_collector import create_sample_training_data
from backend.models.game_models import GameEvent, TeamMomentumIndex
from backend.services.momentum_engine import PossessionFeatures


class TestMomentumPredictor(unittest.TestCase):
    """Test cases for MomentumPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.predictor = MomentumPredictor(model_path=self.model_path)
        self.sample_events = create_sample_training_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        self.assertIsNotNone(self.predictor)
        self.assertEqual(self.predictor.model_path, self.model_path)
        self.assertFalse(self.predictor.is_trained)
        self.assertIsNone(self.predictor.model)
        self.assertIsNone(self.predictor.scaler)
        self.assertEqual(self.predictor.feature_names, [])
    
    def test_collect_historical_data(self):
        """Test historical data collection."""
        training_data = self.predictor.collect_historical_data(self.sample_events)
        
        self.assertIsNotNone(training_data)
        self.assertGreater(len(training_data), 0)
        self.assertIn('momentum_continued', training_data.columns)
        
        # Check that we have expected feature columns
        expected_features = [
            'avg_points_scored', 'avg_fg_percentage', 'avg_turnovers',
            'avg_rebounds', 'avg_fouls', 'points_trend', 'fg_pct_trend'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, training_data.columns)
    
    def test_model_training(self):
        """Test model training functionality."""
        # Collect training data
        training_data = self.predictor.collect_historical_data(self.sample_events)
        
        # Train model
        results = self.predictor.train_model(training_data)
        
        # Verify training results
        self.assertIsNotNone(results)
        self.assertIn('train_accuracy', results)
        self.assertIn('test_accuracy', results)
        self.assertIn('feature_importance', results)
        
        # Check accuracy is reasonable (should be > 0.5 for binary classification)
        self.assertGreater(results['train_accuracy'], 0.5)
        self.assertGreater(results['test_accuracy'], 0.5)
        
        # Verify model is trained
        self.assertTrue(self.predictor.is_trained)
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertGreater(len(self.predictor.feature_names), 0)
    
    def test_feature_extraction(self):
        """Test feature extraction for prediction."""
        # Create sample TMI history
        tmi_history = [
            TeamMomentumIndex(
                game_id="test_game",
                team_tricode="LAL",
                timestamp=datetime.utcnow(),
                tmi_value=0.5,
                feature_contributions={},
                rolling_window_size=5,
                prediction_probability=0.6,
                confidence_score=0.8
            )
        ]
        
        # Create sample possession features
        possession_features = [
            PossessionFeatures(
                team_tricode="LAL",
                possession_id="test_poss_1",
                points_scored=2,
                fg_attempts=1,
                fg_made=1,
                fg_percentage=1.0,
                turnovers=0,
                rebounds=1,
                fouls=0,
                possession_duration=24.0,
                pace=2.5
            ),
            PossessionFeatures(
                team_tricode="LAL",
                possession_id="test_poss_2",
                points_scored=0,
                fg_attempts=1,
                fg_made=0,
                fg_percentage=0.0,
                turnovers=1,
                rebounds=0,
                fouls=1,
                possession_duration=18.0,
                pace=3.3
            )
        ]
        
        # Extract features
        features = self.predictor.extract_prediction_features(tmi_history, possession_features)
        
        # Verify feature extraction
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 1)  # One sample
        self.assertGreater(features.shape[1], 0)  # Multiple features
    
    def test_prediction_without_training(self):
        """Test prediction behavior when model is not trained."""
        tmi_history = []
        possession_features = []
        
        prob, confidence = self.predictor.predict_momentum_continuation(tmi_history, possession_features)
        
        # Should return neutral prediction
        self.assertEqual(prob, 0.5)
        self.assertEqual(confidence, 0.0)
    
    def test_prediction_with_training(self):
        """Test prediction after model training."""
        # Train model first
        training_data = self.predictor.collect_historical_data(self.sample_events)
        self.predictor.train_model(training_data)
        
        # Create test data
        tmi_history = []
        possession_features = [
            PossessionFeatures(
                team_tricode="LAL",
                possession_id="test_poss",
                points_scored=2,
                fg_attempts=1,
                fg_made=1,
                fg_percentage=1.0,
                turnovers=0,
                rebounds=1,
                fouls=0,
                possession_duration=24.0,
                pace=2.5
            )
        ]
        
        # Make prediction
        prob, confidence = self.predictor.predict_momentum_continuation(tmi_history, possession_features)
        
        # Verify prediction
        self.assertIsInstance(prob, float)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train model
        training_data = self.predictor.collect_historical_data(self.sample_events)
        self.predictor.train_model(training_data)
        
        # Save model
        save_success = self.predictor.save_model()
        self.assertTrue(save_success)
        self.assertTrue(os.path.exists(self.model_path))
        
        # Create new predictor and load model
        new_predictor = MomentumPredictor(model_path=self.model_path)
        load_success = new_predictor.load_model()
        
        self.assertTrue(load_success)
        self.assertTrue(new_predictor.is_trained)
        self.assertIsNotNone(new_predictor.model)
        self.assertIsNotNone(new_predictor.scaler)
        self.assertEqual(new_predictor.feature_names, self.predictor.feature_names)
    
    def test_model_loading_nonexistent_file(self):
        """Test loading model from non-existent file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pkl")
        predictor = MomentumPredictor(model_path=nonexistent_path)
        
        load_success = predictor.load_model()
        self.assertFalse(load_success)
        self.assertFalse(predictor.is_trained)
    
    def test_insufficient_training_data(self):
        """Test training with insufficient data."""
        # Create minimal training data
        minimal_events = self.sample_events[:10]  # Very few events
        
        with self.assertRaises(ValueError):
            training_data = self.predictor.collect_historical_data(minimal_events)
            self.predictor.train_model(training_data)
    
    def test_feature_consistency(self):
        """Test that feature extraction is consistent."""
        # Train model
        training_data = self.predictor.collect_historical_data(self.sample_events)
        self.predictor.train_model(training_data)
        
        # Create identical test data
        possession_features = [
            PossessionFeatures(
                team_tricode="LAL",
                possession_id="test_poss",
                points_scored=2,
                fg_attempts=1,
                fg_made=1,
                fg_percentage=1.0,
                turnovers=0,
                rebounds=1,
                fouls=0,
                possession_duration=24.0,
                pace=2.5
            )
        ]
        
        # Extract features multiple times
        features1 = self.predictor.extract_prediction_features([], possession_features)
        features2 = self.predictor.extract_prediction_features([], possession_features)
        
        # Should be identical
        self.assertTrue((features1 == features2).all())


class TestTrainMomentumModel(unittest.TestCase):
    """Test cases for train_momentum_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.sample_events = create_sample_training_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_train_momentum_model_function(self):
        """Test the train_momentum_model utility function."""
        results = train_momentum_model(self.sample_events, self.model_path)
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIn('train_accuracy', results)
        self.assertIn('test_accuracy', results)
        self.assertIn('feature_importance', results)
        
        # Verify model file was created
        self.assertTrue(os.path.exists(self.model_path))
        
        # Verify model can be loaded
        predictor = MomentumPredictor(model_path=self.model_path)
        load_success = predictor.load_model()
        self.assertTrue(load_success)
        self.assertTrue(predictor.is_trained)


class TestMLModelAccuracy(unittest.TestCase):
    """Test cases for ML model accuracy and performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "accuracy_test_model.pkl")
        self.sample_events = create_sample_training_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_accuracy_threshold(self):
        """Test that model achieves minimum accuracy threshold."""
        # Train model
        results = train_momentum_model(self.sample_events, self.model_path)
        
        # Check accuracy meets minimum threshold (60% as per requirements)
        min_accuracy = 0.6
        self.assertGreaterEqual(results['test_accuracy'], min_accuracy,
                               f"Model accuracy {results['test_accuracy']:.3f} below threshold {min_accuracy}")
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for identical inputs."""
        # Train model
        train_momentum_model(self.sample_events, self.model_path)
        
        # Load model
        predictor = MomentumPredictor(model_path=self.model_path)
        
        # Create test data
        possession_features = [
            PossessionFeatures(
                team_tricode="LAL",
                possession_id="test_poss",
                points_scored=2,
                fg_attempts=1,
                fg_made=1,
                fg_percentage=1.0,
                turnovers=0,
                rebounds=1,
                fouls=0,
                possession_duration=24.0,
                pace=2.5
            )
        ]
        
        # Make multiple predictions with same input
        predictions = []
        for _ in range(5):
            prob, confidence = predictor.predict_momentum_continuation([], possession_features)
            predictions.append((prob, confidence))
        
        # All predictions should be identical
        first_prediction = predictions[0]
        for prediction in predictions[1:]:
            self.assertEqual(prediction[0], first_prediction[0])
            self.assertEqual(prediction[1], first_prediction[1])
    
    def test_feature_importance_validity(self):
        """Test that feature importance values are valid."""
        # Train model
        results = train_momentum_model(self.sample_events, self.model_path)
        
        feature_importance = results['feature_importance']
        
        # Check that all importance values are non-negative
        for feature, importance in feature_importance.items():
            self.assertGreaterEqual(importance, 0.0,
                                   f"Feature {feature} has negative importance: {importance}")
        
        # Check that we have reasonable number of features
        self.assertGreater(len(feature_importance), 5,
                          "Too few features in importance ranking")


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    unittest.main(verbosity=2)