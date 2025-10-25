#!/usr/bin/env python3
"""
Test script for ML integration in MomentumML.

This script tests the ML prediction functionality with sample data.
"""

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.services.ml_predictor import MomentumPredictor, train_momentum_model
from backend.services.historical_data_collector import create_sample_training_data
from backend.services.momentum_engine import create_momentum_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ml_training():
    """Test ML model training with sample data."""
    logger.info("Testing ML model training")
    
    try:
        # Create sample training data
        sample_events = create_sample_training_data()
        logger.info(f"Created {len(sample_events)} sample events")
        
        # Train model
        results = train_momentum_model(sample_events, "models/test_model.pkl")
        
        logger.info("Training results:")
        logger.info(f"  Train accuracy: {results['train_accuracy']:.3f}")
        logger.info(f"  Test accuracy: {results['test_accuracy']:.3f}")
        logger.info(f"  Training samples: {results['train_samples']}")
        logger.info(f"  Test samples: {results['test_samples']}")
        
        return True
        
    except Exception as e:
        logger.error(f"ML training test failed: {e}")
        return False


def test_ml_prediction():
    """Test ML model prediction functionality."""
    logger.info("Testing ML model prediction")
    
    try:
        # Load trained model
        predictor = MomentumPredictor("models/test_model.pkl")
        
        if not predictor.is_trained:
            logger.error("Model not trained")
            return False
        
        # Test prediction with empty data (should return neutral prediction)
        prob, confidence = predictor.predict_momentum_continuation([], [])
        
        logger.info(f"Prediction result: probability={prob:.3f}, confidence={confidence:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ML prediction test failed: {e}")
        return False


def test_momentum_engine_integration():
    """Test momentum engine integration with ML predictor."""
    logger.info("Testing momentum engine integration")
    
    try:
        # Create momentum engine with ML prediction enabled
        engine = create_momentum_engine(enable_ml_prediction=True)
        
        # Create sample events
        sample_events = create_sample_training_data()
        
        # Test with a subset of events
        test_events = sample_events[:50]  # Use first 50 events
        
        # Segment possessions
        possessions = engine.segment_possessions(test_events)
        logger.info(f"Segmented {len(test_events)} events into {len(possessions)} possessions")
        
        # Calculate features
        features = engine.calculate_possession_features(possessions)
        logger.info(f"Calculated features for {len(features)} possessions")
        
        # Compute TMI for a team
        if possessions:
            team = possessions[0].team_tricode
            game_id = possessions[0].game_id
            
            tmi = engine.compute_tmi(team, game_id, features[:5])  # Use first 5 features
            
            logger.info(f"TMI calculation result:")
            logger.info(f"  Team: {tmi.team_tricode}")
            logger.info(f"  TMI Value: {tmi.tmi_value:.3f}")
            logger.info(f"  Prediction Probability: {tmi.prediction_probability:.3f}")
            logger.info(f"  Confidence Score: {tmi.confidence_score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Momentum engine integration test failed: {e}")
        return False


def main():
    """Run all ML integration tests."""
    logger.info("Starting ML integration tests")
    
    tests = [
        ("ML Training", test_ml_training),
        ("ML Prediction", test_ml_prediction),
        ("Momentum Engine Integration", test_momentum_engine_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("Test Summary")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "PASSED" if passed_test else "FAILED"
        logger.info(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! ML integration is working correctly.")
        return 0
    else:
        logger.error("Some tests failed. Please check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())