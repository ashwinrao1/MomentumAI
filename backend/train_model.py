#!/usr/bin/env python3
"""
Model training script for MomentumML.

This script collects historical data and trains the momentum prediction model.
Can be run standalone or imported as a module.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from services.historical_data_collector import HistoricalDataCollector, create_sample_training_data
from services.ml_predictor import MomentumPredictor, train_momentum_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_sample_data(model_path: str = "models/momentum_predictor.pkl") -> dict:
    """
    Train the model using sample data (for testing/demo purposes).
    
    Args:
        model_path: Path to save the trained model
        
    Returns:
        Training results dictionary
    """
    logger.info("Training model with sample data")
    
    # Create sample training data
    sample_events = create_sample_training_data()
    
    # Train model
    results = train_momentum_model(sample_events, model_path)
    
    return results


def train_with_historical_data(
    days_back: int = 30,
    max_games: int = 200,
    season: str = "2023-24",
    model_path: str = "models/momentum_predictor.pkl"
) -> dict:
    """
    Train the model using real historical NBA data.
    
    Args:
        days_back: Number of days back to collect data
        max_games: Maximum number of games to process
        season: NBA season to collect from
        model_path: Path to save the trained model
        
    Returns:
        Training results dictionary
    """
    logger.info(f"Training model with historical data from last {days_back} days")
    
    try:
        # Collect historical data
        collector = HistoricalDataCollector()
        historical_events = collector.collect_historical_dataset(
            days_back=days_back,
            max_games=max_games,
            season=season
        )
        
        if not historical_events:
            logger.error("No historical data collected, falling back to sample data")
            return train_with_sample_data(model_path)
        
        # Train model
        results = train_momentum_model(historical_events, model_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error training with historical data: {e}")
        logger.info("Falling back to sample data training")
        return train_with_sample_data(model_path)


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train MomentumML prediction model")
    
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use sample data instead of real NBA data"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Number of days back to collect historical data (default: 30)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=200,
        help="Maximum number of games to process (default: 200)"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2023-24",
        help="NBA season to collect data from (default: 2023-24)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/momentum_predictor.pkl",
        help="Path to save the trained model (default: models/momentum_predictor.pkl)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting MomentumML model training")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        if args.use_sample_data:
            results = train_with_sample_data(args.model_path)
        else:
            results = train_with_historical_data(
                days_back=args.days_back,
                max_games=args.max_games,
                season=args.season,
                model_path=args.model_path
            )
        
        # Print results
        logger.info("Training completed successfully!")
        logger.info(f"Training accuracy: {results['train_accuracy']:.3f}")
        logger.info(f"Test accuracy: {results['test_accuracy']:.3f}")
        logger.info(f"Training samples: {results['train_samples']}")
        logger.info(f"Test samples: {results['test_samples']}")
        
        # Print feature importance
        logger.info("Feature importance:")
        for feature, importance in sorted(
            results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  {feature}: {importance:.3f}")
        
        logger.info(f"Model saved to: {args.model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()