#!/usr/bin/env python3
"""
Final integration test for the complete ML prediction system.
"""

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.services.momentum_engine import create_momentum_engine
from backend.services.historical_data_collector import create_sample_training_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_complete_pipeline():
    """Test the complete ML prediction pipeline."""
    logger.info("Testing complete ML prediction pipeline")
    
    try:
        # Create momentum engine with ML prediction
        engine = create_momentum_engine(enable_ml_prediction=True)
        
        # Create sample events
        sample_events = create_sample_training_data()
        logger.info(f"Created {len(sample_events)} sample events")
        
        # Use events from one game
        game_events = [e for e in sample_events if e.game_id == "sample_game_000"]
        logger.info(f"Using {len(game_events)} events from one game")
        
        # Segment possessions
        possessions = engine.segment_possessions(game_events)
        logger.info(f"Segmented into {len(possessions)} possessions")
        
        # Process possessions for both teams
        teams = list(set(p.team_tricode for p in possessions))
        logger.info(f"Found teams: {teams}")
        
        for team in teams:
            team_possessions = [p for p in possessions if p.team_tricode == team]
            
            if len(team_possessions) >= 5:  # Need minimum possessions
                # Calculate features
                features = engine.calculate_possession_features(team_possessions)
                
                # Compute TMI with ML prediction
                tmi = engine.compute_tmi(team, "sample_game_000", features[:5])
                
                logger.info(f"Results for {team}:")
                logger.info(f"  TMI Value: {tmi.tmi_value:.3f}")
                logger.info(f"  Prediction Probability: {tmi.prediction_probability:.3f}")
                logger.info(f"  Confidence Score: {tmi.confidence_score:.3f}")
                logger.info(f"  Feature Contributions: {tmi.feature_contributions}")
        
        logger.info("Complete pipeline test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Complete pipeline test FAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)