#!/usr/bin/env python3
"""
Advanced model training script with real NBA data and improved methodology.

This script implements the recommendations from the model effectiveness analysis:
1. Collect real NBA data (when available)
2. Use enhanced feature engineering
3. Implement proper train/test splits
4. Use ensemble methods
5. Comprehensive evaluation
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from services.real_nba_data_collector import create_real_nba_collector
    from services.advanced_ml_predictor import train_advanced_momentum_model
    from services.historical_data_collector import create_sample_training_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_real_nba_data(
    days_back: int = 14,
    max_games: int = 50,
    season: str = "2023-24",
    model_path: str = "models/advanced_momentum_predictor.pkl"
) -> dict:
    """
    Train the advanced model using real NBA data.
    
    Args:
        days_back: Number of days back to collect data
        max_games: Maximum number of games to process
        season: NBA season to collect from
        model_path: Path to save the trained model
        
    Returns:
        Training results dictionary
    """
    logger.info(f"Training advanced model with real NBA data from last {days_back} days")
    
    try:
        # Create real NBA data collector
        collector = create_real_nba_collector()
        
        # Collect real NBA training data
        logger.info("Collecting real NBA training data...")
        historical_events = collector.collect_training_dataset(
            days_back=days_back,
            max_games=max_games,
            season=season
        )
        
        if not historical_events:
            logger.error("No real NBA data collected, falling back to enhanced sample data")
            return train_with_enhanced_sample_data(model_path)
        
        logger.info(f"Collected {len(historical_events)} events from real NBA games")
        
        # Train advanced model
        results = train_advanced_momentum_model(historical_events, model_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error training with real NBA data: {e}")
        logger.info("Falling back to enhanced sample data training")
        return train_with_enhanced_sample_data(model_path)


def train_with_enhanced_sample_data(
    model_path: str = "models/advanced_momentum_predictor.pkl"
) -> dict:
    """
    Train the model using enhanced sample data with realistic patterns.
    
    Args:
        model_path: Path to save the trained model
        
    Returns:
        Training results dictionary
    """
    logger.info("Training advanced model with enhanced sample data")
    
    try:
        # Create enhanced sample data with more realistic patterns
        sample_events = create_enhanced_sample_data()
        
        # Train advanced model
        results = train_advanced_momentum_model(sample_events, model_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error training with enhanced sample data: {e}")
        raise


def create_enhanced_sample_data():
    """Create enhanced sample data with more realistic basketball patterns."""
    import numpy as np
    from datetime import datetime
    from backend.models.game_models import GameEvent
    
    logger.info("Creating enhanced sample data with realistic basketball patterns")
    
    np.random.seed(42)  # For reproducibility
    sample_events = []
    
    # Create sample events for multiple games with realistic patterns
    teams = ['LAL', 'GSW', 'BOS', 'MIA', 'PHX', 'MIL', 'BKN', 'DEN']
    
    for game_num in range(30):  # 30 sample games
        # Pick two random teams
        home_team, away_team = np.random.choice(teams, 2, replace=False)
        game_id = f"enhanced_game_{game_num:03d}"
        
        # Game characteristics
        competitive_game = np.random.random() > 0.3  # 70% competitive games
        high_scoring = np.random.random() > 0.5  # 50% high-scoring games
        
        # Create events for each team with realistic patterns
        for team_idx, team in enumerate([home_team, away_team]):
            # Team strength (affects performance)
            team_strength = np.random.uniform(0.4, 0.9)
            
            # Create possessions with momentum patterns
            current_momentum = 0.0
            scoring_run = 0
            
            for possession in range(25):  # 25 possessions per team
                # Momentum affects performance
                momentum_boost = current_momentum * 0.2
                performance_level = team_strength + momentum_boost + np.random.normal(0, 0.1)
                performance_level = max(0.1, min(0.95, performance_level))
                
                # Create events for this possession
                possession_events = []
                
                # Shot attempt (most possessions end with a shot)
                if np.random.random() < 0.85:  # 85% of possessions have shots
                    shot_made = np.random.random() < performance_level
                    
                    # Shot type based on performance and momentum
                    if shot_made:
                        if np.random.random() < 0.3:  # 30% are 3-pointers
                            shot_desc = f"{team} makes 3pt shot"
                            points = 3
                        else:
                            shot_desc = f"{team} makes 2pt shot"
                            points = 2
                        
                        # Update scoring run
                        scoring_run += points
                        current_momentum = min(1.0, current_momentum + 0.3)
                        
                    else:
                        if np.random.random() < 0.3:
                            shot_desc = f"{team} misses 3pt shot"
                        else:
                            shot_desc = f"{team} misses 2pt shot"
                        points = 0
                        
                        # Reset scoring run on miss
                        if scoring_run > 0:
                            scoring_run = max(0, scoring_run - 1)
                        current_momentum = max(-0.5, current_momentum - 0.2)
                    
                    # Create shot event
                    shot_event = GameEvent(
                        event_id=f"{game_id}_{team}_{possession}_shot",
                        game_id=game_id,
                        team_tricode=team,
                        player_name=f"Player_{possession % 8}",
                        event_type='shot',
                        clock=f"{11 - (possession // 5)}:{(60 - possession * 2) % 60:02d}",
                        period=1 + (possession // 25),
                        points_total=possession * 2 + points,
                        shot_result='Made' if shot_made else 'Missed',
                        timestamp=datetime.utcnow(),
                        description=shot_desc
                    )
                    
                    # Add enhanced attributes
                    shot_event.event_value = points if shot_made else -1.0
                    shot_event.time_remaining = max(0, 48 - possession * 1.5)
                    shot_event.score_margin = abs(np.random.randint(0, 15))
                    
                    possession_events.append(shot_event)
                
                # Rebound (if shot missed)
                if possession_events and possession_events[-1].shot_result == 'Missed':
                    if np.random.random() < 0.7:  # 70% of misses have rebounds
                        rebound_type = "offensive" if np.random.random() < 0.3 else "defensive"
                        rebound_event = GameEvent(
                            event_id=f"{game_id}_{team}_{possession}_rebound",
                            game_id=game_id,
                            team_tricode=team,
                            player_name=f"Player_{(possession + 1) % 8}",
                            event_type='rebound',
                            clock=possession_events[-1].clock,
                            period=possession_events[-1].period,
                            points_total=possession_events[-1].points_total,
                            shot_result=None,
                            timestamp=datetime.utcnow(),
                            description=f"{team} {rebound_type} rebound"
                        )
                        
                        rebound_event.event_value = 1.5 if rebound_type == "offensive" else 0.5
                        rebound_event.time_remaining = possession_events[-1].time_remaining
                        rebound_event.score_margin = possession_events[-1].score_margin
                        
                        possession_events.append(rebound_event)
                
                # Assist (if shot made)
                if possession_events and possession_events[-1].shot_result == 'Made':
                    if np.random.random() < 0.6:  # 60% of made shots have assists
                        assist_event = GameEvent(
                            event_id=f"{game_id}_{team}_{possession}_assist",
                            game_id=game_id,
                            team_tricode=team,
                            player_name=f"Player_{(possession + 2) % 8}",
                            event_type='assist',
                            clock=possession_events[-1].clock,
                            period=possession_events[-1].period,
                            points_total=possession_events[-1].points_total,
                            shot_result=None,
                            timestamp=datetime.utcnow(),
                            description=f"{team} assist"
                        )
                        
                        assist_event.event_value = 1.0
                        assist_event.time_remaining = possession_events[-1].time_remaining
                        assist_event.score_margin = possession_events[-1].score_margin
                        
                        possession_events.append(assist_event)
                
                # Turnover (some possessions end in turnovers)
                if not possession_events and np.random.random() < 0.15:  # 15% turnover rate
                    turnover_event = GameEvent(
                        event_id=f"{game_id}_{team}_{possession}_turnover",
                        game_id=game_id,
                        team_tricode=team,
                        player_name=f"Player_{possession % 8}",
                        event_type='turnover',
                        clock=f"{11 - (possession // 5)}:{(60 - possession * 2) % 60:02d}",
                        period=1 + (possession // 25),
                        points_total=possession * 2,
                        shot_result=None,
                        timestamp=datetime.utcnow(),
                        description=f"{team} turnover"
                    )
                    
                    turnover_event.event_value = -2.0
                    turnover_event.time_remaining = max(0, 48 - possession * 1.5)
                    turnover_event.score_margin = abs(np.random.randint(0, 15))
                    
                    possession_events.append(turnover_event)
                    
                    # Turnovers hurt momentum
                    current_momentum = max(-1.0, current_momentum - 0.4)
                    scoring_run = 0
                
                # Steal (defensive play)
                if np.random.random() < 0.08:  # 8% steal rate
                    steal_event = GameEvent(
                        event_id=f"{game_id}_{team}_{possession}_steal",
                        game_id=game_id,
                        team_tricode=team,
                        player_name=f"Player_{(possession + 3) % 8}",
                        event_type='steal',
                        clock=f"{11 - (possession // 5)}:{(60 - possession * 2) % 60:02d}",
                        period=1 + (possession // 25),
                        points_total=possession * 2,
                        shot_result=None,
                        timestamp=datetime.utcnow(),
                        description=f"{team} steal"
                    )
                    
                    steal_event.event_value = 2.0
                    steal_event.time_remaining = max(0, 48 - possession * 1.5)
                    steal_event.score_margin = abs(np.random.randint(0, 15))
                    
                    possession_events.append(steal_event)
                    
                    # Steals boost momentum
                    current_momentum = min(1.0, current_momentum + 0.4)
                
                # Add all possession events to sample
                sample_events.extend(possession_events)
                
                # Momentum decay over time
                current_momentum *= 0.95
    
    logger.info(f"Created {len(sample_events)} enhanced sample events from 30 games")
    return sample_events


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train Advanced MomentumML prediction model")
    
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use enhanced sample data instead of real NBA data"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=14,
        help="Number of days back to collect real NBA data (default: 14)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=50,
        help="Maximum number of games to process (default: 50)"
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
        default="models/advanced_momentum_predictor.pkl",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Advanced MomentumML model training")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        if args.use_sample_data:
            results = train_with_enhanced_sample_data(args.model_path)
        else:
            results = train_with_real_nba_data(
                days_back=args.days_back,
                max_games=args.max_games,
                season=args.season,
                model_path=args.model_path
            )
        
        # Print comprehensive results
        logger.info("=" * 60)
        logger.info("ADVANCED MODEL TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        # Individual model results
        if 'individual_models' in results:
            logger.info("Individual Model Performance:")
            for model_name, metrics in results['individual_models'].items():
                logger.info(f"  {model_name}:")
                logger.info(f"    Validation Accuracy: {metrics['val_accuracy']:.4f}")
                logger.info(f"    Validation AUC: {metrics['val_auc']:.4f}")
                logger.info(f"    Validation F1: {metrics['val_f1']:.4f}")
        
        # Ensemble results
        if 'ensemble_results' in results:
            ensemble = results['ensemble_results']
            logger.info("Ensemble Model Performance:")
            logger.info(f"  Test Accuracy: {ensemble['accuracy']:.4f}")
            logger.info(f"  Test Precision: {ensemble['precision']:.4f}")
            logger.info(f"  Test Recall: {ensemble['recall']:.4f}")
            logger.info(f"  Test F1-Score: {ensemble['f1']:.4f}")
            logger.info(f"  Test AUC: {ensemble['auc']:.4f}")
        
        # Baseline comparison
        if 'baseline_results' in results:
            baseline = results['baseline_results']
            ensemble_acc = results['ensemble_results']['accuracy']
            improvement = ensemble_acc - baseline['most_frequent_accuracy']
            
            logger.info("Baseline Comparison:")
            logger.info(f"  Most Frequent Class: {baseline['most_frequent_accuracy']:.4f}")
            logger.info(f"  Random Prediction: {baseline['random_accuracy']:.4f}")
            logger.info(f"  Ensemble Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
        
        # Ensemble weights
        if 'ensemble_weights' in results:
            logger.info("Ensemble Model Weights:")
            for model_name, weight in results['ensemble_weights'].items():
                logger.info(f"  {model_name}: {weight:.4f}")
        
        # Data statistics
        logger.info("Training Data Statistics:")
        logger.info(f"  Training samples: {results.get('train_samples', 'N/A')}")
        logger.info(f"  Validation samples: {results.get('val_samples', 'N/A')}")
        logger.info(f"  Test samples: {results.get('test_samples', 'N/A')}")
        
        logger.info(f"Advanced model saved to: {args.model_path}")
        
        # Performance assessment
        if 'ensemble_results' in results:
            test_acc = results['ensemble_results']['accuracy']
            if test_acc > 0.7:
                logger.info("✅ EXCELLENT: Model shows strong performance!")
            elif test_acc > 0.6:
                logger.info("✅ GOOD: Model shows decent performance")
            elif test_acc > 0.55:
                logger.info("⚠️  MODERATE: Model shows some predictive ability")
            else:
                logger.info("❌ POOR: Model needs significant improvement")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Advanced training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()