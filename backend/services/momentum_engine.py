"""
Momentum calculation engine for MomentumML.

This module implements the core Team Momentum Index (TMI) calculation,
including possession segmentation, feature engineering, and z-score normalization.
"""

import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.models.game_models import GameEvent, Possession, TeamMomentumIndex
from backend.utils.error_handling import (
    DataProcessingError, MLModelError,
    safe_execute, validate_data, log_error,
    ErrorSeverity, graceful_degradation, health_checker
)

# Configure logging
logger = logging.getLogger("momentum_ml.momentum_engine")


@dataclass
class PossessionFeatures:
    """Features calculated for a single possession."""
    team_tricode: str
    possession_id: str
    points_scored: int
    fg_attempts: int
    fg_made: int
    fg_percentage: float
    turnovers: int
    rebounds: int
    fouls: int
    possession_duration: float  # seconds
    pace: float  # possessions per minute


@dataclass
class TeamFeatureStats:
    """Rolling statistics for team features used in z-score normalization."""
    fg_percentage_history: deque
    turnover_rate_history: deque
    rebound_rate_history: deque
    pace_history: deque
    foul_rate_history: deque


class MomentumEngine:
    """
    Core momentum calculation engine.
    
    Handles possession segmentation, feature engineering, TMI calculation,
    z-score normalization, and ML-based momentum prediction.
    """
    
    def __init__(
        self,
        rolling_window_size: int = 5,
        tmi_weights: Optional[Dict[str, float]] = None,
        normalization_window: int = 20,
        ml_predictor=None
    ):
        """
        Initialize the momentum engine.
        
        Args:
            rolling_window_size: Number of possessions for TMI calculation
            tmi_weights: Feature weights for TMI calculation
            normalization_window: Number of possessions for z-score calculation
            ml_predictor: Optional ML predictor for momentum continuation prediction
        """
        self.rolling_window_size = rolling_window_size
        self.normalization_window = normalization_window
        self.ml_predictor = ml_predictor
        
        # Default TMI weights: score, fg%, rebounds, turnovers, fouls
        self.tmi_weights = tmi_weights or {
            'points_scored': 0.4,
            'fg_percentage': 0.25,
            'rebounds': 0.15,
            'turnovers': -0.15,  # Negative because turnovers hurt momentum
            'fouls': -0.05       # Negative because fouls hurt momentum
        }
        
        # Rolling windows for each team
        self.team_possessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=rolling_window_size))
        self.team_feature_stats: Dict[str, TeamFeatureStats] = defaultdict(
            lambda: TeamFeatureStats(
                fg_percentage_history=deque(maxlen=normalization_window),
                turnover_rate_history=deque(maxlen=normalization_window),
                rebound_rate_history=deque(maxlen=normalization_window),
                pace_history=deque(maxlen=normalization_window),
                foul_rate_history=deque(maxlen=normalization_window)
            )
        )
        
        # Cache for latest TMI values and history
        self.latest_tmi: Dict[str, TeamMomentumIndex] = {}
        self.tmi_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    
    def segment_possessions(self, events: List[GameEvent]) -> List[Possession]:
        """
        Segment game events into team possessions.
        
        A possession ends when:
        - Team makes a shot (2 or 3 points)
        - Team commits a turnover
        - Opponent gets a defensive rebound
        - Period ends
        
        Args:
            events: List of game events in chronological order
            
        Returns:
            List of Possession objects
        """
        try:
            # Validate input
            validate_data(
                events,
                lambda e: isinstance(e, list) and len(e) > 0,
                "Events list is empty or invalid"
            )
            
            possessions = []
            current_possession = None
            possession_counter = 0
            
            # Sort events by period and time with error handling
            try:
                sorted_events = sorted(
                    events, 
                    key=lambda e: (
                        safe_execute(lambda: e.period, 0, log_errors=False),
                        safe_execute(lambda: self._time_to_seconds(e.clock), 0.0, log_errors=False)
                    )
                )
            except Exception as e:
                log_error(e, context={"events_count": len(events)}, severity=ErrorSeverity.MEDIUM)
                # Use unsorted events as fallback
                sorted_events = events
            
            for event in sorted_events:
                try:
                    # Validate individual event
                    if not hasattr(event, 'event_type') or not hasattr(event, 'team_tricode'):
                        logger.warning(f"Skipping invalid event: {event}")
                        continue
                    
                    # Skip non-possession events
                    if event.event_type in ['substitution', 'timeout', 'period_start']:
                        continue
                    
                    # Start new possession if needed
                    if current_possession is None or self._should_end_possession(event, current_possession):
                        # End current possession if exists
                        if current_possession is not None:
                            try:
                                possessions.append(self._finalize_possession(current_possession))
                            except Exception as e:
                                log_error(
                                    e,
                                    context={"possession_id": current_possession.get('possession_id', 'unknown')},
                                    severity=ErrorSeverity.LOW
                                )
                                continue
                        
                        # Start new possession
                        possession_counter += 1
                        current_possession = {
                            'possession_id': f"{event.game_id}_P{possession_counter}",
                            'game_id': event.game_id,
                            'team_tricode': event.team_tricode,
                            'start_time': event.clock,
                            'start_period': event.period,
                            'events': [],
                            'points_scored': 0,
                            'fg_attempts': 0,
                            'fg_made': 0,
                            'turnovers': 0,
                            'rebounds': 0,
                            'fouls': 0
                        }
                    
                    # Add event to current possession
                    current_possession['events'].append(event)
                    
                    # Update possession stats with error handling
                    try:
                        self._update_possession_stats(current_possession, event)
                    except Exception as e:
                        log_error(
                            e,
                            context={
                                "event_id": getattr(event, 'event_id', 'unknown'),
                                "event_type": getattr(event, 'event_type', 'unknown')
                            },
                            severity=ErrorSeverity.LOW
                        )
                        continue
                        
                except Exception as e:
                    log_error(
                        e,
                        context={"event": str(event)[:200]},
                        severity=ErrorSeverity.LOW
                    )
                    continue
            
            # Finalize last possession
            if current_possession is not None:
                try:
                    possessions.append(self._finalize_possession(current_possession))
                except Exception as e:
                    log_error(
                        e,
                        context={"possession_id": current_possession.get('possession_id', 'unknown')},
                        severity=ErrorSeverity.LOW
                    )
            
            logger.info(f"Segmented {len(events)} events into {len(possessions)} possessions")
            return possessions
            
        except DataProcessingError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise DataProcessingError(
                f"Failed to segment possessions: {str(e)}",
                original_error=e,
                details={"events_count": len(events) if events else 0}
            )
    
    def calculate_possession_features(self, possessions: List[Possession]) -> List[PossessionFeatures]:
        """
        Calculate features for each possession.
        
        Args:
            possessions: List of Possession objects
            
        Returns:
            List of PossessionFeatures objects
        """
        features = []
        
        for possession in possessions:
            # Calculate basic percentages
            fg_percentage = (possession.fg_made / possession.fg_attempts) if possession.fg_attempts > 0 else 0.0
            
            # Calculate possession duration (approximate)
            duration = self._calculate_possession_duration(possession)
            
            # Calculate pace (possessions per minute)
            pace = 60.0 / duration if duration > 0 else 0.0
            
            feature = PossessionFeatures(
                team_tricode=possession.team_tricode,
                possession_id=possession.possession_id,
                points_scored=possession.points_scored,
                fg_attempts=possession.fg_attempts,
                fg_made=possession.fg_made,
                fg_percentage=fg_percentage,
                turnovers=possession.turnovers,
                rebounds=possession.rebounds,
                fouls=possession.fouls,
                possession_duration=duration,
                pace=pace
            )
            
            features.append(feature)
        
        return features
    
    def compute_tmi(
        self,
        team_tricode: str,
        game_id: str,
        new_features: List[PossessionFeatures]
    ) -> TeamMomentumIndex:
        """
        Compute Team Momentum Index for a team.
        
        Args:
            team_tricode: Team identifier
            game_id: Game identifier
            new_features: New possession features to add to rolling window
            
        Returns:
            TeamMomentumIndex object with calculated TMI and feature contributions
        """
        try:
            # Validate inputs
            validate_data(
                team_tricode,
                lambda t: isinstance(t, str) and len(t) > 0,
                "Invalid team tricode"
            )
            
            validate_data(
                game_id,
                lambda g: isinstance(g, str) and len(g) > 0,
                "Invalid game ID"
            )
            
            validate_data(
                new_features,
                lambda f: isinstance(f, list),
                "Invalid features list"
            )
            
            # Add new features to rolling window
            team_key = f"{game_id}_{team_tricode}"
            
            for feature in new_features:
                try:
                    if hasattr(feature, 'team_tricode') and feature.team_tricode == team_tricode:
                        self.team_possessions[team_key].append(feature)
                        self._update_feature_stats(team_key, feature)
                except Exception as e:
                    log_error(
                        e,
                        context={
                            "team_key": team_key,
                            "feature": str(feature)[:200]
                        },
                        severity=ErrorSeverity.LOW
                    )
                    continue
            
            # Get current rolling window
            current_window = list(self.team_possessions[team_key])
            
            if len(current_window) == 0:
                # Return neutral TMI if no data
                logger.warning(f"No data available for TMI calculation: {team_key}")
                return TeamMomentumIndex(
                    game_id=game_id,
                    team_tricode=team_tricode,
                    timestamp=datetime.utcnow(),
                    tmi_value=0.0,
                    feature_contributions={},
                    rolling_window_size=self.rolling_window_size,
                    prediction_probability=0.5,
                    confidence_score=0.0
                )
            
            # Calculate normalized features with error handling
            try:
                normalized_features = self._normalize_features(team_key, current_window)
            except Exception as e:
                log_error(
                    e,
                    context={"team_key": team_key, "window_size": len(current_window)},
                    severity=ErrorSeverity.MEDIUM
                )
                # Use default normalization as fallback
                normalized_features = {key: 0.0 for key in self.tmi_weights.keys()}
            
            # Calculate weighted TMI
            tmi_value = 0.0
            feature_contributions = {}
            
            for feature_name, weight in self.tmi_weights.items():
                if feature_name in normalized_features:
                    contribution = safe_execute(
                        lambda: weight * normalized_features[feature_name],
                        0.0,
                        f"Failed to calculate contribution for {feature_name}"
                    )
                    tmi_value += contribution
                    feature_contributions[feature_name] = contribution
            
            # Calculate confidence score based on sample size
            confidence_score = safe_execute(
                lambda: min(len(current_window) / self.rolling_window_size, 1.0),
                0.0,
                "Failed to calculate confidence score"
            )
            
            # Get ML prediction if predictor is available
            prediction_probability = 0.5
            if self.ml_predictor:
                try:
                    team_history = list(self.tmi_history[team_key])
                    prediction_probability, ml_confidence = self.ml_predictor.predict_momentum_continuation(
                        team_history, current_window
                    )
                    # Combine confidence scores
                    confidence_score = safe_execute(
                        lambda: (confidence_score + ml_confidence) / 2,
                        confidence_score,
                        "Failed to combine confidence scores"
                    )
                except Exception as e:
                    log_error(
                        e,
                        context={"team_key": team_key, "ml_predictor": str(type(self.ml_predictor))},
                        severity=ErrorSeverity.LOW
                    )
                    # ML prediction failed, continue with default probability
            
            # Create TMI result
            tmi_result = TeamMomentumIndex(
                game_id=game_id,
                team_tricode=team_tricode,
                timestamp=datetime.utcnow(),
                tmi_value=tmi_value,
                feature_contributions=feature_contributions,
                rolling_window_size=len(current_window),
                prediction_probability=prediction_probability,
                confidence_score=confidence_score
            )
            
            # Cache latest TMI and add to history
            self.latest_tmi[team_key] = tmi_result
            self.tmi_history[team_key].append(tmi_result)
            
            logger.info(f"Calculated TMI for {team_tricode}: {tmi_value:.3f} (prediction: {prediction_probability:.3f})")
            return tmi_result
            
        except DataProcessingError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise DataProcessingError(
                f"Failed to compute TMI for {team_tricode} in game {game_id}",
                original_error=e,
                details={
                    "team_tricode": team_tricode,
                    "game_id": game_id,
                    "features_count": len(new_features) if new_features else 0
                }
            )
    
    def update_rolling_window(self, game_id: str, new_possessions: List[Possession]) -> Dict[str, TeamMomentumIndex]:
        """
        Update rolling windows with new possessions and recalculate TMI.
        
        Args:
            game_id: Game identifier
            new_possessions: New possessions to process
            
        Returns:
            Dictionary mapping team_tricode to updated TMI
        """
        # Calculate features for new possessions
        new_features = self.calculate_possession_features(new_possessions)
        
        # Group features by team
        team_features = defaultdict(list)
        for feature in new_features:
            team_features[feature.team_tricode].append(feature)
        
        # Update TMI for each team
        updated_tmi = {}
        for team_tricode, features in team_features.items():
            tmi = self.compute_tmi(team_tricode, game_id, features)
            updated_tmi[team_tricode] = tmi
        
        return updated_tmi
    
    def get_latest_tmi(self, game_id: str, team_tricode: str) -> Optional[TeamMomentumIndex]:
        """Get the latest TMI for a team."""
        team_key = f"{game_id}_{team_tricode}"
        return self.latest_tmi.get(team_key)
    
    def _should_end_possession(self, event: GameEvent, current_possession: dict) -> bool:
        """Determine if current event should end the current possession."""
        # Different team = new possession
        if event.team_tricode != current_possession['team_tricode']:
            return True
        
        # Made shot ends possession
        if event.event_type == 'shot' and event.shot_result == 'Made':
            return True
        
        # Turnover ends possession
        if event.event_type == 'turnover':
            return True
        
        # Period end
        if event.event_type == 'period_end':
            return True
        
        return False
    
    def _update_possession_stats(self, possession: dict, event: GameEvent):
        """Update possession statistics with new event."""
        if event.event_type == 'shot':
            possession['fg_attempts'] += 1
            if event.shot_result == 'Made':
                possession['fg_made'] += 1
                # Estimate points (simplified - would need shot type info)
                possession['points_scored'] += 2  # Assume 2-pointer for now
        
        elif event.event_type == 'turnover':
            possession['turnovers'] += 1
        
        elif event.event_type == 'rebound':
            possession['rebounds'] += 1
        
        elif event.event_type == 'foul':
            possession['fouls'] += 1
    
    def _finalize_possession(self, possession_data: dict) -> Possession:
        """Convert possession data dict to Possession object."""
        return Possession(
            possession_id=possession_data['possession_id'],
            game_id=possession_data['game_id'],
            team_tricode=possession_data['team_tricode'],
            start_time=possession_data['start_time'],
            end_time=possession_data['events'][-1].clock if possession_data['events'] else possession_data['start_time'],
            events=possession_data['events'],
            points_scored=possession_data['points_scored'],
            fg_attempts=possession_data['fg_attempts'],
            fg_made=possession_data['fg_made'],
            turnovers=possession_data['turnovers'],
            rebounds=possession_data['rebounds'],
            fouls=possession_data['fouls']
        )
    
    def _calculate_possession_duration(self, possession: Possession) -> float:
        """Calculate possession duration in seconds (approximate)."""
        if not possession.events:
            return 30.0  # Default possession duration
        
        start_seconds = self._time_to_seconds(possession.start_time)
        end_seconds = self._time_to_seconds(possession.end_time)
        
        # Handle period transitions
        if start_seconds >= end_seconds:
            return 30.0  # Default for period transitions
        
        return start_seconds - end_seconds  # Clock counts down
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS time string to seconds."""
        try:
            if ':' in time_str:
                minutes, seconds = time_str.split(':')
                return float(minutes) * 60 + float(seconds)
            else:
                return float(time_str)
        except (ValueError, AttributeError):
            return 0.0
    
    def _update_feature_stats(self, team_key: str, feature: PossessionFeatures):
        """Update rolling feature statistics for z-score normalization."""
        stats = self.team_feature_stats[team_key]
        
        stats.fg_percentage_history.append(feature.fg_percentage)
        stats.turnover_rate_history.append(feature.turnovers)
        stats.rebound_rate_history.append(feature.rebounds)
        stats.pace_history.append(feature.pace)
        stats.foul_rate_history.append(feature.fouls)
    
    def _normalize_features(self, team_key: str, current_window: List[PossessionFeatures]) -> Dict[str, float]:
        """
        Normalize features using z-scores based on rolling history.
        
        Args:
            team_key: Team identifier
            current_window: Current rolling window of features
            
        Returns:
            Dictionary of normalized feature values
        """
        stats = self.team_feature_stats[team_key]
        normalized = {}
        
        # Calculate current window averages
        if not current_window:
            return normalized
        
        current_fg_pct = statistics.mean([f.fg_percentage for f in current_window])
        current_turnovers = statistics.mean([f.turnovers for f in current_window])
        current_rebounds = statistics.mean([f.rebounds for f in current_window])
        current_pace = statistics.mean([f.pace for f in current_window])
        current_fouls = statistics.mean([f.fouls for f in current_window])
        current_points = statistics.mean([f.points_scored for f in current_window])
        
        # Normalize using z-scores if we have enough history
        normalized['points_scored'] = self._calculate_z_score(
            current_points, 
            [f.points_scored for f in current_window]
        )
        
        normalized['fg_percentage'] = self._calculate_z_score(
            current_fg_pct,
            list(stats.fg_percentage_history)
        )
        
        normalized['turnovers'] = self._calculate_z_score(
            current_turnovers,
            list(stats.turnover_rate_history)
        )
        
        normalized['rebounds'] = self._calculate_z_score(
            current_rebounds,
            list(stats.rebound_rate_history)
        )
        
        normalized['fouls'] = self._calculate_z_score(
            current_fouls,
            list(stats.foul_rate_history)
        )
        
        return normalized
    
    def _calculate_z_score(self, value: float, history: List[float]) -> float:
        """
        Calculate z-score for a value given its history.
        
        Args:
            value: Current value
            history: Historical values for normalization
            
        Returns:
            Z-score normalized value
        """
        if len(history) < 2:
            return 0.0  # Not enough data for normalization
        
        try:
            mean = statistics.mean(history)
            stdev = statistics.stdev(history)
            
            if stdev == 0:
                return 0.0  # No variation in data
            
            return (value - mean) / stdev
        
        except statistics.StatisticsError:
            return 0.0


# Health check for momentum engine
def _check_momentum_engine_health() -> bool:
    """Check if momentum engine can perform basic calculations."""
    try:
        # Create a test engine
        engine = MomentumEngine(rolling_window_size=2)
        
        # Test with minimal data
        test_events = [
            GameEvent(
                event_id="test_1",
                game_id="test_game",
                team_tricode="TEST",
                player_name="Test Player",
                event_type="shot",
                clock="12:00",
                period=1,
                points_total=0,
                shot_result="Made",
                timestamp=datetime.utcnow(),
                description="Test shot"
            )
        ]
        
        # Test possession segmentation
        possessions = engine.segment_possessions(test_events)
        
        # Test feature calculation
        features = engine.calculate_possession_features(possessions)
        
        # Test TMI computation
        if features:
            tmi = engine.compute_tmi("TEST", "test_game", features)
            return tmi is not None
        
        return True
    except Exception:
        return False


# Register health check
health_checker.register_health_check("momentum_engine", _check_momentum_engine_health)


# Utility functions for external use
def create_momentum_engine(
    rolling_window_size: int = 5,
    tmi_weights: Optional[Dict[str, float]] = None,
    enable_ml_prediction: bool = True
) -> MomentumEngine:
    """Create a configured momentum engine instance."""
    ml_predictor = None
    
    if enable_ml_prediction:
        try:
            # Import here to avoid circular imports
            from backend.services.ml_predictor import MomentumPredictor
            ml_predictor = MomentumPredictor()
            logger.info("ML predictor loaded successfully")
            graceful_degradation.set_service_status("ml_predictor", True)
        except Exception as e:
            log_error(e, context={"component": "ml_predictor"}, severity=ErrorSeverity.MEDIUM)
            graceful_degradation.set_service_status("ml_predictor", False)
    
    try:
        engine = MomentumEngine(
            rolling_window_size=rolling_window_size,
            tmi_weights=tmi_weights,
            ml_predictor=ml_predictor
        )
        graceful_degradation.set_service_status("momentum_engine", True)
        return engine
    except Exception as e:
        graceful_degradation.set_service_status("momentum_engine", False)
        raise DataProcessingError(
            "Failed to create momentum engine",
            original_error=e,
            severity=ErrorSeverity.CRITICAL
        )


def calculate_team_momentum(
    events: List[GameEvent],
    team_tricode: str,
    game_id: str,
    engine: Optional[MomentumEngine] = None
) -> TeamMomentumIndex:
    """
    Convenience function to calculate momentum for a team.
    
    Args:
        events: List of game events
        team_tricode: Team identifier
        game_id: Game identifier
        engine: Optional pre-configured momentum engine
        
    Returns:
        TeamMomentumIndex with calculated momentum
    """
    if engine is None:
        engine = create_momentum_engine()
    
    # Segment possessions
    possessions = engine.segment_possessions(events)
    
    # Filter possessions for the team
    team_possessions = [p for p in possessions if p.team_tricode == team_tricode]
    
    # Calculate features
    features = engine.calculate_possession_features(team_possessions)
    
    # Compute TMI
    return engine.compute_tmi(team_tricode, game_id, features)