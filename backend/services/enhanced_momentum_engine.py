"""
Enhanced momentum engine with improved feature engineering and momentum definitions.

This module implements sophisticated momentum calculations based on basketball
domain expertise and advanced statistical methods.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from backend.models.game_models import GameEvent, Possession, TeamMomentumIndex
from backend.services.momentum_engine import MomentumEngine, PossessionFeatures
from backend.services.enhanced_momentum_predictor import get_enhanced_momentum_analysis

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPossessionFeatures:
    """Enhanced possession features with basketball-specific metrics."""
    
    # Basic features (from original)
    points_scored: float
    fg_percentage: float
    turnovers: float
    rebounds: float
    fouls: float
    pace: float
    
    # Enhanced basketball features
    effective_fg_percentage: float  # Accounts for 3-pointers
    true_shooting_percentage: float  # Accounts for free throws
    assist_to_turnover_ratio: float
    offensive_rating: float  # Points per 100 possessions
    defensive_rating: float  # Opponent points per 100 possessions
    
    # Situational features
    time_remaining: float
    score_margin: float
    period: int
    is_clutch_time: bool  # Last 5 minutes, within 5 points
    
    # Momentum-specific features
    scoring_run_length: int  # Current scoring run
    defensive_stops: int  # Consecutive stops
    momentum_events: float  # Weighted sum of momentum-changing events
    energy_level: float  # Composite energy metric
    
    # Advanced metrics
    player_impact: float  # Impact of players on court
    pace_differential: float  # Pace vs season average
    shot_quality: float  # Quality of shots taken
    defensive_pressure: float  # Pressure applied on defense


@dataclass
class MomentumShift:
    """Represents a momentum shift event."""
    
    timestamp: float
    team: str
    shift_magnitude: float  # -1.0 to 1.0
    shift_type: str  # 'scoring_run', 'defensive_stop', 'key_play'
    duration: float  # How long the shift lasted
    context: Dict[str, Any]


class EnhancedMomentumEngine:
    """
    Enhanced momentum engine with sophisticated basketball analytics.
    
    Improvements over original:
    - Better momentum definitions based on basketball expertise
    - Advanced feature engineering
    - Situational awareness (game context)
    - Multiple momentum metrics
    """
    
    def __init__(
        self,
        rolling_window_size: int = 8,
        momentum_decay_factor: float = 0.9,
        clutch_time_threshold: float = 5.0,
        scoring_run_threshold: int = 6
    ):
        """
        Initialize enhanced momentum engine.
        
        Args:
            rolling_window_size: Number of possessions for rolling calculations
            momentum_decay_factor: How quickly momentum decays over time
            clutch_time_threshold: Minutes remaining to be considered clutch time
            scoring_run_threshold: Points needed to constitute a scoring run
        """
        self.rolling_window_size = rolling_window_size
        self.momentum_decay_factor = momentum_decay_factor
        self.clutch_time_threshold = clutch_time_threshold
        self.scoring_run_threshold = scoring_run_threshold
        
        # Momentum tracking
        self.momentum_history: Dict[str, deque] = {}
        self.scoring_runs: Dict[str, List[Dict]] = {}
        self.defensive_stops: Dict[str, int] = {}
        
        # Advanced metrics cache
        self.team_season_averages: Dict[str, Dict] = {}
        self.player_impact_cache: Dict[str, float] = {}
    
    def calculate_enhanced_possession_features(
        self,
        possessions: List[Possession],
        game_context: Dict[str, Any] = None
    ) -> List[EnhancedPossessionFeatures]:
        """
        Calculate enhanced features for each possession.
        
        Args:
            possessions: List of possessions
            game_context: Additional game context information
            
        Returns:
            List of enhanced possession features
        """
        if not possessions:
            return []
        
        logger.info(f"Calculating enhanced features for {len(possessions)} possessions")
        
        enhanced_features = []
        team_stats = self._initialize_team_stats()
        
        for i, possession in enumerate(possessions):
            try:
                # Calculate basic features
                basic_features = self._calculate_basic_features(possession)
                
                # Calculate advanced basketball metrics
                advanced_metrics = self._calculate_advanced_metrics(
                    possession, possessions[:i], game_context
                )
                
                # Calculate situational features
                situational_features = self._calculate_situational_features(
                    possession, game_context
                )
                
                # Calculate momentum-specific features
                momentum_features = self._calculate_momentum_features(
                    possession, possessions[:i], team_stats
                )
                
                # Combine all features
                enhanced_feature = EnhancedPossessionFeatures(
                    **basic_features,
                    **advanced_metrics,
                    **situational_features,
                    **momentum_features
                )
                
                enhanced_features.append(enhanced_feature)
                
                # Update team stats for next iteration
                self._update_team_stats(team_stats, possession, enhanced_feature)
                
            except Exception as e:
                logger.warning(f"Error calculating features for possession {i}: {e}")
                # Add default features to maintain list length
                enhanced_features.append(self._get_default_features())
        
        logger.info(f"Calculated enhanced features for {len(enhanced_features)} possessions")
        return enhanced_features
    
    def _calculate_basic_features(self, possession: Possession) -> Dict[str, float]:
        """Calculate basic possession features."""
        events = possession.events
        
        # Points scored
        points_scored = sum(
            self._get_event_points(event) for event in events
            if event.event_type == 'shot' and event.shot_result == 'Made'
        )
        
        # Shooting metrics
        shots = [e for e in events if e.event_type == 'shot']
        made_shots = [e for e in shots if e.shot_result == 'Made']
        fg_percentage = len(made_shots) / len(shots) if shots else 0.0
        
        # Other basic stats
        turnovers = len([e for e in events if e.event_type == 'turnover'])
        rebounds = len([e for e in events if e.event_type == 'rebound'])
        fouls = len([e for e in events if e.event_type == 'foul'])
        
        # Pace (possessions per minute estimate)
        possession_duration = self._estimate_possession_duration(possession)
        pace = 1.0 / max(possession_duration, 0.1)  # Avoid division by zero
        
        return {
            'points_scored': points_scored,
            'fg_percentage': fg_percentage,
            'turnovers': turnovers,
            'rebounds': rebounds,
            'fouls': fouls,
            'pace': pace
        }
    
    def _calculate_advanced_metrics(
        self,
        possession: Possession,
        previous_possessions: List[Possession],
        game_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate advanced basketball metrics."""
        events = possession.events
        
        # Effective FG% (accounts for 3-pointers)
        shots = [e for e in events if e.event_type == 'shot']
        if shots:
            total_fga = len(shots)
            made_shots = [e for e in shots if e.shot_result == 'Made']
            three_pointers_made = len([
                e for e in made_shots 
                if '3pt' in e.description.lower()
            ])
            effective_fg_pct = (len(made_shots) + 0.5 * three_pointers_made) / total_fga
        else:
            effective_fg_pct = 0.0
        
        # True Shooting % (includes free throws)
        free_throws = [e for e in events if 'free throw' in e.description.lower()]
        fta = len(free_throws)
        points = sum(self._get_event_points(e) for e in events if e.shot_result == 'Made')
        
        if shots or free_throws:
            true_shooting_attempts = len(shots) + 0.44 * fta
            true_shooting_pct = points / (2 * true_shooting_attempts) if true_shooting_attempts > 0 else 0.0
        else:
            true_shooting_pct = 0.0
        
        # Assist to turnover ratio
        assists = len([e for e in events if e.event_type == 'assist'])
        turnovers = len([e for e in events if e.event_type == 'turnover'])
        ast_to_ratio = assists / max(turnovers, 1)
        
        # Offensive/Defensive rating (simplified)
        offensive_rating = points * 100  # Points per 100 possessions (simplified)
        defensive_rating = self._estimate_defensive_rating(possession, previous_possessions)
        
        return {
            'effective_fg_percentage': effective_fg_pct,
            'true_shooting_percentage': true_shooting_pct,
            'assist_to_turnover_ratio': ast_to_ratio,
            'offensive_rating': offensive_rating,
            'defensive_rating': defensive_rating
        }
    
    def _calculate_situational_features(
        self,
        possession: Possession,
        game_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate situational context features."""
        if not possession.events:
            return self._get_default_situational_features()
        
        first_event = possession.events[0]
        
        # Time remaining
        time_remaining = getattr(first_event, 'time_remaining', 0.0)
        
        # Score margin
        score_margin = getattr(first_event, 'score_margin', 0)
        
        # Period
        period = first_event.period
        
        # Clutch time (last 5 minutes, within 5 points)
        is_clutch_time = time_remaining <= self.clutch_time_threshold and score_margin <= 5
        
        return {
            'time_remaining': time_remaining,
            'score_margin': score_margin,
            'period': period,
            'is_clutch_time': float(is_clutch_time)
        }
    
    def _calculate_momentum_features(
        self,
        possession: Possession,
        previous_possessions: List[Possession],
        team_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate momentum-specific features."""
        team = possession.team_tricode
        
        # Scoring run length
        scoring_run_length = self._calculate_scoring_run_length(
            possession, previous_possessions
        )
        
        # Defensive stops
        defensive_stops = self._calculate_defensive_stops(
            possession, previous_possessions
        )
        
        # Momentum events (weighted sum of momentum-changing plays)
        momentum_events = self._calculate_momentum_events_score(possession)
        
        # Energy level (composite metric)
        energy_level = self._calculate_energy_level(
            possession, previous_possessions
        )
        
        # Player impact (simplified)
        player_impact = self._calculate_player_impact(possession)
        
        # Pace differential
        pace_differential = self._calculate_pace_differential(
            possession, team_stats.get(team, {})
        )
        
        # Shot quality
        shot_quality = self._calculate_shot_quality(possession)
        
        # Defensive pressure
        defensive_pressure = self._calculate_defensive_pressure(possession)
        
        return {
            'scoring_run_length': scoring_run_length,
            'defensive_stops': defensive_stops,
            'momentum_events': momentum_events,
            'energy_level': energy_level,
            'player_impact': player_impact,
            'pace_differential': pace_differential,
            'shot_quality': shot_quality,
            'defensive_pressure': defensive_pressure
        }
    
    def _calculate_scoring_run_length(
        self,
        possession: Possession,
        previous_possessions: List[Possession]
    ) -> int:
        """Calculate current scoring run length."""
        team = possession.team_tricode
        run_length = 0
        
        # Look back through recent possessions
        recent_possessions = previous_possessions[-10:]  # Last 10 possessions
        
        for prev_poss in reversed(recent_possessions):
            if prev_poss.team_tricode == team:
                points = sum(
                    self._get_event_points(event) for event in prev_poss.events
                    if event.event_type == 'shot' and event.shot_result == 'Made'
                )
                if points > 0:
                    run_length += points
                else:
                    break
            else:
                # Opponent possession - check if they scored
                opp_points = sum(
                    self._get_event_points(event) for event in prev_poss.events
                    if event.event_type == 'shot' and event.shot_result == 'Made'
                )
                if opp_points > 0:
                    break
        
        return run_length
    
    def _calculate_defensive_stops(
        self,
        possession: Possession,
        previous_possessions: List[Possession]
    ) -> int:
        """Calculate consecutive defensive stops."""
        team = possession.team_tricode
        stops = 0
        
        # Look at opponent possessions
        for prev_poss in reversed(previous_possessions[-5:]):
            if prev_poss.team_tricode != team:  # Opponent possession
                # Check if they scored
                points = sum(
                    self._get_event_points(event) for event in prev_poss.events
                    if event.event_type == 'shot' and event.shot_result == 'Made'
                )
                if points == 0:
                    stops += 1
                else:
                    break
        
        return stops
    
    def _calculate_momentum_events_score(self, possession: Possession) -> float:
        """Calculate weighted momentum events score."""
        momentum_score = 0.0
        
        for event in possession.events:
            # Get event value if available
            event_value = getattr(event, 'event_value', 0.0)
            
            # Apply momentum weights
            if event.event_type == 'steal':
                momentum_score += 3.0
            elif event.event_type == 'block':
                momentum_score += 2.5
            elif event.event_type == 'shot' and event.shot_result == 'Made':
                if '3pt' in event.description.lower():
                    momentum_score += 2.0
                else:
                    momentum_score += 1.0
            elif event.event_type == 'turnover':
                momentum_score -= 2.0
            elif event.event_type == 'foul':
                momentum_score -= 0.5
            
            # Add base event value
            momentum_score += event_value * 0.5
        
        return momentum_score
    
    def _calculate_energy_level(
        self,
        possession: Possession,
        previous_possessions: List[Possession]
    ) -> float:
        """Calculate team energy level (composite metric)."""
        # Factors that contribute to energy:
        # - Fast break opportunities
        # - Defensive intensity (steals, blocks)
        # - Crowd-pleasing plays
        # - Pace of play
        
        energy = 0.0
        
        # Current possession energy
        for event in possession.events:
            if event.event_type in ['steal', 'block']:
                energy += 2.0
            elif event.event_type == 'shot' and event.shot_result == 'Made':
                if '3pt' in event.description.lower():
                    energy += 1.5
                elif 'dunk' in event.description.lower():
                    energy += 2.0
                else:
                    energy += 0.5
        
        # Recent momentum (last 3 possessions)
        recent_energy = 0.0
        for prev_poss in previous_possessions[-3:]:
            if prev_poss.team_tricode == possession.team_tricode:
                for event in prev_poss.events:
                    if event.event_type in ['steal', 'block']:
                        recent_energy += 1.0
                    elif event.event_type == 'shot' and event.shot_result == 'Made':
                        recent_energy += 0.3
        
        # Decay recent energy
        energy += recent_energy * 0.7
        
        return max(0.0, min(10.0, energy))  # Clamp between 0 and 10
    
    def _calculate_player_impact(self, possession: Possession) -> float:
        """Calculate player impact score (simplified)."""
        # In a real implementation, this would use player ratings
        # For now, use a simplified approach based on events
        
        impact = 0.0
        unique_players = set()
        
        for event in possession.events:
            if event.player_name:
                unique_players.add(event.player_name)
                
                # Simple impact scoring
                if event.event_type == 'shot' and event.shot_result == 'Made':
                    impact += 1.0
                elif event.event_type in ['assist', 'steal', 'block']:
                    impact += 1.5
                elif event.event_type == 'turnover':
                    impact -= 1.0
        
        # Normalize by number of players involved
        return impact / max(len(unique_players), 1)
    
    def _calculate_pace_differential(
        self,
        possession: Possession,
        team_stats: Dict[str, Any]
    ) -> float:
        """Calculate pace differential vs team average."""
        # Simplified pace calculation
        possession_duration = self._estimate_possession_duration(possession)
        current_pace = 1.0 / max(possession_duration, 0.1)
        
        # Use team average pace (or league average if not available)
        team_avg_pace = team_stats.get('avg_pace', 1.0)
        
        return current_pace - team_avg_pace
    
    def _calculate_shot_quality(self, possession: Possession) -> float:
        """Calculate quality of shots taken."""
        shots = [e for e in possession.events if e.event_type == 'shot']
        
        if not shots:
            return 0.0
        
        quality_score = 0.0
        
        for shot in shots:
            # Factors that indicate shot quality:
            # - Assisted shots are typically better
            # - Open shots vs contested
            # - Shot location
            
            base_quality = 0.5  # Neutral quality
            
            # Check if shot was assisted (look for assist in nearby events)
            assisted = any(
                e.event_type == 'assist' for e in possession.events
            )
            if assisted:
                base_quality += 0.3
            
            # 3-point shots have different quality considerations
            if '3pt' in shot.description.lower():
                base_quality += 0.1  # 3-pointers are valuable when made
            
            # Made shots indicate good quality
            if shot.shot_result == 'Made':
                base_quality += 0.2
            
            quality_score += base_quality
        
        return quality_score / len(shots)
    
    def _calculate_defensive_pressure(self, possession: Possession) -> float:
        """Calculate defensive pressure applied."""
        pressure = 0.0
        
        # Indicators of defensive pressure:
        # - Steals and blocks
        # - Forced turnovers
        # - Contested shots (missed shots can indicate pressure)
        
        for event in possession.events:
            if event.event_type == 'steal':
                pressure += 3.0
            elif event.event_type == 'block':
                pressure += 2.5
            elif event.event_type == 'turnover':
                pressure += 1.5
            elif event.event_type == 'shot' and event.shot_result == 'Missed':
                pressure += 0.5  # Might indicate defensive pressure
        
        return pressure
    
    def compute_enhanced_tmi(
        self,
        team: str,
        game_id: str,
        enhanced_features: List[EnhancedPossessionFeatures],
        ml_predictor=None
    ) -> TeamMomentumIndex:
        """
        Compute enhanced Team Momentum Index.
        
        Args:
            team: Team tricode
            game_id: Game ID
            enhanced_features: List of enhanced possession features
            ml_predictor: Optional ML predictor for momentum continuation
            
        Returns:
            Enhanced TeamMomentumIndex
        """
        if not enhanced_features:
            return self._get_default_tmi(team, game_id)
        
        # Calculate multiple momentum components
        scoring_momentum = self._calculate_scoring_momentum(enhanced_features)
        defensive_momentum = self._calculate_defensive_momentum(enhanced_features)
        situational_momentum = self._calculate_situational_momentum(enhanced_features)
        energy_momentum = self._calculate_energy_momentum(enhanced_features)
        
        # Weighted combination
        base_tmi = (
            scoring_momentum * 0.35 +
            defensive_momentum * 0.25 +
            situational_momentum * 0.20 +
            energy_momentum * 0.20
        )
        
        # Apply situational modifiers
        latest_features = enhanced_features[-1]
        
        # Clutch time modifier
        if latest_features.is_clutch_time:
            base_tmi *= 1.2
        
        # Scoring run modifier
        if latest_features.scoring_run_length >= self.scoring_run_threshold:
            base_tmi *= 1.15
        
        # Defensive stops modifier
        if latest_features.defensive_stops >= 3:
            base_tmi *= 1.1
        
        # Clamp TMI to reasonable range
        final_tmi = max(-1.0, min(1.0, base_tmi))
        
        # Get ML prediction if available
        prediction_prob = 0.5
        confidence = 0.0
        
        if ml_predictor and hasattr(ml_predictor, 'predict_momentum_continuation'):
            try:
                prediction_prob, confidence = ml_predictor.predict_momentum_continuation(
                    [], enhanced_features  # Pass enhanced features
                )
            except Exception as e:
                logger.warning(f"Error getting ML prediction: {e}")
        
        return TeamMomentumIndex(
            team_tricode=team,
            game_id=game_id,
            tmi_value=final_tmi,
            timestamp=datetime.utcnow(),
            prediction_probability=prediction_prob,
            confidence_score=confidence,
            contributing_factors={
                'scoring_momentum': scoring_momentum,
                'defensive_momentum': defensive_momentum,
                'situational_momentum': situational_momentum,
                'energy_momentum': energy_momentum,
                'scoring_run_length': latest_features.scoring_run_length,
                'defensive_stops': latest_features.defensive_stops,
                'is_clutch_time': latest_features.is_clutch_time
            }
        )
    
    # Helper methods
    def _get_event_points(self, event: GameEvent) -> int:
        """Get points scored from an event."""
        if event.event_type == 'shot' and event.shot_result == 'Made':
            if '3pt' in event.description.lower():
                return 3
            elif 'free throw' in event.description.lower():
                return 1
            else:
                return 2
        return 0
    
    def _estimate_possession_duration(self, possession: Possession) -> float:
        """Estimate possession duration in minutes."""
        if len(possession.events) < 2:
            return 0.5  # Default duration
        
        # Simple estimation based on number of events
        return min(2.0, len(possession.events) * 0.1)
    
    def _estimate_defensive_rating(
        self,
        possession: Possession,
        previous_possessions: List[Possession]
    ) -> float:
        """Estimate defensive rating (simplified)."""
        # Look at opponent scoring in recent possessions
        opponent_points = 0
        opponent_possessions = 0
        
        for prev_poss in previous_possessions[-5:]:
            if prev_poss.team_tricode != possession.team_tricode:
                points = sum(
                    self._get_event_points(event) for event in prev_poss.events
                    if event.event_type == 'shot' and event.shot_result == 'Made'
                )
                opponent_points += points
                opponent_possessions += 1
        
        if opponent_possessions > 0:
            return (opponent_points / opponent_possessions) * 100
        else:
            return 100.0  # League average
    
    def _initialize_team_stats(self) -> Dict[str, Dict]:
        """Initialize team statistics tracking."""
        return {}
    
    def _update_team_stats(
        self,
        team_stats: Dict[str, Dict],
        possession: Possession,
        features: EnhancedPossessionFeatures
    ):
        """Update team statistics with new possession data."""
        team = possession.team_tricode
        
        if team not in team_stats:
            team_stats[team] = {
                'total_possessions': 0,
                'total_points': 0,
                'avg_pace': 1.0
            }
        
        team_stats[team]['total_possessions'] += 1
        team_stats[team]['total_points'] += features.points_scored
        team_stats[team]['avg_pace'] = (
            team_stats[team]['avg_pace'] * 0.9 + features.pace * 0.1
        )
    
    def _get_default_features(self) -> EnhancedPossessionFeatures:
        """Get default features for error cases."""
        return EnhancedPossessionFeatures(
            points_scored=0.0, fg_percentage=0.0, turnovers=0.0,
            rebounds=0.0, fouls=0.0, pace=1.0,
            effective_fg_percentage=0.0, true_shooting_percentage=0.0,
            assist_to_turnover_ratio=1.0, offensive_rating=100.0,
            defensive_rating=100.0, time_remaining=0.0,
            score_margin=0.0, period=1, is_clutch_time=False,
            scoring_run_length=0, defensive_stops=0,
            momentum_events=0.0, energy_level=0.0,
            player_impact=0.0, pace_differential=0.0,
            shot_quality=0.5, defensive_pressure=0.0
        )
    
    def _get_default_situational_features(self) -> Dict[str, float]:
        """Get default situational features."""
        return {
            'time_remaining': 0.0,
            'score_margin': 0.0,
            'period': 1,
            'is_clutch_time': 0.0
        }
    
    def _get_default_tmi(self, team: str, game_id: str) -> TeamMomentumIndex:
        """Get default TMI for error cases."""
        return TeamMomentumIndex(
            team_tricode=team,
            game_id=game_id,
            tmi_value=0.0,
            timestamp=datetime.utcnow(),
            prediction_probability=0.5,
            confidence_score=0.0
        )
    
    # Momentum calculation methods
    def _calculate_scoring_momentum(self, features: List[EnhancedPossessionFeatures]) -> float:
        """Calculate scoring-based momentum."""
        if not features:
            return 0.0
        
        recent_features = features[-5:]  # Last 5 possessions
        
        # Scoring trend
        scoring_trend = np.mean([f.points_scored for f in recent_features])
        
        # Shooting efficiency
        shooting_efficiency = np.mean([f.effective_fg_percentage for f in recent_features])
        
        # Scoring run bonus
        latest_run = recent_features[-1].scoring_run_length if recent_features else 0
        run_bonus = min(0.3, latest_run / 20.0)  # Cap at 0.3
        
        return (scoring_trend * 0.4 + shooting_efficiency * 0.4 + run_bonus * 0.2)
    
    def _calculate_defensive_momentum(self, features: List[EnhancedPossessionFeatures]) -> float:
        """Calculate defense-based momentum."""
        if not features:
            return 0.0
        
        recent_features = features[-5:]
        
        # Defensive stops
        avg_stops = np.mean([f.defensive_stops for f in recent_features])
        
        # Defensive pressure
        avg_pressure = np.mean([f.defensive_pressure for f in recent_features])
        
        # Defensive rating
        avg_def_rating = np.mean([f.defensive_rating for f in recent_features])
        def_rating_normalized = max(0, (110 - avg_def_rating) / 20)  # Lower is better
        
        return (avg_stops * 0.4 + avg_pressure * 0.3 + def_rating_normalized * 0.3) / 10
    
    def _calculate_situational_momentum(self, features: List[EnhancedPossessionFeatures]) -> float:
        """Calculate situation-based momentum."""
        if not features:
            return 0.0
        
        latest = features[-1]
        
        # Clutch time performance
        clutch_bonus = 0.2 if latest.is_clutch_time else 0.0
        
        # Close game bonus
        close_game_bonus = 0.1 if latest.score_margin <= 5 else 0.0
        
        # Time pressure (more momentum value late in game)
        time_factor = max(0, (48 - latest.time_remaining) / 48) * 0.1
        
        return clutch_bonus + close_game_bonus + time_factor
    
    def _calculate_energy_momentum(self, features: List[EnhancedPossessionFeatures]) -> float:
        """Calculate energy-based momentum."""
        if not features:
            return 0.0
        
        recent_features = features[-3:]  # Last 3 possessions
        
        # Average energy level
        avg_energy = np.mean([f.energy_level for f in recent_features])
        
        # Momentum events
        avg_momentum_events = np.mean([f.momentum_events for f in recent_features])
        
        # Normalize to 0-1 range
        energy_normalized = avg_energy / 10.0
        momentum_events_normalized = max(0, min(1, avg_momentum_events / 5.0))
        
        return (energy_normalized * 0.6 + momentum_events_normalized * 0.4)


def create_enhanced_momentum_engine(**kwargs) -> EnhancedMomentumEngine:
    """Create an enhanced momentum engine instance."""
    return EnhancedMomentumEngine(**kwargs)