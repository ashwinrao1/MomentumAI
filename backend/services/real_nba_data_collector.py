"""
Real NBA data collection service with improved data quality and features.

This module replaces the synthetic data collector with real NBA API integration,
better feature engineering, and proper momentum definitions.
"""

import logging
import time
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams
from nba_api.live.nba.endpoints import scoreboard

from backend.models.game_models import GameEvent, GameInfo

logger = logging.getLogger(__name__)


class RealNBADataCollector:
    """
    Collects real NBA play-by-play data with enhanced features for momentum prediction.
    
    Note: This collector is designed for historical training data from the 2024-25 season.
    For live/current games from the 2025-26 season, use the LiveDataFetcher instead.
    
    Improvements over the original collector:
    - Better error handling and rate limiting
    - Enhanced feature extraction
    - Game context information
    - Player and team performance metrics
    """
    
    def __init__(self, rate_limit_delay: float = 0.6):
        """Initialize with configurable rate limiting."""
        self.rate_limit_delay = rate_limit_delay
        self.collected_games = set()
        self.team_cache = {}
        self._load_team_data()
    
    def _load_team_data(self):
        """Load and cache team information."""
        try:
            nba_teams = teams.get_teams()
            for team in nba_teams:
                self.team_cache[team['id']] = team
            logger.info(f"Loaded {len(self.team_cache)} NBA teams")
        except Exception as e:
            logger.error(f"Error loading team data: {e}")
    
    def collect_recent_completed_games(
        self,
        days_back: int = 7,
        max_games: int = 50,
        season: str = "2024-25"
    ) -> List[GameInfo]:
        """
        Collect recently completed games with better filtering.
        
        Args:
            days_back: Days to look back for games
            max_games: Maximum games to collect
            season: NBA season
            
        Returns:
            List of completed GameInfo objects
        """
        logger.info(f"Collecting completed games from last {days_back} days")
        
        try:
            # Get recent games using scoreboard
            end_date = datetime.now()
            games = []
            
            # Check each day going back
            for days_ago in range(days_back):
                check_date = end_date - timedelta(days=days_ago)
                date_str = check_date.strftime('%Y-%m-%d')
                
                try:
                    # Get games for this date
                    board = scoreboard.ScoreBoard(game_date=date_str)
                    daily_games = board.games.get_dict()
                    
                    for game in daily_games:
                        if game['gameStatus'] == 3:  # Game finished
                            game_info = GameInfo(
                                game_id=game['gameId'],
                                home_team=game['homeTeam']['teamTricode'],
                                away_team=game['awayTeam']['teamTricode'],
                                game_date=date_str,
                                status="Final",
                                period=game.get('period', 4),
                                clock="00:00",
                                home_score=game['homeTeam']['score'],
                                away_score=game['awayTeam']['score']
                            )
                            games.append(game_info)
                    
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Error getting games for {date_str}: {e}")
                    continue
                
                if len(games) >= max_games:
                    break
            
            # Remove duplicates and sort by date
            unique_games = {game.game_id: game for game in games}
            result = list(unique_games.values())[:max_games]
            
            logger.info(f"Found {len(result)} completed games")
            return result
            
        except Exception as e:
            logger.error(f"Error collecting recent games: {e}")
            return []
    
    def collect_enhanced_game_events(self, game_id: str) -> Tuple[List[GameEvent], Dict[str, Any]]:
        """
        Collect play-by-play events with enhanced context information.
        
        Args:
            game_id: NBA game ID
            
        Returns:
            Tuple of (events, game_context)
        """
        if game_id in self.collected_games:
            logger.info(f"Game {game_id} already collected")
            return [], {}
        
        try:
            logger.info(f"Collecting enhanced data for game {game_id}")
            
            # Get play-by-play data
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            pbp_data = pbp.get_data_frames()[0]
            
            # Get box score for additional context
            box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            team_stats = box_score.team_stats.get_dict()['data']
            
            # Extract game context
            game_context = self._extract_game_context(team_stats, game_id)
            
            # Process events with enhanced features
            events = []
            for _, row in pbp_data.iterrows():
                try:
                    event = self._parse_enhanced_play_by_play_row(row, game_id, game_context)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue
            
            self.collected_games.add(game_id)
            logger.info(f"Collected {len(events)} enhanced events for game {game_id}")
            
            time.sleep(self.rate_limit_delay)
            return events, game_context
            
        except Exception as e:
            logger.error(f"Error collecting enhanced events for game {game_id}: {e}")
            return [], {}
    
    def _extract_game_context(self, team_stats: List[Dict], game_id: str) -> Dict[str, Any]:
        """Extract contextual information about the game."""
        context = {
            'game_id': game_id,
            'total_points': 0,
            'pace_estimate': 0,
            'competitive_game': False,
            'team_performance': {}
        }
        
        try:
            if len(team_stats) >= 2:
                team1, team2 = team_stats[0], team_stats[1]
                
                # Basic game stats
                team1_pts = team1.get('PTS', 0)
                team2_pts = team2.get('PTS', 0)
                context['total_points'] = team1_pts + team2_pts
                
                # Competitive game (within 15 points)
                context['competitive_game'] = abs(team1_pts - team2_pts) <= 15
                
                # Pace estimate (possessions per team)
                team1_fga = team1.get('FGA', 0)
                team1_fta = team1.get('FTA', 0)
                team1_tov = team1.get('TOV', 0)
                team1_oreb = team1.get('OREB', 0)
                team2_dreb = team2.get('DREB', 0)
                
                if team1_fga > 0:
                    possessions = team1_fga + 0.44 * team1_fta + team1_tov - team1_oreb
                    context['pace_estimate'] = possessions
                
                # Team performance metrics
                for i, team in enumerate([team1, team2]):
                    team_id = team.get('TEAM_ID')
                    context['team_performance'][team_id] = {
                        'fg_pct': team.get('FG_PCT', 0),
                        'fg3_pct': team.get('FG3_PCT', 0),
                        'ft_pct': team.get('FT_PCT', 0),
                        'rebounds': team.get('REB', 0),
                        'assists': team.get('AST', 0),
                        'turnovers': team.get('TOV', 0),
                        'steals': team.get('STL', 0),
                        'blocks': team.get('BLK', 0)
                    }
        
        except Exception as e:
            logger.warning(f"Error extracting game context: {e}")
        
        return context
    
    def _parse_enhanced_play_by_play_row(
        self, 
        row: Any, 
        game_id: str, 
        game_context: Dict[str, Any]
    ) -> Optional[GameEvent]:
        """Parse play-by-play row with enhanced features."""
        try:
            # Basic event information
            event_id = f"{game_id}_{row.get('EVENTNUM', 0)}"
            period = int(row.get('PERIOD', 1))
            clock = str(row.get('PCTIMESTRING', '12:00'))
            
            # Enhanced description parsing
            home_desc = str(row.get('HOMEDESCRIPTION', '') or '')
            away_desc = str(row.get('VISITORDESCRIPTION', '') or '')
            description = home_desc if home_desc else away_desc
            
            if not description:
                return None
            
            # Determine team and player
            team_tricode = None
            player_name = row.get('PLAYER1_NAME')
            
            if row.get('PLAYER1_TEAM_ABBREVIATION'):
                team_tricode = row.get('PLAYER1_TEAM_ABBREVIATION')
            
            if not team_tricode:
                return None
            
            # Enhanced event classification
            event_type, shot_result, event_value = self._classify_enhanced_event(description, row)
            
            # Calculate game time remaining (for context)
            time_remaining = self._calculate_time_remaining(period, clock)
            
            # Extract score information
            score_info = self._parse_score(row.get('SCORE', '0-0'))
            
            return GameEvent(
                event_id=event_id,
                game_id=game_id,
                team_tricode=team_tricode,
                player_name=player_name,
                event_type=event_type,
                clock=clock,
                period=period,
                points_total=score_info['total_points'],
                shot_result=shot_result,
                timestamp=datetime.utcnow(),
                description=description,
                # Enhanced fields
                event_value=event_value,
                time_remaining=time_remaining,
                score_margin=score_info['margin'],
                game_context=game_context
            )
            
        except Exception as e:
            logger.warning(f"Error parsing enhanced row: {e}")
            return None
    
    def _classify_enhanced_event(self, description: str, row: Any) -> Tuple[str, Optional[str], float]:
        """
        Enhanced event classification with value scoring.
        
        Returns:
            Tuple of (event_type, shot_result, event_value)
        """
        desc_lower = description.lower()
        event_value = 0.0
        
        # Shot events
        if any(word in desc_lower for word in ['makes', 'made']):
            if '3pt' in desc_lower:
                return 'shot', 'Made', 3.0
            elif 'free throw' in desc_lower:
                return 'shot', 'Made', 1.0
            else:
                return 'shot', 'Made', 2.0
        
        elif any(word in desc_lower for word in ['misses', 'missed']):
            if 'free throw' in desc_lower:
                return 'shot', 'Missed', -0.5
            else:
                return 'shot', 'Missed', -1.0
        
        # Positive events
        elif 'rebound' in desc_lower:
            if 'offensive' in desc_lower:
                return 'rebound', None, 1.5
            else:
                return 'rebound', None, 0.5
        
        elif 'assist' in desc_lower:
            return 'assist', None, 1.0
        
        elif 'steal' in desc_lower:
            return 'steal', None, 2.0
        
        elif 'block' in desc_lower:
            return 'block', None, 1.5
        
        # Negative events
        elif 'turnover' in desc_lower:
            return 'turnover', None, -2.0
        
        elif 'foul' in desc_lower:
            if 'flagrant' in desc_lower:
                return 'foul', None, -3.0
            elif 'technical' in desc_lower:
                return 'foul', None, -2.0
            else:
                return 'foul', None, -0.5
        
        else:
            return 'other', None, 0.0
    
    def _calculate_time_remaining(self, period: int, clock: str) -> float:
        """Calculate total time remaining in game (minutes)."""
        try:
            # Parse clock (MM:SS format)
            if ':' in clock:
                minutes, seconds = map(int, clock.split(':'))
                period_time_remaining = minutes + seconds / 60.0
            else:
                period_time_remaining = 0.0
            
            # Calculate total time remaining
            if period <= 4:
                # Regular time
                periods_remaining = 4 - period
                total_remaining = periods_remaining * 12 + period_time_remaining
            else:
                # Overtime
                total_remaining = period_time_remaining
            
            return max(0.0, total_remaining)
            
        except Exception:
            return 0.0
    
    def _parse_score(self, score_str: str) -> Dict[str, Any]:
        """Parse score string and calculate margin."""
        try:
            if '-' in score_str:
                away_score, home_score = map(int, score_str.split('-'))
                return {
                    'home_score': home_score,
                    'away_score': away_score,
                    'total_points': home_score + away_score,
                    'margin': abs(home_score - away_score)
                }
        except Exception:
            pass
        
        return {
            'home_score': 0,
            'away_score': 0,
            'total_points': 0,
            'margin': 0
        }
    
    def collect_training_dataset(
        self,
        days_back: int = 14,
        max_games: int = 100,
        season: str = "2024-25",
        save_cache: bool = True
    ) -> List[GameEvent]:
        """
        Collect a comprehensive training dataset with caching.
        
        Args:
            days_back: Days to look back
            max_games: Maximum games to collect
            season: NBA season
            save_cache: Whether to cache the results
            
        Returns:
            List of enhanced GameEvent objects
        """
        cache_file = f"data/nba_training_data_{season}_{days_back}d_{max_games}g.pkl"
        
        # Try to load from cache first
        if save_cache:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    logger.info(f"Loaded {len(cached_data)} events from cache")
                    return cached_data
            except FileNotFoundError:
                logger.info("No cache found, collecting fresh data")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        logger.info("Starting comprehensive training dataset collection")
        
        # Get list of games
        games = self.collect_recent_completed_games(days_back, max_games, season)
        
        if not games:
            logger.error("No games found for dataset collection")
            return []
        
        # Collect events for each game
        all_events = []
        game_contexts = {}
        
        for i, game in enumerate(games):
            logger.info(f"Processing game {i+1}/{len(games)}: {game.game_id}")
            
            try:
                events, context = self.collect_enhanced_game_events(game.game_id)
                all_events.extend(events)
                game_contexts[game.game_id] = context
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1} games, collected {len(all_events)} total events")
                
            except Exception as e:
                logger.error(f"Error processing game {game.game_id}: {e}")
                continue
        
        # Save to cache
        if save_cache and all_events:
            try:
                import os
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(all_events, f)
                logger.info(f"Cached {len(all_events)} events to {cache_file}")
            except Exception as e:
                logger.warning(f"Error saving cache: {e}")
        
        logger.info(f"Dataset collection complete: {len(all_events)} events from {len(games)} games")
        return all_events


def create_real_nba_collector(rate_limit_delay: float = 0.6) -> RealNBADataCollector:
    """Create a real NBA data collector instance."""
    return RealNBADataCollector(rate_limit_delay=rate_limit_delay)


# Enhanced GameEvent model to support new fields
class EnhancedGameEvent(GameEvent):
    """Enhanced GameEvent with additional context fields."""
    
    def __init__(self, **kwargs):
        # Extract enhanced fields
        self.event_value = kwargs.pop('event_value', 0.0)
        self.time_remaining = kwargs.pop('time_remaining', 0.0)
        self.score_margin = kwargs.pop('score_margin', 0)
        self.game_context = kwargs.pop('game_context', {})
        
        # Initialize base class
        super().__init__(**kwargs)