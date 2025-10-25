"""
Live NBA data fetcher module for MomentumML.

This module handles real-time data collection from the NBA API,
including game event polling, parsing, and error handling.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from nba_api.live.nba.endpoints import scoreboard, boxscore
from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder

from backend.models.game_models import GameEvent, GameInfo
from backend.utils.error_handling import (
    APIError, NetworkError, DataProcessingError,
    retry_with_exponential_backoff, handle_api_errors,
    graceful_degradation, safe_execute, validate_data,
    log_error, ErrorSeverity, health_checker
)

# Configure logging
logger = logging.getLogger("momentum_ml.live_fetcher")


class LiveDataFetcher:
    """
    Handles live NBA data fetching and processing.
    
    This class manages API polling, rate limiting, error handling,
    and data standardization for the MomentumML system.
    """
    
    def __init__(self, poll_interval: int = 25, max_retries: int = 3):
        """
        Initialize the live data fetcher.
        
        Args:
            poll_interval: Seconds between API polls (default 25)
            max_retries: Maximum retry attempts for failed requests
        """
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.last_poll_time = {}  # Track last poll time per game
        self.cached_game_data = {}  # Cache for resilience
        
    @retry_with_exponential_backoff(
        max_retries=3,
        base_delay=1.0,
        exceptions=(Exception,)
    )
    @handle_api_errors
    async def get_active_games(self) -> List[GameInfo]:
        """
        Fetch current NBA games from this week (2025-26 season).
        
        Note: This fetches live/current games from the 2025-26 season.
        Training data uses historical 2024-25 season games.
        
        Returns:
            List of GameInfo objects for current games (live, scheduled, or recently finished)
            
        Raises:
            APIError: If API request fails after retries
        """
        try:
            logger.info("Fetching current NBA games from this week (2025-26 season)")
            
            # Check if NBA API service is healthy
            if not graceful_degradation.is_service_healthy("nba_api"):
                # Try to get cached games
                cached_games = graceful_degradation.get_fallback_data("active_games", max_age_seconds=300)
                if cached_games:
                    logger.warning("NBA API unhealthy, using cached active games")
                    return cached_games
            
            current_games = []
            
            # Get today's games first (this is what ScoreBoard() returns by default)
            try:
                logger.info("Getting today's NBA games")
                board = scoreboard.ScoreBoard()
                games_data = board.get_dict()
                
                # Validate API response structure
                validate_data(
                    games_data,
                    lambda data: isinstance(data, dict) and 'scoreboard' in data,
                    "Invalid scoreboard API response structure"
                )
                
                if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
                    for game in games_data['scoreboard']['games']:
                        try:
                            # Validate individual game data
                            validate_data(
                                game,
                                lambda g: all(key in g for key in ['gameId', 'homeTeam', 'awayTeam', 'gameStatus']),
                                f"Invalid game data structure for game {game.get('gameId', 'unknown')}"
                            )
                            
                            # Include all games (scheduled, live, or finished)
                            # gameStatus: 1=Scheduled, 2=Live, 3=Final
                            if game['gameStatus'] in [1, 2, 3]:
                                # Extract game date from the game data
                                game_date = game.get('gameTimeUTC', datetime.now().isoformat())[:10]
                                
                                game_info = GameInfo(
                                    game_id=game['gameId'],
                                    home_team=game['homeTeam']['teamTricode'],
                                    away_team=game['awayTeam']['teamTricode'],
                                    game_date=game_date,
                                    status=self._get_game_status(game['gameStatus']),
                                    period=game.get('period', 0),
                                    clock=game.get('gameClock', ''),
                                    home_score=game['homeTeam'].get('score', 0),
                                    away_score=game['awayTeam'].get('score', 0)
                                )
                                current_games.append(game_info)
                                
                        except DataProcessingError as e:
                            log_error(e, context={"game_id": game.get('gameId', 'unknown')})
                            continue  # Skip invalid games but continue processing others
                
            except Exception as e:
                logger.warning(f"Failed to get today's games: {e}")
                
            # Get recent completed games from past few days using stats API
            try:
                logger.info("Getting recent completed games from past few days")
                from nba_api.stats.endpoints import leaguegamefinder
                
                # Get games from past 3 days
                for days_ago in range(1, 4):  # 1, 2, 3 days ago
                    try:
                        past_date = datetime.now() - timedelta(days=days_ago)
                        date_str = past_date.strftime('%m/%d/%Y')
                        
                        logger.info(f"Checking completed games for {date_str}")
                        
                        # Use LeagueGameFinder to get games for specific date
                        game_finder = leaguegamefinder.LeagueGameFinder(
                            date_from_nullable=date_str,
                            date_to_nullable=date_str,
                            season_nullable='2025-26',
                            season_type_nullable='Regular Season'
                        )
                        
                        games_df = game_finder.get_data_frames()[0]
                        
                        if not games_df.empty:
                            # Process each game (note: each game appears twice, once for each team)
                            processed_game_ids = set()
                            
                            for _, row in games_df.iterrows():
                                game_id = str(row['GAME_ID'])
                                
                                # Skip if we already processed this game
                                if game_id in processed_game_ids:
                                    continue
                                    
                                processed_game_ids.add(game_id)
                                
                                # Find the matching row for the other team
                                other_team_row = games_df[
                                    (games_df['GAME_ID'] == row['GAME_ID']) & 
                                    (games_df['TEAM_ID'] != row['TEAM_ID'])
                                ]
                                
                                if not other_team_row.empty:
                                    other_row = other_team_row.iloc[0]
                                    
                                    # Determine home/away based on matchup string
                                    matchup = row['MATCHUP']
                                    if ' @ ' in matchup:
                                        # This team is away
                                        away_team = row['TEAM_ABBREVIATION']
                                        home_team = other_row['TEAM_ABBREVIATION']
                                        away_score = row['PTS']
                                        home_score = other_row['PTS']
                                    else:
                                        # This team is home
                                        home_team = row['TEAM_ABBREVIATION']
                                        away_team = other_row['TEAM_ABBREVIATION']
                                        home_score = row['PTS']
                                        away_score = other_row['PTS']
                                    
                                    game_info = GameInfo(
                                        game_id=game_id,
                                        home_team=home_team,
                                        away_team=away_team,
                                        game_date=past_date.strftime('%Y-%m-%d'),
                                        status="Final",
                                        period=4,
                                        clock="0:00",
                                        home_score=int(home_score) if home_score else 0,
                                        away_score=int(away_score) if away_score else 0
                                    )
                                    current_games.append(game_info)
                        
                        # Small delay between requests
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.warning(f"Failed to get completed games for {date_str}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to get recent completed games: {e}")
            
            # Remove duplicates based on game_id and sort by date
            unique_games = {}
            for game in current_games:
                if game.game_id not in unique_games:
                    unique_games[game.game_id] = game
            
            final_games = list(unique_games.values())
            final_games.sort(key=lambda g: (g.game_date, g.game_id))
            
            # Cache successful result
            graceful_degradation.set_fallback_data("active_games", final_games)
            graceful_degradation.set_service_status("nba_api", True)
            
            logger.info(f"Successfully fetched {len(final_games)} current NBA games from this week")
            return final_games
            
        except Exception as e:
            graceful_degradation.set_service_status("nba_api", False)
            
            # Try to return cached data as fallback
            cached_games = graceful_degradation.get_fallback_data("active_games", max_age_seconds=600)
            if cached_games:
                logger.warning("API failed, returning cached active games")
                return cached_games
            
            # No fallback available
            raise APIError(
                f"Failed to fetch current NBA games: {str(e)}",
                original_error=e,
                severity=ErrorSeverity.HIGH
            )
    
    @retry_with_exponential_backoff(
        max_retries=3,
        base_delay=2.0,
        exceptions=(Exception,)
    )
    @handle_api_errors
    async def fetch_live_game_data(self, game_id: str) -> Tuple[GameInfo, List[GameEvent]]:
        """
        Fetch live play-by-play data for a specific game.
        
        Args:
            game_id: NBA game ID
            
        Returns:
            Tuple of (GameInfo, List[GameEvent])
            
        Raises:
            APIError: If API request fails after retries
        """
        try:
            logger.info(f"Fetching game data for {game_id}")
            
            # Validate game_id
            validate_data(
                game_id,
                lambda gid: isinstance(gid, str) and len(gid) > 0,
                "Invalid game ID provided"
            )
            
            # Check rate limiting
            if not self._can_poll_game(game_id):
                logger.info(f"Rate limiting active for game {game_id}")
                cached_data = self._get_cached_data(game_id)
                if cached_data[0] is not None:
                    return cached_data
                else:
                    raise APIError(
                        f"Rate limited and no cached data available for game {game_id}",
                        severity=ErrorSeverity.MEDIUM
                    )
            
            # Check if NBA API service is healthy
            if not graceful_degradation.is_service_healthy("nba_api"):
                cached_data = self._get_cached_data(game_id)
                if cached_data[0] is not None:
                    logger.warning(f"NBA API unhealthy, using cached data for game {game_id}")
                    return cached_data
            
            # Get live boxscore for game info
            try:
                box = boxscore.BoxScore(game_id=game_id)
                box_data = box.get_dict()
            except Exception as e:
                raise APIError(
                    f"Failed to fetch boxscore for game {game_id}",
                    original_error=e,
                    severity=ErrorSeverity.HIGH
                )
            
            # Extract game info with error handling
            try:
                game_info = self._parse_game_info(box_data, game_id)
            except Exception as e:
                raise DataProcessingError(
                    f"Failed to parse game info for game {game_id}",
                    original_error=e,
                    details={"game_id": game_id}
                )
            
            # Get play-by-play data
            try:
                pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
                pbp_data = pbp.get_data_frames()[0]  # Get the main dataframe
            except Exception as e:
                raise APIError(
                    f"Failed to fetch play-by-play data for game {game_id}",
                    original_error=e,
                    severity=ErrorSeverity.HIGH
                )
            
            # Parse events with error handling
            try:
                events = self._parse_events(pbp_data, game_id)
            except Exception as e:
                raise DataProcessingError(
                    f"Failed to parse events for game {game_id}",
                    original_error=e,
                    details={"game_id": game_id, "events_count": len(pbp_data) if pbp_data is not None else 0}
                )
            
            # Update cache and poll time
            self._update_cache(game_id, game_info, events)
            self.last_poll_time[game_id] = time.time()
            
            # Mark NBA API as healthy
            graceful_degradation.set_service_status("nba_api", True)
            
            logger.info(f"Successfully fetched {len(events)} events for game {game_id}")
            return game_info, events
            
        except (APIError, DataProcessingError):
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Mark NBA API as potentially unhealthy
            graceful_degradation.set_service_status("nba_api", False)
            
            # Try to return cached data as fallback
            cached_data = self._get_cached_data(game_id)
            if cached_data[0] is not None:
                logger.warning(f"API failed, returning cached data for game {game_id}")
                return cached_data
            
            # No fallback available
            raise APIError(
                f"Failed to fetch game data for {game_id}: {str(e)}",
                original_error=e,
                severity=ErrorSeverity.HIGH,
                details={"game_id": game_id}
            )
    
    def _parse_game_info(self, box_data: Dict, game_id: str) -> GameInfo:
        """Parse game information from boxscore data."""
        try:
            # Validate boxscore data structure
            validate_data(
                box_data,
                lambda data: isinstance(data, dict) and 'game' in data,
                "Invalid boxscore data structure"
            )
            
            game = box_data['game']
            
            # Validate game data structure
            validate_data(
                game,
                lambda g: all(key in g for key in ['homeTeam', 'awayTeam', 'gameStatus']),
                "Invalid game data structure in boxscore"
            )
            
            home_team = game['homeTeam']
            away_team = game['awayTeam']
            
            # Validate team data
            for team_name, team_data in [('home', home_team), ('away', away_team)]:
                validate_data(
                    team_data,
                    lambda t: isinstance(t, dict) and 'teamTricode' in t,
                    f"Invalid {team_name} team data structure"
                )
            
            return GameInfo(
                game_id=game_id,
                home_team=home_team['teamTricode'],
                away_team=away_team['teamTricode'],
                game_date=safe_execute(
                    lambda: game['gameTimeUTC'][:10],
                    datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                    "Failed to parse game date"
                ),
                status=self._get_game_status(game.get('gameStatus', 1)),
                period=safe_execute(lambda: int(game.get('period', 0)), 0, "Failed to parse period"),
                clock=game.get('gameClock', ''),
                home_score=safe_execute(lambda: int(home_team.get('score', 0)), 0, "Failed to parse home score"),
                away_score=safe_execute(lambda: int(away_team.get('score', 0)), 0, "Failed to parse away score")
            )
            
        except (DataProcessingError, ValueError) as e:
            raise DataProcessingError(
                f"Failed to parse game info for {game_id}",
                original_error=e,
                details={"game_id": game_id, "data_keys": list(box_data.keys()) if isinstance(box_data, dict) else None}
            )
        except Exception as e:
            raise DataProcessingError(
                f"Unexpected error parsing game info for {game_id}",
                original_error=e,
                details={"game_id": game_id}
            )
    
    def _parse_events(self, pbp_df: pd.DataFrame, game_id: str) -> List[GameEvent]:
        """
        Parse play-by-play dataframe into standardized GameEvent objects.
        
        Args:
            pbp_df: Play-by-play dataframe from NBA API
            game_id: NBA game ID
            
        Returns:
            List of GameEvent objects
        """
        events = []
        
        try:
            # Validate dataframe
            validate_data(
                pbp_df,
                lambda df: isinstance(df, pd.DataFrame) and not df.empty,
                "Invalid or empty play-by-play dataframe"
            )
            
            for index, row in pbp_df.iterrows():
                try:
                    # Skip null events
                    if pd.isna(row.get('EVENTMSGTYPE')):
                        continue
                    
                    # Safely parse each field with error handling
                    event_id = safe_execute(
                        lambda: str(row.get('EVENTNUM', f"{game_id}_{index}")),
                        f"{game_id}_{index}",
                        f"Failed to parse event ID for row {index}"
                    )
                    
                    team_tricode = safe_execute(
                        lambda: self._get_team_tricode(row),
                        'UNK',
                        f"Failed to parse team tricode for event {event_id}"
                    )
                    
                    event_type = safe_execute(
                        lambda: self._parse_event_type(row.get('EVENTMSGTYPE', 0)),
                        'other',
                        f"Failed to parse event type for event {event_id}"
                    )
                    
                    period = safe_execute(
                        lambda: int(row.get('PERIOD', 0)),
                        0,
                        f"Failed to parse period for event {event_id}"
                    )
                    
                    points_total = safe_execute(
                        lambda: self._parse_points_total(row.get('SCORE', '0-0')),
                        0,
                        f"Failed to parse points total for event {event_id}"
                    )
                    
                    shot_result = safe_execute(
                        lambda: self._parse_shot_result(row),
                        None,
                        f"Failed to parse shot result for event {event_id}",
                        log_errors=False  # Shot result is optional
                    )
                    
                    # Create standardized event
                    event = GameEvent(
                        event_id=event_id,
                        game_id=game_id,
                        team_tricode=team_tricode,
                        player_name=safe_execute(
                            lambda: row.get('PLAYER1_NAME'),
                            None,
                            f"Failed to parse player name for event {event_id}",
                            log_errors=False
                        ),
                        event_type=event_type,
                        clock=safe_execute(
                            lambda: row.get('PCTIMESTRING', ''),
                            '',
                            f"Failed to parse clock for event {event_id}",
                            log_errors=False
                        ),
                        period=period,
                        points_total=points_total,
                        shot_result=shot_result,
                        timestamp=datetime.now(timezone.utc),
                        description=safe_execute(
                            lambda: row.get('HOMEDESCRIPTION', '') or row.get('VISITORDESCRIPTION', '') or '',
                            '',
                            f"Failed to parse description for event {event_id}",
                            log_errors=False
                        )
                    )
                    
                    events.append(event)
                    
                except Exception as e:
                    # Log individual event parsing error but continue processing
                    log_error(
                        e,
                        context={
                            "game_id": game_id,
                            "row_index": index,
                            "event_data": str(row.to_dict())[:200]  # Truncate for logging
                        },
                        severity=ErrorSeverity.LOW
                    )
                    continue
                
        except DataProcessingError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise DataProcessingError(
                f"Failed to parse play-by-play events for game {game_id}",
                original_error=e,
                details={
                    "game_id": game_id,
                    "dataframe_shape": pbp_df.shape if pbp_df is not None else None,
                    "events_parsed": len(events)
                }
            )
        
        logger.info(f"Successfully parsed {len(events)} events from {len(pbp_df)} rows for game {game_id}")
        return events
    
    def _parse_points_total(self, score_str: str) -> int:
        """Parse points total from score string like '105-98'."""
        try:
            if not score_str or score_str == '0-0':
                return 0
            # Take the first number (home team score)
            return int(score_str.split('-')[0])
        except (ValueError, IndexError):
            return 0
    
    def _get_team_tricode(self, row: pd.Series) -> str:
        """Extract team tricode from play-by-play row."""
        # Try to get team from player team ID or description
        if not pd.isna(row.get('PLAYER1_TEAM_ID')):
            # This would need team ID to tricode mapping
            # For now, extract from description
            pass
        
        # Extract from description as fallback
        home_desc = row.get('HOMEDESCRIPTION', '')
        visitor_desc = row.get('VISITORDESCRIPTION', '')
        
        if home_desc:
            return 'HOME'  # Placeholder - would need actual team mapping
        elif visitor_desc:
            return 'AWAY'  # Placeholder - would need actual team mapping
        
        return 'UNK'
    
    def _parse_event_type(self, event_msg_type: int) -> str:
        """Convert NBA API event message type to standardized event type."""
        event_type_map = {
            1: 'shot',
            2: 'shot',  # Made shot
            3: 'free_throw',
            4: 'rebound',
            5: 'turnover',
            6: 'foul',
            7: 'violation',
            8: 'substitution',
            9: 'timeout',
            10: 'jump_ball',
            11: 'ejection',
            12: 'period_start',
            13: 'period_end'
        }
        
        return event_type_map.get(event_msg_type, 'other')
    
    def _parse_shot_result(self, row: pd.Series) -> Optional[str]:
        """Parse shot result from play-by-play row."""
        event_type = row.get('EVENTMSGTYPE', 0)
        
        if event_type == 1:
            return 'Missed'
        elif event_type == 2:
            return 'Made'
        
        return None
    
    def _get_game_status(self, status_code: int) -> str:
        """Convert NBA API game status code to readable status."""
        status_map = {
            1: 'Scheduled',
            2: 'Live', 
            3: 'Final',
            4: 'Final OT'
        }
        return status_map.get(status_code, 'Unknown')
    
    def _can_poll_game(self, game_id: str) -> bool:
        """Check if enough time has passed since last poll for rate limiting."""
        if game_id not in self.last_poll_time:
            return True
        
        time_since_last_poll = time.time() - self.last_poll_time[game_id]
        return time_since_last_poll >= self.poll_interval
    
    def _update_cache(self, game_id: str, game_info: GameInfo, events: List[GameEvent]):
        """Update cached game data for resilience."""
        self.cached_game_data[game_id] = {
            'game_info': game_info,
            'events': events,
            'cached_at': time.time()
        }
    
    def _get_cached_data(self, game_id: str) -> Tuple[Optional[GameInfo], List[GameEvent]]:
        """Retrieve cached game data."""
        if game_id in self.cached_game_data:
            cached = self.cached_game_data[game_id]
            return cached['game_info'], cached['events']
        
        return None, []


# Health check for NBA API
def _check_nba_api_health() -> bool:
    """Check if NBA API is accessible."""
    try:
        # Simple test to see if we can create a scoreboard instance
        board = scoreboard.ScoreBoard()
        return True
    except Exception:
        return False


# Register health check
health_checker.register_health_check("nba_api", _check_nba_api_health)


# Utility functions for external use
async def get_live_games() -> List[GameInfo]:
    """Get current NBA games from this week and season (2024-25)."""
    fetcher = LiveDataFetcher()
    try:
        # Get real active games from NBA API
        return await fetcher.get_active_games()
    except Exception as e:
        logger.error(f"Failed to fetch real NBA games: {e}")
        # Return empty list instead of mock data
        return []


async def fetch_game_events(game_id: str) -> Tuple[GameInfo, List[GameEvent]]:
    """Convenience function to fetch game events."""
    fetcher = LiveDataFetcher()
    return await fetcher.fetch_live_game_data(game_id)