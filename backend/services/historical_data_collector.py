"""
Historical data collection service for ML model training.

This module provides functionality to collect historical NBA play-by-play data
for training the momentum prediction model.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder
from nba_api.stats.static import teams

from backend.models.game_models import GameEvent, GameInfo

# Configure logging
logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """
    Collects historical NBA play-by-play data for model training.
    
    Uses the NBA API to fetch completed games and their play-by-play events
    for training the momentum prediction model.
    """
    
    def __init__(self, rate_limit_delay: float = 0.6):
        """
        Initialize the historical data collector.
        
        Args:
            rate_limit_delay: Delay between API calls to respect rate limits
        """
        self.rate_limit_delay = rate_limit_delay
        self.collected_games = set()
    
    def collect_recent_games(
        self,
        days_back: int = 30,
        max_games: int = 200,
        season: str = "2023-24"
    ) -> List[GameInfo]:
        """
        Collect information about recent completed games.
        
        Args:
            days_back: Number of days back to look for games
            max_games: Maximum number of games to collect
            season: NBA season (e.g., "2023-24")
            
        Returns:
            List of GameInfo objects for completed games
        """
        logger.info(f"Collecting recent games from last {days_back} days")
        
        try:
            # Get all teams
            nba_teams = teams.get_teams()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            games = []
            
            # Collect games for each team (will have duplicates)
            for team in nba_teams[:5]:  # Limit to first 5 teams for demo
                try:
                    team_games = self._get_team_games(
                        team['id'], 
                        start_date, 
                        end_date, 
                        season
                    )
                    games.extend(team_games)
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                    if len(games) >= max_games:
                        break
                        
                except Exception as e:
                    logger.error(f"Error collecting games for team {team['full_name']}: {e}")
                    continue
            
            # Remove duplicates and filter completed games
            unique_games = {}
            for game in games:
                if game.game_id not in unique_games and game.status == "Final":
                    unique_games[game.game_id] = game
            
            result = list(unique_games.values())[:max_games]
            logger.info(f"Collected {len(result)} unique completed games")
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting recent games: {e}")
            return []
    
    def collect_game_events(self, game_id: str) -> List[GameEvent]:
        """
        Collect play-by-play events for a specific game.
        
        Args:
            game_id: NBA game ID
            
        Returns:
            List of GameEvent objects
        """
        if game_id in self.collected_games:
            logger.info(f"Game {game_id} already collected, skipping")
            return []
        
        try:
            logger.info(f"Collecting play-by-play data for game {game_id}")
            
            # Get play-by-play data
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            pbp_data = pbp.get_data_frames()[0]  # First DataFrame contains play-by-play
            
            events = []
            
            for _, row in pbp_data.iterrows():
                try:
                    event = self._parse_play_by_play_row(row, game_id)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Error parsing play-by-play row: {e}")
                    continue
            
            self.collected_games.add(game_id)
            logger.info(f"Collected {len(events)} events for game {game_id}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return events
            
        except Exception as e:
            logger.error(f"Error collecting events for game {game_id}: {e}")
            return []
    
    def collect_historical_dataset(
        self,
        days_back: int = 30,
        max_games: int = 200,
        season: str = "2023-24"
    ) -> List[GameEvent]:
        """
        Collect a complete historical dataset for model training.
        
        Args:
            days_back: Number of days back to look for games
            max_games: Maximum number of games to collect
            season: NBA season
            
        Returns:
            List of all collected GameEvent objects
        """
        logger.info("Starting historical dataset collection")
        
        # Get list of games
        games = self.collect_recent_games(days_back, max_games, season)
        
        if not games:
            logger.error("No games found for historical data collection")
            return []
        
        # Collect events for each game
        all_events = []
        
        for i, game in enumerate(games):
            logger.info(f"Processing game {i+1}/{len(games)}: {game.game_id}")
            
            try:
                game_events = self.collect_game_events(game.game_id)
                all_events.extend(game_events)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1} games, collected {len(all_events)} total events")
                
            except Exception as e:
                logger.error(f"Error processing game {game.game_id}: {e}")
                continue
        
        logger.info(f"Historical data collection complete: {len(all_events)} events from {len(games)} games")
        return all_events
    
    def _get_team_games(
        self,
        team_id: int,
        start_date: datetime,
        end_date: datetime,
        season: str
    ) -> List[GameInfo]:
        """
        Get games for a specific team in a date range.
        
        Args:
            team_id: NBA team ID
            start_date: Start date for game search
            end_date: End date for game search
            season: NBA season
            
        Returns:
            List of GameInfo objects
        """
        try:
            # Use LeagueGameFinder to get games
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable="Regular Season"
            )
            
            games_df = gamefinder.get_data_frames()[0]
            
            games = []
            for _, row in games_df.iterrows():
                try:
                    # Parse game date
                    game_date = datetime.strptime(row['GAME_DATE'], '%Y-%m-%d')
                    
                    # Filter by date range
                    if start_date <= game_date <= end_date:
                        game_info = GameInfo(
                            game_id=row['GAME_ID'],
                            home_team=row['MATCHUP'].split(' vs. ')[-1] if ' vs. ' in row['MATCHUP'] else row['MATCHUP'].split(' @ ')[0],
                            away_team=row['MATCHUP'].split(' @ ')[-1] if ' @ ' in row['MATCHUP'] else row['MATCHUP'].split(' vs. ')[0],
                            game_date=row['GAME_DATE'],
                            status="Final",  # Historical games are completed
                            period=4,  # Assume regulation game
                            clock="00:00",
                            home_score=0,  # Will be filled from play-by-play
                            away_score=0
                        )
                        games.append(game_info)
                        
                except Exception as e:
                    logger.warning(f"Error parsing game row: {e}")
                    continue
            
            return games
            
        except Exception as e:
            logger.error(f"Error getting team games: {e}")
            return []
    
    def _parse_play_by_play_row(self, row: Any, game_id: str) -> Optional[GameEvent]:
        """
        Parse a single play-by-play row into a GameEvent.
        
        Args:
            row: Pandas row from play-by-play data
            game_id: Game ID
            
        Returns:
            GameEvent object or None if parsing fails
        """
        try:
            # Extract basic information
            event_id = f"{game_id}_{row.get('EVENTNUM', 0)}"
            period = int(row.get('PERIOD', 1))
            clock = str(row.get('PCTIMESTRING', '12:00'))
            description = str(row.get('HOMEDESCRIPTION', '') or row.get('VISITORDESCRIPTION', '') or '')
            
            # Determine team
            team_tricode = None
            if row.get('PLAYER1_TEAM_ABBREVIATION'):
                team_tricode = row.get('PLAYER1_TEAM_ABBREVIATION')
            elif 'HOME' in description:
                team_tricode = "HOME"  # Will need to map to actual team
            elif 'AWAY' in description:
                team_tricode = "AWAY"  # Will need to map to actual team
            
            if not team_tricode:
                return None
            
            # Determine event type and details
            event_type, shot_result = self._classify_event(description)
            
            # Extract player name
            player_name = row.get('PLAYER1_NAME') or self._extract_player_from_description(description)
            
            # Calculate points total (simplified)
            points_total = int(row.get('SCORE', '0-0').split('-')[0]) if row.get('SCORE') else 0
            
            return GameEvent(
                event_id=event_id,
                game_id=game_id,
                team_tricode=team_tricode,
                player_name=player_name,
                event_type=event_type,
                clock=clock,
                period=period,
                points_total=points_total,
                shot_result=shot_result,
                timestamp=datetime.utcnow(),
                description=description
            )
            
        except Exception as e:
            logger.warning(f"Error parsing play-by-play row: {e}")
            return None
    
    def _classify_event(self, description: str) -> tuple[str, Optional[str]]:
        """
        Classify event type from description.
        
        Args:
            description: Play description
            
        Returns:
            Tuple of (event_type, shot_result)
        """
        description_lower = description.lower()
        
        if 'makes' in description_lower or 'made' in description_lower:
            return 'shot', 'Made'
        elif 'misses' in description_lower or 'missed' in description_lower:
            return 'shot', 'Missed'
        elif 'rebound' in description_lower:
            return 'rebound', None
        elif 'turnover' in description_lower:
            return 'turnover', None
        elif 'foul' in description_lower:
            return 'foul', None
        elif 'free throw' in description_lower:
            if 'makes' in description_lower:
                return 'shot', 'Made'
            else:
                return 'shot', 'Missed'
        else:
            return 'other', None
    
    def _extract_player_from_description(self, description: str) -> Optional[str]:
        """
        Extract player name from play description.
        
        Args:
            description: Play description
            
        Returns:
            Player name or None
        """
        # Simple extraction - look for patterns like "PLAYER makes..."
        words = description.split()
        if len(words) >= 2:
            # Assume first word(s) before action verb is player name
            for i, word in enumerate(words):
                if word.lower() in ['makes', 'misses', 'rebound', 'turnover', 'foul']:
                    if i > 0:
                        return ' '.join(words[:i])
                    break
        
        return None


# Utility functions
def collect_training_data(
    days_back: int = 30,
    max_games: int = 200,
    season: str = "2023-24"
) -> List[GameEvent]:
    """
    Collect historical training data.
    
    Args:
        days_back: Number of days back to collect
        max_games: Maximum number of games
        season: NBA season
        
    Returns:
        List of GameEvent objects
    """
    collector = HistoricalDataCollector()
    return collector.collect_historical_dataset(days_back, max_games, season)


def create_sample_training_data() -> List[GameEvent]:
    """
    Create sample training data for testing when NBA API is not available.
    
    Returns:
        List of sample GameEvent objects
    """
    logger.info("Creating sample training data for testing")
    
    sample_events = []
    
    # Create sample events for multiple games
    for game_num in range(10):  # 10 sample games
        game_id = f"sample_game_{game_num:03d}"
        
        # Create events for each team
        for team_idx, team in enumerate(['LAL', 'GSW']):
            for possession in range(20):  # 20 possessions per team
                for event_num in range(3):  # 3 events per possession
                    event_id = f"{game_id}_{team}_{possession}_{event_num}"
                    
                    # Vary event types
                    if event_num == 0:
                        event_type = 'shot'
                        shot_result = 'Made' if possession % 3 == 0 else 'Missed'
                    elif event_num == 1:
                        event_type = 'rebound'
                        shot_result = None
                    else:
                        event_type = 'turnover' if possession % 5 == 0 else 'foul'
                        shot_result = None
                    
                    event = GameEvent(
                        event_id=event_id,
                        game_id=game_id,
                        team_tricode=team,
                        player_name=f"Player_{possession % 5}",
                        event_type=event_type,
                        clock=f"{11 - (possession // 4)}:{(60 - possession * 3) % 60:02d}",
                        period=1 + (possession // 20),
                        points_total=possession * 2,
                        shot_result=shot_result,
                        timestamp=datetime.utcnow(),
                        description=f"{team} {event_type} event"
                    )
                    
                    sample_events.append(event)
    
    logger.info(f"Created {len(sample_events)} sample training events")
    return sample_events