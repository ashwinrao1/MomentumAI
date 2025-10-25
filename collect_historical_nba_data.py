#!/usr/bin/env python3
"""
Historical NBA Data Collection Script

Collects 5 years of NBA play-by-play data for comprehensive model training.
This script handles large-scale data collection with proper rate limiting,
caching, and error recovery.
"""

import argparse
import logging
import time
import pickle
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# NBA API imports
from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from models.game_models import GameEvent, GameInfo
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HistoricalNBADataCollector:
    """
    Comprehensive NBA data collector for 5 years of historical data.
    
    Features:
    - Multi-season data collection
    - Robust error handling and recovery
    - Progress tracking and resumption
    - Data validation and cleaning
    - Efficient caching system
    """
    
    def __init__(self, rate_limit_delay: float = 0.6, cache_dir: str = "data/nba_cache"):
        """Initialize the historical data collector."""
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.cache_dir / "collection_progress.json"
        self.collected_games = set()
        self.failed_games = set()
        
        # Load existing progress
        self._load_progress()
        
        # Team data cache
        self.teams_data = {}
        self._load_teams_data()
    
    def _load_teams_data(self):
        """Load NBA teams data."""
        try:
            nba_teams = teams.get_teams()
            for team in nba_teams:
                self.teams_data[team['id']] = team
            logger.info(f"Loaded {len(self.teams_data)} NBA teams")
        except Exception as e:
            logger.error(f"Error loading teams data: {e}")
    
    def _load_progress(self):
        """Load collection progress from disk."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.collected_games = set(progress.get('collected_games', []))
                    self.failed_games = set(progress.get('failed_games', []))
                logger.info(f"Loaded progress: {len(self.collected_games)} collected, {len(self.failed_games)} failed")
            except Exception as e:
                logger.warning(f"Error loading progress: {e}")
    
    def _save_progress(self):
        """Save collection progress to disk."""
        try:
            progress = {
                'collected_games': list(self.collected_games),
                'failed_games': list(self.failed_games),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving progress: {e}")    

    def collect_games_for_seasons(
        self,
        seasons: List[str],
        max_games_per_season: int = 1000,
        season_types: List[str] = ["Regular Season", "Playoffs"]
    ) -> List[GameInfo]:
        """
        Collect games for multiple NBA seasons.
        
        Args:
            seasons: List of season strings (e.g., ["2019-20", "2020-21"])
            max_games_per_season: Maximum games per season
            season_types: Types of games to collect
            
        Returns:
            List of GameInfo objects
        """
        logger.info(f"Collecting games for seasons: {seasons}")
        
        all_games = []
        
        for season in seasons:
            logger.info(f"Processing season {season}")
            
            for season_type in season_types:
                try:
                    season_games = self._collect_season_games(
                        season, season_type, max_games_per_season
                    )
                    all_games.extend(season_games)
                    
                    logger.info(f"Collected {len(season_games)} games for {season} {season_type}")
                    
                    # Rate limiting between seasons
                    time.sleep(self.rate_limit_delay * 2)
                    
                except Exception as e:
                    logger.error(f"Error collecting {season} {season_type}: {e}")
                    continue
        
        # Remove duplicates
        unique_games = {}
        for game in all_games:
            if game.game_id not in unique_games:
                unique_games[game.game_id] = game
        
        result = list(unique_games.values())
        logger.info(f"Total unique games collected: {len(result)}")
        
        return result
    
    def _collect_season_games(
        self,
        season: str,
        season_type: str,
        max_games: int
    ) -> List[GameInfo]:
        """Collect games for a specific season."""
        cache_file = self.cache_dir / f"games_{season}_{season_type.replace(' ', '_')}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_games = pickle.load(f)
                logger.info(f"Loaded {len(cached_games)} games from cache for {season} {season_type}")
                return cached_games
            except Exception as e:
                logger.warning(f"Error loading cache for {season}: {e}")
        
        games = []
        
        try:
            # Use LeagueGameFinder to get games
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type
            )
            
            games_df = gamefinder.get_data_frames()[0]
            
            # Process games
            processed_games = set()  # Track to avoid duplicates
            
            for _, row in games_df.iterrows():
                try:
                    game_id = row['GAME_ID']
                    
                    # Skip if already processed
                    if game_id in processed_games:
                        continue
                    
                    processed_games.add(game_id)
                    
                    # Parse game information
                    game_info = self._parse_game_row(row, season)
                    if game_info:
                        games.append(game_info)
                    
                    if len(games) >= max_games:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error parsing game row: {e}")
                    continue
            
            # Cache the results
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(games, f)
                logger.info(f"Cached {len(games)} games for {season} {season_type}")
            except Exception as e:
                logger.warning(f"Error caching games: {e}")
            
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.error(f"Error collecting season games: {e}")
        
        return games
    
    def _parse_game_row(self, row: Any, season: str) -> Optional[GameInfo]:
        """Parse a game row into GameInfo object."""
        try:
            game_id = row['GAME_ID']
            matchup = row['MATCHUP']
            game_date = row['GAME_DATE']
            
            # Parse matchup to get teams
            if ' vs. ' in matchup:
                home_team = matchup.split(' vs. ')[1]
                away_team = matchup.split(' vs. ')[0]
            elif ' @ ' in matchup:
                away_team = matchup.split(' @ ')[0]
                home_team = matchup.split(' @ ')[1]
            else:
                logger.warning(f"Could not parse matchup: {matchup}")
                return None
            
            return GameInfo(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                status="Final",  # Historical games are completed
                period=4,  # Default to regulation
                clock="00:00",
                home_score=0,  # Will be filled from play-by-play
                away_score=0
            )
            
        except Exception as e:
            logger.warning(f"Error parsing game row: {e}")
            return None   
 
    def collect_play_by_play_data(
        self,
        games: List[GameInfo],
        max_games: Optional[int] = None
    ) -> List[GameEvent]:
        """
        Collect play-by-play data for a list of games.
        
        Args:
            games: List of GameInfo objects
            max_games: Maximum number of games to process
            
        Returns:
            List of GameEvent objects
        """
        if max_games:
            games = games[:max_games]
        
        logger.info(f"Collecting play-by-play data for {len(games)} games")
        
        all_events = []
        processed_count = 0
        
        for i, game in enumerate(games):
            # Skip if already collected
            if game.game_id in self.collected_games:
                logger.debug(f"Skipping already collected game: {game.game_id}")
                continue
            
            # Skip if previously failed
            if game.game_id in self.failed_games:
                logger.debug(f"Skipping previously failed game: {game.game_id}")
                continue
            
            try:
                logger.info(f"Processing game {i+1}/{len(games)}: {game.game_id}")
                
                # Collect events for this game
                game_events = self._collect_game_play_by_play(game)
                
                if game_events:
                    all_events.extend(game_events)
                    self.collected_games.add(game.game_id)
                    processed_count += 1
                    
                    logger.info(f"Collected {len(game_events)} events from game {game.game_id}")
                else:
                    self.failed_games.add(game.game_id)
                    logger.warning(f"No events collected for game {game.game_id}")
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(games)} games processed, {len(all_events)} total events")
                    self._save_progress()
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error processing game {game.game_id}: {e}")
                self.failed_games.add(game.game_id)
                continue
        
        # Final progress save
        self._save_progress()
        
        logger.info(f"Play-by-play collection complete: {len(all_events)} events from {processed_count} games")
        return all_events
    
    def _collect_game_play_by_play(self, game: GameInfo) -> List[GameEvent]:
        """Collect play-by-play events for a single game."""
        cache_file = self.cache_dir / f"pbp_{game.game_id}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_events = pickle.load(f)
                return cached_events
            except Exception as e:
                logger.warning(f"Error loading cached events for {game.game_id}: {e}")
        
        try:
            # Get play-by-play data
            pbp = playbyplayv2.PlayByPlayV2(game_id=game.game_id)
            pbp_data = pbp.get_data_frames()[0]
            
            # Get box score for additional context
            try:
                box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game.game_id)
                team_stats = box_score.team_stats.get_dict()['data']
                game_context = self._extract_game_context(team_stats, game.game_id)
            except Exception as e:
                logger.warning(f"Could not get box score for {game.game_id}: {e}")
                game_context = {}
            
            # Process events
            events = []
            for _, row in pbp_data.iterrows():
                try:
                    event = self._parse_play_by_play_row(row, game.game_id, game_context)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.debug(f"Error parsing play-by-play row: {e}")
                    continue
            
            # Cache the events
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(events, f)
            except Exception as e:
                logger.warning(f"Error caching events for {game.game_id}: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"Error collecting play-by-play for {game.game_id}: {e}")
            return []
    
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
                
                # Pace estimate
                team1_fga = team1.get('FGA', 0)
                team1_fta = team1.get('FTA', 0)
                team1_tov = team1.get('TOV', 0)
                team1_oreb = team1.get('OREB', 0)
                
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
  
    def _parse_play_by_play_row(
        self,
        row: Any,
        game_id: str,
        game_context: Dict[str, Any]
    ) -> Optional[GameEvent]:
        """Parse a play-by-play row into a GameEvent."""
        try:
            # Basic event information
            event_id = f"{game_id}_{row.get('EVENTNUM', 0)}"
            period = int(row.get('PERIOD', 1))
            clock = str(row.get('PCTIMESTRING', '12:00'))
            
            # Enhanced description parsing
            home_desc = str(row.get('HOMEDESCRIPTION', '') or '')
            away_desc = str(row.get('VISITORDESCRIPTION', '') or '')
            description = home_desc if home_desc else away_desc
            
            if not description or description == 'nan':
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
            
            # Calculate game time remaining
            time_remaining = self._calculate_time_remaining(period, clock)
            
            # Extract score information
            score_info = self._parse_score(row.get('SCORE', '0-0'))
            
            # Create enhanced GameEvent
            event = GameEvent(
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
                description=description
            )
            
            # Add enhanced attributes
            event.event_value = event_value
            event.time_remaining = time_remaining
            event.score_margin = score_info['margin']
            event.game_context = game_context
            
            return event
            
        except Exception as e:
            logger.debug(f"Error parsing play-by-play row: {e}")
            return None
    
    def _classify_enhanced_event(self, description: str, row: Any) -> tuple:
        """Enhanced event classification with value scoring."""
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
            if ':' in clock:
                minutes, seconds = map(int, clock.split(':'))
                period_time_remaining = minutes + seconds / 60.0
            else:
                period_time_remaining = 0.0
            
            if period <= 4:
                periods_remaining = 4 - period
                total_remaining = periods_remaining * 12 + period_time_remaining
            else:
                total_remaining = period_time_remaining
            
            return max(0.0, total_remaining)
            
        except Exception:
            return 0.0
    
    def _parse_score(self, score_str: str) -> Dict[str, Any]:
        """Parse score string and calculate margin."""
        try:
            if '-' in score_str and score_str != '0-0':
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
    
    def save_dataset(self, events: List[GameEvent], filename: str):
        """Save the collected dataset."""
        output_file = self.cache_dir / filename
        
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(events, f)
            
            logger.info(f"Dataset saved to {output_file}")
            logger.info(f"Total events: {len(events)}")
            
            # Save summary statistics
            self._save_dataset_summary(events, filename)
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
    
    def _save_dataset_summary(self, events: List[GameEvent], filename: str):
        """Save dataset summary statistics."""
        try:
            # Calculate statistics
            games = set(event.game_id for event in events)
            teams = set(event.team_tricode for event in events)
            event_types = {}
            
            for event in events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            
            summary = {
                'filename': filename,
                'total_events': len(events),
                'total_games': len(games),
                'total_teams': len(teams),
                'event_type_distribution': event_types,
                'collection_date': datetime.now().isoformat(),
                'teams': sorted(teams),
                'sample_games': sorted(list(games))[:10]
            }
            
            summary_file = self.cache_dir / f"{filename}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Dataset summary saved to {summary_file}")
            
        except Exception as e:
            logger.warning(f"Error saving dataset summary: {e}")


def main():
    """Main data collection script."""
    parser = argparse.ArgumentParser(description="Collect 5 years of NBA historical data")
    
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],
        help="NBA seasons to collect (default: last 5 seasons)"
    )
    parser.add_argument(
        "--max-games-per-season",
        type=int,
        default=1000,
        help="Maximum games per season (default: 1000)"
    )
    parser.add_argument(
        "--max-total-games",
        type=int,
        default=2000,
        help="Maximum total games to process for play-by-play (default: 2000)"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="nba_5year_dataset.pkl",
        help="Output filename for the dataset"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/nba_cache",
        help="Directory for caching data"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.6,
        help="Rate limit delay between API calls (seconds)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting 5-year NBA data collection")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Max games per season: {args.max_games_per_season}")
    logger.info(f"Max total games for play-by-play: {args.max_total_games}")
    
    try:
        # Create collector
        collector = HistoricalNBADataCollector(
            rate_limit_delay=args.rate_limit,
            cache_dir=args.cache_dir
        )
        
        # Step 1: Collect game information
        logger.info("Step 1: Collecting game information...")
        games = collector.collect_games_for_seasons(
            seasons=args.seasons,
            max_games_per_season=args.max_games_per_season
        )
        
        if not games:
            logger.error("No games collected. Exiting.")
            return 1
        
        logger.info(f"Collected information for {len(games)} games")
        
        # Step 2: Collect play-by-play data
        logger.info("Step 2: Collecting play-by-play data...")
        events = collector.collect_play_by_play_data(
            games=games,
            max_games=args.max_total_games
        )
        
        if not events:
            logger.error("No play-by-play events collected. Exiting.")
            return 1
        
        # Step 3: Save dataset
        logger.info("Step 3: Saving dataset...")
        collector.save_dataset(events, args.output_filename)
        
        logger.info("=" * 60)
        logger.info("NBA DATA COLLECTION COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Total events collected: {len(events):,}")
        logger.info(f"Total games processed: {len(set(e.game_id for e in events))}")
        logger.info(f"Dataset saved to: {collector.cache_dir / args.output_filename}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())