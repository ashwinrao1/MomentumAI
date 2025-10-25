"""High-level database service layer."""

import logging
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy.orm import Session

from .repositories import GameRepository, EventRepository, TMIRepository
from .config import get_database, SessionLocal
from backend.models.game_models import GameEvent, TeamMomentumIndex, GameInfo

logger = logging.getLogger(__name__)


class DatabaseService:
    """High-level database service providing business logic operations."""

    def __init__(self, db: Session = None):
        self.db = db or next(get_database())
        self.game_repo = GameRepository(self.db)
        self.event_repo = EventRepository(self.db)
        self.tmi_repo = TMIRepository(self.db)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            self.db.close()

    # Game operations
    def create_or_update_game(self, game_info: GameInfo) -> bool:
        """Create a new game or update existing game status."""
        try:
            existing_game = self.game_repo.get_game(game_info.game_id)
            
            if existing_game:
                # Update existing game status
                self.game_repo.update_game_status(game_info.game_id, game_info.status)
                logger.info(f"Updated game {game_info.game_id} status to {game_info.status}")
            else:
                # Create new game
                self.game_repo.create_game(
                    game_id=game_info.game_id,
                    home_team=game_info.home_team,
                    away_team=game_info.away_team,
                    game_date=game_info.game_date,
                    status=game_info.status
                )
                logger.info(f"Created new game {game_info.game_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create/update game {game_info.game_id}: {str(e)}")
            return False

    def get_active_games(self) -> List[GameInfo]:
        """Get all currently active games."""
        try:
            db_games = self.game_repo.get_active_games()
            return [
                GameInfo(
                    game_id=game.game_id,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    game_date=game.game_date,
                    status=game.status,
                    period=0,  # Will be updated from live data
                    clock="",  # Will be updated from live data
                    home_score=0,  # Will be updated from live data
                    away_score=0   # Will be updated from live data
                )
                for game in db_games
            ]
        except Exception as e:
            logger.error(f"Failed to get active games: {str(e)}")
            return []

    def get_games_by_date(self, game_date: str) -> List[GameInfo]:
        """Get all games for a specific date."""
        try:
            db_games = self.game_repo.get_games_by_date(game_date)
            return [
                GameInfo(
                    game_id=game.game_id,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    game_date=game.game_date,
                    status=game.status,
                    period=0,
                    clock="",
                    home_score=0,
                    away_score=0
                )
                for game in db_games
            ]
        except Exception as e:
            logger.error(f"Failed to get games for date {game_date}: {str(e)}")
            return []

    # Event operations
    def store_events(self, events: List[GameEvent]) -> int:
        """Store multiple events, skipping duplicates."""
        try:
            new_events = []
            for event in events:
                if not self.event_repo.event_exists(event.event_id):
                    new_events.append(event)
            
            if new_events:
                self.event_repo.create_events_batch(new_events)
                logger.info(f"Stored {len(new_events)} new events")
            
            return len(new_events)
        except Exception as e:
            logger.error(f"Failed to store events: {str(e)}")
            return 0

    def get_game_events(self, game_id: str, team_tricode: Optional[str] = None) -> List[GameEvent]:
        """Get events for a game, optionally filtered by team."""
        try:
            if team_tricode:
                db_events = self.event_repo.get_events_by_team(game_id, team_tricode)
            else:
                db_events = self.event_repo.get_events_by_game(game_id)
            
            return [
                GameEvent(
                    event_id=event.event_id,
                    game_id=event.game_id,
                    team_tricode=event.team_tricode,
                    player_name=event.player_name,
                    event_type=event.event_type,
                    clock=event.clock,
                    period=event.period,
                    points_total=event.points_total,
                    shot_result=event.shot_result,
                    timestamp=event.timestamp,
                    description=event.description or ""
                )
                for event in db_events
            ]
        except Exception as e:
            logger.error(f"Failed to get events for game {game_id}: {str(e)}")
            return []

    def get_recent_events(self, game_id: str, minutes: int = 5) -> List[GameEvent]:
        """Get events from the last N minutes."""
        try:
            db_events = self.event_repo.get_recent_events(game_id, minutes)
            return [
                GameEvent(
                    event_id=event.event_id,
                    game_id=event.game_id,
                    team_tricode=event.team_tricode,
                    player_name=event.player_name,
                    event_type=event.event_type,
                    clock=event.clock,
                    period=event.period,
                    points_total=event.points_total,
                    shot_result=event.shot_result,
                    timestamp=event.timestamp,
                    description=event.description or ""
                )
                for event in db_events
            ]
        except Exception as e:
            logger.error(f"Failed to get recent events for game {game_id}: {str(e)}")
            return []

    def get_latest_event_timestamp(self, game_id: str) -> Optional[datetime]:
        """Get the timestamp of the most recent event for a game."""
        try:
            return self.event_repo.get_latest_event_timestamp(game_id)
        except Exception as e:
            logger.error(f"Failed to get latest event timestamp for game {game_id}: {str(e)}")
            return None

    # TMI operations
    def store_tmi_calculation(self, tmi: TeamMomentumIndex) -> bool:
        """Store a TMI calculation."""
        try:
            self.tmi_repo.create_tmi_calculation(tmi)
            logger.info(f"Stored TMI calculation for {tmi.team_tricode} in game {tmi.game_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store TMI calculation: {str(e)}")
            return False

    def get_latest_tmi(self, game_id: str, team_tricode: str) -> Optional[TeamMomentumIndex]:
        """Get the most recent TMI calculation for a team."""
        try:
            db_tmi = self.tmi_repo.get_latest_tmi(game_id, team_tricode)
            if db_tmi:
                return self.tmi_repo.to_domain_model(db_tmi)
            return None
        except Exception as e:
            logger.error(f"Failed to get latest TMI for {team_tricode} in game {game_id}: {str(e)}")
            return None

    def get_tmi_history(self, game_id: str, team_tricode: str, limit: int = 50) -> List[TeamMomentumIndex]:
        """Get TMI calculation history for a team."""
        try:
            db_tmis = self.tmi_repo.get_tmi_history(game_id, team_tricode, limit)
            return [self.tmi_repo.to_domain_model(db_tmi) for db_tmi in db_tmis]
        except Exception as e:
            logger.error(f"Failed to get TMI history for {team_tricode} in game {game_id}: {str(e)}")
            return []

    def get_game_tmi_summary(self, game_id: str) -> Dict[str, Optional[TeamMomentumIndex]]:
        """Get latest TMI for both teams in a game."""
        try:
            # Get all recent TMI calculations for the game
            db_tmis = self.tmi_repo.get_game_tmi_history(game_id, limit=10)
            
            # Group by team and get the latest for each
            team_tmis = {}
            for db_tmi in db_tmis:
                if db_tmi.team_tricode not in team_tmis:
                    team_tmis[db_tmi.team_tricode] = self.tmi_repo.to_domain_model(db_tmi)
            
            return team_tmis
        except Exception as e:
            logger.error(f"Failed to get TMI summary for game {game_id}: {str(e)}")
            return {}

    def cleanup_old_data(self, game_id: str, keep_events: int = 1000, keep_tmi: int = 100) -> Dict[str, int]:
        """Clean up old data to prevent database bloat."""
        try:
            # Clean up old TMI calculations
            tmi_deleted = self.tmi_repo.delete_old_calculations(game_id, keep_tmi)
            
            # Note: We don't delete events as they're needed for historical analysis
            # In a production system, you might archive old events to a separate table
            
            logger.info(f"Cleaned up {tmi_deleted} old TMI calculations for game {game_id}")
            
            return {
                "tmi_deleted": tmi_deleted,
                "events_deleted": 0  # Not implemented for safety
            }
        except Exception as e:
            logger.error(f"Failed to cleanup old data for game {game_id}: {str(e)}")
            return {"tmi_deleted": 0, "events_deleted": 0}


# Convenience function for getting a database service instance
def get_database_service() -> DatabaseService:
    """Get a new database service instance."""
    return DatabaseService()