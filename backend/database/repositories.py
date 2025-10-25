"""Data access layer with repository pattern."""

import json
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func, text
from sqlalchemy.sql import select

from .models import Game, Event, TMICalculation
from backend.models.game_models import GameEvent, TeamMomentumIndex
from backend.utils.cache import get_cache, cached
from backend.utils.performance_monitor import get_metrics_collector, time_operation


class GameRepository:
    """Repository for game data operations."""

    def __init__(self, db: Session):
        self.db = db

    def create_game(self, game_id: str, home_team: str, away_team: str, 
                   game_date: str, status: str) -> Game:
        """Create a new game record."""
        db_game = Game(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            status=status
        )
        self.db.add(db_game)
        self.db.commit()
        self.db.refresh(db_game)
        return db_game

    def get_game(self, game_id: str) -> Optional[Game]:
        """Get a game by ID."""
        return self.db.query(Game).filter(Game.game_id == game_id).first()

    def get_games_by_date(self, game_date: str) -> List[Game]:
        """Get all games for a specific date."""
        return self.db.query(Game).filter(Game.game_date == game_date).all()

    def get_active_games(self) -> List[Game]:
        """Get all games with active status."""
        return self.db.query(Game).filter(Game.status.in_(["Live", "In Progress"])).all()

    def update_game_status(self, game_id: str, status: str) -> Optional[Game]:
        """Update game status."""
        db_game = self.get_game(game_id)
        if db_game:
            db_game.status = status
            self.db.commit()
            self.db.refresh(db_game)
        return db_game

    def delete_game(self, game_id: str) -> bool:
        """Delete a game and all related data."""
        db_game = self.get_game(game_id)
        if db_game:
            self.db.delete(db_game)
            self.db.commit()
            return True
        return False


class EventRepository:
    """Repository for game event data operations."""

    def __init__(self, db: Session):
        self.db = db

    def create_event(self, event: GameEvent) -> Event:
        """Create a new event record."""
        db_event = Event(
            event_id=event.event_id,
            game_id=event.game_id,
            team_tricode=event.team_tricode,
            player_name=event.player_name,
            event_type=event.event_type,
            clock=event.clock,
            period=event.period,
            points_total=event.points_total,
            shot_result=event.shot_result,
            description=event.description,
            timestamp=event.timestamp
        )
        self.db.add(db_event)
        self.db.commit()
        self.db.refresh(db_event)
        return db_event

    def create_events_batch(self, events: List[GameEvent]) -> List[Event]:
        """Create multiple events in a batch."""
        db_events = []
        for event in events:
            db_event = Event(
                event_id=event.event_id,
                game_id=event.game_id,
                team_tricode=event.team_tricode,
                player_name=event.player_name,
                event_type=event.event_type,
                clock=event.clock,
                period=event.period,
                points_total=event.points_total,
                shot_result=event.shot_result,
                description=event.description,
                timestamp=event.timestamp
            )
            db_events.append(db_event)
        
        self.db.add_all(db_events)
        self.db.commit()
        return db_events

    def get_event(self, event_id: str) -> Optional[Event]:
        """Get an event by ID."""
        return self.db.query(Event).filter(Event.event_id == event_id).first()

    def get_events_by_game(self, game_id: str, limit: Optional[int] = None) -> List[Event]:
        """Get all events for a game, optionally limited."""
        with time_operation("db_query_events_by_game"):
            # Use index on (game_id, timestamp) for optimal performance
            query = self.db.query(Event).filter(Event.game_id == game_id).order_by(Event.timestamp)
            if limit:
                query = query.limit(limit)
            return query.all()

    def get_events_by_team(self, game_id: str, team_tricode: str) -> List[Event]:
        """Get all events for a specific team in a game."""
        with time_operation("db_query_events_by_team"):
            # Use composite index on (game_id, team_tricode, timestamp)
            return self.db.query(Event).filter(
                and_(Event.game_id == game_id, Event.team_tricode == team_tricode)
            ).order_by(Event.timestamp).all()

    def get_recent_events(self, game_id: str, minutes: int = 5) -> List[Event]:
        """Get events from the last N minutes."""
        with time_operation("db_query_recent_events"):
            cutoff_time = datetime.now().timestamp() - (minutes * 60)
            # Use index on (game_id, timestamp) for time-based queries
            return self.db.query(Event).filter(
                and_(
                    Event.game_id == game_id,
                    Event.timestamp >= datetime.fromtimestamp(cutoff_time)
                )
            ).order_by(Event.timestamp).all()

    def event_exists(self, event_id: str) -> bool:
        """Check if an event already exists."""
        return self.db.query(Event).filter(Event.event_id == event_id).first() is not None

    def get_latest_event_timestamp(self, game_id: str) -> Optional[datetime]:
        """Get the timestamp of the most recent event for a game."""
        latest_event = self.db.query(Event).filter(Event.game_id == game_id).order_by(desc(Event.timestamp)).first()
        return latest_event.timestamp if latest_event else None


class TMIRepository:
    """Repository for TMI calculation data operations."""

    def __init__(self, db: Session):
        self.db = db

    def create_tmi_calculation(self, tmi: TeamMomentumIndex) -> TMICalculation:
        """Create a new TMI calculation record."""
        db_tmi = TMICalculation(
            game_id=tmi.game_id,
            team_tricode=tmi.team_tricode,
            tmi_value=tmi.tmi_value,
            feature_contributions=json.dumps(tmi.feature_contributions),
            prediction_probability=tmi.prediction_probability,
            confidence_score=tmi.confidence_score,
            rolling_window_size=tmi.rolling_window_size,
            calculated_at=tmi.timestamp
        )
        self.db.add(db_tmi)
        self.db.commit()
        self.db.refresh(db_tmi)
        return db_tmi

    def get_latest_tmi(self, game_id: str, team_tricode: str) -> Optional[TMICalculation]:
        """Get the most recent TMI calculation for a team."""
        # Check cache first
        cache = get_cache()
        cached_tmi = cache.get_momentum(game_id, team_tricode)
        if cached_tmi:
            return cached_tmi
        
        with time_operation("db_query_latest_tmi"):
            # Use composite index on (game_id, team_tricode, calculated_at)
            result = self.db.query(TMICalculation).filter(
                and_(
                    TMICalculation.game_id == game_id,
                    TMICalculation.team_tricode == team_tricode
                )
            ).order_by(desc(TMICalculation.calculated_at)).first()
            
            # Cache the result
            if result:
                cache.cache_momentum(game_id, team_tricode, result)
            
            return result

    def get_tmi_history(self, game_id: str, team_tricode: str, limit: int = 50) -> List[TMICalculation]:
        """Get TMI calculation history for a team."""
        with time_operation("db_query_tmi_history"):
            # Use composite index for efficient history queries
            return self.db.query(TMICalculation).filter(
                and_(
                    TMICalculation.game_id == game_id,
                    TMICalculation.team_tricode == team_tricode
                )
            ).order_by(desc(TMICalculation.calculated_at)).limit(limit).all()

    def get_game_tmi_history(self, game_id: str, limit: int = 100) -> List[TMICalculation]:
        """Get TMI calculation history for all teams in a game."""
        return self.db.query(TMICalculation).filter(
            TMICalculation.game_id == game_id
        ).order_by(desc(TMICalculation.calculated_at)).limit(limit).all()

    def delete_old_calculations(self, game_id: str, keep_latest: int = 100) -> int:
        """Delete old TMI calculations, keeping only the most recent ones."""
        # Get the timestamp of the Nth most recent calculation
        cutoff_calculation = self.db.query(TMICalculation).filter(
            TMICalculation.game_id == game_id
        ).order_by(desc(TMICalculation.calculated_at)).offset(keep_latest).first()
        
        if cutoff_calculation:
            deleted_count = self.db.query(TMICalculation).filter(
                and_(
                    TMICalculation.game_id == game_id,
                    TMICalculation.calculated_at < cutoff_calculation.calculated_at
                )
            ).delete()
            self.db.commit()
            return deleted_count
        return 0

    def to_domain_model(self, db_tmi: TMICalculation) -> TeamMomentumIndex:
        """Convert database model to domain model."""
        feature_contributions = json.loads(db_tmi.feature_contributions) if db_tmi.feature_contributions else {}
        
        return TeamMomentumIndex(
            game_id=db_tmi.game_id,
            team_tricode=db_tmi.team_tricode,
            timestamp=db_tmi.calculated_at,
            tmi_value=db_tmi.tmi_value,
            feature_contributions=feature_contributions,
            rolling_window_size=db_tmi.rolling_window_size or 5,
            prediction_probability=db_tmi.prediction_probability or 0.0,
            confidence_score=db_tmi.confidence_score or 0.0
        )