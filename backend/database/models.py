"""SQLAlchemy database models."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from .config import Base


class Game(Base):
    """Games table model."""
    __tablename__ = "games"

    game_id = Column(String, primary_key=True, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    game_date = Column(String, nullable=False)  # Store as string for NBA API compatibility
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    events = relationship("Event", back_populates="game", cascade="all, delete-orphan")
    tmi_calculations = relationship("TMICalculation", back_populates="game", cascade="all, delete-orphan")


class Event(Base):
    """Events table model."""
    __tablename__ = "events"

    event_id = Column(String, primary_key=True, index=True)
    game_id = Column(String, ForeignKey("games.game_id"), nullable=False, index=True)
    team_tricode = Column(String, nullable=False)
    player_name = Column(String, nullable=True)
    event_type = Column(String, nullable=False)
    clock = Column(String, nullable=False)
    period = Column(Integer, nullable=False)
    points_total = Column(Integer, default=0)
    shot_result = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=func.now())

    # Relationships
    game = relationship("Game", back_populates="events")
    
    # Performance indexes
    __table_args__ = (
        Index('idx_events_game_timestamp', 'game_id', 'timestamp'),
        Index('idx_events_game_team_timestamp', 'game_id', 'team_tricode', 'timestamp'),
        Index('idx_events_type_timestamp', 'event_type', 'timestamp'),
    )


class TMICalculation(Base):
    """TMI calculations table model."""
    __tablename__ = "tmi_calculations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    game_id = Column(String, ForeignKey("games.game_id"), nullable=False, index=True)
    team_tricode = Column(String, nullable=False)
    tmi_value = Column(Float, nullable=False)
    feature_contributions = Column(Text, nullable=True)  # JSON string
    prediction_probability = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    rolling_window_size = Column(Integer, nullable=True)
    calculated_at = Column(DateTime, default=func.now())

    # Relationships
    game = relationship("Game", back_populates="tmi_calculations")
    
    # Performance indexes
    __table_args__ = (
        Index('idx_tmi_game_team_calculated', 'game_id', 'team_tricode', 'calculated_at'),
        Index('idx_tmi_calculated_at', 'calculated_at'),
        Index('idx_tmi_game_calculated', 'game_id', 'calculated_at'),
    )