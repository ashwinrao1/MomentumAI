from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel


@dataclass
class GameInfo:
    """Basic game information structure"""
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    status: str
    period: int
    clock: str
    home_score: int
    away_score: int


@dataclass
class GameEvent:
    """Represents a single game event from NBA play-by-play data"""
    event_id: str
    game_id: str
    team_tricode: str
    player_name: Optional[str]
    event_type: str  # shot, rebound, turnover, foul
    clock: str
    period: int
    points_total: int
    shot_result: Optional[str]  # Made, Missed
    timestamp: datetime
    description: str


@dataclass
class Possession:
    """Represents a team possession with aggregated events"""
    possession_id: str
    game_id: str
    team_tricode: str
    start_time: str
    end_time: str
    events: List[GameEvent]
    points_scored: int
    fg_attempts: int
    fg_made: int
    turnovers: int
    rebounds: int
    fouls: int


@dataclass
class TeamMomentumIndex:
    """Team Momentum Index calculation result"""
    game_id: str
    team_tricode: str
    timestamp: datetime
    tmi_value: float
    feature_contributions: Dict[str, float]
    rolling_window_size: int
    prediction_probability: float
    confidence_score: float


# Pydantic models for API serialization
class GameEventResponse(BaseModel):
    event_id: str
    game_id: str
    team_tricode: str
    player_name: Optional[str]
    event_type: str
    clock: str
    period: int
    points_total: int
    shot_result: Optional[str]
    timestamp: str


class PossessionResponse(BaseModel):
    possession_id: str
    game_id: str
    team_tricode: str
    start_time: str
    end_time: str
    points_scored: int
    fg_attempts: int
    fg_made: int
    turnovers: int
    rebounds: int
    fouls: int


class TMIResponse(BaseModel):
    game_id: str
    team_tricode: str
    timestamp: str
    tmi_value: float
    feature_contributions: Dict[str, float]
    rolling_window_size: int
    prediction_probability: float
    confidence_score: float


class GameSelectionResponse(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    status: str


class MomentumUpdateMessage(BaseModel):
    """WebSocket message for momentum updates"""
    type: str = "momentum_update"
    game_id: str
    timestamp: str
    game_status: str
    teams: Dict[str, TMIResponse]
    event_count: int


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure"""
    type: str
    timestamp: str
    message: Optional[str] = None


class SubscriptionMessage(BaseModel):
    """WebSocket subscription message"""
    action: str  # subscribe, unsubscribe, ping
    game_id: Optional[str] = None