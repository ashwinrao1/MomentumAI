"""Database management API endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.config import get_database
from database.service import DatabaseService
from models.game_models import GameSelectionResponse, GameEventResponse

router = APIRouter(prefix="/database", tags=["database"])


@router.get("/games", response_model=List[GameSelectionResponse])
async def get_active_games(db: Session = Depends(get_database)):
    """Get all active games from database."""
    try:
        db_service = DatabaseService(db)
        games = db_service.get_active_games()
        
        return [
            GameSelectionResponse(
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                game_date=game.game_date,
                status=game.status
            )
            for game in games
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve games: {str(e)}")


@router.get("/games/{game_id}/events", response_model=List[GameEventResponse])
async def get_game_events(game_id: str, db: Session = Depends(get_database)):
    """Get all events for a specific game."""
    try:
        db_service = DatabaseService(db)
        events = db_service.get_game_events(game_id)
        
        return [
            GameEventResponse(
                event_id=event.event_id,
                game_id=event.game_id,
                team_tricode=event.team_tricode,
                player_name=event.player_name,
                event_type=event.event_type,
                clock=event.clock,
                period=event.period,
                points_total=event.points_total,
                shot_result=event.shot_result,
                timestamp=event.timestamp.isoformat()
            )
            for event in events
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve events: {str(e)}")


@router.get("/health")
async def database_health():
    """Check database health."""
    try:
        from database.migrations import check_database_health
        is_healthy = check_database_health()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "database": "SQLite",
            "tables": ["games", "events", "tmi_calculations"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")