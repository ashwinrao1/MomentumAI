"""
Momentum-specific API endpoints for MomentumML.

This module provides REST API endpoints for momentum calculations,
predictions, and real-time momentum data access.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.services.momentum_engine import MomentumEngine, create_momentum_engine
from backend.services.ml_predictor import MomentumPredictor
from backend.services.production_momentum_predictor import get_production_predictor
from backend.services.live_fetcher import fetch_game_events
from backend.services.realtime_pipeline import get_cached_game_momentum, force_game_update
from backend.models.game_models import TMIResponse
from backend.utils.error_handling import (
    APIError, DataProcessingError, MLModelError, NetworkError,
    log_error, ErrorSeverity, graceful_degradation, health_checker,
    safe_execute
)
from backend.utils.cache import get_cache
from backend.utils.performance_monitor import get_metrics_collector, time_operation

# Configure logging
logger = logging.getLogger("momentum_ml.api.momentum")

# Create router
router = APIRouter(prefix="/api/momentum", tags=["momentum"])

# Global momentum engine instance
momentum_engine: Optional[MomentumEngine] = None
ml_predictor = MomentumPredictor()
production_predictor = get_production_predictor()


def get_momentum_engine() -> MomentumEngine:
    """Get or create the global momentum engine instance."""
    global momentum_engine
    if momentum_engine is None:
        momentum_engine = create_momentum_engine(
            rolling_window_size=5,
            enable_ml_prediction=True
        )
        logger.info("Momentum engine initialized")
    return momentum_engine


# Request/Response models
class MomentumCurrentRequest(BaseModel):
    """Request model for current momentum endpoint."""
    game_id: str = Field(..., description="NBA game ID")
    team_tricode: Optional[str] = Field(None, description="Specific team code (optional)")


class MomentumCurrentResponse(BaseModel):
    """Response model for current momentum data."""
    game_id: str
    timestamp: str
    teams: Dict[str, TMIResponse]
    game_status: str
    last_updated: str


class MomentumPredictRequest(BaseModel):
    """Request model for momentum prediction endpoint."""
    game_id: str = Field(..., description="NBA game ID")
    team_tricode: str = Field(..., description="Team code for prediction")


class MomentumPredictResponse(BaseModel):
    """Response model for momentum prediction."""
    game_id: str
    team_tricode: str
    prediction_probability: float = Field(..., description="Probability momentum continues (0-1)")
    confidence_score: float = Field(..., description="Prediction confidence (0-1)")
    prediction_class: str = Field(..., description="'continue' or 'shift'")
    current_tmi: float = Field(..., description="Current TMI value")
    feature_contributions: Dict[str, float]
    timestamp: str


class GameSelectionResponse(BaseModel):
    """Response model for game selection."""
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    status: str
    home_score: int = 0
    away_score: int = 0


@router.get("/current", response_model=MomentumCurrentResponse)
async def get_current_momentum(
    game_id: str = Query(..., description="NBA game ID"),
    team_tricode: Optional[str] = Query(None, description="Specific team code (optional)")
):
    """
    Get current Team Momentum Index (TMI) values for a game.
    
    This endpoint fetches the latest play-by-play data, processes it through
    the momentum engine, and returns current TMI values for both teams.
    
    Args:
        game_id: NBA game ID
        team_tricode: Optional specific team code to filter results
        
    Returns:
        Current momentum data including TMI values and feature contributions
        
    Raises:
        HTTPException: If game not found or processing fails
    """
    try:
        logger.info(f"Getting current momentum for game {game_id}")
        
        # Record metrics
        metrics = get_metrics_collector()
        cache = get_cache()
        
        # Validate input parameters
        if not game_id or len(game_id.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Game ID is required and cannot be empty"
            )
        
        # Check cache first for better performance
        with time_operation("cache_lookup_momentum"):
            cache_key = f"current_momentum:{game_id}:{team_tricode or 'all'}"
            cached_response = cache.get_api_response("current_momentum", cache_key)
            
            if cached_response:
                logger.info(f"Returning cached momentum data for game {game_id}")
                metrics.increment_counter("cache_hits", tags={"endpoint": "current_momentum"})
                return cached_response
        
        # Try to get cached data first for better performance
        cached_data = safe_execute(
            lambda: get_cached_game_momentum(game_id),
            None,
            f"Failed to get cached data for game {game_id}",
            log_errors=False
        )
        
        if cached_data:
            logger.info(f"Using cached momentum data for game {game_id}")
            
            try:
                # Filter by specific team if requested
                teams_data = cached_data["teams"]
                if team_tricode:
                    if team_tricode not in teams_data:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No momentum data found for team {team_tricode} in game {game_id}"
                        )
                    teams_data = {team_tricode: teams_data[team_tricode]}
                
                # Convert to response format
                teams_response = {}
                for team, tmi_data in teams_data.items():
                    teams_response[team] = TMIResponse(
                        game_id=game_id,
                        team_tricode=tmi_data["team_tricode"],
                        timestamp=cached_data["last_updated"],
                        tmi_value=tmi_data["tmi_value"],
                        feature_contributions=tmi_data["feature_contributions"],
                        rolling_window_size=tmi_data["rolling_window_size"],
                        prediction_probability=tmi_data["prediction_probability"],
                        confidence_score=tmi_data["confidence_score"]
                    )
                
                response = MomentumCurrentResponse(
                    game_id=game_id,
                    timestamp=datetime.utcnow().isoformat(),
                    teams=teams_response,
                    game_status=cached_data["game_info"]["status"],
                    last_updated=cached_data["last_updated"]
                )
                
                logger.info(f"Returning cached momentum data for {len(teams_response)} teams in game {game_id}")
                return response
                
            except Exception as e:
                log_error(
                    e,
                    context={"game_id": game_id, "cached_data_keys": list(cached_data.keys())},
                    severity=ErrorSeverity.MEDIUM
                )
                # Continue to fallback calculation
        
        # Fallback to direct calculation if no cached data
        logger.info(f"No cached data available, calculating momentum for game {game_id}")
        
        # Check if services are healthy
        if not graceful_degradation.is_service_healthy("nba_api"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NBA API service is currently unavailable. Please try again later."
            )
        
        if not graceful_degradation.is_service_healthy("momentum_engine"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Momentum calculation service is currently unavailable. Please try again later."
            )
        
        # Fetch latest game data
        try:
            game_info, events = await fetch_game_events(game_id)
        except APIError as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Game {game_id} not found"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Unable to fetch game data: {str(e)}"
                )
        except Exception as e:
            log_error(e, context={"game_id": game_id}, severity=ErrorSeverity.HIGH)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch game data from NBA API"
            )
        
        if game_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Game {game_id} not found or no data available"
            )
        
        if not events:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No events found for game {game_id}"
            )
        
        # Get momentum engine
        try:
            engine = get_momentum_engine()
        except Exception as e:
            log_error(e, context={"game_id": game_id}, severity=ErrorSeverity.CRITICAL)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Momentum calculation engine is not available"
            )
        
        # Segment possessions and calculate momentum
        try:
            possessions = engine.segment_possessions(events)
        except DataProcessingError as e:
            log_error(e, context={"game_id": game_id, "events_count": len(events)})
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process game events: {str(e)}"
            )
        
        if not possessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No possessions found for game {game_id}"
            )
        
        # Update momentum for all teams
        try:
            team_momentum = engine.update_rolling_window(game_id, possessions)
        except DataProcessingError as e:
            log_error(e, context={"game_id": game_id, "possessions_count": len(possessions)})
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to calculate momentum: {str(e)}"
            )
        
        # If no momentum calculated yet, calculate for all teams in the game
        if not team_momentum:
            teams_in_game = set(event.team_tricode for event in events if event.team_tricode and event.team_tricode != 'UNK')
            for team in teams_in_game:
                try:
                    team_possessions = [p for p in possessions if p.team_tricode == team]
                    if team_possessions:
                        features = engine.calculate_possession_features(team_possessions)
                        tmi = engine.compute_tmi(team, game_id, features)
                        team_momentum[team] = tmi
                except Exception as e:
                    log_error(
                        e,
                        context={"game_id": game_id, "team": team},
                        severity=ErrorSeverity.MEDIUM
                    )
                    continue
        
        # Filter by specific team if requested
        if team_tricode:
            if team_tricode not in team_momentum:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No momentum data found for team {team_tricode} in game {game_id}"
                )
            team_momentum = {team_tricode: team_momentum[team_tricode]}
        
        # Convert to response format
        teams_response = {}
        for team, tmi in team_momentum.items():
            try:
                teams_response[team] = TMIResponse(
                    game_id=tmi.game_id,
                    team_tricode=tmi.team_tricode,
                    timestamp=tmi.timestamp.isoformat(),
                    tmi_value=tmi.tmi_value,
                    feature_contributions=tmi.feature_contributions,
                    rolling_window_size=tmi.rolling_window_size,
                    prediction_probability=tmi.prediction_probability,
                    confidence_score=tmi.confidence_score
                )
            except Exception as e:
                log_error(
                    e,
                    context={"game_id": game_id, "team": team},
                    severity=ErrorSeverity.LOW
                )
                continue
        
        if not teams_response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No valid momentum data could be calculated for game {game_id}"
            )
        
        response = MomentumCurrentResponse(
            game_id=game_id,
            timestamp=datetime.utcnow().isoformat(),
            teams=teams_response,
            game_status=game_info.status,
            last_updated=datetime.utcnow().isoformat()
        )
        
        # Cache the response
        with time_operation("cache_store_momentum"):
            cache.cache_api_response("current_momentum", cache_key, response)
        
        metrics.increment_counter("cache_misses", tags={"endpoint": "current_momentum"})
        logger.info(f"Returning momentum data for {len(teams_response)} teams in game {game_id}")
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except (APIError, DataProcessingError, MLModelError) as e:
        log_error(e, context={"game_id": game_id}, severity=ErrorSeverity.HIGH)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {str(e)}"
        )
    except Exception as e:
        log_error(e, context={"game_id": game_id}, severity=ErrorSeverity.CRITICAL)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )


@router.get("/predict", response_model=MomentumPredictResponse)
async def predict_momentum(
    game_id: str = Query(..., description="NBA game ID"),
    team_tricode: str = Query(..., description="Team code for prediction")
):
    """
    Get momentum continuation prediction for a specific team.
    
    This endpoint uses the trained ML model to predict the probability
    that the current team momentum will continue in the next possession.
    
    Args:
        game_id: NBA game ID
        team_tricode: Team code for prediction
        
    Returns:
        Momentum prediction with probability and confidence scores
        
    Raises:
        HTTPException: If game/team not found or prediction fails
    """
    try:
        logger.info(f"Getting momentum prediction for team {team_tricode} in game {game_id}")
        
        # Check if production ML model is available
        if not production_predictor.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Production ML model not available. Please check model deployment."
            )
        
        # Get momentum engine
        engine = get_momentum_engine()
        
        # Get current TMI for the team
        current_tmi = engine.get_latest_tmi(game_id, team_tricode)
        
        if current_tmi is None:
            # Try to calculate momentum first
            game_info, events = await fetch_game_events(game_id)
            
            if game_info is None or not events:
                raise HTTPException(
                    status_code=404,
                    detail=f"Game {game_id} not found or no data available"
                )
            
            # Calculate momentum
            possessions = engine.segment_possessions(events)
            team_possessions = [p for p in possessions if p.team_tricode == team_tricode]
            
            if not team_possessions:
                raise HTTPException(
                    status_code=404,
                    detail=f"No possessions found for team {team_tricode} in game {game_id}"
                )
            
            features = engine.calculate_possession_features(team_possessions)
            current_tmi = engine.compute_tmi(team_tricode, game_id, features)
        
        # Get TMI history for the team
        team_key = f"{game_id}_{team_tricode}"
        tmi_history = list(engine.tmi_history.get(team_key, []))
        
        # Get current possession features
        current_possessions = list(engine.team_possessions.get(team_key, []))
        
        # Get ML prediction using production model
        prediction_prob, confidence, feature_details = production_predictor.predict_momentum_continuation(
            events, team_tricode
        )
        
        # Determine prediction class
        prediction_class = "continue" if prediction_prob > 0.5 else "shift"
        
        response = MomentumPredictResponse(
            game_id=game_id,
            team_tricode=team_tricode,
            prediction_probability=prediction_prob,
            confidence_score=confidence,
            prediction_class=prediction_class,
            current_tmi=current_tmi.tmi_value,
            feature_contributions=feature_details,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Prediction for {team_tricode}: {prediction_prob:.3f} ({prediction_class})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting momentum for {team_tricode} in game {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict momentum: {str(e)}"
        )


@router.get("/games", response_model=List[GameSelectionResponse])
async def get_available_games():
    """
    Get list of available games for momentum analysis.
    
    This endpoint provides a list of currently active and recent NBA games
    that can be selected for momentum analysis.
    
    Returns:
        List of available games with basic information
        
    Raises:
        HTTPException: If unable to fetch game list
    """
    try:
        logger.info("Fetching available games for momentum analysis")
        
        # Import here to avoid circular imports
        from backend.services.live_fetcher import get_live_games
        
        # Get active games
        games = await get_live_games()
        
        # Convert to response format
        response_games = []
        for game in games:
            response_games.append(GameSelectionResponse(
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                game_date=game.game_date,
                status=game.status,
                home_score=getattr(game, 'home_score', 0),
                away_score=getattr(game, 'away_score', 0)
            ))
        
        logger.info(f"Returning {len(response_games)} available games")
        return response_games
        
    except Exception as e:
        logger.error(f"Error fetching available games: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch available games: {str(e)}"
        )


@router.get("/status")
async def get_momentum_service_status():
    """
    Get the current status of the momentum calculation service.
    
    Returns:
        Service status including engine state and ML model availability
    """
    try:
        global momentum_engine
        
        # Get comprehensive health status
        health_status = health_checker.get_overall_health()
        
        # Check momentum engine status
        engine_status = "initialized" if momentum_engine is not None else "not_initialized"
        
        # Check ML model status
        ml_status = "trained" if ml_predictor.is_trained else "not_trained"
        production_ml_status = "loaded" if production_predictor.is_loaded else "not_loaded"
        
        # Get active games count with error handling
        active_games_count = safe_execute(
            lambda: _get_active_games_count(),
            0,
            "Failed to get active games count",
            log_errors=False
        )
        
        # Get service health statuses
        service_statuses = {
            "nba_api": graceful_degradation.is_service_healthy("nba_api"),
            "momentum_engine": graceful_degradation.is_service_healthy("momentum_engine"),
            "ml_predictor": graceful_degradation.is_service_healthy("ml_predictor")
        }
        
        # Determine overall status
        overall_status = "healthy" if health_status["overall_healthy"] else "degraded"
        if not any(service_statuses.values()):
            overall_status = "unhealthy"
        
        # Get production model info
        production_model_info = production_predictor.get_model_info()
        
        return JSONResponse(content={
            "status": overall_status,
            "momentum_engine": engine_status,
            "ml_model": ml_status,
            "production_ml_model": production_ml_status,
            "production_model_info": production_model_info,
            "ml_model_path": getattr(ml_predictor, 'model_path', None),
            "feature_count": len(ml_predictor.feature_names) if hasattr(ml_predictor, 'feature_names') and ml_predictor.feature_names else 0,
            "active_games_count": active_games_count,
            "service_health": service_statuses,
            "detailed_health": health_status,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        log_error(e, context={"endpoint": "status"}, severity=ErrorSeverity.MEDIUM)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": "Service status check failed",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


async def _get_active_games_count() -> int:
    """Helper function to get active games count."""
    from backend.services.live_fetcher import get_live_games
    games = await get_live_games()
    return len(games)


@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint for monitoring.
    
    Returns:
        Detailed health information for all services
    """
    try:
        # Perform health checks
        health_status = health_checker.get_overall_health()
        
        # Add additional service information
        health_status["services"]["api_endpoints"] = {
            "healthy": True,
            "last_check": datetime.utcnow()
        }
        
        # Check database connectivity if available
        try:
            from backend.database.service import DatabaseService
            db_service = DatabaseService()
            db_healthy = db_service.test_connection()
            health_status["services"]["database"] = {
                "healthy": db_healthy,
                "last_check": datetime.utcnow()
            }
        except Exception as e:
            health_status["services"]["database"] = {
                "healthy": False,
                "last_check": datetime.utcnow(),
                "error": str(e)
            }
        
        # Determine HTTP status code
        http_status = status.HTTP_200_OK if health_status["overall_healthy"] else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=http_status,
            content=health_status
        )
        
    except Exception as e:
        log_error(e, context={"endpoint": "health"}, severity=ErrorSeverity.HIGH)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "overall_healthy": False,
                "error": "Health check failed",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.post("/refresh")
async def refresh_momentum_data(
    game_id: str = Query(..., description="NBA game ID")
):
    """
    Force refresh of momentum data for a specific game.
    
    This endpoint uses the real-time pipeline to force an immediate
    update for the specified game.
    
    Args:
        game_id: NBA game ID to refresh
        
    Returns:
        Refresh status and updated momentum data
    """
    try:
        logger.info(f"Refreshing momentum data for game {game_id}")
        
        # Use the pipeline to force an update
        result = await force_game_update(game_id)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=404,
                detail=result["message"]
            )
        
        # Get the updated cached data
        cached_data = get_cached_game_momentum(game_id)
        
        if cached_data:
            teams_response = {}
            for team, tmi_data in cached_data["teams"].items():
                teams_response[team] = {
                    "team_tricode": tmi_data["team_tricode"],
                    "tmi_value": tmi_data["tmi_value"],
                    "prediction_probability": tmi_data["prediction_probability"],
                    "confidence_score": tmi_data["confidence_score"],
                    "rolling_window_size": tmi_data["rolling_window_size"]
                }
            
            return JSONResponse(content={
                "status": "refreshed",
                "game_id": game_id,
                "teams": teams_response,
                "event_count": cached_data["event_count"],
                "timestamp": result["timestamp"],
                "pipeline_result": result
            })
        else:
            return JSONResponse(content={
                "status": "refreshed",
                "game_id": game_id,
                "teams": {},
                "event_count": result.get("events_processed", 0),
                "timestamp": result["timestamp"],
                "pipeline_result": result
            })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing momentum data for game {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh momentum data: {str(e)}"
        )