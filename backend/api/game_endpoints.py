"""
Game data API endpoints for MomentumML.

This module provides REST API endpoints for game data fetching,
processing, and real-time updates.
"""

import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from backend.services.live_fetcher import LiveDataFetcher, get_live_games, fetch_game_events
from backend.services.ml_predictor import MomentumPredictor, train_momentum_model
from backend.services.historical_data_collector import create_sample_training_data
from backend.services.enhanced_momentum_predictor import get_enhanced_momentum_analysis, get_momentum_visualization_data
from backend.models.game_models import GameSelectionResponse, GameEventResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/games", tags=["games"])

# Global instances
fetcher = LiveDataFetcher()
ml_predictor = MomentumPredictor()


@router.get("/active", response_model=List[GameSelectionResponse])
async def get_active_games():
    """
    Get list of currently active NBA games.
    
    Returns:
        List of active games with basic information
        
    Raises:
        HTTPException: If API request fails
    """
    try:
        logger.info("Fetching active games")
        games = await get_live_games()
        
        # Convert to response format
        response_games = [
            GameSelectionResponse(
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                game_date=game.game_date,
                status=game.status
            )
            for game in games
        ]
        
        logger.info(f"Returning {len(response_games)} active games")
        return response_games
        
    except Exception as e:
        logger.error(f"Error fetching active games: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch active games: {str(e)}"
        )


@router.get("/{game_id}/fetch")
async def fetch_game_data(game_id: str):
    """
    Fetch latest play-by-play data for a specific game.
    
    Args:
        game_id: NBA game ID
        
    Returns:
        Game information and events
        
    Raises:
        HTTPException: If game not found or API request fails
    """
    try:
        logger.info(f"Fetching data for game {game_id}")
        
        game_info, events = await fetch_game_events(game_id)
        
        if game_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Game {game_id} not found or no data available"
            )
        
        # Convert events to response format
        event_responses = [
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
        
        response_data = {
            "game_info": {
                "game_id": game_info.game_id,
                "home_team": game_info.home_team,
                "away_team": game_info.away_team,
                "game_date": game_info.game_date,
                "status": game_info.status,
                "period": game_info.period,
                "clock": game_info.clock,
                "home_score": game_info.home_score,
                "away_score": game_info.away_score
            },
            "events": event_responses,
            "event_count": len(events)
        }
        
        logger.info(f"Returning {len(events)} events for game {game_id}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching game data for {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch game data: {str(e)}"
        )


@router.post("/{game_id}/process")
async def process_game_data(game_id: str, background_tasks: BackgroundTasks):
    """
    Process game data and compute features (placeholder for momentum engine).
    
    Args:
        game_id: NBA game ID
        background_tasks: FastAPI background tasks
        
    Returns:
        Processing status
    """
    try:
        logger.info(f"Processing game data for {game_id}")
        
        # Fetch latest data
        game_info, events = await fetch_game_events(game_id)
        
        if game_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Game {game_id} not found"
            )
        
        # Add background task for processing (placeholder)
        background_tasks.add_task(
            _process_game_events_background,
            game_id,
            events
        )
        
        return JSONResponse(content={
            "status": "processing",
            "game_id": game_id,
            "event_count": len(events),
            "message": "Game data processing started"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing game data for {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process game data: {str(e)}"
        )


@router.get("/{game_id}/status")
async def get_game_status(game_id: str):
    """
    Get current status and basic info for a game.
    
    Args:
        game_id: NBA game ID
        
    Returns:
        Game status information
    """
    try:
        logger.info(f"Getting status for game {game_id}")
        
        game_info, _ = await fetch_game_events(game_id)
        
        if game_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Game {game_id} not found"
            )
        
        return JSONResponse(content={
            "game_id": game_info.game_id,
            "home_team": game_info.home_team,
            "away_team": game_info.away_team,
            "status": game_info.status,
            "period": game_info.period,
            "clock": game_info.clock,
            "home_score": game_info.home_score,
            "away_score": game_info.away_score,
            "last_updated": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting game status for {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get game status: {str(e)}"
        )


async def _process_game_events_background(game_id: str, events: List):
    """
    Background task for processing game events.
    
    This is a placeholder for the momentum engine integration.
    """
    try:
        logger.info(f"Background processing started for game {game_id} with {len(events)} events")
        
        # Placeholder for momentum calculations
        # This will be implemented in task 3 (momentum calculation engine)
        
        logger.info(f"Background processing completed for game {game_id}")
        
    except Exception as e:
        logger.error(f"Error in background processing for game {game_id}: {e}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for game data services."""
    try:
        # Test NBA API connectivity
        games = await get_live_games()
        
        return JSONResponse(content={
            "status": "healthy",
            "nba_api_status": "connected",
            "active_games_count": len(games),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "nba_api_status": "disconnected",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# ML Prediction Endpoints
@router.post("/ml/train")
async def train_ml_model(background_tasks: BackgroundTasks, use_sample_data: bool = True):
    """
    Train the momentum prediction model.
    
    Args:
        background_tasks: FastAPI background tasks
        use_sample_data: Whether to use sample data (True) or real NBA data (False)
        
    Returns:
        Training status
    """
    try:
        logger.info("Starting ML model training")
        
        # Add background task for training
        background_tasks.add_task(
            _train_model_background,
            use_sample_data
        )
        
        return JSONResponse(content={
            "status": "training_started",
            "use_sample_data": use_sample_data,
            "message": "Model training started in background",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model training: {str(e)}"
        )


@router.get("/ml/status")
async def get_ml_model_status():
    """
    Get the current status of the ML model.
    
    Returns:
        Model status information
    """
    try:
        return JSONResponse(content={
            "model_loaded": ml_predictor.is_trained,
            "model_path": ml_predictor.model_path,
            "feature_count": len(ml_predictor.feature_names) if ml_predictor.feature_names else 0,
            "feature_names": ml_predictor.feature_names,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting ML model status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )


@router.post("/ml/predict")
async def predict_momentum(game_id: str):
    """
    Get momentum prediction for a specific game.
    
    Args:
        game_id: NBA game ID
        
    Returns:
        Momentum prediction results
    """
    try:
        if not ml_predictor.is_trained:
            raise HTTPException(
                status_code=400,
                detail="ML model not trained. Please train the model first."
            )
        
        logger.info(f"Getting momentum prediction for game {game_id}")
        
        # For now, return a placeholder prediction
        # In a full implementation, this would get current game state
        # and use the momentum engine to calculate features
        
        prediction_prob, confidence = ml_predictor.predict_momentum_continuation([], [])
        
        return JSONResponse(content={
            "game_id": game_id,
            "prediction_probability": prediction_prob,
            "confidence_score": confidence,
            "prediction_class": "continue" if prediction_prob > 0.5 else "shift",
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting momentum prediction for {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get momentum prediction: {str(e)}"
        )


# Enhanced Momentum Endpoints
@router.get("/{game_id}/momentum/enhanced")
async def get_enhanced_momentum(game_id: str):
    """
    Get enhanced momentum analysis including individual team momentum
    and overall game momentum with team highlighting.
    
    Args:
        game_id: NBA game ID
        
    Returns:
        Enhanced momentum analysis with game-level momentum
    """
    try:
        logger.info(f"Getting enhanced momentum analysis for game {game_id}")
        
        # Fetch game events
        game_info, events = await fetch_game_events(game_id)
        
        if game_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Game {game_id} not found"
            )
        
        # Convert events to dict format for analysis
        event_dicts = [
            {
                'event_id': event.event_id,
                'game_id': event.game_id,
                'team_tricode': event.team_tricode,
                'player_name': event.player_name,
                'event_type': event.event_type,
                'clock': event.clock,
                'period': event.period,
                'points_total': event.points_total,
                'shot_result': event.shot_result,
                'timestamp': event.timestamp.isoformat(),
                'description': getattr(event, 'description', '')
            }
            for event in events
        ]
        
        # Get enhanced momentum analysis
        momentum_analysis = get_enhanced_momentum_analysis(event_dicts)
        
        # Add game info to response
        response_data = {
            "game_info": {
                "game_id": game_info.game_id,
                "home_team": game_info.home_team,
                "away_team": game_info.away_team,
                "status": game_info.status,
                "period": game_info.period,
                "clock": game_info.clock,
                "home_score": game_info.home_score,
                "away_score": game_info.away_score
            },
            "momentum_analysis": momentum_analysis,
            "events_analyzed": len(events)
        }
        
        logger.info(f"Enhanced momentum analysis complete for game {game_id}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhanced momentum for {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get enhanced momentum analysis: {str(e)}"
        )


@router.get("/{game_id}/momentum/visualization")
async def get_momentum_visualization(game_id: str):
    """
    Get momentum data formatted specifically for frontend visualization
    including momentum meter, team highlighting, and momentum bars.
    
    Args:
        game_id: NBA game ID
        
    Returns:
        Momentum visualization data
    """
    try:
        logger.info(f"Getting momentum visualization data for game {game_id}")
        
        # Fetch game events
        game_info, events = await fetch_game_events(game_id)
        
        if game_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Game {game_id} not found"
            )
        
        # Convert events to dict format
        event_dicts = [
            {
                'event_id': event.event_id,
                'game_id': event.game_id,
                'team_tricode': event.team_tricode,
                'player_name': event.player_name,
                'event_type': event.event_type,
                'clock': event.clock,
                'period': event.period,
                'points_total': event.points_total,
                'shot_result': event.shot_result,
                'timestamp': event.timestamp.isoformat(),
                'description': getattr(event, 'description', '')
            }
            for event in events
        ]
        
        # Get visualization data
        viz_data = get_momentum_visualization_data(event_dicts)
        
        # Add game context
        response_data = {
            "game_id": game_id,
            "teams": {
                "home": game_info.home_team,
                "away": game_info.away_team
            },
            "scores": {
                "home": game_info.home_score,
                "away": game_info.away_score
            },
            "game_status": {
                "period": game_info.period,
                "clock": game_info.clock,
                "status": game_info.status
            },
            "momentum_data": viz_data
        }
        
        logger.info(f"Momentum visualization data ready for game {game_id}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting momentum visualization for {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get momentum visualization data: {str(e)}"
        )





async def _train_model_background(use_sample_data: bool = True):
    """
    Background task for training the ML model.
    
    Args:
        use_sample_data: Whether to use sample data or real NBA data
    """
    try:
        logger.info(f"Background ML training started (sample_data={use_sample_data})")
        
        if use_sample_data:
            # Use sample data for training
            sample_events = create_sample_training_data()
            results = train_momentum_model(sample_events)
        else:
            # Use real NBA data (would need to implement historical data collection)
            logger.warning("Real NBA data training not fully implemented, using sample data")
            sample_events = create_sample_training_data()
            results = train_momentum_model(sample_events)
        
        # Reload the global predictor
        global ml_predictor
        ml_predictor = MomentumPredictor()
        
        logger.info(f"Background ML training completed - Accuracy: {results['test_accuracy']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in background ML training: {e}")