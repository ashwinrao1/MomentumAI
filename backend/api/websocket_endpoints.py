"""
WebSocket endpoints for real-time momentum data streaming.

This module provides WebSocket connections for live momentum updates,
enabling real-time dashboard communication.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from backend.services.momentum_engine import create_momentum_engine
from backend.services.live_fetcher import fetch_game_events
from backend.services.realtime_pipeline import get_pipeline, add_game_to_pipeline, remove_game_from_pipeline

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Connection manager for WebSocket clients
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.game_subscriptions: Dict[str, Set[WebSocket]] = {}
        self.connection_heartbeats: Dict[WebSocket, datetime] = {}
        self.momentum_engine = create_momentum_engine()
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_timeout = 60  # seconds
        self.pipeline = get_pipeline()
        
        # Message throttling to optimize performance
        self.last_broadcast_time: Dict[str, datetime] = {}
        self.min_broadcast_interval = 2.0  # Minimum 2 seconds between broadcasts
        self.message_queue: Dict[str, dict] = {}  # Queue latest message per game
        
        # Set this manager as the WebSocket broadcaster for the pipeline
        self.pipeline.set_websocket_callback(self._pipeline_broadcast_callback)
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_heartbeats[websocket] = datetime.utcnow()
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from game subscriptions
        for game_id, subscribers in self.game_subscriptions.items():
            if websocket in subscribers:
                subscribers.remove(websocket)
        
        # Remove from heartbeat tracking
        if websocket in self.connection_heartbeats:
            del self.connection_heartbeats[websocket]
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def subscribe_to_game(self, websocket: WebSocket, game_id: str):
        """Subscribe a WebSocket to game updates."""
        if game_id not in self.game_subscriptions:
            self.game_subscriptions[game_id] = set()
        
        self.game_subscriptions[game_id].add(websocket)
        
        # Add game to real-time pipeline if not already active
        await add_game_to_pipeline(game_id)
        
        logger.info(f"WebSocket subscribed to game {game_id}")
    
    async def unsubscribe_from_game(self, websocket: WebSocket, game_id: str):
        """Unsubscribe a WebSocket from game updates."""
        if game_id in self.game_subscriptions and websocket in self.game_subscriptions[game_id]:
            self.game_subscriptions[game_id].remove(websocket)
            
            # Remove game from pipeline if no more subscribers
            if not self.game_subscriptions[game_id]:
                await remove_game_from_pipeline(game_id)
                del self.game_subscriptions[game_id]
            
            logger.info(f"WebSocket unsubscribed from game {game_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending personal message: {e}")
                self.disconnect(websocket)
    
    async def broadcast_to_game(self, message: dict, game_id: str):
        """Broadcast a message to all subscribers of a game with throttling."""
        if game_id not in self.game_subscriptions:
            return
        
        # Throttle broadcasts to avoid overwhelming clients
        current_time = datetime.utcnow()
        last_broadcast = self.last_broadcast_time.get(game_id)
        
        if last_broadcast:
            time_since_last = (current_time - last_broadcast).total_seconds()
            if time_since_last < self.min_broadcast_interval:
                # Queue the message instead of sending immediately
                self.message_queue[game_id] = message
                return
        
        # Send queued message if available, otherwise send current message
        message_to_send = self.message_queue.pop(game_id, message)
        
        disconnected_clients = []
        successful_sends = 0
        
        for websocket in self.game_subscriptions[game_id]:
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    # Compress JSON for smaller payload
                    json_str = json.dumps(message_to_send, separators=(',', ':'))
                    await websocket.send_text(json_str)
                    successful_sends += 1
                except Exception as e:
                    logger.error(f"Error broadcasting to game {game_id}: {e}")
                    disconnected_clients.append(websocket)
            else:
                disconnected_clients.append(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected_clients:
            self.game_subscriptions[game_id].discard(websocket)
        
        # Update last broadcast time
        self.last_broadcast_time[game_id] = current_time
        
        # Log performance metrics
        if successful_sends > 0:
            from backend.utils.performance_monitor import get_metrics_collector
            metrics = get_metrics_collector()
            metrics.record_websocket_message(sent=True)
            metrics.increment_counter("websocket_broadcasts", successful_sends)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected_clients = []
        
        for websocket in self.active_connections:
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to all: {e}")
                    disconnected_clients.append(websocket)
            else:
                disconnected_clients.append(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected_clients:
            self.disconnect(websocket)
    
    async def process_momentum_update(self, game_id: str):
        """Process momentum update for a game and broadcast to subscribers."""
        try:
            # Check if we have cached data first
            cached_data = self.pipeline.get_cached_momentum(game_id)
            
            if cached_data:
                # Optimize payload size by only sending essential data
                optimized_teams = {}
                for team, tmi_data in cached_data["teams"].items():
                    optimized_teams[team] = {
                        "t": tmi_data["team_tricode"],  # Shortened field names
                        "v": round(tmi_data["tmi_value"], 3),  # Rounded values
                        "p": round(tmi_data["prediction_probability"], 3),
                        "c": round(tmi_data["confidence_score"], 3),
                        "f": {k: round(v, 3) for k, v in tmi_data["feature_contributions"].items()}
                    }
                
                # Compressed message format
                message = {
                    "type": "momentum_update",
                    "game_id": game_id,
                    "ts": int(datetime.utcnow().timestamp()),  # Unix timestamp
                    "status": cached_data["game_info"]["status"],
                    "teams": optimized_teams,
                    "cached": True
                }
                
                await self.broadcast_to_game(message, game_id)
                logger.info(f"Broadcasted optimized cached momentum update for game {game_id}")
                return
            
            # Fallback to direct calculation if no cached data
            game_info, events = await fetch_game_events(game_id)
            
            if game_info is None or not events:
                logger.warning(f"No data available for game {game_id}")
                return
            
            # Calculate momentum
            possessions = self.momentum_engine.segment_possessions(events)
            team_momentum = self.momentum_engine.update_rolling_window(game_id, possessions)
            
            if not team_momentum:
                logger.warning(f"No momentum calculated for game {game_id}")
                return
            
            # Prepare optimized broadcast message
            teams_data = {}
            for team, tmi in team_momentum.items():
                teams_data[team] = {
                    "t": tmi.team_tricode,
                    "v": round(tmi.tmi_value, 3),
                    "p": round(tmi.prediction_probability, 3),
                    "c": round(tmi.confidence_score, 3),
                    "f": {k: round(v, 3) for k, v in tmi.feature_contributions.items()}
                }
            
            # Compressed message format
            message = {
                "type": "momentum_update",
                "game_id": game_id,
                "ts": int(datetime.utcnow().timestamp()),
                "status": game_info.status,
                "game_info": {
                    "h": game_info.home_team,
                    "a": game_info.away_team,
                    "p": game_info.period,
                    "c": game_info.clock,
                    "hs": game_info.home_score,
                    "as": game_info.away_score
                },
                "teams": teams_data,
                "cached": False
            }
            
            # Broadcast to game subscribers
            await self.broadcast_to_game(message, game_id)
            logger.info(f"Broadcasted momentum update for game {game_id} to {len(self.game_subscriptions.get(game_id, []))} clients")
            
        except Exception as e:
            logger.error(f"Error processing momentum update for game {game_id}: {e}")
    
    async def _pipeline_broadcast_callback(self, message: dict, game_id: str):
        """Callback function for pipeline to broadcast momentum updates."""
        try:
            await self.broadcast_to_game(message, game_id)
            logger.debug(f"Pipeline broadcast completed for game {game_id}")
        except Exception as e:
            logger.error(f"Pipeline broadcast failed for game {game_id}: {e}")
    
    def update_heartbeat(self, websocket: WebSocket):
        """Update heartbeat timestamp for a connection."""
        if websocket in self.connection_heartbeats:
            self.connection_heartbeats[websocket] = datetime.utcnow()
    
    async def check_connection_health(self):
        """Check connection health and remove stale connections."""
        current_time = datetime.utcnow()
        stale_connections = []
        
        for websocket, last_heartbeat in self.connection_heartbeats.items():
            time_since_heartbeat = (current_time - last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.heartbeat_timeout:
                stale_connections.append(websocket)
                logger.warning(f"Connection stale, removing: {time_since_heartbeat}s since last heartbeat")
        
        # Remove stale connections
        for websocket in stale_connections:
            self.disconnect(websocket)
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(code=1000, reason="Connection timeout")
            except Exception as e:
                logger.error(f"Error closing stale connection: {e}")
    
    async def send_heartbeat_ping(self):
        """Send heartbeat ping to all connected clients."""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_all(ping_message)
        logger.debug(f"Sent heartbeat ping to {len(self.active_connections)} connections")


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/momentum")
async def websocket_momentum_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time momentum updates.
    
    Clients can connect to this endpoint to receive live momentum data.
    The connection supports subscribing to specific games and receiving
    automatic updates when momentum changes occur.
    
    Message format:
    - Client -> Server: {"action": "subscribe", "game_id": "game_id"}
    - Client -> Server: {"action": "unsubscribe", "game_id": "game_id"}
    - Client -> Server: {"action": "ping"}
    - Server -> Client: {"type": "momentum_update", "game_id": "...", "teams": {...}}
    - Server -> Client: {"type": "pong"}
    - Server -> Client: {"type": "error", "message": "..."}
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connected",
            "message": "Connected to MomentumML WebSocket",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                action = message.get("action")
                
                if action == "subscribe":
                    game_id = message.get("game_id")
                    if game_id:
                        await manager.subscribe_to_game(websocket, game_id)
                        
                        # Send current momentum data if available
                        await manager.process_momentum_update(game_id)
                        
                        await manager.send_personal_message({
                            "type": "subscribed",
                            "game_id": game_id,
                            "message": f"Subscribed to game {game_id}",
                            "timestamp": datetime.utcnow().isoformat()
                        }, websocket)
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "game_id required for subscription"
                        }, websocket)
                
                elif action == "unsubscribe":
                    game_id = message.get("game_id")
                    if game_id:
                        await manager.unsubscribe_from_game(websocket, game_id)
                        await manager.send_personal_message({
                            "type": "unsubscribed",
                            "game_id": game_id,
                            "message": f"Unsubscribed from game {game_id}",
                            "timestamp": datetime.utcnow().isoformat()
                        }, websocket)
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "game_id required for unsubscription"
                        }, websocket)
                
                elif action == "ping":
                    manager.update_heartbeat(websocket)
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }, websocket)
                
                elif action == "pong":
                    # Client responding to server ping
                    manager.update_heartbeat(websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown action: {action}"
                    }, websocket)
            
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
            
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/status")
async def websocket_status_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for system status updates.
    
    Provides periodic status updates about the momentum service,
    including active games, connection counts, and service health.
    """
    await websocket.accept()
    
    try:
        while True:
            # Send status update every 30 seconds
            status_message = {
                "type": "status_update",
                "timestamp": datetime.utcnow().isoformat(),
                "active_connections": len(manager.active_connections),
                "game_subscriptions": {
                    game_id: len(subscribers) 
                    for game_id, subscribers in manager.game_subscriptions.items()
                },
                "heartbeat_status": {
                    "tracked_connections": len(manager.connection_heartbeats),
                    "heartbeat_interval": manager.heartbeat_interval,
                    "heartbeat_timeout": manager.heartbeat_timeout
                },
                "service_status": "healthy"
            }
            
            await websocket.send_text(json.dumps(status_message))
            await asyncio.sleep(30)  # Send status every 30 seconds
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Status WebSocket error: {e}")


@router.get("/ws/stats")
async def get_websocket_stats():
    """
    REST endpoint to get current WebSocket connection statistics.
    
    Returns:
        JSON object with connection statistics and health information
    """
    return get_connection_stats()


@router.get("/ws/pipeline/stats")
async def get_pipeline_stats():
    """
    REST endpoint to get real-time pipeline statistics.
    
    Returns:
        JSON object with pipeline performance metrics
    """
    from backend.services.realtime_pipeline import get_pipeline_statistics
    return get_pipeline_statistics()


@router.post("/ws/pipeline/force-update/{game_id}")
async def force_pipeline_update(game_id: str):
    """
    Force an immediate pipeline update for a specific game.
    
    Args:
        game_id: NBA game ID to update
        
    Returns:
        Update results
    """
    from backend.services.realtime_pipeline import force_game_update
    return await force_game_update(game_id)


# Utility functions for external integration
async def trigger_momentum_update(game_id: str):
    """
    Trigger a momentum update for a specific game.
    
    This function can be called from other parts of the application
    to push momentum updates to connected WebSocket clients.
    
    Args:
        game_id: NBA game ID to update
    """
    await manager.process_momentum_update(game_id)


def get_connection_stats() -> Dict:
    """
    Get current WebSocket connection statistics.
    
    Returns:
        Dictionary with connection statistics
    """
    return {
        "total_connections": len(manager.active_connections),
        "game_subscriptions": {
            game_id: len(subscribers) 
            for game_id, subscribers in manager.game_subscriptions.items()
        },
        "active_games": list(manager.game_subscriptions.keys()),
        "heartbeat_status": {
            "tracked_connections": len(manager.connection_heartbeats),
            "heartbeat_interval": manager.heartbeat_interval,
            "heartbeat_timeout": manager.heartbeat_timeout
        }
    }


# Background task for periodic momentum updates
async def periodic_momentum_updates():
    """
    Background task that manages WebSocket connections and starts the real-time pipeline.
    
    This function manages connection health and delegates momentum processing
    to the real-time pipeline.
    """
    heartbeat_counter = 0
    
    # Start the real-time pipeline in the background
    pipeline = get_pipeline()
    asyncio.create_task(pipeline.start_pipeline())
    
    while True:
        try:
            # Check connection health every 60 seconds (2 cycles)
            if heartbeat_counter % 2 == 0:
                await manager.check_connection_health()
                await manager.send_heartbeat_ping()
            
            heartbeat_counter += 1
            
            # Wait 30 seconds before next health check cycle
            await asyncio.sleep(30)
        
        except Exception as e:
            logger.error(f"Error in periodic WebSocket management: {e}")
            await asyncio.sleep(60)  # Wait longer on error