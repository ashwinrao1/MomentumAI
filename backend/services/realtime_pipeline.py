"""
Real-time data pipeline for MomentumML.

This module integrates all components to provide a complete real-time
momentum analysis pipeline from NBA API to dashboard updates.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict

from backend.services.live_fetcher import LiveDataFetcher
from backend.services.momentum_engine import MomentumEngine, create_momentum_engine
from backend.database.service import DatabaseService
from backend.models.game_models import GameEvent, GameInfo, TeamMomentumIndex

# Configure logging
logger = logging.getLogger(__name__)


class RealtimePipeline:
    """
    Real-time data pipeline that orchestrates the complete momentum analysis flow.
    
    This class manages:
    - Live data fetching from NBA API
    - Automatic momentum recalculation
    - Data persistence
    - WebSocket broadcasting
    - Performance caching
    """
    
    def __init__(
        self,
        poll_interval: int = 25,
        cache_ttl: int = 300,  # 5 minutes
        max_cached_games: int = 10
    ):
        """
        Initialize the real-time pipeline.
        
        Args:
            poll_interval: Seconds between data polls
            cache_ttl: Cache time-to-live in seconds
            max_cached_games: Maximum number of games to cache
        """
        self.poll_interval = poll_interval
        self.cache_ttl = cache_ttl
        self.max_cached_games = max_cached_games
        
        # Initialize components
        self.live_fetcher = LiveDataFetcher(poll_interval=poll_interval)
        self.momentum_engine = create_momentum_engine()
        self.db_service = DatabaseService()
        
        # Pipeline state
        self.active_games: Set[str] = set()
        self.last_poll_times: Dict[str, float] = {}
        self.last_event_counts: Dict[str, int] = {}
        
        # Performance caching
        self.game_cache: Dict[str, Dict] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
        # WebSocket callback for broadcasting updates
        self.websocket_callback = None
        
        # Pipeline statistics
        self.stats = {
            'total_polls': 0,
            'successful_polls': 0,
            'events_processed': 0,
            'tmi_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'websocket_broadcasts': 0
        }
        
        logger.info("Real-time pipeline initialized")
    
    def set_websocket_callback(self, callback):
        """Set callback function for WebSocket broadcasting."""
        self.websocket_callback = callback
        logger.info("WebSocket callback registered")
    
    async def start_pipeline(self):
        """Start the real-time pipeline processing loop."""
        logger.info("Starting real-time pipeline")
        
        while True:
            try:
                await self._pipeline_cycle()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Pipeline cycle error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def add_game(self, game_id: str) -> bool:
        """
        Add a game to the real-time pipeline.
        
        Args:
            game_id: NBA game ID to monitor
            
        Returns:
            True if game was added successfully
        """
        try:
            # Verify game exists and get initial data
            game_info, events = await self.live_fetcher.fetch_live_game_data(game_id)
            
            if game_info is None:
                logger.warning(f"Game {game_id} not found, cannot add to pipeline")
                return False
            
            # Add to active games
            self.active_games.add(game_id)
            self.last_event_counts[game_id] = len(events)
            
            # Store initial game data
            self.db_service.create_or_update_game(game_info)
            new_events = self.db_service.store_events(events)
            
            # Calculate initial momentum if we have events
            if events:
                await self._process_momentum_update(game_id, game_info, events)
            
            logger.info(f"Added game {game_id} to pipeline ({len(events)} initial events)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add game {game_id} to pipeline: {e}")
            return False
    
    async def remove_game(self, game_id: str):
        """Remove a game from the real-time pipeline."""
        if game_id in self.active_games:
            self.active_games.remove(game_id)
            
            # Clean up state
            self.last_poll_times.pop(game_id, None)
            self.last_event_counts.pop(game_id, None)
            self._invalidate_cache(game_id)
            
            logger.info(f"Removed game {game_id} from pipeline")
    
    async def force_update(self, game_id: str) -> Dict:
        """
        Force an immediate update for a specific game.
        
        Args:
            game_id: NBA game ID to update
            
        Returns:
            Update results dictionary
        """
        try:
            logger.info(f"Forcing update for game {game_id}")
            
            # Fetch latest data
            game_info, events = await self.live_fetcher.fetch_live_game_data(game_id)
            
            if game_info is None:
                return {"status": "error", "message": "Game not found"}
            
            # Process the update
            result = await self._process_game_update(game_id, game_info, events)
            
            # Invalidate cache to force fresh data
            self._invalidate_cache(game_id)
            
            return {
                "status": "success",
                "game_id": game_id,
                "events_processed": result.get("new_events", 0),
                "momentum_updated": result.get("momentum_updated", False),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to force update for game {game_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_cached_momentum(self, game_id: str) -> Optional[Dict]:
        """
        Get cached momentum data for a game.
        
        Args:
            game_id: NBA game ID
            
        Returns:
            Cached momentum data or None if not available/expired
        """
        if game_id not in self.game_cache:
            self.stats['cache_misses'] += 1
            return None
        
        # Check cache expiry
        cache_time = self.cache_timestamps.get(game_id, 0)
        if time.time() - cache_time > self.cache_ttl:
            self._invalidate_cache(game_id)
            self.stats['cache_misses'] += 1
            return None
        
        self.stats['cache_hits'] += 1
        return self.game_cache[game_id]
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline performance statistics."""
        return {
            **self.stats,
            "active_games": len(self.active_games),
            "cached_games": len(self.game_cache),
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
            "cache_hit_rate": (
                self.stats['cache_hits'] / 
                max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
            )
        }
    
    async def _pipeline_cycle(self):
        """Execute one cycle of the pipeline processing."""
        if not self.active_games:
            return
        
        logger.debug(f"Pipeline cycle: processing {len(self.active_games)} games")
        
        # Process each active game
        for game_id in list(self.active_games):  # Copy to avoid modification during iteration
            try:
                await self._process_single_game(game_id)
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
        
        # Clean up old cache entries
        self._cleanup_cache()
        
        self.stats['total_polls'] += 1
    
    async def _process_single_game(self, game_id: str):
        """Process updates for a single game."""
        try:
            # Check if we should poll this game (rate limiting)
            if not self._should_poll_game(game_id):
                return
            
            # Fetch latest data
            game_info, events = await self.live_fetcher.fetch_live_game_data(game_id)
            
            if game_info is None:
                logger.warning(f"No data available for game {game_id}")
                return
            
            # Process the update
            await self._process_game_update(game_id, game_info, events)
            
            self.last_poll_times[game_id] = time.time()
            self.stats['successful_polls'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process game {game_id}: {e}")
    
    async def _process_game_update(self, game_id: str, game_info: GameInfo, events: List[GameEvent]) -> Dict:
        """Process a game update with new data."""
        result = {"new_events": 0, "momentum_updated": False}
        
        # Check if we have new events
        previous_count = self.last_event_counts.get(game_id, 0)
        current_count = len(events)
        
        if current_count <= previous_count:
            logger.debug(f"No new events for game {game_id} ({current_count} vs {previous_count})")
            return result
        
        # Update game status first
        self.db_service.create_or_update_game(game_info)
        
        # Store new events in database
        new_events_stored = self.db_service.store_events(events)
        result["new_events"] = new_events_stored
        self.stats['events_processed'] += new_events_stored
        
        # Calculate momentum if we have new events
        if new_events_stored > 0:
            await self._process_momentum_update(game_id, game_info, events)
            result["momentum_updated"] = True
        
        # Update event count
        self.last_event_counts[game_id] = current_count
        
        logger.info(f"Processed {new_events_stored} new events for game {game_id}")
        return result
    
    async def _process_momentum_update(self, game_id: str, game_info: GameInfo, events: List[GameEvent]):
        """Process momentum calculations and broadcasting."""
        try:
            # Store game info in database first
            self.db_service.create_or_update_game(game_info)
            
            # Store events in database
            self.db_service.store_events(events)
            
            # Segment possessions and calculate momentum
            possessions = self.momentum_engine.segment_possessions(events)
            team_momentum = self.momentum_engine.update_rolling_window(game_id, possessions)
            
            if not team_momentum:
                logger.debug(f"No momentum calculated for game {game_id}")
                return
            
            # Store TMI calculations in database
            for team, tmi in team_momentum.items():
                self.db_service.store_tmi_calculation(tmi)
                self.stats['tmi_calculations'] += 1
            
            # Update cache
            self._update_cache(game_id, game_info, team_momentum, events)
            
            # Broadcast to WebSocket clients
            if self.websocket_callback:
                await self._broadcast_momentum_update(game_id, game_info, team_momentum, events)
            
            logger.info(f"Updated momentum for {len(team_momentum)} teams in game {game_id}")
            
        except Exception as e:
            logger.error(f"Failed to process momentum update for game {game_id}: {e}")
    
    async def _broadcast_momentum_update(
        self,
        game_id: str,
        game_info: GameInfo,
        team_momentum: Dict[str, TeamMomentumIndex],
        events: List[GameEvent]
    ):
        """Broadcast momentum update via WebSocket."""
        try:
            # Prepare broadcast message
            teams_data = {}
            for team, tmi in team_momentum.items():
                teams_data[team] = {
                    "team_tricode": tmi.team_tricode,
                    "tmi_value": tmi.tmi_value,
                    "feature_contributions": tmi.feature_contributions,
                    "prediction_probability": tmi.prediction_probability,
                    "confidence_score": tmi.confidence_score,
                    "rolling_window_size": tmi.rolling_window_size
                }
            
            message = {
                "type": "momentum_update",
                "game_id": game_id,
                "timestamp": datetime.utcnow().isoformat(),
                "game_status": game_info.status,
                "game_info": {
                    "home_team": game_info.home_team,
                    "away_team": game_info.away_team,
                    "period": game_info.period,
                    "clock": game_info.clock,
                    "home_score": game_info.home_score,
                    "away_score": game_info.away_score
                },
                "teams": teams_data,
                "event_count": len(events)
            }
            
            # Call the WebSocket callback
            await self.websocket_callback(message, game_id)
            self.stats['websocket_broadcasts'] += 1
            
        except Exception as e:
            logger.error(f"Failed to broadcast momentum update for game {game_id}: {e}")
    
    def _should_poll_game(self, game_id: str) -> bool:
        """Check if enough time has passed to poll a game."""
        last_poll = self.last_poll_times.get(game_id, 0)
        return time.time() - last_poll >= self.poll_interval
    
    def _update_cache(
        self,
        game_id: str,
        game_info: GameInfo,
        team_momentum: Dict[str, TeamMomentumIndex],
        events: List[GameEvent]
    ):
        """Update the performance cache with latest data."""
        # Prepare cache data
        cache_data = {
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
            "teams": {},
            "event_count": len(events),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Add team momentum data
        for team, tmi in team_momentum.items():
            cache_data["teams"][team] = {
                "team_tricode": tmi.team_tricode,
                "tmi_value": tmi.tmi_value,
                "feature_contributions": tmi.feature_contributions,
                "prediction_probability": tmi.prediction_probability,
                "confidence_score": tmi.confidence_score,
                "rolling_window_size": tmi.rolling_window_size
            }
        
        # Update cache
        self.game_cache[game_id] = cache_data
        self.cache_timestamps[game_id] = time.time()
        
        # Limit cache size
        if len(self.game_cache) > self.max_cached_games:
            self._cleanup_cache(force=True)
    
    def _invalidate_cache(self, game_id: str):
        """Invalidate cache for a specific game."""
        self.game_cache.pop(game_id, None)
        self.cache_timestamps.pop(game_id, None)
    
    def _cleanup_cache(self, force: bool = False):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_games = []
        
        for game_id, cache_time in self.cache_timestamps.items():
            if force or (current_time - cache_time > self.cache_ttl):
                expired_games.append(game_id)
        
        # Remove expired entries
        for game_id in expired_games:
            self._invalidate_cache(game_id)
        
        if expired_games:
            logger.debug(f"Cleaned up {len(expired_games)} expired cache entries")


# Global pipeline instance
_pipeline_instance: Optional[RealtimePipeline] = None


def get_pipeline() -> RealtimePipeline:
    """Get or create the global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RealtimePipeline()
        _pipeline_instance._start_time = time.time()
    return _pipeline_instance


async def start_realtime_pipeline():
    """Start the global real-time pipeline."""
    pipeline = get_pipeline()
    await pipeline.start_pipeline()


# Utility functions for external integration
async def add_game_to_pipeline(game_id: str) -> bool:
    """Add a game to the real-time pipeline."""
    pipeline = get_pipeline()
    return await pipeline.add_game(game_id)


async def remove_game_from_pipeline(game_id: str):
    """Remove a game from the real-time pipeline."""
    pipeline = get_pipeline()
    await pipeline.remove_game(game_id)


async def force_game_update(game_id: str) -> Dict:
    """Force an immediate update for a specific game."""
    pipeline = get_pipeline()
    return await pipeline.force_update(game_id)


def get_cached_game_momentum(game_id: str) -> Optional[Dict]:
    """Get cached momentum data for a game."""
    pipeline = get_pipeline()
    return pipeline.get_cached_momentum(game_id)


def get_pipeline_statistics() -> Dict:
    """Get pipeline performance statistics."""
    pipeline = get_pipeline()
    return pipeline.get_pipeline_stats()


def set_websocket_broadcaster(callback):
    """Set the WebSocket broadcasting callback."""
    pipeline = get_pipeline()
    pipeline.set_websocket_callback(callback)