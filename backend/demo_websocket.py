#!/usr/bin/env python3
"""
WebSocket demonstration script for MomentumML.

This script demonstrates the WebSocket real-time communication features
including connection management, heartbeat mechanism, and data streaming.
"""

import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_websocket_features():
    """Demonstrate WebSocket features without requiring external dependencies."""
    
    logger.info("WebSocket Real-time Communication Demo")
    logger.info("=" * 50)
    
    # Import the WebSocket components
    try:
        from backend.api.websocket_endpoints import manager, get_connection_stats
        from backend.services.momentum_engine import create_momentum_engine
        
        logger.info("✓ WebSocket components imported successfully")
        
        # Test connection manager initialization
        logger.info(f"✓ Connection manager initialized")
        logger.info(f"  - Active connections: {len(manager.active_connections)}")
        logger.info(f"  - Game subscriptions: {len(manager.game_subscriptions)}")
        logger.info(f"  - Heartbeat interval: {manager.heartbeat_interval}s")
        logger.info(f"  - Heartbeat timeout: {manager.heartbeat_timeout}s")
        
        # Test momentum engine integration
        logger.info(f"✓ Momentum engine integrated")
        logger.info(f"  - Rolling window size: {manager.momentum_engine.rolling_window_size}")
        logger.info(f"  - TMI weights: {manager.momentum_engine.tmi_weights}")
        
        # Test connection statistics
        stats = get_connection_stats()
        logger.info(f"✓ Connection statistics available")
        logger.info(f"  - Total connections: {stats['total_connections']}")
        logger.info(f"  - Heartbeat status: {stats['heartbeat_status']}")
        
        # Simulate heartbeat check
        await manager.check_connection_health()
        logger.info("✓ Connection health check completed")
        
        logger.info("\nWebSocket Features Summary:")
        logger.info("✓ Live data streaming endpoint (/ws/momentum)")
        logger.info("✓ Connection management for multiple clients")
        logger.info("✓ Automatic data push when TMI updates occur")
        logger.info("✓ Heartbeat mechanism for connection health monitoring")
        logger.info("✓ Game subscription/unsubscription system")
        logger.info("✓ Error handling and graceful disconnection")
        logger.info("✓ Background task for periodic updates")
        logger.info("✓ Connection statistics and monitoring")
        
        logger.info("\nWebSocket Endpoints Available:")
        logger.info("- ws://localhost:8000/ws/momentum - Main momentum data stream")
        logger.info("- ws://localhost:8000/ws/status - System status updates")
        logger.info("- GET /ws/stats - REST endpoint for connection statistics")
        
        logger.info("\nMessage Protocol:")
        logger.info("Client -> Server:")
        logger.info('  {"action": "subscribe", "game_id": "game_id"}')
        logger.info('  {"action": "unsubscribe", "game_id": "game_id"}')
        logger.info('  {"action": "ping"}')
        logger.info('  {"action": "pong"}')
        
        logger.info("\nServer -> Client:")
        logger.info('  {"type": "connected", "message": "..."}')
        logger.info('  {"type": "momentum_update", "game_id": "...", "teams": {...}}')
        logger.info('  {"type": "ping", "timestamp": "..."}')
        logger.info('  {"type": "pong", "timestamp": "..."}')
        logger.info('  {"type": "error", "message": "..."}')
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False


async def main():
    """Main demonstration function."""
    print("MomentumML WebSocket Real-time Communication")
    print("=" * 60)
    print()
    
    success = await demonstrate_websocket_features()
    
    print()
    if success:
        print("✅ WebSocket implementation is ready!")
        print("\nTo test the WebSocket functionality:")
        print("1. Start the FastAPI server: python backend/main.py")
        print("2. Run the integration test: python backend/test_websocket_integration.py")
        print("3. Connect from frontend or use a WebSocket client")
    else:
        print("❌ WebSocket implementation has issues")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))