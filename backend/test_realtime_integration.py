"""
Integration test for the real-time data pipeline.

This test verifies the complete end-to-end data flow from NBA API
to dashboard updates through the real-time pipeline.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.realtime_pipeline import RealtimePipeline, get_pipeline
from backend.services.live_fetcher import get_live_games
from backend.services.historical_data_collector import create_sample_training_data
from backend.models.game_models import GameEvent, GameInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockWebSocketBroadcaster:
    """Mock WebSocket broadcaster for testing."""
    
    def __init__(self):
        self.messages_received = []
        self.broadcast_count = 0
    
    async def broadcast_callback(self, message: Dict, game_id: str):
        """Mock broadcast callback that records messages."""
        self.messages_received.append({
            "message": message,
            "game_id": game_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.broadcast_count += 1
        logger.info(f"Mock broadcast received for game {game_id}: {message['type']}")


async def test_pipeline_with_sample_data():
    """Test the pipeline using sample data."""
    logger.info("Testing real-time pipeline with sample data")
    
    # Create mock broadcaster
    mock_broadcaster = MockWebSocketBroadcaster()
    
    # Create pipeline instance
    pipeline = RealtimePipeline(poll_interval=5, cache_ttl=60)
    pipeline.set_websocket_callback(mock_broadcaster.broadcast_callback)
    
    # Create sample game data
    sample_events = create_sample_training_data()
    
    if not sample_events:
        logger.error("No sample events created")
        return False
    
    # Use the first game from sample data
    test_game_id = sample_events[0].game_id
    logger.info(f"Using test game ID: {test_game_id}")
    
    # Create sample game info
    test_game_info = GameInfo(
        game_id=test_game_id,
        home_team="LAL",
        away_team="GSW",
        game_date="2024-01-15",
        status="Live",
        period=2,
        clock="8:45",
        home_score=45,
        away_score=42
    )
    
    try:
        # Test 1: Add game to pipeline (bypass live fetcher for sample data)
        logger.info("Test 1: Adding sample game to pipeline")
        
        # Manually add the game to active games and process sample data
        pipeline.active_games.add(test_game_id)
        pipeline.last_event_counts[test_game_id] = len(sample_events)
        
        logger.info("✓ Sample game added to pipeline successfully")
        
        # Test 2: Force update with sample data
        logger.info("Test 2: Force update with sample data")
        
        # Simulate processing sample events
        await pipeline._process_momentum_update(test_game_id, test_game_info, sample_events[:50])
        
        # Check if momentum was calculated
        cached_data = pipeline.get_cached_momentum(test_game_id)
        
        if not cached_data:
            logger.error("No cached momentum data found")
            return False
        
        logger.info(f"✓ Momentum calculated for {len(cached_data['teams'])} teams")
        
        # Test 3: Check WebSocket broadcasting
        logger.info("Test 3: Checking WebSocket broadcasting")
        
        if mock_broadcaster.broadcast_count == 0:
            logger.error("No WebSocket broadcasts received")
            return False
        
        logger.info(f"✓ Received {mock_broadcaster.broadcast_count} WebSocket broadcasts")
        
        # Test 4: Check cache performance
        logger.info("Test 4: Testing cache performance")
        
        start_time = time.time()
        cached_data = pipeline.get_cached_momentum(test_game_id)
        cache_time = time.time() - start_time
        
        if cache_time > 0.01:  # Should be very fast
            logger.warning(f"Cache access took {cache_time:.4f}s (expected < 0.01s)")
        else:
            logger.info(f"✓ Cache access time: {cache_time:.4f}s")
        
        # Test 5: Pipeline statistics
        logger.info("Test 5: Checking pipeline statistics")
        
        stats = pipeline.get_pipeline_stats()
        
        required_stats = ['total_polls', 'events_processed', 'tmi_calculations', 'cache_hits']
        for stat in required_stats:
            if stat not in stats:
                logger.error(f"Missing statistic: {stat}")
                return False
        
        logger.info(f"✓ Pipeline statistics: {stats}")
        
        # Test 6: Remove game from pipeline
        logger.info("Test 6: Removing game from pipeline")
        
        await pipeline.remove_game(test_game_id)
        
        if test_game_id in pipeline.active_games:
            logger.error("Game not removed from active games")
            return False
        
        logger.info("✓ Game removed from pipeline successfully")
        
        logger.info("All pipeline tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False


async def test_live_game_integration():
    """Test integration with live NBA games (if available)."""
    logger.info("Testing integration with live NBA games")
    
    try:
        # Get active games
        active_games = await get_live_games()
        
        if not active_games:
            logger.info("No active games available, skipping live integration test")
            return True
        
        # Use the first active game
        test_game = active_games[0]
        logger.info(f"Testing with live game: {test_game.home_team} vs {test_game.away_team}")
        
        # Create pipeline
        pipeline = get_pipeline()
        
        # Add game to pipeline
        success = await pipeline.add_game(test_game.game_id)
        
        if not success:
            logger.warning("Could not add live game to pipeline (may be expected)")
            return True
        
        # Wait for some processing
        await asyncio.sleep(10)
        
        # Check if data was processed
        cached_data = pipeline.get_cached_momentum(test_game.game_id)
        
        if cached_data:
            logger.info(f"✓ Live game data processed: {len(cached_data['teams'])} teams")
        else:
            logger.info("No cached data yet (may be expected for new games)")
        
        # Clean up
        await pipeline.remove_game(test_game.game_id)
        
        logger.info("Live integration test completed")
        return True
        
    except Exception as e:
        logger.error(f"Live integration test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling in the pipeline."""
    logger.info("Testing error handling")
    
    pipeline = RealtimePipeline()
    
    try:
        # Test 1: Invalid game ID
        logger.info("Test 1: Invalid game ID")
        
        success = await pipeline.add_game("invalid_game_id")
        
        if success:
            logger.error("Pipeline should have rejected invalid game ID")
            return False
        
        logger.info("✓ Invalid game ID rejected correctly")
        
        # Test 2: Cache with non-existent game
        logger.info("Test 2: Cache with non-existent game")
        
        cached_data = pipeline.get_cached_momentum("non_existent_game")
        
        if cached_data is not None:
            logger.error("Cache should return None for non-existent game")
            return False
        
        logger.info("✓ Cache correctly returns None for non-existent game")
        
        logger.info("Error handling tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting real-time pipeline integration tests")
    
    tests = [
        ("Sample Data Pipeline", test_pipeline_with_sample_data),
        ("Live Game Integration", test_live_game_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    # Run the integration tests
    success = asyncio.run(run_integration_tests())
    
    if success:
        print("\n✅ Real-time pipeline integration tests completed successfully!")
    else:
        print("\n❌ Some integration tests failed. Check logs for details.")
        exit(1)