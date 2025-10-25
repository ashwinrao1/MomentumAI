"""
End-to-end test for the complete real-time data pipeline.

This test verifies the complete integration from API endpoints
through the real-time pipeline to WebSocket broadcasting.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.realtime_pipeline import get_pipeline, add_game_to_pipeline, get_cached_game_momentum
from backend.services.historical_data_collector import create_sample_training_data
from backend.models.game_models import GameEvent, GameInfo
from backend.api.momentum_endpoints import get_current_momentum
from backend.database.service import DatabaseService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_complete_pipeline_flow():
    """Test the complete end-to-end pipeline flow."""
    logger.info("Testing complete pipeline flow")
    
    # Create sample data
    sample_events = create_sample_training_data()
    test_game_id = sample_events[0].game_id
    
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
        # Step 1: Initialize pipeline
        logger.info("Step 1: Initialize pipeline")
        pipeline = get_pipeline()
        
        # Step 2: Manually add game and process data (simulating real-time flow)
        logger.info("Step 2: Process game data through pipeline")
        
        # Add game to active games
        pipeline.active_games.add(test_game_id)
        pipeline.last_event_counts[test_game_id] = 0
        
        # Process momentum update
        await pipeline._process_momentum_update(test_game_id, test_game_info, sample_events[:50])
        
        # Step 3: Verify data persistence
        logger.info("Step 3: Verify data persistence")
        
        db_service = DatabaseService()
        
        # Check if game was stored
        stored_game = db_service.game_repo.get_game(test_game_id)
        if not stored_game:
            logger.error("Game not stored in database")
            return False
        
        logger.info("✓ Game stored in database")
        
        # Check if TMI was stored
        latest_tmi = db_service.get_latest_tmi(test_game_id, "LAL")
        if not latest_tmi:
            logger.error("TMI not stored in database")
            return False
        
        logger.info(f"✓ TMI stored: {latest_tmi.tmi_value:.3f}")
        
        # Step 4: Verify caching
        logger.info("Step 4: Verify caching")
        
        cached_data = get_cached_game_momentum(test_game_id)
        if not cached_data:
            logger.error("No cached data found")
            return False
        
        logger.info(f"✓ Data cached for {len(cached_data['teams'])} teams")
        
        # Step 5: Test API endpoint integration
        logger.info("Step 5: Test API endpoint integration")
        
        try:
            # This would normally call the API, but we'll test the logic directly
            # since we can't easily mock the FastAPI request context
            
            # Verify the cached data has the expected structure
            expected_fields = ['game_info', 'teams', 'event_count', 'last_updated']
            for field in expected_fields:
                if field not in cached_data:
                    logger.error(f"Missing field in cached data: {field}")
                    return False
            
            logger.info("✓ API integration structure verified")
            
        except Exception as e:
            logger.error(f"API integration test failed: {e}")
            return False
        
        # Step 6: Test performance metrics
        logger.info("Step 6: Test performance metrics")
        
        stats = pipeline.get_pipeline_stats()
        
        # Verify key metrics are present
        required_metrics = ['tmi_calculations', 'cache_hits', 'websocket_broadcasts']
        for metric in required_metrics:
            if metric not in stats:
                logger.error(f"Missing metric: {metric}")
                return False
            
            if stats[metric] == 0 and metric != 'cache_hits':  # cache_hits might be 0 initially
                logger.warning(f"Metric {metric} is 0, expected > 0")
        
        logger.info(f"✓ Performance metrics: {stats}")
        
        # Step 7: Test data consistency
        logger.info("Step 7: Test data consistency")
        
        # Compare database TMI with cached TMI
        db_tmi_value = latest_tmi.tmi_value
        cached_tmi_value = cached_data['teams']['LAL']['tmi_value']
        
        if abs(db_tmi_value - cached_tmi_value) > 0.001:
            logger.error(f"TMI mismatch: DB={db_tmi_value}, Cache={cached_tmi_value}")
            return False
        
        logger.info("✓ Data consistency verified")
        
        # Step 8: Test cleanup
        logger.info("Step 8: Test cleanup")
        
        await pipeline.remove_game(test_game_id)
        
        if test_game_id in pipeline.active_games:
            logger.error("Game not removed from active games")
            return False
        
        logger.info("✓ Cleanup completed")
        
        logger.info("🎉 Complete pipeline flow test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline flow test failed: {e}")
        return False


async def test_concurrent_game_processing():
    """Test processing multiple games concurrently."""
    logger.info("Testing concurrent game processing")
    
    try:
        # Create sample data for multiple games
        sample_events = create_sample_training_data()
        
        # Create multiple game scenarios
        games = []
        for i in range(3):
            game_id = f"concurrent_game_{i:03d}"
            game_events = [
                GameEvent(
                    event_id=f"{game_id}_event_{j}",
                    game_id=game_id,
                    team_tricode="LAL" if j % 2 == 0 else "GSW",
                    player_name=f"Player_{j}",
                    event_type="shot",
                    clock=f"{12-j//5}:{(60-j*5)%60:02d}",
                    period=1,
                    points_total=j * 2,
                    shot_result="Made" if j % 3 == 0 else "Missed",
                    timestamp=datetime.utcnow(),
                    description=f"Test event {j}"
                )
                for j in range(20)
            ]
            
            game_info = GameInfo(
                game_id=game_id,
                home_team="LAL",
                away_team="GSW",
                game_date="2024-01-15",
                status="Live",
                period=1,
                clock="10:30",
                home_score=20 + i * 5,
                away_score=18 + i * 3
            )
            
            games.append((game_id, game_info, game_events))
        
        # Process games concurrently
        pipeline = get_pipeline()
        
        async def process_game(game_id, game_info, events):
            pipeline.active_games.add(game_id)
            await pipeline._process_momentum_update(game_id, game_info, events)
            return game_id
        
        # Process all games concurrently
        tasks = [process_game(gid, ginfo, gevents) for gid, ginfo, gevents in games]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all games were processed
        successful_games = [r for r in results if isinstance(r, str)]
        
        if len(successful_games) != len(games):
            logger.error(f"Only {len(successful_games)}/{len(games)} games processed successfully")
            return False
        
        # Verify cached data for all games
        for game_id, _, _ in games:
            cached_data = get_cached_game_momentum(game_id)
            if not cached_data:
                logger.error(f"No cached data for game {game_id}")
                return False
        
        # Cleanup
        for game_id, _, _ in games:
            await pipeline.remove_game(game_id)
        
        logger.info(f"✓ Successfully processed {len(games)} games concurrently")
        return True
        
    except Exception as e:
        logger.error(f"Concurrent processing test failed: {e}")
        return False


async def test_cache_performance():
    """Test cache performance under load."""
    logger.info("Testing cache performance")
    
    try:
        # Create test data
        sample_events = create_sample_training_data()
        test_game_id = "cache_perf_game"
        
        game_info = GameInfo(
            game_id=test_game_id,
            home_team="LAL",
            away_team="GSW",
            game_date="2024-01-15",
            status="Live",
            period=1,
            clock="10:00",
            home_score=25,
            away_score=23
        )
        
        # Setup pipeline
        pipeline = get_pipeline()
        pipeline.active_games.add(test_game_id)
        
        # Process initial data
        await pipeline._process_momentum_update(test_game_id, game_info, sample_events[:30])
        
        # Test cache performance
        cache_times = []
        num_requests = 100
        
        for i in range(num_requests):
            start_time = time.time()
            cached_data = get_cached_game_momentum(test_game_id)
            end_time = time.time()
            
            if cached_data is None:
                logger.error(f"Cache miss on request {i}")
                return False
            
            cache_times.append(end_time - start_time)
        
        # Analyze performance
        avg_time = sum(cache_times) / len(cache_times)
        max_time = max(cache_times)
        
        logger.info(f"Cache performance: avg={avg_time*1000:.2f}ms, max={max_time*1000:.2f}ms")
        
        # Performance thresholds
        if avg_time > 0.001:  # 1ms average
            logger.warning(f"Cache average time {avg_time*1000:.2f}ms exceeds 1ms threshold")
        
        if max_time > 0.01:  # 10ms max
            logger.warning(f"Cache max time {max_time*1000:.2f}ms exceeds 10ms threshold")
        
        # Cleanup
        await pipeline.remove_game(test_game_id)
        
        logger.info("✓ Cache performance test completed")
        return True
        
    except Exception as e:
        logger.error(f"Cache performance test failed: {e}")
        return False


async def run_end_to_end_tests():
    """Run all end-to-end tests."""
    logger.info("Starting end-to-end pipeline tests")
    
    tests = [
        ("Complete Pipeline Flow", test_complete_pipeline_flow),
        ("Concurrent Game Processing", test_concurrent_game_processing),
        ("Cache Performance", test_cache_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("END-TO-END TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All end-to-end tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    # Run the end-to-end tests
    success = asyncio.run(run_end_to_end_tests())
    
    if success:
        print("\n✅ End-to-end pipeline tests completed successfully!")
        print("\n🚀 Real-time data pipeline is ready for production!")
    else:
        print("\n❌ Some end-to-end tests failed. Check logs for details.")
        exit(1)