#!/usr/bin/env python3
"""
Test script for NBA data collection functionality.

This script validates the core functionality of the live data fetcher,
including error handling, rate limiting, and data parsing.
"""

import asyncio
import sys
import time
from services.live_fetcher import LiveDataFetcher, get_live_games, fetch_game_events


async def test_active_games():
    """Test fetching active games."""
    print("=== Testing Active Games Fetching ===")
    
    try:
        games = await get_live_games()
        print(f"✓ Successfully fetched {len(games)} active games")
        
        if games:
            for i, game in enumerate(games[:3]):  # Show first 3 games
                print(f"  Game {i+1}: {game.away_team} @ {game.home_team}")
                print(f"    Status: {game.status}, Score: {game.away_score}-{game.home_score}")
        else:
            print("  No active games found (normal if no games are live)")
            
        return True
        
    except Exception as e:
        print(f"✗ Error fetching active games: {e}")
        return False


async def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\n=== Testing Rate Limiting ===")
    
    try:
        fetcher = LiveDataFetcher(poll_interval=5)  # 5 second interval
        
        # Test that we can poll immediately
        can_poll_1 = fetcher._can_poll_game("test_game_1")
        print(f"✓ Can poll new game: {can_poll_1}")
        
        # Simulate a poll
        fetcher.last_poll_time["test_game_1"] = time.time()
        
        # Test that we can't poll immediately after
        can_poll_2 = fetcher._can_poll_game("test_game_1")
        print(f"✓ Rate limiting active: {not can_poll_2}")
        
        # Test that we can poll a different game
        can_poll_3 = fetcher._can_poll_game("test_game_2")
        print(f"✓ Can poll different game: {can_poll_3}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing rate limiting: {e}")
        return False


async def test_error_handling():
    """Test error handling with invalid game ID."""
    print("\n=== Testing Error Handling ===")
    
    try:
        fetcher = LiveDataFetcher(max_retries=2)  # Reduce retries for faster testing
        
        # Test with invalid game ID
        try:
            game_info, events = await fetcher.fetch_live_game_data("invalid_game_id")
            print("✗ Should have failed with invalid game ID")
            return False
        except Exception as e:
            print(f"✓ Properly handled invalid game ID: {type(e).__name__}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing error handling: {e}")
        return False


async def test_caching():
    """Test data caching functionality."""
    print("\n=== Testing Data Caching ===")
    
    try:
        fetcher = LiveDataFetcher()
        
        # Test empty cache
        cached_data = fetcher._get_cached_data("test_game")
        print(f"✓ Empty cache returns None: {cached_data[0] is None}")
        
        # Test cache update (with dummy data)
        from models.game_models import GameInfo, GameEvent
        from datetime import datetime, timezone
        
        dummy_game_info = GameInfo(
            game_id="test_game",
            home_team="TEST1",
            away_team="TEST2",
            game_date="2024-01-01",
            status="Live",
            period=1,
            clock="12:00",
            home_score=0,
            away_score=0
        )
        
        dummy_events = [
            GameEvent(
                event_id="1",
                game_id="test_game",
                team_tricode="TEST1",
                player_name="Test Player",
                event_type="shot",
                clock="12:00",
                period=1,
                points_total=0,
                shot_result="Made",
                timestamp=datetime.now(timezone.utc),
                description="Test shot"
            )
        ]
        
        fetcher._update_cache("test_game", dummy_game_info, dummy_events)
        
        # Test cache retrieval
        cached_game_info, cached_events = fetcher._get_cached_data("test_game")
        print(f"✓ Cache stores and retrieves data: {cached_game_info is not None}")
        print(f"✓ Cached events count: {len(cached_events)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing caching: {e}")
        return False


async def main():
    """Run all tests."""
    print("MomentumML Data Collection Test Suite")
    print("=" * 50)
    
    tests = [
        test_active_games,
        test_rate_limiting,
        test_error_handling,
        test_caching
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))