#!/usr/bin/env python3
"""Simple test script to verify database functionality."""

import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from database.service import DatabaseService
from models.game_models import GameInfo, GameEvent


def test_database_operations():
    """Test basic database operations."""
    print("Testing database operations...")
    
    # Test game creation
    print("\n1. Testing game creation...")
    game_info = GameInfo(
        game_id="test_game_001",
        home_team="LAL",
        away_team="GSW",
        game_date="2024-01-15",
        status="Live",
        period=1,
        clock="12:00",
        home_score=0,
        away_score=0
    )
    
    with DatabaseService() as db_service:
        success = db_service.create_or_update_game(game_info)
        print(f"Game creation: {'✓' if success else '✗'}")
        
        # Test retrieving active games
        print("\n2. Testing active games retrieval...")
        active_games = db_service.get_active_games()
        print(f"Active games found: {len(active_games)}")
        for game in active_games:
            print(f"  - {game.game_id}: {game.away_team} @ {game.home_team}")
        
        # Test event creation
        print("\n3. Testing event creation...")
        test_event = GameEvent(
            event_id="test_event_001",
            game_id="test_game_001",
            team_tricode="LAL",
            player_name="LeBron James",
            event_type="shot",
            clock="11:45",
            period=1,
            points_total=2,
            shot_result="Made",
            timestamp=datetime.now(),
            description="LeBron James makes 2-pt shot"
        )
        
        stored_count = db_service.store_events([test_event])
        print(f"Events stored: {stored_count}")
        
        # Test event retrieval
        print("\n4. Testing event retrieval...")
        events = db_service.get_game_events("test_game_001")
        print(f"Events retrieved: {len(events)}")
        for event in events:
            print(f"  - {event.event_type}: {event.player_name} ({event.clock})")
        
        # Test latest event timestamp
        print("\n5. Testing latest event timestamp...")
        latest_timestamp = db_service.get_latest_event_timestamp("test_game_001")
        print(f"Latest event timestamp: {latest_timestamp}")
        
        print("\n✓ All database tests completed successfully!")


if __name__ == "__main__":
    try:
        test_database_operations()
    except Exception as e:
        print(f"✗ Database test failed: {str(e)}")
        sys.exit(1)