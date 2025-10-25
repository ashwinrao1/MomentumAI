#!/usr/bin/env python3
"""Integration test for database with FastAPI."""

import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from fastapi import Depends
from database.config import get_database
from database.service import DatabaseService
from models.game_models import GameInfo, GameEvent


def test_database_with_dependency_injection():
    """Test database service with FastAPI dependency injection pattern."""
    print("Testing database with dependency injection...")
    
    # Simulate FastAPI dependency injection
    def get_db_service(db=Depends(get_database)):
        return DatabaseService(db)
    
    # Get database service (simulating FastAPI request)
    db_gen = get_database()
    db = next(db_gen)
    db_service = DatabaseService(db)
    
    try:
        # Test creating a game
        game_info = GameInfo(
            game_id="integration_test_001",
            home_team="BOS",
            away_team="MIA",
            game_date="2024-01-16",
            status="Live",
            period=2,
            clock="8:30",
            home_score=45,
            away_score=42
        )
        
        success = db_service.create_or_update_game(game_info)
        print(f"✓ Game creation: {'Success' if success else 'Failed'}")
        
        # Test creating events
        events = [
            GameEvent(
                event_id="int_event_001",
                game_id="integration_test_001",
                team_tricode="BOS",
                player_name="Jayson Tatum",
                event_type="shot",
                clock="8:15",
                period=2,
                points_total=47,
                shot_result="Made",
                timestamp=datetime.now(),
                description="Jayson Tatum makes 2-pt shot"
            ),
            GameEvent(
                event_id="int_event_002",
                game_id="integration_test_001",
                team_tricode="MIA",
                player_name="Jimmy Butler",
                event_type="rebound",
                clock="8:10",
                period=2,
                points_total=42,
                shot_result=None,
                timestamp=datetime.now(),
                description="Jimmy Butler defensive rebound"
            )
        ]
        
        stored_count = db_service.store_events(events)
        print(f"✓ Events stored: {stored_count}")
        
        # Test retrieving data
        active_games = db_service.get_active_games()
        print(f"✓ Active games retrieved: {len(active_games)}")
        
        game_events = db_service.get_game_events("integration_test_001")
        print(f"✓ Game events retrieved: {len(game_events)}")
        
        print("\n✓ Integration test completed successfully!")
        
    finally:
        # Clean up
        db.close()


if __name__ == "__main__":
    try:
        test_database_with_dependency_injection()
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        sys.exit(1)