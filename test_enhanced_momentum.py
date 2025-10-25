#!/usr/bin/env python3
"""
Test the enhanced momentum prediction system.
"""

import sys
import os
sys.path.append('backend')

from backend.services.enhanced_momentum_predictor import get_enhanced_momentum_analysis, get_momentum_visualization_data
from datetime import datetime
import json

def create_test_events():
    """Create test events to verify the system works."""
    
    # Simulate a Lakers vs Warriors game with momentum shifts
    events = [
        # Lakers scoring run
        {'event_id': '1', 'game_id': 'test_001', 'team_tricode': 'LAL', 'player_name': 'LeBron James', 
         'event_type': 'shot', 'shot_result': 'Made', 'points_total': 2, 'period': 1, 'clock': '11:30', 
         'timestamp': '2025-10-25T10:00:00', 'description': 'LeBron James makes 2-pt shot'},
        
        {'event_id': '2', 'game_id': 'test_001', 'team_tricode': 'LAL', 'player_name': 'Anthony Davis', 
         'event_type': 'steal', 'shot_result': None, 'points_total': 0, 'period': 1, 'clock': '11:15', 
         'timestamp': '2025-10-25T10:00:15', 'description': 'Anthony Davis steal'},
        
        {'event_id': '3', 'game_id': 'test_001', 'team_tricode': 'LAL', 'player_name': 'Russell Westbrook', 
         'event_type': 'shot', 'shot_result': 'Made', 'points_total': 3, 'period': 1, 'clock': '11:00', 
         'timestamp': '2025-10-25T10:00:30', 'description': 'Russell Westbrook makes 3-pt shot'},
        
        # Warriors response
        {'event_id': '4', 'game_id': 'test_001', 'team_tricode': 'GSW', 'player_name': 'Stephen Curry', 
         'event_type': 'shot', 'shot_result': 'Made', 'points_total': 3, 'period': 1, 'clock': '10:45', 
         'timestamp': '2025-10-25T10:00:45', 'description': 'Stephen Curry makes 3-pt shot'},
        
        {'event_id': '5', 'game_id': 'test_001', 'team_tricode': 'GSW', 'player_name': 'Klay Thompson', 
         'event_type': 'shot', 'shot_result': 'Made', 'points_total': 2, 'period': 1, 'clock': '10:30', 
         'timestamp': '2025-10-25T10:01:00', 'description': 'Klay Thompson makes 2-pt shot'},
        
        # More Lakers momentum
        {'event_id': '6', 'game_id': 'test_001', 'team_tricode': 'LAL', 'player_name': 'LeBron James', 
         'event_type': 'assist', 'shot_result': None, 'points_total': 0, 'period': 1, 'clock': '10:15', 
         'timestamp': '2025-10-25T10:01:15', 'description': 'LeBron James assist'},
        
        {'event_id': '7', 'game_id': 'test_001', 'team_tricode': 'LAL', 'player_name': 'Anthony Davis', 
         'event_type': 'shot', 'shot_result': 'Made', 'points_total': 2, 'period': 1, 'clock': '10:00', 
         'timestamp': '2025-10-25T10:01:30', 'description': 'Anthony Davis makes 2-pt shot'},
        
        # Warriors turnover
        {'event_id': '8', 'game_id': 'test_001', 'team_tricode': 'GSW', 'player_name': 'Draymond Green', 
         'event_type': 'turnover', 'shot_result': None, 'points_total': 0, 'period': 1, 'clock': '9:45', 
         'timestamp': '2025-10-25T10:01:45', 'description': 'Draymond Green turnover'},
        
        # More Lakers momentum
        {'event_id': '9', 'game_id': 'test_001', 'team_tricode': 'LAL', 'player_name': 'Russell Westbrook', 
         'event_type': 'shot', 'shot_result': 'Made', 'points_total': 2, 'period': 1, 'clock': '9:30', 
         'timestamp': '2025-10-25T10:02:00', 'description': 'Russell Westbrook makes 2-pt shot'},
        
        {'event_id': '10', 'game_id': 'test_001', 'team_tricode': 'LAL', 'player_name': 'Anthony Davis', 
         'event_type': 'block', 'shot_result': None, 'points_total': 0, 'period': 1, 'clock': '9:15', 
         'timestamp': '2025-10-25T10:02:15', 'description': 'Anthony Davis block'}
    ]
    
    return events

def test_enhanced_momentum():
    """Test the enhanced momentum prediction system."""
    
    print("🧪 TESTING ENHANCED MOMENTUM SYSTEM")
    print("="*50)
    
    # Create test events
    events = create_test_events()
    print(f"📊 Created {len(events)} test events")
    
    try:
        # Test enhanced momentum analysis
        print("\n🔍 Testing enhanced momentum analysis...")
        analysis = get_enhanced_momentum_analysis(events)
        
        print("✅ Enhanced analysis results:")
        print(json.dumps(analysis, indent=2, default=str))
        
        # Test visualization data
        print("\n🎨 Testing visualization data...")
        viz_data = get_momentum_visualization_data(events)
        
        print("✅ Visualization data:")
        print(json.dumps(viz_data, indent=2, default=str))
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_momentum()
    exit(0 if success else 1)