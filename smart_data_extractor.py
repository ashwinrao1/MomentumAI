#!/usr/bin/env python3
"""
Smart extractor that can handle pickled data with class dependencies.
"""

import pickle
import pandas as pd
import sys
import os
from tqdm import tqdm
import json

class GameEvent:
    """Dummy GameEvent class to handle unpickling."""
    def __init__(self, *args, **kwargs):
        # Store all arguments as attributes
        for i, arg in enumerate(args):
            setattr(self, f'arg_{i}', arg)
        
        # Store all keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __setstate__(self, state):
        """Handle pickle state restoration."""
        self.__dict__.update(state)

class GameInfo:
    """Dummy GameInfo class."""
    def __init__(self, *args, **kwargs):
        for i, arg in enumerate(args):
            setattr(self, f'arg_{i}', arg)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __setstate__(self, state):
        self.__dict__.update(state)

# Create dummy modules to handle imports
class DummyModule:
    GameEvent = GameEvent
    GameInfo = GameInfo
    
    def __getattr__(self, name):
        return GameEvent

def setup_dummy_modules():
    """Set up dummy modules to handle missing imports."""
    dummy = DummyModule()
    
    # Add to sys.modules
    sys.modules['models'] = dummy
    sys.modules['models.game_models'] = dummy
    sys.modules['backend'] = dummy
    sys.modules['backend.models'] = dummy
    sys.modules['backend.models.game_models'] = dummy

def extract_event_data(event):
    """Extract data from event object."""
    data = {}
    
    # Get all attributes
    if hasattr(event, '__dict__'):
        for key, value in event.__dict__.items():
            if key.startswith('_'):
                continue
            
            # Convert datetime objects
            if hasattr(value, 'isoformat'):
                data[key] = value.isoformat()
            elif value is None:
                data[key] = None
            else:
                data[key] = value
    
    return data

def extract_main_dataset():
    """Extract from the main dataset file."""
    print("🎯 Extracting from main dataset file...")
    
    # Set up dummy modules
    setup_dummy_modules()
    
    main_file = "data/nba_cache/nba_5year_dataset.pkl"
    
    try:
        with open(main_file, 'rb') as f:
            print("📖 Loading main dataset...")
            data = pickle.load(f)
            
            print(f"✅ Loaded data type: {type(data)}")
            
            if isinstance(data, list):
                print(f"📊 Processing {len(data)} events...")
                
                all_events = []
                
                # Process with progress bar
                for event in tqdm(data, desc="Extracting events"):
                    try:
                        event_data = extract_event_data(event)
                        if event_data:
                            all_events.append(event_data)
                    except Exception as e:
                        continue
                
                if all_events:
                    df = pd.DataFrame(all_events)
                    
                    # Clean up the data
                    if 'game_id' in df.columns:
                        df = df.dropna(subset=['game_id'])
                    
                    print(f"✅ Extracted {len(df)} events from {df['game_id'].nunique() if 'game_id' in df.columns else 'unknown'} games")
                    
                    # Save to CSV
                    output_file = "data/nba_extracted_dataset.csv"
                    df.to_csv(output_file, index=False)
                    
                    print(f"💾 Saved to {output_file}")
                    
                    # Save summary
                    summary = {
                        'total_events': len(df),
                        'total_games': df['game_id'].nunique() if 'game_id' in df.columns else 0,
                        'columns': df.columns.tolist(),
                        'event_types': df['event_type'].value_counts().to_dict() if 'event_type' in df.columns else {},
                        'teams': sorted(df['team_tricode'].unique().tolist()) if 'team_tricode' in df.columns else [],
                        'extraction_date': pd.Timestamp.now().isoformat()
                    }
                    
                    with open('data/nba_extracted_summary.json', 'w') as f:
                        json.dump(summary, f, indent=2, default=str)
                    
                    return df
                
    except Exception as e:
        print(f"❌ Error extracting main dataset: {e}")
        return None

def extract_from_individual_files(max_files=100):
    """Extract from individual game files as backup."""
    print(f"🎯 Extracting from individual files (max {max_files})...")
    
    setup_dummy_modules()
    
    cache_dir = "data/nba_cache"
    pbp_files = [f for f in os.listdir(cache_dir) if f.startswith('pbp_') and f.endswith('.pkl')]
    
    # Limit files for testing
    test_files = pbp_files[:max_files]
    
    all_events = []
    successful_files = 0
    
    for filename in tqdm(test_files, desc="Processing files"):
        filepath = os.path.join(cache_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                if isinstance(data, list):
                    for event in data:
                        try:
                            event_data = extract_event_data(event)
                            if event_data:
                                all_events.append(event_data)
                        except:
                            continue
                
                successful_files += 1
                
        except Exception as e:
            continue
    
    print(f"📊 Processed {successful_files}/{len(test_files)} files")
    print(f"📈 Extracted {len(all_events)} events")
    
    if all_events:
        df = pd.DataFrame(all_events)
        
        if 'game_id' in df.columns:
            df = df.dropna(subset=['game_id'])
        
        output_file = "data/nba_individual_extracted.csv"
        df.to_csv(output_file, index=False)
        
        print(f"💾 Saved to {output_file}")
        return df
    
    return None

if __name__ == "__main__":
    print("🚀 SMART NBA DATA EXTRACTION")
    print("=" * 50)
    
    # Try main dataset first
    df = extract_main_dataset()
    
    if df is None:
        print("\n⚠️  Main dataset extraction failed, trying individual files...")
        df = extract_from_individual_files(max_files=200)
    
    if df is not None:
        print(f"\n🎉 SUCCESS!")
        print(f"📊 Final dataset: {len(df)} events")
        if 'game_id' in df.columns:
            print(f"🏀 Games: {df['game_id'].nunique()}")
        if 'team_tricode' in df.columns:
            print(f"🏟️  Teams: {df['team_tricode'].nunique()}")
        if 'event_type' in df.columns:
            print(f"📋 Event types: {df['event_type'].value_counts().to_dict()}")
    else:
        print(f"\n❌ Extraction failed completely")