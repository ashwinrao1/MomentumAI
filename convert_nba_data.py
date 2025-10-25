#!/usr/bin/env python3
"""
Convert NBA pickle data to clean CSV format without class dependencies.
"""

import os
import pickle
import pandas as pd
from tqdm import tqdm
import json

def extract_event_data(event):
    """Extract data from event object without class dependencies."""
    if hasattr(event, '__dict__'):
        data = event.__dict__.copy()
    elif isinstance(event, dict):
        data = event.copy()
    else:
        # Try to extract basic attributes
        data = {}
        for attr in ['event_id', 'game_id', 'team_tricode', 'player_name', 
                    'event_type', 'clock', 'period', 'points_total', 
                    'shot_result', 'timestamp', 'description']:
            try:
                data[attr] = getattr(event, attr, None)
            except:
                data[attr] = None
    
    # Clean up the data
    clean_data = {}
    for key, value in data.items():
        if key.startswith('_'):
            continue
        
        # Convert datetime to string
        if hasattr(value, 'isoformat'):
            clean_data[key] = value.isoformat()
        elif value is None:
            clean_data[key] = None
        else:
            clean_data[key] = str(value) if not isinstance(value, (int, float, bool)) else value
    
    return clean_data

def convert_pickle_files():
    """Convert all pickle files to clean format."""
    cache_dir = "data/nba_cache"
    output_file = "data/nba_clean_dataset.csv"
    
    # Get all pickle files
    game_files = [f for f in os.listdir(cache_dir) if f.startswith('pbp_') and f.endswith('.pkl')]
    
    print(f"Found {len(game_files)} game files")
    
    all_events = []
    successful_games = 0
    
    # Process files with progress bar
    for game_file in tqdm(game_files, desc="Converting games"):
        try:
            file_path = os.path.join(cache_dir, game_file)
            
            # Try different loading methods
            events = None
            
            # Method 1: Direct pickle load
            try:
                with open(file_path, 'rb') as f:
                    events = pickle.load(f)
            except Exception as e1:
                # Method 2: Load with different protocol
                try:
                    with open(file_path, 'rb') as f:
                        events = pickle.load(f, encoding='latin1')
                except Exception as e2:
                    continue
            
            if events:
                # Extract data from each event
                for event in events:
                    try:
                        event_data = extract_event_data(event)
                        if event_data.get('game_id'):  # Only add if we have a game_id
                            all_events.append(event_data)
                    except Exception as e:
                        continue
                
                successful_games += 1
                
        except Exception as e:
            continue
    
    print(f"Successfully processed {successful_games} games")
    print(f"Extracted {len(all_events)} events")
    
    if all_events:
        # Convert to DataFrame and save
        df = pd.DataFrame(all_events)
        
        # Clean up columns
        df = df.dropna(subset=['game_id'])  # Remove events without game_id
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"Saved clean dataset to {output_file}")
        print(f"Final dataset: {len(df)} events from {df['game_id'].nunique()} games")
        
        # Save summary
        summary = {
            'total_events': len(df),
            'total_games': df['game_id'].nunique(),
            'event_types': df['event_type'].value_counts().to_dict(),
            'teams': sorted(df['team_tricode'].unique().tolist()),
            'sample_columns': df.columns.tolist()
        }
        
        with open('data/nba_clean_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return df
    else:
        print("No events could be extracted")
        return None

if __name__ == "__main__":
    convert_pickle_files()