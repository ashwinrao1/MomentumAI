#!/usr/bin/env python3
"""
Extract data from our collected NBA files using a different approach.
"""

import os
import pandas as pd
import json
from tqdm import tqdm

def extract_data_smart():
    """Try multiple approaches to extract our collected data."""
    
    # Check what files we have
    cache_dir = "data/nba_cache"
    
    print("📁 Checking available data files...")
    
    # List all files
    all_files = os.listdir(cache_dir)
    print(f"Found {len(all_files)} files in cache directory")
    
    # Check for the main dataset file
    main_file = "data/nba_cache/nba_5year_dataset.pkl"
    if os.path.exists(main_file):
        print(f"📊 Main dataset file exists: {os.path.getsize(main_file) / (1024*1024):.1f} MB")
        
        # Try to read it with different methods
        try:
            # Method 1: Try to read just the structure
            import pickle
            with open(main_file, 'rb') as f:
                # Read first few bytes to see structure
                data = f.read(1000)
                print(f"File starts with: {data[:100]}")
        except Exception as e:
            print(f"Error reading main file: {e}")
    
    # Check individual game files
    pbp_files = [f for f in all_files if f.startswith('pbp_') and f.endswith('.pkl')]
    print(f"📋 Found {len(pbp_files)} individual game files")
    
    if pbp_files:
        # Try to read one file to see structure
        sample_file = os.path.join(cache_dir, pbp_files[0])
        print(f"🔍 Examining sample file: {pbp_files[0]}")
        
        try:
            import pickle
            with open(sample_file, 'rb') as f:
                # Try different loading methods
                try:
                    data = pickle.load(f)
                    print(f"✅ Successfully loaded sample file")
                    print(f"Type: {type(data)}")
                    if hasattr(data, '__len__'):
                        print(f"Length: {len(data)}")
                    
                    if isinstance(data, list) and len(data) > 0:
                        sample_event = data[0]
                        print(f"Sample event type: {type(sample_event)}")
                        
                        # Try to extract attributes without importing classes
                        if hasattr(sample_event, '__dict__'):
                            attrs = list(sample_event.__dict__.keys())
                            print(f"Event attributes: {attrs[:10]}...")  # Show first 10
                        
                        return True
                        
                except Exception as e:
                    print(f"Error with pickle.load: {e}")
                    
                    # Try with different encoding
                    f.seek(0)
                    try:
                        data = pickle.load(f, encoding='latin1')
                        print(f"✅ Loaded with latin1 encoding")
                        return True
                    except Exception as e2:
                        print(f"Error with latin1: {e2}")
        
        except Exception as e:
            print(f"Error opening sample file: {e}")
    
    # Check for any CSV files
    csv_files = [f for f in all_files if f.endswith('.csv')]
    if csv_files:
        print(f"📄 Found {len(csv_files)} CSV files: {csv_files}")
    
    # Check for JSON files
    json_files = [f for f in all_files if f.endswith('.json')]
    if json_files:
        print(f"📄 Found {len(json_files)} JSON files: {json_files}")
        
        # Read summary if available
        summary_file = os.path.join(cache_dir, 'nba_5year_dataset.pkl_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                print(f"📊 Dataset summary:")
                print(f"   Total events: {summary.get('total_events', 'Unknown')}")
                print(f"   Total games: {summary.get('total_games', 'Unknown')}")
                print(f"   Event types: {summary.get('event_type_distribution', {})}")
    
    return False

def try_manual_extraction():
    """Try to manually extract data without class dependencies."""
    cache_dir = "data/nba_cache"
    pbp_files = [f for f in os.listdir(cache_dir) if f.startswith('pbp_') and f.endswith('.pkl')]
    
    print(f"\n🔧 Attempting manual extraction from {len(pbp_files)} files...")
    
    all_events = []
    successful_files = 0
    
    # Try first 10 files as a test
    test_files = pbp_files[:10]
    
    for filename in tqdm(test_files, desc="Testing extraction"):
        filepath = os.path.join(cache_dir, filename)
        
        try:
            # Read raw pickle data
            import pickle
            with open(filepath, 'rb') as f:
                # Try to load without class dependencies
                import sys
                
                # Create a dummy module to catch missing imports
                class DummyModule:
                    def __getattr__(self, name):
                        return DummyClass
                
                class DummyClass:
                    def __init__(self, *args, **kwargs):
                        self.__dict__.update(kwargs)
                        for i, arg in enumerate(args):
                            setattr(self, f'arg_{i}', arg)
                
                # Temporarily replace missing modules
                original_modules = {}
                missing_modules = ['models.game_models', 'backend.models.game_models']
                
                for module_name in missing_modules:
                    if module_name not in sys.modules:
                        sys.modules[module_name] = DummyModule()
                        original_modules[module_name] = None
                    else:
                        original_modules[module_name] = sys.modules[module_name]
                        sys.modules[module_name] = DummyModule()
                
                try:
                    data = pickle.load(f)
                    
                    # Extract data from objects
                    if isinstance(data, list):
                        for item in data:
                            event_dict = {}
                            
                            # Try to extract all attributes
                            if hasattr(item, '__dict__'):
                                for key, value in item.__dict__.items():
                                    if not key.startswith('_'):
                                        # Convert to simple types
                                        if hasattr(value, 'isoformat'):  # datetime
                                            event_dict[key] = value.isoformat()
                                        elif value is None:
                                            event_dict[key] = None
                                        else:
                                            event_dict[key] = str(value)
                            
                            if event_dict:
                                all_events.append(event_dict)
                    
                    successful_files += 1
                    
                finally:
                    # Restore original modules
                    for module_name, original in original_modules.items():
                        if original is None:
                            if module_name in sys.modules:
                                del sys.modules[module_name]
                        else:
                            sys.modules[module_name] = original
        
        except Exception as e:
            continue
    
    print(f"\n📊 Extraction results:")
    print(f"   Successful files: {successful_files}/{len(test_files)}")
    print(f"   Total events extracted: {len(all_events)}")
    
    if all_events:
        # Convert to DataFrame and save
        df = pd.DataFrame(all_events)
        
        # Clean up
        if 'game_id' in df.columns:
            df = df.dropna(subset=['game_id'])
            
            output_file = "data/nba_extracted_dataset.csv"
            df.to_csv(output_file, index=False)
            
            print(f"✅ Saved extracted data to {output_file}")
            print(f"📈 Final dataset: {len(df)} events from {df['game_id'].nunique()} games")
            
            return df
    
    return None

if __name__ == "__main__":
    print("🔍 ANALYZING COLLECTED NBA DATA")
    print("=" * 50)
    
    # First, analyze what we have
    can_read = extract_data_smart()
    
    if can_read:
        print("\n✅ Data appears readable, attempting extraction...")
        df = try_manual_extraction()
        
        if df is not None:
            print(f"\n🎉 SUCCESS! Extracted {len(df)} events")
        else:
            print(f"\n❌ Extraction failed")
    else:
        print("\n❌ Data files appear corrupted or incompatible")