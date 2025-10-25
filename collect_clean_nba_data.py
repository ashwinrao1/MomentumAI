#!/usr/bin/env python3
"""
Collect clean NBA data directly without class dependencies.
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from tqdm import tqdm
from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder

def collect_clean_nba_data(max_games=500):
    """Collect clean NBA data directly."""
    print(f"🏀 Collecting clean NBA data for {max_games} games...")
    
    # Get recent games
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable="2023-24",
            season_type_nullable="Regular Season"
        )
        games_df = gamefinder.get_data_frames()[0]
        
        # Get unique games (remove duplicates from home/away)
        unique_games = games_df['GAME_ID'].unique()[:max_games]
        
        print(f"Found {len(unique_games)} games to process")
        
        all_events = []
        
        for i, game_id in enumerate(tqdm(unique_games, desc="Processing games")):
            try:
                # Get play-by-play data
                pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
                pbp_data = pbp.get_data_frames()[0]
                
                # Process each event
                for _, row in pbp_data.iterrows():
                    # Extract basic event info
                    event = {
                        'event_id': f"{game_id}_{row.get('EVENTNUM', 0)}",
                        'game_id': game_id,
                        'period': int(row.get('PERIOD', 1)),
                        'clock': str(row.get('PCTIMESTRING', '12:00')),
                        'event_type': 'other',  # Will classify below
                        'team_tricode': None,
                        'player_name': row.get('PLAYER1_NAME'),
                        'description': '',
                        'shot_result': None,
                        'points_total': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Get description and team
                    home_desc = str(row.get('HOMEDESCRIPTION', '') or '')
                    away_desc = str(row.get('VISITORDESCRIPTION', '') or '')
                    description = home_desc if home_desc != 'nan' and home_desc else away_desc
                    
                    if description == 'nan' or not description:
                        continue
                    
                    event['description'] = description
                    
                    # Get team
                    if row.get('PLAYER1_TEAM_ABBREVIATION'):
                        event['team_tricode'] = row.get('PLAYER1_TEAM_ABBREVIATION')
                    
                    if not event['team_tricode']:
                        continue
                    
                    # Classify event type and extract value
                    desc_lower = description.lower()
                    
                    if any(word in desc_lower for word in ['makes', 'made']):
                        event['event_type'] = 'shot'
                        event['shot_result'] = 'Made'
                        if '3pt' in desc_lower:
                            event['points_total'] = 3
                        elif 'free throw' in desc_lower:
                            event['points_total'] = 1
                        else:
                            event['points_total'] = 2
                    
                    elif any(word in desc_lower for word in ['misses', 'missed']):
                        event['event_type'] = 'shot'
                        event['shot_result'] = 'Missed'
                        event['points_total'] = 0
                    
                    elif 'rebound' in desc_lower:
                        event['event_type'] = 'rebound'
                    
                    elif 'assist' in desc_lower:
                        event['event_type'] = 'assist'
                    
                    elif 'steal' in desc_lower:
                        event['event_type'] = 'steal'
                    
                    elif 'block' in desc_lower:
                        event['event_type'] = 'block'
                    
                    elif 'turnover' in desc_lower:
                        event['event_type'] = 'turnover'
                    
                    elif 'foul' in desc_lower:
                        event['event_type'] = 'foul'
                    
                    all_events.append(event)
                
                # Rate limiting
                time.sleep(0.6)
                
                # Progress update
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1} games, {len(all_events)} events collected")
                
            except Exception as e:
                print(f"  Error processing game {game_id}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        
        if len(df) > 0:
            # Save to CSV
            output_file = "data/nba_clean_dataset.csv"
            df.to_csv(output_file, index=False)
            
            print(f"\n✅ Collection complete!")
            print(f"📊 Total events: {len(df)}")
            print(f"🏀 Total games: {df['game_id'].nunique()}")
            print(f"🏟️  Teams: {df['team_tricode'].nunique()}")
            print(f"💾 Saved to: {output_file}")
            
            # Save summary
            summary = {
                'total_events': len(df),
                'total_games': df['game_id'].nunique(),
                'total_teams': df['team_tricode'].nunique(),
                'event_type_distribution': df['event_type'].value_counts().to_dict(),
                'teams': sorted(df['team_tricode'].unique().tolist()),
                'collection_date': datetime.now().isoformat()
            }
            
            with open('data/nba_clean_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return df
        else:
            print("❌ No events collected")
            return None
            
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        return None

if __name__ == "__main__":
    collect_clean_nba_data(max_games=500)