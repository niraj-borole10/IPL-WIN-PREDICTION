import os
import glob
import pandas as pd
import numpy as np

def clean_all_data():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Parse all *_info.csv files
    info_files = glob.glob(os.path.join(raw_dir, "*_info.csv"))
    print(f"Found {len(info_files)} match info files to parse.")
    
    TEAM_MAPPING = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Rising Pune Supergiants': 'Rising Pune Supergiant',
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
    }
    
    matches_list = []
    
    for filepath in info_files:
        match_id = os.path.basename(filepath).replace("_info.csv", "")
        
        info_dict = {
            'match_id': int(match_id),
            'season': None,
            'date': None,
            'team1': None,
            'team2': None,
            'venue': None,
            'city': None,
            'toss_winner': None,
            'toss_decision': None,
            'winner': None,
            'win_by_runs': 0,
            'win_by_wickets': 0,
            'outcome': 'normal',
            'player_of_match': None
        }
        
        teams = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[0] == 'info':
                    key = parts[1]
                    value = ','.join(parts[2:]).strip('"')
                    
                    if key == 'team':
                        teams.append(value)
                    elif key == 'season':
                        info_dict['season'] = value
                    elif key == 'date':
                        info_dict['date'] = value
                    elif key == 'venue':
                        info_dict['venue'] = value
                    elif key == 'city':
                        info_dict['city'] = value
                    elif key == 'toss_winner':
                        info_dict['toss_winner'] = value
                    elif key == 'toss_decision':
                        info_dict['toss_decision'] = value
                    elif key == 'winner':
                        info_dict['winner'] = value
                    elif key == 'winner_runs':
                        info_dict['win_by_runs'] = int(value)
                    elif key == 'winner_wickets':
                        info_dict['win_by_wickets'] = int(value)
                    elif key == 'outcome':
                        info_dict['outcome'] = value
                    elif key == 'player_of_match':
                        info_dict['player_of_match'] = value
                        
        if len(teams) >= 2:
            info_dict['team1'] = teams[0]
            info_dict['team2'] = teams[1]
        
        # Standardize team names
        for col in ['team1', 'team2', 'toss_winner', 'winner']:
            val = info_dict[col]
            if val in TEAM_MAPPING:
                info_dict[col] = TEAM_MAPPING[val]
                
        matches_list.append(info_dict)
        
    df_matches = pd.DataFrame(matches_list)
    # Sort matches by date
    df_matches['date'] = pd.to_datetime(df_matches['date'])
    df_matches = df_matches.sort_values(by='date').reset_index(drop=True)
    
    matches_csv_path = os.path.join(processed_dir, "cleaned_matches.csv")
    df_matches.to_csv(matches_csv_path, index=False)
    print(f"Saved {len(df_matches)} cleaned matches to {matches_csv_path}")
    
    # 2. Clean ball-by-ball deliveries
    all_deliveries_path = os.path.join(raw_dir, "all_matches.csv")
    print(f"Reading ball-by-ball data from {all_deliveries_path}...")
    df_deliv = pd.read_csv(all_deliveries_path)
    
    # Apply team mapping
    df_deliv['batting_team'] = df_deliv['batting_team'].map(TEAM_MAPPING).fillna(df_deliv['batting_team'])
    df_deliv['bowling_team'] = df_deliv['bowling_team'].map(TEAM_MAPPING).fillna(df_deliv['bowling_team'])
    
    # Let's save processed deliveries
    deliv_csv_path = os.path.join(processed_dir, "cleaned_deliveries.csv")
    df_deliv.to_csv(deliv_csv_path, index=False)
    print(f"Saved cleaned deliveries to {deliv_csv_path}")

if __name__ == "__main__":
    clean_all_data()
