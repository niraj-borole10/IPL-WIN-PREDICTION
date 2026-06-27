import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Home Ground Venues Mapping
HOME_VENUES = {
    'Mumbai Indians': ['Wankhede', 'Brabourne', 'Dr DY Patil'],
    'Chennai Super Kings': ['MA Chidambaram', 'Chepauk'],
    'Kolkata Knight Riders': ['Eden Gardens'],
    'Royal Challengers Bangalore': ['M Chinnaswamy', 'M.Chinnaswamy', 'Bengaluru'],
    'Delhi Capitals': ['Arun Jaitley', 'Feroz Shah Kotla', 'Delhi'],
    'Rajasthan Royals': ['Sawai Mansingh', 'Jaipur'],
    'Punjab Kings': ['Punjab Cricket Association', 'PCA Stadium', 'Mohali', 'Dharamsala'],
    'Sunrisers Hyderabad': ['Rajiv Gandhi International', 'Uppal', 'Hyderabad'],
    'Gujarat Titans': ['Narendra Modi Stadium', 'Motera', 'Ahmedabad'],
    'Lucknow Super Giants': ['Ekana', 'Lucknow']
}

def train_advanced_model():
    processed_dir = os.path.join("data", "processed")
    matches_path = os.path.join(processed_dir, "cleaned_matches.csv")
    deliveries_path = os.path.join(processed_dir, "cleaned_deliveries.csv")
    
    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        print("Data files not found. Run cleaning first.")
        return
        
    df_matches = pd.read_csv(matches_path)
    df_deliv = pd.read_csv(deliveries_path)
    
    # Sort matches chronologically to calculate running stats
    df_matches['date'] = pd.to_datetime(df_matches['date'])
    df_matches = df_matches.sort_values(by='date').reset_index(drop=True)
    
    # Running player stats
    player_runs = {}
    player_dismissals = {}
    player_balls_faced = {}
    player_wickets = {}
    player_runs_conceded = {}
    player_balls_bowled = {}
    player_matches = {}
    
    # Player recent form lists (max 5 records)
    player_recent_runs = {}
    player_recent_wickets = {}
    
    # Head-to-Head win tracking
    h2h_wins = {}
    
    # Roster tracking: team to set of players
    team_rosters = {}
    
    # Pre-calculate wickets indicator
    bowler_wickets_types = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
    df_deliv['is_bowler_wicket'] = df_deliv['wicket_type'].isin(bowler_wickets_types).astype(int)
    
    # Pre-calculate dismissals (exclude retired hurt)
    df_deliv['is_dismissal'] = df_deliv['player_dismissed'].notna() & (df_deliv['wicket_type'] != 'retired hurt')
    
    # Pre-calculate bowler runs conceded
    df_deliv['is_bye_legbye'] = df_deliv['byes'].notna() | df_deliv['legbyes'].notna()
    df_deliv['bowler_runs'] = df_deliv['runs_off_bat'] + df_deliv['extras'].fillna(0).astype(int)
    df_deliv.loc[df_deliv['is_bye_legbye'], 'bowler_runs'] = df_deliv['runs_off_bat']
    
    df_deliv['is_wide'] = df_deliv['wides'].notna() & (df_deliv['wides'] > 0)
    
    print("Computing running player stats chronologically with player form...")
    
    # Group deliveries by match_id for fast lookup
    grouped_deliv = df_deliv.groupby('match_id')
    
    match_features = []
    
    for idx, row in df_matches.iterrows():
        match_id = row['match_id']
        team1 = row['team1']
        team2 = row['team2']
        venue = row['venue']
        winner = row['winner']
        outcome = row['outcome']
        
        # Skip matches without clear winner
        if outcome != 'normal' or pd.isna(winner) or pd.isna(team1) or pd.isna(team2):
            continue
            
        if match_id not in grouped_deliv.groups:
            continue
            
        m_deliv = grouped_deliv.get_group(match_id)
        
        # Identify players in this match
        t1_players = set()
        t2_players = set()
        
        # Batting team 1 / 2
        for t_name, p_set in [(team1, t1_players), (team2, t2_players)]:
            batting_players = set(m_deliv[m_deliv['batting_team'] == t_name]['striker'].dropna().unique()) | \
                              set(m_deliv[m_deliv['batting_team'] == t_name]['non_striker'].dropna().unique())
            bowling_players = set(m_deliv[m_deliv['bowling_team'] == t_name]['bowler'].dropna().unique())
            p_set.update(batting_players)
            p_set.update(bowling_players)
            
        # Update rosters database
        for t_name, p_set in [(team1, t1_players), (team2, t2_players)]:
            if t_name not in team_rosters:
                team_rosters[t_name] = {}
            for p in p_set:
                team_rosters[t_name][p] = team_rosters[t_name].get(p, 0) + 1
                
        # Check Home Ground Advantage
        is_home = 0
        if team1 in HOME_VENUES:
            for v_key in HOME_VENUES[team1]:
                if v_key.lower() in str(venue).lower():
                    is_home = 1
                    break
                    
        # Head-to-Head stats
        t1_vs_t2_wins = h2h_wins.get((team1, team2), 0)
        t2_vs_t1_wins = h2h_wins.get((team2, team1), 0)
        total_h2h = t1_vs_t2_wins + t2_vs_t1_wins
        h2h_win_rate = t1_vs_t2_wins / total_h2h if total_h2h > 0 else 0.5
        
        # Compute pre-match averages (Career & Form)
        def get_team_stats(player_set):
            sum_avg_runs = 0
            sum_avg_wickets = 0
            sum_recent_runs = 0
            sum_recent_wickets = 0
            for p in player_set:
                # Career Averages
                runs = player_runs.get(p, 0)
                dism = player_dismissals.get(p, 0)
                avg_runs = runs / max(1, dism)
                
                wicks = player_wickets.get(p, 0)
                match_count = player_matches.get(p, 0)
                avg_wicks = wicks / max(1, match_count)
                
                sum_avg_runs += avg_runs
                sum_avg_wickets += avg_wicks
                
                # Recent Form (last 5 matches)
                rec_runs = player_recent_runs.get(p, [])
                rec_avg_runs = sum(rec_runs) / max(1, len(rec_runs))
                sum_recent_runs += rec_avg_runs
                
                rec_wicks = player_recent_wickets.get(p, [])
                rec_avg_wicks = sum(rec_wicks) / max(1, len(rec_wicks))
                sum_recent_wickets += rec_avg_wicks
                
            return sum_avg_runs, sum_avg_wickets, sum_recent_runs, sum_recent_wickets
            
        t1_bat_avg, t1_bowl_avg, t1_rec_runs, t1_rec_bowl = get_team_stats(t1_players)
        t2_bat_avg, t2_bowl_avg, t2_rec_runs, t2_rec_bowl = get_team_stats(t2_players)
        
        # Target: 1 if team1 wins, 0 if team2 wins
        target = 1 if winner == team1 else 0
        
        match_features.append({
            'match_id': match_id,
            'team1_bat_avg': t1_bat_avg,
            'team1_bowl_avg': t1_bowl_avg,
            'team2_bat_avg': t2_bat_avg,
            'team2_bowl_avg': t2_bowl_avg,
            'team1_recent_bat_runs': t1_rec_runs,
            'team1_recent_bowl_wickets': t1_rec_bowl,
            'team2_recent_bat_runs': t2_rec_runs,
            'team2_recent_bowl_wickets': t2_rec_bowl,
            'team1_is_home': is_home,
            'h2h_win_rate': h2h_win_rate,
            'venue': venue,
            'target': target
        })
        
        # Update running stats *after* this match completes
        # A. Batting stats
        bat_stats = m_deliv.groupby('striker').agg(
            runs=('runs_off_bat', 'sum'),
            balls_faced=('is_wide', lambda x: (~x).sum()),
            dism=('is_dismissal', 'sum')
        ).reset_index()
        
        for _, r in bat_stats.iterrows():
            p = r['striker']
            player_runs[p] = player_runs.get(p, 0) + r['runs']
            player_dismissals[p] = player_dismissals.get(p, 0) + r['dism']
            player_balls_faced[p] = player_balls_faced.get(p, 0) + r['balls_faced']
            player_matches[p] = player_matches.get(p, 0) + 1
            
            # Update recent form list (last 5)
            if p not in player_recent_runs:
                player_recent_runs[p] = []
            player_recent_runs[p].append(r['runs'])
            if len(player_recent_runs[p]) > 5:
                player_recent_runs[p].pop(0)
            
        # B. Bowling stats
        bowl_stats = m_deliv.groupby('bowler').agg(
            wickets=('is_bowler_wicket', 'sum'),
            runs_conceded=('bowler_runs', 'sum'),
            balls_bowled=('is_wide', lambda x: (~x).sum())
        ).reset_index()
        
        for _, r in bowl_stats.iterrows():
            p = r['bowler']
            player_wickets[p] = player_wickets.get(p, 0) + r['wickets']
            player_runs_conceded[p] = player_runs_conceded.get(p, 0) + r['runs_conceded']
            player_balls_bowled[p] = player_balls_bowled.get(p, 0) + r['balls_bowled']
            if p not in bat_stats['striker'].values:
                player_matches[p] = player_matches.get(p, 0) + 1
                
            # Update recent form list (last 5)
            if p not in player_recent_wickets:
                player_recent_wickets[p] = []
            player_recent_wickets[p].append(r['wickets'])
            if len(player_recent_wickets[p]) > 5:
                player_recent_wickets[p].pop(0)
                
        # Update Head-to-Head win totals
        h2h_wins[(winner, team1 if winner == team2 else team2)] = h2h_wins.get((winner, team1 if winner == team2 else team2), 0) + 1
                
    df_features = pd.DataFrame(match_features)
    print(f"Generated features for {len(df_features)} matches.")
    
    # Train Random Forest Pipeline
    feature_cols = [
        'team1_bat_avg', 'team1_bowl_avg', 'team2_bat_avg', 'team2_bowl_avg',
        'team1_recent_bat_runs', 'team1_recent_bowl_wickets',
        'team2_recent_bat_runs', 'team2_recent_bowl_wickets',
        'team1_is_home', 'h2h_win_rate', 'venue'
    ]
    X = df_features[feature_cols]
    y = df_features['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['venue'])
        ],
        remainder='passthrough'
    )
    
    # Replaced Logistic Regression with RandomForestClassifier
    match_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    match_pipeline.fit(X_train, y_train)
    train_score = accuracy_score(y_train, match_pipeline.predict(X_train))
    test_score = accuracy_score(y_test, match_pipeline.predict(X_test))
    
    print(f"Advanced model train accuracy: {train_score:.3f}")
    print(f"Advanced model test accuracy: {test_score:.3f}")
    
    # Save the model
    joblib.dump(match_pipeline, os.path.join("models", "match_predictor.pkl"))
    
    # Build final player profile lookups
    batsman_stats = []
    for p in player_runs.keys():
        runs = player_runs[p]
        dism = player_dismissals[p]
        balls = player_balls_faced[p]
        avg = runs / max(1, dism)
        sr = (runs / max(1, balls)) * 100
        batsman_stats.append({
            'striker': p,
            'career_runs': runs,
            'career_matches': player_matches.get(p, 0),
            'career_avg_runs': avg,
            'career_sr': sr
        })
        
    bowler_stats = []
    for p in player_wickets.keys():
        wicks = player_wickets[p]
        runs_c = player_runs_conceded[p]
        balls_b = player_balls_bowled[p]
        avg_w = wicks / max(1, player_matches.get(p, 0))
        econ = (runs_c / (max(1, balls_b) / 6))
        bowler_stats.append({
            'bowler': p,
            'career_wickets': wicks,
            'career_matches': player_matches.get(p, 0),
            'career_avg_wickets': avg_w,
            'career_economy': econ
        })
        
    # Standardize rosters to list sorted by match count (top active players)
    final_rosters = {}
    for team, p_dict in team_rosters.items():
        sorted_players = sorted(p_dict.items(), key=lambda x: x[1], reverse=True)
        final_rosters[team] = [p for p, count in sorted_players]
        
    player_predictor = {
        'batsman_stats': pd.DataFrame(batsman_stats).set_index('striker').to_dict(orient='index'),
        'bowler_stats': pd.DataFrame(bowler_stats).set_index('bowler').to_dict(orient='index'),
        'player_recent_runs': player_recent_runs,
        'player_recent_wickets': player_recent_wickets,
        'h2h_wins': h2h_wins,
        'team_rosters': final_rosters,
        'teams': sorted(list(team_rosters.keys())),
        'venues': sorted(list(df_matches['venue'].dropna().unique()))
    }
    
    joblib.dump(player_predictor, os.path.join("models", "player_predictor.pkl"))
    print("Saved advanced player-based models and rosters database.")

if __name__ == "__main__":
    train_advanced_model()
