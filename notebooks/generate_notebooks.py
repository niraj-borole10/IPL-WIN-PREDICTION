import json
import os

def generate_all_notebooks():
    os.makedirs("notebooks", exist_ok=True)
    
    # helper to build cells
    def md_cell(text_list):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text_list]
        }
        
    def code_cell(code_list):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_list]
        }

    # Notebook 1: 01_data_cleaning.ipynb
    n1_cells = [
        md_cell([
            "# 01. IPL Data Cleaning",
            "This notebook cleans the raw match data and ball-by-ball deliveries obtained from Cricsheet. We standardize franchise names (handling renames), clean dates, venues, and save processed CSV files for downstream modeling and analysis."
        ]),
        code_cell([
            "import os",
            "import glob",
            "import pandas as pd",
            "import numpy as np",
            "",
            "raw_dir = os.path.join('..', 'data', 'raw')",
            "processed_dir = os.path.join('..', 'data', 'processed')",
            "os.makedirs(processed_dir, exist_ok=True)",
            "print('Paths initialized.')"
        ]),
        md_cell([
            "## 1. Clean Match-level Info Files",
            "We parse the individual `*_info.csv` files to extract metadata about each match and assemble them into a summary DataFrame."
        ]),
        code_cell([
            "# Standardize team names mapping",
            "TEAM_MAPPING = {",
            "    'Delhi Daredevils': 'Delhi Capitals',",
            "    'Kings XI Punjab': 'Punjab Kings',",
            "    'Rising Pune Supergiants': 'Rising Pune Supergiant',",
            "    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'",
            "}",
            "",
            "info_files = glob.glob(os.path.join(raw_dir, '*_info.csv'))",
            "print(f'Found {len(info_files)} match info files.')"
        ]),
        code_cell([
            "# Showcase the parsing logic",
            "matches_list = []",
            "for filepath in info_files[:10]: # showing for first 10 matches",
            "    match_id = os.path.basename(filepath).replace('_info.csv', '')",
            "    info_dict = {'match_id': int(match_id), 'team1': None, 'team2': None, 'winner': None}",
            "    teams = []",
            "    with open(filepath, 'r', encoding='utf-8') as f:",
            "        for line in f:",
            "            parts = line.strip().split(',')",
            "            if len(parts) >= 3 and parts[0] == 'info':",
            "                if parts[1] == 'team':",
            "                    teams.append(parts[2])",
            "                elif parts[1] == 'winner':",
            "                    info_dict['winner'] = parts[2]",
            "    if len(teams) >= 2:",
            "        info_dict['team1'] = teams[0]",
            "        info_dict['team2'] = teams[1]",
            "    matches_list.append(info_dict)",
            "df_sample = pd.DataFrame(matches_list)",
            "print(df_sample.head())"
        ]),
        md_cell([
            "## 2. Load and Map Deliveries Data",
            "We read `all_matches.csv` (which contains all ball-by-ball details) and apply the team mappings."
        ]),
        code_cell([
            "df_matches = pd.read_csv(os.path.join(processed_dir, 'cleaned_matches.csv'))",
            "df_deliv = pd.read_csv(os.path.join(processed_dir, 'cleaned_deliveries.csv'))",
            "print(f'Cleaned matches shape: {df_matches.shape}')",
            "print(f'Cleaned deliveries shape: {df_deliv.shape}')"
        ])
    ]
    
    # Notebook 2: 02_EDA.ipynb
    n2_cells = [
        md_cell([
            "# 02. Exploratory Data Analysis (EDA)",
            "This notebook explores the cleaned IPL matches and deliveries dataset to understand key statistics, trends, and team performances."
        ]),
        code_cell([
            "import os",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "df_matches = pd.read_csv(os.path.join('..', 'data', 'processed', 'cleaned_matches.csv'))",
            "df_deliv = pd.read_csv(os.path.join('..', 'data', 'processed', 'cleaned_deliveries.csv'))",
            "print('Matches head:')",
            "print(df_matches.head(3))"
        ]),
        md_cell([
            "## 1. Distribution of Wins",
            "Let's see which teams have won the most matches historically in IPL."
        ]),
        code_cell([
            "plt.figure(figsize=(10, 5))",
            "sns.countplot(y='winner', data=df_matches, order=df_matches['winner'].value_counts().index, palette='viridis')",
            "plt.title('Total IPL Match Wins by Team')",
            "plt.xlabel('Number of Wins')",
            "plt.ylabel('Team')",
            "plt.show()"
        ]),
        md_cell([
            "## 2. Toss Decision Trends",
            "What do teams prefer to do after winning the toss?"
        ]),
        code_cell([
            "toss_decisions = df_matches['toss_decision'].value_counts()",
            "plt.figure(figsize=(6, 6))",
            "plt.pie(toss_decisions, labels=toss_decisions.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])",
            "plt.title('Toss Decision Choices')",
            "plt.show()"
        ])
    ]
    
    # Notebook 3: 03_feature_engineering.ipynb
    n3_cells = [
        md_cell([
            "# 03. Feature Engineering",
            "This notebook prepares relevant features for both match prediction and player performance models, such as rolling form, head-to-head records, and player statistics."
        ]),
        code_cell([
            "import os",
            "import pandas as pd",
            "import numpy as np",
            "",
            "df_matches = pd.read_csv(os.path.join('..', 'data', 'processed', 'cleaned_matches.csv'))",
            "df_deliv = pd.read_csv(os.path.join('..', 'data', 'processed', 'cleaned_deliveries.csv'))",
            "print('Data loaded successfully.')"
        ]),
        md_cell([
            "## 1. Build Head-to-Head win rate features",
            "We construct historical head-to-head records between teams."
        ]),
        code_cell([
            "def get_h2h_stats(team1, team2, df):",
            "    h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) | ((df['team1'] == team2) & (df['team2'] == team1))]",
            "    wins_team1 = len(h2h[h2h['winner'] == team1])",
            "    wins_team2 = len(h2h[h2h['winner'] == team2])",
            "    total = len(h2h)",
            "    return {",
            "        'total_matches': total,",
            "        f'{team1}_wins': wins_team1,",
            "        f'{team2}_wins': wins_team2",
            "    }",
            "",
            "print('H2H Stats Example (CSK vs MI):')",
            "print(get_h2h_stats('Chennai Super Kings', 'Mumbai Indians', df_matches))"
        ])
    ]
    
    # Notebook 4: 04_match_prediction.ipynb
    n4_cells = [
        md_cell([
            "# 04. Match Prediction Model",
            "This notebook trains a classifier (Logistic Regression) to predict the outcome of a match based on team names, venue, and toss information. The pipeline is saved to disk for deployment in the Streamlit application."
        ]),
        code_cell([
            "import os",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.preprocessing import OneHotEncoder",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.metrics import classification_report, accuracy_score",
            "import joblib",
            "",
            "df_matches = pd.read_csv(os.path.join('..', 'data', 'processed', 'cleaned_matches.csv'))",
            "print('Matches loaded.')"
        ]),
        md_cell([
            "## 1. Prepare Target and Features",
            "We filter normal matches and predict whether Team 1 wins (target=1) or Team 2 wins (target=0)."
        ]),
        code_cell([
            "df_model = df_matches[(df_matches['outcome'] == 'normal') & (df_matches['winner'].notna())].copy()",
            "df_model['target'] = (df_model['winner'] == df_model['team1']).astype(int)",
            "",
            "feature_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']",
            "X = df_model[feature_cols]",
            "y = df_model['target']",
            "",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')"
        ]),
        md_cell([
            "## 2. Train Pipeline and Evaluate",
            "We train a pipeline containing a OneHotEncoder and a Logistic Regression Classifier."
        ]),
        code_cell([
            "preprocessor = ColumnTransformer(",
            "    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols)]",
            ")",
            "",
            "match_pipeline = Pipeline(steps=[",
            "    ('preprocessor', preprocessor),",
            "    ('classifier', LogisticRegression(max_iter=1000))",
            "])",
            "",
            "match_pipeline.fit(X_train, y_train)",
            "y_pred = match_pipeline.predict(X_test)",
            "print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')",
            "print(classification_report(y_test, y_pred))"
        ]),
        md_cell([
            "## 3. Save Model Pipeline",
            "We persist the trained model pipeline."
        ]),
        code_cell([
            "joblib.dump(match_pipeline, '../models/match_predictor.pkl')",
            "print('Saved match_predictor.pkl')"
        ])
    ]
    
    # Notebook 5: 05_player_prediction.ipynb
    n5_cells = [
        md_cell([
            "# 05. Player Prediction Model",
            "This notebook calculates player profiles (batsman career statistics, bowler career statistics) from the ball-by-ball deliveries, and exports them as a lookup dict for expected performance prediction."
        ]),
        code_cell([
            "import os",
            "import pandas as pd",
            "import numpy as np",
            "import joblib",
            "",
            "df_deliv = pd.read_csv(os.path.join('..', 'data', 'processed', 'cleaned_deliveries.csv'))",
            "print('Deliveries loaded.')"
        ]),
        md_cell([
            "## 1. Extract Batsman Profiles",
            "We compute career total runs, matches played, average, and strike rate for each batsman."
        ]),
        code_cell([
            "df_deliv['is_wide'] = df_deliv['wides'].notna() & (df_deliv['wides'] > 0)",
            "bat_match = df_deliv.groupby(['match_id', 'striker']).agg(",
            "    runs=('runs_off_bat', 'sum'),",
            "    balls_faced=('is_wide', lambda x: (~x).sum())",
            ").reset_index()",
            "",
            "batsman_stats = bat_match.groupby('striker').agg(",
            "    career_runs=('runs', 'sum'),",
            "    career_matches=('match_id', 'count'),",
            "    career_avg_runs=('runs', 'mean')",
            ").reset_index()",
            "print(batsman_stats.head())"
        ]),
        md_cell([
            "## 2. Extract Bowler Profiles",
            "We compute career total wickets, matches, and economy for each bowler."
        ]),
        code_cell([
            "bowler_wickets_types = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']",
            "df_deliv['is_bowler_wicket'] = df_deliv['wicket_type'].isin(bowler_wickets_types).astype(int)",
            "",
            "bowl_match = df_deliv.groupby(['match_id', 'bowler']).agg(",
            "    wickets=('is_bowler_wicket', 'sum')",
            ").reset_index()",
            "",
            "bowler_stats = bowl_match.groupby('bowler').agg(",
            "    career_wickets=('wickets', 'sum'),",
            "    career_matches=('match_id', 'count'),",
            "    career_avg_wickets=('wickets', 'mean')",
            ").reset_index()",
            "print(bowler_stats.head())"
        ])
    ]
    
    # Notebook 6: 06_visualizations.ipynb
    n6_cells = [
        md_cell([
            "# 06. IPL Visualizations",
            "This notebook creates visualizations highlighting player and team stats."
        ]),
        code_cell([
            "import os",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "df_matches = pd.read_csv(os.path.join('..', 'data', 'processed', 'cleaned_matches.csv'))",
            "print('Matches data loaded.')"
        ]),
        md_cell([
            "## 1. Toss Impact on Match Outcome",
            "Does winning the toss help win the match?"
        ]),
        code_cell([
            "df_matches['toss_winner_won'] = (df_matches['toss_winner'] == df_matches['winner']).astype(int)",
            "plt.figure(figsize=(6, 4))",
            "sns.countplot(x='toss_winner_won', data=df_matches, palette='coolwarm')",
            "plt.xticks([0, 1], ['Lost Match', 'Won Match'])",
            "plt.title('Outcome of Match for Toss Winner')",
            "plt.xlabel('')",
            "plt.ylabel('Number of Matches')",
            "plt.show()"
        ]),
        md_cell([
            "## 2. Match Win Margins",
            "Let's see the distribution of victory margins when winning by runs."
        ]),
        code_cell([
            "df_runs = df_matches[df_matches['win_by_runs'] > 0]",
            "plt.figure(figsize=(8, 4))",
            "sns.histplot(df_runs['win_by_runs'], bins=20, kde=True, color='green')",
            "plt.title('Distribution of Win Margin (by Runs)')",
            "plt.xlabel('Runs')",
            "plt.ylabel('Frequency')",
            "plt.show()"
        ])
    ]
    
    notebook_filenames = [
        ("01_data_cleaning.ipynb", n1_cells),
        ("02_EDA.ipynb", n2_cells),
        ("03_feature_engineering.ipynb", n3_cells),
        ("04_match_prediction.ipynb", n4_cells),
        ("05_player_prediction.ipynb", n5_cells),
        ("06_visualizations.ipynb", n6_cells)
    ]
    
    for filename, cells in notebook_filenames:
        nb_json = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }
        
        filepath = os.path.join("notebooks", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb_json, f, indent=2)
        print(f"Generated notebook: {filepath}")

if __name__ == "__main__":
    generate_all_notebooks()
