import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="IPL Prediction & Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Home Ground Venues Mapping (matches train_models.py)
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

# Custom CSS for premium dark aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        background-attachment: fixed;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .main-title {
        font-size: 3.2rem;
        color: #ff4b4b;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .stExpander {
        background: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }
    
    .streamlit-expanderHeader {
        background-color: transparent !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    div[data-baseweb="select"] {
        background-color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #f8fafc !important;
    }
    
    div.stButton > button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: none !important;
        width: 100% !important;
    }
    
    div.stButton > button:hover {
        background-color: #e13f3f !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_models():
    match_model_path = os.path.join("models", "match_predictor.pkl")
    player_model_path = os.path.join("models", "player_predictor.pkl")
    
    match_model = None
    player_data = None
    
    if os.path.exists(match_model_path):
        match_model = joblib.load(match_model_path)
    if os.path.exists(player_model_path):
        player_data = joblib.load(player_model_path)
        
    return match_model, player_data

# Cache matches loading
@st.cache_data
def load_matches_data():
    matches_path = os.path.join("data", "processed", "cleaned_matches.csv")
    if os.path.exists(matches_path):
        return pd.read_csv(matches_path)
    return None

match_model, player_data = load_models()
df_matches = load_matches_data()

st.markdown('<div class="main-title">IPL Prediction & Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;">Real-time match win-loss forecasting and player career profile metrics</div>', unsafe_allow_html=True)

if match_model is None or player_data is None or df_matches is None:
    st.error("Model files or cleaned data not found. Please run data preparation and training first.")
    st.info("Ensure you have executed the data cleaning and model training scripts in the project directory.")
else:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose Dashboard Section:",
        ["Match Winner Predictor", "Player Stats Lookup", "Historical Analytics"]
    )
    
    # ------------------
    # Section 1: Match Winner Predictor
    # ------------------
    if app_mode == "Match Winner Predictor":
        st.header("Match Winner Predictor")
        st.write("Predict match outcomes based on the playing XI of both teams and the venue.")
        
        teams = player_data['teams']
        venues = player_data['venues']
        team_rosters = player_data.get('team_rosters', {})
        batsman_stats = player_data['batsman_stats']
        bowler_stats = player_data['bowler_stats']
        player_recent_runs = player_data.get('player_recent_runs', {})
        player_recent_wickets = player_data.get('player_recent_wickets', {})
        h2h_wins = player_data.get('h2h_wins', {})
        
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1 (Home Team)", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)
            venue = st.selectbox("Select Venue", venues)
            
        with col2:
            team2 = st.selectbox("Select Team 2 (Away Team)", [t for t in teams if t != team1], index=0)
            
        # Get rosters for Team 1 and Team 2
        roster1 = team_rosters.get(team1, [])
        roster2 = team_rosters.get(team2, [])
        
        # Default to top 11 players
        default_xi_1 = roster1[:11] if len(roster1) >= 11 else roster1
        default_xi_2 = roster2[:11] if len(roster2) >= 11 else roster2
        
        st.subheader("Selected Playing XI")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            with st.expander(f"{team1} Lineup Selection", expanded=True):
                selected_xi_1 = st.multiselect(
                    f"Select Playing XI for {team1}",
                    options=roster1,
                    default=default_xi_1
                )
                if len(selected_xi_1) != 11:
                    st.caption(f"Selected {len(selected_xi_1)} players. Standard lineup is 11.")
                    
        with col_t2:
            with st.expander(f"{team2} Lineup Selection", expanded=True):
                selected_xi_2 = st.multiselect(
                    f"Select Playing XI for {team2}",
                    options=roster2,
                    default=default_xi_2
                )
                if len(selected_xi_2) != 11:
                    st.caption(f"Selected {len(selected_xi_2)} players. Standard lineup is 11.")
                    
        if st.button("Predict Winning Probabilities", type="primary"):
            if len(selected_xi_1) == 0 or len(selected_xi_2) == 0:
                st.error("Please select at least 1 player for both teams.")
            else:
                # Aggregate stats for selected XI
                def aggregate_team_stats(selected_players):
                    sum_avg_runs = 0
                    sum_avg_wickets = 0
                    sum_recent_runs = 0
                    sum_recent_wickets = 0
                    for p in selected_players:
                        # Career
                        bat_p = batsman_stats.get(p, {})
                        avg_runs = bat_p.get('career_avg_runs', 0)
                        sum_avg_runs += avg_runs
                        
                        bowl_p = bowler_stats.get(p, {})
                        avg_wicks = bowl_p.get('career_avg_wickets', 0)
                        sum_avg_wickets += avg_wicks
                        
                        # Recent Form (rolling 5)
                        rec_runs = player_recent_runs.get(p, [])
                        rec_avg_runs = sum(rec_runs) / max(1, len(rec_runs))
                        sum_recent_runs += rec_avg_runs
                        
                        rec_wicks = player_recent_wickets.get(p, [])
                        rec_avg_wicks = sum(rec_wicks) / max(1, len(rec_wicks))
                        sum_recent_wickets += rec_avg_wicks
                        
                    return sum_avg_runs, sum_avg_wickets, sum_recent_runs, sum_recent_wickets
                    
                t1_bat_avg, t1_bowl_avg, t1_rec_runs, t1_rec_bowl = aggregate_team_stats(selected_xi_1)
                t2_bat_avg, t2_bowl_avg, t2_rec_runs, t2_rec_bowl = aggregate_team_stats(selected_xi_2)
                
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
                
                input_df = pd.DataFrame([{
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
                    'venue': venue
                }])
                
                try:
                    probs = match_model.predict_proba(input_df)[0]
                    prob_team1 = probs[1] * 100
                    prob_team2 = probs[0] * 100
                    
                    st.success("Prediction complete!")
                    
                    st.subheader("Win Probability Breakdown")
                    prob_df = pd.DataFrame({
                        "Team": [team1, team2],
                        "Probability (%)": [prob_team1, prob_team2]
                    })
                    fig = px.bar(
                        prob_df, 
                        x="Probability (%)", 
                        y="Team", 
                        orientation='h', 
                        color="Team",
                        text="Probability (%)",
                        color_discrete_sequence=["#1f77b4", "#ff7f0e"]
                    )
                    fig.update_layout(xaxis=dict(range=[0, 100]), showlegend=False)
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, width="stretch")
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric(label=f"{team1} Win Chance", value=f"{prob_team1:.1f}%")
                    col_b.metric(label=f"{team2} Win Chance", value=f"{prob_team2:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                
    # ------------------
    # Section 2: Player Stats Lookup
    # ------------------
    elif app_mode == "Player Stats Lookup":
        st.header("Player Performance Profiles")
        st.write("Look up historical career statistics for batsmen and bowlers.")
        
        batsman_dict = player_data['batsman_stats']
        bowler_dict = player_data['bowler_stats']
        
        player_type = st.radio("Select Player Role:", ["Batsman", "Bowler"])
        
        if player_type == "Batsman":
            all_batsmen = sorted(list(batsman_dict.keys()))
            player_name = st.selectbox("Select Batsman", all_batsmen)
            
            if player_name in batsman_dict:
                stats = batsman_dict[player_name]
                st.subheader(f"Batting Profile: {player_name}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Matches Played", int(stats['career_matches']))
                col2.metric("Total Runs", int(stats['career_runs']))
                col3.metric("Batting Average", f"{stats['career_avg_runs']:.2f}")
                col4.metric("Strike Rate", f"{stats['career_sr']:.2f}")
                
                # Show expected runs description
                st.info(f"Based on historical IPL career performance (where retired hurt counts as not out), **{player_name}** scores an average of **{stats['career_avg_runs']:.1f} runs** per innings at a strike rate of **{stats['career_sr']:.1f}**.")
                
        else:
            all_bowlers = sorted(list(bowler_dict.keys()))
            player_name = st.selectbox("Select Bowler", all_bowlers)
            
            if player_name in bowler_dict:
                stats = bowler_dict[player_name]
                st.subheader(f"Bowling Profile: {player_name}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Matches Played", int(stats['career_matches']))
                col2.metric("Total Wickets", int(stats['career_wickets']))
                col3.metric("Wickets Per Match", f"{stats['career_avg_wickets']:.2f}")
                col4.metric("Economy Rate", f"{stats['career_economy']:.2f}")
                
                # Show expected wickets description
                st.info(f"Based on historical IPL career performance, **{player_name}** takes an average of **{stats['career_avg_wickets']:.2f} wickets** per match with an economy rate of **{stats['career_economy']:.2f}**.")

    # ------------------
    # Section 3: Historical Analytics
    # ------------------
    elif app_mode == "Historical Analytics":
        st.header("Historical IPL Match Trends")
        st.write("Browse historical dataset statistics and match distribution charts.")
        
        # Win breakdown
        st.subheader("Wins by Team")
        wins_df = df_matches['winner'].value_counts().reset_index()
        wins_df.columns = ['Team', 'Wins']
        fig_wins = px.bar(wins_df, x='Wins', y='Team', orientation='h', title='Total Match Wins in IPL (All Seasons)', color='Team')
        st.plotly_chart(fig_wins, width="stretch")
        
        # Toss choice breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Toss Decisions")
            toss_df = df_matches['toss_decision'].value_counts().reset_index()
            toss_df.columns = ['Decision', 'Count']
            fig_toss = px.pie(toss_df, values='Count', names='Decision', title='Toss Decision Choices', hole=0.4)
            st.plotly_chart(fig_toss, width="stretch")
            
        with col2:
            st.subheader("Match Outcome Types")
            outcome_df = df_matches['outcome'].value_counts().reset_index()
            outcome_df.columns = ['Outcome', 'Count']
            fig_out = px.pie(outcome_df, values='Count', names='Outcome', title='Match Outcomes')
            st.plotly_chart(fig_out, width="stretch")
            
        # Matches Data Table
        st.subheader("Browse Historical Matches")
        st.dataframe(df_matches[['match_id', 'season', 'date', 'team1', 'team2', 'venue', 'winner', 'win_by_runs', 'win_by_wickets', 'player_of_match']])
