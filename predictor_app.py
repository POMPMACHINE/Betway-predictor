import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Betway-Style AI Match Predictor")

# Team name inputs
home_team = st.text_input("Enter Home Team Name", "Team A")
away_team = st.text_input("Enter Away Team Name", "Team B")

# Stat inputs
home_team_attack = st.slider(f"{home_team} Attack Rating", 0.0, 2.5, 1.2)
away_team_attack = st.slider(f"{away_team} Attack Rating", 0.0, 2.5, 1.0)
home_team_defense = st.slider(f"{home_team} Defense Rating", 0.0, 2.5, 0.8)
away_team_defense = st.slider(f"{away_team} Defense Rating", 0.0, 2.5, 1.0)
home_recent_form = st.slider(f"{home_team} Recent Form (0-5 wins)", 0, 5, 3)
away_recent_form = st.slider(f"{away_team} Recent Form (0-5 wins)", 0, 5, 2)
home_star_players_missing = st.slider(f"{home_team} Star Players Missing", 0, 5, 1)
away_star_players_missing = st.slider(f"{away_team} Star Players Missing", 0, 5, 2)

# Package into dataframe for prediction
data = pd.DataFrame({
    'home_team_attack': [home_team_attack],
    'away_team_attack': [away_team_attack],
    'home_team_defense': [home_team_defense],
    'away_team_defense': [away_team_defense],
    'home_recent_form': [home_recent_form],
    'away_recent_form': [away_recent_form],
    'home_star_players_missing': [home_star_players_missing],
    'away_star_players_missing': [away_star_players_missing],
})

# Sample training data
X_train = pd.DataFrame({
    'home_team_attack': [1.4, 0.8, 1.2, 0.7],
    'away_team_attack': [1.0, 1.1, 0.9, 1.3],
    'home_team_defense': [0.7, 1.1, 0.8, 1.2],
    'away_team_defense': [1.0, 0.9, 1.0, 0.7],
    'home_recent_form': [4, 1, 3, 2],
    'away_recent_form': [2, 3, 1, 4],
    'home_star_players_missing': [0, 1, 0, 2],
    'away_star_players_missing': [1, 0, 2, 1],
})
y_train_winner = [1, 0, 1, 0]
y_train_ou25 = [1, 0, 1, 1]

# Train models
model_winner = LogisticRegression().fit(X_train, y_train_winner)
model_ou25 = LogisticRegression().fit(X_train, y_train_ou25)

# Predict on button click
if st.button("Predict"):
    pred_winner = model_winner.predict(data)[0]
    pred_ou25 = model_ou25.predict(data)[0]

    st.subheader("Prediction Results")
    st.write("Match Winner:", f"**{home_team}**" if pred_winner == 1 else f"**{away_team} or Draw**")
    st.write("Total Goals:", "> 2.5 goals" if pred_ou25 == 1 else "<= 2.5 goals")