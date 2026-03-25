import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ----------------------------
# ÖRNEK VERİ (gömülü)
# ----------------------------
data = pd.DataFrame({
    "home_team": ["Galatasaray", "Fenerbahce", "Besiktas", "Galatasaray", "Trabzonspor"],
    "away_team": ["Fenerbahce", "Besiktas", "Galatasaray", "Trabzonspor", "Fenerbahce"],
    "home_goals": [2, 3, 1, 2, 1],
    "away_goals": [1, 2, 1, 0, 2]
})

# ----------------------------
# MODEL
# ----------------------------
def calculate_team_strengths(df):
    teams = pd.concat([df['home_team'], df['away_team']]).unique()

    attack = {}
    defense = {}

    for team in teams:
        home_games = df[df['home_team'] == team]
        away_games = df[df['away_team'] == team]

        attack[team] = (
            home_games['home_goals'].mean() + away_games['away_goals'].mean()
        ) / 2

        defense[team] = (
            home_games['away_goals'].mean() + away_games['home_goals'].mean()
        ) / 2

    return attack, defense


def predict_score(home_team, away_team, attack, defense, max_goals=5):
    home_lambda = attack[home_team] * defense[away_team]
    away_lambda = attack[away_team] * defense[home_team]

    results = []

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
            results.append((i, j, prob))

    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("⚽ Maç Skor Tahmin AI")

attack, defense = calculate_team_strengths(data)

teams = sorted(list(set(data['home_team'])))

home_team = st.selectbox("Ev Sahibi", teams)
away_team = st.selectbox("Deplasman", teams)

if st.button("Tahmin Et"):
    predictions = predict_score(home_team, away_team, attack, defense)

    st.subheader("📊 En Olası Skorlar")

    for home, away, prob in predictions[:5]:
        st.write(f"{home}-{away} → %{round(prob*100,2)}")
