import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ----------------------------
# ÖRNEK VERİ
# ----------------------------
data = pd.DataFrame({
    "home_team": ["Galatasaray", "Fenerbahce", "Besiktas", "Galatasaray", "Trabzonspor"],
    "away_team": ["Fenerbahce", "Besiktas", "Galatasaray", "Trabzonspor", "Fenerbahce"],
    "home_goals": [2, 3, 1, 2, 1],
    "away_goals": [1, 2, 1, 0, 2]
})

# ----------------------------
# ELO RATING (elle başlatıyoruz)
# ----------------------------
elo = {
    "Galatasaray": 1600,
    "Fenerbahce": 1580,
    "Besiktas": 1550,
    "Trabzonspor": 1500
}

# ----------------------------
# MODEL
# ----------------------------
def calculate_team_strengths(df):
    df = df.copy()

    # Weighted form (son maçlar daha önemli)
    df["weight"] = np.linspace(0.5, 1.5, len(df))

    teams = pd.concat([df['home_team'], df['away_team']]).unique()

    attack = {}
    defense = {}

    for team in teams:
        home_games = df[df['home_team'] == team]
        away_games = df[df['away_team'] == team]

        attack[team] = (
            (home_games['home_goals'] * home_games['weight']).mean() +
            (away_games['away_goals'] * away_games['weight']).mean()
        ) / 2

        defense[team] = (
            (home_games['away_goals'] * home_games['weight']).mean() +
            (away_games['home_goals'] * away_games['weight']).mean()
        ) / 2

    return attack, defense


def predict_score(home_team, away_team, attack, defense, max_goals=5):

    home_lambda = attack[home_team] * defense[away_team]
    away_lambda = attack[away_team] * defense[home_team]

    # Ev sahibi avantajı
    home_lambda *= 1.2

    # Elo farkı etkisi
    elo_diff = (elo[home_team] - elo[away_team]) / 400
    home_lambda *= (1 + elo_diff)
    away_lambda *= (1 - elo_diff)

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
st.title("⚽ Gelişmiş Maç Tahmin AI")

attack, defense = calculate_team_strengths(data)

teams = sorted(list(set(data['home_team'])))

home_team = st.selectbox("Ev Sahibi", teams)
away_team = st.selectbox("Deplasman", teams)

if st.button("Tahmin Et"):
    predictions = predict_score(home_team, away_team, attack, defense)

    st.subheader("📊 En Olası Skorlar")

    for home, away, prob in predictions[:5]:
        st.write(f"{home}-{away} → %{round(prob*100,2)}")
