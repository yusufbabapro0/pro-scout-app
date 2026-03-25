import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# VERİ YÜKLE
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("matches.csv")

df = load_data()

# ----------------------------
# ELO
# ----------------------------
teams = pd.concat([df['home_team'], df['away_team']]).unique()
elo = {team: 1500 for team in teams}

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def create_features(df):
    df = df.copy()

    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["result"] = df["goal_diff"].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))

    df["elo_home"] = df["home_team"].map(elo)
    df["elo_away"] = df["away_team"].map(elo)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    df["total_goals"] = df["home_goals"] + df["away_goals"]

    # bahis oranlarını olasılığa çevir
    df["prob_home"] = 1 / df["home_odds"]
    df["prob_draw"] = 1 / df["draw_odds"]
    df["prob_away"] = 1 / df["away_odds"]

    return df

df_ml = create_features(df)

# ----------------------------
# ML MODEL
# ----------------------------
features = ["elo_diff", "total_goals", "prob_home", "prob_draw", "prob_away"]

X = df_ml[features]
y = df_ml["result"]

model = RandomForestClassifier(n_estimators=300)
model.fit(X, y)

# ----------------------------
# POISSON
# ----------------------------
def calculate_strengths(df):
    df = df.copy()
    df["weight"] = np.linspace(0.5, 1.5, len(df))

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

attack, defense = calculate_strengths(df)

# ----------------------------
# TAHMİN
# ----------------------------
def predict(home_team, away_team, home_odds, draw_odds, away_odds):

    home_lambda = attack[home_team] * defense[away_team] * 1.3
    away_lambda = attack[away_team] * defense[home_team]

    elo_diff = elo[home_team] - elo[away_team]

    prob_home = 1 / home_odds
    prob_draw = 1 / draw_odds
    prob_away = 1 / away_odds

    ml_input = pd.DataFrame([[elo_diff, home_lambda + away_lambda, prob_home, prob_draw, prob_away]],
                            columns=features)

    result = model.predict(ml_input)[0]

    results = []

    for i in range(6):
        for j in range(6):
            prob = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)

            if result == 1 and i > j:
                prob *= 1.4
            elif result == -1 and i < j:
                prob *= 1.4
            elif result == 0 and i == j:
                prob *= 1.4

            results.append((i, j, prob))

    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results[:5], result

# ----------------------------
# UI
# ----------------------------
st.title("🚀 Ultimate Maç Tahmin AI")

home_team = st.selectbox("Ev Sahibi", teams)
away_team = st.selectbox("Deplasman", teams)

st.subheader("💰 Bahis Oranları Gir")

home_odds = st.number_input("Ev Sahibi Oranı", value=2.0)
draw_odds = st.number_input("Beraberlik Oranı", value=3.2)
away_odds = st.number_input("Deplasman Oranı", value=3.5)

if st.button("Tahmin Et"):

    scores, result = predict(home_team, away_team, home_odds, draw_odds, away_odds)

    st.subheader("📊 Skor Tahminleri")
    for h, a, p in scores:
        st.write(f"{h}-{a} → %{round(p*100,2)}")

    st.subheader("🎯 Maç Sonucu")

    if result == 1:
        st.write("Ev Sahibi Kazanır")
    elif result == -1:
        st.write("Deplasman Kazanır")
    else:
        st.write("Beraberlik")
