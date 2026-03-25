import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# API AYARI
# ----------------------------
API_KEY = "865a20d4f77b4d92a52002d071ccfa04"

headers = {"X-Auth-Token": API_KEY}

# Premier League (örnek)
url = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"

@st.cache_data
def load_data():
    res = requests.get(url, headers=headers)
    data = res.json()

    matches = []

    for m in data["matches"]:
        if m["score"]["fullTime"]["home"] is not None:
            matches.append({
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": m["score"]["fullTime"]["home"],
                "away_goals": m["score"]["fullTime"]["away"]
            })

    return pd.DataFrame(matches)

df = load_data()

# ----------------------------
# ELO OLUŞTUR
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

    return df

df_ml = create_features(df)

# ----------------------------
# ML MODEL
# ----------------------------
features = ["elo_diff", "total_goals"]
X = df_ml[features]
y = df_ml["result"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# ----------------------------
# POISSON
# ----------------------------
def calculate_strengths(df):
    df = df.copy()
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

attack, defense = calculate_strengths(df)

# ----------------------------
# TAHMİN
# ----------------------------
def predict(home_team, away_team):

    home_lambda = attack[home_team] * defense[away_team] * 1.2
    away_lambda = attack[away_team] * defense[home_team]

    elo_diff = elo[home_team] - elo[away_team]

    ml_input = pd.DataFrame([[elo_diff, home_lambda + away_lambda]],
                            columns=["elo_diff", "total_goals"])

    result = model.predict(ml_input)[0]

    results = []

    for i in range(6):
        for j in range(6):
            prob = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)

            if result == 1 and i > j:
                prob *= 1.2
            elif result == -1 and i < j:
                prob *= 1.2
            elif result == 0 and i == j:
                prob *= 1.2

            results.append((i, j, prob))

    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results[:5], result

# ----------------------------
# UI
# ----------------------------
st.title("🔥 Gerçek Veri Maç Tahmin AI")

teams = sorted(list(set(df['home_team'])))

home_team = st.selectbox("Ev Sahibi", teams)
away_team = st.selectbox("Deplasman", teams)

if st.button("Tahmin Et"):

    scores, result = predict(home_team, away_team)

    st.subheader("📊 Skor Tahminleri")
    for h, a, p in scores:
        st.write(f"{h}-{a} → %{round(p*100,2)}")

    st.subheader("🎯 Maç Sonucu")

    if result == 1:
        st.write("Ev Sahibi Kazanır")
    elif result == -1:
        st.write("Deplasman Kazanır")
    else:
        st.write("Beraberlik")apı
