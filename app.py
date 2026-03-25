import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# TAKIMLAR (SÜPER LİG)
# ----------------------------
teams = [
    "Galatasaray","Fenerbahce","Besiktas","Trabzonspor",
    "Basaksehir","Adana Demirspor","Kasimpasa","Antalyaspor"
]

np.random.seed(42)

# ----------------------------
# SAHTE AMA GERÇEKÇİ VERİ
# ----------------------------
rows = []
for _ in range(800):  # daha büyük veri
    home = np.random.choice(teams)
    away = np.random.choice([t for t in teams if t != home])

    home_goals = np.random.poisson(1.6)
    away_goals = np.random.poisson(1.2)

    rows.append([home, away, home_goals, away_goals])

df = pd.DataFrame(rows, columns=["home_team","away_team","home_goals","away_goals"])

# ----------------------------
# ELO
# ----------------------------
elo = {team: 1500 + np.random.randint(-100,100) for team in teams}

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
X = df_ml[["elo_diff","total_goals"]]
y = df_ml["result"]

model = RandomForestClassifier(n_estimators=200)
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
def predict(home_team, away_team):

    home_lambda = attack[home_team] * defense[away_team] * 1.25
    away_lambda = attack[away_team] * defense[home_team]

    elo_diff = elo[home_team] - elo[away_team]

    ml_input = pd.DataFrame([[elo_diff, home_lambda + away_lambda]],
                            columns=["elo_diff","total_goals"])

    result = model.predict(ml_input)[0]

    results = []

    for i in range(6):
        for j in range(6):
            prob = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)

            if result == 1 and i > j:
                prob *= 1.3
            elif result == -1 and i < j:
                prob *= 1.3
            elif result == 0 and i == j:
                prob *= 1.3

            results.append((i,j,prob))

    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results[:5], result

# ----------------------------
# UI
# ----------------------------
st.title("🔥 Süper Lig Tahmin AI (API'siz)")

home_team = st.selectbox("Ev Sahibi", teams)
away_team = st.selectbox("Deplasman", teams)

if st.button("Tahmin Et"):

    scores, result = predict(home_team, away_team)

    st.subheader("📊 Skor Tahminleri")
    for h,a,p in scores:
        st.write(f"{h}-{a} → %{round(p*100,2)}")

    st.subheader("🎯 Maç Sonucu")

    if result == 1:
        st.write("Ev Sahibi Kazanır")
    elif result == -1:
        st.write("Deplasman Kazanır")
    else:
        st.write("Beraberlik")
