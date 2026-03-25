import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# CSV YÜKLE / DEMO
# ----------------------------
uploaded_file = st.file_uploader("CSV yükle (opsiyonel)", type=["csv"])

def load_data():
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        teams = ["Galatasaray","Fenerbahce","Besiktas","Trabzonspor"]
        np.random.seed(42)

        rows = []
        for _ in range(800):
            home = np.random.choice(teams)
            away = np.random.choice([t for t in teams if t != home])

            rows.append([
                home, away,
                np.random.poisson(1.6),
                np.random.poisson(1.2),
                np.random.uniform(1.5, 3.5),
                np.random.uniform(2.5, 4.0),
                np.random.uniform(2.5, 4.5)
            ])

        return pd.DataFrame(rows, columns=[
            "home_team","away_team","home_goals","away_goals",
            "home_odds","draw_odds","away_odds"
        ])

df = load_data()

teams = pd.concat([df['home_team'], df['away_team']]).unique()
elo = {team: 1500 for team in teams}

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def create_features(df):
    df = df.copy()

    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["result"] = df["goal_diff"].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))

    df["total_goals"] = df["home_goals"] + df["away_goals"]

    df["prob_home"] = 1 / df["home_odds"]
    df["prob_draw"] = 1 / df["draw_odds"]
    df["prob_away"] = 1 / df["away_odds"]

    return df

df_ml = create_features(df)

# ----------------------------
# ML MODEL
# ----------------------------
features = ["total_goals","prob_home","prob_draw","prob_away"]

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
def predict(home_team, away_team):

    home_lambda = attack[home_team] * defense[away_team] * 1.25
    away_lambda = attack[away_team] * defense[home_team]

    results = []

    for i in range(6):
        for j in range(6):
            prob = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
            results.append((i, j, prob))

    results = sorted(results, key=lambda x: x[2], reverse=True)

    # olasılıklar
    home_win = sum(p for i,j,p in results if i > j)
    draw = sum(p for i,j,p in results if i == j)
    away_win = sum(p for i,j,p in results if i < j)

    over25 = sum(p for i,j,p in results if (i+j) > 2)
    under25 = 1 - over25

    btts_yes = sum(p for i,j,p in results if i > 0 and j > 0)
    btts_no = 1 - btts_yes

    # HT tahmini (basit yaklaşım)
    ht_home = home_win * 0.6
    ht_draw = draw * 0.7
    ht_away = away_win * 0.6

    # en güvenli bahis
    markets = {
        "Ev Sahibi": home_win,
        "Beraberlik": draw,
        "Deplasman": away_win,
        "2.5 Üst": over25,
        "2.5 Alt": under25,
        "KG Var": btts_yes,
        "KG Yok": btts_no
    }

    best_bet = max(markets, key=markets.get)
    confidence = markets[best_bet]

    return results[:5], markets, best_bet, confidence

# ----------------------------
# UI
# ----------------------------
st.title("🔥 Gelişmiş Bahis Tahmin AI")

home_team = st.selectbox("Ev Sahibi", teams)
away_team = st.selectbox("Deplasman", teams)

if st.button("Tahmin Et"):

    scores, markets, best_bet, confidence = predict(home_team, away_team)

    st.subheader("📊 Skor Tahminleri")
    for h,a,p in scores:
        st.write(f"{h}-{a} → %{round(p*100,2)}")

    st.subheader("📈 Bahis Analizi")

    for k,v in markets.items():
        st.write(f"{k} → %{round(v*100,2)}")

    st.subheader("💎 En Güvenli Bahis")
    st.success(best_bet)

    st.subheader("📊 Güven Skoru")
    st.write(f"%{round(confidence*100,2)}")
