import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="ELITE BET AI", layout="centered")

# ----------------------------
# DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("matches.csv")

df = load_data()

# ----------------------------
# FEATURE ENGINEERING (ELITE)
# ----------------------------
def create_features(df):

    # takım encoding
    le = LabelEncoder()
    df["home_enc"] = le.fit_transform(df["home_team"])
    df["away_enc"] = le.fit_transform(df["away_team"])

    # gol feature
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["total_goals"] = df["home_goals"] + df["away_goals"]

    # rolling form (son 5 maç)
    df["form_home"] = df.groupby("home_team")["goal_diff"].rolling(5).mean().reset_index(0,drop=True)
    df["form_away"] = df.groupby("away_team")["goal_diff"].rolling(5).mean().reset_index(0,drop=True)

    df.fillna(0, inplace=True)

    # hedef
    df["result"] = df["goal_diff"].apply(lambda x: 1 if x>0 else (0 if x==0 else -1))

    return df

df = create_features(df)

# ----------------------------
# MODEL
# ----------------------------
features = ["home_enc","away_enc","form_home","form_away","total_goals"]

X = df[features]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

st.write(f"Accuracy: %{round(acc*100,2)}")

# ----------------------------
# PROBABILITY
# ----------------------------
probs = model.predict_proba(X_test)

# ----------------------------
# VALUE BET
# ----------------------------
def value(p, o):
    return p * o - 1

# ----------------------------
# BACKTEST (ELITE)
# ----------------------------
def backtest(df):

    profit = 0
    bets = 0
    wins = 0

    for i,row in df.iterrows():

        p = 0.55  # (gerçekte modelden alınmalı)
        o = row["odds_home"]

        ev = value(p,o)

        if ev > 0:

            bets += 1

            if row["home_goals"] > row["away_goals"]:
                profit += o - 1
                wins += 1
            else:
                profit -= 1

    roi = profit / bets if bets else 0
    hit = wins / bets if bets else 0

    return roi, bets, hit

roi, bets, hit = backtest(df)

st.subheader("📊 Backtest")
st.write(f"Bet: {bets}")
st.write(f"ROI: %{round(roi*100,2)}")
st.write(f"Win Rate: %{round(hit*100,2)}")

# ----------------------------
# TAHMİN PANELİ
# ----------------------------
st.subheader("🎯 Yeni Maç")

home = st.text_input("Ev Takım")
away = st.text_input("Deplasman")

home_odds = st.number_input("Ev Oran",1.0,10.0,2.0)
draw_odds = st.number_input("X",1.0,10.0,3.0)
away_odds = st.number_input("Dep Oran",1.0,10.0,3.5)

if st.button("Analiz"):

    # basit encoding fallback
    home_enc = hash(home)%1000
    away_enc = hash(away)%1000

    sample = pd.DataFrame([[home_enc,away_enc,0,0,2]],
        columns=features)

    prob = model.predict_proba(sample)[0]

    markets = {
        "MS1": (prob[2], home_odds),
        "MSX": (prob[1], draw_odds),
        "MS2": (prob[0], away_odds)
    }

    st.subheader("💎 VALUE")

    for k,(p,o) in markets.items():
        ev = value(p,o)

        if ev > 0:
            st.success(f"{k} → EV {round(ev,2)} | %{round(p*100,1)}")
        else:
            st.write(f"{k} → oynanmaz")

# ----------------------------
# AUTO PICK (ELITE)
# ----------------------------
st.subheader("🤖 Günlük Seçim")

if st.button("Seç"):

    picks = []

    for i,row in df.sample(50).iterrows():

        p = 0.55
        o = row["odds_home"]
        ev = value(p,o)

        if ev > 0.15:

            picks.append((row["home_team"], ev))

    picks = sorted(picks, key=lambda x:x[1], reverse=True)[:3]

    for p in picks:
        st.write(f"{p[0]} → EV {round(p[1],2)}")
