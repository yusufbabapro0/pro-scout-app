import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AUTO BET AI FINAL", layout="centered")

# ----------------------------
# DATA (HATASIZ)
# ----------------------------
@st.cache_data(ttl=3600)
def load_data():
    try:
        return pd.read_csv("matches.csv")
    except:
        # fallback dataset (hatasız çalışır)
        return pd.DataFrame({
            "date":[datetime.date.today()]*50,
            "league":["TR","ENG","ESP","GER","ITA"]*10,
            "home_team":[f"T{i}" for i in range(50)],
            "away_team":[f"A{i}" for i in range(50)],
            "home_goals":np.random.randint(0,4,50),
            "away_goals":np.random.randint(0,4,50),
            "odds_home":np.random.uniform(1.5,3.5,50),
            "odds_draw":np.random.uniform(2.5,4.0,50),
            "odds_away":np.random.uniform(2.0,5.0,50)
        })

df = load_data()

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
le = LabelEncoder()

df["home_enc"] = le.fit_transform(df["home_team"])
df["away_enc"] = le.fit_transform(df["away_team"])

df["goal_diff"] = df["home_goals"] - df["away_goals"]
df["total_goals"] = df["home_goals"] + df["away_goals"]

df["result"] = df["goal_diff"].apply(lambda x: 1 if x>0 else (0 if x==0 else -1))

features = ["home_enc","away_enc","goal_diff","total_goals"]

# ----------------------------
# LIG BAZLI MODEL
# ----------------------------
models = {}

for league in df["league"].unique():
    sub = df[df["league"] == league]

    if len(sub) < 5:
        continue

    X = sub[features]
    y = sub["result"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X,y)

    models[league] = model

# ----------------------------
# VALUE BET
# ----------------------------
def value(p,o):
    return p*o - 1

# ----------------------------
# ANALİZ
# ----------------------------
def analyze(df):

    results = []

    for _,row in df.iterrows():

        if row["league"] not in models:
            continue

        model = models[row["league"]]

        sample = pd.DataFrame([[row["home_enc"],row["away_enc"],0,2]],
                              columns=features)

        probs = model.predict_proba(sample)[0]

        markets = {
            "MS1": (probs[2], row["odds_home"]),
            "MSX": (probs[1], row["odds_draw"]),
            "MS2": (probs[0], row["odds_away"])
        }

        for m,(p,o) in markets.items():

            ev = value(p,o)

            if ev > 0.10:
                results.append({
                    "match": f"{row['home_team']} vs {row['away_team']}",
                    "league": row["league"],
                    "bet": m,
                    "odds": round(o,2),
                    "prob": round(p,2),
                    "ev": round(ev,2)
                })

    return pd.DataFrame(results)

# ----------------------------
# UI
# ----------------------------
st.title("💀 AUTO BET AI (FINAL)")

# ----------------------------
# KUPON
# ----------------------------
if st.button("🔥 Günlük Kupon"):

    res = analyze(df)

    if res.empty:
        st.warning("Value bet yok")
    else:
        res = res.sort_values(by="ev", ascending=False)

        picks = res.head(5)

        total_odds = 1

        st.subheader("💎 Seçimler")

        for _,r in picks.iterrows():
            st.write(f"{r['match']} | {r['bet']} | {r['odds']} | EV:{r['ev']}")
            total_odds *= r["odds"]

        st.success(f"Toplam Oran: {round(total_odds,2)}")

# ----------------------------
# LIVE TRADE
# ----------------------------
st.subheader("⚡ Canlı Trade")

initial_odds = st.number_input("Giriş Oranı",1.0,10.0,2.0)
live_odds = st.number_input("Canlı Oran",1.0,10.0,1.6)

if st.button("AI Karar"):

    if live_odds < initial_odds:
        st.success("💰 Hedge / Cashout")

    elif live_odds > initial_odds:
        st.warning("📉 Risk")

    else:
        st.write("Bekle")

# ----------------------------
# HEDGE
# ----------------------------
st.subheader("💰 Hedge")

stake = st.number_input("Stake",10,10000,100)

if st.button("Hedge Hesapla"):

    hedge_stake = (stake * initial_odds) / live_odds
    st.success(f"Hedge: {round(hedge_stake,2)}")
