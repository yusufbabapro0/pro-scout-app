import streamlit as st
import pandas as pd
import numpy as np
import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AUTO BET AI", layout="centered")

# ----------------------------
# AUTO DATA (fallback dahil)
# ----------------------------
@st.cache_data(ttl=3600)
def load_data():
    try:
        return pd.read_csv("matches.csv")
    except:
        return pd.DataFrame({
            "date":[datetime.date.today()]*20,
            "league":["TR","ENG","ESP","GER"]*5,
            "home_team":[f"T{i}" for i in range(20)],
            "away_team":[f"A{i}" for i in range(20)],
            "home_goals":np.random.randint(0,4,20),
            "away_goals":np.random.randint(0,4,20),
            "odds_home":np.random.uniform(1.5,3.5,20),
            "odds_draw":np.random.uniform(2.5,4.0,20),
            "odds_away":np.random.uniform(2.0,5.0,20)
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

    model = XGBClassifier(n_estimators=150)
    model.fit(X,y)

    models[league] = model

# ----------------------------
# VALUE BET
# ----------------------------
def value(p,o):
    return p*o - 1

# ----------------------------
# AUTO MATCH ANALYSIS
# ----------------------------
def analyze_matches(df):

    results = []

    for _,row in df.iterrows():

        league = row["league"]

        if league not in models:
            continue

        model = models[league]

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
                    "league": league,
                    "bet": m,
                    "odds": round(o,2),
                    "prob": round(p,2),
                    "ev": round(ev,2)
                })

    return pd.DataFrame(results)

# ----------------------------
# AUTO PICK ENGINE
# ----------------------------
st.title("🤖 AUTO BET AI")

if st.button("🔥 Günlük Kupon Oluştur"):

    result_df = analyze_matches(df)

    if result_df.empty:
        st.warning("Value bet bulunamadı")
    else:
        result_df = result_df.sort_values(by="ev", ascending=False)

        st.subheader("💎 En İyi Seçimler")

        picks = result_df.head(5)

        total_odds = 1

        for _,row in picks.iterrows():
            st.write(f"{row['match']} | {row['bet']} | {row['odds']} | EV:{row['ev']}")
            total_odds *= row["odds"]

        st.success(f"Toplam Kupon Oranı: {round(total_odds,2)}")

# ----------------------------
# LIVE AUTO DECISION
# ----------------------------
st.subheader("⚡ Canlı Trade")

initial_odds = st.number_input("Giriş Oranı",1.0,10.0,2.0)
live_odds = st.number_input("Canlı Oran",1.0,10.0,1.6)

if st.button("AI Karar"):

    if live_odds < initial_odds:
        st.success("💰 Kârda → Hedge / Cashout")

    elif live_odds > initial_odds:
        st.warning("📉 Risk → çık düşün")

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
