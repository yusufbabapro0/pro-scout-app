import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Bet AI Final", layout="centered")

# ----------------------------
# DATA
# ----------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("matches.csv")
    except:
        # fallback (hatasız demo veri)
        return pd.DataFrame({
            "home_team":[f"T{i}" for i in range(30)],
            "away_team":[f"A{i}" for i in range(30)],
            "home_goals":np.random.randint(0,4,30),
            "away_goals":np.random.randint(0,4,30),
            "odds_home":np.random.uniform(1.5,3.0,30),
            "odds_draw":np.random.uniform(2.5,4.0,30),
            "odds_away":np.random.uniform(2.0,5.0,30)
        })

df = load_data()

# ----------------------------
# TEAM ENCODING
# ----------------------------
le = LabelEncoder()
teams = pd.concat([df["home_team"], df["away_team"]]).unique()
le.fit(teams)

df["home_enc"] = le.transform(df["home_team"])
df["away_enc"] = le.transform(df["away_team"])

# ----------------------------
# FEATURES
# ----------------------------
df["goal_diff"] = df["home_goals"] - df["away_goals"]
df["total_goals"] = df["home_goals"] + df["away_goals"]

df["result"] = df["goal_diff"].apply(lambda x: 1 if x>0 else (0 if x==0 else -1))

features = ["home_enc","away_enc","goal_diff","total_goals"]

# ----------------------------
# MODEL
# ----------------------------
X = df[features]
y = df["result"]

model = RandomForestClassifier(n_estimators=150)
model.fit(X,y)

# ----------------------------
# VALUE FUNCTION
# ----------------------------
def value(p,o):
    return p*o - 1

# ----------------------------
# UI
# ----------------------------
st.title("💀 Bet AI Final")

home_team = st.selectbox("Ev Sahibi", sorted(teams))
away_team = st.selectbox("Deplasman", sorted(teams))

home_odds = st.number_input("MS1 Oran",1.0,10.0,2.0)
draw_odds = st.number_input("MSX Oran",1.0,10.0,3.2)
away_odds = st.number_input("MS2 Oran",1.0,10.0,3.5)

# ----------------------------
# PREDICT
# ----------------------------
if st.button("Analiz Et"):

    home_enc = le.transform([home_team])[0]
    away_enc = le.transform([away_team])[0]

    sample = pd.DataFrame([[home_enc, away_enc, 0, 2]],
                          columns=features)

    probs = model.predict_proba(sample)[0]

    p_home = probs[2]
    p_draw = probs[1]
    p_away = probs[0]

    st.subheader("📊 Olasılıklar")
    st.write(f"Ev: %{round(p_home*100,1)}")
    st.write(f"Beraberlik: %{round(p_draw*100,1)}")
    st.write(f"Dep: %{round(p_away*100,1)}")

    st.subheader("💎 Value Bet")

    if value(p_home, home_odds) > 0:
        st.success("MS1 VALUE")

    if value(p_draw, draw_odds) > 0:
        st.success("MSX VALUE")

    if value(p_away, away_odds) > 0:
        st.success("MS2 VALUE")

# ----------------------------
# LIVE TRADE
# ----------------------------
st.subheader("⚡ Canlı Trade")

initial_odds = st.number_input("Giriş Oranı",1.0,10.0,2.0)
live_odds = st.number_input("Canlı Oran",1.0,10.0,1.6)

if st.button("Karar Ver"):

    if live_odds < initial_odds:
        st.success("💰 Hedge / Cashout")

    elif live_odds > initial_odds:
        st.warning("📉 Risk")

    else:
        st.write("Bekle")

# ----------------------------
# HEDGE
# ----------------------------
st.subheader("💰 Hedge Hesapla")

stake = st.number_input("Stake",10,10000,100)

if st.button("Hedge"):

    hedge_stake = (stake * initial_odds) / live_odds
    st.success(f"Hedge miktarı: {round(hedge_stake,2)}")
