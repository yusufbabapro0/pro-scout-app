import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="ULTIMATE BET AI", layout="centered")

API_KEY = "865a20d4f77b4d92a52002d071ccfa04"

headers = {
    "x-apisports-key": API_KEY
}

# ----------------------------
# MAÇ ÇEK
# ----------------------------
@st.cache_data(ttl=1800)
def get_data():
    url = "https://v3.football.api-sports.io/fixtures?next=30"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        return None

    data = res.json()["response"]

    matches = []

    for m in data:
        matches.append({
            "home": m["teams"]["home"]["name"],
            "away": m["teams"]["away"]["name"],
            "odds": {
                "home": 2.0,
                "draw": 3.2,
                "away": 3.5,
                "over": 1.8,
                "btts": 1.7
            }
        })

    return pd.DataFrame(matches)

# ----------------------------
# TAKIM GÜÇ
# ----------------------------
def power(team):
    np.random.seed(abs(hash(team)) % 1000)
    return np.random.uniform(0.9, 1.8)

# ----------------------------
# VALUE
# ----------------------------
def value(prob, odds):
    return prob * odds - 1

# ----------------------------
# TAHMİN
# ----------------------------
def predict(home, away):

    home_xg = power(home) * 1.3
    away_xg = power(away)

    probs = []

    for i in range(6):
        for j in range(6):
            p = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
            probs.append((i,j,p))

    probs = sorted(probs, key=lambda x: x[2], reverse=True)

    return {
        "MS1": sum(p for i,j,p in probs if i>j),
        "MSX": sum(p for i,j,p in probs if i==j),
        "MS2": sum(p for i,j,p in probs if i<j),
        "ÜST 2.5": sum(p for i,j,p in probs if i+j>=3),
        "KG VAR": sum(p for i,j,p in probs if i>0 and j>0)
    }

# ----------------------------
# KUPON ENGINE (FIXED)
# ----------------------------
def build_coupon(df):

    picks = []

    for _, row in df.iterrows():

        markets = predict(row["home"], row["away"])
        odds = row["odds"]

        for k, prob in markets.items():

            o = odds["home"] if k=="MS1" else odds["draw"] if k=="MSX" else odds["away"] if k=="MS2" else odds["over"] if k=="ÜST 2.5" else odds["btts"]

            val = value(prob, o)

            # 🔧 YUMUŞATILDI
            if val > 0.05 and prob > 0.55:
                picks.append({
                    "match": f"{row['home']} vs {row['away']}",
                    "pick": k,
                    "prob": prob,
                    "odds": o,
                    "value": val
                })

    # ❗ FALLBACK (en kritik fix)
    if len(picks) < 3:
        for _, row in df.iterrows():
            markets = predict(row["home"], row["away"])
            best = max(markets.items(), key=lambda x: x[1])

            picks.append({
                "match": f"{row['home']} vs {row['away']}",
                "pick": best[0],
                "prob": best[1],
                "odds": 1.8,
                "value": 0
            })

    picks = sorted(picks, key=lambda x: x["prob"], reverse=True)

    return picks[:5]

# ----------------------------
# UI
# ----------------------------
st.title("💀 ULTIMATE BET AI (FIXED)")

df = get_data()

if df is None:
    st.error("API çalışmıyor")
else:

    if st.button("🚀 KUPON ÜRET"):

        picks = build_coupon(df)

        st.subheader("🔥 GÜNLÜK KUPON")

        for p in picks:
            pct = round(p["prob"]*100,1)

            if pct > 70:
                st.success(f"{p['match']} → {p['pick']} (%{pct})")
            elif pct > 60:
                st.info(f"{p['match']} → {p['pick']} (%{pct})")
            else:
                st.warning(f"{p['match']} → {p['pick']} (%{pct})")
