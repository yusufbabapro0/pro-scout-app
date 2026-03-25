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
# MAÇ + ORAN ÇEK
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

        odds = {"home":2.0,"draw":3.0,"away":3.0,"over":1.8,"btts":1.7}

        # oran endpoint pahalı olduğu için fallback
        matches.append({
            "league": m["league"]["name"],
            "home": m["teams"]["home"]["name"],
            "away": m["teams"]["away"]["name"],
            "odds": odds
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

    markets = {
        "MS1": sum(p for i,j,p in probs if i>j),
        "MSX": sum(p for i,j,p in probs if i==j),
        "MS2": sum(p for i,j,p in probs if i<j),
        "ÜST 2.5": sum(p for i,j,p in probs if i+j>=3),
        "KG VAR": sum(p for i,j,p in probs if i>0 and j>0)
    }

    return markets

# ----------------------------
# KUPON ENGINE (EN KRİTİK)
# ----------------------------
def build_coupon(df):

    picks = []

    for _, row in df.iterrows():

        markets = predict(row["home"], row["away"])
        odds = row["odds"]

        for k, prob in markets.items():

            if k == "MS1":
                o = odds["home"]
            elif k == "MSX":
                o = odds["draw"]
            elif k == "MS2":
                o = odds["away"]
            elif k == "ÜST 2.5":
                o = odds["over"]
            else:
                o = odds["btts"]

            val = value(prob, o)

            # 🔥 EN KRİTİK FİLTRE
            if val > 0.15 and prob > 0.60:

                picks.append({
                    "match": f"{row['home']} vs {row['away']}",
                    "pick": k,
                    "prob": prob,
                    "odds": o,
                    "value": val
                })

    # en iyi value sıralama
    picks = sorted(picks, key=lambda x: x["value"], reverse=True)

    banko = [p for p in picks if p["prob"] > 0.70][:2]
    orta = [p for p in picks if 0.60 < p["prob"] <= 0.70][:2]
    risk = [p for p in picks if p["value"] > 0.25][:1]

    return banko, orta, risk

# ----------------------------
# UI
# ----------------------------
st.title("💀 ULTIMATE BETTING AI")

df = get_data()

if df is None:
    st.error("API limit dolmuş olabilir")
else:

    if st.button("🚀 GÜNLÜK KUPON ÜRET"):

        banko, orta, risk = build_coupon(df)

        st.subheader("🔥 BANKO (En Güvenli)")
        for b in banko:
            st.success(f"{b['match']} → {b['pick']} | %{round(b['prob']*100,1)} | oran {b['odds']} | value {round(b['value'],2)}")

        st.subheader("⚖️ ORTA RİSK")
        for o in orta:
            st.info(f"{o['match']} → {o['pick']} | %{round(o['prob']*100,1)} | oran {o['odds']}")

        st.subheader("🎲 HIGH VALUE (Riskli)")
        for r in risk:
            st.warning(f"{r['match']} → {r['pick']} | value {round(r['value'],2)}")

        st.subheader("💰 AKILLI KUPON")

        combo = banko[:1] + orta[:1]

        for c in combo:
            st.write(f"{c['match']} → {c['pick']}")
