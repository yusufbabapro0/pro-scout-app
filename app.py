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
# SESSION STATE
# ----------------------------
if "picks" not in st.session_state:
    st.session_state.picks = []

# ----------------------------
# MAÇ ÇEK
# ----------------------------
@st.cache_data(ttl=1800)
def get_data():
    try:
        url = "https://v3.football.api-sports.io/fixtures?next=30"
        res = requests.get(url, headers=headers)

        if res.status_code != 200:
            return pd.DataFrame()

        data = res.json()["response"]

        matches = []

        for m in data:
            matches.append({
                "home": m["teams"]["home"]["name"],
                "away": m["teams"]["away"]["name"]
            })

        return pd.DataFrame(matches)

    except:
        return pd.DataFrame()

# ----------------------------
# FALLBACK DATA (KRİTİK)
# ----------------------------
def fallback_data():
    return pd.DataFrame([
        {"home":"Galatasaray","away":"Besiktas"},
        {"home":"Fenerbahce","away":"Trabzonspor"},
        {"home":"Barcelona","away":"Sevilla"},
        {"home":"Arsenal","away":"Chelsea"}
    ])

# ----------------------------
# MODEL
# ----------------------------
def power(team):
    np.random.seed(abs(hash(team)) % 1000)
    return np.random.uniform(0.9, 1.8)

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
# KUPON OLUŞTUR
# ----------------------------
def build_coupon(df):

    picks = []

    for _, row in df.iterrows():
        markets = predict(row["home"], row["away"])
        best = max(markets.items(), key=lambda x: x[1])

        picks.append({
            "match": f"{row['home']} vs {row['away']}",
            "pick": best[0],
            "prob": best[1]
        })

    picks = sorted(picks, key=lambda x: x["prob"], reverse=True)

    return picks[:5]

# ----------------------------
# UI
# ----------------------------
st.title("💀 ULTIMATE BET AI (FINAL FIX)")

df = get_data()

# ❗ API boşsa fallback kullan
if df.empty:
    st.warning("API veri gelmedi → demo kupon gösteriliyor")
    df = fallback_data()

if st.button("🚀 KUPON ÜRET"):

    st.session_state.picks = build_coupon(df)

# ----------------------------
# SONUÇ GÖSTER
# ----------------------------
if st.session_state.picks:

    st.subheader("🔥 GÜNLÜK KUPON")

    for p in st.session_state.picks:

        pct = round(p["prob"] * 100, 1)

        if pct > 70:
            st.success(f"{p['match']} → {p['pick']} (%{pct})")
        elif pct > 60:
            st.info(f"{p['match']} → {p['pick']} (%{pct})")
        else:
            st.warning(f"{p['match']} → {p['pick']} (%{pct})")
