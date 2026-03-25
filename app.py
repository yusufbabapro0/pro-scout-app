import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import pickle
from scipy.stats import poisson

st.set_page_config(page_title="FINAL BOSS AI", layout="centered")

API_KEY = "865a20d4f77b4d92a52002d071ccfa04"

headers = {
    "x-apisports-key": API_KEY
}

# ----------------------------
# TAKIM İSTATİSTİK (GERÇEK)
# ----------------------------
@st.cache_data(ttl=3600)
def get_team_stats(team_id, league_id, season=2024):

    url = f"https://v3.football.api-sports.io/teams/statistics?team={team_id}&league={league_id}&season={season}"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        return None

    data = res.json()["response"]

    return {
        "attack": data["goals"]["for"]["average"]["total"],
        "defense": data["goals"]["against"]["average"]["total"]
    }

# ----------------------------
# MAÇLAR (GERÇEK)
# ----------------------------
@st.cache_data(ttl=300)
def get_matches():

    url = "https://v3.football.api-sports.io/fixtures?next=20"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        return None

    data = res.json()["response"]

    matches = []

    for m in data:
        matches.append({
            "league_id": m["league"]["id"],
            "league": m["league"]["name"],
            "home": m["teams"]["home"]["name"],
            "away": m["teams"]["away"]["name"],
            "home_id": m["teams"]["home"]["id"],
            "away_id": m["teams"]["away"]["id"]
        })

    return pd.DataFrame(matches)

# ----------------------------
# MODEL (KAYITLI)
# ----------------------------
MODEL_FILE = "model.pkl"

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        weights = pickle.load(f)
else:
    weights = {"attack":1.2, "defense":1.0}

# ----------------------------
# VALUE BET
# ----------------------------
def value(prob, odds):
    return prob * odds - 1

# ----------------------------
# TAHMİN (xG benzeri)
# ----------------------------
def predict(home_stats, away_stats):

    home_xg = home_stats["attack"] * weights["attack"] / away_stats["defense"]
    away_xg = away_stats["attack"] * weights["attack"] / home_stats["defense"]

    probs = []

    for i in range(6):
        for j in range(6):
            p = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
            probs.append((i,j,p))

    probs = sorted(probs, key=lambda x: x[2], reverse=True)

    home_win = sum(p for i,j,p in probs if i>j)
    draw = sum(p for i,j,p in probs if i==j)
    away_win = sum(p for i,j,p in probs if i<j)

    over25 = sum(p for i,j,p in probs if i+j>=3)
    btts = sum(p for i,j,p in probs if i>0 and j>0)

    markets = {
        "Ev Sahibi": home_win,
        "Beraberlik": draw,
        "Deplasman": away_win,
        "2.5 Üst": over25,
        "KG Var": btts
    }

    return probs[:3], markets

# ----------------------------
# UI
# ----------------------------
st.title("🔥 FINAL BOSS Betting AI")

df = get_matches()

if df is None:
    st.error("API çalışmıyor (limit dolmuş olabilir)")
else:

    match_names = df.apply(lambda x: f"{x['home']} vs {x['away']} ({x['league']})", axis=1)

    selected = st.selectbox("Maç seç", match_names)

    row = df.iloc[match_names.tolist().index(selected)]

    st.subheader("💰 Oran Gir")

    odds = {
        "Ev Sahibi": st.number_input("Ev", 1.0, 10.0, 2.0),
        "Beraberlik": st.number_input("X", 1.0, 10.0, 3.2),
        "Deplasman": st.number_input("2", 1.0, 10.0, 3.5),
        "2.5 Üst": st.number_input("Üst", 1.0, 10.0, 1.8),
        "KG Var": st.number_input("KG", 1.0, 10.0, 1.7)
    }

    if st.button("FINAL ANALİZ"):

        home_stats = get_team_stats(row["home_id"], row["league_id"])
        away_stats = get_team_stats(row["away_id"], row["league_id"])

        if not home_stats or not away_stats:
            st.error("Takım verisi çekilemedi")
        else:

            scores, markets = predict(home_stats, away_stats)

            st.subheader("📊 Skor Tahmini")
            for h,a,p in scores:
                st.write(f"{h}-{a} → %{round(p*100,1)}")

            st.subheader("📈 Marketler")

            sorted_m = sorted(markets.items(), key=lambda x: x[1], reverse=True)

            for k,v in sorted_m:
                pct = round(v*100,1)

                if pct > 75:
                    st.success(f"{k} → %{pct} (BANKO)")
                elif pct > 60:
                    st.info(f"{k} → %{pct}")
                else:
                    st.warning(f"{k} → %{pct}")

            st.subheader("💎 VALUE BET")

            for k,v in markets.items():
                val = value(v, odds[k])
                if val > 0:
                    st.success(f"{k} → VALUE: {round(val,2)}")

# ----------------------------
# MODEL GELİŞTİRME
# ----------------------------
st.subheader("🧠 Model Geliştir")

if st.button("Modeli Güçlendir"):

    weights["attack"] += 0.05
    weights["defense"] += 0.02

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(weights, f)

    st.success("Model geliştirildi 🚀")
