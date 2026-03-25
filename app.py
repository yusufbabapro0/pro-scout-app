import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Live Bet AI", layout="centered")

# ----------------------------
# API AYAR
# ----------------------------
API_KEY = "865a20d4f77b4d92a52002d071ccfa04"

headers = {
    "x-apisports-key": API_KEY
}

# ----------------------------
# MAÇLARI ÇEK
# ----------------------------
@st.cache_data(ttl=3600)
def get_matches():

    url = "https://v3.football.api-sports.io/fixtures?live=all"

    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        return None

    data = res.json()

    matches = []

    for m in data["response"]:
        matches.append({
            "league": m["league"]["name"],
            "home": m["teams"]["home"]["name"],
            "away": m["teams"]["away"]["name"],
            "minute": m["fixture"]["status"]["elapsed"] or 0
        })

    return pd.DataFrame(matches)

# ----------------------------
# TAKIM GÜÇLERİ (DEMO fallback)
# ----------------------------
def fake_strength(team):
    np.random.seed(abs(hash(team)) % 1000)
    return np.random.uniform(0.8, 1.8)

# ----------------------------
# TAHMİN
# ----------------------------
def predict(home, away, minute):

    home_l = fake_strength(home) * 1.3
    away_l = fake_strength(away)

    # canlı etkisi
    factor = minute / 90
    home_l *= (1 - factor)
    away_l *= (1 - factor)

    probs = []

    for i in range(6):
        for j in range(6):
            p = poisson.pmf(i, home_l) * poisson.pmf(j, away_l)
            probs.append((i,j,p))

    probs = sorted(probs, key=lambda x: x[2], reverse=True)

    markets = {
        "Ev Sahibi": sum(p for i,j,p in probs if i>j),
        "Beraberlik": sum(p for i,j,p in probs if i==j),
        "Deplasman": sum(p for i,j,p in probs if i<j),
        "2.5 Üst": sum(p for i,j,p in probs if i+j>=3),
        "KG Var": sum(p for i,j,p in probs if i>0 and j>0)
    }

    sorted_markets = sorted(markets.items(), key=lambda x: x[1], reverse=True)

    return probs[:3], sorted_markets

# ----------------------------
# UI
# ----------------------------
st.title("🔥 Live Betting AI")

df = get_matches()

if df is None or df.empty:
    st.error("API veri çekemedi (limit dolmuş olabilir)")
else:

    st.subheader("📡 Canlı Maçlar")

    match_list = df.apply(lambda x: f"{x['home']} vs {x['away']} ({x['minute']} dk)", axis=1)

    selected = st.selectbox("Maç seç", match_list)

    row = df.iloc[match_list.tolist().index(selected)]

    if st.button("Analiz Et"):

        scores, markets = predict(row["home"], row["away"], row["minute"])

        st.subheader("📊 Skor Tahmini")
        for h,a,p in scores:
            st.write(f"{h}-{a} → %{round(p*100,1)}")

        st.subheader("📈 En İyi Seçimler")

        for k,v in markets:
            pct = round(v*100,1)

            if pct > 75:
                st.success(f"{k} → %{pct} (BANKO)")
            elif pct > 60:
                st.info(f"{k} → %{pct}")
            else:
                st.warning(f"{k} → %{pct}")

        st.subheader("🔥 Kupon")

        for b in markets[:2]:
            st.write(f"{b[0]} → %{round(b[1]*100,1)}")
