import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ----------------------------
# CSV / DEMO
# ----------------------------
uploaded_file = st.file_uploader("CSV yükle", type=["csv"])

def load_data():
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        teams = ["Galatasaray","Fenerbahce","Besiktas","Trabzonspor"]
        np.random.seed(42)

        rows = []
        for _ in range(1000):
            home = np.random.choice(teams)
            away = np.random.choice([t for t in teams if t != home])

            rows.append([
                home, away,
                np.random.poisson(1.6),
                np.random.poisson(1.2)
            ])

        return pd.DataFrame(rows, columns=[
            "home_team","away_team","home_goals","away_goals"
        ])

df = load_data()

teams = pd.concat([df['home_team'], df['away_team']]).unique()

# ----------------------------
# TAKIM GÜÇLERİ
# ----------------------------
def calculate_strengths(df):
    attack = {}
    defense = {}

    for team in teams:
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]

        attack[team] = (home["home_goals"].mean() + away["away_goals"].mean()) / 2
        defense[team] = (home["away_goals"].mean() + away["home_goals"].mean()) / 2

    return attack, defense

attack, defense = calculate_strengths(df)

# ----------------------------
# TAHMİN
# ----------------------------
def predict(home, away):

    home_l = attack[home] * defense[away] * 1.25
    away_l = attack[away] * defense[home]

    probs = []

    for i in range(6):
        for j in range(6):
            p = poisson.pmf(i, home_l) * poisson.pmf(j, away_l)
            probs.append((i,j,p))

    probs = sorted(probs, key=lambda x: x[2], reverse=True)

    # temel
    home_win = sum(p for i,j,p in probs if i>j)
    draw = sum(p for i,j,p in probs if i==j)
    away_win = sum(p for i,j,p in probs if i<j)

    # over
    over05 = sum(p for i,j,p in probs if (i+j)>=1)
    over15 = sum(p for i,j,p in probs if (i+j)>=2)
    over25 = sum(p for i,j,p in probs if (i+j)>=3)

    # btts
    btts = sum(p for i,j,p in probs if i>0 and j>0)

    # double chance
    dc_1x = home_win + draw
    dc_x2 = draw + away_win
    dc_12 = home_win + away_win

    # HT (yaklaşık)
    ht_home = home_win * 0.6
    ht_draw = draw * 0.7
    ht_away = away_win * 0.6

    # HT/FT
    htft = {
        "1/1": ht_home * home_win,
        "X/1": ht_draw * home_win,
        "2/1": ht_away * home_win,
        "1/X": ht_home * draw,
        "X/X": ht_draw * draw,
        "2/X": ht_away * draw,
        "1/2": ht_home * away_win,
        "X/2": ht_draw * away_win,
        "2/2": ht_away * away_win,
    }

    markets = {
        "Ev Sahibi": home_win,
        "Beraberlik": draw,
        "Deplasman": away_win,
        "1X": dc_1x,
        "X2": dc_x2,
        "12": dc_12,
        "0.5 Üst": over05,
        "1.5 Üst": over15,
        "2.5 Üst": over25,
        "KG Var": btts
    }

    best = max(markets, key=markets.get)
    confidence = markets[best]

    # risk
    if confidence > 0.75:
        risk = "Düşük Risk"
    elif confidence > 0.60:
        risk = "Orta Risk"
    else:
        risk = "Yüksek Risk"

    # kombine öneri (en iyi 2)
    sorted_markets = sorted(markets.items(), key=lambda x: x[1], reverse=True)
    combo = sorted_markets[:2]

    return probs[:5], markets, htft, best, confidence, risk, combo

# ----------------------------
# UI
# ----------------------------
st.title("🔥 ULTRA Betting AI")

home = st.selectbox("Ev Sahibi", teams)
away = st.selectbox("Deplasman", teams)

if st.button("Tahmin Et"):

    scores, markets, htft, best, conf, risk, combo = predict(home, away)

    st.subheader("📊 Skor Tahminleri")
    for h,a,p in scores:
        st.write(f"{h}-{a} → %{round(p*100,2)}")

    st.subheader("📈 Tüm Marketler")
    for k,v in markets.items():
        st.write(f"{k} → %{round(v*100,2)}")

    st.subheader("⏱️ HT/FT")
    for k,v in htft.items():
        st.write(f"{k} → %{round(v*100,2)}")

    st.subheader("💎 En Güvenli Bahis")
    st.success(best)

    st.subheader("📊 Güven")
    st.write(f"%{round(conf*100,2)}")

    st.subheader("⚠️ Risk")
    st.warning(risk)

    st.subheader("🔥 Kombine Kupon Önerisi")
    for c in combo:
        st.write(f"{c[0]} → %{round(c[1]*100,2)}")
