import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Legend Bet AI", layout="centered")

# ----------------------------
# DEMO / CSV
# ----------------------------
uploaded_file = st.file_uploader("CSV yükle", type=["csv"])

def load_data():
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        leagues = ["Super Lig","Premier League","La Liga"]

        teams = {
            "Super Lig": ["Galatasaray","Fenerbahce","Besiktas","Trabzonspor"],
            "Premier League": ["Man City","Liverpool","Arsenal","Chelsea"],
            "La Liga": ["Real Madrid","Barcelona","Atletico","Sevilla"]
        }

        np.random.seed(42)
        rows = []

        for league in leagues:
            for _ in range(500):
                home = np.random.choice(teams[league])
                away = np.random.choice([t for t in teams[league] if t != home])

                rows.append([
                    league,
                    home,
                    away,
                    np.random.poisson(1.6),
                    np.random.poisson(1.2)
                ])

        return pd.DataFrame(rows, columns=[
            "league","home_team","away_team","home_goals","away_goals"
        ])

df = load_data()

# ----------------------------
# LİG
# ----------------------------
league = st.selectbox("Lig", df["league"].unique())
df = df[df["league"] == league]

teams = pd.concat([df['home_team'], df['away_team']]).unique()

# ----------------------------
# FORM
# ----------------------------
def get_form(team):
    matches = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(5)

    pts = 0
    for _, r in matches.iterrows():
        if r["home_team"] == team:
            if r["home_goals"] > r["away_goals"]: pts += 3
            elif r["home_goals"] == r["away_goals"]: pts += 1
        else:
            if r["away_goals"] > r["home_goals"]: pts += 3
            elif r["away_goals"] == r["home_goals"]: pts += 1

    return pts / 15

# ----------------------------
# GÜÇ
# ----------------------------
def strengths(df):
    attack = {}
    defense = {}

    for t in teams:
        h = df[df["home_team"] == t]
        a = df[df["away_team"] == t]

        attack[t] = (h["home_goals"].mean() + a["away_goals"].mean()) / 2
        defense[t] = (h["away_goals"].mean() + a["home_goals"].mean()) / 2

    return attack, defense

attack, defense = strengths(df)

# ----------------------------
# VALUE BET
# ----------------------------
def value(prob, odds):
    return prob * odds - 1

# ----------------------------
# TAHMİN
# ----------------------------
def predict(home, away, odds):

    home_l = attack[home] * defense[away] * 1.2
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

    over25 = sum(p for i,j,p in probs if i+j>=3)
    btts = sum(p for i,j,p in probs if i>0 and j>0)

    markets = {
        "Ev Sahibi": home_win,
        "Beraberlik": draw,
        "Deplasman": away_win,
        "2.5 Üst": over25,
        "KG Var": btts
    }

    # VALUE hesap
    values = {}
    for k in markets:
        values[k] = value(markets[k], odds[k])

    # EN İYİLER
    best = sorted(markets.items(), key=lambda x: x[1], reverse=True)
    value_best = sorted(values.items(), key=lambda x: x[1], reverse=True)

    return probs[:3], best, value_best

# ----------------------------
# UI
# ----------------------------
st.title("🔥 LEGEND Betting AI")

home = st.selectbox("Ev Sahibi", teams)
away = st.selectbox("Deplasman", teams)

st.subheader("💰 Oran Gir")

odds = {
    "Ev Sahibi": st.number_input("Ev Sahibi", 1.0, 10.0, 2.0),
    "Beraberlik": st.number_input("Beraberlik", 1.0, 10.0, 3.2),
    "Deplasman": st.number_input("Deplasman", 1.0, 10.0, 3.5),
    "2.5 Üst": st.number_input("2.5 Üst", 1.0, 10.0, 1.8),
    "KG Var": st.number_input("KG Var", 1.0, 10.0, 1.7)
}

if st.button("Analiz"):

    scores, best, value_best = predict(home, away, odds)

    st.subheader("📊 Skor")
    for h,a,p in scores:
        st.write(f"{h}-{a} → %{round(p*100,1)}")

    st.subheader("📈 En Güvenli")
    for b in best[:3]:
        st.write(f"{b[0]} → %{round(b[1]*100,1)}")

    st.subheader("💎 Value Bet (PARA BURADA)")
    for v in value_best[:3]:
        st.success(f"{v[0]} → Değer: {round(v[1],2)}")

    st.subheader("🔥 Kupon Önerisi")
    combo = [best[0][0], value_best[0][0]]
    for c in combo:
        st.write(c)
