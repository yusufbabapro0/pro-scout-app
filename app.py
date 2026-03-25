import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson

# ============ CONFIG ============
st.set_page_config(page_title="PRO SCOUT ELITE v3", layout="wide")

ODDS_API_KEY = "YOUR_ODDS_API_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# ============ DATA ============
@st.cache_data
def load_data(lig):
    df_all = pd.DataFrame()
    for s in ["2324", "2425", "2526"]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/{lig}.csv"
            df = pd.read_csv(url)
            df_all = pd.concat([df_all, df])
        except:
            continue
    return df_all

# ============ FEATURES (xG) ============
def create_xg(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    xg = {}

    for t in teams:
        last = df[(df['HomeTeam']==t)|(df['AwayTeam']==t)].tail(10)

        scored = last[last['HomeTeam']==t]['FTHG'].mean()
        conceded = last[last['HomeTeam']==t]['FTAG'].mean()

        xg[t] = {
            "attack": scored if not np.isnan(scored) else 1,
            "defense": conceded if not np.isnan(conceded) else 1
        }

    return xg

# ============ MODEL ============
@st.cache_data
def train(df):
    le = LabelEncoder()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    le.fit(teams)

    X, y1, y2 = [], [], []

    for _, r in df.iterrows():
        X.append([
            le.transform([r['HomeTeam']])[0],
            le.transform([r['AwayTeam']])[0]
        ])
        y1.append(r['FTHG'])
        y2.append(r['FTAG'])

    m1 = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500).fit(X, y1)
    m2 = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500).fit(X, y2)

    return m1, m2, le

# ============ ODDS API ============
def get_real_odds():
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
    res = requests.get(url)

    if res.status_code != 200:
        return []

    data = res.json()
    matches = []

    for g in data:
        try:
            home = g['home_team']
            away = g['away_team']
            odds = g['bookmakers'][0]['markets'][0]['outcomes']

            matches.append({
                "home": home,
                "away": away,
                "o1": odds[0]['price'],
                "o2": odds[1]['price'],
                "ox": odds[2]['price']
            })
        except:
            continue

    return matches

# ============ ENGINE ============
def predict(home, away, m1, m2, le, xg):
    try:
        h = le.transform([home])[0]
        a = le.transform([away])[0]
    except:
        return None

    gh = m1.predict([[h, a]])[0]
    ga = m2.predict([[h, a]])[0]

    gh *= xg.get(home, {"attack":1})["attack"]
    ga *= xg.get(away, {"attack":1})["attack"]

    gh, ga = max(0.2, gh), max(0.2, ga)

    mat = np.outer(
        [poisson.pmf(i, gh) for i in range(6)],
        [poisson.pmf(j, ga) for j in range(6)]
    )

    hw = np.sum(np.tril(mat, -1))*100
    dr = np.sum(np.diag(mat))*100
    aw = np.sum(np.triu(mat, 1))*100

    o25 = sum(mat[i,j] for i in range(6) for j in range(6) if i+j>2.5)*100
    btts = (1 - poisson.pmf(0, gh))*(1 - poisson.pmf(0, ga))*100

    return hw, dr, aw, o25, btts, gh, ga

# ============ VALUE ============
def value(prob, odd):
    return (prob/100)*odd

# ============ AI COMMENT ============
def ai_comment(home, away, hw, dr, aw, o25, btts):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"{home} vs {away}. Ev:{hw} Ber:{dr} Dep:{aw} Üst:{o25} KG:{btts}. Kısa analiz."

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )

        return res.choices[0].message.content
    except:
        return "AI yorum alınamadı."

# ============ VALUE BOT ============
def find_value_bets(matches, m1, m2, le, xg):
    picks = []

    for m in matches:
        pred = predict(m['home'], m['away'], m1, m2, le, xg)
        if not pred:
            continue

        hw, dr, aw, _, _, _, _ = pred

        v1 = value(hw, m['o1'])
        vx = value(dr, m['ox'])
        v2 = value(aw, m['o2'])

        best = max([("MS1", v1), ("X", vx), ("MS2", v2)], key=lambda x: x[1])

        if best[1] > 1.05:
            picks.append({
                "match": f"{m['home']} - {m['away']}",
                "bet": best[0],
                "value": round(best[1],2)
            })

    return picks

# ============ COUPON ============
def build_coupon(picks):
    picks = sorted(picks, key=lambda x: x['value'], reverse=True)
    return picks[:4]

# ============ UI ============
st.title("💎 PRO SCOUT ELITE v3")

LIGLER = {
    "Türkiye": "T1",
    "Premier": "E0",
    "La Liga": "SP1"
}

lig = st.selectbox("Lig seç", list(LIGLER.keys()))
df = load_data(LIGLER[lig])

if df.empty:
    st.error("Veri yok")
else:
    m1, m2, le = train(df)
    xg = create_xg(df)

    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())

    st.subheader("📊 Maç Analizi")
    c1, c2 = st.columns(2)
    home = c1.selectbox("Ev Sahibi", teams)
    away = c2.selectbox("Deplasman", teams)

    if st.button("ANALİZ ET"):
        res = predict(home, away, m1, m2, le, xg)

        if res:
            hw, dr, aw, o25, btts, gh, ga = res

            st.write(f"🏠 %{hw:.1f} | 🤝 %{dr:.1f} | 🚀 %{aw:.1f}")
            st.write(f"🔥 2.5 ÜST: %{o25:.1f} | ⚽ KG: %{btts:.1f}")
            st.success(f"Skor: {round(gh)}-{round(ga)}")

            st.subheader("🤖 AI Yorum")
            st.info(ai_comment(home, away, hw, dr, aw, o25, btts))

    # API Odds
    st.subheader("📡 Canlı Oran + Value")
    if st.button("ORANLARI ÇEK"):
        matches = get_real_odds()
        picks = find_value_bets(matches, m1, m2, le, xg)
        coupon = build_coupon(picks)

        st.write("🔥 VALUE MAÇLAR")
        for p in picks:
            st.write(p)

        st.write("🧾 KUPON")
        for c in coupon:
            st.success(c)
