import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson
from openai import OpenAI

# ============ CONFIG ============
st.set_page_config(page_title="PRO SCOUT ELITE", layout="wide")

# 🔐 API KEY (ENV)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ============ DATA ============
@st.cache_data
def load_data():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    return pd.read_csv(url)

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

    m1 = MLPRegressor(max_iter=300).fit(X, y1)
    m2 = MLPRegressor(max_iter=300).fit(X, y2)

    return m1, m2, le

# ============ PREDICT ============
def predict(home, away, m1, m2, le):
    h = le.transform([home])[0]
    a = le.transform([away])[0]

    gh = max(0.2, m1.predict([[h,a]])[0])
    ga = max(0.2, m2.predict([[h,a]])[0])

    hw = (gh / (gh+ga)) * 100
    aw = (ga / (gh+ga)) * 100
    dr = 100 - hw - aw

    o25 = min(100, (gh+ga)*20)
    btts = min(100, gh*ga*25)

    return hw, dr, aw, o25, btts, gh, ga

# ============ AI ============
def ai_comment(home, away, hw, dr, aw, o25, btts):
    if not OPENAI_API_KEY:
        return "API key yok"

    prompt = f"""
    Sen profesyonel bahis analistisin.

    Maç: {home} vs {away}
    Ev: %{hw:.1f} | Ber: %{dr:.1f} | Dep: %{aw:.1f}
    Üst: %{o25:.1f} | KG: %{btts:.1f}

    1. En olası senaryo
    2. Risk
    3. En iyi bahis

    Kısa yaz.
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Hata: {e}"

# ============ UI ============
st.title("💎 PRO SCOUT ELITE + AI")

df = load_data()
m1, m2, le = train(df)

teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())

col1, col2 = st.columns(2)
home = col1.selectbox("Ev Sahibi", teams)
away = col2.selectbox("Deplasman", teams)

if st.button("ANALİZ ET"):
    hw, dr, aw, o25, btts, gh, ga = predict(home, away, m1, m2, le)

    st.write(f"🏠 %{hw:.1f} | 🤝 %{dr:.1f} | 🚀 %{aw:.1f}")
    st.write(f"🔥 ÜST: %{o25:.1f} | ⚽ KG: %{btts:.1f}")
    st.success(f"Skor: {round(gh)}-{round(ga)}")

    st.subheader("🤖 AI Yorum")
    st.info(ai_comment(home, away, hw, dr, aw, o25, btts))
