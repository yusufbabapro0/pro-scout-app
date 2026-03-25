import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson
from openai import OpenAI

# ================= CONFIG =================
st.set_page_config(page_title="PRO SCOUT ULTIMATE", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ================= DATA =================
@st.cache_data
def load_data():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    return pd.read_csv(url)

# ================= MODEL =================
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

    m1 = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=400).fit(X, y1)
    m2 = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=400).fit(X, y2)

    return m1, m2, le

# ================= ENGINE =================
def simulate_match(gh, ga):
    matrix = np.outer(
        [poisson.pmf(i, gh) for i in range(6)],
        [poisson.pmf(j, ga) for j in range(6)]
    )

    hw = np.sum(np.tril(matrix, -1))*100
    dr = np.sum(np.diag(matrix))*100
    aw = np.sum(np.triu(matrix, 1))*100

    o15 = sum(matrix[i,j] for i in range(6) for j in range(6) if i+j>1.5)*100
    o25 = sum(matrix[i,j] for i in range(6) for j in range(6) if i+j>2.5)*100
    o35 = sum(matrix[i,j] for i in range(6) for j in range(6) if i+j>3.5)*100

    btts = (1 - poisson.pmf(0, gh))*(1 - poisson.pmf(0, ga))*100

    exact_scores = {}
    for i in range(4):
        for j in range(4):
            exact_scores[f"{i}-{j}"] = matrix[i][j]*100

    return hw, dr, aw, o15, o25, o35, btts, exact_scores

def predict(home, away, m1, m2, le):
    h = le.transform([home])[0]
    a = le.transform([away])[0]

    gh = max(0.2, m1.predict([[h,a]])[0])
    ga = max(0.2, m2.predict([[h,a]])[0])

    return simulate_match(gh, ga), gh, ga

# ================= VALUE =================
def value(prob, odd):
    return (prob/100)*odd

# ================= AI =================
def ai_comment(home, away, stats):
    if not OPENAI_API_KEY:
        return "API key yok"

    hw, dr, aw, o15, o25, o35, btts, _ = stats

    prompt = f"""
    Profesyonel bahis analisti gibi yorum yap.

    {home} vs {away}
    MS1:{hw:.1f} X:{dr:.1f} MS2:{aw:.1f}
    ÜST2.5:{o25:.1f} KG:{btts:.1f}

    En iyi bahis + risk analizi + kısa yorum ver.
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return res.choices[0].message.content
    except:
        return "AI çalışmadı"

# ================= RISK =================
def risk_score(hw, dr, aw):
    mx = max(hw, dr, aw)
    if mx > 65:
        return "🟢 Düşük Risk"
    elif mx > 50:
        return "🟡 Orta Risk"
    else:
        return "🔴 Yüksek Risk"

# ================= UI =================
st.title("💎 PRO SCOUT ULTIMATE v4")

df = load_data()
m1, m2, le = train(df)

teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())

col1, col2 = st.columns(2)
home = col1.selectbox("Ev Sahibi", teams)
away = col2.selectbox("Deplasman", teams)

st.subheader("💰 ORAN GİR")
c1, c2, c3 = st.columns(3)
o1 = c1.number_input("MS1", value=2.0)
ox = c2.number_input("X", value=3.2)
o2 = c3.number_input("MS2", value=3.0)

if st.button("🔥 ANALİZ ET"):
    stats, gh, ga = predict(home, away, m1, m2, le)
    hw, dr, aw, o15, o25, o35, btts, scores = stats

    st.subheader("📊 MAÇ OLASILIKLARI")
    st.write(f"MS1: %{hw:.1f} | X: %{dr:.1f} | MS2: %{aw:.1f}")

    st.subheader("⚽ GOL MARKETLERİ")
    st.write(f"ÜST1.5: %{o15:.1f}")
    st.write(f"ÜST2.5: %{o25:.1f}")
    st.write(f"ÜST3.5: %{o35:.1f}")
    st.write(f"KG VAR: %{btts:.1f}")

    st.subheader("🎯 SKOR TAHMİNİ")
    top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for s in top_scores:
        st.write(f"{s[0]} → %{s[1]:.1f}")

    st.success(f"Tahmini Skor: {round(gh)}-{round(ga)}")

    # VALUE
    st.subheader("💰 VALUE ANALİZİ")
    v1 = value(hw, o1)
    vx = value(dr, ox)
    v2 = value(aw, o2)

    st.write(f"MS1 Value: {v1:.2f}")
    st.write(f"X Value: {vx:.2f}")
    st.write(f"MS2 Value: {v2:.2f}")

    best = max([("MS1", v1), ("X", vx), ("MS2", v2)], key=lambda x:x[1])

    if best[1] > 1.05:
        st.success(f"🔥 VALUE BET: {best[0]}")
    else:
        st.warning("Value yok")

    # RISK
    st.subheader("⚠️ RİSK ANALİZİ")
    st.write(risk_score(hw, dr, aw))

    # AI
    st.subheader("🤖 AI ANALİZ")
    st.info(ai_comment(home, away, stats))
