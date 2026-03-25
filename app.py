import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson

# ============ CONFIG ============
st.set_page_config(page_title="PRO SCOUT ULTIMATE", layout="wide")

# ============ DATA ============
@st.cache_data
def load_data():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    df = pd.read_csv(url)
    df = df[['HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
    return df

# ============ MODEL ============
@st.cache_resource
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

    m1 = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=300)
    m2 = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=300)

    m1.fit(X, y1)
    m2.fit(X, y2)

    return m1, m2, le

# ============ ENGINE ============
def simulate(gh, ga):
    matrix = np.outer(
        [poisson.pmf(i, gh) for i in range(6)],
        [poisson.pmf(j, ga) for j in range(6)]
    )

    hw = np.sum(np.tril(matrix, -1))*100
    dr = np.sum(np.diag(matrix))*100
    aw = np.sum(np.triu(matrix, 1))*100

    o25 = sum(matrix[i,j] for i in range(6) for j in range(6) if i+j>2.5)*100
    btts = (1 - poisson.pmf(0, gh))*(1 - poisson.pmf(0, ga))*100

    return hw, dr, aw, o25, btts

def predict(home, away, m1, m2, le):
    try:
        h = le.transform([home])[0]
        a = le.transform([away])[0]
    except:
        return None

    gh = max(0.2, m1.predict([[h,a]])[0])
    ga = max(0.2, m2.predict([[h,a]])[0])

    return simulate(gh, ga), gh, ga

# ============ VALUE ============
def value(prob, odd):
    return (prob/100)*odd

# ============ AI (SAFE) ============
def ai_comment(home, away, hw, dr, aw):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not os.getenv("OPENAI_API_KEY"):
            return "API key yok"

        prompt = f"{home}-{away} için kısa bahis yorumu. Ev:{hw:.1f} X:{dr:.1f} Dep:{aw:.1f}"

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )

        return res.choices[0].message.content

    except Exception as e:
        return f"AI hata: {e}"

# ============ UI ============
st.title("💎 PRO SCOUT ULTIMATE (STABLE)")

df = load_data()
m1, m2, le = train(df)

teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())

col1, col2 = st.columns(2)
home = col1.selectbox("Ev Sahibi", teams)
away = col2.selectbox("Deplasman", teams)

st.subheader("💰 ORAN GİR")
c1, c2, c3 = st.columns(3)
o1 = c1.number_input("MS1", value=2.0)
ox = c2.number_input("X", value=3.0)
o2 = c3.number_input("MS2", value=3.0)

if st.button("ANALİZ ET"):

    if home == away:
        st.error("Aynı takım seçilemez")
    else:
        res = predict(home, away, m1, m2, le)

        if not res:
            st.error("Takım bulunamadı")
        else:
            (hw, dr, aw, o25, btts), gh, ga = res

            st.subheader("📊 SONUÇ")
            st.write(f"MS1: %{hw:.1f} | X: %{dr:.1f} | MS2: %{aw:.1f}")
            st.write(f"ÜST2.5: %{o25:.1f} | KG: %{btts:.1f}")
            st.success(f"Skor: {round(gh)}-{round(ga)}")

            # VALUE
            st.subheader("💰 VALUE")
            v1 = value(hw, o1)
            vx = value(dr, ox)
            v2 = value(aw, o2)

            st.write(f"MS1: {v1:.2f} | X: {vx:.2f} | MS2: {v2:.2f}")

            best = max([("MS1",v1),("X",vx),("MS2",v2)], key=lambda x:x[1])

            if best[1] > 1.05:
                st.success(f"🔥 VALUE: {best[0]}")
            else:
                st.warning("Value yok")

            # AI
            st.subheader("🤖 AI YORUM")
            st.info(ai_comment(home, away, hw, dr, aw))
