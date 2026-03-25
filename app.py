import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import os
import pickle

st.set_page_config(page_title="Bet AI Pro", layout="centered")

# ----------------------------
# LOGIN (BASİT PANEL)
# ----------------------------
USERS = {"admin":"1234", "user":"1111"}

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Giriş")

    u = st.text_input("Kullanıcı")
    p = st.text_input("Şifre", type="password")

    if st.button("Giriş"):
        if u in USERS and USERS[u] == p:
            st.session_state.login = True
            st.success("Giriş başarılı")
        else:
            st.error("Hatalı giriş")

    st.stop()

# ----------------------------
# VERİ YÜKLE
# ----------------------------
DATA_FILE = "data.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        leagues = ["Super Lig","Premier League"]

        teams = {
            "Super Lig": ["Galatasaray","Fenerbahce","Besiktas","Trabzonspor"],
            "Premier League": ["Man City","Liverpool","Arsenal","Chelsea"]
        }

        rows = []
        for league in leagues:
            for _ in range(500):
                home = np.random.choice(teams[league])
                away = np.random.choice([t for t in teams[league] if t != home])

                rows.append([
                    league, home, away,
                    np.random.poisson(1.5),
                    np.random.poisson(1.2)
                ])

        df = pd.DataFrame(rows, columns=[
            "league","home_team","away_team","home_goals","away_goals"
        ])
        df.to_csv(DATA_FILE, index=False)
        return df

df = load_data()

# ----------------------------
# MODEL YÜKLE / TRAIN
# ----------------------------
MODEL_FILE = "model.pkl"

def train_model(df):
    attack = {}
    defense = {}

    teams = pd.concat([df['home_team'], df['away_team']]).unique()

    for t in teams:
        h = df[df["home_team"] == t]
        a = df[df["away_team"] == t]

        attack[t] = (h["home_goals"].mean() + a["away_goals"].mean()) / 2
        defense[t] = (h["away_goals"].mean() + a["home_goals"].mean()) / 2

    return attack, defense

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        attack, defense = pickle.load(f)
else:
    attack, defense = train_model(df)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((attack, defense), f)

# ----------------------------
# GÜNLÜK MAÇ (MANUEL + AUTO)
# ----------------------------
st.title("🔥 Bet AI Pro")

st.subheader("📅 Günlük Maç Ekle")

league = st.selectbox("Lig", df["league"].unique())
teams = pd.concat([df['home_team'], df['away_team']]).unique()

home = st.selectbox("Ev Sahibi", teams)
away = st.selectbox("Deplasman", teams)

if st.button("Maçı Kaydet"):
    new = pd.DataFrame([[league,home,away,0,0]],
        columns=["league","home_team","away_team","home_goals","away_goals"])
    df = pd.concat([df,new])
    df.to_csv(DATA_FILE, index=False)
    st.success("Maç eklendi")

# ----------------------------
# TAHMİN
# ----------------------------
def predict(home, away):

    home_l = attack.get(home,1.2) * defense.get(away,1.2)
    away_l = attack.get(away,1.2) * defense.get(home,1.2)

    probs = []
    for i in range(6):
        for j in range(6):
            p = poisson.pmf(i, home_l) * poisson.pmf(j, away_l)
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

    return sorted(probs, key=lambda x: x[2], reverse=True)[:3], markets

# ----------------------------
# ANALİZ
# ----------------------------
st.subheader("📊 Maç Analizi")

if st.button("Tahmin Et"):

    scores, markets = predict(home, away)

    st.write("### Skor")
    for h,a,p in scores:
        st.write(f"{h}-{a} → %{round(p*100,1)}")

    st.write("### Marketler")

    sorted_m = sorted(markets.items(), key=lambda x: x[1], reverse=True)

    for k,v in sorted_m:
        if v > 0.7:
            st.success(f"{k} → %{round(v*100,1)}")
        else:
            st.write(f"{k} → %{round(v*100,1)}")

    st.write("### 🎯 Kupon")

    best2 = sorted_m[:2]
    for b in best2:
        st.write(b[0])

# ----------------------------
# MODEL GÜNCELLEME
# ----------------------------
st.subheader("🧠 Sonuç Gir (AI öğrenir)")

hg = st.number_input("Ev Gol", 0, 10)
ag = st.number_input("Dep Gol", 0, 10)

if st.button("Sonucu Kaydet & Model Güncelle"):

    new = pd.DataFrame([[league,home,away,hg,ag]],
        columns=["league","home_team","away_team","home_goals","away_goals"])

    df = pd.concat([df,new])
    df.to_csv(DATA_FILE, index=False)

    attack, defense = train_model(df)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((attack, defense), f)

    st.success("Model güncellendi 🚀")
