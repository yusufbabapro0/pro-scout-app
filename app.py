import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson

st.set_page_config(page_title="PRO SCOUT vNext", layout="wide")

# --- DATA LOADER ---
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

# --- FEATURE ENGINEERING ---
def create_features(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    form = {}

    for t in teams:
        last = df[(df['HomeTeam']==t)|(df['AwayTeam']==t)].tail(5)

        scored = last[last['HomeTeam']==t]['FTHG'].sum() + last[last['AwayTeam']==t]['FTAG'].sum()
        conceded = last[last['HomeTeam']==t]['FTAG'].sum() + last[last['AwayTeam']==t]['FTHG'].sum()

        form[t] = {
            "attack": scored/5,
            "defense": conceded/5
        }

    return form

# --- MODEL ---
@st.cache_data
def train_model(df):
    le = LabelEncoder()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    le.fit(teams)

    X = []
    y_home = []
    y_away = []

    for _, r in df.iterrows():
        X.append([
            le.transform([r['HomeTeam']])[0],
            le.transform([r['AwayTeam']])[0]
        ])
        y_home.append(r['FTHG'])
        y_away.append(r['FTAG'])

    model_home = GradientBoostingRegressor().fit(X, y_home)
    model_away = GradientBoostingRegressor().fit(X, y_away)

    return model_home, model_away, le

# --- ENGINE ---
def predict_match(home, away, model_home, model_away, le, form):
    h_id = le.transform([home])[0]
    a_id = le.transform([away])[0]

    base_home = model_home.predict([[h_id, a_id]])[0]
    base_away = model_away.predict([[h_id, a_id]])[0]

    # form adjust
    base_home *= (1 + form[home]['attack'] - form[away]['defense'])
    base_away *= (1 + form[away]['attack'] - form[home]['defense'])

    base_home = max(base_home, 0.2)
    base_away = max(base_away, 0.2)

    matrix = np.outer(
        [poisson.pmf(i, base_home) for i in range(6)],
        [poisson.pmf(j, base_away) for j in range(6)]
    )

    home_win = np.sum(np.tril(matrix, -1))*100
    draw = np.sum(np.diag(matrix))*100
    away_win = np.sum(np.triu(matrix, 1))*100

    over25 = sum(matrix[i,j] for i in range(6) for j in range(6) if i+j>2.5)*100
    btts = (1 - poisson.pmf(0, base_home)) * (1 - poisson.pmf(0, base_away)) * 100

    return home_win, draw, away_win, over25, btts, base_home, base_away

# --- UI ---
st.title("💎 PRO SCOUT vNext")

LIGLER = {
    "Türkiye Süper Lig": "T1",
    "Premier League": "E0",
    "La Liga": "SP1"
}

lig = st.selectbox("Lig seç", list(LIGLER.keys()))

df = load_data(LIGLER[lig])

if df.empty:
    st.error("Veri çekilemedi")
else:
    model_home, model_away, le = train_model(df)
    form = create_features(df)

    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())

    col1, col2 = st.columns(2)
    home = col1.selectbox("Ev Sahibi", teams)
    away = col2.selectbox("Deplasman", teams)

    if st.button("Analiz Et"):
        hw, dr, aw, o25, btts, gh, ga = predict_match(home, away, model_home, model_away, le, form)

        st.subheader(f"{home} vs {away}")

        st.write(f"🏠 Ev Kazanır: %{hw:.1f}")
        st.write(f"🤝 Beraberlik: %{dr:.1f}")
        st.write(f"🚀 Deplasman: %{aw:.1f}")

        st.write("---")

        st.write(f"🔥 2.5 ÜST: %{o25:.1f}")
        st.write(f"⚽ KG VAR: %{btts:.1f}")

        st.success(f"Tahmini skor: {round(gh)} - {round(ga)}")

        # AI yorum
        if hw > 65:
            st.info("💡 Ana bahis: MS1")
        elif o25 > 65:
            st.info("💡 Ana bahis: 2.5 ÜST")
        elif btts > 60:
            st.info("💡 Ana bahis: KG VAR")
        else:
            st.warning("Net bahis yok")
