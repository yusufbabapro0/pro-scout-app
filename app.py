!pip install streamlit xgboost pyngrok --quiet

import streamlit as st
import pandas as pd
import numpy as np
import inspect
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from scipy.stats import poisson
import matplotlib.pyplot as plt
from pyngrok import ngrok

# =========================
# STREAMLIT APP
# =========================
def run_app():

    st.set_page_config(page_title="PRO-SCOUT ELITE", layout="wide")

    st.markdown("""
    <style>
    body {background-color:#020617;}
    .title {text-align:center;font-size:42px;color:white;font-weight:900;}
    .card {
        background: linear-gradient(135deg,#1e293b,#020617);
        padding:20px;border-radius:16px;
        text-align:center;color:white;
        box-shadow:0 0 20px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>⚽ PRO-SCOUT v70 ELITE</div>", unsafe_allow_html=True)

    LEAGUES = {
        'TR Süper Lig': 'T1','ENG Premier': 'E0',
        'ESP La Liga': 'SP1','GER Bundesliga': 'D1',
        'ITA Serie A': 'I1','FRA Ligue 1': 'F1'
    }

    # =========================
    # DATA LOAD
    # =========================
    @st.cache_data
    def load_data(code):
        df = pd.DataFrame()

        for season in ["2324","2425","2526"]:
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
                tmp = pd.read_csv(url)
                df = pd.concat([df, tmp[['HomeTeam','AwayTeam','FTHG','FTAG']]])
            except:
                continue

        df.dropna(inplace=True)

        le = LabelEncoder()
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        le.fit(teams)

        df['h'] = le.transform(df['HomeTeam'])
        df['a'] = le.transform(df['AwayTeam'])

        X = df[['h','a']]

        model_h = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9
        )

        model_a = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9
        )

        model_h.fit(X, df['FTHG'])
        model_a.fit(X, df['FTAG'])

        return df, model_h, model_a, le, teams

    # =========================
    # TEAM FORM
    # =========================
    def get_team_stats(df, team, is_home=True):
        if is_home:
            sub = df[df['HomeTeam']==team].tail(5)
            att = sub['FTHG'].mean()
            deff = sub['FTAG'].mean()
        else:
            sub = df[df['AwayTeam']==team].tail(5)
            att = sub['FTAG'].mean()
            deff = sub['FTHG'].mean()

        return att if not np.isnan(att) else 1.2, deff if not np.isnan(deff) else 1.2

    # =========================
    # PREDICT
    # =========================
    def predict(home, away, df, mh, ma, le):

        h = le.transform([home])[0]
        a = le.transform([away])[0]

        h_att, h_def = get_team_stats(df, home, True)
        a_att, a_def = get_team_stats(df, away, False)

        X_pred = [[h, a]]

        gh = max(mh.predict(X_pred)[0] * (h_att / max(a_def,0.5)), 0.2)
        ga = max(ma.predict(X_pred)[0] * (a_att / max(h_def,0.5)), 0.2)

        max_goals = 8

        probs = np.outer(
            [poisson.pmf(i, gh) for i in range(max_goals)],
            [poisson.pmf(j, ga) for j in range(max_goals)]
        )

        hw = np.sum(np.tril(probs,-1))*100
        dr = np.sum(np.diag(probs))*100
        aw = np.sum(np.triu(probs,1))*100

        # Over/Under
        over25 = 100 * (1 - sum(
            poisson.pmf(i, gh)*poisson.pmf(j, ga)
            for i in range(3) for j in range(3) if i+j<=2
        ))

        return gh, ga, hw, dr, aw, over25

    # =========================
    # VALUE BET
    # =========================
    def is_value(hw, dr, aw):
        return max(hw,dr,aw) > 70

    # =========================
    # POWER SCORE
    # =========================
    def power(att, deff):
        return (att*1.5) - deff

    # =========================
    # UI
    # =========================
    league = st.selectbox("Lig", list(LEAGUES.keys()))
    data = load_data(LEAGUES[league])

    if data:
        df, mh, ma, le, teams = data

        tab1, tab2, tab3 = st.tabs(["Tahmin","Radar","Banko AI"])

        # =========================
        # TAHMİN
        # =========================
        with tab1:
            c1,c2 = st.columns(2)
            home = c1.selectbox("Ev Sahibi", teams)
            away = c2.selectbox("Deplasman", teams)

            if st.button("Tahmin Et"):

                gh, ga, hw, dr, aw, over25 = predict(home, away, df, mh, ma, le)

                m1,m2,m3 = st.columns(3)
                m1.markdown(f"<div class='card'>EV %{hw:.1f}</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='card'>BER %{dr:.1f}</div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='card'>DEP %{aw:.1f}</div>", unsafe_allow_html=True)

                st.success(f"Skor Tahmini: {round(gh)} - {round(ga)}")
                st.info(f"Over 2.5: %{over25:.1f}")

        # =========================
        # RADAR
        # =========================
        with tab2:
            team = st.selectbox("Takım", teams)

            att, deff = get_team_stats(df, team, True)
            p = power(att, deff)

            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)

            vals = [att, deff, p]
            angles = np.linspace(0,2*np.pi,len(vals),endpoint=False)

            vals = np.append(vals, vals[0])
            angles = np.append(angles, angles[0])

            ax.plot(angles, vals)
            ax.fill(angles, vals, alpha=0.2)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(['Hücum','Savunma','Power'])

            st.pyplot(fig)

        # =========================
        # BANKO AI
        # =========================
        with tab3:
            if st.button("Banko Tara"):

                for i in range(len(teams)-1):

                    gh, ga, hw, dr, aw, _ = predict(
                        teams[i], teams[i+1], df, mh, ma, le
                    )

                    if is_value(hw, dr, aw):
                        st.write(f"🔥 {teams[i]} vs {teams[i+1]} → %{max(hw,aw):.1f}")

# =========================
# RUN STREAMLIT
# =========================
from threading import Thread

def start():
    import os
    os.system("streamlit run app.py")

with open("app.py","w") as f:
    f.write(inspect.getsource(run_app)+"\n\nrun_app()")

url = ngrok.connect(8501)
print("LINK:", url)

Thread(target=start).start()
