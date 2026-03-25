import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import poisson

st.set_page_config(page_title="Match Predictor", layout="wide")

# ================= DATA =================
@st.cache_data
def load_data():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    df = pd.read_csv(url)
    df = df[['HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
    return df

# ================= MODEL =================
@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    le.fit(teams)

    X = []
    y_home = []
    y_away = []

    for _, row in df.iterrows():
        X.append([
            le.transform([row['HomeTeam']])[0],
            le.transform([row['AwayTeam']])[0]
        ])
        y_home.append(row['FTHG'])
        y_away.append(row['FTAG'])

    model_home = LinearRegression()
    model_away = LinearRegression()

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    return model_home, model_away, le

# ================= PREDICTION =================
def predict_match(home, away, model_home, model_away, le):
    try:
        h = le.transform([home])[0]
        a = le.transform([away])[0]
    except:
        return None

    gh = max(0.2, model_home.predict([[h, a]])[0])
    ga = max(0.2, model_away.predict([[h, a]])[0])

    # Poisson matrix
    matrix = np.outer(
        [poisson.pmf(i, gh) for i in range(6)],
        [poisson.pmf(j, ga) for j in range(6)]
    )

    home_win = np.sum(np.tril(matrix, -1)) * 100
    draw = np.sum(np.diag(matrix)) * 100
    away_win = np.sum(np.triu(matrix, 1)) * 100

    over25 = sum(matrix[i, j] for i in range(6) for j in range(6) if i + j > 2.5) * 100
    btts = (1 - poisson.pmf(0, gh)) * (1 - poisson.pmf(0, ga)) * 100

    return home_win, draw, away_win, over25, btts, gh, ga

# ================= UI =================
st.title("⚽ Match Prediction App (Stable)")

df = load_data()
model_home, model_away, le = train_model(df)

teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home Team", teams)
away_team = col2.selectbox("Away Team", teams)

if st.button("Predict"):

    if home_team == away_team:
        st.error("Same teams selected!")
    else:
        result = predict_match(home_team, away_team, model_home, model_away, le)

        if result is None:
            st.error("Prediction failed!")
        else:
            hw, dr, aw, o25, btts, gh, ga = result

            st.subheader("📊 Match Probabilities")
            st.write(f"Home Win: %{hw:.1f}")
            st.write(f"Draw: %{dr:.1f}")
            st.write(f"Away Win: %{aw:.1f}")

            st.subheader("⚽ Goals")
            st.write(f"Over 2.5: %{o25:.1f}")
            st.write(f"BTTS: %{btts:.1f}")

            st.subheader("🎯 Predicted Score")
            st.success(f"{round(gh)} - {round(ga)}")
