import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Maç Tahmin", layout="centered")

st.title("⚽ Futbol Maç Skor Tahmin Uygulaması")

# -----------------------------
# ÖRNEK DATA (gerçek projede CSV bağlayabilirsin)
# -----------------------------
data = {
    "Team": ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"],
    "Attack": [1.8, 1.7, 1.5, 1.4],
    "Defense": [1.2, 1.1, 1.3, 1.4]
}

df = pd.DataFrame(data)

# -----------------------------
# TAKIM SEÇİMİ
# -----------------------------
home_team = st.selectbox("Ev Sahibi", df["Team"])
away_team = st.selectbox("Deplasman", df["Team"])

# -----------------------------
# MODEL FONKSİYONU
# -----------------------------
def predict_score(home, away):
    home_attack = df[df["Team"] == home]["Attack"].values[0]
    home_defense = df[df["Team"] == home]["Defense"].values[0]

    away_attack = df[df["Team"] == away]["Attack"].values[0]
    away_defense = df[df["Team"] == away]["Defense"].values[0]

    home_lambda = home_attack * away_defense
    away_lambda = away_attack * home_defense

    max_goals = 5
    matrix = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i][j] = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)

    return matrix

# -----------------------------
# TAHMİN BUTONU
# -----------------------------
if st.button("Tahmin Et"):
    if home_team == away_team:
        st.warning("Aynı takımı seçemezsin!")
    else:
        matrix = predict_score(home_team, away_team)

        # en olası skor
        result = np.unravel_index(np.argmax(matrix), matrix.shape)
        home_goals, away_goals = result

        st.subheader("📊 Tahmin Sonucu")
        st.write(f"**{home_team} {home_goals} - {away_goals} {away_team}**")

        # kazanma ihtimalleri
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))

        st.subheader("📈 Olasılıklar")
        st.write(f"Ev Sahibi Kazanır: %{home_win*100:.1f}")
        st.write(f"Beraberlik: %{draw*100:.1f}")
        st.write(f"Deplasman Kazanır: %{away_win*100:.1f}")

        # skor matrisi göster
        st.subheader("🔢 Skor Olasılık Matrisi")
        st.dataframe(pd.DataFrame(matrix))
