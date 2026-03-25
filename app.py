import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Learning AI", layout="centered")

API_KEY = "865a20d4f77b4d92a52002d071ccfa04"

headers = {
    "x-apisports-key": API_KEY
}

DATA_FILE = "training_data.csv"
MODEL_FILE = "ml_model.pkl"

# ----------------------------
# MAÇLARI ÇEK
# ----------------------------
@st.cache_data(ttl=1800)
def get_matches():
    url = "https://v3.football.api-sports.io/fixtures?next=20"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        return None

    data = res.json()["response"]

    matches = []

    for m in data:
        matches.append({
            "home": m["teams"]["home"]["name"],
            "away": m["teams"]["away"]["name"]
        })

    return pd.DataFrame(matches)

# ----------------------------
# FEATURE (ÖZELLİK)
# ----------------------------
def team_power(team):
    np.random.seed(abs(hash(team)) % 1000)
    return np.random.uniform(0.8, 1.8)

def create_features(home, away):
    return [
        team_power(home),
        team_power(away),
        team_power(home) - team_power(away)
    ]

# ----------------------------
# MODEL YÜKLE / TRAIN
# ----------------------------
def load_model():

    if os.path.exists(MODEL_FILE):
        return pickle.load(open(MODEL_FILE, "rb"))

    else:
        model = RandomForestClassifier(n_estimators=100)
        return model

def train_model():

    if not os.path.exists(DATA_FILE):
        return None

    df = pd.read_csv(DATA_FILE)

    X = df[["f1","f2","f3"]]
    y = df["result"]

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    pickle.dump(model, open(MODEL_FILE, "wb"))

    return model

model = load_model()

# ----------------------------
# TAHMİN
# ----------------------------
def predict(home, away):

    X = np.array(create_features(home, away)).reshape(1,-1)

    probs = model.predict_proba(X)[0]

    return {
        "MS1": probs[0],
        "MSX": probs[1],
        "MS2": probs[2]
    }

# ----------------------------
# UI
# ----------------------------
st.title("🧠 Learning Betting AI")

df = get_matches()

if df is None:
    st.error("API veri çekemedi")
else:

    match_names = df.apply(lambda x: f"{x['home']} vs {x['away']}", axis=1)
    selected = st.selectbox("Maç seç", match_names)

    row = df.iloc[match_names.tolist().index(selected)]

    if st.button("Tahmin Et"):

        preds = predict(row["home"], row["away"])

        for k,v in preds.items():
            st.write(f"{k} → %{round(v*100,1)}")

# ----------------------------
# SONUÇ GİR (ÖĞRENME)
# ----------------------------
st.subheader("📊 Sonuç Gir (AI Öğrenir)")

result = st.selectbox("Sonuç", ["MS1","MSX","MS2"])

if st.button("Kaydet & Öğret"):

    features = create_features(row["home"], row["away"])

    new = pd.DataFrame([{
        "f1": features[0],
        "f2": features[1],
        "f3": features[2],
        "result": result
    }])

    if os.path.exists(DATA_FILE):
        old = pd.read_csv(DATA_FILE)
        new = pd.concat([old, new])

    new.to_csv(DATA_FILE, index=False)

    train_model()

    st.success("AI kendini geliştirdi 🚀")
