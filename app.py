import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check whether it is **Fraud or Normal**.")

features = []

# IMPORTANT: Order must match training
# Time
time = st.number_input("Time (seconds since first transaction)", value=0.0)
features.append(time)

# V1 to V28
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    features.append(val)

# Amount
amount = st.number_input("Transaction Amount", value=0.0)
features.append(amount)

st.write("Total features entered:", len(features))  # must be 30

# Prediction
if st.button("Check Transaction"):
    input_data = np.array(features).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected\n\nFraud Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction\n\nFraud Probability: {probability:.2f}")

