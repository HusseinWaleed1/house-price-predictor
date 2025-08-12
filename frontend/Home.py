import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/house_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page config
st.set_page_config(page_title="🏠 California House Price Prediction", page_icon="🏠", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        h1 {
            color: #2E86C1;
            text-align: center;
            font-size: 38px !important;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏠 California House Price Prediction")
st.write("Fill in the details below to estimate the house price in California.")

# Form layout
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Median Income", min_value=0.0, step=0.1)
    HouseAge = st.number_input("House Age", min_value=0.0, step=1.0)
    AveRooms = st.number_input("Average Rooms", min_value=0.0, step=0.1)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, step=0.1)

with col2:
    Population = st.number_input("Population", min_value=0.0, step=1.0)
    AveOccup = st.number_input("Average Occupancy", min_value=0.0, step=0.1)
    Latitude = st.number_input("Latitude", step=0.01)
    Longitude = st.number_input("Longitude", step=0.01)

# Predict button
if st.button("🔍 Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    price = prediction[0] * 100000

    st.success(f"💰 Estimated House Price: ${price:,.2f}")
