import streamlit as st
import pandas as pd
import joblib

# Load the saved model and label encoder
model = joblib.load('weather_model.pkl')
le = joblib.load('location_encoder.pkl')

st.title("🌦️ Weather Predictor Dashboard")

st.sidebar.header("Input Weather Features")

location = st.sidebar.selectbox("Select Location:", list(le.classes_))
location_encoded = le.transform([location])[0]

humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 100.0, 0.0)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2024)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=7)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)

input_df = pd.DataFrame([[location_encoded, humidity, precipitation, year, month, day]],
                        columns=["Location", "Humidity_pct", "Precipitation_mm", "year", "month", "day"])

if st.button("Predict Weather"):
    prediction = model.predict(input_df)
    temp, wind = prediction[0]
    st.success(f"🌡️ Predicted Temperature: {temp:.2f} °C")
    st.success(f"💨 Predicted Wind Speed: {wind:.2f} km/h")
