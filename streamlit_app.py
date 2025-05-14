
import streamlit as st
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("polynomial_temperature_model.pkl")
scaler = joblib.load("temperature_scaler.pkl")

st.title("🌡️ Temperature Forecast in Đà Nẵng")

mode = st.radio("Choose a mode:", ["🔢 Manual Prediction", "📈 7-Day Forecast"])

if mode == "🔢 Manual Prediction":
    st.header("Manual Temperature Prediction")
    humidity = st.slider("Relative Humidity (%)", 0, 100, 85)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 50.0, 10.0)
    dayofyear = st.slider("Day of Year", 1, 366, datetime.datetime.now().timetuple().tm_yday)

    X_input = np.array([[humidity, wind_speed, dayofyear]])
    X_scaled = scaler.transform(X_input)
    predicted_temp = model.predict(X_scaled)[0]

    st.success(f"🌤️ Predicted Temperature: {predicted_temp:.2f} °C")

elif mode == "📈 7-Day Forecast":
    st.header("Forecast Next 7 Days")
    start_day = st.slider("Start Day of Year", 1, 359, datetime.datetime.now().timetuple().tm_yday)
    humidity = st.slider("Assumed Humidity (%)", 0, 100, 80)
    wind_speed = st.slider("Assumed Wind Speed (m/s)", 0.0, 50.0, 12.0)

    days = np.array([start_day + i for i in range(7)])
    inputs = np.array([[humidity, wind_speed, d] for d in days])
    X_scaled = scaler.transform(inputs)
    predictions = model.predict(X_scaled)

    # Plot
    st.subheader("📊 Temperature Forecast")
    fig, ax = plt.subplots()
    ax.plot(days, predictions, marker='o')
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Predicted Temp (°C)")
    ax.set_title("7-Day Temperature Forecast")
    ax.grid(True)
    st.pyplot(fig)
