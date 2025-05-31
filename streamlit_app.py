import streamlit as st
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Cấu hình trang
st.set_page_config(page_title="Dự báo nhiệt độ Đà Nẵng", layout="centered")

# Load model và scaler
model = joblib.load("poly_model_deg3.pkl")
scaler = joblib.load("temperature_scaler.pkl")

st.title("🌡️ Dự báo nhiệt độ - Đà Nẵng")

# Radio chọn chế độ
mode = st.radio("Chọn chế độ:", ["🔢 Manual Prediction", "📈 7-Day Forecast", "📊 Dữ liệu & Phân tích"])


# ======= Chế độ 1: Dự đoán thủ công =======
if mode == "🔢 Manual Prediction":
    st.markdown("### 🎛️ Điều chỉnh thông số dự đoán")

    col1, col2, col3 = st.columns(3)
    with col1:
        humidity = st.slider("💧 Độ ẩm (%)", 0, 100, 85)
    with col2:
        wind_speed = st.slider("💨 Tốc độ gió (m/s)", 0.0, 50.0, 10.0)
    with col3:
        dayofyear = st.slider("📅 Ngày trong năm", 1, 366, datetime.datetime.now().timetuple().tm_yday)
        date_display = datetime.datetime.strptime(f"{dayofyear}", "%j").strftime("%d/%m")
        st.caption(f"🗓️ Tương ứng: {date_display}")

    # Dự đoán
    X_input = np.array([[humidity, wind_speed, dayofyear]])
    X_scaled = scaler.transform(X_input)
    predicted_temp = model.predict(X_scaled)[0]

    st.markdown("---")
    st.markdown("### 🌤️ Kết quả dự đoán")
    st.markdown(f"""
    <div style='
        padding: 25px;
        background-color: #e3f2fd;
        border-radius: 15px;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #0d47a1;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
        🌡️ {predicted_temp:.2f} °C
    </div>
    """, unsafe_allow_html=True)

# ======= Chế độ 2: Dự báo 7 ngày =======
elif mode == "📈 7-Day Forecast":
    st.markdown("### 📆 Dự báo 7 ngày tiếp theo")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_day = st.slider("📅 Bắt đầu từ ngày", 1, 359, datetime.datetime.now().timetuple().tm_yday)
        date_start = datetime.datetime.strptime(f"{start_day}", "%j").strftime("%d/%m")
        st.caption(f"🗓️ Ngày bắt đầu: {date_start}")

    with col2:
        humidity = st.slider("💧 Độ ẩm giả định (%)", 0, 100, 80)
    with col3:
        wind_speed = st.slider("💨 Gió giả định (m/s)", 0.0, 50.0, 12.0)

    # Dự đoán
    days = np.array([start_day + i for i in range(7)])
    inputs = np.array([[humidity, wind_speed, d] for d in days])
    X_scaled = scaler.transform(inputs)
    predictions = model.predict(X_scaled)

    # Vẽ biểu đồ
    st.markdown("---")
    st.markdown("### 📊 Biểu đồ dự báo nhiệt độ")
    fig, ax = plt.subplots()
    ax.plot(days, predictions, marker='o', linestyle='-', color='#1e88e5')
    ax.set_xticks(days)
    ax.set_xlabel("📅 Ngày trong năm")
    ax.set_ylabel("🌡️ Nhiệt độ dự đoán (°C)")
    ax.set_title("Dự báo nhiệt độ 7 ngày")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # Hiển thị box từng ngày
    st.markdown("### 📦 Chi tiết từng ngày")
    cols = st.columns(7)
    st.markdown("### 📦 Chi tiết từng ngày")
    for i in range(7):
        day_label = datetime.datetime.strptime(str(days[i]), "%j").strftime("%d/%m")
        temp = predictions[i]
        icon = "🔥" if temp >= 35 else "🌤️" if temp >= 30 else "🌥️" if temp >= 25 else "🌦️"
        st.markdown(f"**{icon} {day_label}**")
        st.metric(label="Nhiệt độ", value=f"{temp:.2f}°C")
        st.markdown("---")


elif mode == "📊 Dữ liệu & Phân tích":
    st.markdown("## 🧪 Phân tích dữ liệu gốc (EDA)")

# Đọc và xử lý dữ liệu gốc
    def load_data():
        df = pd.read_csv("weather_hourly_danang_2020_2025.csv")
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour
        df["dayofyear"] = df["time"].dt.dayofyear
        df["year"] = df["time"].dt.year
        df = df.dropna()
        return df

    df = load_data()

    # --- 1. Phân phối dữ liệu ---
    st.markdown("### 📊 Phân phối nhiệt độ (Histogram & Boxplot)")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df["temperature_2m"], bins=30, kde=True, ax=ax1, color="skyblue")
        ax1.set_title("Histogram nhiệt độ")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.boxplot(y=df["temperature_2m"], ax=ax2, color="orange")
        ax2.set_title("Boxplot nhiệt độ")
        st.pyplot(fig2)

    selected_year = st.selectbox("Chọn năm để xem nhiệt độ theo thời gian", sorted(df["year"].unique()))

    yearly_df = df[df["year"] == selected_year].resample("D", on="time").mean(numeric_only=True)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(yearly_df.index, yearly_df["temperature_2m"], color="green")
    ax3.set_title(f"Nhiệt độ trung bình mỗi ngày ({selected_year})")
    ax3.set_xlabel("Ngày")
    ax3.set_ylabel("Nhiệt độ (°C)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig3)



    # --- 3. Outlier detection ---
    st.markdown("### 🚨 Phát hiện ngoại lệ (Outliers)")
    q1 = df["temperature_2m"].quantile(0.25)
    q3 = df["temperature_2m"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_outliers = df[(df["temperature_2m"] < lower_bound) | (df["temperature_2m"] > upper_bound)]
    df_clean = df[(df["temperature_2m"] >= lower_bound) & (df["temperature_2m"] <= upper_bound)]

    col3, col4 = st.columns(2)
    with col3:
        fig4, ax4 = plt.subplots()
        sns.boxplot(y=df["temperature_2m"], ax=ax4, color="tomato")
        ax4.set_title("Trước khi loại bỏ ngoại lệ")
        st.pyplot(fig4)

    with col4:
        fig5, ax5 = plt.subplots()
        sns.boxplot(y=df_clean["temperature_2m"], ax=ax5, color="limegreen")
        ax5.set_title("Sau khi loại bỏ ngoại lệ")
        st.pyplot(fig5)

    st.success(f"Số lượng điểm ngoại lệ: {len(df_outliers)} | Dữ liệu còn lại: {len(df_clean)}")

