import streamlit as st
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests

def get_latest_weather_data():
    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast?latitude=16.07&longitude=108.22&current=temperature_2m,relative_humidity_2m,cloud_cover,cloud_cover_mid,pressure_msl,surface_pressure")
        if response.status_code == 200:
            data = response.json().get("current", {})
            return {
                "humidity": data.get("relative_humidity_2m", 85),
                "cloud_mid": data.get("cloud_cover_mid", 20),
                "cloud_total": data.get("cloud_cover", 60),
                "pressure_msl": data.get("pressure_msl", 1013),
                "pressure_surface": data.get("surface_pressure", 1010),
            }
        else:
            return {}
    except Exception:
        return {}


# Cấu hình trang
st.set_page_config(page_title="Dự báo nhiệt độ Đà Nẵng", layout="centered")

# Sidebar for mode selection including new training tab
mode = st.sidebar.radio("Chọn chế độ:", ["🔢 Manual Prediction", "📊 Dữ liệu & Phân tích", "🔧 Huấn luyện mô hình"])

# Load model và scaler if available
try:
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("temperature_scaler.pkl")
except:
    model = None
    scaler = None


# Function to load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("weather_hourly_danang_2020_2025.csv")
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["dayofyear"] = df["time"].dt.dayofyear
    df["year"] = df["time"].dt.year
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df = df.dropna()
    return df

# Cache model training and evaluation
@st.cache_data
def train_all_models():
    df = load_data()
    X = df[[
        "relative_humidity_2m", 
        "cloud_cover_mid",
        "cloud_cover",
        "pressure_msl", 
        "surface_pressure", 
        "hour_sin", "hour_cos",
        "day_sin", "day_cos"
    ]].values
    y = df["temperature_2m"].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    # Linear
    lin_model = LinearRegression()
    lin_model.fit(X_scaled, y)
    y_pred = lin_model.predict(X_scaled)
    lin_rmse = np.sqrt(mean_squared_error(y, y_pred))
    lin_r2 = r2_score(y, y_pred)
    lin_cv_rmse = -cross_val_score(lin_model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    lin_cv_r2 = cross_val_score(lin_model, X_scaled, y, cv=5, scoring='r2')
    joblib.dump(lin_model, "linear_model.pkl")
    results.append(("Linear", lin_rmse, lin_r2, lin_cv_rmse.mean(), lin_cv_rmse.std(), lin_cv_r2.mean(), lin_cv_r2.std()))

    # Polynomial
    for d in range(1, 7):
        poly = PolynomialFeatures(degree=d)
        X_poly = poly.fit_transform(X_scaled)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        cv_rmse = -cross_val_score(model, X_poly, y, cv=5, scoring='neg_root_mean_squared_error')
        cv_r2 = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
        if d == 5:
            joblib.dump(model, f"poly_model_deg{d}.pkl")
            joblib.dump(d, "trained_degree.pkl")
        results.append((f"Polynomial (deg={d})", rmse, r2, cv_rmse.mean(), cv_rmse.std(), cv_r2.mean(), cv_r2.std()))


    # SVR
    # from sklearn.svm import SVR
    # svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    # svr.fit(X_scaled, y)
    # y_pred = svr.predict(X_scaled)
    # rmse = np.sqrt(mean_squared_error(y, y_pred))
    # r2 = r2_score(y, y_pred)
    # cv_rmse = -cross_val_score(svr, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    # cv_r2 = cross_val_score(svr, X_scaled, y, cv=5, scoring='r2')
    # joblib.dump(svr, "svr_model.pkl")
    # results.append(("SVR", rmse, r2, cv_rmse.mean(), cv_rmse.std(), cv_r2.mean(), cv_r2.std()))

    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    y_pred = rf.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    cv_rmse = -cross_val_score(rf, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_r2 = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
    joblib.dump(rf, "rf_model.pkl")
    results.append(("Random Forest", rmse, r2, cv_rmse.mean(), cv_rmse.std(), cv_r2.mean(), cv_r2.std()))

    joblib.dump(scaler, "temperature_scaler.pkl")
    return results

if mode == "🔧 Huấn luyện mô hình":
    st.sidebar.markdown("## 🔧 Huấn luyện mô hình")
    results = train_all_models()
    # if st.button("🔄 Huấn luyện lại từ đầu"):
    #     results = train_all_models()
    # else:
    #     results = train_all_models()
    st.markdown("### 📊 Kết quả các mô hình đã huấn luyện")

    # Convert to DataFrame
    results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2", "CV_RMSE", "CV_RMSE_STD", "CV_R2", "CV_R2_STD"])

    # Show summary table
    st.dataframe(results_df[["Model", "RMSE", "R2", "CV_RMSE", "CV_R2"]])

    # Chart: RMSE
    st.markdown("#### 📉 So sánh RMSE")
    fig_rmse, ax_rmse = plt.subplots()
    sns.barplot(data=results_df, x="Model", y="RMSE", ax=ax_rmse, palette="Blues_d")
    ax_rmse.set_title("RMSE trên tập huấn luyện")
    ax_rmse.set_xticklabels(results_df["Model"], rotation=45, ha="right")
    st.pyplot(fig_rmse)

    # Chart: R²
    st.markdown("#### 📈 So sánh R²")
    fig_r2, ax_r2 = plt.subplots()
    sns.barplot(data=results_df, x="Model", y="R2", ax=ax_r2, palette="Greens_d")
    ax_r2.set_title("R² trên tập huấn luyện")
    ax_r2.set_xticklabels(results_df["Model"], rotation=45, ha="right")
    st.pyplot(fig_r2)

    # Optional: Show CV RMSE chart
    st.markdown("#### 🔁 So sánh CV RMSE (5-fold)")
    fig_cv_rmse, ax_cv_rmse = plt.subplots()
    sns.barplot(data=results_df, x="Model", y="CV_RMSE", ax=ax_cv_rmse, palette="Purples_d")
    ax_cv_rmse.set_title("Cross-validated RMSE")
    ax_cv_rmse.set_xticklabels(results_df["Model"], rotation=45, ha="right")
    st.pyplot(fig_cv_rmse)

elif mode == "🔢 Manual Prediction":
    st.title("🌡️ Dự báo nhiệt độ - Đà Nẵng")

    # Load model and scaler if not already loaded
    if model is None or scaler is None:
        st.error("Mô hình hoặc scaler chưa được huấn luyện hoặc tải lên.")
    else:
        scaler = joblib.load("temperature_scaler.pkl")
        default_data = get_latest_weather_data()
        st.markdown("### 🎛️ Điều chỉnh thông số dự đoán")

        col1, col2, col3 = st.columns(3)
        with col1:
            humidity = st.slider("💧 Độ ẩm (%)", 0, 100, default_data.get("humidity", 85))
            cloud_mid = st.slider("☁️ Mây trung bình (%)", 0, 100, default_data.get("cloud_mid", 20))
            if cloud_mid < 0 or cloud_mid > 100:
                st.error("Giá trị mây trung bình phải trong khoảng 0-100%")
                cloud_mid = 20
        if cloud_mid < 0 or cloud_mid > 100:
            st.error("Giá trị mây trung bình phải trong khoảng 0-100%")
            cloud_mid = 20
        with col2:
            cloud_total = st.slider("☁️ Tổng mây (%)", 0, 100, default_data.get("cloud_total", 60))
            pressure_msl = st.slider(
    "🌡️ Áp suất mực nước biển (hPa)",
    900,
    1100,
    int(default_data.get("pressure_msl", 1013))
)
            pressure_surface = st.slider("🌡️ Áp suất bề mặt (hPa)", 900, 1100, int(default_data.get("pressure_surface", 1010)))
        with col3:
            dayofyear = st.slider("📅 Ngày trong năm", 1, 366, datetime.datetime.now().timetuple().tm_yday)
            date_display = datetime.datetime.strptime(f"{dayofyear}", "%j").strftime("%d/%m")
            st.caption(f"🗓️ Tương ứng: {date_display}")

        day_sin = np.sin(2 * np.pi * dayofyear / 365)
        day_cos = np.cos(2 * np.pi * dayofyear / 365)

        X_input = np.array([[humidity, cloud_mid, cloud_total,
                             pressure_msl, pressure_surface,
                             np.sin(2 * np.pi * datetime.datetime.now().hour / 24),
                             np.cos(2 * np.pi * datetime.datetime.now().hour / 24),
                             day_sin, day_cos]])
        X_scaled = scaler.transform(X_input)
        try:
            trained_degree = joblib.load("trained_degree.pkl")
        except:
            trained_degree = 1

        if trained_degree > 1 and "poly" in str(type(model)).lower():
            poly = PolynomialFeatures(degree=trained_degree)
            X_poly = poly.fit_transform(X_scaled)
            predictions = model.predict(X_poly)
            predicted_temp = predictions[0]
        else:
            predicted_temp = model.predict(X_scaled)[0]

        st.markdown("---")
        st.markdown("### 🌤️ Kết quả dự đoán")
        if predicted_temp is not None:
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
        else:
            st.error("Không thể dự đoán nhiệt độ. Vui lòng kiểm tra lại mô hình đã huấn luyện hoặc scaler.")

elif mode == "📊 Dữ liệu & Phân tích":
    st.title("🌡️ Dự báo nhiệt độ - Đà Nẵng")
    st.markdown("## 🧪 Phân tích dữ liệu gốc (EDA)")

    df = load_data()

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

    st.markdown("### 🔗 Phân tích nhị biến (Bivariate)")

    features_to_compare = ["temperature_2m",
        "relative_humidity_2m",
    "cloud_cover_low",
    "cloud_cover_mid",     # optional
    "cloud_cover",         # optional
    "pressure_msl",
    "surface_pressure",
    "hour",
    "dayofyear"]
    for feature in features_to_compare:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature, y="temperature_2m", ax=ax, alpha=0.4)
        ax.set_title(f"Nhiệt độ vs {feature}")
        st.pyplot(fig)

    st.markdown("### 🔍 Phân tích đa biến (Multivariate)")

    st.markdown("#### 📈 Ma trận tương quan")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    selected_features = [
    "temperature_2m",  # target
    "relative_humidity_2m", "cloud_cover_low", "cloud_cover_mid", "cloud_cover",
    "pressure_msl", "surface_pressure", "hour", "dayofyear"
]
    sns.heatmap(df[selected_features].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
