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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


st.set_page_config(page_title="Temperature Forecast - Da Nang", layout="centered")

# Sidebar for mode selection including new training tab
mode = st.sidebar.radio("Select mode:", [ "📊 Data & Analysis", "🔧 Train Model", "🔢 Manual Prediction"])

# Load model và scaler if available
try:
    model = joblib.load("poly_model_deg4.pkl")
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
    # Remove duplicate rows if any
    df = df.drop_duplicates()
    # Optional: print number of duplicates removed (for logging/debugging)
    print("Duplicates removed:", df.duplicated().sum())
    # Outlier removal using IQR
    q1 = df["temperature_2m"].quantile(0.25)
    q3 = df["temperature_2m"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_clean = df[(df["temperature_2m"] >= lower_bound) & (df["temperature_2m"] <= upper_bound)]
    return df_clean

# Cache model training and evaluation
@st.cache_data
def train_all_models():
    df = load_data()
    X = df[[
        "relative_humidity_2m", 
        "cloud_cover_mid",
        "cloud_cover",
        "pressure_msl", 
        # "surface_pressure", 
        # "hour_sin", 
        "hour_cos",
        # "day_sin", 
        "day_cos"
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
        if d == 4:  # Update this if you're training degree 4 by default
            joblib.dump(model, f"poly_model_deg{d}.pkl")
            joblib.dump(d, "trained_degree.pkl")
            joblib.dump(poly, f"poly_transformer_deg{d}.pkl")
        results.append((f"Polynomial (deg={d})", rmse, r2, cv_rmse.mean(), cv_rmse.std(), cv_r2.mean(), cv_r2.std()))

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

if mode == "🔧 Train Model":
    st.sidebar.markdown("## 🔧 Train Model")
    if st.button("Train All Models"):
        results = train_all_models()
        st.session_state["train_results"] = results
    elif "train_results" in st.session_state:
        results = st.session_state["train_results"]
    else:
        st.warning("Click 'Train All Models' to start model training.")

    if "results" in locals():
        st.markdown("### 📊 Trained Model Results")

        # Convert to DataFrame
        results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2", "CV_RMSE", "CV_RMSE_STD", "CV_R2", "CV_R2_STD"])

        # Show summary table
        st.dataframe(results_df[["Model", "RMSE", "R2", "CV_RMSE", "CV_R2"]])

        # Chart: RMSE
        st.markdown("#### 📉 RMSE Comparison")
        fig_rmse, ax_rmse = plt.subplots()
        sns.barplot(data=results_df, x="Model", y="RMSE", ax=ax_rmse, palette="Blues_d")
        ax_rmse.set_title("RMSE on Training Set")
        ax_rmse.set_xticklabels(results_df["Model"], rotation=45, ha="right")
        st.pyplot(fig_rmse)

        # Chart: R²
        st.markdown("#### 📈 R² Comparison")
        fig_r2, ax_r2 = plt.subplots()
        sns.barplot(data=results_df, x="Model", y="R2", ax=ax_r2, palette="Greens_d")
        ax_r2.set_title("R² on Training Set")
        ax_r2.set_xticklabels(results_df["Model"], rotation=45, ha="right")
        st.pyplot(fig_r2)

        # Optional: Show CV RMSE chart
        st.markdown("#### 🔁 Cross-validated RMSE (5-fold)")
        fig_cv_rmse, ax_cv_rmse = plt.subplots()
        sns.barplot(data=results_df, x="Model", y="CV_RMSE", ax=ax_cv_rmse, palette="Purples_d")
        ax_cv_rmse.set_title("Cross-validated RMSE")
        ax_cv_rmse.set_xticklabels(results_df["Model"], rotation=45, ha="right")
        st.pyplot(fig_cv_rmse)

elif mode == "🔢 Manual Prediction":
    st.title("🌡️ Temperature Forecast - Da Nang")

    # Load model and scaler if not already loaded
    if model is None or scaler is None:
        st.error("Model or scaler not trained or uploaded.")
    else:
        scaler = joblib.load("temperature_scaler.pkl")
        default_data = get_latest_weather_data()
        st.markdown("### 🎛️ Adjust Prediction Parameters")

        col1, col2, col3 = st.columns(3)
        with col1:
            humidity = st.slider("💧 Humidity (%)", 0, 100, default_data.get("humidity", 85))
            cloud_mid = st.slider("☁️ Mid-level Cloud (%)", 0, 100, default_data.get("cloud_mid", 20))
            if cloud_mid < 0 or cloud_mid > 100:
                st.error("Mid-level cloud value must be between 0-100%")
                cloud_mid = 20
        if cloud_mid < 0 or cloud_mid > 100:
            st.error("Mid-level cloud value must be between 0-100%")
            cloud_mid = 20
        with col2:
            cloud_total = st.slider("☁️ Total Cloud (%)", 0, 100, default_data.get("cloud_total", 60))
            pressure_msl = st.slider(
    "🌡️ Mean Sea Level Pressure (hPa)",
    900,
    1100,
    int(default_data.get("pressure_msl", 1013))
)
            # pressure_surface = st.slider("🌡️ Surface Pressure (hPa)", 900, 1100, int(default_data.get("pressure_surface", 1010)))
        with col3:
            dayofyear = st.slider("📅 Day of Year", 1, 366, datetime.datetime.now().timetuple().tm_yday)
            date_display = datetime.datetime.strptime(f"{dayofyear}", "%j").strftime("%d/%m")
            st.caption(f"🗓️ Corresponds to: {date_display}")

        day_sin = np.sin(2 * np.pi * dayofyear / 365)
        day_cos = np.cos(2 * np.pi * dayofyear / 365)

        X_input = np.array([[humidity, cloud_mid, cloud_total,
                             pressure_msl,
                            #  np.sin(2 * np.pi * datetime.datetime.now().hour / 24),
                             np.cos(2 * np.pi * datetime.datetime.now().hour / 24),
                            #  day_sin, 
                             day_cos]])
        X_scaled = scaler.transform(X_input)
        try:
            trained_degree = joblib.load("trained_degree.pkl")
        except:
            trained_degree = 1

        if trained_degree > 1:
            poly_transformer = joblib.load(f"poly_transformer_deg{trained_degree}.pkl")
            X_poly = poly_transformer.transform(X_scaled)
            predicted_temp = model.predict(X_poly)[0]
        else:
            predicted_temp = model.predict(X_scaled)[0]

        st.markdown("---")
        st.markdown("### 🌤️ Prediction Result")
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
            st.error("Unable to predict temperature. Please check if the model is trained or the scaler is available.")

elif mode == "📊 Data & Analysis":
    st.title("🌡️ Temperature Forecast - Da Nang")
    st.markdown("## 🧪 Original Data Analysis (EDA)")

    df = load_data()

    st.markdown("### 📊 Temperature Distribution (Histogram & Boxplot)")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df["temperature_2m"], bins=30, kde=True, ax=ax1, color="skyblue")
        ax1.set_title("Temperature Histogram")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.boxplot(y=df["temperature_2m"], ax=ax2, color="orange")
        ax2.set_title("Temperature Boxplot")
        st.pyplot(fig2)

    selected_year = st.selectbox("Select year to view temperature over time", sorted(df["year"].unique()))

    yearly_df = df[df["year"] == selected_year].resample("D", on="time").mean(numeric_only=True)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(yearly_df.index, yearly_df["temperature_2m"], color="green")
    ax3.set_title(f"Average Daily Temperature ({selected_year})")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Temperature (°C)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig3)

    st.markdown("### 🚨 Outlier Detection")
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
        ax4.set_title("Before Outlier Removal")
        st.pyplot(fig4)

    with col4:
        fig5, ax5 = plt.subplots()
        sns.boxplot(y=df_clean["temperature_2m"], ax=ax5, color="limegreen")
        ax5.set_title("After Outlier Removal")
        st.pyplot(fig5)
    

    st.success(f"Number of outliers: {len(df_outliers)} | Remaining data: {len(df_clean)}")

    st.markdown("### 🔗 Bivariate Analysis")

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
        ax.set_title(f"Temperature vs {feature}")
        st.pyplot(fig)

    st.markdown("### 🔍 Multivariate Analysis")

    st.markdown("#### 📈 Full Feature Correlation Matrix")
    fig_full_corr, ax_full_corr = plt.subplots(figsize=(12, 10))
    # all these features temperature_2m,dew_point_2m,apparent_temperature,relative_humidity_2m,precipitation,rain,cloud_cover,cloud_cover_mid,cloud_cover_low,cloud_cover_high,pressure_msl,surface_pressure,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,vapour_pressure_deficit
    full_features = [
        "temperature_2m", "dew_point_2m", "apparent_temperature",
        "relative_humidity_2m", "precipitation", "rain",
        "cloud_cover", "cloud_cover_mid", "cloud_cover_low",
        "cloud_cover_high", "pressure_msl", "surface_pressure",
        "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
        "vapour_pressure_deficit", "hour", "dayofyear"
    ]
    sns.heatmap(df[full_features].corr(), annot=True, cmap="coolwarm", ax=ax_full_corr)
    st.pyplot(fig_full_corr)

    st.markdown("#### 📈 Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    
    selected_features = [
    "temperature_2m",  # target
    "relative_humidity_2m", "cloud_cover_low", "cloud_cover_mid", "cloud_cover",
    "pressure_msl", "hour", "dayofyear"
]
    
    sns.heatmap(df[selected_features].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    col5 = st.columns(1)[0]

    with col5:
        st.markdown("### 📈 Univariate Analysis")
        st.markdown("#### 📊 Temperature Distribution Chart")
        fig6, ax6 = plt.subplots()
        sns.histplot(df["temperature_2m"], bins=30, kde=True, ax=ax6, color="skyblue")
        ax6.set_title("Temperature Distribution")
        st.pyplot(fig6)

        st.markdown("---")
        st.markdown("### 🎯 Principal Component Analysis (PCA with Cyclical Encoding)")
        cyclical_features = [
            "relative_humidity_2m",
            "cloud_cover_low",
            "cloud_cover_mid",     # optional
            "cloud_cover",         # optional
            "pressure_msl",
            # "surface_pressure",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos"
        ]
        X_cyclical = df[cyclical_features].dropna()
        scaler_cyc = StandardScaler()
        X_scaled_cyc = scaler_cyc.fit_transform(X_cyclical)

        pca_cyc = PCA(n_components=len(cyclical_features))
        X_pca_cyc = pca_cyc.fit_transform(X_scaled_cyc)

        st.markdown("#### 🔍 Scree Plot (PCA with Cyclical Encoding)")
        fig_cyc, ax_cyc = plt.subplots()
        ax_cyc.plot(np.cumsum(pca_cyc.explained_variance_ratio_), marker='o')
        ax_cyc.axhline(y=0.95, color="r", linestyle="--", label="95% Threshold")
        ax_cyc.set_xlabel("Number of Principal Components")
        ax_cyc.set_ylabel("Cumulative Explained Variance Ratio")
        ax_cyc.set_title("Elbow Plot - PCA (cyclical)")
        ax_cyc.grid(True, linestyle="--", alpha=0.5)
        ax_cyc.legend()
        st.pyplot(fig_cyc)

        # Calculate feature importance from PCA loadings (cyclical)
        pca_components = np.abs(pca_cyc.components_)
        feature_importance = pca_components.sum(axis=0)
        sorted_indices = np.argsort(feature_importance)[::-1]
        sorted_features = np.array(cyclical_features)[sorted_indices]
        sorted_importance = feature_importance[sorted_indices]

        st.markdown("#### 🔍 Feature Contribution (PCA with Cyclical Encoding)")
        fig_feat_cyc, ax_feat_cyc = plt.subplots()
        sns.barplot(x=sorted_importance, y=sorted_features, ax=ax_feat_cyc, palette="Blues_d")
        ax_feat_cyc.set_xlabel("Total Contribution to PCA")
        ax_feat_cyc.set_title("Feature Importance - PCA (cyclical)")
        st.pyplot(fig_feat_cyc)

        st.markdown("### 🎯 Principal Component Analysis (PCA)")
        features = [
            "relative_humidity_2m",
            "cloud_cover_low",
            "cloud_cover_mid",     # optional
            "cloud_cover",         # optional
            "pressure_msl",
            # "surface_pressure",
            "hour",
            "dayofyear"
        ]
        # Standardize data
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=len(features))
        X_pca = pca.fit_transform(X_scaled)

        # Scree Plot
        st.markdown("#### 🔍 Scree Plot (Select Number of Principal Components)")
        fig7, ax7 = plt.subplots()
        ax7.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        ax7.axhline(y=0.95, color="r", linestyle="--", label="95% Threshold")
        ax7.set_xlabel("Number of Principal Components")
        ax7.set_ylabel("Cumulative Explained Variance Ratio")
        ax7.set_title("Elbow Plot - PCA")
        ax7.grid(True, linestyle="--", alpha=0.5)
        ax7.legend()
        st.pyplot(fig7)

        # Feature importance from standard PCA
        pca_components_std = np.abs(pca.components_)
        feature_importance_std = pca_components_std.sum(axis=0)
        sorted_indices_std = np.argsort(feature_importance_std)[::-1]
        sorted_features_std = np.array(features)[sorted_indices_std]
        sorted_importance_std = feature_importance_std[sorted_indices_std]

        st.markdown("#### 🔍 Feature Contribution (Standard PCA)")
        fig_feat_std, ax_feat_std = plt.subplots()
        sns.barplot(x=sorted_importance_std, y=sorted_features_std, ax=ax_feat_std, palette="Greens_d")
        ax_feat_std.set_xlabel("Total Contribution to PCA")
        ax_feat_std.set_title("Feature Importance - PCA")
        st.pyplot(fig_feat_std)
