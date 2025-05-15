import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean


def train_model(csv_path="D:\Python\ML_Final\weather_hourly_danang_2020_2025.csv", degree=3):
    # Load data
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["dayofyear"] = df["time"].dt.dayofyear
    df["year"] = df["time"].dt.year
    df = df.dropna()

    # Remove outliers
    cols_to_check = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "wind_speed_10m"]
    df = remove_outliers_iqr(df, cols_to_check)

    # Correlation-based feature selection
    correlation_columns = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "rain", "wind_speed_10m", "hour", "dayofyear"
    ]
    corr = df[correlation_columns].corr()
    print("📊 Correlation Matrix:")
    print(corr["temperature_2m"].sort_values(ascending=False))

    # Final selected features
    final_features = ["relative_humidity_2m", "wind_speed_10m", "dayofyear"]
    target = "temperature_2m"

    X = df[final_features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model using Polynomial Regression
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lin_reg", LinearRegression())
    ])
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n✅ Polynomial Degree {degree} Model Results:")
    print(f"RMSE: {rmse:.3f} °C")
    print(f"R² Score: {r2:.3f}")

    # Save model
    joblib.dump(model, f"poly_model_deg{degree}.pkl")

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted Temperature (Test Set)")
    plt.xlabel("Sample Index")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model, rmse, r2
model, rmse, r2 = train_model("D:\Python\ML_Final\weather_hourly_danang_2020_2025.csv", degree=3)
