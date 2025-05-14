import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Or any other model like Random Forest, etc.
from sklearn.metrics import mean_absolute_error
import joblib

def train_model(df):
    # Assuming the 'temperature' is the target variable
    X = df.drop(columns='temperature')  # Features (weather-related data)
    y = df['temperature']  # Target variable (predicted temperature)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model (e.g., Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model using MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae:.2f} degrees')
    
    # Save the model for later use
    joblib.dump(model, 'model.pkl')
    return model, mae
