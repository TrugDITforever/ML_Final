import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_data
from model import train_model
import joblib

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('weather_hourly_danang_2020_2025.csv')  # Replace with your data file
    
    # Preprocess data
    print("Preprocessing data...")
    processed_df = preprocess_data(df)
    
    # Train model
    print("Training model...")
    model, mae = train_model(processed_df)
    
    # Visualize results (basic example)
    print("Generating visualizations...")
    # Load the model to make predictions on the entire dataset
    model = joblib.load('model.pkl')
    
    # If you want to visualize actual vs. predicted
    X = processed_df.drop(columns='temperature')
    actual = processed_df['temperature']
    predicted = model.predict(X)
    
    # Create simple visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title(f'Actual vs Predicted Temperatures (MAE: {mae:.2f}°)')
    plt.savefig('temperature_prediction_results.png')
    plt.show()
    
    print("Done! Model saved as 'model.pkl'")

if __name__ == "__main__":
    main()