import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Handle missing values by filling with mean for numerical columns
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
        ('scaler', StandardScaler())                 # Standardize the numerical columns
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncoding for categorical features
    ])

    # Apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform data
    df_processed = preprocessor.fit_transform(df)
    
    # Convert the transformed dataframe to a DataFrame (use column names for encoding)
    processed_df = pd.DataFrame(df_processed)
    
    return processed_df
