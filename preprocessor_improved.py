
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from distance_transformer import DistanceToCenterTransformer



def preprocess_data(df):
    # Handling missing values in 'last_review'
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')  # Coerce errors will turn invalid parsing into NaT
    
    # Extract additional datetime features
    df['year'] = df['last_review'].dt.year
    df['month'] = df['last_review'].dt.month
    df['day'] = df['last_review'].dt.day
    df['weekday'] = df['last_review'].dt.weekday
    
    # Impute datetime features where 'last_review' is NaT
    datetime_features = ['year', 'month', 'day', 'weekday']
    for feature in datetime_features:
        df[feature].fillna(df[feature].median(), inplace=True)
    

    if 'reviews_per_month' in df.columns:
        df['reviews_per_day'] = df['number_of_reviews'] / ((df['last_review'] - df['last_review'].min()).dt.days + 1)
    
    # Numeric features include engineered features
    numeric_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'availability_365', 'year', 'month', 'day', 'weekday', 'reviews_per_day']
    
    # Handling outliers for numeric features
    for feature in numeric_features:
        if feature in df.columns:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[df[feature] < lower_bound, feature] = lower_bound
            df.loc[df[feature] > upper_bound, feature] = upper_bound
    
    # Categorical features for one-hot encoding
    categorical_features = ['neighbourhood_group', 'neighbourhood', 'room_type']
    
    # Pipelines for numeric and categorical preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combining preprocessing steps in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Exclude all other columns not specified in transformers
    )
    
    # Select only relevant features before applying transformations
    relevant_features = numeric_features + categorical_features
    X = df[relevant_features]
    y = df['price']
    
    
    return preprocessor, X, y

# # Re-run the function with the corrected feature selection
# preprocessed_data_onehot_corrected, target_onehot_corrected = preprocess_train_evaluate_onehot_corrected(data)
# preprocessed_data_onehot_corrected.shape

