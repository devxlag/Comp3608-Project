
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from distance_transformer import DistanceToCenterTransformer



def preprocess_data(df):
    
    # if 'reviews_per_month' in df.columns:
    #     df['reviews_per_day'] = df['number_of_reviews'] / ((df['last_review'] - df['last_review'].min()).dt.days + 1)
    
    # Numeric features include engineered features
    numeric_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    
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
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    
    return preprocessor, X, y


