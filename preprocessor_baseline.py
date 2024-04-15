from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


def preprocess_data(df):
    # Selecting relevant features for the model
    df.drop(['id', 'name', 'host_id', 'host_name'], axis=1, inplace=True)


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
    
    # Defining preprocessing for numeric columns
    numeric_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Defining preprocessing for categorical columns
    categorical_features = ['neighbourhood_group', 'neighbourhood', 'room_type']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combining preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X = df.drop('price', axis=1)
    y = df['price']

    return preprocessor, X, y        