from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import numpy as np
import pandas as pd

# Function to preprocess the Airbnb dataset
def preprocess_data(df, city_center_latitude=1.3521, city_center_longitude=103.8198):
    # Define numeric features
    numeric_features = [
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365'
    ]

    # Define categorical features
    categorical_features = ['neighbourhood_group', 'neighbourhood', 'room_type']

    # Log transformation to manage outliers in the 'price' column
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    df['log_price'] = log_transformer.transform(df[['price']])

    # Define numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Prepare features (excluding 'price') and target variable (log-transformed 'price')
    X = df.drop(['price', 'log_price'], axis=1)  # Drop 'price' and 'log_price' from features
    y_log = df['log_price']  # Log-transformed 'price'

    return preprocessor, X, y_log



 # Handling outliers for numeric features
    # for feature in numeric_features:
    #     if feature in df.columns:
    #         Q1 = df[feature].quantile(0.25)
    #         Q3 = df[feature].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 1.5 * IQR
    #         df.loc[df[feature] < lower_bound, feature] = lower_bound
    #         df.loc[df[feature] > upper_bound, feature] = upper_bound