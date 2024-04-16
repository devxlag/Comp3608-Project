import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

def xg_boost(X_train, X_test, y_train, y_test, preprocessor):

    # Create and fit the pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1))
    ])

    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)

    return rmse, mae, explained_variance

 
