import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline

def gbm(X_train, X_test, y_train, y_test, preprocessor, n_estimators=100, max_depth=5, learning_rate=0.1):

    # Create and fit the pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate))
    ])

    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
 

    return mse,mae
