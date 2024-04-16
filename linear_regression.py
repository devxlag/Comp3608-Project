from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score


def linear_regression(X_train, X_test, y_train, y_test, preprocessor):
    
    # Creating and training the Linear Regression model
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    
    # Predicting on the test set
    y_pred = lr_pipeline.predict(X_test)

    # Calculating the evaluation metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)

    return rmse, mae, explained_variance