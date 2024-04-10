from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline


def random_forest_regressor(X_train, X_test, y_train, y_test, preprocessor):


    # Creating and training the Random Forest model
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    rf_pipeline.fit(X_train, y_train)


    # Evaluating the model
    y_pred = rf_pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    return rmse, mae
    # print(f'RMSE: {rmse}, MAE: {mae}')