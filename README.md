## Project Summary

In this project, we tackled the challenge of accurately predicting Airbnb listing prices across three distinct datasets: Singapore, New York City, and Madrid. Through an extensive exploration of outlier management strategies and model optimization approaches, we gained valuable insights into the complexities of this regression problem.

### Initial Observations
Our initial baseline models, which included Linear Regression, Random Forest, and XGBoost, served as a solid foundation for understanding the datasets' characteristics and identifying areas for improvement. However, the presence of outliers and skewness in the data significantly impacted model performance, as evidenced by the high error metrics and low R-squared scores.

### Strategies for Improvement
To address these challenges, we employed a comprehensive set of strategies, including feature engineering, principal component analysis (PCA), and various outlier management techniques. Among these strategies, outlier management emerged as the most influential factor in enhancing model accuracy and reliability.

The application of log transformations to the target variable and numeric features proved effective, particularly for the larger and more skewed New York dataset. This approach helped mitigate the impact of extreme values and normalize the data distribution, leading to improved model performance.

However, the most substantial improvements were achieved through the removal of outliers using the interquartile range (IQR) method. By systematically identifying and eliminating outliers in both the target variable and numeric features, we witnessed significant reductions in error metrics (RMSE and MAE) and substantial increases in R-squared scores across all three datasets and algorithms.

Subsequent hyperparameter tuning further optimized the models' performance, unlocking their full potential and demonstrating the importance of tailoring model configurations to the specific characteristics of the data.

### Results
Here are the performance metrics for each algorithm:

| Algorithm         | RMSE  | MAE   | R2 Score |
|-------------------|-------|-------|----------|
| Random Forest     | 39.07 | 27.83 | 0.563    |
| Linear Regression | 42.82 | 31.36 | 0.488    |
| XGBoost           | 40.07 | 28.50 | 0.546    |

Random Forest has the best RMSE and MAE scores, suggesting it has the lowest average errors. XGBoost shows competitive performance, especially with a balanced RMSE and the second-best R2 score. Linear Regression has the highest RMSE and MAE values, and the lowest R2 score, indicating it generally performs worse than the others.

### Looking Ahead
Looking ahead, further research could explore more advanced outlier detection and handling techniques, as well as incorporate additional domain-specific features or external data sources to capture the nuances of Airbnb pricing dynamics better.

### Trade-offs
The act of removing outliers must be considered carefully, as it can lead to the loss of critical information. For stakeholders, these results suggest that outlier removal should be a targeted and context-dependent process. It can lead to improvements in model performance, as seen with the Madrid Dataset. However, it can also decrease model performance if valuable information is discarded, as evidenced by the Singapore Dataset particularly for the XGBoost model.

NB: Unfortunately, the datasets we used weren’t part of any competition on Kaggle, so we could not obtain a Kaggle score.
