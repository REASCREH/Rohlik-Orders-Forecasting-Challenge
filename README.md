
# Rohlik-Orders-Forecasting-Challenge
 Project Overview: Predicting Daily Order Volumes for Rohlik 

 This project tackles the Rohlik Orders Forecasting Challenge, aiming to build a robust and accurate system for predicting daily order volumes for Rohlik, a leading online grocery delivery service. By leveraging historical data and a rich set of contextual features, we developed a predictive model that not only offers high accuracy but also provides actionable insights for critical operational planning.

 Key Achievements:

Exceptional Accuracy: Our XGBoost model achieved a remarkable Mean Absolute Percentage Error (MAPE) of 3.37% on unseen test data. This outstanding accuracy makes the model highly reliable for real-world predictions, especially considering the inherent variability in demand forecasting.
Strong Generalization: With an R² score of 0.9856 on the test set, the model explains nearly all the variance in future order patterns. This indicates excellent generalization capabilities and minimal overfitting, ensuring reliable performance on new data.
Comprehensive Insights: Through detailed Exploratory Data Analysis (EDA) and feature importance analysis, we identified key drivers of order volumes. These include warehouse-specific trends, strong weekly and monthly seasonality, and the significant impact of user activity metrics. These insights are invaluable for strategic decision-making.








## Data Used in This Project 📊

Our project uses daily order data from Rohlik, an online grocery service. This data helps us understand and predict how many orders they'll get.

Data Details:

When? We trained our model with data from December 2020 to March 2024. We tested it on data from March to May 2024 to see how well it predicts future orders.
Where? The data comes from 7 different warehouses in Central Europe:

Czech Republic: Prague (3 locations), Brno (1 location)
Germany: Munich (1 location), Frankfurt (1 location)
Hungary: Budapest (1 location)
How often? We have data for every single day, which helps us spot detailed patterns.

What Each Column Means (Simple Breakdown):
Here's a look at the important information in our dataset:

Basic Order Info:

id: A unique code for each day at each warehouse (e.g., "Prague_1_2020-12-05").

warehouse: The specific location (like "Prague_1" or "Budapest_1"). This helps us understand regional differences.

date: The day the orders were placed. Super important for looking at trends over time.

orders: This is what we want to predict! It's the total number of orders for that day.

Business Events:

These columns tell us about special days or operational changes:

holiday_name: The name of a holiday if there was one (e.g., "Christmas"). We also created special numbers from these names to help the model learn.

holiday: A simple "yes" (1) or "no" (0) if it was a public holiday. Holidays often change how people order.

shutdown: "Yes" (1) or "no" (0) if there was a major stop in operations. This means very few orders.

mini_shutdown: "Yes" (1) or "no" (0) if there was a smaller disruption.

shops_closed: "Yes" (1) or "no" (0) if physical stores were closed. This might make more people order online.
Seasonal & Calendar Info:

These help us see regular patterns:

winter_school_holidays: "Yes" (1) or "no" (0) for winter school breaks.
school_holidays: "Yes" (1) or "no" (0) for any school holidays.
blackout: "Yes" (1) or "no" (0) for power outages. Rare, but important.
Outside Factors:
Things not directly related to Rohlik but impacting orders:

mov_change: "Yes" (1) or "no" (0) for a change that might be related to minimum order value.
frankfurt_shutdown: "Yes" (1) or "no" (0) specifically for shutdowns in Frankfurt.
Weather Info:
Weather can definitely affect online ordering!

precipitation: How much rain or snow fell that day (in millimeters). Bad weather often means more online orders.
snow: How much snow fell (in millimeters). Strong impact on orders and delivery.
User Activity:

These show how busy Rohlik's platforms are:

user_activity_1: A number showing some kind of user activity. It strongly relates to more orders.

user_activity_2: Another number for user activity. Also strongly relates to more orders.
New Features We Made:

We created these from the basic data to help our model learn better:

country: Which country the warehouse is in (Czech Republic, Germany, Hungary). Useful for national trends.
month, day_of_week, year, day, quarter, month_name, week: Simple parts of the date to capture daily, weekly, and yearly patterns.

_sin & _cos (Cyclical Features): Special math tricks to tell the model that things like "December" are close to "January" in a cycle (for months, days, etc.).

total_holidays_month, total_shops_closed_week: Counts of holidays or shop closures over a month/week.

holiday_name_tfidf_0 to holiday_name_tfidf_9: Numbers created from holiday_name to understand different types of holidays.

holiday_before, holiday_after: "Yes" (1) or "no" (0) if a holiday was yesterday or is tomorrow. This helps the model understand how orders change around holidays.



## Model Training and Feature Engineering

We leveraged XGBoost (Extreme Gradient Boosting), a highly efficient and flexible open-source library, for this forecasting task. Its ability to handle diverse data types, manage missing values, and prevent overfitting makes it an excellent choice.

The meticulous model training and feature engineering process were documented and executed in this Kaggle Notebook.

Feature Engineering Details:
The creation of new, insightful features from raw data was fundamental to our approach:

Date-based Features: Extraction of year, month, day, quarter, week, day_of_week, and month_name.
Cyclical Transformations: Sine and cosine transformations applied to year, month, day, quarter, and a custom group feature. This helps the model correctly interpret the cyclical nature of time.
Aggregated Features: total_holidays_month and total_shops_closed_week provide broader context for holiday or closure intensity.
Lag/Lead Features: holiday_before and holiday_after capture the anticipatory or residual effects of holidays, which are vital for time-series forecasting.
Text Feature Engineering: TF-IDF and Truncated SVD applied to holiday_name to convert text into meaningful numerical features, capturing semantic patterns.
Categorical Encoding: One-hot encoding was used for categorical features like warehouse, holiday, day_of_week, month_name, and country, making them suitable for the XGBoost algorithm.
XGBoost Model Configuration:
Our XGBoost model was carefully configured to balance performance and generalization:

Booster: DART (Dropout Approximately Rank-one Tensor) was chosen. DART uses dropout to prevent overfitting by randomly dropping trees during training.
Learning Rate (eta / learning_rate): 0.06: A moderate rate balancing training speed and accuracy.
Maximum Tree Depth (max_depth): 8: Allows complex interactions while guarding against overfitting.
Regularization (alpha, lambda): alpha=9 (L1), lambda=8 (L2): Strong regularization values (alpha for L1, lambda for L2) were used to penalize large weights and encourage sparsity, effectively preventing overfitting.
Subsampling (subsample, colsample_bytree): subsample=0.7, colsample_bytree=0.7: 70% of data instances and features were randomly sampled for each tree. This increases model robustness by reducing variance.
Enhanced Stopping Mechanism: A custom EnhancedStopper callback monitored MAPE and R² score. It included early stopping (if no improvement for 50 rounds) to efficiently prevent overfitting and save computational resources.

Model Performance and Visualizations 📊
The XGBoost model demonstrated exceptional performance, both on the training and unseen test datasets.

Final Metrics:

Metric	Training Set	Test Set	Interpretation
MAPE	2.05%	3.37%	Predictions are, on average, within 3.37% of the actual order volumes on unseen data. This is an excellent level of accuracy for business forecasting.

R²	0.9957	0.9856

	The model explains 98.56% of the variance in order volumes on the test set, demonstrating very strong predictive power and a near-perfect fit.

Key Observations:

The model's MAPE is significantly better than typical benchmarks for order forecasting.
The minimal difference between training and test performance (a gap of only ~1.3% in MAPE) signifies robust generalization and minimal overfitting.
R² values consistently above 0.98 confirm the model effectively captures the underlying patterns driving order volumes.
Diagnostic Plots:
Visualizations were used to confirm the model's accuracy and inspect its behavior:

Actual vs. Predicted Orders Plot:
A tight clustering of points around the ideal 45-degree line visually confirms the high accuracy of the model, showing that predictions closely match actuals.

##Model Deployment and Live Application 🚀

To make the forecasting model accessible and interactive, it has been deployed as a web application using Streamlit. The pre-trained XGBoost model was saved from the Kaggle notebook and then integrated into a Python script (app01.py) for deployment.

Streamlit App: You can access the live Streamlit application here: https://rohlik-orders-forecasting-challenge-htxwnjy5vm3sjudwkhsz5t.streamlit.app/

GitHub Repository: The source code for the Streamlit application (app01.py) and other project files are available on GitHub: https://github.com/REASCREH/Rohlik-Orders-Forecasting-Challenge/blob/main/app01.py

Kaggle Notebook: The complete training and feature engineering process is detailed in this Kaggle notebook

 :https://www.kaggle.com/code/qamarmath/fork-of-rohlik-orders-forecasting-xgboost-with-da








