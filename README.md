
# Rohlik-Orders-Forecasting-Challenge

 Overview

This project provides a complete solution for forecasting daily orders for Rohlik, an online grocery delivery service, using an XGBoost machine learning model. The system includes:

1-Exploratory Data Analysis (EDA) - Comprehensive visualization of the dataset

2-Machine Learning Model - Pre-trained XGBoost model for order forecasting

3-Streamlit Web Application - User-friendly interface for analysis and predictions

🏗️ Project Structure
text
rohlik-forecasting/
├── model_output/               # Saved model and plots
│   ├── xgboost_model.joblib    # Pre-trained XGBoost model
│   └── *.png                   # Visualization images
├── train.csv                   # Training dataset
├── test.csv                    # Test dataset
├── app.py                      # Streamlit application
└── README.md                   # This documentation
🚀 Key Features
1. Data Exploration

Interactive visualizations of order patterns

1-Time series analysis by warehouse

2-Holiday/special event impact analysis

3-Geographical comparisons

4-Weather impact analysis

2. Forecasting Model

XGBoost with DART booster (Dropout Additive Regression Trees)

Advanced feature engineering:

Temporal features (cyclic encoding)

Holiday effects

Warehouse-specific patterns

TF-IDF for holiday names

Custom early stopping based on MAPE and R² metrics

3. Deployment Features

Model performance monitoring

Feature importance visualization

Prediction distribution analysis

Easy download of forecast results

🔧 Technical Requirements
To run this project, you'll need:

Python 3.12+

Required Python packages:

text
streamlit==1.22.0
xgboost==1.7.5
scikit-learn==1.2.2
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
joblib==1.2.0
🛠️ Installation
Clone the repository:

bash
git clone https://github.com/yourusername/rohlik-forecasting.git
cd rohlik-forecasting

Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
pip install -r requirements.txt
🖥️ Running the Application
Start the Streamlit app with:

bash
streamlit run app.py

The application will open in your default browser at http://localhost:8501

📊 Using the Application

1. Data Exploration Section
View dataset statistics and missing values

Explore time series patterns

Compare warehouses and countries

Analyze holiday/special event impacts

2. Model Prediction Section

Load the pre-trained XGBoost model

Generate predictions on test data

View prediction distributions

Examine feature importance

Download forecast results as CSV

🧠 Model Details
Training Metrics

Train MAPE: 2.05%

Test MAPE: 3.37%

Train R²: 0.9957

Test R²: 0.9856

Key Features Used

The model uses 60+ engineered features including:

Temporal features (day, week, month with cyclic encoding)

Holiday indicators and patterns

Warehouse and country information

Derived features from holiday names (TF-IDF)

Special event flags (shutdowns, school holidays)

📈 Performance Visualization

The model provides several visualizations:

Actual vs Predicted values

Residual analysis

Error distribution

Feature importance

Learning curves

Decision tree visualization

