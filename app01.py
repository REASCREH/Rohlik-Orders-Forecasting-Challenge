import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance, plot_tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import streamlit as st
import joblib
import os
from urllib.request import urlopen
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

plt.style.use('ggplot')

# Model configuration
MODEL_URL = "https://github.com/mianhamzaashraf/Rohlik-Orders-Forecasting-Challenge/raw/main/xgboost_model.joblib"
LOCAL_MODEL_PATH = "xgboost_model.joblib"

TRAIN_FEATURES = [
    'year', 'day', 'month', 'week', 'year_sin', 'year_cos', 'month_sin',
    'month_cos', 'day_sin', 'day_cos', 'group', 'total_holidays_month',
    'total_shops_closed_week', 'group_sin', 'group_cos',
    'holiday_name_tfidf_0', 'holiday_name_tfidf_1', 'holiday_name_tfidf_2',
    'holiday_name_tfidf_3', 'holiday_name_tfidf_4', 'holiday_name_tfidf_5',
    'holiday_name_tfidf_6', 'holiday_name_tfidf_7', 'holiday_name_tfidf_8',
    'holiday_name_tfidf_9', 'holiday_before', 'holiday_after',
    'warehouse_Brno_1', 'warehouse_Budapest_1', 'warehouse_Frankfurt_1',
    'warehouse_Munich_1', 'warehouse_Prague_1', 'warehouse_Prague_2',
    'warehouse_Prague_3', 'holiday_0', 'holiday_1', 'shops_closed_0',
    'shops_closed_1', 'winter_school_holidays_0', 'winter_school_holidays_1',
    'school_holidays_0', 'school_holidays_1', 'month_name_April',
    'month_name_August', 'month_name_December', 'month_name_February',
    'month_name_January', 'month_name_July', 'month_name_June',
    'month_name_March', 'month_name_May', 'month_name_November',
    'month_name_October', 'month_name_September', 'day_of_week_Friday',
    'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday',
    'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday',
    'country_Czech Republic', 'country_Germany', 'country_Hungary'
]

# Helper functions (load_and_preprocess_eda_data and load_and_preprocess_model_data remain the same as before)

@st.cache_resource
def load_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info(f"Downloading model from {MODEL_URL}...")
        try:
            with urlopen(MODEL_URL) as response, open(LOCAL_MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            return None
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main Streamlit app
st.set_page_config(layout="wide", page_title="Rohlik Orders Forecasting Analysis")
st.title("Rohlik Orders Forecasting Analysis 📈")

# Section 1: EDA (same as before)
# ...

# Section 2: Model Predictions with Enhanced Visualizations
st.header("2. Model Predictions on Test Data 🔮")
model = load_model()

if model is not None:
    with st.spinner('Preparing test data for predictions...'):
        test_data, original_test_df = load_and_preprocess_model_data(
            'train (8).csv', 'test (2).csv', expected_features=TRAIN_FEATURES
        )
        
        # Add date column back for time series visualization
        original_test_df['date'] = pd.to_datetime(original_test_df['date'])
        warehouse_mapping = original_test_df['warehouse']

    if test_data is not None:
        try:
            # Make predictions
            if isinstance(model, xgb.Booster):
                dmatrix = xgb.DMatrix(test_data)
                predictions = model.predict(dmatrix)
            else:
                predictions = model.predict(test_data)
            
            # Create prediction dataframe with all relevant information
            predictions_df = pd.DataFrame({
                'id': original_test_df.index,
                'date': original_test_df['date'],
                'warehouse': warehouse_mapping,
                'predicted_orders': predictions
            })
            
            # Add country information
            city_to_country = {
                'Munich_1': 'Germany',
                'Frankfurt_1': 'Germany',
                'Budapest_1': 'Hungary',
                'Brno_1': 'Czech Republic',
                'Prague_1': 'Czech Republic',
                'Prague_2': 'Czech Republic',
                'Prague_3': 'Czech Republic'
            }
            predictions_df['country'] = predictions_df['warehouse'].map(city_to_country)
            
            # Add temporal features for visualization
            predictions_df['month'] = predictions_df['date'].dt.month
            predictions_df['day_of_week'] = predictions_df['date'].dt.day_name()
            predictions_df['week'] = predictions_df['date'].dt.isocalendar().week.astype(int)
            
            st.success("Predictions generated successfully!")
            
            # =============================================
            # COMPREHENSIVE PREDICTION VISUALIZATIONS
            # =============================================
            
            st.header("📊 Prediction Visualizations")
            
            # 1. Time Series Plot by Warehouse
            st.subheader("1. Time Series of Predictions by Warehouse")
            fig1 = plt.figure(figsize=(16, 8))
            sns.lineplot(
                data=predictions_df, 
                x='date', 
                y='predicted_orders', 
                hue='warehouse',
                style='country',
                markers=True,
                dashes=False
            )
            plt.title('Predicted Orders Over Time by Warehouse')
            plt.ylabel('Predicted Orders')
            plt.grid(True)
            st.pyplot(fig1)
            
            # 2. Distribution Plots
            st.subheader("2. Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig2 = plt.figure(figsize=(12, 6))
                sns.histplot(predictions_df['predicted_orders'], kde=True, bins=30)
                plt.title('Overall Distribution of Predicted Orders')
                st.pyplot(fig2)
                
            with col2:
                fig3 = plt.figure(figsize=(12, 6))
                stats.probplot(predictions_df['predicted_orders'], plot=plt)
                plt.title('Q-Q Plot of Predicted Orders')
                st.pyplot(fig3)
            
            # 3. Box Plots by Different Categories
            st.subheader("3. Category-wise Distribution")
            
            # 3.1 By Warehouse
            fig4 = plt.figure(figsize=(14, 6))
            sns.boxplot(
                data=predictions_df,
                x='warehouse',
                y='predicted_orders',
                palette='viridis'
            )
            plt.title('Predicted Orders Distribution by Warehouse')
            plt.xticks(rotation=45)
            st.pyplot(fig4)
            
            # 3.2 By Country
            fig5 = plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=predictions_df,
                x='country',
                y='predicted_orders',
                palette='magma'
            )
            plt.title('Predicted Orders Distribution by Country')
            st.pyplot(fig5)
            
            # 3.3 By Day of Week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            predictions_df['day_of_week'] = pd.Categorical(
                predictions_df['day_of_week'],
                categories=day_order,
                ordered=True
            )
            
            fig6 = plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=predictions_df,
                x='day_of_week',
                y='predicted_orders',
                palette='coolwarm'
            )
            plt.title('Predicted Orders Distribution by Day of Week')
            st.pyplot(fig6)
            
            # 4. Temporal Patterns
            st.subheader("4. Temporal Patterns")
            
            # 4.1 Monthly Trends
            fig7 = plt.figure(figsize=(14, 6))
            sns.lineplot(
                data=predictions_df.groupby(['month', 'warehouse'])['predicted_orders'].mean().reset_index(),
                x='month',
                y='predicted_orders',
                hue='warehouse',
                marker='o',
                markersize=8
            )
            plt.title('Average Predicted Monthly Orders by Warehouse')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            st.pyplot(fig7)
            
            # 4.2 Weekly Trends
            fig8 = plt.figure(figsize=(14, 6))
            weekly_data = predictions_df.groupby(['week', 'warehouse'])['predicted_orders'].mean().reset_index()
            sns.lineplot(
                data=weekly_data,
                x='week',
                y='predicted_orders',
                hue='warehouse'
            )
            plt.title('Weekly Predicted Orders Trend by Warehouse')
            st.pyplot(fig8)
            
            # 5. Autocorrelation Analysis
            st.subheader("5. Time Series Decomposition")
            
            # For each warehouse
            warehouses = predictions_df['warehouse'].unique()
            selected_warehouse = st.selectbox("Select warehouse for time series analysis:", warehouses)
            
            warehouse_data = predictions_df[predictions_df['warehouse'] == selected_warehouse]
            warehouse_data = warehouse_data.set_index('date').sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig9 = plt.figure(figsize=(12, 6))
                plot_acf(warehouse_data['predicted_orders'], lags=30, ax=plt.gca())
                plt.title(f'ACF for {selected_warehouse}')
                st.pyplot(fig9)
                
            with col2:
                fig10 = plt.figure(figsize=(12, 6))
                plot_pacf(warehouse_data['predicted_orders'], lags=30, ax=plt.gca())
                plt.title(f'PACF for {selected_warehouse}')
                st.pyplot(fig10)
            
            # 6. Scatter Matrix (for numeric features)
            st.subheader("6. Feature Relationships")
            
            # Create numeric features for scatter matrix
            scatter_df = predictions_df.copy()
            scatter_df['day_of_week_num'] = scatter_df['day_of_week'].cat.codes
            scatter_df['month_sin'] = np.sin(2 * np.pi * scatter_df['month'] / 12)
            scatter_df['month_cos'] = np.cos(2 * np.pi * scatter_df['month'] / 12)
            
            numeric_cols = ['predicted_orders', 'month', 'day_of_week_num', 'month_sin', 'month_cos']
            
            fig11 = plt.figure(figsize=(14, 14))
            sns.pairplot(
                scatter_df[numeric_cols],
                kind='reg',
                plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'alpha': 0.1}}
            )
            plt.suptitle('Pairwise Relationships of Numerical Features', y=1.02)
            st.pyplot(fig11)
            
            # 7. Heatmap of Correlations
            fig12 = plt.figure(figsize=(12, 8))
            corr_matrix = scatter_df[numeric_cols].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1
            )
            plt.title('Correlation Heatmap')
            st.pyplot(fig12)
            
            # 8. Top N Days Analysis
            st.subheader("7. Peak Days Analysis")
            
            n_days = st.slider("Select number of peak days to display:", 5, 50, 10)
            top_days = predictions_df.nlargest(n_days, 'predicted_orders')
            
            fig13 = plt.figure(figsize=(14, 6))
            sns.barplot(
                data=top_days,
                x='date',
                y='predicted_orders',
                hue='warehouse',
                dodge=False
            )
            plt.title(f'Top {n_days} Days with Highest Predicted Orders')
            plt.xticks(rotation=45)
            st.pyplot(fig13)
            
            # Show the actual data table
            st.dataframe(top_days[['date', 'warehouse', 'predicted_orders']].sort_values(
                'predicted_orders', ascending=False
            ).reset_index(drop=True))
            
            # Download predictions
            st.subheader("📁 Download Predictions")
            csv = predictions_df[['id', 'predicted_orders']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='rohlik_order_predictions.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error during prediction or visualization: {str(e)}")
    else:
        st.error("Test data preprocessing failed. No predictions can be made.")

st.success("Analysis complete!")
