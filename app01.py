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

plt.style.use('ggplot')

# IMPORTANT: Replace this with the actual raw URL to your trained model on GitHub
MODEL_URL = "https://github.com/mianhamzaashraf/Rohlik-Orders-Forecasting-Challenge/raw/main/xgboost_model.joblib"
LOCAL_MODEL_PATH = "xgboost_model.joblib"

# Manually defined TRAIN_FEATURES based on training notebook output
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

@st.cache_data
def load_and_preprocess_eda_data():
    train_eda = pd.read_csv('train (8).csv', index_col='id')
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 5 rows:**")
        st.dataframe(train_eda.head())
    with col2:
        st.write("**Basic Statistics:**")
        st.dataframe(train_eda.describe())

    st.subheader("🔍 Missing Values Analysis")
    missing_data = train_eda.isnull().sum().to_frame(name="Missing Values")
    missing_data["Percentage"] = (missing_data["Missing Values"] / len(train_eda)) * 100
    st.dataframe(missing_data.sort_values(by="Percentage", ascending=False))

    st.subheader("📄 Column Information")
    col_info = pd.DataFrame({
        'Column': train_eda.columns,
        'Data Type': train_eda.dtypes,
        'Unique Values': [train_eda[col].nunique() for col in train_eda.columns]
    })
    st.dataframe(col_info)

    train_eda['date'] = pd.to_datetime(train_eda['date'])
    city_to_country = {
        'Munich_1': 'Germany',
        'Frankfurt_1': 'Germany',
        'Budapest_1': 'Hungary',
        'Brno_1': 'Czech Republic',
        'Prague_1': 'Czech Republic',
        'Prague_2': 'Czech Republic',
        'Prague_3': 'Czech Republic'
    }
    train_eda['country'] = train_eda['warehouse'].map(city_to_country)
    train_eda['month'] = train_eda['date'].dt.month
    train_eda['day_of_week'] = train_eda['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    train_eda['day_of_week'] = pd.Categorical(train_eda['day_of_week'], categories=day_order, ordered=True)
    return train_eda

@st.cache_data
def load_and_preprocess_model_data(train_df_path, test_df_path, expected_features=None):
    train = pd.read_csv(train_df_path, index_col='id')
    test = pd.read_csv(test_df_path, index_col='id')

    target_column = 'orders'

    # Combine for consistent preprocessing
    data = pd.concat([train.drop(columns=[target_column]) if target_column in train.columns else train, test], axis=0)

    # Fill NaNs for 'holiday_name'
    data['holiday_name'] = data['holiday_name'].fillna('None')

    # Map warehouse to country
    city_to_country = {
        'Munich_1': 'Germany',
        'Frankfurt_1': 'Germany',
        'Budapest_1': 'Hungary',
        'Brno_1': 'Czech Republic',
        'Prague_1': 'Czech Republic',
        'Prague_2': 'Czech Republic',
        'Prague_3': 'Czech Republic'
    }
    data['country'] = data['warehouse'].map(city_to_country)

    def Process_Date(Df):
        Df['date'] = pd.to_datetime(Df['date'])
        Df['year'] = Df['date'].dt.year
        Df['day'] = Df['date'].dt.day
        Df['month'] = Df['date'].dt.month
        Df['quarter'] = Df['date'].dt.quarter
        Df['month_name'] = Df['date'].dt.month_name()
        Df['day_of_week'] = Df['date'].dt.day_name()
        Df['week'] = Df['date'].dt.isocalendar().week.astype(int)
        Df['year_sin'] = np.sin(2 * np.pi * Df['year'])
        Df['year_cos'] = np.cos(2 * np.pi * Df['year'])
        Df['month_sin'] = np.sin(2 * np.pi * Df['month'] / 12)
        Df['month_cos'] = np.cos(2 * np.pi * Df['month'] / 12)
        Df['day_sin'] = np.sin(2 * np.pi * Df['day'] / 31)
        Df['day_cos'] = np.cos(2 * np.pi * Df['day'] / 31)
        Df['quarter_sin'] = np.sin(2 * np.pi * Df['quarter'] / 4)
        Df['quarter_cos'] = np.cos(2 * np.pi * Df['quarter'] / 4)

        min_year = Df['year'].min()
        Df['group'] = (Df['year'] - min_year) * 48 + Df['month'] * 4 + Df['day'] // 7
        max_group = Df['group'].max()
        if max_group > 0:
            Df['group_sin'] = np.sin(2 * np.pi * Df['group'] / max_group)
            Df['group_cos'] = np.cos(2 * np.pi * Df['group'] / max_group)
        else:
            Df['group_sin'] = 0
            Df['group_cos'] = 0

        if 'holiday' in Df.columns:
            Df['total_holidays_month'] = Df.groupby(['year', 'month'])['holiday'].transform('sum')
        else:
            Df['total_holidays_month'] = 0

        if 'shops_closed' in Df.columns:
            Df['total_shops_closed_week'] = Df.groupby(['year', 'week'])['shops_closed'].transform('sum')
        else:
            Df['total_shops_closed_week'] = 0

        cols_to_drop_if_present = [
            'precipitation', 'snow', 'user_activity_1', 'user_activity_2',
            'mov_change', 'shutdown', 'blackout', 'mini_shutdown', 'frankfurt_shutdown'
        ]
        for col in cols_to_drop_if_present:
            if col in Df.columns:
                Df = Df.drop(columns=[col])

        return Df

    data = Process_Date(data)

    def apply_tfidf_svd(df, text_column, max_features=1000, n_components=10):
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        vectors = vectorizer.fit_transform(df[text_column])
        svd = TruncatedSVD(n_components)
        x_sv = svd.fit_transform(vectors)
        tfidf_df = pd.DataFrame(x_sv)
        cols = [(text_column + "_tfidf_" + str(f)) for f in tfidf_df.columns.to_list()]
        tfidf_df.columns = cols
        tfidf_df.index = df.index
        df = pd.concat([df, tfidf_df], axis="columns")
        return df

    data = apply_tfidf_svd(data, 'holiday_name')
    data = data.drop(columns=['holiday_name'])
    if 'date' in data.columns:
        data = data.drop(columns=['date'])
    if 'quarter' in data.columns:
        data = data.drop(columns=['quarter'])
    if 'quarter_sin' in data.columns:
        data = data.drop(columns=['quarter_sin'])
    if 'quarter_cos' in data.columns:
        data = data.drop(columns=['quarter_cos'])

    data['holiday_before'] = data['holiday'].shift(1).fillna(0).astype(int)
    data['holiday_after'] = data['holiday'].shift(-1).fillna(0).astype(int)

    categorical_columns = ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays', 'day_of_week', 'month_name', 'warehouse', 'country']
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype('category')

    # One-hot encode the categorical columns
    data = pd.get_dummies(data)

    # Split back into train and test parts
    test_processed = data.loc[test.index].copy()

    if expected_features is not None:
        aligned_test_data = pd.DataFrame(0, index=test_processed.index, columns=expected_features)
        common_cols = list(set(test_processed.columns) & set(expected_features))
        aligned_test_data[common_cols] = test_processed[common_cols]
        aligned_test_data = aligned_test_data[expected_features]
        return aligned_test_data, test
    else:
        st.error("Error: Expected features list (TRAIN_FEATURES) was not provided to preprocessing.")
        return None, test

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
        
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            if set(model.feature_names_in_) != set(TRAIN_FEATURES):
                st.warning("Model's internal feature names differ from TRAIN_FEATURES!")
        elif hasattr(model, 'get_booster') and model.get_booster().feature_names is not None:
            if set(model.get_booster().feature_names) != set(TRAIN_FEATURES):
                st.warning("Model Booster's feature names differ from TRAIN_FEATURES!")
        else:
            st.info("Model feature names not directly accessible.")

        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main Streamlit app
st.set_page_config(layout="wide", page_title="Rohlik Orders Forecasting Analysis")
st.title("Rohlik Orders Forecasting Analysis 📈")
st.write("This application provides comprehensive EDA of the dataset and predictions using a pre-trained XGBoost model.")

# Section 1: Exploratory Data Analysis (EDA)
st.header("1. Exploratory Data Analysis (EDA) 🔍")
train_eda = load_and_preprocess_eda_data()

# Time Series Analysis
st.subheader("Time Series Analysis 📊")
fig1 = plt.figure(figsize=(15, 7))
sns.lineplot(data=train_eda, x='date', y='orders', hue='warehouse', marker='o')
plt.title('Daily Orders Over Time by Warehouse')
plt.grid(True)
st.pyplot(fig1)

# Distribution Analysis
st.subheader("Distribution Analysis 📈")
col1, col2 = st.columns(2)
with col1:
    fig2 = plt.figure(figsize=(10, 6))
    sns.histplot(train_eda['orders'], kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Orders')
    st.pyplot(fig2)
with col2:
    fig3 = plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_eda, x='warehouse', y='orders', palette='viridis')
    plt.title('Orders Distribution by Warehouse')
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# Temporal Patterns
st.subheader("Temporal Patterns 🗓️")
col1, col2 = st.columns(2)
with col1:
    fig4 = plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_eda, x='day_of_week', y='orders', palette='pastel')
    plt.title('Orders by Day of Week')
    st.pyplot(fig4)
with col2:
    fig5 = plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_eda.groupby(['month', 'warehouse'])['orders'].mean().reset_index(),
                 x='month', y='orders', hue='warehouse', marker='o')
    plt.title('Average Monthly Orders by Warehouse')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    st.pyplot(fig5)

# Geographical Analysis
st.subheader("Geographical Analysis 🌍")
fig6 = plt.figure(figsize=(10, 6))
avg_orders = train_eda.groupby('country')['orders'].mean().sort_values(ascending=False)
sns.barplot(x=avg_orders.index, y=avg_orders.values, palette='magma')
plt.title('Average Daily Orders by Country')
st.pyplot(fig6)

# Section 2: Model Predictions
st.header("2. Model Predictions on Test Data 🔮")
st.write("Using a pre-trained XGBoost model to generate predictions on unseen test data.")
model = load_model()

if model is not None:
    with st.spinner('Preparing test data for predictions...'):
        test_data, original_test_df = load_and_preprocess_model_data('train (8).csv', 'test (2).csv', expected_features=TRAIN_FEATURES)

    if test_data is not None:
        st.subheader("Feature Alignment Check")
        st.write(f"Number of features in test data: {len(test_data.columns)}")
        st.write(f"Number of expected features: {len(TRAIN_FEATURES)}")
        
        if set(test_data.columns) == set(TRAIN_FEATURES):
            st.success("All expected features are present!")
        else:
            st.error("Feature mismatch detected!")

        try:
            if isinstance(model, xgb.Booster):
                dmatrix = xgb.DMatrix(test_data)
                predictions = model.predict(dmatrix)
            else:
                predictions = model.predict(test_data)

            st.subheader("📊 Prediction Results")
            st.write(f"Generated predictions for {len(predictions)} test samples.")
            st.dataframe(pd.Series(predictions).describe().to_frame('Predictions'))

            col1, col2 = st.columns(2)
            with col1:
                fig7 = plt.figure(figsize=(10, 6))
                sns.histplot(predictions, kde=True, bins=30, color='purple')
                plt.title('Distribution of Predicted Orders')
                st.pyplot(fig7)
            with col2:
                fig8 = plt.figure(figsize=(10, 6))
                plt.plot(predictions, 'o', alpha=0.5)
                plt.title('Predicted Orders Sequence')
                st.pyplot(fig8)

            st.subheader("✨ Feature Importance")
            if hasattr(model, 'feature_importances_'):
                fig9 = plt.figure(figsize=(12, 8))
                plot_importance(model, max_num_features=15)
                plt.title('Top 15 Important Features')
                st.pyplot(fig9)
            else:
                st.warning("Feature importance not available for this model type")

            st.subheader("📁 Download Predictions")
            predictions_df = pd.DataFrame({'id': original_test_df.index, 'predicted_orders': predictions})
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='rohlik_order_predictions.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        st.error("Test data preprocessing failed")

st.success("Analysis complete!")
