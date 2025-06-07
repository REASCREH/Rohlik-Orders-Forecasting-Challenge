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
# Example: "https://raw.githubusercontent.com/your_username/your_repo/main/xgboost_model.joblib"
MODEL_URL = "https://github.com/mianhamzaashraf/Rohlik-Orders-Forecasting-Challenge/raw/main/xgboost_model.joblib" # Assuming this is your correct URL
LOCAL_MODEL_PATH = "xgboost_model.joblib"

# --- CRITICAL FIX: MANUALLY DEFINE TRAIN_FEATURES BASED ON YOUR TRAINING NOTEBOOK OUTPUT ---
# This list MUST exactly match the columns (excluding 'orders') that your model was trained on, in the correct order.
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
    train_eda = pd.read_csv(r'train (8).csv', index_col='id')
    st.subheader("\U0001F4CB Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 5 rows:**")
        st.dataframe(train_eda.head())
    with col2:
        st.write("**Basic Statistics:**")
        st.dataframe(train_eda.describe())

    st.subheader("\U0001F50D Missing Values Analysis")
    missing_data = train_eda.isnull().sum().to_frame(name="Missing Values")
    missing_data["Percentage"] = (missing_data["Missing Values"] / len(train_eda)) * 100
    st.dataframe(missing_data.sort_values(by="Percentage", ascending=False))

    st.subheader("\U0001F4C2 Column Information")
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

    # Combine for consistent preprocessing, preserving original indices
    # We will use the full `train` and `test` data here initially to ensure all
    # original columns are available for mapping/feature creation.
    data = pd.concat([train.drop(columns=[target_column]) if target_column in train.columns else train, test], axis=0)

    # Fill NaNs for 'holiday_name' early
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

        # Ensure 'holiday' and 'shops_closed' exist before groupby transforms
        if 'holiday' in Df.columns:
            Df['total_holidays_month'] = Df.groupby(['year', 'month'])['holiday'].transform('sum')
        else:
            Df['total_holidays_month'] = 0 # Default if 'holiday' not available

        if 'shops_closed' in Df.columns:
            Df['total_shops_closed_week'] = Df.groupby(['year', 'week'])['shops_closed'].transform('sum')
        else:
            Df['total_shops_closed_week'] = 0 # Default if 'shops_closed' not available

        # **IMPORTANT**: Remove or comment out features that were NOT in your training data
        # Based on your error message, these were not in your training features:
        # 'precipitation', 'snow', 'user_activity_1', 'user_activity_2',
        # 'mov_change', 'shutdown', 'blackout', 'mini_shutdown', 'frankfurt_shutdown'
        # If these columns *were* in your raw `train (8).csv` but your model was trained
        # *after dropping them*, then ensure they are dropped here as well.
        # Given your `train.head` output, they are not present in your final training features.
        # So, we should ensure they are not carried forward into the processed data if they somehow appear.
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
        # Fit on combined data to ensure all vocabulary is learned
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

    data['holiday_before'] = data['holiday'].shift(1).fillna(0).astype(int)
    data['holiday_after'] = data['holiday'].shift(-1).fillna(0).astype(int)

    categorical_columns = ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays', 'day_of_week', 'month_name', 'warehouse', 'country']
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype('category')

    # One-hot encode the categorical columns
    data = pd.get_dummies(data)

    # Now, split the combined and processed data back into train and test parts
    test_processed = data.loc[test.index].copy()

    # If `expected_features` are provided (which they should be from TRAIN_FEATURES),
    # then strictly align the test data to those features.
    if expected_features is not None:
        # Create a new DataFrame with all expected features, initialized to 0
        aligned_test_data = pd.DataFrame(0, index=test_processed.index, columns=expected_features)

        # Copy values for features that exist in the processed test data
        common_cols = list(set(test_processed.columns) & set(expected_features))
        aligned_test_data[common_cols] = test_processed[common_cols]

        # Ensure the order is exactly as expected
        aligned_test_data = aligned_test_data[expected_features]
        return aligned_test_data
    else:
        st.error("Error: Expected features list (TRAIN_FEATURES) was not provided to preprocessing. Cannot ensure alignment.")
        return None # Return None or raise an error to prevent further issues


@st.cache_resource
def load_model():
    global TRAIN_FEATURES # This global variable will be set manually based on your training data
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info(f"Downloading model from {MODEL_URL}...")
        try:
            with urlopen(MODEL_URL) as response, open(LOCAL_MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}. Please ensure the MODEL_URL is correct and accessible.")
            return None
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully!")
        # The TRAIN_FEATURES are now manually set, so we don't need to try retrieving them from the model
        # However, for debugging, you might still want to check if the model's internal features match TRAIN_FEATURES
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            if set(model.feature_names_in_) != set(TRAIN_FEATURES):
                st.warning("Model's internal feature names differ from the manually defined TRAIN_FEATURES. This is critical!")
                st.write("Model features (first 5):", model.feature_names_in_[:5].tolist())
                st.write("TRAIN_FEATURES (first 5):", TRAIN_FEATURES[:5])
        elif hasattr(model, 'get_booster') and model.get_booster().feature_names is not None:
            if set(model.get_booster().feature_names) != set(TRAIN_FEATURES):
                 st.warning("Model (Booster)'s internal feature names differ from the manually defined TRAIN_FEATURES. This is critical!")
                 st.write("Model Booster features (first 5):", model.get_booster().feature_names[:5])
                 st.write("TRAIN_FEATURES (first 5):", TRAIN_FEATURES[:5])
        else:
            st.info("Model feature names not directly accessible for comparison, relying on manual TRAIN_FEATURES definition.")

        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

st.set_page_config(layout="wide", page_title="Rohlik Orders Forecasting Analysis")
st.title("Rohlik Orders Forecasting Analysis \U0001F4CA")
st.write("This application provides comprehensive EDA of the dataset and predictions using a pre-trained XGBoost model.")

st.header("1. Exploratory Data Analysis (EDA) \U0001F50D")
train_eda = load_and_preprocess_eda_data()

st.subheader("\U0001F4C8 Time Series Analysis")
fig1 = plt.figure(figsize=(15, 7))
sns.lineplot(data=train_eda, x='date', y='orders', hue='warehouse', marker='o')
plt.title('Daily Orders Over Time by Warehouse')
plt.grid(True)
st.pyplot(fig1)

st.subheader("\U0001F4CA Distribution Analysis")
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

st.subheader("\U0001F5D3️ Temporal Patterns")
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

st.subheader("\U0001F30D Geographical Analysis")
fig6 = plt.figure(figsize=(10, 6))
avg_orders = train_eda.groupby('country')['orders'].mean().sort_values(ascending=False)
sns.barplot(x=avg_orders.index, y=avg_orders.values, palette='magma')
plt.title('Average Daily Orders by Country')
st.pyplot(fig6)

st.header("2. Model Predictions on Test Data \U0001F52E")
st.write("Using a pre-trained XGBoost model to generate predictions on unseen test data.")
model = load_model()

if model is not None:
    with st.spinner('Preparing test data for predictions...'):
        # Pass the original train and test paths, and the expected features from TRAIN_FEATURES
        # It's crucial that TRAIN_FEATURES is correctly defined at the top of the script
        test_data = load_and_preprocess_model_data(r'train (8).csv', r'test (2).csv', expected_features=TRAIN_FEATURES)

    if test_data is not None and (len(test_data.columns) != len(TRAIN_FEATURES) or any(test_data.columns != pd.Index(TRAIN_FEATURES))):
        st.warning("Warning: The preprocessed test data columns still do not exactly match the model's expected features after alignment. This is critical!")
        st.write("Test Data Columns (first 5):", test_data.columns.tolist()[:5])
        st.write("Model Expected Features (first 5):", TRAIN_FEATURES[:5])
        missing_in_test = set(TRAIN_FEATURES) - set(test_data.columns)
        extra_in_test = set(test_data.columns) - set(TRAIN_FEATURES)
        if missing_in_test:
            st.error(f"Missing in test data: {missing_in_test}")
        if extra_in_test:
            st.error(f"Extra in test data: {extra_in_test}")
        if not missing_in_test and not extra_in_test:
            st.warning("Columns match, but order is different. This can still cause issues.")
            st.write("Test Data Columns:", test_data.columns.tolist())
            st.write("Model Expected Features:", TRAIN_FEATURES)


    try:
        if test_data is None: # Handle case where preprocessing failed
            st.error("Preprocessing of test data failed. Cannot make predictions.")
        else:
            if isinstance(model, xgb.Booster): # Raw XGBoost Booster
                dmatrix = xgb.DMatrix(test_data)
                predictions = model.predict(dmatrix)
            else: # scikit-learn API wrapper
                predictions = model.predict(test_data)

            st.subheader("\U0001F4CA Prediction Results")
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

            st.subheader("\u2728 Feature Importance")
            if TRAIN_FEATURES: # Check if TRAIN_FEATURES is not empty
                # For plotting importance, ensure the model has feature names associated
                if hasattr(model, 'feature_names_in_'): # Scikit-learn API
                    fig9 = plt.figure(figsize=(12, 8))
                    plot_importance(model, max_num_features=15)
                    plt.title('Top 15 Important Features')
                    st.pyplot(fig9)
                elif isinstance(model, xgb.Booster) and model.feature_names is not None: # Raw Booster
                    fig9 = plt.figure(figsize=(12, 8))
                    plot_importance(model, max_num_features=15, importance_type='weight')
                    plt.title('Top 15 Important Features')
                    st.pyplot(fig9)
                else:
                    st.warning("Feature importance plot cannot be generated as feature names are not readily available in the model object.")
            else:
                st.warning("Feature importance plot cannot be generated as model feature names (TRAIN_FEATURES) were not set.")

            st.subheader("\U0001F333 Example Decision Tree")
            st.info("Decision tree visualization is currently commented out due to potential large size/complexity. Uncomment if needed.")

            st.subheader("\U0001F4C2 Download Predictions")
            predictions_df = pd.DataFrame({'id': test.index, 'predicted_orders': predictions}) # Use original test index
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='rohlik_order_predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please ensure that the columns of your preprocessed test data exactly match the features the model was trained on. Check the console for full error details.")

st.success("Analysis complete!")
