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

MODEL_URL = "https://github.com/your_username/your_repo/raw/main/xgboost_model.joblib" # Replace with your actual model URL
LOCAL_MODEL_PATH = "xgboost_model.joblib"

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

# Store the feature names used during training globally or pass them around
# This list needs to be populated during training and saved with the model,
# or derived from the training data before the model is saved.
# For now, I'll assume you can get them from the loaded model's feature_names_in_ attribute.
# If not, you'll need to reconstruct the training data preprocessing to get the exact columns.
TRAIN_FEATURES = None # This will be populated after model load if available

@st.cache_data
def load_and_preprocess_model_data(train_df_path, test_df_path, expected_features=None):
    train = pd.read_csv(train_df_path, index_col='id')
    test = pd.read_csv(test_df_path, index_col='id')

    # It's crucial that the data used for training and testing is processed identically
    # To achieve this, it's often best to concatenate before processing and then split.
    # We will only process the columns that are common or derived for both.

    # Identify the target variable
    target_column = 'orders'
    train_has_target = target_column in train.columns
    test_has_target = target_column in test.columns

    # Combine for consistent preprocessing
    if train_has_target and test_has_target: # This scenario is unlikely for typical train/test sets
        data = pd.concat([train, test], axis=0)
    elif train_has_target and not test_has_target:
        # We need to process both train and test to get all possible categorical levels for one-hot encoding
        # and consistent feature engineering.
        # Create a placeholder 'orders' column in test if it doesn't exist to allow concatenation.
        test_temp = test.copy()
        test_temp[target_column] = np.nan
        data = pd.concat([train, test_temp], axis=0)
    elif not train_has_target and not test_has_target: # E.g., if you're loading a preprocessed test set
        data = test.copy()
    else: # If test has target but train doesn't (unlikely)
        raise ValueError("Invalid train/test target column configuration.")

    data['holiday_name'] = data['holiday_name'].fillna('None')
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

        # Handle 'group' and related sin/cos transformations carefully for potential new values in test set
        min_year = Df['year'].min()
        Df['group'] = (Df['year'] - min_year) * 48 + Df['month'] * 4 + Df['day'] // 7
        max_group = Df['group'].max()
        if max_group > 0: # Ensure division by zero is avoided if all group values are 0
            Df['group_sin'] = np.sin(2 * np.pi * Df['group'] / max_group)
            Df['group_cos'] = np.cos(2 * np.pi * Df['group'] / max_group)
        else:
            Df['group_sin'] = 0
            Df['group_cos'] = 0

        # These transformations rely on groupby, which should be consistent across train and test
        Df['total_holidays_month'] = Df.groupby(['year', 'month'])['holiday'].transform('sum')
        Df['total_shops_closed_week'] = Df.groupby(['year', 'week'])['shops_closed'].transform('sum')
        return Df

    data = Process_Date(data)

    # Ensure all required columns for training are present in the final dataset.
    # The `required_cols` list should reflect all features used in your *trained* model.
    required_cols_base = ['warehouse', 'holiday_name', 'holiday', 'shops_closed',
                          'winter_school_holidays', 'school_holidays', 'year', 'day', 'month',
                          'month_name', 'day_of_week', 'week', 'year_sin', 'year_cos',
                          'month_sin', 'month_cos', 'day_sin', 'day_cos', 'group',
                          'total_holidays_month', 'total_shops_closed_week', 'group_sin',
                          'group_cos', 'country', 'precipitation', 'snow', 'user_activity_1',
                          'user_activity_2', 'mov_change', 'shutdown', 'blackout', 'mini_shutdown',
                          'frankfurt_shutdown']

    # Add 'orders' only if it exists in the original data or is a placeholder
    if target_column in data.columns:
        required_cols_base.append(target_column)

    data = data[list(set(required_cols_base).intersection(data.columns))]

    def apply_tfidf_svd(df, text_column, max_features=1000, n_components=10, fitted_vectorizer=None, fitted_svd=None):
        df[text_column] = df[text_column].fillna('')

        # For consistent TF-IDF and SVD, you should fit on training data and transform on test data.
        # If you're combining train/test before, fit_transform on the combined.
        # For prediction, it's better to load pre-fitted objects.
        if fitted_vectorizer and fitted_svd:
            vectors = fitted_vectorizer.transform(df[text_column])
            x_sv = fitted_svd.transform(vectors)
        else:
            # This path is for initial data prep. For prediction, it's not ideal unless
            # the combined data ensures all training-time vocabulary is present.
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

    data = apply_tfidf_svd(data, 'holiday_name') # No pre-fitted vectorizer/svd here, assumes combined data handles this
    data = data.drop(columns=['holiday_name'])
    if 'date' in data.columns:
        data = data.drop(columns=['date'])

    # Shift operations: Be careful. If you're processing train and test separately
    # (which you seem to be doing by concatenating then splitting based on 'orders' nullity),
    # then these shifts will not be consistent at the train/test split boundary.
    # Ideally, perform these shifts *after* processing train/test separately if the boundary is important.
    # For now, keeping as is, assuming a single continuous time series processing.
    data['holiday_before'] = data['holiday'].shift(1).fillna(0).astype(int)
    data['holiday_after'] = data['holiday'].shift(-1).fillna(0).astype(int)

    categorical_columns = ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays', 'day_of_week', 'month_name', 'warehouse', 'country']
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype('category')

    # Perform one-hot encoding
    # Crucially, ensure the categories for one-hot encoding are consistent with training.
    # A robust way is to fit the OneHotEncoder on the combined (train+test) categorical data
    # or save the encoder from training and load it.
    # For simplicity, pd.get_dummies will generate columns based on categories present in `data`.
    # This might lead to missing columns if a category present in training is not in test, and vice versa.
    # You MUST ensure consistent columns after this step.
    data = pd.get_dummies(data)

    test_processed = data[data[target_column].isnull()]
    test_processed = test_processed.drop(columns=[target_column])

    # Now, explicitly align columns with the expected features from the model
    if expected_features is not None:
        # Add missing columns (fill with 0, or appropriate default)
        missing_in_test = set(expected_features) - set(test_processed.columns)
        for col in missing_in_test:
            test_processed[col] = 0

        # Drop extra columns
        extra_in_test = set(test_processed.columns) - set(expected_features)
        test_processed = test_processed.drop(columns=list(extra_in_test))

        # Ensure order is consistent
        test_processed = test_processed[expected_features]

    return test_processed

@st.cache_resource
def load_model():
    global TRAIN_FEATURES
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info(f"Downloading model from {MODEL_URL}...")
        try:
            with urlopen(MODEL_URL) as response, open(LOCAL_MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}. Please check the MODEL_URL.")
            return None
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully!")
        # Store feature names if available in the model
        if hasattr(model, 'feature_names_in_'):
            TRAIN_FEATURES = model.feature_names_in_.tolist()
        elif hasattr(model, 'get_booster'): # For raw XGBoost Booster objects
            TRAIN_FEATURES = model.get_booster().feature_names
        else:
            st.warning("Could not automatically retrieve feature names from the loaded model. Ensure your preprocessing aligns.")
            # If feature names cannot be retrieved, you'll need to manually define them
            # based on your training data's final column set.
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
        # Pass the original train and test paths, and the expected features from the model
        test_data = load_and_preprocess_model_data(r'train (8).csv', r'test (2).csv', expected_features=TRAIN_FEATURES)

    try:
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
        # Ensure plot_importance can access feature names correctly
        if hasattr(model, 'feature_names_in_'):
            fig9 = plt.figure(figsize=(12, 8))
            plot_importance(model, max_num_features=15)
            plt.title('Top 15 Important Features')
            st.pyplot(fig9)
        elif isinstance(model, xgb.Booster) and model.feature_names is not None:
            # For raw Booster, feature_names should be set or pass feature_names to plot_importance
            fig9 = plt.figure(figsize=(12, 8))
            plot_importance(model, max_num_features=15, importance_type='weight') # You might need to specify importance_type
            plt.title('Top 15 Important Features')
            st.pyplot(fig9)
        else:
            st.warning("Feature importance plot cannot be generated as feature names are not available in the model.")


        st.subheader("\U0001F333 Example Decision Tree")
        # Plotting a single tree might be very large, consider only for small trees or specific cases
        # fig10 = plt.figure(figsize=(20, 10))
        # plot_tree(model, num_trees=0)
        # plt.title('First Tree in the Model')
        # st.pyplot(fig10)
        st.info("Decision tree visualization is currently commented out due to potential large size/complexity. Uncomment if needed.")

        st.subheader("\U0001F4C2 Download Predictions")
        predictions_df = pd.DataFrame({'id': test_data.index, 'predicted_orders': predictions})
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='rohlik_order_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please ensure that the columns of your preprocessed test data exactly match the features the model was trained on.")

st.success("Analysis complete!")
