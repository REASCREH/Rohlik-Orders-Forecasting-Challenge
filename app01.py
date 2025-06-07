import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import os
from urllib.request import urlopen

plt.style.use('ggplot')

MODEL_URL = "https://github.com/mianhamzaashraf/Rohlik-Orders-Forecasting-Challenge/raw/main/xgboost_model.joblib"
LOCAL_MODEL_PATH = "xgboost_model.joblib"

TRAIN_FEATURES = [
    # (same list as you provided)
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
    missing_data = train_eda.isnull().sum().to_frame("Missing Values")
    missing_data["Percentage"] = (missing_data["Missing Values"] / len(train_eda)) * 100
    st.dataframe(missing_data.sort_values("Percentage", ascending=False))

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
    train_eda['day_of_week'] = pd.Categorical(train_eda['day_of_week'],
                                              categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                                              ordered=True)
    return train_eda

@st.cache_data
def load_and_preprocess_model_data(train_df_path, test_df_path, expected_features=None):
    train = pd.read_csv(train_df_path, index_col='id')
    test = pd.read_csv(test_df_path, index_col='id')
    data = pd.concat([train.drop(columns=['orders'], errors='ignore'), test], axis=0)
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

    def process_date(df):
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        df['year_sin'] = np.sin(2 * np.pi * df['year'])
        df['year_cos'] = np.cos(2 * np.pi * df['year'])
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        min_year = df['year'].min()
        df['group'] = (df['year'] - min_year) * 48 + df['month'] * 4 + df['day'] // 7
        max_group = df['group'].max() or 1
        df['group_sin'] = np.sin(2 * np.pi * df['group'] / max_group)
        df['group_cos'] = np.cos(2 * np.pi * df['group'] / max_group)

        df['total_holidays_month'] = df.groupby(['year', 'month'])['holiday'].transform('sum') if 'holiday' in df.columns else 0
        df['total_shops_closed_week'] = df.groupby(['year', 'week'])['shops_closed'].transform('sum') if 'shops_closed' in df.columns else 0

        # drop unwanted columns if present...
        for col in ['date', 'quarter', 'quarter_sin', 'quarter_cos']:
            df.drop(columns=[col], inplace=True, errors='ignore')

        return df

    data = process_date(data)

    def apply_tfidf_svd(df):
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        vectors = vectorizer.fit_transform(df['holiday_name'])
        svd = TruncatedSVD(n_components=10)
        tfidf_df = pd.DataFrame(svd.fit_transform(vectors),
                                 index=df.index,
                                 columns=[f'holiday_name_tfidf_{i}' for i in range(10)])
        return pd.concat([df.drop(columns=['holiday_name']), tfidf_df], axis=1)

    data = apply_tfidf_svd(data)

    data['holiday_before'] = data['holiday'].shift(1).fillna(0).astype(int)
    data['holiday_after'] = data['holiday'].shift(-1).fillna(0).astype(int)

    categorical_columns = ['holiday', 'shops_closed', 'winter_school_holidays',
                           'school_holidays', 'day_of_week', 'month_name', 'warehouse', 'country']
    for col in categorical_columns:
        if col in data:
            data[col] = data[col].astype('category')

    data = pd.get_dummies(data)

    test_processed = data.loc[test.index]

    if expected_features:
        aligned = pd.DataFrame(0, index=test_processed.index, columns=expected_features)
        common = test_processed.columns.intersection(expected_features)
        aligned[common] = test_processed[common]
        return aligned[expected_features], test

    return None, test

@st.cache_resource
def load_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info(f"Downloading model from {MODEL_URL}...")
        try:
            with urlopen(MODEL_URL) as r, open(LOCAL_MODEL_PATH, 'wb') as f:
                f.write(r.read())
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None

    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

st.set_page_config(layout="wide", page_title="Rohlik Orders Forecasting Analysis")
st.title("Rohlik Orders Forecasting Analysis 📈")

# 1. EDA
train_eda = load_and_preprocess_eda_data()
# Build your lineplots, histograms, etc. as in your original code...

# 2. Model Prediction
model = load_model()
if model:
    :contentReference[oaicite:1]{index=1}
        :contentReference[oaicite:2]{index=2}
    )
    :contentReference[oaicite:3]{index=3}
        :contentReference[oaicite:4]{index=4}

        try:
            dmatrix = xgb.DMatrix(test_data) if isinstance(model, xgb.Booster) else None
            predictions = model.predict(dmatrix) if dmatrix else model.predict(test_data)

            st.subheader("📊 Predictions Overview")
            st.dataframe(pd.Series(predictions).describe().to_frame("Predictions"))

            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            sns.histplot(predictions, kde=True, ax=axs[0], color='purple')
            axs[0].set_title("Predicted Orders Distribution")
            axs[1].plot(predictions, 'o', alpha=0.5)
            axs[1].set_title("Predicted Orders Sequence")
            axs[1].set_xlabel("Index")
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("✨ Feature Importance")
            fi_fig = plt.figure(figsize=(12, 8))
            plot_importance(model, max_num_features=15)
            st.pyplot(fi_fig)

            st.subheader("🌳 Example Tree")
            tree_fig = plt.figure(figsize=(20, 10))
            plot_tree(model, num_trees=0, rankdir='LR', ax=tree_fig.gca())
            st.pyplot(tree_fig)

            # Download CSV
            out_df = pd.DataFrame({
                "id": original_test_df.index,
                "predicted_orders": predictions
            })
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download predictions as CSV",
                data=csv,
                file_name="rohlik_order_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Preprocessing test data failed.")
else:
    :contentReference[oaicite:5]{index=5}

:contentReference[oaicite:6]{index=6}
