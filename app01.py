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

MODEL_URL = "xgboost_model.joblib"
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

@st.cache_data
def load_and_preprocess_model_data():
    train = pd.read_csv(r'train (8).csv', index_col='id')
    test = pd.read_csv(r'test (2).csv', index_col='id')
    data = pd.concat([train, test], axis=0)
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
        min_year = Df['year'].min()
        Df['group'] = (Df['year'] - min_year) * 48 + Df['month'] * 4 + Df['day'] // 7
        max_group = Df['group'].max()
        if max_group > 0:
            Df['group_sin'] = np.sin(2 * np.pi * Df['group'] / max_group)
            Df['group_cos'] = np.cos(2 * np.pi * Df['group'] / max_group)
        else:
            Df['group_sin'] = 0
            Df['group_cos'] = 0
        Df['total_holidays_month'] = Df.groupby(['year', 'month'])['holiday'].transform('sum')
        Df['total_shops_closed_week'] = Df.groupby(['year', 'week'])['shops_closed'].transform('sum')
        return Df

    data = Process_Date(data)
    required_cols = ['warehouse', 'holiday_name', 'holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays', 'year', 'day', 'month', 'month_name', 'day_of_week', 'week', 'year_sin', 'year_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'group', 'total_holidays_month', 'total_shops_closed_week', 'group_sin', 'group_cos', 'country', 'precipitation', 'snow', 'user_activity_1', 'user_activity_2', 'mov_change', 'shutdown', 'blackout', 'mini_shutdown', 'frankfurt_shutdown']
    if 'orders' in data.columns:
        required_cols.append('orders')
    data = data[list(set(required_cols).intersection(data.columns))]

    def apply_tfidf_svd(df, text_column, max_features=1000, n_components=10):
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        df[text_column] = df[text_column].fillna('')
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
    data = pd.get_dummies(data)
    test_processed = data[data['orders'].isnull()]
    test_processed = test_processed.drop(columns=['orders'])
    return test_processed

@st.cache_resource
def load_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info("Downloading model from GitHub...")
        with urlopen(MODEL_URL) as response, open(LOCAL_MODEL_PATH, 'wb') as out_file:
            out_file.write(response.read())
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully!")
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
        test_data = load_and_preprocess_model_data()
    try:
        feature_names = model.feature_names_in_
    except AttributeError:
        feature_names = test_data.columns.tolist()
    missing_features = set(feature_names) - set(test_data.columns)
    for feat in missing_features:
        test_data[feat] = 0
    test_data = test_data[feature_names]

    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(test_data)
        predictions = model.predict(dmatrix)
    else:
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
    fig9 = plt.figure(figsize=(12, 8))
    plot_importance(model, max_num_features=15)
    plt.title('Top 15 Important Features')
    st.pyplot(fig9)

    st.subheader("\U0001F333 Example Decision Tree")
    fig10 = plt.figure(figsize=(20, 10))
    plot_tree(model, num_trees=0)
    plt.title('First Tree in the Model')
    st.pyplot(fig10)

    st.subheader("\U0001F4C2 Download Predictions")
    predictions_df = pd.DataFrame({'id': test_data.index, 'predicted_orders': predictions})
    csv = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='rohlik_order_predictions.csv',
        mime='text/csv'
    )

st.success("Analysis complete!")
