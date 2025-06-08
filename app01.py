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

# Update these paths to your local files
TRAIN_DATA_PATH = "train (8).csv"
TEST_DATA_PATH = "test (2).csv"
MODEL_PATH = "xgboost_model.joblib"

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
    train_eda = pd.read_csv(TRAIN_DATA_PATH, index_col='id')
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
    try:
        model = joblib.load(MODEL_PATH)
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
plt.title('Daily Orders Over Time by Warehouse 📈', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig1)

# Distribution Analysis
st.subheader("Distribution Analysis 📈")
col1, col2 = st.columns(2)
with col1:
    fig2 = plt.figure(figsize=(10, 6))
    sns.histplot(train_eda['orders'], kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Orders 📊', fontsize=16)
    plt.xlabel('Number of Orders', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y')
    plt.tight_layout()
    st.pyplot(fig2)
with col2:
    fig3 = plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_eda, x='warehouse', y='orders', palette='viridis')
    plt.title('Orders Distribution by Warehouse 🏠', fontsize=16)
    plt.xlabel('Warehouse', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig3)

# Temporal Patterns
st.subheader("Temporal Patterns 🗓️")
col1, col2 = st.columns(2)
with col1:
    fig4 = plt.figure(figsize=(12, 7))
    sns.boxplot(data=train_eda, x='day_of_week', y='orders', palette='pastel')
    plt.title('Orders Distribution by Day of Week 🗓️', fontsize=16)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    st.pyplot(fig4)
with col2:
    fig5 = plt.figure(figsize=(12, 7))
    sns.lineplot(data=train_eda.groupby(['month', 'warehouse'])['orders'].mean().reset_index(),
                 x='month', y='orders', hue='warehouse', marker='o', palette='tab10')
    plt.title('Average Monthly Orders by Warehouse 🗓️', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Number of Orders', fontsize=12)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig5)

# Geographical Analysis
st.subheader("Geographical Analysis 🌍")
fig6 = plt.figure(figsize=(10, 6))
avg_orders = train_eda.groupby('country')['orders'].mean().sort_values(ascending=False)
sns.barplot(x=avg_orders.index, y=avg_orders.values, palette='magma')
plt.title('Average Daily Orders by Country 🌍', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Average Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig6)

# Orders: Holiday vs. Non-Holiday
st.subheader("Holiday vs. Non-Holiday Orders 🎁")
fig_holiday = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='holiday', y='orders', palette='viridis')
plt.title('Orders: Holiday vs. Non-Holiday 🎁', fontsize=16)
plt.xlabel('Is Holiday? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
plt.tight_layout()
st.pyplot(fig_holiday)

# Orders: Shutdown vs. Non-Shutdown
st.subheader("Shutdown vs. Non-Shutdown Orders 🛑")
fig_shutdown = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='shutdown', y='orders', palette='mako')
plt.title('Orders: Shutdown vs. Non-Shutdown 🛑', fontsize=16)
plt.xlabel('Is Shutdown? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['Non-Shutdown', 'Shutdown'])
plt.tight_layout()
st.pyplot(fig_shutdown)

# Correlation Matrix of Numerical Features
st.subheader("Correlation Matrix of Numerical Features 🔗")
numerical_cols = ['orders', 'precipitation', 'snow', 'user_activity_1', 'user_activity_2', 'mov_change']
corr_matrix = train_eda[numerical_cols].corr()
fig_corr = plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features 🔗', fontsize=16)
plt.tight_layout()
st.pyplot(fig_corr)

# Orders vs. User Activity 1
st.subheader("Orders vs. User Activity 1 📈")
fig_ua1 = plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_eda, x='user_activity_1', y='orders', hue='warehouse', alpha=0.6, palette='tab10')
plt.title('Orders vs. User Activity 1 📈', fontsize=16)
plt.xlabel('User Activity 1', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig_ua1)

# Orders vs. User Activity 2
st.subheader("Orders vs. User Activity 2 📈")
fig_ua2 = plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_eda, x='user_activity_2', y='orders', hue='warehouse', alpha=0.6, palette='tab10')
plt.title('Orders vs. User Activity 2 📈', fontsize=16)
plt.xlabel('User Activity 2', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig_ua2)

# Total Orders by Warehouse
st.subheader("Total Orders by Warehouse 📦")
fig_total_orders_warehouse = plt.figure(figsize=(10, 6))
sns.barplot(data=train_eda, x='warehouse', y='orders', estimator=sum, palette='viridis')
plt.title('Total Orders by Warehouse 📦', fontsize=16)
plt.xlabel('Warehouse', fontsize=12)
plt.ylabel('Total Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_total_orders_warehouse)

# Orders: School Holidays vs. Non-School Holidays
st.subheader("Orders: School Holidays vs. Non-School Holidays 🏫")
fig_school_holidays = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='school_holidays', y='orders', palette='cividis')
plt.title('Orders: School Holidays vs. Non-School Holidays 🏫', fontsize=16)
plt.xlabel('Is School Holiday? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['Non-School Holiday', 'School Holiday'])
plt.tight_layout()
st.pyplot(fig_school_holidays)

# Orders: Blackout vs. Non-Blackout
st.subheader("Orders: Blackout vs. Non-Blackout 🌑")
fig_blackout = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='blackout', y='orders', palette='magma')
plt.title('Orders: Blackout vs. Non-Blackout 🌑', fontsize=16)
plt.xlabel('Is Blackout? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['Non-Blackout', 'Blackout'])
plt.tight_layout()
st.pyplot(fig_blackout)

# Orders vs. User Activity 1 (with Regression Line)
st.subheader("Orders vs. User Activity 1 (with Regression Line) 🚀")
fig_reg_ua1 = plt.figure(figsize=(10, 6))
sns.regplot(data=train_eda, x='user_activity_1', y='orders', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Orders vs. User Activity 1 (with Regression Line) 🚀', fontsize=16)
plt.xlabel('User Activity 1', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig_reg_ua1)

# Orders vs. User Activity 2 (with Regression Line)
st.subheader("Orders vs. User Activity 2 (with Regression Line) 🚀")
fig_reg_ua2 = plt.figure(figsize=(10, 6))
sns.regplot(data=train_eda, x='user_activity_2', y='orders', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Orders vs. User Activity 2 (with Regression Line) 🚀', fontsize=16)
plt.xlabel('User Activity 2', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig_reg_ua2)

# Distribution of Precipitation
st.subheader("Distribution of Precipitation 🌧️")
fig_precipitation = plt.figure(figsize=(10, 6))
sns.histplot(train_eda['precipitation'], kde=True, bins=20, color='teal')
plt.title('Distribution of Precipitation 🌧️', fontsize=16)
plt.xlabel('Precipitation (mm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
st.pyplot(fig_precipitation)

# Distribution of Snow
st.subheader("Distribution of Snow ❄️")
fig_snow = plt.figure(figsize=(10, 6))
sns.histplot(train_eda['snow'], kde=True, bins=20, color='lightsteelblue')
plt.title('Distribution of Snow ❄️', fontsize=16)
plt.xlabel('Snow (mm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
st.pyplot(fig_snow)

# Orders vs. Precipitation (with Regression Line)
st.subheader("Orders vs. Precipitation (with Regression Line) ☔")
fig_reg_precipitation = plt.figure(figsize=(10, 6))
sns.regplot(data=train_eda, x='precipitation', y='orders', scatter_kws={'alpha':0.6}, line_kws={'color':'darkgreen'})
plt.title('Orders vs. Precipitation (with Regression Line) ☔', fontsize=16)
plt.xlabel('Precipitation (mm)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig_reg_precipitation)

# Orders vs. Snow (with Regression Line)
st.subheader("Orders vs. Snow (with Regression Line) 🌨️")
fig_reg_snow = plt.figure(figsize=(10, 6))
sns.regplot(data=train_eda, x='snow', y='orders', scatter_kws={'alpha':0.6}, line_kws={'color':'purple'})
plt.title('Orders vs. Snow (with Regression Line) 🌨️', fontsize=16)
plt.xlabel('Snow (mm)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True)
plt.tight_layout()
st.pyplot(fig_reg_snow)

# Orders: Mini Shutdown vs. Non-Mini Shutdown
st.subheader("Orders: Mini Shutdown vs. Non-Mini Shutdown ⚙️")
fig_mini_shutdown = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='mini_shutdown', y='orders', palette='rocket')
plt.title('Orders: Mini Shutdown vs. Non-Mini Shutdown ⚙️', fontsize=16)
plt.xlabel('Is Mini Shutdown? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['Non-Mini Shutdown', 'Mini Shutdown'])
plt.tight_layout()
st.pyplot(fig_mini_shutdown)

# Orders: Shops Closed vs. Non-Shops Closed
st.subheader("Orders: Shops Closed vs. Non-Shops Closed 🏬")
fig_shops_closed = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='shops_closed', y='orders', palette='cool')
plt.title('Orders: Shops Closed vs. Non-Shops Closed 🏬', fontsize=16)
plt.xlabel('Are Shops Closed? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['Shops Open', 'Shops Closed'])
plt.tight_layout()
st.pyplot(fig_shops_closed)

# Orders: Winter School Holidays vs. Non-Winter School Holidays
st.subheader("Orders: Winter School Holidays vs. Non-Winter School Holidays ☃️")
fig_winter_school_holidays = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='winter_school_holidays', y='orders', palette='cubehelix')
plt.title('Orders: Winter School Holidays vs. Non-Winter School Holidays ☃️', fontsize=16)
plt.xlabel('Is Winter School Holiday? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['Normal Days', 'Winter School Holiday'])
plt.tight_layout()
st.pyplot(fig_winter_school_holidays)

# Orders: Frankfurt Shutdown vs. Non-Frankfurt Shutdown
st.subheader("Orders: Frankfurt Shutdown vs. Non-Frankfurt Shutdown 🇩🇪")
fig_frankfurt_shutdown = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='frankfurt_shutdown', y='orders', palette='crest')
plt.title('Orders: Frankfurt Shutdown vs. Non-Frankfurt Shutdown 🇩🇪', fontsize=16)
plt.xlabel('Is Frankfurt Shutdown? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['Not Shutdown', 'Shutdown'])
plt.tight_layout()
st.pyplot(fig_frankfurt_shutdown)

# Orders: MOV Change vs. No MOV Change
st.subheader("Orders: MOV Change vs. No MOV Change 💰")
fig_mov_change = plt.figure(figsize=(8, 6))
sns.boxplot(data=train_eda, x='mov_change', y='orders', palette='flare')
plt.title('Orders: MOV Change vs. No MOV Change 💰', fontsize=16)
plt.xlabel('Is MOV Change? (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks([0, 1], ['No Change', 'Change'])
plt.tight_layout()
st.pyplot(fig_mov_change)

# Orders per Day of Week by Warehouse
st.subheader("Orders per Day of Week by Warehouse 🗓️🏠")
g = sns.catplot(data=train_eda, x='day_of_week', y='orders', col='warehouse',
                kind='box', col_wrap=2, height=5, aspect=1.2, palette='Set3', sharey=True)
g.set_axis_labels("Day of Week", "Number of Orders")
g.set_titles("Warehouse: {col_name}")
g.set_xticklabels(rotation=45, ha='right')
g.fig.suptitle('Orders Distribution by Day of Week for Each Warehouse 🗓️🏠', y=1.02, fontsize=16)
plt.tight_layout()
st.pyplot(g)


# Section 2: Model Predictions
st.header("2. Model Predictions on Test Data 🔮")
st.write("Using a pre-trained XGBoost model to generate predictions on unseen test data.")
model = load_model()

if model is not None:
    with st.spinner('Preparing test data for predictions...'):
        test_data, original_test_df = load_and_preprocess_model_data(TRAIN_DATA_PATH, TEST_DATA_PATH, expected_features=TRAIN_FEATURES)

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
