import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance, plot_tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import streamlit as st

# Set up matplotlib style for aesthetics
plt.style.use('ggplot')

# --- Data Loading and Initial Preprocessing for EDA plots ---
@st.cache_data
def load_and_preprocess_eda_data():
    # Adjusted path for local execution. Replace with your actual path.
    train_eda = pd.read_csv(r'train (8).csv', index_col='id') # <-- No C:\Users\Qamar\Downloads\

    train_eda['date'] = pd.to_datetime(train_eda['date'])

    # Dictionary mapping cities to their countries
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

# --- Data Loading and Preprocessing for Model Training ---
@st.cache_data
def load_and_preprocess_model_data():
    # Adjusted paths for local execution. Replace with your actual paths.

    train = pd.read_csv(r'train (8).csv', index_col='id') # <-- No C:\Users\Qamar\Downloads\
    test = pd.read_csv(r'test (2).csv', index_col='id')   # <-- No C:\Users\Qamar\Downloads\

    data = pd.concat([train, test], axis=0)
    data['holiday_name'] = data['holiday_name'].fillna('None')

    # Dictionary mapping cities to their countries
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

    required_cols = ['warehouse', 'holiday_name', 'holiday', 'shops_closed',
                     'winter_school_holidays', 'school_holidays', 'year', 'day', 'month',
                     'month_name', 'day_of_week', 'week', 'year_sin', 'year_cos',
                     'month_sin', 'month_cos', 'day_sin', 'day_cos', 'group',
                     'total_holidays_month', 'total_shops_closed_week',
                     'group_sin', 'group_cos', 'country',
                     'precipitation', 'snow', 'user_activity_1', 'user_activity_2',
                     'mov_change', 'shutdown', 'blackout', 'mini_shutdown', 'frankfurt_shutdown'] # Added missing numerical/boolean columns for model

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
        df = df.reset_index(drop=True)
        df = pd.concat([df, tfidf_df], axis="columns")
        return df

    data = apply_tfidf_svd(data, 'holiday_name')
    data = data.drop(columns=['holiday_name'])
    if 'date' in data.columns:
        data = data.drop(columns=['date'])

    data['holiday_before'] = data['holiday'].shift(1).fillna(0).astype(int)
    data['holiday_after'] = data['holiday'].shift(-1).fillna(0).astype(int)

    categorical_columns = ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays', 'day_of_week', 'month_name', 'warehouse', 'country'] # Added more categorical columns for dummification
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype('category')

    data = pd.get_dummies(data)

    train_processed = data[~data['orders'].isnull()]
    test_processed = data[data['orders'].isnull()]
    test_processed = test_processed.drop(columns=['orders'])

    return train_processed, test_processed

# --- Model Training ---
@st.cache_resource
def train_xgboost_model(train_data_df):
    target = train_data_df['orders']
    features = train_data_df.drop(columns=['orders'])

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.09, random_state=42)
    
    # Align columns - crucial for consistent feature sets
    train_cols = X_train.columns
    test_cols = X_test.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0
    X_test = X_test[train_cols] # Ensure test set has the same order of columns as train set

    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)

    params = {
        'booster': 'dart',
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'learning_rate': 0.06,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'lambda': 8.0,
        'alpha': 9.0,
        'rate_drop': 0.4,
        'skip_drop': 0.8,
        'normalize_type': 'tree',
        'sample_type': 'uniform',
        'verbosity': 0,
    }

    class EnhancedStopper(xgb.callback.TrainingCallback):
        def __init__(self, target_mape=None, rounds=100, train_dmatrix=None, test_dmatrix=None, y_train_actual=None, y_test_actual=None):
            self.target_mape = target_mape
            self.rounds = rounds
            self.history = {
                'train_mape': [], 'test_mape': [],
                'train_mae': [], 'test_mae': [],
                'train_r2': [], 'test_r2': []
            }
            self.train_dmatrix = train_dmatrix
            self.test_dmatrix = test_dmatrix
            self.y_train_actual = y_train_actual
            self.y_test_actual = y_test_actual
            
        def after_iteration(self, model, epoch, evals_log):
            if self.train_dmatrix is None or self.test_dmatrix is None:
                return False

            y_pred_train = model.predict(self.train_dmatrix)
            y_pred_test = model.predict(self.test_dmatrix)
            
            # Ensure no division by zero for MAPE
            train_mape = np.mean(np.abs((self.y_train_actual - y_pred_train) / self.y_train_actual[self.y_train_actual != 0])) * 100 if np.sum(self.y_train_actual != 0) > 0 else np.nan
            test_mape = np.mean(np.abs((self.y_test_actual - y_pred_test) / self.y_test_actual[self.y_test_actual != 0])) * 100 if np.sum(self.y_test_actual != 0) > 0 else np.nan

            train_mae = mean_absolute_error(self.y_train_actual, y_pred_train)
            test_mae = mean_absolute_error(self.y_test_actual, y_pred_test)
            train_r2 = r2_score(self.y_train_actual, y_pred_train)
            test_r2 = r2_score(self.y_test_actual, y_pred_test)

            self.history['train_mape'].append(train_mape)
            self.history['test_mape'].append(test_mape)
            self.history['train_mae'].append(train_mae)
            self.history['test_mae'].append(test_mae)
            self.history['train_r2'].append(train_r2)
            self.history['test_r2'].append(test_r2)

            return False # Continue training until early stopping rounds or num_boost_round is reached

    stopper = EnhancedStopper(target_mape=0.019, rounds=100, train_dmatrix=train_data, test_dmatrix=test_data, y_train_actual=y_train, y_test_actual=y_test)

    model = xgb.train(
        params,
        train_data,
        num_boost_round=500,
        evals=[(train_data, 'train'), (test_data, 'test')],
        early_stopping_rounds=100,
        callbacks=[stopper],
        verbose_eval=False
    )
    
    return model, X_train, X_test, y_train, y_test, stopper.history

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Rohlik Orders Forecasting Analysis")

st.title("Rohlik Orders Forecasting Analysis 📊")
st.write("This application provides a comprehensive visualization of the Rohlik orders dataset and the performance of the trained XGBoost model.")

# --- Initial EDA Plots ---
st.header("Exploratory Data Analysis (EDA) 🔎")
st.write("These plots provide insights into the raw data, helping us understand trends, distributions, and relationships before modeling.")

with st.spinner('Generating EDA plots...'):
    train_eda = load_and_preprocess_eda_data()

    ## Daily Orders Over Time by Warehouse 📈
    fig = plt.figure(figsize=(15, 7))
    sns.lineplot(data=train_eda, x='date', y='orders', hue='warehouse', marker='o')
    plt.title('Daily Orders Over Time by Warehouse 📈', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.grid(True)
    st.pyplot(fig)

    ## Distribution of Orders 📊
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(train_eda['orders'], kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Orders 📊', fontsize=16)
    plt.xlabel('Number of Orders', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y')
    st.pyplot(fig)

    ## Orders: Holiday vs. Non-Holiday 🎁
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(data=train_eda, x='holiday', y='orders', palette='viridis')
    plt.title('Orders: Holiday vs. Non-Holiday 🎁', fontsize=16)
    plt.xlabel('Is Holiday? (0: No, 1: Yes)', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
    st.pyplot(fig)

    ## Orders: Shutdown vs. Non-Shutdown 🛑
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(data=train_eda, x='shutdown', y='orders', palette='mako')
    plt.title('Orders: Shutdown vs. Non-Shutdown 🛑', fontsize=16)
    plt.xlabel('Is Shutdown? (0: No, 1: Yes)', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks([0, 1], ['Non-Shutdown', 'Shutdown'])
    st.pyplot(fig)

    ## Correlation Matrix of Numerical Features 🔗
    numerical_cols = ['orders', 'precipitation', 'snow', 'user_activity_1', 'user_activity_2', 'mov_change']
    # Ensure all numerical_cols exist in train_eda before calculating correlation
    actual_numerical_cols = [col for col in numerical_cols if col in train_eda.columns]
    
    if actual_numerical_cols: # Only plot if there are numerical columns to correlate
        corr_matrix = train_eda[actual_numerical_cols].corr()
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numerical Features 🔗', fontsize=16)
        st.pyplot(fig)
    else:
        st.write("No relevant numerical columns found for correlation matrix.")


    ## Orders vs. User Activity 1 📈
    if 'user_activity_1' in train_eda.columns:
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=train_eda, x='user_activity_1', y='orders', hue='warehouse', alpha=0.6, palette='tab10')
        plt.title('Orders vs. User Activity 1 📈', fontsize=16)
        plt.xlabel('User Activity 1', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.write("`user_activity_1` column not found for plotting.")

    ## Orders vs. User Activity 2 📈
    if 'user_activity_2' in train_eda.columns:
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=train_eda, x='user_activity_2', y='orders', hue='warehouse', alpha=0.6, palette='tab10')
        plt.title('Orders vs. User Activity 2 📈', fontsize=16)
        plt.xlabel('User Activity 2', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.write("`user_activity_2` column not found for plotting.")


    ## Average Daily Orders per Warehouse 🏠
    avg_orders_per_warehouse = train_eda.groupby('warehouse')['orders'].mean().sort_values(ascending=False)
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_orders_per_warehouse.index, y=avg_orders_per_warehouse.values, palette='Blues_d')
    plt.title('Average Daily Orders per Warehouse 🏠', fontsize=16) 
    plt.xlabel('Warehouse', fontsize=12)
    plt.ylabel('Average Number of Orders', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Added tight_layout
    st.pyplot(fig)

    ## Total Orders by Warehouse 📦
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=train_eda, x='warehouse', y='orders', estimator=sum, palette='viridis')
    plt.title('Total Orders by Warehouse 📦', fontsize=16)
    plt.xlabel('Warehouse', fontsize=12)
    plt.ylabel('Total Number of Orders', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Added tight_layout
    st.pyplot(fig)

    ## Average Monthly Orders by Warehouse 🗓️
    fig = plt.figure(figsize=(12, 7))
    sns.lineplot(data=train_eda.groupby(['month', 'warehouse'])['orders'].mean().reset_index(),
                x='month', y='orders', hue='warehouse', marker='o', palette='tab10')
    plt.title('Average Monthly Orders by Warehouse 🗓️', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Number of Orders', fontsize=12)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    plt.tight_layout() # Added tight_layout
    st.pyplot(fig)

    ## Orders Distribution by Day of Week 🗓️
    fig = plt.figure(figsize=(12, 7))
    sns.boxplot(data=train_eda, x='day_of_week', y='orders', palette='pastel')
    plt.title('Orders Distribution by Day of Week 🗓️', fontsize=16)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout() # Added tight_layout
    st.pyplot(fig)

    ## Orders: School Holidays vs. Non-School Holidays 🏫
    if 'school_holidays' in train_eda.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=train_eda, x='school_holidays', y='orders', palette='cividis')
        plt.title('Orders: School Holidays vs. Non-School Holidays 🏫', fontsize=16)
        plt.xlabel('Is School Holiday? (0: No, 1: Yes)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.xticks([0, 1], ['Non-School Holiday', 'School Holiday'])
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`school_holidays` column not found for plotting.")

    ## Orders: Blackout vs. Non-Blackout 🌑
    if 'blackout' in train_eda.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=train_eda, x='blackout', y='orders', palette='magma')
        plt.title('Orders: Blackout vs. Non-Blackout 🌑', fontsize=16)
        plt.xlabel('Is Blackout? (0: No, 1: Yes)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.xticks([0, 1], ['Non-Blackout', 'Blackout'])
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`blackout` column not found for plotting.")

    ## Orders vs. User Activity 1 (with Regression Line) 🚀
    if 'user_activity_1' in train_eda.columns:
        fig = plt.figure(figsize=(10, 6))
        sns.regplot(data=train_eda, x='user_activity_1', y='orders', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        plt.title('Orders vs. User Activity 1 (with Regression Line) 🚀', fontsize=16)
        plt.xlabel('User Activity 1', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.grid(True)
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`user_activity_1` column not found for plotting.")


    ## Orders vs. User Activity 2 (with Regression Line) 🚀
    if 'user_activity_2' in train_eda.columns:
        fig = plt.figure(figsize=(10, 6))
        sns.regplot(data=train_eda, x='user_activity_2', y='orders', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        plt.title('Orders vs. User Activity 2 (with Regression Line) 🚀', fontsize=16)
        plt.xlabel('User Activity 2', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.grid(True)
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`user_activity_2` column not found for plotting.")

    ## Distribution of Precipitation 🌧️
    if 'precipitation' in train_eda.columns:
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(train_eda['precipitation'], kde=True, bins=20, color='teal')
        plt.title('Distribution of Precipitation 🌧️', fontsize=16)
        plt.xlabel('Precipitation (mm)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y')
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`precipitation` column not found for plotting.")

    ## Distribution of Snow ❄️
    if 'snow' in train_eda.columns:
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(train_eda['snow'], kde=True, bins=20, color='lightsteelblue')
        plt.title('Distribution of Snow ❄️', fontsize=16)
        plt.xlabel('Snow (mm)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y')
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`snow` column not found for plotting.")

    ## Orders vs. Precipitation (with Regression Line) ☔
    if 'precipitation' in train_eda.columns:
        fig = plt.figure(figsize=(10, 6))
        sns.regplot(data=train_eda, x='precipitation', y='orders', scatter_kws={'alpha':0.6}, line_kws={'color':'darkgreen'})
        plt.title('Orders vs. Precipitation (with Regression Line) ☔', fontsize=16)
        plt.xlabel('Precipitation (mm)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.grid(True)
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`precipitation` column not found for plotting.")

    ## Orders vs. Snow (with Regression Line) 🌨️
    if 'snow' in train_eda.columns:
        fig = plt.figure(figsize=(10, 6))
        sns.regplot(data=train_eda, x='snow', y='orders', scatter_kws={'alpha':0.6}, line_kws={'color':'purple'})
        plt.title('Orders vs. Snow (with Regression Line) 🌨️', fontsize=16)
        plt.xlabel('Snow (mm)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.grid(True)
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`snow` column not found for plotting.")

    ## Orders: Mini Shutdown vs. Non-Mini Shutdown ⚙️
    if 'mini_shutdown' in train_eda.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=train_eda, x='mini_shutdown', y='orders', palette='rocket')
        plt.title('Orders: Mini Shutdown vs. Non-Mini Shutdown ⚙️', fontsize=16)
        plt.xlabel('Is Mini Shutdown? (0: No, 1: Yes)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.xticks([0, 1], ['Non-Mini Shutdown', 'Mini Shutdown'])
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`mini_shutdown` column not found for plotting.")

    ## Orders: Shops Closed vs. Non-Shops Closed 🏬
    if 'shops_closed' in train_eda.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=train_eda, x='shops_closed', y='orders', palette='cool')
        plt.title('Orders: Shops Closed vs. Non-Shops Closed 🏬', fontsize=16)
        plt.xlabel('Are Shops Closed? (0: No, 1: Yes)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.xticks([0, 1], ['Shops Open', 'Shops Closed'])
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`shops_closed` column not found for plotting.")

    ## Orders: Winter School Holidays vs. Non-Winter School Holidays ☃️
    if 'winter_school_holidays' in train_eda.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=train_eda, x='winter_school_holidays', y='orders', palette='cubehelix')
        plt.title('Orders: Winter School Holidays vs. Non-Winter School Holidays ☃️', fontsize=16)
        plt.xlabel('Is Winter School Holiday? (0: No, 1: Yes)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.xticks([0, 1], ['Normal Days', 'Winter School Holiday'])
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`winter_school_holidays` column not found for plotting.")

    ## Orders: Frankfurt Shutdown vs. Non-Frankfurt Shutdown 🇩🇪
    if 'frankfurt_shutdown' in train_eda.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=train_eda, x='frankfurt_shutdown', y='orders', palette='crest')
        plt.title('Orders: Frankfurt Shutdown vs. Non-Frankfurt Shutdown 🇩🇪', fontsize=16)
        plt.xlabel('Is Frankfurt Shutdown? (0: No, 1: Yes)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.xticks([0, 1], ['Not Shutdown', 'Shutdown'])
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`frankfurt_shutdown` column not found for plotting.")

    ## Orders: MOV Change vs. No MOV Change 💰
    if 'mov_change' in train_eda.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data=train_eda, x='mov_change', y='orders', palette='flare')
        plt.title('Orders: MOV Change vs. No MOV Change 💰', fontsize=16)
        plt.xlabel('Is MOV Change? (0: No, 1: Yes)', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.xticks([0, 1], ['No Change', 'Change'])
        plt.tight_layout() # Added tight_layout
        st.pyplot(fig)
    else:
        st.write("`mov_change` column not found for plotting.")

    ## Orders per Day of Week by Warehouse 🗓️🏠
    g = sns.catplot(data=train_eda, x='day_of_week', y='orders', col='warehouse',
                    kind='box', col_wrap=2, height=5, aspect=1.2, palette='Set3', sharey=True)
    g.set_axis_labels("Day of Week", "Number of Orders")
    g.set_titles("Warehouse: {col_name}")
    g.set_xticklabels(rotation=45, ha='right')
    g.fig.suptitle('Orders Distribution by Day of Week for Each Warehouse 🗓️🏠', y=1.02, fontsize=16)
    plt.tight_layout() # Added tight_layout
    st.pyplot(g.fig) # Pass the figure object from catplot

    ## Average Daily Orders by Country 🌍
    avg_orders_per_country = train_eda.groupby('country')['orders'].mean().sort_values(ascending=False)
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_orders_per_country.index, y=avg_orders_per_country.values, palette='magma')
    plt.title('Average Daily Orders by Country 🌍', fontsize=16)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Average Number of Orders', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Added tight_layout
    st.pyplot(fig)

    # Note: Pair Plot is commented out as it can be computationally intensive for web apps.
    # if 'numerical_cols' in locals():
    #     st.subheader("Pair Plot of Key Numerical Features (Potentially Intensive)")
    #     st.write("Uncomment this section in the code to view the pair plot.")
    #     # fig = sns.pairplot(train_eda[numerical_cols])
    #     # plt.suptitle('Pair Plot of Key Numerical Features 🔄', y=1.02, fontsize=16)
    #     # st.pyplot(fig)

st.success('EDA plots generated!')



st.header("Model Performance on Test Set (Split from Training Data) 📈")
st.write("These plots show how well the model performed on the portion of the training data set aside for testing its generalization ability.")

# Load and preprocess data for model training
with st.spinner('Loading and preprocessing data for model...'):
    train_data_processed, test_data_processed = load_and_preprocess_model_data()
st.success('Data loaded and preprocessed for model!')

# Train model
with st.spinner('Training XGBoost model...'):
    model, X_train, X_test, y_train, y_test, history = train_xgboost_model(train_data_processed)
st.success('Model training complete!')

# Final predictions for the evaluation test set
y_pred_train = model.predict(xgb.DMatrix(X_train))
y_pred_test = model.predict(xgb.DMatrix(X_test))

# Final metrics
final_metrics = {
    'train': {
        'MAPE': np.mean(np.abs((y_train - y_pred_train) / y_train[y_train != 0])) * 100 if np.sum(y_train != 0) > 0 else np.nan, # Handle division by zero
        'MAE': mean_absolute_error(y_train, y_pred_train),
        'R2': r2_score(y_train, y_pred_train)
    },
    'test': {
        'MAPE': np.mean(np.abs((y_test - y_pred_test) / y_test[y_test != 0])) * 100 if np.sum(y_test != 0) > 0 else np.nan, # Handle division by zero
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'R2': r2_score(y_test, y_pred_test)
    }
}

st.subheader("Final Model Metrics")
col1, col2 = st.columns(2)
with col1:
    st.write("**Train Set:**")
    for metric, value in final_metrics['train'].items():
        st.write(f"- {metric}: {value:.4f}")
with col2:
    st.write("**Test Set:**")
    for metric, value in final_metrics['test'].items():
        st.write(f"- {metric}: {value:.4f}")

# Plots for the evaluation test set
st.subheader("Evaluation Plots")

fig1, axes1 = plt.subplots(3, 2, figsize=(15, 18))

# Actual vs Predicted
axes1[0, 0].scatter(y_test, y_pred_test, alpha=0.5)
axes1[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
axes1[0, 0].set_xlabel('Actual Values')
axes1[0, 0].set_ylabel('Predicted Values')
axes1[0, 0].set_title('Actual vs Predicted Values (Test Set)')

# Residual plot
residuals = y_test - y_pred_test
axes1[0, 1].scatter(y_pred_test, residuals, alpha=0.5)
axes1[0, 1].axhline(y=0, color='r', linestyle='--')
axes1[0, 1].set_xlabel('Predicted Values')
axes1[0, 1].set_ylabel('Residuals')
axes1[0, 1].set_title('Residual Plot (Test Set)')

# Error distribution
sns.histplot(residuals, kde=True, ax=axes1[1, 0])
axes1[1, 0].set_xlabel('Prediction Error')
axes1[1, 0].set_title('Error Distribution (Test Set)')

# Feature importance (Requires getting feature names from X_train)
feature_names = X_train.columns.tolist()
importance = model.get_score(importance_type='weight')
importance_df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance']).sort_values(by='importance', ascending=False)
sns.barplot(x='importance', y='feature', data=importance_df.head(15), ax=axes1[1, 1])
axes1[1, 1].set_title('Feature Importance')


# MAPE over iterations
axes1[2, 0].plot(history['train_mape'], label='Train MAPE')
axes1[2, 0].plot(history['test_mape'], label='Test MAPE')
axes1[2, 0].set_xlabel('Iteration')
axes1[2, 0].set_ylabel('MAPE (%)') # Changed label to reflect percentage
axes1[2, 0].set_title('MAPE Progression During Training')
axes1[2, 0].legend()

# MAE over iterations
axes1[2, 1].plot(history['train_mae'], label='Train MAE')
axes1[2, 1].plot(history['test_mae'], label='Test MAE')
axes1[2, 1].set_xlabel('Iteration')
axes1[2, 1].set_ylabel('MAE')
axes1[2, 1].set_title('MAE Progression During Training')
axes1[2, 1].legend()

plt.tight_layout()
st.pyplot(fig1)

# Learning curves for remaining metrics
st.subheader("Learning Curves for Other Metrics")
for metric_name in ['R2']:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history[f'train_{metric_name.lower()}'], label=f'Train {metric_name}')
    ax.plot(history[f'test_{metric_name.lower()}'], label=f'Test {metric_name}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Progression During Training')
    ax.legend()
    plt.tight_layout() # Added tight_layout
    st.pyplot(fig)

# Prediction error by actual value - Completed
st.subheader("Prediction Error by Actual Value Range")
fig_error_by_value, ax_error_by_value = plt.subplots(figsize=(10, 6))
# Ensure y_test is a pandas Series or DataFrame for alignment in plotting
y_test_series = pd.Series(y_test, index=y_test.index)
y_pred_test_series = pd.Series(y_pred_test, index=y_test.index)

# Create bins based on the range of actual values
# Use np.ceil and np.floor to ensure full range is covered if min/max are not exact multiples
min_val = np.floor(y_test.min())
max_val = np.ceil(y_test.max())
bins = np.linspace(min_val, max_val, 10) # Adjust number of bins as needed
bin_centers = (bins[:-1] + bins[1:]) / 2

# Calculate MAE for each bin
bin_errors = []
for i in range(len(bins) - 1):
    lower_bound = bins[i]
    upper_bound = bins[i+1]
    
    # Filter data points within the current bin
    mask = (y_test_series >= lower_bound) & (y_test_series < upper_bound)
    
    if np.sum(mask) > 0:
        actual_in_bin = y_test_series[mask]
        predicted_in_bin = y_pred_test_series[mask]
        bin_errors.append(mean_absolute_error(actual_in_bin, predicted_in_bin))
    else:
        bin_errors.append(np.nan) # Append NaN if no data points in bin

valid_bin_centers = [bc for bc, be in zip(bin_centers, bin_errors) if not np.isnan(be)]
valid_bin_errors = [be for be in bin_errors if not np.isnan(be)]

if valid_bin_centers:
    ax_error_by_value.bar(valid_bin_centers, valid_bin_errors, width=(bins[1]-bins[0])*0.8, color='darkorange', edgecolor='black')
    ax_error_by_value.set_xlabel('Actual Value (binned)')
    ax_error_by_value.set_ylabel('Mean Absolute Error')
    ax_error_by_value.set_title('Prediction Error by Actual Value Range')
    plt.tight_layout()
    st.pyplot(fig_error_by_value)
else:
    st.write("Not enough data to plot Prediction Error by Actual Value Range or all bins are empty.")


# Visualize the first tree
st.subheader("Example Tree Visualization (First Tree in the Model)")
fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
plot_tree(model, num_trees=0, ax=ax_tree)
ax_tree.set_title('Example Tree Visualization')
plt.tight_layout()
st.pyplot(fig_tree)



st.header("Predictions on Unseen Test Data 📦")
st.write("This section visualizes the distribution of orders predicted by the model for the original `test.csv` dataset (the unseen data).")

# Align columns for prediction on the unseen test set
# Get the columns from the training features
train_feature_columns = X_train.columns

# Add missing columns to test_data_processed and ensure order
missing_cols_in_test_for_pred = set(train_feature_columns) - set(test_data_processed.columns)
for col in missing_cols_in_test_for_pred:
    test_data_processed[col] = 0

# Drop extra columns from test_data_processed if they exist
extra_cols_in_test_for_pred = set(test_data_processed.columns) - set(train_feature_columns)
test_data_processed = test_data_processed.drop(columns=list(extra_cols_in_test_for_pred))

# Ensure the order of columns matches the training data
test_data_processed = test_data_processed[train_feature_columns]

submission_data = xgb.DMatrix(test_data_processed)

preds = model.predict(submission_data)

st.write(f"Generated {len(preds)} predictions for the unseen test data.")

# Distribution of Predictions
fig_preds_dist, ax_preds_dist = plt.subplots(figsize=(10, 6))
sns.histplot(preds, kde=True, bins=30, color='purple', ax=ax_preds_dist)
ax_preds_dist.set_title('Distribution of Predicted Orders for Unseen Data')
ax_preds_dist.set_xlabel('Predicted Number of Orders')
ax_preds_dist.set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig_preds_dist)

st.subheader("Predicted Orders Statistics (Unseen Data)")
st.write(pd.Series(preds).describe())
