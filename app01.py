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
import os

# Initialize Streamlit with cache configuration
st.set_page_config(layout="wide", page_title="Rohlik Orders Forecasting Analysis")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Clear caches at startup to prevent CacheStorageKeyNotFoundError
@st.cache_resource
def clear_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
clear_caches()

# Set up matplotlib style for aesthetics
plt.style.use('ggplot')

# --- Data Loading and Initial Preprocessing for EDA plots ---
@st.cache_data(persist=True, show_spinner=False)
def load_and_preprocess_eda_data():
    try:
        # Use relative path and check file existence
        train_path = 'train.csv'  # Make sure this file is in your repository
        if not os.path.exists(train_path):
            st.error(f"File not found: {train_path}")
            return None
            
        train_eda = pd.read_csv(train_path)
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
    except Exception as e:
        st.error(f"Error loading EDA data: {str(e)}")
        return None

# --- Data Loading and Preprocessing for Model Training ---
@st.cache_data(persist=True, show_spinner=False)
def load_and_preprocess_model_data():
    try:
        # Use relative paths and check file existence
        train_path = 'train.csv'
        test_path = 'test.csv'
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            st.error("Required data files not found!")
            return None, None
            
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path, index_col='id')
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

        # Rest of your processing code remains the same...
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
                         'mov_change', 'shutdown', 'blackout', 'mini_shutdown', 'frankfurt_shutdown']

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

        categorical_columns = ['holiday', 'shops_closed', 'winter_school_holidays', 'school_holidays', 'day_of_week', 'month_name', 'warehouse', 'country']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category')

        data = pd.get_dummies(data)

        train_processed = data[~data['orders'].isnull()]
        test_processed = data[data['orders'].isnull()]
        test_processed = test_processed.drop(columns=['orders'])

        return train_processed, test_processed
        
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        return None, None

# --- Model Training ---
@st.cache_resource(show_spinner=False)
def train_xgboost_model(train_data_df):
    try:
        if train_data_df is None:
            return None, None, None, None, None, None
            
        target = train_data_df['orders']
        features = train_data_df.drop(columns=['orders'])

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.09, random_state=42)
        
        # Align columns
        train_cols = X_train.columns
        test_cols = X_test.columns
        
        missing_in_test = set(train_cols) - set(test_cols)
        for c in missing_in_test:
            X_test[c] = 0
        missing_in_train = set(test_cols) - set(train_cols)
        for c in missing_in_train:
            X_train[c] = 0
        X_test = X_test[train_cols]

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

                return False

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
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None, None

# --- Main App ---
def main():
    st.title("Rohlik Orders Forecasting Analysis 📊")
    st.write("This application provides a comprehensive visualization of the Rohlik orders dataset and the performance of the trained XGBoost model.")

    # --- Initial EDA Plots ---
    st.header("Exploratory Data Analysis (EDA) 🔎")
    st.write("These plots provide insights into the raw data, helping us understand trends, distributions, and relationships before modeling.")

    with st.spinner('Generating EDA plots...'):
        train_eda = load_and_preprocess_eda_data()
        
        if train_eda is not None:
            # Your EDA plotting code remains the same...
            # Just add None checks before each plotting section
            
            ## Daily Orders Over Time by Warehouse 📈
            if train_eda is not None:
                fig = plt.figure(figsize=(15, 7))
                sns.lineplot(data=train_eda, x='date', y='orders', hue='warehouse', marker='o')
                plt.title('Daily Orders Over Time by Warehouse 📈', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Number of Orders', fontsize=12)
                plt.grid(True)
                st.pyplot(fig)

            # Continue with all your other plots, each wrapped in a None check
            
            st.success('EDA plots generated!')
        else:
            st.error("Failed to load EDA data")

    # --- Model Section ---
    st.header("Model Performance on Test Set (Split from Training Data) 📈")
    st.write("These plots show how well the model performed on the portion of the training data set aside for testing its generalization ability.")

    with st.spinner('Loading and preprocessing data for model...'):
        train_data_processed, test_data_processed = load_and_preprocess_model_data()
        
    if train_data_processed is not None and test_data_processed is not None:
        st.success('Data loaded and preprocessed for model!')
        
        with st.spinner('Training XGBoost model...'):
            model, X_train, X_test, y_train, y_test, history = train_xgboost_model(train_data_processed)
            
        if model is not None:
            st.success('Model training complete!')
            
            # Rest of your model evaluation and plotting code...
            # Make sure to add None checks before each section
            
        else:
            st.error("Model training failed")
    else:
        st.error("Failed to load model data")

if __name__ == "__main__":
    main()
