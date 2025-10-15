import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# --- Helper Functions ---
def fill_missing_values(df_input):
    """Fills numeric NaNs with mean and categorical NaNs with mode."""
    df = df_input.copy()
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include='object').columns:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
    return df

def prepare_regression_features(df_prophet_data, lags, additional_features_df):
    """
    Prepares data for regression models by adding lag features and merging additional features.
    df_prophet_data: DataFrame with 'ds' and 'y'.
    additional_features_df: DataFrame with 'ds' as index and one-hot encoded features.
    """
    df_temp = df_prophet_data.copy().set_index('ds')
    for i in range(1, lags + 1):
        df_temp[f'lag_{i}'] = df_temp['y'].shift(i)
    if not additional_features_df.empty:
        df_temp = df_temp.merge(additional_features_df.reindex(df_temp.index), left_index=True, right_index=True, how='left')
    return df_temp.dropna()

def extrapolate_future_features(df_processed_history, periods, additional_features_df):
    """
    Extrapolates future values for additional features for regression models.
    Uses ffill for simplicity.
    """
    if df_processed_history.empty:
        return pd.DataFrame()
    extrapolation_start_date = df_processed_history.index[-1] + timedelta(days=1)
    future_dates = pd.date_range(start=extrapolation_start_date, periods=periods, freq='D')
    if not additional_features_df.empty:
        return additional_features_df.reindex(future_dates).fillna(method='ffill').fillna(0)
    return pd.DataFrame(index=future_dates)

def recursive_forecast(model, X_train_cols, last_known_row, future_dates_range, future_features_extrapolated, lags):
    """
    Performs recursive forecasting for regression models (XGBoost, LR, RF).
    Updates lags with predicted values.
    """
    future_preds = []
    current_last_row = last_known_row.copy()
    for i in range(len(future_dates_range)):
        x_input_dict = {f'lag_{j}': current_last_row.get(f'lag_{j}', 0) for j in range(1, lags + 1)}
        if not future_features_extrapolated.empty:
            current_future_features = future_features_extrapolated.iloc[i]
            x_input_dict.update(current_future_features.to_dict())
        x_input = pd.DataFrame([x_input_dict]).reindex(columns=X_train_cols, fill_value=0)
        pred = model.predict(x_input)[0]
        future_preds.append(pred)
        if lags > 0:
            for j in range(lags, 1, -1):
                current_last_row[f'lag_{j}'] = current_last_row.get(f'lag_{j-1}', 0)
            current_last_row['lag_1'] = pred
        if not future_features_extrapolated.empty:
            current_last_row.update(current_future_features)
    return pd.DataFrame({'ds': future_dates_range, 'yhat': future_preds}).set_index('ds')

# --- Forecast Functions ---
def forecast_prophet(df, periods, additional_features_df):
    df_prophet_model = df.copy()
    if not additional_features_df.empty:
        df_prophet_model = df_prophet_model.set_index('ds').merge(additional_features_df, left_index=True, right_index=True, how='left').reset_index()
    model = Prophet()
    if not additional_features_df.empty:
        for feat in additional_features_df.columns:
            model.add_regressor(feat)
    model.fit(df_prophet_model)
    future = model.make_future_dataframe(periods=periods)
    if not additional_features_df.empty:
        full_date_range_start = df_prophet_model['ds'].min() if not df_prophet_model.empty else pd.Timestamp.now()
        full_date_range = pd.date_range(start=full_date_range_start, end=future['ds'].max(), freq='D')
        extended_features = additional_features_df.reindex(full_date_range).fillna(method='ffill').fillna(0)
        for feat in additional_features_df.columns:
            future[feat] = extended_features.loc[future['ds'], feat].values if feat in extended_features.columns else 0
    return model.predict(future)[['ds', 'yhat']].set_index('ds')

def forecast_arima(df, periods):
    df = df.set_index('ds')
    if df['y'].isnull().all() or df.empty:
        st.error("ARIMA: Target series is empty or all nulls. Cannot fit ARIMA.")
        return pd.DataFrame()
    if len(df) < 10:
        st.warning("ARIMA: Not enough data points to reliably fit the model. Consider a longer dataset.")
        return pd.DataFrame()
    try:
        model_fit = ARIMA(df['y'], order=(5, 1, 0)).fit()
        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=periods, freq='D')
        return pd.DataFrame({'ds': future_dates, 'yhat': model_fit.forecast(steps=periods).values}).set_index('ds')
    except Exception as e:
        st.error(f"ARIMA model fitting failed: {e}. Please check your data or try a different model.")
        return pd.DataFrame()

def forecast_regression_model(model_class, df_prophet_data, periods, lags, all_selected_features_df):
    df_processed = prepare_regression_features(df_prophet_data, lags, all_selected_features_df)
    if df_processed.empty:
        st.warning(f"{model_class.__name__}: Not enough data after preparing features and lags. Cannot run forecast.")
        return pd.DataFrame()
    features_for_model = [f'lag_{i}' for i in range(1, lags + 1)] + all_selected_features_df.columns.tolist()
    for f in features_for_model:
        if f not in df_processed.columns:
            df_processed[f] = 0
    X, y = df_processed[features_for_model], df_processed['y']
    if X.empty or y.empty:
        st.warning(f"{model_class.__name__}: No data to train on after feature preparation. Cannot run forecast.")
        return pd.DataFrame()
    model = model_class(n_estimators=100, random_state=42) if model_class in [RandomForestRegressor, XGBRegressor] else model_class()
    if model_class == XGBRegressor:
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X, y)
    last_row = df_processed.iloc[-1].copy()
    future_dates = pd.date_range(start=df_processed.index[-1] + timedelta(days=1), periods=periods, freq='D')
    future_features_extrapolated = extrapolate_future_features(df_processed, periods, all_selected_features_df)
    st.session_state['trained_model'] = model
    st.session_state['X_train_cols'] = X.columns.tolist()
    st.session_state['df_prophet_data_for_profit'] = df_prophet_data.copy()
    return recursive_forecast(model, X.columns, last_row, future_dates, future_features_extrapolated, lags)

# --- Streamlit UI ---
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📈 Analytics For Business Forecasting")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Initialize session state variables
for key in ['trained_model', 'X_train_cols', 'df_prophet_data_for_profit', 'forecast_df_sales', 'df_cleaned_global', 'date_col_global', 'inventory_col_global', 'promotion_col_global', 'categorical_promotion_col']:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame() if 'df' in key else None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data (Before Cleaning)")
    st.write(df.head())

    st.sidebar.header("Preprocessing")
    st.sidebar.markdown("Missing value handling will be applied automatically.")
    df_cleaned = fill_missing_values(df.drop_duplicates())
    st.session_state['df_cleaned_global'] = df_cleaned.copy()
    st.subheader("Cleaned Data Preview")
    st.write(df_cleaned.head())

    st.sidebar.header("Select Columns for Time Frame Summary")
    all_columns = df_cleaned.columns.tolist()
    date_col = st.sidebar.selectbox("Select Date Column", all_columns)
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    target_col = st.sidebar.selectbox("Select Target (Sales) Column", numeric_cols)
    st.session_state['date_col_global'] = date_col

    try:
        df_cleaned[date_col] = pd.to_datetime(df_cleaned[date_col])
    except Exception as e:
        st.error(f"❌ Could not parse the selected date column: {e}. Please check the data.")
        st.stop()

    df_prophet = df_cleaned[[date_col, target_col]].dropna().drop_duplicates(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df_prophet.columns = ['ds', 'y']

    
    days_options = [7, 30, 60, 90, 180, 365, "All"]
    selected_days = st.sidebar.selectbox("View last N days", days_options, index=1)
    df_viz = df_prophet[df_prophet['ds'] >= (df_prophet['ds'].max() - pd.Timedelta(days=int(selected_days)))] if selected_days != "All" else df_prophet.copy()

    st.sidebar.header(" Feature Selection for Model Building")
    available_numeric_features = [col for col in numeric_cols if col not in [target_col]]
    available_categorical_features = df_cleaned.select_dtypes(include='object').columns.tolist()

    potential_inventory_cols = [col for col in available_numeric_features if 'inventory' in col.lower()]
    st.session_state['inventory_col_global'] = st.sidebar.selectbox("Select Inventory Column (for Profit Calculation)", ['None'] + potential_inventory_cols, key='inventory_select')

    # Allow selection of categorical 'promotion' columns
    potential_promotion_cols = [col for col in available_numeric_features + available_categorical_features if 'promotion' in col.lower()]
    selected_promotion_col_name = st.sidebar.selectbox("Select Promotion Column (for Profit Calculation)", ['None'] + potential_promotion_cols, key='promotion_select')

    st.session_state['promotion_col_global'] = selected_promotion_col_name if selected_promotion_col_name != 'None' else None

    # Store if the selected promotion column is categorical
    if st.session_state['promotion_col_global'] and st.session_state['promotion_col_global'] in available_categorical_features:
        st.session_state['categorical_promotion_col'] = st.session_state['promotion_col_global']
    else:
        st.session_state['categorical_promotion_col'] = None

    selected_numeric_features = st.sidebar.multiselect("Select additional numeric features", options=available_numeric_features)
    # Exclude the selected promotion column if it's categorical to avoid double processing
    selected_categorical_features_for_model = [col for col in available_categorical_features if col != st.session_state['categorical_promotion_col']]
    selected_categorical_features = st.sidebar.multiselect("Select categorical features (one-hot encoded)", options=selected_categorical_features_for_model)

    lag_count = st.sidebar.slider("Number of lag features (XGBoost, Linear Reg., Random Forest)", min_value=0, max_value=10, value=2)


    cols_to_select_for_features = [date_col] + selected_numeric_features
    if selected_categorical_features:
        cols_to_select_for_features.extend(selected_categorical_features)

    # Add the selected promotion column to features if it's not already in numeric or categorical for model
    if st.session_state['promotion_col_global'] and st.session_state['promotion_col_global'] not in selected_numeric_features and st.session_state['promotion_col_global'] not in selected_categorical_features:
        cols_to_select_for_features.append(st.session_state['promotion_col_global'])

    all_selected_features_df = df_cleaned[cols_to_select_for_features].copy().rename(columns={date_col: 'ds'}).set_index('ds')

    # Apply one-hot encoding to all selected categorical features, including the potential promotion column
    features_to_onehot_encode = selected_categorical_features.copy()
    if st.session_state['categorical_promotion_col'] and st.session_state['categorical_promotion_col'] not in features_to_onehot_encode:
        features_to_onehot_encode.append(st.session_state['categorical_promotion_col'])

    if features_to_onehot_encode:
        all_selected_features_df = pd.get_dummies(all_selected_features_df, columns=features_to_onehot_encode, drop_first=True)


    st.subheader("🕒 Time Frame Summary")
    st.markdown(f"**Start Date:** `{df_viz['ds'].min().date()}`")
    st.markdown(f"**End Date:** `{df_viz['ds'].max().date()}`")
    st.markdown(f"**Total Observations:** `{df_viz.shape[0]}`")
    diffs = df_viz['ds'].diff().dropna()
    if not diffs.empty:
        st.markdown(f"**Most Common Frequency:** `{diffs.value_counts().idxmax()}`")
    st.line_chart(df_viz.set_index('ds')['y'])

    # --- Correlation Map Section ---
    st.subheader("📈 Feature Correlation Map for Numeric Column Selection")
    st.markdown("Examine correlations between numeric features and the target variable.")

    correlation_cols = [col for col in numeric_cols if col != date_col] # Exclude date if it's numeric and shouldn't be correlated
    if target_col not in correlation_cols:
        correlation_cols.append(target_col) # Ensure target is always there for correlation

    df_for_correlation_map = df_cleaned[correlation_cols].copy()

    # Only keep numeric types for correlation calculation, in case some non-numeric columns were in numeric_cols
    df_for_correlation_map = df_for_correlation_map.select_dtypes(include=np.number)

    if not df_for_correlation_map.empty and len(df_for_correlation_map.columns) > 1: # Need at least 2 columns for a correlation matrix
        # Calculate the correlation matrix
        corr_matrix = df_for_correlation_map.corr()

        # Create the heatmap
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
        ax_corr.set_title("Correlation Matrix of Numeric Features")
        st.pyplot(fig_corr)

        st.markdown("---") # Separator
    else:
        st.info("Not enough numeric columns available (or only one) for a meaningful correlation matrix.")
    # --- End Correlation Map Section ---


    st.subheader("📊 Custom Data Visualizations")
    categorical_columns_for_viz = df_cleaned.select_dtypes(include='object').columns.tolist()
    viz_type = st.radio("Choose Visualization Type", ["Histogram", "Box Plot", "Scatter Plot"], horizontal=True)

    if viz_type == "Histogram":
        selected_col = st.selectbox("Select column for Histogram", numeric_cols)
        kde_toggle = st.checkbox("Show KDE", value=True)
        bins = st.slider("Number of Bins", 5, 100, 30)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df_cleaned[selected_col], kde=kde_toggle, bins=bins, color='skyblue', ax=ax)
        ax.set_title(f"Histogram of {selected_col}")
        st.pyplot(fig)
    elif viz_type == "Box Plot":
        selected_col = st.selectbox("Select column for Box Plot", numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=df_cleaned[selected_col], color='salmon', ax=ax)
        ax.set_title(f"Box Plot of {selected_col}")
        st.pyplot(fig)
    elif viz_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis", numeric_cols, index=0)
        y_col = st.selectbox("Select Y-axis", numeric_cols, index=1)
        color_col = st.selectbox("Color by (optional)", ["None"] + categorical_columns_for_viz)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_cleaned, x=x_col, y=y_col, hue=color_col if color_col != "None" else None, ax=ax)
        ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
        st.pyplot(fig)

    if categorical_columns_for_viz:
        st.subheader("📊 Categorical Distribution")
        selected_cat = st.selectbox("Choose a Categorical Column", categorical_columns_for_viz)
        fig, ax = plt.subplots(figsize=(10, 5))
        df_cleaned[selected_cat].value_counts().plot(kind='bar', color='mediumseagreen', ax=ax)
        ax.set_title(f"Distribution of {selected_cat}")
        st.pyplot(fig)

    if len(categorical_columns_for_viz) >= 2:
        st.subheader("📊 Grouped Bar Chart (Percentage Distribution)")
        group_x = st.selectbox("Select X-axis Category", categorical_columns_for_viz, index=0, key="group_x")
        group_hue = st.selectbox("Select Hue (Color Grouping)", [col for col in categorical_columns_for_viz if col != group_x], index=0, key="group_hue")
        group_df = df_cleaned.groupby([group_x, group_hue]).size().reset_index(name='Count')
        group_df['Percent'] = group_df.groupby(group_x)['Count'].transform(lambda x: 100 * x / x.sum())
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=group_df, x=group_x, y="Percent", hue=group_hue, palette='Pastel1', ax=ax)
        ax.set_title(f"Percentage Distribution of {group_hue} within each {group_x}")
        ax.set_ylabel("Percentage (%)")
        plt.xticks(rotation=25)
        st.pyplot(fig)

    st.sidebar.header("Forecast")
    model_choice = st.sidebar.selectbox("Choose Model", ["Prophet", "ARIMA", "XGBoost", "Linear Regression", "Random Forest"])
    periods = st.sidebar.slider("Forecast Horizon (days)", 7, 365, 30)

    if st.sidebar.button("Run Forecast"):
        forecast_df = pd.DataFrame()
        try:
            if model_choice == "Prophet":
                forecast_df = forecast_prophet(df_prophet.copy(), periods, all_selected_features_df.copy())
            elif model_choice == "ARIMA":
                forecast_df = forecast_arima(df_prophet.copy(), periods)
            elif model_choice == "XGBoost":
                forecast_df = forecast_regression_model(XGBRegressor, df_prophet.copy(), periods, lag_count, all_selected_features_df.copy())
            elif model_choice == "Linear Regression":
                forecast_df = forecast_regression_model(LinearRegression, df_prophet.copy(), periods, lag_count, all_selected_features_df.copy())
            elif model_choice == "Random Forest":
                forecast_df = forecast_regression_model(RandomForestRegressor, df_prophet.copy(), periods, lag_count, all_selected_features_df.copy())

        except Exception as e:
            st.error(f"Error during forecasting with {model_choice}: {e}")
            forecast_df = pd.DataFrame()

        if not forecast_df.empty:
            plot_start_date = df_prophet['ds'].max() - pd.Timedelta(days=periods * 2)
            actuals_for_plot = df_prophet[df_prophet['ds'] >= max(plot_start_date, df_prophet['ds'].min())].copy().rename(columns={'y': 'y_actual'})
            forecast_df_renamed = forecast_df.reset_index().rename(columns={'yhat': 'yhat_forecast'})

            full_plot_df = pd.merge(actuals_for_plot[['ds', 'y_actual']], forecast_df_renamed[['ds', 'yhat_forecast']], on='ds', how='outer')
            plot_df_for_streamlit = full_plot_df.set_index('ds').rename(columns={'y_actual': 'y', 'yhat_forecast': 'yhat'})

            st.subheader(f"{model_choice} Forecast vs Actual")
            if not plot_df_for_streamlit.empty and 'y' in plot_df_for_streamlit.columns and 'yhat' in plot_df_for_streamlit.columns:
                st.line_chart(plot_df_for_streamlit[['y', 'yhat']])
            else:
                st.warning("No 'y' or 'yhat' data available in the final plotting DataFrame.")

            st.subheader("📊 Forecast Metrics")
            merged_for_metrics = pd.DataFrame()

            if model_choice in ["XGBoost", "Linear Regression", "Random Forest"] and st.session_state['trained_model'] is not None:
                model = st.session_state['trained_model']
                X_train_cols = st.session_state['X_train_cols']
                metrics_lookback_start_date = df_prophet['ds'].max() - pd.Timedelta(days=periods)
                actuals_for_metrics_period = df_prophet[df_prophet['ds'] >= metrics_lookback_start_date].copy()

                if not actuals_for_metrics_period.empty:
                    temp_df_for_historical_features = df_cleaned[df_cleaned[date_col] >= metrics_lookback_start_date].copy().rename(columns={date_col: 'ds'}).set_index('ds')

                    # Ensure all relevant selected features (numeric and categorical) are included for historical processing
                    historical_features_selected = [col for col in selected_numeric_features if col in temp_df_for_historical_features.columns]
                    historical_features_cat_selected_for_metrics = [col for col in selected_categorical_features if col in temp_df_for_historical_features.columns]

                    # Add the promotion column if it was used in training and is in temp_df_for_historical_features
                    if st.session_state['promotion_col_global'] and st.session_state['promotion_col_global'] in temp_df_for_historical_features.columns and st.session_state['promotion_col_global'] not in historical_features_selected and st.session_state['promotion_col_global'] not in historical_features_cat_selected_for_metrics:
                         # Decide whether to treat it as numeric or categorical here for metrics preparation
                        if pd.api.types.is_numeric_dtype(temp_df_for_historical_features[st.session_state['promotion_col_global']]):
                            historical_features_selected.append(st.session_state['promotion_col_global'])
                        else:
                            historical_features_cat_selected_for_metrics.append(st.session_state['promotion_col_global'])


                    historical_additional_features_df = temp_df_for_historical_features[historical_features_selected + historical_features_cat_selected_for_metrics].copy()

                    if historical_features_cat_selected_for_metrics:
                        historical_additional_features_df = pd.get_dummies(historical_additional_features_df, columns=historical_features_cat_selected_for_metrics, drop_first=True)

                    df_historical_processed = prepare_regression_features(actuals_for_metrics_period, lag_count, historical_additional_features_df)

                    if not df_historical_processed.empty:
                        X_historical = df_historical_processed.reindex(columns=X_train_cols, fill_value=0)
                        y_historical_actual = df_historical_processed['y']

                        if not X_historical.empty and not y_historical_actual.empty:
                            y_historical_pred = model.predict(X_historical)
                            merged_for_metrics = pd.DataFrame({'ds': y_historical_actual.index, 'y': y_historical_actual.values, 'yhat': y_historical_pred}).set_index('ds')
                        else:
                            st.warning("Historical data for metrics is empty after feature preparation. Cannot calculate in-sample metrics.")
                    else:
                        st.warning("Historical data for metrics is empty after feature preparation. Cannot calculate in-sample metrics.")
                else:
                    st.warning("Not enough recent actuals data to calculate in-sample metrics for the selected horizon.")

            elif model_choice == "Prophet" and st.session_state['trained_model'] is not None:
                model = st.session_state['trained_model']
                historical_future = model.make_future_dataframe(periods=0, include_history=True)
                if not all_selected_features_df.empty:
                    full_date_range_start = df_prophet['ds'].min() if not df_prophet.empty else pd.Timestamp.now()
                    full_date_range = pd.date_range(start=full_date_range_start, end=historical_future['ds'].max(), freq='D')
                    extended_features = all_selected_features_df.reindex(full_date_range).fillna(method='ffill').fillna(0)
                    for feat in all_selected_features_df.columns:
                        historical_future[feat] = extended_features.loc[historical_future['ds'], feat].values if feat in extended_features.columns else 0

                historical_forecast = model.predict(historical_future)
                merged_for_metrics = pd.merge(df_prophet.set_index('ds'), historical_forecast[['ds', 'yhat']].set_index('ds'), on='ds', how='inner', suffixes=('_actual', '_forecast'))
                merged_for_metrics = merged_for_metrics.rename(columns={'y_actual': 'y', 'yhat_forecast': 'yhat'})
                metrics_lookback_start_date = df_prophet['ds'].max() - pd.Timedelta(days=periods)
                merged_for_metrics = merged_for_metrics[merged_for_metrics.index >= metrics_lookback_start_date].dropna()

            elif model_choice == "ARIMA" and st.session_state['trained_model'] is not None:
                st.warning("ARIMA in-sample metrics are not directly implemented in this demo for the look-back period. Metrics will only show for overlapping actuals and forecasts if available (e.g., if your actuals extend into the forecast period).")

            if not merged_for_metrics.empty and 'y' in merged_for_metrics.columns and 'yhat' in merged_for_metrics.columns:
                mae = mean_absolute_error(merged_for_metrics['y'], merged_for_metrics['yhat'])
                rmse = sqrt(mean_squared_error(merged_for_metrics['y'], merged_for_metrics['yhat']))
                mape_calc = np.abs((merged_for_metrics['y'] - merged_for_metrics['yhat']) / merged_for_metrics['y'])
                mape_calc = mape_calc[np.isfinite(mape_calc)]
                mape = np.mean(mape_calc) * 100 if not mape_calc.empty else float('nan')

                st.write(f"**MAE**: {mae:.2f}")
                st.write(f"**RMSE**: {rmse:.2f}")
                st.write(f"**MAPE**: {mape:.2f}%" if not np.isnan(mape) else "**MAPE**: Not applicable (contains zero or very small actuals)")
            else:
                st.warning("No overlapping actuals and forecasts to calculate metrics. This is expected if your forecast period extends beyond your historical data. Metrics shown are for model performance on recent historical data where actuals are available.")

            st.download_button("📥 Download Sales Forecast", forecast_df.reset_index().to_csv(index=False), file_name="sales_forecast.csv")
            st.session_state['forecast_df_sales'] = forecast_df.reset_index()

        else:
            st.warning("Sales Forecast could not be generated. Please check your data and selected settings.")

st.subheader("💰 Profit Forecasting")
st.markdown("Adjust the markup percentage to see the projected profit based on the sales forecast.")

if st.session_state['trained_model'] is not None and not st.session_state['forecast_df_sales'].empty:
    model = st.session_state['trained_model']
    X_train_cols = st.session_state['X_train_cols']
    df_cleaned_global = st.session_state['df_cleaned_global']
    date_col = st.session_state['date_col_global']
    inventory_col = st.session_state['inventory_col_global']
    promotion_col = st.session_state['promotion_col_global'] # This will be the original column name
    categorical_promotion_col = st.session_state['categorical_promotion_col'] # This will be the original column name if it was categorical

    sales_forecast_df = st.session_state['forecast_df_sales']

    future_dates = pd.to_datetime(sales_forecast_df['ds'])
    sales_forecast = sales_forecast_df['yhat'].values

    last_known_features = df_cleaned_global.set_index(date_col).iloc[-1]
    last_inventory = last_known_features[inventory_col] if inventory_col and inventory_col in last_known_features else 0

    # Determine the 'promotion' state for future prediction
    last_promotion_state = 0 # Default to no promotion
    if promotion_col and promotion_col in last_known_features:
        if categorical_promotion_col: # If promotion column was categorical ('yes'/'no')
            # Assuming 'yes' maps to a positive influence, so we'd need to know the one-hot encoded column name
            # For simplicity, if the last known state was 'yes', assume future is 'yes' (represented as 1)
            # This is a basic assumption; more advanced models would predict future feature states.
            promotion_value_map = {'yes': 1, 'no': 0} # Adjust this mapping as per your data
            last_promotion_state = promotion_value_map.get(str(last_known_features[promotion_col]).lower(), 0)

            # If 'promotion_yes' or similar is in X_train_cols, we need to set that
            # For the profit calculation, we'll use a simple approach for now,
            # assuming it's either on (1) or off (0) for the future for a simple calculation.
        else: # If promotion column was numeric
            last_promotion_state = last_known_features[promotion_col]

    profit_future_df = pd.DataFrame(index=future_dates)

    if 'day_of_week' in X_train_cols:
        profit_future_df['day_of_week'] = future_dates.dayofweek
    if 'month' in X_train_cols:
        profit_future_df['month'] = future_dates.month

    for col in X_train_cols:
        if inventory_col and col == inventory_col and inventory_col in last_known_features:
            profit_future_df[col] = last_inventory
        # Check for one-hot encoded promotion column names
        elif categorical_promotion_col and f"{categorical_promotion_col}_yes" in col and f"{categorical_promotion_col}_yes" in X_train_cols:
            profit_future_df[col] = last_promotion_state # This sets the one-hot encoded 'yes' column
        elif promotion_col and col == promotion_col and promotion_col in last_known_features and not categorical_promotion_col: # Numeric promotion
            profit_future_df[col] = last_promotion_state
        elif col in last_known_features and 'lag_' not in col:
            if pd.api.types.is_numeric_dtype(last_known_features[col]):
                profit_future_df[col] = last_known_features[col]
            else:
                profit_future_df[col] = 0

    for col in X_train_cols:
        if col not in profit_future_df.columns:
            profit_future_df[col] = 0

    markup = st.slider('Markup (%)', min_value=10, max_value=100, value=45, step=5)

    # Simple profit calculation: sales * markup.
    # If you want promotion to directly impact profit calculation (e.g., reduce costs),
    # you'd need a more complex formula incorporating last_promotion_state or a cost component.
    profit_forecast = sales_forecast * markup / 100

    forecast_profit_df = pd.DataFrame({
        'date': future_dates,
        'predicted_sales': sales_forecast.round(2),
        'predicted_profit': profit_forecast.round(2)
    })
    st.dataframe(forecast_profit_df)

    st.download_button("📥 Download Profit Forecast", forecast_profit_df.to_csv(index=False), file_name="profit_forecast.csv")

elif st.session_state['trained_model'] is None:
    st.info("Please upload a CSV file and run a sales forecast using XGBoost, Linear Regression, or Random Forest to enable profit forecasting.")
else:
    st.warning("No sales forecast available to calculate profit. Please ensure the sales forecast was generated successfully.")