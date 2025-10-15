# 📈 Generalized Sales Forecasting Dashboard

This Streamlit application provides a comprehensive solution for analyzing historical sales data and forecasting future sales using various machine learning models. It also includes a basic profit forecasting feature based on the sales predictions.

## ✨ Features

* **Data Upload & Preprocessing:** Easily upload your sales data in CSV format. The application handles missing values automatically (numeric NaNs with mean, categorical NaNs with mode) and removes duplicate entries.
* **Dynamic Column Selection:** Select your date and target (sales) columns dynamically from your uploaded dataset.
* **Time Frame Filtering:** Visualize your historical sales data for specific timeframes (e.g., last 7, 30, 90 days, or all available data).
* **Feature Selection for Forecasting:**
    * Select additional numeric and categorical features to be used by regression models (XGBoost, Linear Regression, Random Forest).
    * Specify the number of lag features for regression models.
    * Identify optional 'Inventory' and 'Promotion' columns for profit calculation.
* **Interactive Visualizations:**
    * **Time Series Plot:** Visualize your historical sales data over time.
    * **Feature Correlation Map:** Understand the relationships between your numeric features and the target variable.
    * **Custom Data Visualizations:** Create histograms, box plots, and scatter plots for any selected numeric columns.
    * **Categorical Distribution:** Analyze the distribution of your categorical data with bar charts.
    * **Grouped Bar Charts:** Explore percentage distributions within categorical groups.
* **Multiple Forecasting Models:** Choose from a selection of popular time series and regression models:
    * **Prophet:** A robust forecasting library from Facebook, designed for business forecasts.
    * **ARIMA:** AutoRegressive Integrated Moving Average model, a classic time series forecasting method.
    * **XGBoost:** A powerful gradient boosting framework known for its performance and speed.
    * **Linear Regression:** A fundamental statistical model for understanding the relationship between variables.
    * **Random Forest:** An ensemble learning method that builds multiple decision trees for robust prediction.
* **Forecast Horizon Control:** Adjust the number of days into the future you want to forecast.
* **Performance Metrics:** Evaluate model performance with key metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) on recent historical data.
* **Download Forecasts:** Download the generated sales and profit forecasts as CSV files.
* **Profit Forecasting:** Project future profits based on the sales forecast and a user-defined markup percentage.

## 🛠️ Technologies Used

* **Streamlit** - Web framework for interactive data apps.
* **Pandas, NumPy** - Data handling and numerical computation.
* **Matplotlib, Seaborn** - Visualization.
* **Scikit-learn** - ML utilities (Linear Regression, RandomForest, metrics).
* **Statsmodels** - ARIMA modeling.
* **XGBoost** - Gradient boosting model.
* **Prophet (Meta)** - Business forecasting.

## 🚀 Getting Started

### Installation

1. Save the script (e.g., `sales_forecast_app.py`) in your working folder.

2. Create a file named `requirements.txt` in the same directory with the following content:

    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    statsmodels
    xgboost
    prophet
    ```

3. Then install the dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

    > ⚠️ If you face issues installing `prophet`, try installing these manually:
    > ```bash
    > pip install pystan==2.19.1.1
    > pip install prophet
    > ```

### Run the Application

```bash
streamlit run sales_forecast_app.py


## 👨‍💻 How to Use

1. Upload a CSV under "Step 1: Upload Dataset".
2. The app auto-cleans your data.
3. Select date and sales columns.
4. (Optional) Filter timeframes, select extra features.
5. View visualizations for numeric & categorical features.
6. Choose a forecasting model and duration.
7. Run the forecast and analyze results.
8. Download results as CSV.

## 👥 Authors

* **Rohit Gupta**  
* **Sagar Kumar**  
* **Sheetal Patil**