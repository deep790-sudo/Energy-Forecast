# Appliances Energy Consumption Forecasting


## Project Overview

This project focuses on predicting household appliance energy consumption using the **Appliances Energy Prediction** dataset from UCI.

The work involves data cleaning, preprocessing, feature engineering, model training, and evaluation using multiple machine learning techniques.

A Streamlit application has also been developed to allow interactive forecasting and visualization of results.

---

## Dataset

* **Source:** [Appliances Energy Prediction — UCI / Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)
* **Size:** 19,735 rows and 28 features
* **Duration:** 4.5 months (Jan–May 2016), recorded at 10-minute intervals
* **Target Variable:** `Appliances` — energy consumption in Wh

**Key Features:**
- Temperature readings from 9 rooms (T1–T9)
- Humidity readings from 9 rooms (RH_1–RH_9)
- Outdoor weather: temperature, humidity, wind speed, visibility, dewpoint, pressure

---

## Preprocessing and Feature Engineering

1. **Missing Values:** Checked and confirmed no missing values in the dataset.
2. **Column Removal:** Dropped `rv1`, `rv2` (random test variables) and `lights` (data leakage risk).
3. **Feature Engineering:** Created time-based features (`Hour`, `DayOfWeek`, `Month`, `Is_Weekend`).
4. **Temperature Aggregation:** Computed average indoor temperature (mean of T1–T9) and indoor-outdoor temperature difference.
5. **Humidity Aggregation:** Computed average indoor humidity (mean of RH_1–RH_9).
6. **Lag Features:** Added lag features (1–6 steps) on the target variable.
7. **Rolling Statistics:** Added rolling mean (3, 6, 12 steps) and rolling standard deviation (6 steps).

---

## Models Used

* **Random Forest Regressor**
* **Gradient Boosting Regressor**
* **XGBoost Regressor** (tuned with Optuna)
* **LightGBM Regressor**

---

## Evaluation and Metrics

Evaluation metrics used:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **R² Score (Coefficient of Determination)**

**Results:**

|              Model |  RMSE  |   MAE   |   R²   |
| -----------------: | :----: | :-----: | :----: |
|      Random Forest | 64.511 | 33.648  | 0.4932 |
|  Gradient Boosting | 71.034 | 40.018  | 0.3855 |
|   XGBoost (Optuna) | 57.939 | 26.321  | 0.5912 |
|           LightGBM | 62.463 | 30.913  | 0.5249 |

**Naive Baseline:**
R² = 0.4653 | RMSE = — | MAE = —

**XGBoost (Optuna-tuned) achieved the best performance with the lowest RMSE and highest R².**

> *Note: The Appliances Energy dataset is inherently noisy due to human behavioral patterns. R² values in the 0.50–0.60 range are consistent with published benchmarks for this dataset.*

---

## Streamlit Application

The Streamlit application provides an interactive platform to forecast energy consumption:

* Users can input **Hour, Day of Week, Month**, and **select a Model**.
* Predictions are generated for the next **6 intervals** (1 hour at 10-minute granularity).
* Outputs include both **numerical predictions** and a **line chart visualization**.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook to train models and save them.

3. Launch the Streamlit app:
   ```bash
   streamlit run Streamlit_App.py
   ```

---
