
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta


st.set_page_config(page_title="Energy Forecast App", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        color: #e94560;
        margin-bottom: 20px;
    }

    .stButton>button {
        background: linear-gradient(to right, #e94560, #c23152);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .info-box {
        background-color: #16213edd;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0px 6px 12px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Title ---
st.title("⚡ Appliances Energy Forecast")
st.markdown("Predict household appliance energy consumption for the next hour using ML models.")


# --- Sidebar ---
st.sidebar.header("🔧 Configuration")

selected_hour = st.sidebar.selectbox("Select Hour", list(range(24)))
selected_day = st.sidebar.selectbox("Select Day (0 = Monday)", list(range(7)))
selected_month = st.sidebar.selectbox("Select Month", list(range(1, 13)))

model_name = st.sidebar.radio(
    "Choose Model",
    ["XGBoost", "Random Forest", "Gradient Boosting", "LightGBM"]
)

model_scores = {
    "Random Forest": "0.49",
    "Gradient Boosting": "0.39",
    "XGBoost": "0.59",
    "LightGBM": "0.52",
}

st.markdown(f"**Selected:** {model_name} (R² ≈ {model_scores.get(model_name)})")
st.sidebar.markdown("---")


# --- Load Models and Data ---
@st.cache_resource
def load_resources():
    rf = joblib.load("Models/random_forest.pkl")
    gb = joblib.load("Models/gradient_boosting.pkl")
    xgb_model = joblib.load("Models/xgb_model.pkl")
    lgb_model = joblib.load("Models/lgb_model.pkl")

    df_rf = pd.read_csv("Dataframes/df_rf.csv")
    df_gb = pd.read_csv("Dataframes/df_gb.csv")
    df_xgb = pd.read_csv("Dataframes/df_xgb.csv")
    df_lgb = pd.read_csv("Dataframes/df_lgb.csv")

    return rf, gb, xgb_model, lgb_model, df_rf, df_gb, df_xgb, df_lgb


rf, gb, xgb_model, lgb_model, df_rf, df_gb, df_xgb, df_lgb = load_resources()


# --- Helper: get matching rows ---
def get_matching_rows(df, hour, day, month):
    subset = df[(df['Hour'] == hour) & (df['DayOfWeek'] == day) & (df['Month'] == month)]
    if subset.empty:
        st.warning("⚠️ No matching data found for this combination. Try different inputs.")
        return None
    return subset.head(6)


# --- Prediction functions ---
def predict_with_model(model, df, hour, day, month):
    features = [col for col in df.columns if col != 'Appliances']
    data = get_matching_rows(df, hour, day, month)
    if data is not None:
        return model.predict(data[features])
    return None


# --- Predict Button ---
if st.button("🔮 Predict Energy Usage"):
    with st.spinner("Running prediction..."):

        predictions = None

        if model_name == "Random Forest":
            predictions = predict_with_model(rf, df_rf, selected_hour, selected_day, selected_month)
        elif model_name == "Gradient Boosting":
            predictions = predict_with_model(gb, df_gb, selected_hour, selected_day, selected_month)
        elif model_name == "XGBoost":
            predictions = predict_with_model(xgb_model, df_xgb, selected_hour, selected_day, selected_month)
        elif model_name == "LightGBM":
            predictions = predict_with_model(lgb_model, df_lgb, selected_hour, selected_day, selected_month)

        if predictions is not None:
            st.success(f"✅ Prediction Complete using {model_name}")
            st.subheader("📈 Forecast (Next 1 Hour):")

            # timestamps at 10-min intervals
            base_time = datetime(2025, 1, 1, selected_hour, 0, 0)
            time_stamps = [(base_time + timedelta(minutes=10 * (i + 1))) for i in range(6)]
            formatted_times = [ts.strftime("%I:%M %p") for ts in time_stamps]

            prediction_df = pd.DataFrame({
                "Time": formatted_times,
                "Predicted Energy (Wh)": predictions[:6]
            })

            # show table
            st.dataframe(prediction_df.style.format({"Predicted Energy (Wh)": "{:.2f}"}))
            st.markdown("📆 Predicted appliance energy consumption at 10-minute intervals.")

            # show chart
            st.line_chart(
                data=prediction_df.set_index("Time"),
                use_container_width=True
            )
