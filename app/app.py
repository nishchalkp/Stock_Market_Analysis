import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Load saved SARIMAX model
@st.cache_resource
def load_model():
    return joblib.load("eda/timeseries.pkl")


model = load_model()

# Streamlit UI
st.title("Time Series Forecasting with SARIMAX")

# User Input: Number of days to forecast
n_months = st.number_input("Enter number of months to predict:", min_value=1, max_value=60, value=12)


# Predict Function
def predict_future(n_months):
    pred = model.get_forecast(steps=n_months)
    pred_ci = pred.conf_int()
    return pred.predicted_mean, pred_ci


# Forecasting
if st.button("Predict"):
    forecast, confidence_interval = predict_future(n_months)

    # Display results
    last_date = pd.Timestamp("2019-12")  # Adjust based on your dataset
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq="M")
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})

    st.write("### Forecasted Values")
    st.dataframe(forecast_df)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(forecast_dates, forecast, label="Predicted", color="blue")
    ax.fill_between(forecast_dates, confidence_interval.iloc[:, 0], confidence_interval.iloc[:, 1], color="lightblue",
                    alpha=0.3)
    ax.legend()
    st.pyplot(fig)

