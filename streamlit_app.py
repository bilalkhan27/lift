import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import io

st.set_page_config(page_title="Lift Breakdown Forecasting", layout="wide")
st.title("🚀 Enhanced Lift Breakdown Dashboard")

progress = st.progress(0)
status_text = st.empty()

# Upload
uploaded_file = st.file_uploader("📁 Upload Breakdown Excel File", type=["xlsx"])
if not uploaded_file:
    st.info("Please upload a breakdown Excel file to proceed.")
    st.stop()

# Map column names
COLUMN_MAP = {
    "Actual Start": ["Actual Start", "Actual_Start", "Start Time", "Start", "ActualStart"],
    "Date Created": ["Date Created", "Date_Created", "Reported Time", "Created", "Breakdown Time"],
    "Actual Finish": ["Actual Finish", "Actual_Finish", "Finish Time", "End Time", "ActualFinish"],
    "Site": ["Site", "Site ID", "Location", "Building"],
    "Fault": ["Fault", "Fault Code", "Error", "Error Type"],
}

def _find_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None

def load_data(xlsx_file):
    df = pd.read_excel(io.BytesIO(xlsx_file.read()))
    rename_map = {}
    for canonical, aliases in COLUMN_MAP.items():
        found_col = _find_column(df, aliases)
        if found_col:
            rename_map[found_col] = canonical
    df = df.rename(columns=rename_map)

    tz = pytz.timezone("Europe/London")
    for col in ["Actual Start", "Date Created", "Actual Finish"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")

    if "Actual Finish" in df.columns:
        df["Actual Finish"] = df["Actual Finish"].fillna(pd.Timestamp.now(tz))
    if "Actual Start" in df.columns and "Date Created" in df.columns:
        df["Response Delay_hours"] = (df["Actual Start"] - df["Date Created"]).dt.total_seconds() / 3600
    if "Actual Finish" in df.columns and "Actual Start" in df.columns:
        df["Resolution_minutes"] = (df["Actual Finish"] - df["Actual Start"]).dt.total_seconds() / 60
    if "Date Created" in df.columns:
        df["Hour"] = df["Date Created"].dt.hour
        df["DoW"] = df["Date Created"].dt.day_name()

    return df

status_text.text("Step 1/5: Loading and processing data...")
progress.progress(20)
df = load_data(uploaded_file)

progress.progress(40)

# Forecast Prep
def prepare_series(df, date_col="Date Created"):
    return df.groupby(df[date_col].dt.date).size().rename("Calls").asfreq("D", fill_value=0)

series = prepare_series(df)
progress.progress(60)

# Prophet Forecast
df_prophet = series.reset_index()
df_prophet.columns = ["ds", "y"]
prophet = Prophet(interval_width=0.8, weekly_seasonality=True)
prophet.fit(df_prophet)
future = prophet.make_future_dataframe(periods=7)
forecast = prophet.predict(future)

# SARIMA Forecast
sarima_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
sarima_result = sarima_model.fit(disp=False)
sarima_forecast = sarima_result.forecast(steps=7)

# Naive and Moving Average
naive_forecast = [series[-1]] * 7
moving_avg_forecast = [series[-7:].mean()] * 7

progress.progress(80)

# RMSE Calculation
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

actual = series[-7:].values
prophet_rmse = rmse(actual, forecast.iloc[-7:]["yhat"].values)
sarima_rmse = rmse(actual, sarima_forecast.values)
naive_rmse = rmse(actual, naive_forecast)
ma_rmse = rmse(actual, moving_avg_forecast)

best_model = "Prophet" if prophet_rmse < min(sarima_rmse, naive_rmse, ma_rmse) else "Other"

# 📊 Line Chart Forecast
st.subheader("📊 Breakdown Call Forecast – Next 7 Days")
fig, ax = plt.subplots(figsize=(10, 4))
series.plot(ax=ax, label="Historical")
forecast.set_index("ds")["yhat"].plot(ax=ax, label="Prophet Forecast")
ax.fill_between(
    forecast["ds"].tail(7).values,
    forecast["yhat_lower"].tail(7).values,
    forecast["yhat_upper"].tail(7).values,
    alpha=0.2,
    label="Confidence Interval"
)
ax.legend()
ax.set_ylabel("Calls per Day")
st.pyplot(fig)

# 🔢 Forecast Table
st.subheader("📅 7-Day Forecast Table")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).rename(columns={
    'ds': 'Date', 'yhat': 'Predicted Calls', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'
}))

# 📈 Forecast RMSE Table
st.subheader("📈 Forecast RMSE Comparison")
st.write(pd.DataFrame({
    "Model": ["Prophet", "SARIMA", "Naïve", "7-day Moving Avg"],
    "RMSE": [prophet_rmse, sarima_rmse, naive_rmse, ma_rmse]
}).sort_values("RMSE"))

# 📋 Summary Table
st.subheader("🧾 Dashboard Summary")
summary_data = {
    "Best-scoring model": [f"{best_model} (RMSE={round(min(prophet_rmse, sarima_rmse, naive_rmse, ma_rmse), 2)})"],
    "Forecast horizon": [f"{int(forecast.iloc[-7:]['yhat'].sum())} calls (≈{round(forecast.iloc[-7:]['yhat'].mean(), 1)} daily avg)"],
    "Avg response delay": [f"{df['Response Delay_hours'].mean():.1f} hours" if 'Response Delay_hours' in df else "N/A"],
    "Avg resolution time": [f"{df['Resolution_minutes'].mean():.1f} min" if 'Resolution_minutes' in df else "N/A"],
    "Top Sites": [", ".join(df['Site'].value_counts().head(3).index) if 'Site' in df else "N/A"],
    "Top Faults": [", ".join(df['Fault'].value_counts().head(3).index) if 'Fault' in df else "N/A"],
}
st.table(pd.DataFrame(summary_data))

# 📄 Show top of raw data
st.subheader("📄 Raw Data Preview")
st.dataframe(df.head(10))

progress.progress(100)
status_text.text("✅ Dashboard complete!")
