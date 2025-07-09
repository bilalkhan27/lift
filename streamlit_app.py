import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import io
import pytz

st.set_page_config(page_title="\ud83d\udcc8 Lift Breakdown Forecasting Dashboard", layout="wide")
st.title("\ud83d\ude80 Enhanced Lift Breakdown Forecasting")

uploaded_file = st.file_uploader("\ud83d\udcc1 Upload Breakdown Excel File", type=["xlsx"])
if not uploaded_file:
    st.info("Please upload a breakdown Excel file to continue.")
    st.stop()

# Mapping actual column names
COLUMN_MAP = {
    "Actual Start": ["Actual Start"],
    "Date Created": ["Date Created"],
    "Actual End": ["Actual End"],
    "Site": ["Site ID (Building Location) (Building Location)"],
    "Fault": ["Fault Code"]
}

def _find_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None

def load_data(file):
    df = pd.read_excel(io.BytesIO(file.read()), sheet_name=0)
    rename_map = {}
    for canonical, aliases in COLUMN_MAP.items():
        found_col = _find_column(df, aliases)
        if found_col:
            rename_map[found_col] = canonical
    df = df.rename(columns=rename_map)

    tz = pytz.timezone("Europe/London")
    for col in ["Actual Start", "Date Created", "Actual End"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")

    df["Actual End"] = df["Actual End"].fillna(pd.Timestamp.now(tz))
    df["Resolution_minutes"] = (df["Actual End"] - df["Actual Start"]).dt.total_seconds() / 60
    df["Hour"] = df["Date Created"].dt.hour
    df["DoW"] = df["Date Created"].dt.day_name()
    return df


df = load_data(uploaded_file)

st.subheader("\ud83d\udccb Uploaded Data Sample")
st.dataframe(df.head(50))

# Time Series Preparation
def prepare_series(df, date_col="Date Created"):
    return df.groupby(df[date_col].dt.date).size().rename("Calls").asfreq("D", fill_value=0)

series = prepare_series(df)

horizon = st.slider("\ud83d\udd2e Select Forecast Horizon (days)", 7, 30, 7)

# Prophet Forecast
df_prophet = series.reset_index()
df_prophet.columns = ["ds", "y"]
prophet = Prophet(interval_width=0.8, weekly_seasonality=True)
prophet.fit(df_prophet)
future = prophet.make_future_dataframe(periods=horizon)
forecast = prophet.predict(future)

# SARIMA Forecast
sarima_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
sarima_result = sarima_model.fit(disp=False)
sarima_forecast = sarima_result.forecast(steps=horizon)

# Evaluation
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

actual = series[-horizon:].values
prophet_rmse = rmse(actual, forecast.iloc[-horizon:]["yhat"].values)
sarima_rmse = rmse(actual, sarima_forecast.values)
naive_rmse = rmse(actual, [series[-1]] * horizon)
ma_rmse = rmse(actual, [series[-7:].mean()] * horizon)

best_model = min(
    [("Prophet", prophet_rmse), ("SARIMA", sarima_rmse), ("Naive", naive_rmse), ("Moving Avg", ma_rmse)],
    key=lambda x: x[1]
)[0]

# Forecast Chart
st.subheader(f"\ud83d\udcc8 Forecasted Breakdown Calls – Next {horizon} Days")
fig, ax = plt.subplots(figsize=(10, 4))
series.plot(ax=ax, label="Historical")
forecast.set_index("ds")["yhat"].plot(ax=ax, label="Prophet Forecast")
ax.fill_between(forecast["ds"].tail(horizon).values,
                forecast["yhat_lower"].tail(horizon).values,
                forecast["yhat_upper"].tail(horizon).values,
                alpha=0.2, label="Confidence Interval")
ax.set_ylabel("Calls per Day")
ax.legend()
st.pyplot(fig)

# Forecast Table
st.subheader("\ud83d\uddd3\ufe0f Forecast Table")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).round(2))

# RMSE
st.subheader("\ud83d\udcca RMSE Comparison")
rmse_df = pd.DataFrame({
    "Model": ["Prophet", "SARIMA", "Naive", "Moving Avg"],
    "RMSE": [prophet_rmse, sarima_rmse, naive_rmse, ma_rmse]
}).sort_values("RMSE")
st.dataframe(rmse_df.style.highlight_min(subset=["RMSE"], color="#85C1E9"))

# Site/Faults
if "Site" in df.columns:
    st.subheader("\ud83c\udfe2 Top Sites by Breakdown Calls")
    top_sites = df["Site"].value_counts().head(10)
    st.bar_chart(top_sites)

if "Fault" in df.columns:
    st.subheader("\u26a0\ufe0f Top Fault Types")
    top_faults = df["Fault"].value_counts().head(10)
    st.bar_chart(top_faults)

# Summary
st.subheader("\ud83d\udccc Dashboard Summary")
avg_resolution = df["Resolution_minutes"].mean() if "Resolution_minutes" in df.columns else None
st.markdown(f"""
- \u2705 **Best Performing Model:** **{best_model}**
- \ud83d\udcde **Total Forecasted Calls (next {horizon} days):** **{int(forecast.iloc[-horizon:]['yhat'].sum())}**
- \ud83d\udee0\ufe0f **Average Resolution Time:** {f"{avg_resolution:.1f} mins" if avg_resolution is not None else "N/A"}
""")
