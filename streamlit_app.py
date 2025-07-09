
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from io import BytesIO
import pytz

st.set_page_config(page_title="Lift Breakdown Forecasting Dashboard", layout="wide")
st.title("Enhanced Lift Breakdown Forecasting")

uploaded_file = st.file_uploader("Upload Breakdown Excel File", type=["xlsx"])
if not uploaded_file:
    st.info("Please upload a breakdown Excel file to continue.")
    st.stop()

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

def load_data(file):
    df = pd.read_excel(file)
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
        df["Month"] = df["Date Created"].dt.to_period("M")
    return df

df = load_data(uploaded_file)

# Dashboard Summary
st.markdown("### Dashboard Summary")
total_breakdowns = df.shape[0]
avg_resolution = df["Resolution_minutes"].mean() if "Resolution_minutes" in df.columns else None
num_sites = df["Site"].nunique() if "Site" in df.columns else "N/A"
num_faults = df["Fault"].nunique() if "Fault" in df.columns else "N/A"
st.markdown(f'''
- Total Breakdowns: {total_breakdowns}
- Average Resolution Time: {avg_resolution:.1f} mins
- Sites Covered: {num_sites}
- Fault Types: {num_faults}
''')

# Forecasting
series = df.groupby(df["Date Created"].dt.date).size().rename("Calls").asfreq("D", fill_value=0)
horizon = st.slider("Forecast Horizon (days)", 7, 30, 7)
df_prophet = series.reset_index()
df_prophet.columns = ["ds", "y"]
model = Prophet(interval_width=0.9, weekly_seasonality=True)
model.fit(df_prophet)
future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

# Forecast Chart
st.subheader("Forecasted Breakdown Calls")
fig, ax = plt.subplots(figsize=(10, 4))
series.plot(ax=ax, label="Historical")
forecast.set_index("ds")["yhat"].plot(ax=ax, label="Forecast")
ax.fill_between(forecast["ds"].tail(horizon),
                forecast["yhat_lower"].tail(horizon),
                forecast["yhat_upper"].tail(horizon),
                alpha=0.2, label="Confidence Interval")
ax.legend()
st.pyplot(fig)

# Forecast Table and Trend Line
st.subheader("Forecast Table")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).round(2))
fig2, ax2 = plt.subplots()
forecast_tail = forecast.tail(horizon)
ax2.plot(forecast_tail["ds"], forecast_tail["yhat"], marker='o', label="Forecasted Calls")
ax2.fill_between(forecast_tail["ds"], forecast_tail["yhat_lower"], forecast_tail["yhat_upper"], alpha=0.2)
ax2.set_title("Forecast Trend")
ax2.set_xlabel("Date")
ax2.set_ylabel("Predicted Calls")
ax2.legend()
st.pyplot(fig2)
