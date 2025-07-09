import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from math import sqrt
import pytz
from io import BytesIO

# Page setup
st.set_page_config(page_title="📈 Lift Breakdown Forecasting Dashboard", layout="wide")
st.title("🚀 Enhanced Lift Breakdown Forecasting")

# File uploader
uploaded_file = st.file_uploader("📁 Upload Breakdown Excel File", type=["xlsx"])
if not uploaded_file:
    st.info("Please upload a breakdown Excel file to continue.")
    st.stop()

# Column mapping for flexibility
COLUMN_MAP = {
    "Actual Start": ["Actual Start", "Actual_Start", "Start Time", "Start", "ActualStart"],
    "Date Created": ["Date Created", "Date_Created", "Reported Time", "Created", "Breakdown Time"],
    "Actual Finish": ["Actual Finish", "Actual_Finish", "Finish Time", "End Time", "ActualFinish"],
    "Site": ["Site", "Site ID", "Location", "Building"],
    "Fault": ["Fault", "Fault Code", "Error", "Error Type"],
}

# Helper to match column aliases
def _find_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None

# Data loader
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

# Load and clean
df = load_data(uploaded_file)

# Show sample
st.subheader("📋 Uploaded Data Sample")
st.dataframe(df.head(50))

# Forecast horizon selector
horizon = st.slider("🔮 Forecast Horizon (days)", 7, 30, 7)

# Prepare time series
series = df.groupby(df["Date Created"].dt.date).size().rename("Calls")
series.index = pd.to_datetime(series.index)
series = series.asfreq("D", fill_value=0)

# Prophet forecasting
df_prophet = series.reset_index()
df_prophet.columns = ["ds", "y"]
prophet = Prophet(interval_width=0.9, weekly_seasonality=True)
prophet.fit(df_prophet)
future = prophet.make_future_dataframe(periods=horizon)
forecast = prophet.predict(future)

# Convert x-axis to datetime for fill_between
forecast["ds"] = pd.to_datetime(forecast["ds"])

# STL decomposition
stl = STL(series, seasonal=7)
res = stl.fit()

# Forecast chart
st.subheader("📈 Forecasted Breakdown Calls")
fig, ax = plt.subplots(figsize=(10, 4))
series.plot(ax=ax, label="Historical")
forecast.set_index("ds")["yhat"].plot(ax=ax, label="Forecast")
ax.fill_between(
    forecast["ds"].tail(horizon).values,
    forecast["yhat_lower"].tail(horizon).values.astype(float),
    forecast["yhat_upper"].tail(horizon).values.astype(float),
    alpha=0.2, label="Confidence Interval"
)
ax.legend()
st.pyplot(fig)

# STL Decomposition
st.subheader("📉 Trend Decomposition")
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axs[0].plot(res.trend); axs[0].set_title("Trend")
axs[1].plot(res.seasonal); axs[1].set_title("Seasonal")
axs[2].plot(res.resid); axs[2].set_title("Residual")
st.pyplot(fig)

# Breakdown per Site
if "Site" in df.columns:
    st.subheader("📊 Breakdown Volume per Site Over Time")
    site_group = df.groupby([df["Date Created"].dt.date, "Site"]).size().unstack().fillna(0)
    st.line_chart(site_group)

# Monthly Fault Frequency
if "Fault" in df.columns:
    st.subheader("📅 Monthly Fault Frequency Trend")
    fault_monthly = df.groupby(["Month", "Fault"]).size().unstack().fillna(0)
    st.bar_chart(fault_monthly)

# Weekly Heatmap
st.subheader("🕓 Weekly Heatmap (Hour vs Day)")
heatmap_data = df.groupby(["DoW", "Hour"]).size().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# Fault Pie Chart
if "Fault" in df.columns:
    st.subheader("📌 Fault Share Pie Chart")
    fault_counts = df["Fault"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(fault_counts, labels=fault_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)

# Forecast Table
st.subheader("🗓️ Forecast Table")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).round(2))

# CSV Export
csv = forecast.to_csv(index=False).encode()
st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

# Summary
st.subheader("📌 Dashboard Summary")
st.markdown(f"""
- 📅 **Forecast Horizon:** Next {horizon} Days  
- 📞 **Total Forecasted Calls:** {int(forecast.iloc[-horizon:]['yhat'].sum())}  
- ⏱️ **Average Resolution Time:** {
    f"{df['Resolution_minutes'].mean():.1f} mins" if 'Resolution_minutes' in df.columns else "N/A"
}
""")
