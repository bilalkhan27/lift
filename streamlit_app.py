import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io

# Step setup
st.set_page_config(page_title="Lift Breakdown Forecasting", layout="wide")
st.title("🚀 Enhanced Lift Breakdown Dashboard")

progress = st.progress(0)
status_text = st.empty()

# Step 1: Upload file
uploaded_file = st.file_uploader("📁 Upload Breakdown Excel File", type=["xlsx"])
if not uploaded_file:
    st.info("Please upload a breakdown Excel file to proceed.")
    st.stop()

status_text.text("Step 1/5: Loading Excel file...")
progress.progress(10)

# Step 2: Read file and map columns
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

    # Safe datetime conversion
    for col in ["Actual Start", "Date Created", "Actual Finish"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")

    # Only fillna if column exists
    if "Actual Finish" in df.columns:
        df["Actual Finish"] = df["Actual Finish"].fillna(pd.Timestamp.now(tz))

    # Conditional derived columns
    if "Actual Start" in df.columns and "Date Created" in df.columns:
        df["Response Delay_hours"] = (df["Actual Start"] - df["Date Created"]).dt.total_seconds() / 3600
    if "Actual Finish" in df.columns and "Actual Start" in df.columns:
        df["Resolution_minutes"] = (df["Actual Finish"] - df["Actual Start"]).dt.total_seconds() / 60

    # Feature engineering only if "Date Created" exists
    if "Date Created" in df.columns:
        df["Hour"] = df["Date Created"].dt.hour
        df["DoW"] = df["Date Created"].dt.day_name()

    return df

df = load_data(uploaded_file)
status_text.text("Step 2/5: File loaded and cleaned.")
progress.progress(30)

# Step 3: Filtering (Optional UI Filters here)
with st.expander("🔍 Optional Filters"):
    if "Site" in df.columns:
        selected_sites = st.multiselect("Select Site(s)", options=df["Site"].unique(), default=df["Site"].unique())
        df = df[df["Site"].isin(selected_sites)]
    if "Fault" in df.columns:
        selected_faults = st.multiselect("Select Fault(s)", options=df["Fault"].unique(), default=df["Fault"].unique())
        df = df[df["Fault"].isin(selected_faults)]

status_text.text("Step 3/5: Filters applied.")
progress.progress(50)

# Step 4: Forecasting
def prepare_series(df, date_col='Date Created'):
    return df.groupby(df[date_col].dt.date).size().rename("Calls").asfreq("D", fill_value=0)

def forecast_prophet(series, horizon=7):
    df_prophet = series.reset_index()
    df_prophet.columns = ["ds", "y"]
    model = Prophet(interval_width=0.8, weekly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future).tail(horizon)
    return forecast

series = prepare_series(df)
forecast = forecast_prophet(series, horizon=7)

status_text.text("Step 4/5: Forecasting completed.")
progress.progress(80)

# Step 5: Show forecast and charts
st.subheader("📊 Forecast – Next 7 Days")
fig, ax = plt.subplots(figsize=(10, 4))
series.plot(ax=ax, label="Historical Calls")
forecast.set_index("ds")["yhat"].plot(ax=ax, label="Forecast")
ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.2,
    label="Confidence Interval"
)
ax.set_ylabel("Calls per Day")
ax.legend()
st.pyplot(fig)

# Forecast Table
st.write("📅 Forecast Table")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

status_text.text("✅ Step 5/5: Visualization complete. Dashboard ready!")
progress.progress(100)
