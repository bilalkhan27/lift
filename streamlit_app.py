
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import io
import pytz

st.set_page_config(page_title="📈 Lift Breakdown Forecasting Dashboard", layout="wide")
st.title("🚀 Enhanced Lift Breakdown Forecasting")

uploaded_file = st.file_uploader("📁 Upload Breakdown Excel File", type=["xlsx"])
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
    df = pd.read_excel(io.BytesIO(file.read()))
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

df = load_data(uploaded_file)

# Show raw data
st.subheader("📋 Uploaded Data Sample")
st.dataframe(df.head(50))

# Time series prep
def prepare_series(df, date_col="Date Created"):
    return df.groupby(df[date_col].dt.date).size().rename("Calls").asfreq("D", fill_value=0)

series = prepare_series(df)

# Forecast horizon selector
horizon = st.slider("🔮 Select Forecast Horizon (days)", 7, 30, 7)

# Prophet
df_prophet = series.reset_index()
df_prophet.columns = ["ds", "y"]
prophet = Prophet(interval_width=0.8, weekly_seasonality=True)
prophet.fit(df_prophet)
future = prophet.make_future_dataframe(periods=horizon)
forecast = prophet.predict(future)

# SARIMA
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

# Visual Forecast
st.subheader(f"📈 Forecasted Breakdown Calls – Next {horizon} Days")
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
st.subheader("🗓️ Forecast Table")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).round(2))

# RMSE Table
st.subheader("📊 RMSE Comparison")
rmse_df = pd.DataFrame({
    "Model": ["Prophet", "SARIMA", "Naive", "Moving Avg"],
    "RMSE": [prophet_rmse, sarima_rmse, naive_rmse, ma_rmse]
}).sort_values("RMSE")
st.dataframe(rmse_df.style.highlight_min(subset=["RMSE"], color="#85C1E9"))

# Top Sites and Faults
if "Site" in df.columns:
    st.subheader("🏢 Top Sites by Breakdown Calls")
    top_sites = df["Site"].value_counts().head(10)
    st.bar_chart(top_sites)

if "Fault" in df.columns:
    st.subheader("⚠️ Top Fault Types")
    top_faults = df["Fault"].value_counts().head(10)
    st.bar_chart(top_faults)

# Summary
st.subheader("📌 Dashboard Summary")
st.markdown(f"""
- ✅ **Best Performing Model:** **{best_model}**
- 📞 **Total Forecasted Calls (next {horizon} days):** **{int(forecast.iloc[-horizon:]['yhat'].sum())}**
- 🛠️ **Average Resolution Time:** {
    f"{df['Resolution_minutes'].mean():.1f} mins"
    if 'Resolution_minutes' in df.columns else "N/A"
}

- 🛠️ **Average Resolution Time:** {df['Resolution_minutes'].mean():.1f} mins
""")
 
