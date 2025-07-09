
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pytz

# Page setup
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

st.subheader("📋 Uploaded Data Sample")
st.dataframe(df.head(50))

horizon = st.slider("🔮 Forecast Horizon (days)", 7, 30, 7)
series = df.groupby(df["Date Created"].dt.date).size().rename("Calls")
series.index = pd.to_datetime(series.index)
series = series.asfreq("D", fill_value=0)

# Prophet model
df_prophet = series.reset_index()
df_prophet.columns = ["ds", "y"]
prophet = Prophet(interval_width=0.9, weekly_seasonality=True)
prophet.fit(df_prophet)
future = prophet.make_future_dataframe(periods=horizon)
forecast_prophet = prophet.predict(future)
forecast_prophet["model"] = "Prophet"

# ARIMA model
arima_model = ARIMA(series, order=(2,1,2))
arima_result = arima_model.fit()
forecast_arima = arima_result.get_forecast(steps=horizon)
forecast_arima_df = forecast_arima.summary_frame()
forecast_arima_df["ds"] = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
forecast_arima_df.rename(columns={"mean": "yhat", "mean_ci_lower": "yhat_lower", "mean_ci_upper": "yhat_upper"}, inplace=True)
forecast_arima_df["model"] = "ARIMA"

# RMSE comparison
prophet_rmse = sqrt(mean_squared_error(series[-horizon:], forecast_prophet.set_index("ds")["yhat"][:horizon]))
arima_rmse = sqrt(mean_squared_error(series[-horizon:], forecast_arima_df.set_index("ds")["yhat"][:horizon]))

best_model = "Prophet" if prophet_rmse < arima_rmse else "ARIMA"

# STL
stl = STL(series, seasonal=7)
res = stl.fit()

# Forecast chart with model comparison
st.subheader("📈 Forecasted Breakdown Calls")
fig, ax = plt.subplots(figsize=(10, 4))
series.plot(ax=ax, label="Historical")
forecast_prophet.set_index("ds")["yhat"].plot(ax=ax, label="Prophet Forecast")
forecast_arima_df.set_index("ds")["yhat"].plot(ax=ax, linestyle="--", label="ARIMA Forecast")
ax.fill_between(
    forecast_prophet["ds"].tail(horizon).values,
    forecast_prophet["yhat_lower"].tail(horizon).values,
    forecast_prophet["yhat_upper"].tail(horizon).values,
    alpha=0.2, label="Prophet CI"
)
ax.fill_between(
    forecast_arima_df["ds"].values,
    forecast_arima_df["yhat_lower"].values,
    forecast_arima_df["yhat_upper"].values,
    alpha=0.1, color="gray", label="ARIMA CI"
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

# Model comparison summary
st.subheader("🤖 Model Comparison")
st.markdown(f'''
- **Prophet RMSE:** {prophet_rmse:.2f}  
- **ARIMA RMSE:** {arima_rmse:.2f}  
- ✅ **Best Performing Model:** `{best_model}`
''')

# Forecast table from best model
st.subheader("🗓️ Forecast Table")
if best_model == "Prophet":
    display_df = forecast_prophet[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).round(2)
else:
    display_df = forecast_arima_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].round(2)
st.dataframe(display_df)

# CSV Export
csv = display_df.to_csv(index=False).encode()
st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

# Summary
st.subheader("📌 Dashboard Summary")
st.markdown(f'''
- 📅 **Forecast Horizon:** Next {horizon} Days  
- 📞 **Total Forecasted Calls:** {int(display_df["yhat"].sum())}  
- 🧠 **Best Model Used:** {best_model}  
- ⏱️ **Average Resolution Time:** {f"{df["Resolution_minutes"].mean():.1f} mins" if "Resolution_minutes" in df.columns else "N/A"}
''')
