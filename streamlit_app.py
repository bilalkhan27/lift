

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
import io
import tempfile
import datetime as dt
from fpdf import FPDF
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

# ---------------------------------------------------------------------------------
# Helper: Load and Clean Data
# ---------------------------------------------------------------------------------

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
@st.cache_data(show_spinner=False)
def load_data(xlsx_file):
    st.info("🔄 Loading uploaded Excel file...")

    # Use BytesIO to avoid ImportError for openpyxl
    try:
        df = pd.read_excel(io.BytesIO(xlsx_file.read()))
    except Exception as e:
        st.error(f"❌ Failed to load Excel file: {e}")
        return pd.DataFrame()

    rename_map = {}
    for canonical, aliases in COLUMN_MAP.items():
        found_col = _find_column(df, aliases)
        if found_col:
            rename_map[found_col] = canonical
        else:
            st.warning(f"⚠️ Could not find column: '{canonical}' – check aliases: {aliases}")
    df = df.rename(columns=rename_map)

    # Timezone setup
    tz = pytz.timezone("Europe/London")
    for col in ["Actual Start", "Date Created", "Actual Finish"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    if "Actual Finish" in df.columns:
        df["Actual Finish"] = df["Actual Finish"].fillna(pd.Timestamp.now(tz))

    # Time differences
    if "Actual Start" in df.columns and "Date Created" in df.columns:
        df["Response Delay_hours"] = (df["Actual Start"] - df["Date Created"]).dt.total_seconds() / 3600
    if "Actual Finish" in df.columns and "Actual Start" in df.columns:
        df["Resolution_minutes"] = (df["Actual Finish"] - df["Actual Start"]).dt.total_seconds() / 60

    if "Date Created" in df.columns:
        df["Hour"] = df["Date Created"].dt.hour
        df["DoW"] = df["Date Created"].dt.day_name()

    st.success("✅ Data loaded and cleaned successfully.")
    st.dataframe(df.head(10))
    return df

# ---------------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------------

def prepare_series(df, date_col='Date Created'):
    series = df.groupby(df[date_col].dt.date).size().rename("Calls").asfreq("D", fill_value=0)
    median, mad = series.median(), np.median(np.abs(series - series.median()))
    clipped = series.clip(lower=median - 3 * 1.4826 * mad, upper=median + 3 * 1.4826 * mad)
    return clipped

def forecast_sarima(series, horizon):
    if PMDARIMA_AVAILABLE and len(series) >= 14:
        model = auto_arima(series, seasonal=True, m=7, trace=False, error_action="ignore", suppress_warnings=True)
        order, s_order = model.order, model.seasonal_order
    else:
        order, s_order = (1, 1, 1), (1, 0, 1, 7)
    sarima_model = SARIMAX(series, order=order, seasonal_order=s_order)
    sarima_fit = sarima_model.fit(disp=False)
    return sarima_fit.forecast(horizon)

def forecast_prophet(series, horizon):
    df_prophet = series.reset_index()
    df_prophet.columns = ["ds", "y"]
    model = Prophet(interval_width=0.8, weekly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(horizon)
    forecast = model.predict(future).tail(horizon)
    return forecast

def get_baseline_forecasts(series, horizon):
    return np.repeat(series.iloc[-1], horizon), np.repeat(series[-min(7, len(series)):].mean(), horizon)

def evaluate_models(actual, sarima_forecast, prophet_forecast, naive, ma7):
    return pd.DataFrame({
        "Model": ["SARIMA", "Prophet", "Naïve", "7-day MA"],
        "MAE": [
            mean_absolute_error(actual, sarima_forecast[:len(actual)]),
            mean_absolute_error(actual, prophet_forecast.yhat[:len(actual)]),
            mean_absolute_error(actual, naive[:len(actual)]),
            mean_absolute_error(actual, ma7[:len(actual)])
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(actual, sarima_forecast[:len(actual)])),
            np.sqrt(mean_squared_error(actual, prophet_forecast.yhat[:len(actual)])),
            np.sqrt(mean_squared_error(actual, naive[:len(actual)])),
            np.sqrt(mean_squared_error(actual, ma7[:len(actual)]))
        ]
    })

# ---------------------------------------------------------------------------------
# App Logic
# ---------------------------------------------------------------------------------

st.set_page_config(page_title="Hart Lifts | Breakdown Dashboard", layout="wide")
st.title("🚀 Enhanced Lift Breakdown Dashboard")

uploaded_file = st.file_uploader("📁 Upload Breakdown Excel File", type=["xlsx"])
if not uploaded_file:
    st.info("Please upload a breakdown Excel file to proceed.")
    st.stop()

df = load_data(uploaded_file)
series = prepare_series(df)
horizon = 7

sarima_fc = forecast_sarima(series, horizon)
prophet_fc = forecast_prophet(series, horizon)
naive_fc, ma7_fc = get_baseline_forecasts(series, horizon)

actual = series[-horizon:] if len(series) >= horizon else series
eval_table = evaluate_models(actual, sarima_fc, prophet_fc, naive_fc, ma7_fc)
best_model = eval_table.loc[eval_table["RMSE"].idxmin(), "Model"]

st.subheader("📊 Forecast Summary")
st.write(eval_table)

# Forecast chart
st.subheader(f"📅 {best_model} Forecast – Next {horizon} Days")
fig, ax = plt.subplots(figsize=(10, 4))
series.plot(ax=ax, label="History", lw=1)
if best_model == "Prophet":
    yhat = prophet_fc.set_index("ds")["yhat"]
    prophet_fc.set_index("ds")["yhat"].plot(ax=ax, label="Prophet Forecast")
    ax.fill_between(yhat.index, prophet_fc.set_index("ds")["yhat_lower"], prophet_fc.set_index("ds")["yhat_upper"], alpha=0.2)
else:
    sarima_fc.plot(ax=ax, label="SARIMA Forecast", linestyle="--")
ax.set_ylabel("Calls per day")
ax.legend()
ax.grid(True, ls=":", lw=0.4)
st.pyplot(fig)
