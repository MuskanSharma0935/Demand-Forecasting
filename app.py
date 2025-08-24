
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import timedelta

# Optional / heavy libs are imported lazily inside functions
# statsmodels for Holt-Winters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# We'll try to use pmdarima for auto_arima; if not available, we'll fallback to a simple ARIMA
try:
    import pmdarima as pm
    HAS_PM = True
except Exception:
    HAS_PM = False

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet (installed as 'prophet' package in modern versions)
def _import_prophet():
    try:
        from prophet import Prophet
        return Prophet
    except Exception:
        try:
            from fbprophet import Prophet  # legacy name
            return Prophet
        except Exception:
            return None

st.set_page_config(page_title="Demand Forecasting App", layout="wide")

st.title("📈 Demand Forecasting (Holt-Winters • ARIMA • Prophet)")
st.write(
    "Upload a time-series CSV, pick your **date** and **target** columns, choose a model, and generate forecasts. "
    "The app supports **Holt-Winters**, **ARIMA** (auto_arima when available), and **Prophet**."
)

with st.expander("ℹ️ Data format help", expanded=False):
    st.markdown(
        """
        **CSV requirements**
        - Must include a date/time column (e.g., `date`, `ds`, `timestamp`) and a numeric target column (e.g., `sales`, `y`).
        - One row per time period. If you have multiple stores/items, filter to one series before uploading.
        - Tip: Missing dates will be auto-filled when you choose a frequency below.
        """
    )

# -------- Sidebar controls --------
st.sidebar.header("1) Upload & Configure Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

freq = st.sidebar.selectbox("Frequency", options=["D (Daily)", "W (Weekly)", "M (Monthly)", "Q (Quarterly)"], index=2)
freq_map = {"D (Daily)": "D", "W (Weekly)": "W", "M (Monthly)": "MS", "Q (Quarterly)": "QS"}
pd_freq = freq_map[freq]

season_default = 12 if pd_freq.endswith("S") else (7 if pd_freq == "D" else 12)
season_length = st.sidebar.number_input("Season length (periods)", min_value=1, value=season_default, step=1,
                                        help="E.g., 7 for weekly seasonality on daily data, 12 for monthly, 4 for quarterly.")

st.sidebar.header("2) Train/Test Split & Horizon")
train_ratio = st.sidebar.slider("Train share (%)", min_value=50, max_value=95, value=80, step=1)
horizon = st.sidebar.number_input("Forecast horizon (periods)", min_value=1, value=12, step=1)

st.sidebar.header("3) Choose Model")
model_name = st.sidebar.radio("Model", ["Holt-Winters", "ARIMA", "Prophet"], index=0)

advanced = st.sidebar.checkbox("Show advanced settings", value=False)
if advanced and model_name == "ARIMA":
    use_auto = st.sidebar.checkbox("Use auto_arima (pmdarima)", value=True, help="Uncheck to use a simple SARIMAX(1,1,1).")
    seasonal = st.sidebar.checkbox("Seasonal ARIMA", value=True)
else:
    use_auto = True
    seasonal = True

def prepare_series(df: pd.DataFrame, date_col: str, target_col: str, freq_code: str) -> pd.Series:
    s = df[[date_col, target_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    s = s.sort_values(date_col).set_index(date_col)[target_col].astype(float)
    # Resample to chosen frequency and forward-fill / interpolate gaps
    s = s.resample(freq_code).sum() if s.index.is_monotonic_increasing else s.asfreq(freq_code)
    if s.isna().any():
        s = s.interpolate(limit_direction="both")
    return s

def train_test_split_series(s: pd.Series, train_ratio: int):
    n = len(s)
    n_train = int(n * train_ratio / 100)
    train = s.iloc[:n_train]
    test = s.iloc[n_train:]
    return train, test

def metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

def fit_holt_winters(train: pd.Series, season_length: int):
    # Auto-select additive/multiplicative seasonality by inspecting variance; keep it simple
    seasonal = "add" if train.std() / (np.abs(train.mean()) + 1e-8) < 0.5 else "mul"
    trend = "add"
    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=season_length, initialization_method="estimated")
    fitted = model.fit(optimized=True)
    return fitted

def fit_arima(train: pd.Series, seasonal: bool, season_length: int, use_auto: bool):
    if use_auto and HAS_PM:
        m = season_length if seasonal else 0
        model = pm.auto_arima(
            train,
            seasonal=seasonal,
            m=season_length if seasonal else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        return model
    # Fallback: simple SARIMAX
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, season_length) if seasonal else (0, 0, 0, 0)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    return fitted

def forecast_arima(model, periods: int):
    # pmdarima and statsmodels have different APIs; handle both
    if HAS_PM and hasattr(model, "predict_in_sample"):
        # pmdarima
        fc = model.predict(n_periods=periods)
        return pd.Series(fc)
    else:
        # statsmodels SARIMAXResults
        fc = model.get_forecast(steps=periods)
        return fc.predicted_mean

def fit_prophet(train: pd.Series, freq_code: str, season_length: int):
    Prophet = _import_prophet()
    if Prophet is None:
        raise ImportError("Prophet is not installed. Please add 'prophet' to requirements.txt or pip install prophet.")
    df = train.reset_index()
    df.columns = ["ds", "y"]
    m = Prophet()
    # Add seasonality roughly aligned with chosen period
    if freq_code == "D":
        m.add_seasonality(name="weekly", period=7, fourier_order=6)
        m.add_seasonality(name="yearly", period=365.25, fourier_order=8)
    elif freq_code == "MS":
        m.add_seasonality(name="yearly", period=12, fourier_order=10)
    elif freq_code == "QS":
        m.add_seasonality(name="annual", period=4, fourier_order=6)
    m.fit(df)
    return m

def forecast_prophet(model, periods: int, last_date: pd.Timestamp, freq_code: str):
    future = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq_code), periods=periods, freq=freq_code)
    future_df = pd.DataFrame({"ds": future})
    fcst = model.predict(future_df)
    return fcst.set_index("ds")["yhat"]

def plot_series(train, test, pred_test=None, pred_future=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    train.plot(ax=ax, label="Train")
    if test is not None and len(test) > 0:
        test.plot(ax=ax, label="Test")
    if pred_test is not None:
        pred_test.index = test.index[:len(pred_test)]
        pred_test.plot(ax=ax, label="Prediction (test)")
    if pred_future is not None:
        pred_future.plot(ax=ax, label="Forecast (future)")
    ax.set_title("Time Series & Forecasts")
    ax.legend()
    st.pyplot(fig)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("👀 Data Preview")
    st.dataframe(df.head())

    # Column selectors
    cols = df.columns.tolist()
    date_col = st.selectbox("Date column", options=cols, index=0)
    target_col = st.selectbox("Target (numeric) column", options=cols, index=min(1, len(cols)-1))

    # Prepare series
    try:
        series = prepare_series(df, date_col, target_col, pd_freq)
    except Exception as e:
        st.error(f"Failed to parse/prepare your series: {e}")
        st.stop()

    st.write(f"Data points after resampling to **{pd_freq}**: {len(series)}")
    st.line_chart(series)

    train, test = train_test_split_series(series, train_ratio)

    if st.button("🚀 Run Forecast"):
        with st.spinner(f"Fitting {model_name} and forecasting..."):
            try:
                if model_name == "Holt-Winters":
                    fitted = fit_holt_winters(train, season_length=season_length)
                    # In-sample forecast for test
                    pred_test = fitted.forecast(steps=len(test)) if len(test) > 0 else None
                    pred_future = fitted.forecast(steps=horizon)

                elif model_name == "ARIMA":
                    model = fit_arima(train, seasonal=seasonal, season_length=season_length, use_auto=use_auto)
                    pred_test = forecast_arima(model, len(test)) if len(test) > 0 else None
                    pred_future = forecast_arima(model, horizon)

                else:  # Prophet
                    m = fit_prophet(train, pd_freq, season_length)
                    pred_test = None
                    if len(test) > 0:
                        pred_test = forecast_prophet(m, len(test), last_date=train.index[-1], freq_code=pd_freq)
                    pred_future = forecast_prophet(m, horizon, last_date=series.index[-1], freq_code=pd_freq)

            except Exception as e:
                st.error(f"Model failed: {e}")
                st.stop()

        # Metrics on test, if available
        if len(test) > 0 and pred_test is not None:
            test_common = test[:len(pred_test)]
            mtr = metrics(test_common.values, pred_test.values)
            st.subheader("📊 Test Metrics")
            st.write({k: float(v) for k, v in mtr.items()})

        # Plot
        st.subheader("📉 Plots")
        plot_series(train, test, pred_test=pred_test, pred_future=pred_future)

        # Download forecast
        st.subheader("⬇️ Download Forecast")
        df_out = pd.DataFrame({
            "date": pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(pd_freq), periods=horizon, freq=pd_freq),
            "forecast": pred_future.values
        }).set_index("date")
        st.dataframe(df_out.head())
        csv = df_out.to_csv().encode("utf-8")
        st.download_button("Download future forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")

else:
    st.info("Upload a CSV to get started. Example columns: `date`, `sales`.")
    st.code("date,sales\n2022-01-01,120\n2022-02-01,98\n2022-03-01,135\n...", language="text")
