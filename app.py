import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="BharatStock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern, responsive Streamlit styling with subtle entrance animations.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bs-bg: #0a0f1b;
        --bs-panel: rgba(18, 27, 44, .78);
        --bs-border: rgba(148, 163, 184, .17);
        --bs-text: #edf4ff;
        --bs-muted: #9aabc3;
        --bs-cyan: #38bdf8;
        --bs-green: #34d399;
        --bs-saffron: #ffad5a;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(14, 116, 144, .17), transparent 36%),
          radial-gradient(ellipse at 92% 10%, rgba(37, 99, 235, .13), transparent 30%),
          var(--bs-bg);
        color: var(--bs-text);
    }
    [data-testid="stHeader"] { background: rgba(10, 15, 27, .72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111b2b 0%, #0d1422 100%);
        border-right: 1px solid var(--bs-border);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.4rem; }
    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        animation: rise-in .55s ease-out both;
    }
    @keyframes rise-in {
        from { opacity: 0; transform: translateY(9px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 0 0 rgba(56,189,248,0); }
        50% { box-shadow: 0 0 26px rgba(56,189,248,.12); }
    }
    .bs-hero {
        position: relative;
        overflow: hidden;
        padding: 1.65rem 1.8rem;
        margin: .15rem 0 1.5rem 0;
        border: 1px solid rgba(125, 211, 252, .23);
        border-radius: 24px;
        background: linear-gradient(120deg, rgba(15, 35, 56, .96), rgba(16, 28, 49, .92) 58%, rgba(19, 45, 58, .88));
        animation: glow-pulse 5s ease-in-out infinite;
    }
    .bs-hero:after {
        content: "";
        position: absolute;
        width: 240px; height: 240px;
        right: -75px; top: -125px;
        border-radius: 50%;
        border: 1px solid rgba(56,189,248,.17);
        box-shadow: 0 0 0 24px rgba(56,189,248,.025), 0 0 0 52px rgba(56,189,248,.02);
        pointer-events: none;
    }
    .bs-brand {
        display: flex; align-items: center; gap: 15px;
        position: relative; z-index: 1;
    }
    .bs-logo {
        display: flex; align-items: center; justify-content: center;
        width: 58px; height: 58px; flex: 0 0 58px;
        border-radius: 17px;
        color: #061321;
        font-size: 27px;
        background: linear-gradient(135deg, #67e8f9, #38bdf8 52%, #34d399);
        box-shadow: 0 9px 26px rgba(56,189,248,.18);
    }
    .bs-title {
        margin: 0;
        color: #f4f8ff;
        font: 700 clamp(1.8rem, 4vw, 2.6rem)/1.08 'Space Grotesk', sans-serif;
        letter-spacing: -.055em;
    }
    .bs-subtitle { margin: 8px 0 0; color: #a9bdd4; font-size: .98rem; }
    .bs-chip {
        display: inline-flex; align-items: center; gap: 7px;
        margin-top: 16px; padding: 6px 10px;
        border-radius: 999px; border: 1px solid rgba(52,211,153,.25);
        color: #a7f3d0; background: rgba(16,185,129,.08);
        font-size: .76rem; font-weight: 700; letter-spacing: .04em;
    }
    .bs-chip-dot { width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 10px #34d399; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.025em; }
    h2 { margin-top: 1.5rem !important; }
    [data-testid="stMetric"] {
        padding: 17px 18px;
        border-radius: 18px;
        border: 1px solid var(--bs-border);
        background: linear-gradient(145deg, rgba(21, 32, 51, .91), rgba(14, 23, 38, .88));
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
        animation: rise-in .55s ease-out both;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(56,189,248,.4);
        box-shadow: 0 12px 28px rgba(0,0,0,.15);
    }
    [data-testid="stMetricLabel"] { color: #9aabc3 !important; font-size: .82rem !important; }
    [data-testid="stMetricValue"] { color: #f2f7ff !important; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stPlotlyChart"] {
        padding: 9px;
        border: 1px solid var(--bs-border);
        border-radius: 18px;
        background: rgba(14, 22, 36, .48);
    }
    div[data-testid="stDataFrame"], .stExpander {
        border: 1px solid var(--bs-border);
        border-radius: 15px;
        overflow: hidden;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid rgba(56,189,248,.35);
        background: linear-gradient(100deg, #0e7490, #0369a1);
        color: white;
        font-weight: 700;
        transition: transform .18s ease, filter .18s ease, box-shadow .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.12);
        box-shadow: 0 8px 22px rgba(14,165,233,.2);
        border-color: rgba(125,211,252,.7);
        color: white;
    }
    .stSelectbox [data-baseweb="select"] > div,
    .stTextInput input, .stFileUploader section {
        border-radius: 12px;
    }
    [data-testid="stAlert"] { border-radius: 14px; }
    .bs-section-note { color: var(--bs-muted); font-size: .9rem; margin-top: -.35rem; margin-bottom: .9rem; }
    @media (max-width: 700px) {
        .block-container { padding: 1rem .8rem 2rem; }
        .bs-hero { padding: 1.25rem; border-radius: 18px; }
        .bs-logo { width: 48px; height: 48px; flex-basis: 48px; border-radius: 14px; }
        .bs-subtitle { font-size: .88rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
    }
    </style>
    <div class="bs-hero">
      <div class="bs-brand">
        <div class="bs-logo">📈</div>
        <div>
          <h1 class="bs-title">BharatStock AI</h1>
          <p class="bs-subtitle">Indian stock market analytics · Machine learning insights</p>
          <div class="bs-chip"><span class="bs-chip-dot"></span> HISTORICAL ANALYSIS&nbsp; • &nbsp;PREDICTIVE MODELS</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

DEFAULT_DATA_DIR = Path(__file__).parent / "data"

def normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

def detect_column(columns, candidates):
    lookup = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        if normalize_name(candidate) in lookup:
            return lookup[normalize_name(candidate)]
    for c in columns:
        nc = normalize_name(c)
        if any(normalize_name(candidate) in nc for candidate in candidates):
            return c
    return None

def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")

def prepare_data(raw, date_col, close_col, ticker_col=None, ticker_value=None):
    df = raw.copy()
    if ticker_col and ticker_value and ticker_value != "(All / no filter)":
        df = df[df[ticker_col].astype(str) == str(ticker_value)].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    df[close_col] = pd.to_numeric(
        df[close_col].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False),
        errors="coerce"
    )
    df = df.dropna(subset=[date_col, close_col])
    df = df[df[close_col] > 0].sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    df = df.rename(columns={date_col: "Date", close_col: "Close"})
    return df

def make_features(df):
    data = df.copy().sort_values("Date").reset_index(drop=True)
    data["Return"] = data["Close"].pct_change()
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year
    for lag in [1, 2, 3, 5, 10]:
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
    data["RollingMean_5"] = data["Close"].shift(1).rolling(5).mean()
    data["RollingMean_10"] = data["Close"].shift(1).rolling(10).mean()
    data["RollingStd_5"] = data["Close"].shift(1).rolling(5).std()
    # Target is the next observed trading day's closing price.
    data["Target"] = data["Close"].shift(-1)
    data["TargetDate"] = data["Date"].shift(-1)
    feature_cols = [
        "DayOfWeek", "Month", "Year", "Close_Lag_1", "Close_Lag_2",
        "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
        "RollingMean_5", "RollingMean_10", "RollingStd_5"
    ]
    data = data.replace([np.inf, -np.inf], np.nan)
    model_data = data.dropna(subset=feature_cols + ["Target"]).copy()
    return data, model_data, feature_cols

def train_models(model_data, feature_cols, test_fraction):
    if len(model_data) < 30:
        raise ValueError("At least 30 usable dated rows are recommended. Please use a larger historical dataset.")
    split = int(len(model_data) * (1 - test_fraction))
    split = min(max(split, 1), len(model_data) - 1)
    train = model_data.iloc[:split]
    test = model_data.iloc[split:]
    X_train, y_train = train[feature_cols], train["Target"]
    X_test, y_test = test[feature_cols], test["Target"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }
    results, predictions = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred) if len(y_test) > 1 else np.nan,
            "model": model
        }
        predictions[name] = pd.DataFrame({
            "Date": test["TargetDate"].values,
            "Actual Close": y_test.values,
            "Predicted Close": pred
        })
    return results, predictions, split

def money(value):
    return f"₹{value:,.2f}"

with st.sidebar:
    st.header("⚙️ Workspace")
    st.write("Load your historical market data to begin analysis.")
    uploaded = st.file_uploader("Upload stock CSV", type=["csv"])
    test_fraction = st.slider("Test data proportion", min_value=0.10, max_value=0.35, value=0.20, step=0.05)
    st.caption("Chronological validation · Earlier observations train the model; later observations test it.")

raw = None
source_name = None
if uploaded is not None:
    try:
        raw = load_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")

if raw is None and DEFAULT_DATA_DIR.exists():
    csv_files = sorted(DEFAULT_DATA_DIR.glob("*.csv"))
    if csv_files:
        chosen = st.sidebar.selectbox("CSV found in data folder", csv_files, format_func=lambda p: p.name)
        try:
            raw = pd.read_csv(chosen)
            source_name = chosen.name
        except Exception as exc:
            st.error(f"Could not read {chosen.name}: {exc}")

if raw is None:
    st.info("Start by uploading your Kaggle CSV in the sidebar. You can also copy CSV files into the project's `data` folder.")
    st.subheader("Expected data")
    st.write("A CSV with a date column and a closing-price column. Common names include `Date` and `Close` or `Close Price`.")
    st.code("Date,Open,High,Low,Close,Volume\n2000-01-03,100,105,98,103,120000\n2000-01-04,103,108,101,107,150000", language="csv")
    st.stop()

st.success(f"Loaded **{source_name}** · {len(raw):,} rows · {len(raw.columns)} columns")
with st.expander("Preview raw data"):
    st.dataframe(raw.head(20), use_container_width=True)

date_guess = detect_column(raw.columns, ["Date", "Datetime", "Timestamp", "Trade Date"])
close_guess = detect_column(raw.columns, ["Close", "Closing Price", "Close Price", "Adj Close", "Adjusted Close", "Price"])
ticker_guess = detect_column(raw.columns, ["Ticker", "Symbol", "Stock", "Company", "Name"])

c1, c2, c3 = st.columns(3)
with c1:
    date_col = st.selectbox("Date column", list(raw.columns), index=list(raw.columns).index(date_guess) if date_guess in raw.columns else 0)
with c2:
    close_col = st.selectbox("Closing-price column", list(raw.columns), index=list(raw.columns).index(close_guess) if close_guess in raw.columns else 0)
with c3:
    ticker_options = ["(All / no filter)"] + ([ticker_guess] if ticker_guess else [])
    ticker_col = st.selectbox("Stock identifier column (optional)", ["(None)"] + list(raw.columns), index=(list(raw.columns).index(ticker_guess) + 1) if ticker_guess in raw.columns else 0)

ticker_value = None
if ticker_col != "(None)":
    values = sorted(raw[ticker_col].dropna().astype(str).unique().tolist())
    if values:
        ticker_value = st.selectbox("Choose stock", ["(All / no filter)"] + values)
    else:
        ticker_col = None

df = prepare_data(raw, date_col, close_col, ticker_col if ticker_col != "(None)" else None, ticker_value)
if len(df) < 35:
    st.error(f"Only {len(df)} valid rows remain after cleaning. Please choose the correct columns or upload more historical data (at least 35 rows).")
    st.stop()

if "Volume" in raw.columns:
    volume_col = detect_column(raw.columns, ["Volume", "Trading Volume"])
else:
    volume_col = None

st.subheader("01  ·  Market overview")
first_close = float(df["Close"].iloc[0])
last_close = float(df["Close"].iloc[-1])
change = (last_close / first_close - 1) * 100
m1, m2, m3, m4 = st.columns(4)
m1.metric("Valid observations", f"{len(df):,}")
m2.metric("Latest close in dataset", money(last_close))
m3.metric("First-to-last change", f"{change:.2f}%")
m4.metric("Date range", f"{df['Date'].min():%Y}–{df['Date'].max():%Y}")

fig = px.line(df, x="Date", y="Close", title="Historical closing price", labels={"Close": "Closing price (₹)", "Date": "Date"})
fig.update_layout(hovermode="x unified", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=12, r=12, t=55, b=12), title_font=dict(family="Space Grotesk", size=20))
st.plotly_chart(fig, use_container_width=True)

raw_dates = pd.to_datetime(raw[date_col], errors="coerce")
if volume_col:
    vol_df = raw[[date_col, volume_col]].copy()
    vol_df[date_col] = pd.to_datetime(vol_df[date_col], errors="coerce")
    vol_df[volume_col] = pd.to_numeric(vol_df[volume_col], errors="coerce")
    vol_df = vol_df.dropna().sort_values(date_col)
    if not vol_df.empty:
        st.plotly_chart(px.bar(vol_df.tail(250), x=date_col, y=volume_col, title="Trading volume (latest 250 observations)", template="plotly_dark"), use_container_width=True)

st.subheader("02  ·  Model lab")
full_features, model_data, feature_cols = make_features(df)
try:
    results, predictions, split = train_models(model_data, feature_cols, test_fraction)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

metric_rows = []
for name, result in results.items():
    metric_rows.append({"Model": name, "MAE (₹)": result["MAE"], "RMSE (₹)": result["RMSE"], "R²": result["R²"]})
metrics_df = pd.DataFrame(metric_rows)
st.dataframe(
    metrics_df.style.format({"MAE (₹)": "{:,.3f}", "RMSE (₹)": "{:,.3f}", "R²": "{:.4f}"}),
    use_container_width=True,
    hide_index=True
)
st.caption("MAE and RMSE use the closing-price unit; lower is better. R² may be negative when a model underperforms a mean baseline.")

selected_model = st.selectbox("Model to inspect", list(results.keys()), index=1)
pred_df = predictions[selected_model]
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Actual Close"], name="Actual close", mode="lines"))
fig_pred.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted Close"], name="Predicted close", mode="lines"))
fig_pred.update_layout(title=f"{selected_model}: actual vs predicted on test period", xaxis_title="Date", yaxis_title="Closing price (₹)", hovermode="x unified", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=12, r=12, t=55, b=12), title_font=dict(family="Space Grotesk", size=19))
st.plotly_chart(fig_pred, use_container_width=True)

st.download_button(
    "Download test predictions as CSV",
    data=pred_df.to_csv(index=False).encode("utf-8"),
    file_name="stock_test_predictions.csv",
    mime="text/csv"
)

st.subheader("03  ·  Next-session estimate")
st.write("This estimate uses the latest available historical row and the selected trained model. It is a model output, not a guaranteed future price.")
latest_row = full_features.dropna(subset=feature_cols).iloc[[-1]]
selected_estimator = results[selected_model]["model"]
next_estimate = float(selected_estimator.predict(latest_row[feature_cols])[0])
latest_actual = float(df["Close"].iloc[-1])
delta_pct = (next_estimate / latest_actual - 1) * 100 if latest_actual else np.nan
p1, p2, p3 = st.columns(3)
p1.metric("Latest available close", money(latest_actual))
p2.metric("Model's next-step estimate", money(next_estimate))
p3.metric("Estimated change vs latest close", f"{delta_pct:+.2f}%")
st.caption("The target is the next row in the dataset, which usually corresponds to the next trading session. The model is not trained to forecast long horizons.")

with st.expander("Methodology and limitations"):
    st.markdown("""
    - Features include lagged closing prices, calendar variables, and rolling statistics calculated from prior observations.
    - The target is the next observed closing price.
    - The train/test split preserves chronological order to reduce look-ahead leakage.
    - The final dataset row has no known next-day target and is used only to create the next-step estimate.
    - This baseline does not include news, macroeconomic indicators, corporate actions, transaction costs, or trading strategy simulation.
    - If multiple stocks are mixed in one CSV, filter to one stock before training. Do not train one sequence across unrelated ticker histories.  
    """)
    st.markdown(
    """
    <div style="
        text-align: center;
        padding: 25px 10px;
        margin-top: 35px;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
        color: #94a3b8;
        font-size: 20px;
        font-weight: 600;
    ">
        © Copyright @Vaibhav Rane 2026
    </div>
    """,
    unsafe_allow_html=True
)
