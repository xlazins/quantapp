"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  FAKE BREAKOUT BACKTEST ENGINE                                            ║
║  Streamlit · Geist Mono · Vectorized Pandas/NumPy · 200MB+ Support        ║
╚═══════════════════════════════════════════════════════════════════════════╝

CHANGES vs original
───────────────────
1. FIX  fig_outcome_donut – duplicate 'margin' kwarg TypeError resolved by
        merging layout overrides into a single dict before unpacking.
2. NEW  Risk Mode: "Fixed %" | "ATR-Based"
        • ATR-Based SL  = sl_k × atr_14 (per-bar ATR from dataset)
        • ATR-Based TP  = tp_k × realized-vol of the range window
          (std-dev of 1-min closes inside the range, annualised to price units)
        Falls back to fixed-% if atr_14 column is absent.
3. CLARIFIED  Breakout detection always runs on 1-minute bars regardless of
        the range timeframe chosen.  Range TF only defines the consolidation
        window; the trigger candle is always the first 1-min bar whose close
        breaks out of that range.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fake Breakout Backtest",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TF_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MAX_TRADE_BARS = 600

C = {
    "bg0": "#070709", "bg1": "#0d0d10", "bg2": "#131318", "bg3": "#1a1a20",
    "border": "#22222a", "border2": "#2e2e38",
    "t0": "#f2f2f4", "t1": "#9898a8", "t2": "#55555f", "t3": "#35353f",
    "green": "#00e87a", "red": "#ff4455", "yellow": "#ffc840",
    "blue": "#3d9eff", "purple": "#9b7aff", "teal": "#00d4c8",
    "green_dim": "rgba(0,232,122,0.10)", "red_dim": "rgba(255,68,85,0.10)",
}

PLOTLY_BASE = dict(
    paper_bgcolor=C["bg0"],
    plot_bgcolor=C["bg1"],
    font=dict(
        family="'Geist Mono','JetBrains Mono','Cascadia Code',monospace",
        color=C["t1"],
        size=11,
    ),
    margin=dict(l=55, r=20, t=50, b=50),
)

AXIS = dict(
    gridcolor=C["border"],
    linecolor=C["border"],
    tickcolor=C["border"],
    zeroline=False,
    tickfont=dict(size=10, color=C["t2"]),
)


def _plotly_layout(**overrides) -> dict:
    """Return PLOTLY_BASE with per-call overrides merged in (no duplicate-kwarg crash)."""
    return {**PLOTLY_BASE, **overrides}


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@300;400;500;600;700&display=swap');

*, html, body {{
  font-family: 'Geist Mono', 'JetBrains Mono', 'Cascadia Code', monospace !important;
  box-sizing: border-box;
}}
.stApp, [data-testid="stAppViewContainer"] {{
  background: {C["bg0"]} !important;
  color: {C["t0"]} !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{C["bg1"]}; }}
::-webkit-scrollbar-thumb {{ background:{C["border2"]}; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:{C["t3"]}; }}

[data-testid="stSidebar"] {{
  background: {C["bg1"]} !important;
  border-right: 1px solid {C["border"]} !important;
}}
[data-testid="stSidebar"] * {{ color: {C["t0"]} !important; }}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {{
  background: {C["bg2"]} !important;
  border: 1px solid {C["border"]} !important;
  border-radius: 5px !important;
}}
[data-testid="stSidebar"] section {{ padding-top: 0 !important; }}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
  background: {C["bg2"]} !important;
  border: 1px solid {C["border"]} !important;
  color: {C["t0"]} !important;
  border-radius: 5px !important;
  font-family: 'Geist Mono', monospace !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
  border-color: {C["green"]} !important;
  box-shadow: 0 0 0 2px rgba(0,232,122,.1) !important;
  outline: none !important;
}}

.stButton > button {{
  background: {C["bg2"]} !important;
  border: 1px solid {C["border2"]} !important;
  color: {C["t1"]} !important;
  border-radius: 5px !important;
  font-family: 'Geist Mono', monospace !important;
  font-size: .78rem !important;
  letter-spacing: .04em !important;
  transition: all .15s ease !important;
  padding: 8px 16px !important;
}}
.stButton > button:hover {{
  border-color: {C["green"]} !important;
  color: {C["green"]} !important;
  background: {C["green_dim"]} !important;
}}

[data-testid="stSlider"] > div > div > div > div {{
  background: {C["green"]} !important;
}}
[data-testid="stSlider"] [role="slider"] {{
  background: {C["green"]} !important;
  border: 2px solid {C["bg0"]} !important;
  box-shadow: 0 0 0 1px {C["green"]} !important;
}}

[data-testid="stFileUploader"] > div {{
  background: {C["bg2"]} !important;
  border: 1px dashed {C["border2"]} !important;
  border-radius: 6px !important;
}}
[data-testid="stFileUploader"] > div:hover {{
  border-color: {C["green"]} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  background: {C["bg1"]} !important;
  border-bottom: 1px solid {C["border"]} !important;
  gap: 0 !important;
  padding: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: {C["t2"]} !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  font-size: .73rem !important;
  letter-spacing: .06em !important;
  padding: 8px 16px !important;
}}
.stTabs [aria-selected="true"] {{
  color: {C["green"]} !important;
  border-bottom: 2px solid {C["green"]} !important;
  background: {C["green_dim"]} !important;
}}

[data-testid="stMetricValue"] {{
  font-size: 1.5rem !important;
  font-weight: 600 !important;
  color: {C["t0"]} !important;
}}
[data-testid="stMetricLabel"] {{
  font-size: .65rem !important;
  color: {C["t2"]} !important;
  letter-spacing: .10em !important;
  text-transform: uppercase !important;
}}

.streamlit-expanderHeader {{
  background: {C["bg2"]} !important;
  border: 1px solid {C["border"]} !important;
  border-radius: 5px !important;
  color: {C["t1"]} !important;
  font-size: .78rem !important;
}}
.streamlit-expanderContent {{
  background: {C["bg1"]} !important;
  border: 1px solid {C["border"]} !important;
  border-top: none !important;
  border-radius: 0 0 5px 5px !important;
}}

.stAlert {{
  background: {C["bg2"]} !important;
  border: 1px solid {C["border"]} !important;
  border-radius: 5px !important;
}}

.stDataFrame {{ background: {C["bg1"]} !important; border-radius: 6px; }}
[data-testid="stDataFrameResizable"] table {{ color: {C["t1"]} !important; }}

.stSpinner > div {{ border-top-color: {C["green"]} !important; }}
.stCheckbox label {{ color: {C["t1"]} !important; font-size: .8rem !important; }}
.stRadio label {{ color: {C["t1"]} !important; font-size: .8rem !important; }}
[data-testid="stProgressBar"] > div {{ background: {C["green"]} !important; }}

.kpi-card {{
  background: {C["bg2"]};
  border: 1px solid {C["border"]};
  border-radius: 6px;
  padding: 14px 18px;
}}
.kpi-label {{
  font-size: .60rem;
  color: {C["t2"]};
  text-transform: uppercase;
  letter-spacing: .12em;
  margin-bottom: 5px;
}}
.kpi-value {{
  font-size: 1.35rem;
  font-weight: 600;
  color: {C["t0"]};
  line-height: 1;
}}
.kpi-value.green {{ color: {C["green"]}; }}
.kpi-value.red   {{ color: {C["red"]}; }}
.kpi-value.yellow {{ color: {C["yellow"]}; }}

.section-label {{
  font-size: .60rem;
  font-weight: 600;
  color: {C["t3"]};
  text-transform: uppercase;
  letter-spacing: .15em;
  padding-bottom: 6px;
  border-bottom: 1px solid {C["border"]};
  margin: 18px 0 10px 0;
}}

.badge {{
  display: inline-block;
  font-size: .60rem;
  padding: 2px 8px;
  border-radius: 3px;
  letter-spacing: .08em;
  text-transform: uppercase;
  font-weight: 500;
}}
.badge-green {{
  background: {C["green_dim"]};
  color: {C["green"]};
  border: 1px solid rgba(0,232,122,.25);
}}
.badge-red {{
  background: {C["red_dim"]};
  color: {C["red"]};
  border: 1px solid rgba(255,68,85,.25);
}}
.badge-dim {{
  background: {C["bg3"]};
  color: {C["t2"]};
  border: 1px solid {C["border"]};
}}
.badge-yellow {{
  background: rgba(255,200,64,.10);
  color: {C["yellow"]};
  border: 1px solid rgba(255,200,64,.25);
}}

.top-header {{
  display: flex;
  align-items: center;
  gap: 14px;
  padding-bottom: 18px;
  border-bottom: 1px solid {C["border"]};
  margin-bottom: 22px;
}}
.top-header .pulse {{
  width: 8px; height: 8px;
  background: {C["green"]};
  border-radius: 50%;
  box-shadow: 0 0 8px {C["green"]};
  flex-shrink: 0;
}}
.top-header .title {{
  font-size: 1.05rem;
  font-weight: 600;
  color: {C["t0"]};
  letter-spacing: .02em;
}}
.top-header .sub {{
  font-size: .65rem;
  color: {C["t2"]};
  margin-top: 3px;
  letter-spacing: .06em;
}}

hr {{ border-color: {C["border"]} !important; }}
div[data-testid="column"] {{ padding: 0 4px !important; }}
</style>
""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=4)
def _load_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


@st.cache_data(show_spinner=False, max_entries=4)
def _load_parquet_path(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, max_entries=8)
def preprocess(df_raw: pd.DataFrame, years: int) -> pd.DataFrame:
    df = df_raw.copy()
    ts_col = next(
        (c for c in df.columns if c.lower() in ("timestamp", "datetime", "time", "date")),
        None,
    )
    if ts_col and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
        df = df.set_index(ts_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False, errors="coerce")

    df.index.name = "timestamp"
    df = df[~df.index.isna()].sort_index()
    df.columns = [c.lower().strip() for c in df.columns]

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["open", "high", "low", "close", "volume", "adx_14", "atr_14", "zscore_20"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    if years > 0:
        cutoff = df.index[-1] - pd.DateOffset(years=years)
        df = df.loc[df.index >= cutoff]

    if df.empty:
        raise ValueError("No data remaining after filtering.")

    df["_date"]         = df.index.date
    df["_weekday"]      = df.index.dayofweek
    df["_weekday_name"] = df.index.day_name()
    df["_bar_idx"]      = np.arange(len(df), dtype=np.int32)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY — RANGE COMPUTATION & BREAKOUT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=8)
def compute_signals(
    df: pd.DataFrame,
    tf_minutes: int,
    custom_start_str: str = "",
    custom_end_str: str = "",
) -> pd.DataFrame:
    """
    Breakout detection always happens on 1-minute bars.
    tf_minutes controls ONLY the consolidation window (range candle).
    The first 1-min close that crosses the range high/low after the range
    closes is the trigger.
    """
    freq = f"{tf_minutes}min"
    range_candles = df.resample(freq).agg(
        range_high=("high", "max"),
        range_low=("low", "min"),
        # Realized vol of the range: std-dev of 1-min closes (price units)
        range_vol=("close", "std"),
    ).dropna(subset=["range_high", "range_low"])
    range_candles.index.name = "range_start"
    range_candles = range_candles.reset_index()
    range_candles["range_end"] = range_candles["range_start"] + pd.Timedelta(minutes=tf_minutes)

    if custom_start_str and custom_end_str:
        try:
            cs = pd.to_datetime(custom_start_str, format="%H:%M").time()
            ce = pd.to_datetime(custom_end_str, format="%H:%M").time()
            mask = (
                (range_candles["range_start"].dt.time >= cs)
                & (range_candles["range_start"].dt.time < ce)
            )
            range_candles = range_candles[mask]
        except Exception:
            pass

    if range_candles.empty:
        return pd.DataFrame()

    extra = [c for c in ["adx_14", "atr_14", "zscore_20"] if c in df.columns]
    bar_cols = ["high", "low", "close", "_date", "_weekday", "_weekday_name", "_bar_idx"] + extra
    bars = df[bar_cols].reset_index()

    bars          = bars.sort_values("timestamp")
    range_candles = range_candles.sort_values("range_end")

    merged = pd.merge_asof(
        bars,
        range_candles[["range_start", "range_end", "range_high", "range_low", "range_vol"]],
        left_on="timestamp",
        right_on="range_end",
        direction="backward",
    )

    merged = merged.dropna(subset=["range_start"])
    merged["_rdate"] = merged["range_start"].dt.date
    merged = merged[merged["_date"] == merged["_rdate"]]
    merged = merged[merged["timestamp"] >= merged["range_end"]]

    if merged.empty:
        return pd.DataFrame()

    short_mask = merged["close"] > merged["range_high"]
    long_mask  = merged["close"] < merged["range_low"]

    short_sig = merged[short_mask].copy()
    short_sig["direction"]   = "short"
    short_sig["entry_price"] = short_sig["close"]

    long_sig = merged[long_mask].copy()
    long_sig["direction"]   = "long"
    long_sig["entry_price"] = long_sig["close"]

    signals = pd.concat([short_sig, long_sig], ignore_index=True)
    if signals.empty:
        return pd.DataFrame()

    signals = signals.sort_values("timestamp")
    signals = (
        signals.groupby(["range_start", "direction"], sort=False)
        .first()
        .reset_index()
    )
    signals["range_time"] = signals["range_start"].dt.strftime("%H:%M")

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# RISK LEVEL COMPUTATION  (ATR-based or fixed-%)
# ══════════════════════════════════════════════════════════════════════════════

def compute_sl_tp(
    entry_price: np.ndarray,
    range_high: np.ndarray,
    range_low: np.ndarray,
    range_vol: np.ndarray,
    atr: np.ndarray,          # per-signal ATR value (NaN if unavailable)
    is_short: np.ndarray,
    risk_mode: str,           # "fixed_pct" | "atr"
    sl_k: float,              # multiplier for SL  (pct or ATR multiple)
    tp_mode: str,             # "range" | "fixed_pct" | "vol"
    tp_k: float,              # multiplier for TP
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (sl_arr, tp_arr) as price levels.

    ATR mode
    --------
    SL  = entry ± sl_k × ATR_14
    TP  = entry ∓ tp_k × range_vol   (realized vol of the consolidation window)
          Falls back to range opposite side when range_vol is NaN/zero,
          and to fixed-% when ATR is also unavailable.
    """
    has_atr = ~np.isnan(atr)

    # ── Stop Loss ─────────────────────────────────────────────────────
    if risk_mode == "atr":
        sl_dist = np.where(has_atr, sl_k * atr, entry_price * sl_k / 100.0)
    else:
        sl_dist = entry_price * (sl_k / 100.0)

    sl_arr = np.where(is_short, entry_price + sl_dist, entry_price - sl_dist)

    # ── Take Profit ───────────────────────────────────────────────────
    if tp_mode == "range":
        tp_arr = np.where(is_short, range_low, range_high)

    elif tp_mode == "vol":
        # TP distance = tp_k × range realized vol (price units)
        # range_vol is already in price units (std of 1-min closes)
        has_vol = ~np.isnan(range_vol) & (range_vol > 0)
        tp_dist = np.where(
            has_vol,
            tp_k * range_vol,
            np.where(has_atr, tp_k * atr, entry_price * tp_k / 100.0),
        )
        tp_arr = np.where(is_short, entry_price - tp_dist, entry_price + tp_dist)

    else:  # fixed_pct
        tp_dist = entry_price * (tp_k / 100.0)
        tp_arr = np.where(is_short, entry_price - tp_dist, entry_price + tp_dist)

    return sl_arr, tp_arr


# ══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=8)
def simulate_trades(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    risk_mode: str,
    sl_k: float,
    tp_mode: str,
    tp_k: float = 1.0,
    max_bars: int = MAX_TRADE_BARS,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n_bars = len(highs)

    entry_bar   = signals["_bar_idx"].values.astype(np.int32)
    entry_price = signals["entry_price"].values.astype(np.float64)
    range_high  = signals["range_high"].values.astype(np.float64)
    range_low   = signals["range_low"].values.astype(np.float64)
    range_vol   = signals["range_vol"].values.astype(np.float64) if "range_vol" in signals.columns else np.full(len(signals), np.nan)
    is_short    = (signals["direction"].values == "short")

    # Per-signal ATR at entry bar
    if "atr_14" in signals.columns:
        atr_vals = signals["atr_14"].values.astype(np.float64)
    else:
        atr_vals = np.full(len(signals), np.nan)

    sl_arr, tp_arr = compute_sl_tp(
        entry_price, range_high, range_low, range_vol, atr_vals,
        is_short, risk_mode, sl_k, tp_mode, tp_k,
    )

    n_trades    = len(signals)
    outcomes    = np.full(n_trades, "timeout", dtype=object)
    exit_prices = entry_price.copy()
    mae_pts     = np.zeros(n_trades, dtype=np.float64)
    mfe_pts     = np.zeros(n_trades, dtype=np.float64)
    dur_bars    = np.zeros(n_trades, dtype=np.int32)

    for i in range(n_trades):
        eb   = int(entry_bar[i])
        ep   = entry_price[i]
        sl   = sl_arr[i]
        tp   = tp_arr[i]
        shrt = bool(is_short[i])

        start = eb + 1
        end   = min(start + max_bars, n_bars)
        if start >= n_bars:
            continue

        fh = highs[start:end]
        fl = lows[start:end]
        fc = closes[start:end]
        n  = len(fh)
        if n == 0:
            continue

        if shrt:
            sl_hit = fh >= sl
            tp_hit = fl <= tp
        else:
            sl_hit = fl <= sl
            tp_hit = fh >= tp

        first_sl = int(np.argmax(sl_hit)) if sl_hit.any() else n
        first_tp = int(np.argmax(tp_hit)) if tp_hit.any() else n

        if not sl_hit.any() and not tp_hit.any():
            exit_local     = n - 1
            outcomes[i]    = "timeout"
            exit_prices[i] = fc[-1]
        elif first_sl <= first_tp:
            exit_local     = first_sl
            outcomes[i]    = "loss"
            exit_prices[i] = sl
        else:
            exit_local     = first_tp
            outcomes[i]    = "win"
            exit_prices[i] = tp

        dur_bars[i] = exit_local + 1

        th = fh[: exit_local + 1]
        tl = fl[: exit_local + 1]
        if len(th) == 0:
            continue

        if shrt:
            mfe_pts[i] = max(0.0, ep - float(np.min(tl)))
            mae_pts[i] = max(0.0, float(np.max(th)) - ep)
        else:
            mfe_pts[i] = max(0.0, float(np.max(th)) - ep)
            mae_pts[i] = max(0.0, ep - float(np.min(tl)))

    res = signals.copy()
    res["sl_price"]   = sl_arr
    res["tp_price"]   = tp_arr
    res["outcome"]    = outcomes
    res["exit_price"] = exit_prices
    res["mae"]        = mae_pts
    res["mfe"]        = mfe_pts
    res["duration"]   = dur_bars
    res["win"]        = outcomes == "win"

    res["pnl_pct"] = np.where(
        is_short,
        (entry_price - exit_prices) / entry_price * 100.0,
        (exit_prices - entry_price) / entry_price * 100.0,
    )
    res["mae_pct"] = mae_pts / entry_price * 100.0
    res["mfe_pct"] = mfe_pts / entry_price * 100.0

    return res


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=16)
def aggregate_heatmap(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    agg = (
        trades.groupby(["_weekday_name", "range_time"])
        .agg(
            n_trades  =("win",     "count"),
            n_wins    =("win",     "sum"),
            avg_mae   =("mae_pct", "mean"),
            avg_mfe   =("mfe_pct", "mean"),
            avg_pnl   =("pnl_pct", "mean"),
        )
        .reset_index()
    )
    agg["strike_rate"] = (agg["n_wins"] / agg["n_trades"] * 100).round(1)
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_heatmap(agg: pd.DataFrame) -> go.Figure:
    avail_days = [d for d in DAY_ORDER if d in agg["_weekday_name"].values]

    pivot  = agg.pivot(index="range_time", columns="_weekday_name", values="strike_rate").reindex(columns=avail_days)
    p_mae  = agg.pivot(index="range_time", columns="_weekday_name", values="avg_mae").reindex(columns=avail_days)
    p_mfe  = agg.pivot(index="range_time", columns="_weekday_name", values="avg_mfe").reindex(columns=avail_days)
    p_n    = agg.pivot(index="range_time", columns="_weekday_name", values="n_trades").reindex(columns=avail_days)
    p_pnl  = agg.pivot(index="range_time", columns="_weekday_name", values="avg_pnl").reindex(columns=avail_days)

    pivot = pivot.sort_index()
    z    = pivot.values
    days = avail_days
    times = pivot.index.tolist()

    hover = []
    for ti, t in enumerate(times):
        row = []
        for di, d in enumerate(days):
            sr  = z[ti, di]
            mae = p_mae.values[ti, di]
            mfe = p_mfe.values[ti, di]
            nn  = p_n.values[ti, di]
            pnl = p_pnl.values[ti, di]
            if np.isnan(sr):
                row.append(f"<b>{d[:3]} · {t}</b><br>No data")
            else:
                win_icon  = "▲" if sr >= 50 else "▼"
                win_color = C["green"] if sr >= 50 else C["red"]
                row.append(
                    f"<b style='font-size:12px'>{d[:3]} · {t}</b><br>"
                    f"<span style='color:{win_color}'>{win_icon} {sr:.1f}% strike rate</span><br>"
                    f"<span style='color:{C['t2']}'>Trades: {int(nn)}</span><br>"
                    f"<span style='color:{C['red']}'>MAE avg  {mae:.3f}%</span><br>"
                    f"<span style='color:{C['green']}'>MFE avg  {mfe:.3f}%</span><br>"
                    f"<span style='color:{C['t2']}'>P&L avg  {pnl:+.3f}%</span>"
                )
        hover.append(row)

    colorscale = [
        [0.00, "#2a0808"], [0.30, "#5a1010"], [0.45, "#8a2800"],
        [0.50, "#222228"], [0.55, "#0a2a18"], [0.70, "#006030"],
        [1.00, C["green"]],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=days, y=times,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=colorscale,
        zmid=50, zmin=0, zmax=100,
        xgap=2, ygap=1,
        colorbar=dict(
            title=dict(text="Strike %", font=dict(color=C["t2"], size=10)),
            tickfont=dict(color=C["t2"], size=9),
            bgcolor=C["bg0"],
            bordercolor=C["border"],
            thickness=12,
            len=0.85,
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
        ),
    ))

    fig.update_layout(
        **_plotly_layout(
            title=dict(
                text="Strike Rate  ·  Fake Breakout  ·  Click a cell for deep analytics",
                font=dict(size=12, color=C["t1"]),
                x=0.01, y=0.98,
            ),
            height=max(380, len(times) * 17 + 160),
            xaxis=dict(side="top", tickfont=dict(size=11, color=C["t1"]),
                       **{k: v for k, v in AXIS.items() if k != "tickfont"}),
            yaxis=dict(autorange="reversed", tickfont=dict(size=9, color=C["t2"]),
                       **{k: v for k, v in AXIS.items() if k != "tickfont"}),
        )
    )
    return fig


def fig_distributions(trades: pd.DataFrame, label: str = "") -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["MAE Distribution  (adverse)", "MFE Distribution  (favorable)"],
        horizontal_spacing=0.08,
    )
    mae_v = trades["mae_pct"].dropna()
    mfe_v = trades["mfe_pct"].dropna()

    fig.add_trace(go.Histogram(x=mae_v, name="MAE %",
        marker_color=C["red"], opacity=0.72, nbinsx=50,
        hovertemplate="MAE %{x:.3f}%<br>Count %{y}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Histogram(x=mfe_v, name="MFE %",
        marker_color=C["green"], opacity=0.72, nbinsx=50,
        hovertemplate="MFE %{x:.3f}%<br>Count %{y}<extra></extra>"), row=1, col=2)

    for pct, lbl in [(25, "p25"), (50, "p50"), (75, "p75")]:
        if len(mae_v):
            v = float(np.percentile(mae_v, pct))
            fig.add_vline(x=v, row=1, col=1,
                line=dict(color=C["red"], dash="dash", width=1),
                annotation=dict(text=lbl, font=dict(color=C["red"], size=8), showarrow=False))
        if len(mfe_v):
            v = float(np.percentile(mfe_v, pct))
            fig.add_vline(x=v, row=1, col=2,
                line=dict(color=C["green"], dash="dash", width=1),
                annotation=dict(text=lbl, font=dict(color=C["green"], size=8), showarrow=False))

    ax_upd = dict(gridcolor=C["border"], linecolor=C["border"],
                  tickfont=dict(size=9, color=C["t2"]), zeroline=False)
    title  = f"MAE / MFE Distributions{('  ·  ' + label) if label else ''}"
    fig.update_layout(
        **_plotly_layout(
            title=dict(text=title, font=dict(size=11, color=C["t1"]), x=0.01),
            height=320, showlegend=True,
            legend=dict(font=dict(size=9, color=C["t2"]), bgcolor="rgba(0,0,0,0)"),
            **{ax: ax_upd for ax in ("xaxis", "yaxis", "xaxis2", "yaxis2")},
        )
    )
    fig.update_annotations(font_size=9)
    return fig


def fig_scatter(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    palette = {"win": (C["green"], .55), "loss": (C["red"], .55), "timeout": (C["yellow"], .40)}
    for outcome, (color, opacity) in palette.items():
        sub = trades[trades["outcome"] == outcome]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["mae_pct"], y=sub["mfe_pct"],
            mode="markers", name=outcome.capitalize(),
            marker=dict(color=color, size=3.5, opacity=opacity, line=dict(width=0)),
            hovertemplate=f"<b>{outcome}</b><br>MAE %{{x:.3f}}%<br>MFE %{{y:.3f}}%<extra></extra>",
        ))
    fig.update_layout(
        **_plotly_layout(
            title=dict(text="MAE vs MFE Scatter", font=dict(size=11, color=C["t1"]), x=0.01),
            height=320,
            xaxis=dict(title="MAE %", **AXIS),
            yaxis=dict(title="MFE %", **AXIS),
            legend=dict(font=dict(size=9, color=C["t2"]), bgcolor="rgba(0,0,0,0)"),
        )
    )
    return fig


def fig_feature_analysis(trades: pd.DataFrame, feature: str) -> go.Figure:
    sub = trades.dropna(subset=[feature]).copy()
    if sub.empty:
        return go.Figure()
    sub["_bin"] = pd.cut(sub[feature], bins=20)
    agg = sub.groupby("_bin", observed=True).agg(
        n=("win", "count"), win_rate=("win", "mean"),
        mae=("mae_pct", "mean"), mfe=("mfe_pct", "mean")
    ).reset_index()
    agg["win_pct"] = agg["win_rate"] * 100
    agg["bin_mid"] = agg["_bin"].apply(lambda x: float(x.mid))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[f"Strike Rate vs {feature}", "Avg MAE / MFE"])
    fig.add_trace(go.Scatter(x=agg["bin_mid"], y=agg["win_pct"],
        mode="lines+markers", name="Strike %",
        line=dict(color=C["green"], width=2), marker=dict(size=4, color=C["green"]),
        hovertemplate=f"{feature} %{{x:.2f}}<br>Strike %{{y:.1f}}%<extra></extra>"),
        row=1, col=1)
    fig.add_hline(y=50, row=1, col=1, line=dict(color=C["t3"], dash="dot", width=1))
    fig.add_trace(go.Bar(x=agg["bin_mid"], y=agg["mae"], name="Avg MAE",
        marker_color=C["red"], opacity=.7,
        hovertemplate=f"{feature} %{{x:.2f}}<br>MAE %{{y:.3f}}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=agg["bin_mid"], y=agg["mfe"], name="Avg MFE",
        marker_color=C["green"], opacity=.7,
        hovertemplate=f"{feature} %{{x:.2f}}<br>MFE %{{y:.3f}}%<extra></extra>"), row=2, col=1)

    ax_upd = dict(gridcolor=C["border"], linecolor=C["border"],
                  tickfont=dict(size=9, color=C["t2"]), zeroline=False)
    fig.update_layout(
        **_plotly_layout(
            height=400, barmode="group",
            title=dict(text=f"Performance Regime — {feature}",
                       font=dict(size=11, color=C["t1"]), x=0.01),
            legend=dict(font=dict(size=9, color=C["t2"]), bgcolor="rgba(0,0,0,0)"),
            **{ax: ax_upd for ax in ("xaxis", "yaxis", "xaxis2", "yaxis2")},
        )
    )
    fig.update_annotations(font_size=9)
    return fig


def fig_outcome_donut(trades: pd.DataFrame) -> go.Figure:
    wins     = int((trades["outcome"] == "win").sum())
    losses   = int((trades["outcome"] == "loss").sum())
    timeouts = int((trades["outcome"] == "timeout").sum())
    fig = go.Figure(go.Pie(
        labels=["Win", "Loss", "Timeout"],
        values=[wins, losses, timeouts],
        hole=0.62,
        marker_colors=[C["green"], C["red"], C["yellow"]],
        textfont=dict(family="'Geist Mono',monospace", size=10),
        hovertemplate="%{label}: %{value}  (%{percent})<extra></extra>",
    ))
    # ── FIX: use _plotly_layout() so margin override doesn't clash with
    #         the margin key already inside PLOTLY_BASE ─────────────────
    fig.update_layout(
        **_plotly_layout(
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),   # override base margin
            showlegend=True,
            legend=dict(font=dict(size=9, color=C["t2"]), bgcolor="rgba(0,0,0,0)"),
        )
    )
    return fig


def fig_pnl_curve(trades: pd.DataFrame) -> go.Figure:
    s   = trades.sort_values("timestamp" if "timestamp" in trades.columns else "range_start")
    cum = s["pnl_pct"].cumsum()
    color = C["green"] if float(cum.iloc[-1]) >= 0 else C["red"]
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cum))), y=cum,
        mode="lines", name="Cumulative P&L %",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.08)",
        hovertemplate="Trade %{x}<br>Cum P&L %{y:.3f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=C["t3"], dash="dot", width=1))
    fig.update_layout(
        **_plotly_layout(
            title=dict(text="Cumulative P&L %", font=dict(size=11, color=C["t1"]), x=0.01),
            height=240,
            xaxis=dict(title="Trade #", **AXIS),
            yaxis=dict(title="Cum P&L %", **AXIS),
            showlegend=False,
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi(label: str, value: str, variant: str = "") -> str:
    cls = f"kpi-value {variant}".strip()
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>"""


def pct_table(trades: pd.DataFrame) -> str:
    rows = ""
    mae_v = trades["mae_pct"].dropna()
    mfe_v = trades["mfe_pct"].dropna()
    for p in [10, 25, 50, 75, 90]:
        mv = f"{np.percentile(mae_v, p):.4f}%" if len(mae_v) else "—"
        fv = f"{np.percentile(mfe_v, p):.4f}%" if len(mfe_v) else "—"
        rows += (
            f"<tr>"
            f"<td style='color:{C['t2']};padding:6px 14px'>p{p}</td>"
            f"<td style='color:{C['red']};padding:6px 14px;text-align:right'>{mv}</td>"
            f"<td style='color:{C['green']};padding:6px 14px;text-align:right'>{fv}</td>"
            f"</tr>"
        )
    return f"""
    <table style="width:100%;background:{C['bg2']};border:1px solid {C['border']};
      border-radius:6px;border-collapse:collapse;
      font-family:'Geist Mono',monospace;font-size:.78rem">
      <thead>
        <tr style="border-bottom:1px solid {C['border']}">
          <th style="color:{C['t3']};padding:8px 14px;text-align:left;
              font-weight:500;letter-spacing:.10em;text-transform:uppercase">PCTILE</th>
          <th style="color:{C['red']};padding:8px 14px;text-align:right;
              font-weight:500;letter-spacing:.06em">MAE %</th>
          <th style="color:{C['green']};padding:8px 14px;text-align:right;
              font-weight:500;letter-spacing:.06em">MFE %</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_analytics(trades: pd.DataFrame, label: str = "") -> None:
    if trades.empty:
        st.warning("No trades for the current selection / filters.")
        return

    n        = len(trades)
    wins     = int((trades["outcome"] == "win").sum())
    losses   = int((trades["outcome"] == "loss").sum())
    timeouts = int((trades["outcome"] == "timeout").sum())
    sr       = wins / n * 100 if n else 0
    avg_pnl  = float(trades["pnl_pct"].mean())
    avg_mae  = float(trades["mae_pct"].mean())
    avg_mfe  = float(trades["mfe_pct"].mean())
    ratio    = avg_mfe / avg_mae if avg_mae > 0 else 0.0

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.markdown(kpi("Trades", f"{n:,}"), unsafe_allow_html=True)
    c2.markdown(kpi("Strike Rate", f"{sr:.1f}%", "green" if sr >= 50 else "red"), unsafe_allow_html=True)
    c3.markdown(kpi("Wins", f"{wins:,}", "green"), unsafe_allow_html=True)
    c4.markdown(kpi("Losses", f"{losses:,}", "red"), unsafe_allow_html=True)
    c5.markdown(kpi("Avg P&L", f"{avg_pnl:+.3f}%", "green" if avg_pnl >= 0 else "red"), unsafe_allow_html=True)
    c6.markdown(kpi("Avg MAE", f"{avg_mae:.3f}%", "red"), unsafe_allow_html=True)
    c7.markdown(kpi("MFE/MAE", f"{ratio:.2f}", "green" if ratio >= 1 else "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊  Distributions", "🔵  Scatter", "📈  Feature Analysis",
        "📋  Percentiles", "📉  P&L Curve", "🗂  Raw Trades",
    ])

    _key = label.replace(" ", "_").replace("·", "").replace(":", "")

    with tab1:
        st.plotly_chart(fig_distributions(trades, label), use_container_width=True, key=f"dist_{_key}")
    with tab2:
        st.plotly_chart(fig_scatter(trades), use_container_width=True, key=f"scat_{_key}")
    with tab3:
        feat_cols = [c for c in ["adx_14", "atr_14", "zscore_20"] if c in trades.columns]
        if feat_cols:
            sel = st.selectbox("Feature", feat_cols, key=f"feat_sel_{_key}")
            st.plotly_chart(fig_feature_analysis(trades, sel), use_container_width=True, key=f"feat_{_key}")
        else:
            st.info("Feature columns (adx_14, atr_14, zscore_20) not found in this dataset.")
    with tab4:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown(pct_table(trades), unsafe_allow_html=True)
        with col_b:
            st.plotly_chart(fig_outcome_donut(trades), use_container_width=True, key=f"donut_{_key}")
    with tab5:
        st.plotly_chart(fig_pnl_curve(trades), use_container_width=True, key=f"pnl_{_key}")
    with tab6:
        show_cols = [c for c in [
            "range_time", "_weekday_name", "direction",
            "entry_price", "sl_price", "tp_price", "exit_price",
            "outcome", "pnl_pct", "mae_pct", "mfe_pct", "duration",
        ] if c in trades.columns]
        st.dataframe(
            trades[show_cols].rename(columns={"_weekday_name": "weekday"}).round(5).head(1000),
            use_container_width=True, height=280,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:18px 0 20px;border-bottom:1px solid {C['border']};margin-bottom:14px">
          <div style="font-size:.60rem;color:{C['t3']};letter-spacing:.14em;
              text-transform:uppercase;margin-bottom:5px">Strategy Engine</div>
          <div style="font-size:.95rem;font-weight:600;color:{C['t0']}">
              ⚡ Fake Breakout
          </div>
          <div style="font-size:.62rem;color:{C['t3']};margin-top:3px">
              MAE · MFE · Regime Analyzer
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Data Source ──────────────────────────────────────────────
        st.markdown(f'<div class="section-label">Data Source</div>', unsafe_allow_html=True)
        src = st.radio("_src", ["Upload File", "Local Path"], label_visibility="collapsed")
        uploaded   = None
        local_path = ""

        if src == "Upload File":
            uploaded = st.file_uploader(
                "_up", type=["csv", "parquet"], label_visibility="collapsed",
                help="CSV or Parquet with timestamp, open, high, low, close columns.",
            )
        else:
            local_path = st.text_input("_lp", placeholder="/data/nq_1m.parquet",
                                       label_visibility="collapsed")

        years = st.slider("Years of data", 1, 10, 3)

        # ── Range Settings ───────────────────────────────────────────
        st.markdown(f'<div class="section-label">Range</div>', unsafe_allow_html=True)

        tf = st.selectbox(
            "Range Timeframe",
            ["1m", "5m", "15m", "1h"], index=1,
            help="Defines the consolidation window. Breakout trigger is always a 1-min bar close.",
        )

        st.markdown(
            f'<div style="font-size:.62rem;color:{C["t3"]};margin:-6px 0 8px 0">'
            f'Trigger candle: always <span style="color:{C["green"]}">1-min bar</span></div>',
            unsafe_allow_html=True,
        )

        custom_range = st.checkbox("Custom Window")
        c_start, c_end = "", ""
        if custom_range:
            ca, cb = st.columns(2)
            with ca:
                c_start = st.text_input("Start", value="09:30", key="cstart")
            with cb:
                c_end = st.text_input("End", value="09:35", key="cend")

        # ── Risk ─────────────────────────────────────────────────────
        st.markdown(f'<div class="section-label">Risk</div>', unsafe_allow_html=True)

        risk_mode_label = st.selectbox(
            "Risk Mode",
            ["Fixed %", "ATR-Based"],
            index=0,
            help=(
                "ATR-Based: SL = k × ATR_14  |  TP = k × realized vol of range window.\n"
                "Falls back to fixed-% when atr_14 is absent."
            ),
        )
        risk_mode = "fixed_pct" if risk_mode_label == "Fixed %" else "atr"

        if risk_mode == "atr":
            sl_k = st.number_input(
                "SL multiplier (× ATR)", 0.5, 10.0, 1.5, 0.1, format="%.1f",
                help="Stop distance = sl_k × ATR_14 per bar",
            )
        else:
            sl_k = st.number_input("Stop Loss %", 0.05, 5.0, 0.5, 0.05, format="%.2f")

        tp_mode_label = st.selectbox(
            "Take Profit Mode",
            ["Opposite Side of Range", "Fixed %", "Realized Vol of Range"],
            index=0,
            help=(
                "Realized Vol: TP distance = k × std-dev of 1-min closes inside the range candle.\n"
                "Reduces TP size in low-vol regimes automatically."
            ),
        )
        tp_mode = {
            "Opposite Side of Range": "range",
            "Fixed %": "fixed_pct",
            "Realized Vol of Range": "vol",
        }[tp_mode_label]

        tp_k = 1.0
        if tp_mode in ("fixed_pct", "vol"):
            tp_label = "TP multiplier (× vol)" if tp_mode == "vol" else "Take Profit %"
            tp_k = st.number_input(tp_label, 0.05, 10.0, 1.0, 0.05, format="%.2f")

        # ── Filters ──────────────────────────────────────────────────
        st.markdown(f'<div class="section-label">Filters</div>', unsafe_allow_html=True)
        direction_filt = st.multiselect("Direction", ["short", "long"], default=["short", "long"])
        weekday_filt   = st.multiselect(
            "Weekdays",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        )

        st.markdown("**ADX Range**")
        adx_r = st.slider("_adx", 0.0, 100.0, (0.0, 100.0), step=1.0, label_visibility="collapsed")
        st.markdown("**ATR Range**")
        atr_r = st.slider("_atr", 0.0, 50.0, (0.0, 50.0), step=0.1, label_visibility="collapsed")
        st.markdown("**Z-Score Range**")
        zsc_r = st.slider("_zsc", -10.0, 10.0, (-10.0, 10.0), step=0.1, label_visibility="collapsed")

        st.markdown("---")
        run = st.button("⚡  Run Backtest", use_container_width=True)

        return dict(
            src=src, uploaded=uploaded, local_path=local_path, years=years,
            tf=tf, custom_range=custom_range, c_start=c_start, c_end=c_end,
            risk_mode=risk_mode, sl_k=sl_k, tp_mode=tp_mode, tp_k=tp_k,
            direction_filt=direction_filt, weekday_filt=weekday_filt,
            adx_r=adx_r, atr_r=atr_r, zsc_r=zsc_r,
            run=run,
        )


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def render_welcome() -> None:
    st.markdown(f"""
    <div style="margin:48px auto;max-width:680px;text-align:center">
      <div style="font-size:2.8rem;margin-bottom:14px;letter-spacing:-.02em">⚡</div>
      <div style="font-size:1.15rem;font-weight:600;color:{C['t0']};margin-bottom:10px">
        Fake Breakout Backtest Engine
      </div>
      <div style="font-size:.78rem;color:{C['t2']};line-height:1.9;margin-bottom:32px">
        Load a 1-minute intraday OHLCV dataset to detect fake breakout patterns,<br>
        simulate trades with MAE/MFE tracking, and explore strike rate behaviour<br>
        across time-of-day windows, weekdays, and market regimes.
      </div>
      <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:40px">
        <span class="badge badge-green">1-min Breakout Trigger</span>
        <span class="badge badge-green">ATR / Vol-Based Risk</span>
        <span class="badge badge-dim">200MB+ Dataset Support</span>
        <span class="badge badge-dim">Parquet / CSV</span>
        <span class="badge badge-green">MAE · MFE · ADX · ATR · Z-Score</span>
      </div>
      <div style="background:{C['bg2']};border:1px solid {C['border']};
          border-radius:8px;padding:20px 24px;text-align:left">
        <div style="font-size:.60rem;color:{C['t3']};letter-spacing:.12em;
            text-transform:uppercase;margin-bottom:12px">Expected Schema</div>
        <code style="font-size:.72rem;color:{C['t2']};line-height:1.9;
            font-family:'Geist Mono',monospace;white-space:pre">
<span style="color:{C['t3']}">timestamp            open      high      low       close   volume  adx_14  atr_14  zscore_20</span>
<span style="color:{C['green']}">2022-01-03T00:29:00</span>  <span style="color:{C['t1']}">1828.72   1829.19   1828.72   1829.07  25      31.1    0.3     2.84</span>
        </code>
      </div>
      <div style="margin-top:24px;font-size:.68rem;color:{C['t3']}">
        Upload a file or specify a local path in the sidebar, then click
        <span style="color:{C['green']}">⚡ Run Backtest</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()

    st.markdown(f"""
    <div class="top-header">
      <div class="pulse"></div>
      <div>
        <div class="title">Fake Breakout Backtest</div>
        <div class="sub">MAE · MFE · Strike Rate Analyzer — 1-min Trigger Engine</div>
      </div>
      <span class="badge badge-green" style="margin-left:auto">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    for key, default in [("trades", None), ("df", None), ("selected_cell", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    p = render_sidebar()

    # ── Load data ────────────────────────────────────────────────────
    df_raw = None

    if p["src"] == "Upload File" and p["uploaded"] is not None:
        f = p["uploaded"]
        with st.spinner(f"Loading **{f.name}** …"):
            try:
                raw_bytes = f.read()
                if f.name.endswith(".parquet"):
                    df_raw = pd.read_parquet(io.BytesIO(raw_bytes))
                else:
                    df_raw = _load_csv_bytes(raw_bytes)
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                return

    elif p["src"] == "Local Path" and p["local_path"].strip():
        path = Path(p["local_path"].strip())
        if not path.exists():
            st.error(f"File not found: `{path}`")
            return
        with st.spinner(f"Loading `{path.name}` …"):
            try:
                df_raw = _load_parquet_path(str(path)) if str(path).endswith(".parquet") \
                    else _load_csv_bytes(path.read_bytes())
            except Exception as e:
                st.error(f"Failed to load: {e}")
                return

    if df_raw is None:
        render_welcome()
        return

    # ── Preprocess ───────────────────────────────────────────────────
    with st.spinner("Preprocessing …"):
        try:
            df = preprocess(df_raw, years=p["years"])
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            return

    st.session_state.df = df

    mem_mb  = df.memory_usage(deep=True).sum() / 1024**2
    n_days  = df["_date"].nunique()
    date_lo = df.index[0].strftime("%Y-%m-%d")
    date_hi = df.index[-1].strftime("%Y-%m-%d")
    feat_ok = [c for c in ["adx_14", "atr_14", "zscore_20"] if c in df.columns]

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val in [
        (c1, "Bars", f"{len(df):,}"),
        (c2, "Trading Days", f"{n_days:,}"),
        (c3, "From", date_lo),
        (c4, "To", date_hi),
        (c5, "Memory", f"{mem_mb:.1f} MB"),
    ]:
        col.markdown(kpi(lbl, val), unsafe_allow_html=True)

    if feat_ok:
        st.markdown(
            '<div style="margin-top:8px;margin-bottom:4px">'
            + "".join(f'<span class="badge badge-green" style="margin-right:6px">{f}</span>' for f in feat_ok)
            + '</div>',
            unsafe_allow_html=True,
        )

    # ATR-mode warning when column absent
    if p["risk_mode"] == "atr" and "atr_14" not in df.columns:
        st.warning(
            "⚠️ ATR-Based risk selected but `atr_14` column not found — "
            "falling back to fixed-% sizing."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run strategy ─────────────────────────────────────────────────
    if p["run"]:
        st.session_state.selected_cell = None
        tf_min = TF_MINUTES[p["tf"]]
        cstart = p["c_start"] if p["custom_range"] else ""
        cend   = p["c_end"]   if p["custom_range"] else ""

        with st.spinner("Detecting breakouts on 1-min bars …"):
            signals = compute_signals(df, tf_min, cstart, cend)

        if signals.empty:
            st.warning("No breakout signals found. Try a different timeframe or range window.")
            st.session_state.trades = pd.DataFrame()
            return

        prog = st.progress(0, text=f"Simulating {len(signals):,} trades …")
        trades = simulate_trades(
            df, signals,
            risk_mode=p["risk_mode"],
            sl_k=p["sl_k"],
            tp_mode=p["tp_mode"],
            tp_k=p["tp_k"],
        )
        prog.progress(100, text="Done.")
        prog.empty()
        st.session_state.trades = trades

    if st.session_state.trades is None:
        st.info("Configure parameters in the sidebar and click **⚡ Run Backtest**.")
        return

    trades_all = st.session_state.trades
    if trades_all.empty:
        st.warning("No trades were generated.")
        return

    # ── Apply sidebar filters ─────────────────────────────────────────
    ft = trades_all.copy()
    if p["direction_filt"]:
        ft = ft[ft["direction"].isin(p["direction_filt"])]
    if p["weekday_filt"]:
        ft = ft[ft["_weekday_name"].isin(p["weekday_filt"])]
    if "adx_14" in ft.columns:
        ft = ft[ft["adx_14"].between(*p["adx_r"], inclusive="both") | ft["adx_14"].isna()]
    if "atr_14" in ft.columns:
        ft = ft[ft["atr_14"].between(*p["atr_r"], inclusive="both") | ft["atr_14"].isna()]
    if "zscore_20" in ft.columns:
        ft = ft[ft["zscore_20"].between(*p["zsc_r"], inclusive="both") | ft["zscore_20"].isna()]

    n_ft   = len(ft)
    sr_ft  = ft["win"].mean() * 100 if n_ft else 0
    sr_cls = "badge-green" if sr_ft >= 50 else "badge-red"
    rm_cls = "badge-yellow" if p["risk_mode"] == "atr" else "badge-dim"
    rm_lbl = "ATR Risk" if p["risk_mode"] == "atr" else "Fixed % Risk"

    st.markdown(
        f'<span class="badge badge-dim" style="margin-right:8px">{n_ft:,} trades after filters</span>'
        f'<span class="badge {sr_cls}" style="margin-right:8px">{sr_ft:.1f}% strike rate</span>'
        f'<span class="badge {rm_cls}">{rm_lbl}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Custom range → skip heatmap ───────────────────────────────────
    if p["custom_range"]:
        st.markdown(
            f'<div class="section-label">Custom Range Analytics'
            f'<span class="badge badge-green" style="margin-left:10px">'
            f'{p["c_start"]} – {p["c_end"]}</span></div>',
            unsafe_allow_html=True,
        )
        render_analytics(ft, f"{p['c_start']}–{p['c_end']}")
        return

    # ── Heatmap ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Strike Rate Heatmap</div>', unsafe_allow_html=True)

    agg_data = aggregate_heatmap(ft)
    if agg_data.empty:
        st.warning("No data to display — adjust filters.")
        return

    event = st.plotly_chart(
        fig_heatmap(agg_data),
        use_container_width=True,
        on_select="rerun",
        key="main_heatmap",
    )

    if event and event.get("selection", {}).get("points"):
        pt = event["selection"]["points"][0]
        cd = pt.get("x")
        ct = pt.get("y")
        if cd and ct:
            st.session_state.selected_cell = (cd, ct)

    if st.session_state.selected_cell:
        day, rtime = st.session_state.selected_cell
        cell_trades = ft[(ft["_weekday_name"] == day) & (ft["range_time"] == rtime)]
        label = f"{day} · {rtime}"
        st.markdown(
            f'<div class="section-label">Analytics Panel'
            f'<span class="badge badge-green" style="margin-left:10px">{label}</span></div>',
            unsafe_allow_html=True,
        )
        render_analytics(cell_trades, label)
    else:
        st.markdown('<div class="section-label">Global Analytics</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:.73rem;color:{C["t2"]};margin-bottom:12px">'
            f'Click a heatmap cell to drill into a specific time window. '
            f'Expand below for aggregate view across all filtered trades.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Show Global Analytics", expanded=False):
            render_analytics(ft, "All Filtered Trades")


if __name__ == "__main__":
    main()
