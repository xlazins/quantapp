import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

PRICE_FLOOR = 1000.0
PRICE_CEIL  = 30000.0
TIMEZONE    = "America/New_York"

# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NQ Magic Hour Backtest", layout="wide", page_icon="📈")
st.title("📈 NQ Magic Hour Backtest")
st.caption("Define your range window, entry logic, and risk parameters in the sidebar, then hit Run.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv", "txt"],
        help="Upload your NQ data file directly."
    )

    st.divider()
    st.subheader("📊 Timeframe")
    tf_label = st.selectbox("Resample to", ["1 min", "5 min", "15 min", "30 min"])
    tf_map   = {"1 min": "1min", "5 min": "5min", "15 min": "15min", "30 min": "30min"}
    timeframe = tf_map[tf_label]

    st.divider()
    st.subheader("🕐 Range Window (ET)")
    c1, c2 = st.columns(2)
    range_start_h = c1.number_input("Start Hour", 0, 23, 7)
    range_start_m = c2.number_input("Start Min",  0, 59, 0)
    c3, c4 = st.columns(2)
    range_end_h   = c3.number_input("End Hour",   0, 23, 7)
    range_end_m   = c4.number_input("End Min",    0, 59, 59)

    st.divider()
    st.subheader("🎯 Entry Logic")
    trade_type = st.radio(
        "Signal Type",
        ["Breakout Fade", "Sweep Fade"],
        help=(
            "**Breakout Fade**: bar *closes* outside the range → enter fade.\n\n"
            "**Sweep Fade**: bar *wicks* outside range but *closes back inside* → enter fade."
        )
    )

    st.divider()
    st.subheader("📐 Risk / Reward")
    stop_pct = st.number_input("Stop Loss %", 0.1, 10.0, 0.5, step=0.1) / 100

    target_type = st.selectbox("Target", ["Midpoint", "Range High", "Range Low", "Custom %"])
    custom_target_pct = None
    if target_type == "Custom %":
        custom_target_pct = st.number_input("Target % from entry", 0.1, 20.0, 1.0, step=0.1) / 100

    analysis_hrs = int(st.number_input("Analysis Window (hrs after range end)", 1, 12, 3))

    st.divider()
    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data(file_bytes: bytes, tf: str) -> pd.DataFrame:

    content = file_bytes.decode("utf-8")
    first   = content.splitlines()[0]
    sep     = "\t" if "\t" in first else ","

    df = pd.read_csv(StringIO(content), sep=sep, on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower()

    # ── Filter to front-month symbol if multi-symbol ───────────────────
    if "symbol" in df.columns:
        top_symbol = df["symbol"].value_counts().idxmax()
        df = df[df["symbol"] == top_symbol].copy()

    # ── Timestamp ──────────────────────────────────────────────────────
    if "date" in df.columns and "time" in df.columns:
        df["ts_raw"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce"
        )
        df["ts_raw"] = df["ts_raw"].dt.tz_localize(
            TIMEZONE, ambiguous="infer", nonexistent="shift_forward"
        )
    else:
        ts_col = next(
            (c for c in df.columns if c in ["ts_event","timestamp","datetime","date"]),
            df.columns[0]
        )
        df["ts_raw"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df["ts_raw"] = df["ts_raw"].dt.tz_convert(TIMEZONE)

    df = df.dropna(subset=["ts_raw"])
    df = df.set_index("ts_raw").sort_index()

    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Price filter ───────────────────────────────────────────────────
    mask = (
        df["open"].between(PRICE_FLOOR, PRICE_CEIL) &
        df["high"].between(PRICE_FLOOR, PRICE_CEIL) &
        df["low"].between(PRICE_FLOOR, PRICE_CEIL)  &
        df["close"].between(PRICE_FLOOR, PRICE_CEIL) &
        (df["low"] <= df["high"])
    )
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        mask &= df["volume"].fillna(0) > 0

    df = df[mask].copy()

    # ── Resample ───────────────────────────────────────────────────────
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    df = df.resample(tf).agg(agg).dropna(subset=["open","close"])

    return df

    # ── Filter to front-month symbol if multi-symbol file ─────────────────
    if "symbol" in df.columns:
        # Pick the symbol with the most rows (front month)
        top_symbol = df["symbol"].value_counts().idxmax()
        df = df[df["symbol"] == top_symbol].copy()

    # ── Timestamp ──────────────────────────────────────────────────────────
    if "Date" in df.columns and "Time" in df.columns:
        df["ts_raw"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
        df["ts_raw"] = df["ts_raw"].dt.tz_localize(
            TIMEZONE, ambiguous="infer", nonexistent="shift_forward"
        )
    else:
        ts_col = next(
            (c for c in df.columns if c.lower() in ["ts_event","timestamp","datetime","date"]),
            df.columns[0]
        )
        df["ts_raw"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df["ts_raw"] = df["ts_raw"].dt.tz_convert(TIMEZONE)

    df = df.dropna(subset=["ts_raw"])

    # ── OHLCV columns ──────────────────────────────────────────────────────
    col_map = {}
    for target in ["open","high","low","close","volume"]:
        for col in df.columns:
            if col.lower() == target:
                col_map[target] = col
                break

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df = df.set_index("ts_raw").sort_index()

    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Price filter ───────────────────────────────────────────────────────
    mask = (
        df["open"].between(PRICE_FLOOR, PRICE_CEIL) &
        df["high"].between(PRICE_FLOOR, PRICE_CEIL) &
        df["low"].between(PRICE_FLOOR, PRICE_CEIL)  &
        df["close"].between(PRICE_FLOOR, PRICE_CEIL) &
        (df["low"] <= df["high"])
    )
    # Volume filter only if column exists and is non-zero
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        mask &= df["volume"].fillna(0) > 0

    df = df[mask].copy()

    # ── Resample ───────────────────────────────────────────────────────────
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    df = df.resample(tf).agg(agg).dropna(subset=["open","close"])

    return df

    # ── Timestamp ─────────────────────────────────────────────────────────────
    # Handle separate Date + Time columns (common in NQ exports)
    if "Date" in df.columns and "Time" in df.columns:
        df["ts_raw"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
        df["ts_raw"] = df["ts_raw"].dt.tz_localize(
            TIMEZONE, ambiguous="infer", nonexistent="shift_forward"
        )
    else:
        ts_col = next(
            (c for c in df.columns if c.lower() in ["ts_event","timestamp","datetime","date"]),
            df.columns[0]
        )
        df["ts_raw"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df["ts_raw"] = df["ts_raw"].dt.tz_convert(TIMEZONE)

    df = df.dropna(subset=["ts_raw"])

    # ── OHLCV columns ─────────────────────────────────────────────────────────
    col_map = {}
    for target in ["open","high","low","close","volume"]:
        for col in df.columns:
            if col.lower() == target:
                col_map[target] = col
                break

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df = df.set_index("ts_raw").sort_index()

    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Price filter ──────────────────────────────────────────────────────────
    mask = (
        df["open"].between(PRICE_FLOOR, PRICE_CEIL) &
        df["high"].between(PRICE_FLOOR, PRICE_CEIL) &
        df["low"].between(PRICE_FLOOR, PRICE_CEIL)  &
        df["close"].between(PRICE_FLOOR, PRICE_CEIL) &
        (df["low"] <= df["high"])
    )
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        mask &= df["volume"] > 0
    df = df[mask].copy()

    # ── Resample ──────────────────────────────────────────────────────────────
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    df = df.resample(tf).agg(agg).dropna(subset=["open","close"])

    return df


def compute_target(entry: float, is_short: bool, range_high: float, range_low: float,
                   midpoint: float, target_type: str, custom_pct: float | None) -> float:
    if target_type == "Midpoint":
        return midpoint
    elif target_type == "Range High":
        return range_high
    elif target_type == "Range Low":
        return range_low
    else:  # Custom %
        return entry * (1 - custom_pct) if is_short else entry * (1 + custom_pct)


def run_backtest(df, range_start_h, range_start_m, range_end_h, range_end_m,
                 trade_type, stop_pct, target_type, custom_target_pct, analysis_hrs):
    results = []
    days = sorted(set(df.index.date))

    range_start_minutes = range_start_h * 60 + range_start_m
    range_end_minutes   = range_end_h   * 60 + range_end_m

    for day in days:
        day_data = df[df.index.date == day]
        if len(day_data) < 5:
            continue

        # ── Build range ───────────────────────────────────────────────────────
        def in_range_window(ts):
            m = ts.hour * 60 + ts.minute
            return range_start_minutes <= m <= range_end_minutes

        range_bars = day_data[[in_range_window(ts) for ts in day_data.index]]
        if len(range_bars) < 2:
            continue

        range_high = range_bars["high"].max()
        range_low  = range_bars["low"].min()
        range_size = range_high - range_low
        midpoint   = (range_high + range_low) / 2.0

        if range_size <= 0:
            continue

        # ── Post-range bars ───────────────────────────────────────────────────
        range_end_ts = range_bars.index[-1]
        cutoff_ts    = range_end_ts + pd.Timedelta(hours=analysis_hrs)

        post_bars = day_data[
            (day_data.index > range_end_ts) &
            (day_data.index <= cutoff_ts)
        ]

        if len(post_bars) == 0:
            continue

        # ── Find signal ───────────────────────────────────────────────────────
        entry_ts    = None
        entry_price = None
        is_short    = None

        for ts, bar in post_bars.iterrows():
            if trade_type == "Breakout Fade":
                # Close outside the range
                if bar["close"] > range_high:
                    entry_ts, entry_price, is_short = ts, bar["close"], True
                    break
                elif bar["close"] < range_low:
                    entry_ts, entry_price, is_short = ts, bar["close"], False
                    break
            else:  # Sweep Fade
                # Wick outside, close back inside
                if bar["high"] > range_high and bar["close"] <= range_high:
                    entry_ts, entry_price, is_short = ts, bar["close"], True
                    break
                elif bar["low"] < range_low and bar["close"] >= range_low:
                    entry_ts, entry_price, is_short = ts, bar["close"], False
                    break

        if entry_ts is None:
            results.append({
                "date": day, "range_high": range_high, "range_low": range_low,
                "range_size": range_size, "has_signal": False,
                "target_hit": False, "stop_hit": False,
                "mae": np.nan, "mae_pct_range": np.nan, "mae_pct_price": np.nan,
                "extension": np.nan,
            })
            continue

        # ── Trade tracking ────────────────────────────────────────────────────
        stop_price   = entry_price * (1 + stop_pct if is_short else 1 - stop_pct)
        target_price = compute_target(
            entry_price, is_short, range_high, range_low,
            midpoint, target_type, custom_target_pct
        )

        # Sanity: target must be in the right direction
        if is_short and target_price >= entry_price:
            target_price = midpoint
        if not is_short and target_price <= entry_price:
            target_price = midpoint

        after_entry = post_bars[post_bars.index >= entry_ts]
        mae         = 0.0
        target_hit  = False
        stop_hit    = False

        for ts, bar in after_entry.iterrows():
            if is_short:
                adverse = bar["high"] - entry_price
                if adverse > mae:
                    mae = adverse
                if bar["high"] >= stop_price:
                    stop_hit = True
                    break
                if bar["low"] <= target_price:
                    target_hit = True
                    break
            else:
                adverse = entry_price - bar["low"]
                if adverse > mae:
                    mae = adverse
                if bar["low"] <= stop_price:
                    stop_hit = True
                    break
                if bar["high"] >= target_price:
                    target_hit = True
                    break

        extension = abs(entry_price - (range_high if is_short else range_low))

        results.append({
            "date":          day,
            "range_high":    range_high,
            "range_low":     range_low,
            "range_size":    range_size,
            "midpoint":      midpoint,
            "has_signal":    True,
            "is_short":      is_short,
            "entry_price":   entry_price,
            "stop_price":    stop_price,
            "target_price":  target_price,
            "target_hit":    target_hit,
            "stop_hit":      stop_hit,
            "mae":           mae,
            "mae_pct_price": mae / entry_price * 100 if entry_price > 0 else np.nan,
            "mae_pct_range": mae / range_size * 100   if range_size > 0  else np.nan,
            "extension":     extension / range_size * 100,
        })

    return pd.DataFrame(results)


def render_results(results: pd.DataFrame):
    trades = results[results["has_signal"]].copy()
    completed = trades[trades["target_hit"] | trades["stop_hit"]].copy()

    total_days  = len(results)
    signal_days = len(trades)
    comp_n      = len(completed)

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    wins     = int(completed["target_hit"].sum())
    losses   = int(completed["stop_hit"].sum())
    win_rate = wins / comp_n * 100 if comp_n > 0 else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Days",       f"{total_days:,}")
    k2.metric("Signal Days",      f"{signal_days:,}",
              f"{signal_days/total_days*100:.1f}% of days")
    k3.metric("Completed Trades", f"{comp_n:,}")
    k4.metric("Win Rate",         f"{win_rate:.1f}%",  f"{wins}W / {losses}L")
    k5.metric("Avg MAE (% Range)",
              f"{completed['mae_pct_range'].mean():.1f}%" if comp_n > 0 else "—")

    if comp_n == 0:
        st.warning("No completed trades found with these settings.")
        return

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Win rate by month
    completed["ym"] = pd.to_datetime(completed["date"]).dt.to_period("M")
    monthly = (
        completed.groupby("ym")
        .agg(wr=("target_hit", "mean"), n=("target_hit", "count"),
             mae_p75=("mae_pct_range", lambda x: x.quantile(0.75)))
        .reset_index()
    )
    monthly["ym_str"] = monthly["ym"].astype(str)

    with col_left:
        st.subheader("Monthly Win Rate")
        fig_wr = go.Figure()
        fig_wr.add_trace(go.Bar(
            x=monthly["ym_str"],
            y=(monthly["wr"] * 100).round(1),
            marker_color=[
                "#ef4444" if v < 50 else "#f97316" if v < 60 else "#22c55e"
                for v in monthly["wr"] * 100
            ],
            text=(monthly["wr"] * 100).round(1).astype(str) + "%",
            textposition="outside"
        ))
        fig_wr.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.4)
        fig_wr.update_layout(
            yaxis_title="Win Rate %", xaxis_title="Month",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 105]), height=350,
            font=dict(color="white")
        )
        st.plotly_chart(fig_wr, use_container_width=True)

    # MAE distribution
    with col_right:
        st.subheader("MAE Distribution (% of Range)")
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Histogram(
            x=completed["mae_pct_range"].dropna(),
            nbinsx=30,
            marker_color="#6366f1",
            opacity=0.85
        ))
        fig_mae.update_layout(
            xaxis_title="MAE as % of Range", yaxis_title="# Trades",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=350, font=dict(color="white")
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    # ── Extension zones + MAE stats ───────────────────────────────────────────
    col_ext, col_stats = st.columns(2)

    with col_ext:
        st.subheader("Extension Zones")
        ext = completed["extension"].dropna()
        zones = {
            "Z1 0–25%":   (ext <= 25).sum(),
            "Z2 25–50%":  ((ext > 25) & (ext <= 50)).sum(),
            "Z3 50–75%":  ((ext > 50) & (ext <= 75)).sum(),
            "Z4 75–100%": ((ext > 75) & (ext <= 100)).sum(),
            "INV >100%":  (ext > 100).sum(),
        }
        fig_ext = go.Figure(go.Bar(
            x=list(zones.keys()),
            y=list(zones.values()),
            marker_color=["#22c55e","#84cc16","#f59e0b","#f97316","#ef4444"],
            text=list(zones.values()), textposition="outside"
        ))
        fig_ext.update_layout(
            yaxis_title="# Trades",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=320, font=dict(color="white")
        )
        st.plotly_chart(fig_ext, use_container_width=True)

    with col_stats:
        st.subheader("MAE Percentiles")
        mae_s = completed["mae_pct_range"].dropna()
        stats_df = pd.DataFrame({
            "Percentile": ["P25", "P50 (Median)", "P75", "P90", "P95"],
            "MAE % of Range": [
                f"{mae_s.quantile(0.25):.1f}%",
                f"{mae_s.quantile(0.50):.1f}%",
                f"{mae_s.quantile(0.75):.1f}%",
                f"{mae_s.quantile(0.90):.1f}%",
                f"{mae_s.quantile(0.95):.1f}%",
            ],
            "MAE % of Price": [
                f"{completed['mae_pct_price'].quantile(0.25):.3f}%",
                f"{completed['mae_pct_price'].quantile(0.50):.3f}%",
                f"{completed['mae_pct_price'].quantile(0.75):.3f}%",
                f"{completed['mae_pct_price'].quantile(0.90):.3f}%",
                f"{completed['mae_pct_price'].quantile(0.95):.3f}%",
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    st.divider()

    # ── Month-by-month table ──────────────────────────────────────────────────
    st.subheader("Month-by-Month Breakdown")
    monthly_display = monthly.copy()
    monthly_display["Win Rate"] = (monthly_display["wr"] * 100).round(1).astype(str) + "%"
    monthly_display["MAE P75"]  = monthly_display["mae_p75"].round(1).astype(str) + "%"
    monthly_display = monthly_display.rename(columns={"ym_str": "Month", "n": "Trades"})
    st.dataframe(
        monthly_display[["Month", "Trades", "Win Rate", "MAE P75"]],
        hide_index=True, use_container_width=True
    )

    st.divider()

    # ── Raw results download ──────────────────────────────────────────────────
    st.subheader("Raw Trade Log")
    st.dataframe(completed.reset_index(drop=True), use_container_width=True, height=300)

    csv = completed.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Results CSV", csv, "magic_hour_results.csv", "text/csv")


# ── Main ──────────────────────────────────────────────────────────────────────
if run_btn:
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
        st.stop()

    try:
        df = load_data(uploaded_file.getvalue(), timeframe)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    st.success(f"Loaded {len(df):,} bars | {df.index.min().date()} → {df.index.max().date()}")

    with st.spinner("Running backtest..."):
        results = run_backtest(
            df,
            range_start_h, range_start_m,
            range_end_h,   range_end_m,
            trade_type, stop_pct,
            target_type, custom_target_pct,
            analysis_hrs
        )

    render_results(results)

else:
    st.info("👈 Configure your settings in the sidebar and click **Run Backtest**.")
