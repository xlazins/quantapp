"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  FAKE BREAKOUT BACKTEST ENGINE  v3                                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  CHANGES FROM v2                                                          ║
║  • Heatmap: static grid, strike % on cells, "09:30–09:35" range labels   ║
║  • Heatmap click → analytics update fixed via @st.fragment               ║
║  • Removed P&L entirely — goal is repeatable MAE/MFE setup discovery     ║
║  • Removed weekday filter from sidebar                                    ║
║  • Removed ADX/ATR/Z-Score range sliders from sidebar                    ║
║  • New Regime tab: how each feature bin shifts MAE and MFE               ║
║  • @st.fragment on heatmap+analytics — only that section rerenders       ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Fake Breakout Backtest",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TF_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MAX_TRADE_BARS = 600

C = {
    "bg0":"#070709","bg1":"#0d0d10","bg2":"#131318","bg3":"#1a1a20",
    "border":"#22222a","border2":"#2e2e38",
    "t0":"#f2f2f4","t1":"#9898a8","t2":"#55555f","t3":"#35353f",
    "green":"#00e87a","red":"#ff4455","yellow":"#ffc840","blue":"#3d9eff",
    "green_dim":"rgba(0,232,122,0.10)","red_dim":"rgba(255,68,85,0.10)",
}

PLOTLY_BASE = dict(
    paper_bgcolor=C["bg0"], plot_bgcolor=C["bg1"],
    font=dict(family="'Geist Mono','JetBrains Mono',monospace", color=C["t1"], size=11),
    margin=dict(l=55, r=20, t=50, b=50),
)

AXIS = dict(
    gridcolor=C["border"], linecolor=C["border"], tickcolor=C["border"],
    zeroline=False, tickfont=dict(size=10, color=C["t2"]),
)


def _layout(**kw) -> dict:
    """Merge PLOTLY_BASE with per-call overrides — no duplicate-kwarg TypeError."""
    return {**PLOTLY_BASE, **kw}


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@300;400;500;600;700&display=swap');
*,html,body{{font-family:'Geist Mono','JetBrains Mono',monospace!important;box-sizing:border-box}}
.stApp,[data-testid="stAppViewContainer"]{{background:{C["bg0"]}!important;color:{C["t0"]}!important}}
[data-testid="stHeader"]{{background:transparent!important}}
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:{C["bg1"]}}}
::-webkit-scrollbar-thumb{{background:{C["border2"]};border-radius:3px}}
[data-testid="stSidebar"]{{background:{C["bg1"]}!important;border-right:1px solid {C["border"]}!important}}
[data-testid="stSidebar"] *{{color:{C["t0"]}!important}}
[data-testid="stSidebar"] .stSelectbox>div>div,[data-testid="stSidebar"] .stMultiSelect>div>div{{
  background:{C["bg2"]}!important;border:1px solid {C["border"]}!important;border-radius:5px!important}}
[data-testid="stSidebar"] section{{padding-top:0!important}}
.stTextInput>div>div>input,.stNumberInput>div>div>input{{
  background:{C["bg2"]}!important;border:1px solid {C["border"]}!important;
  color:{C["t0"]}!important;border-radius:5px!important;font-family:'Geist Mono',monospace!important}}
.stTextInput>div>div>input:focus,.stNumberInput>div>div>input:focus{{
  border-color:{C["green"]}!important;box-shadow:0 0 0 2px rgba(0,232,122,.1)!important;outline:none!important}}
.stButton>button{{
  background:{C["bg2"]}!important;border:1px solid {C["border2"]}!important;color:{C["t1"]}!important;
  border-radius:5px!important;font-family:'Geist Mono',monospace!important;
  font-size:.78rem!important;letter-spacing:.04em!important;padding:8px 16px!important;transition:all .15s!important}}
.stButton>button:hover{{border-color:{C["green"]}!important;color:{C["green"]}!important;background:{C["green_dim"]}!important}}
[data-testid="stSlider"]>div>div>div>div{{background:{C["green"]}!important}}
[data-testid="stSlider"] [role="slider"]{{background:{C["green"]}!important;border:2px solid {C["bg0"]}!important;box-shadow:0 0 0 1px {C["green"]}!important}}
[data-testid="stFileUploader"]>div{{background:{C["bg2"]}!important;border:1px dashed {C["border2"]}!important;border-radius:6px!important}}
[data-testid="stFileUploader"]>div:hover{{border-color:{C["green"]}!important}}
.stTabs [data-baseweb="tab-list"]{{background:{C["bg1"]}!important;border-bottom:1px solid {C["border"]}!important;gap:0!important;padding:0!important}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:{C["t2"]}!important;border:none!important;
  border-bottom:2px solid transparent!important;font-size:.73rem!important;letter-spacing:.06em!important;padding:8px 16px!important}}
.stTabs [aria-selected="true"]{{color:{C["green"]}!important;border-bottom:2px solid {C["green"]}!important;background:{C["green_dim"]}!important}}
.streamlit-expanderHeader{{background:{C["bg2"]}!important;border:1px solid {C["border"]}!important;
  border-radius:5px!important;color:{C["t1"]}!important;font-size:.78rem!important}}
.streamlit-expanderContent{{background:{C["bg1"]}!important;border:1px solid {C["border"]}!important;
  border-top:none!important;border-radius:0 0 5px 5px!important}}
.stAlert{{background:{C["bg2"]}!important;border:1px solid {C["border"]}!important;border-radius:5px!important}}
.stDataFrame{{background:{C["bg1"]}!important;border-radius:6px}}
.stSpinner>div{{border-top-color:{C["green"]}!important}}
.stCheckbox label,.stRadio label{{color:{C["t1"]}!important;font-size:.8rem!important}}
.kpi-card{{background:{C["bg2"]};border:1px solid {C["border"]};border-radius:6px;padding:14px 18px}}
.kpi-label{{font-size:.60rem;color:{C["t2"]};text-transform:uppercase;letter-spacing:.12em;margin-bottom:5px}}
.kpi-value{{font-size:1.35rem;font-weight:600;color:{C["t0"]};line-height:1}}
.kpi-value.green{{color:{C["green"]}}}.kpi-value.red{{color:{C["red"]}}}.kpi-value.yellow{{color:{C["yellow"]}}}
.section-label{{font-size:.60rem;font-weight:600;color:{C["t3"]};text-transform:uppercase;
  letter-spacing:.15em;padding-bottom:6px;border-bottom:1px solid {C["border"]};margin:18px 0 10px 0}}
.badge{{display:inline-block;font-size:.60rem;padding:2px 8px;border-radius:3px;letter-spacing:.08em;text-transform:uppercase;font-weight:500}}
.badge-green{{background:{C["green_dim"]};color:{C["green"]};border:1px solid rgba(0,232,122,.25)}}
.badge-red{{background:{C["red_dim"]};color:{C["red"]};border:1px solid rgba(255,68,85,.25)}}
.badge-dim{{background:{C["bg3"]};color:{C["t2"]};border:1px solid {C["border"]}}}
.badge-yellow{{background:rgba(255,200,64,.10);color:{C["yellow"]};border:1px solid rgba(255,200,64,.25)}}
.top-header{{display:flex;align-items:center;gap:14px;padding-bottom:18px;border-bottom:1px solid {C["border"]};margin-bottom:22px}}
.top-header .pulse{{width:8px;height:8px;background:{C["green"]};border-radius:50%;box-shadow:0 0 8px {C["green"]};flex-shrink:0}}
.top-header .title{{font-size:1.05rem;font-weight:600;color:{C["t0"]};letter-spacing:.02em}}
.top-header .sub{{font-size:.65rem;color:{C["t2"]};margin-top:3px;letter-spacing:.06em}}
.cell-banner{{background:{C["bg2"]};border:1px solid {C["green"]};border-radius:6px;
  padding:10px 16px;margin-bottom:14px;display:flex;align-items:center;gap:12px;font-size:.78rem}}
hr{{border-color:{C["border"]}!important}}
div[data-testid="column"]{{padding:0 4px!important}}
</style>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=4)
def _load_csv(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))

@st.cache_data(show_spinner=False, max_entries=4)
def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False, max_entries=4)
def preprocess(df_raw: pd.DataFrame, years: int) -> pd.DataFrame:
    df = df_raw.copy()
    ts = next((c for c in df.columns if c.lower() in ("timestamp","datetime","time","date")), None)
    if ts:
        df[ts] = pd.to_datetime(df[ts], utc=False, errors="coerce")
        df = df.set_index(ts)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    df.index.name = "timestamp"
    df = df[~df.index.isna()].sort_index()
    df.columns = [c.lower().strip() for c in df.columns]
    miss = [c for c in ["open","high","low","close"] if c not in df.columns]
    if miss:
        raise ValueError(f"Missing: {miss}")
    for col in ["open","high","low","close","volume","adx_14","atr_14","zscore_20"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])
    if years > 0:
        df = df.loc[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    if df.empty:
        raise ValueError("No data after filtering.")
    df["_date"]         = df.index.date
    df["_weekday_name"] = df.index.day_name()
    df["_bar_idx"]      = np.arange(len(df), dtype=np.int32)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SIGNALS  — breakout ALWAYS detected on 1-min bars
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=8)
def compute_signals(df: pd.DataFrame, tf_minutes: int,
                    cs: str = "", ce: str = "") -> pd.DataFrame:
    freq = f"{tf_minutes}min"
    rc = df.resample(freq).agg(
        range_high=("high","max"),
        range_low =("low", "min"),
        range_vol =("close","std"),      # realized vol of consolidation window
    ).dropna(subset=["range_high","range_low"])
    rc.index.name = "range_start"
    rc = rc.reset_index()
    rc["range_end"]   = rc["range_start"] + pd.Timedelta(minutes=tf_minutes)
    rc["range_label"] = (rc["range_start"].dt.strftime("%H:%M") + "–" +
                         rc["range_end"].dt.strftime("%H:%M"))

    if cs and ce:
        try:
            t0 = pd.to_datetime(cs, format="%H:%M").time()
            t1 = pd.to_datetime(ce, format="%H:%M").time()
            rc = rc[(rc["range_start"].dt.time >= t0) & (rc["range_start"].dt.time < t1)]
        except Exception:
            pass

    if rc.empty:
        return pd.DataFrame()

    extra = [c for c in ["adx_14","atr_14","zscore_20"] if c in df.columns]
    bars  = df[["high","low","close","_date","_weekday_name","_bar_idx"] + extra].reset_index()
    bars  = bars.sort_values("timestamp")
    rc    = rc.sort_values("range_end")

    merged = pd.merge_asof(
        bars,
        rc[["range_start","range_end","range_high","range_low","range_vol","range_label"]],
        left_on="timestamp", right_on="range_end", direction="backward",
    ).dropna(subset=["range_start"])

    merged["_rdate"] = merged["range_start"].dt.date
    merged = merged[(merged["_date"] == merged["_rdate"]) &
                    (merged["timestamp"] >= merged["range_end"])]
    if merged.empty:
        return pd.DataFrame()

    sh = merged[merged["close"] > merged["range_high"]].copy()
    sh["direction"] = "short"; sh["entry_price"] = sh["close"]
    lo = merged[merged["close"] < merged["range_low"]].copy()
    lo["direction"] = "long";  lo["entry_price"] = lo["close"]

    sig = pd.concat([sh, lo], ignore_index=True).sort_values("timestamp")
    sig = sig.groupby(["range_start","direction"], sort=False).first().reset_index()
    return sig


# ──────────────────────────────────────────────────────────────────────────────
# RISK LEVELS
# ──────────────────────────────────────────────────────────────────────────────
def _levels(ep, rh, rl, rvol, atr, is_short, risk_mode, sl_k, tp_mode, tp_k):
    has_atr = ~np.isnan(atr)
    sl_dist = np.where(has_atr, sl_k*atr, ep*sl_k/100) if risk_mode=="atr" else ep*(sl_k/100)
    sl = np.where(is_short, ep+sl_dist, ep-sl_dist)
    if tp_mode == "range":
        tp = np.where(is_short, rl, rh)
    elif tp_mode == "vol":
        has_vol = ~np.isnan(rvol) & (rvol > 0)
        td = np.where(has_vol, tp_k*rvol, np.where(has_atr, tp_k*atr, ep*tp_k/100))
        tp = np.where(is_short, ep-td, ep+td)
    else:
        tp = np.where(is_short, ep*(1-tp_k/100), ep*(1+tp_k/100))
    return sl, tp


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION  — MAE/MFE only, no P&L
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=8)
def simulate_trades(df: pd.DataFrame, signals: pd.DataFrame,
                    risk_mode: str, sl_k: float,
                    tp_mode: str,   tp_k: float = 1.0,
                    max_bars: int = MAX_TRADE_BARS) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    N      = len(highs)

    ep      = signals["entry_price"].values.astype(np.float64)
    rh      = signals["range_high"].values.astype(np.float64)
    rl      = signals["range_low"].values.astype(np.float64)
    rvol    = signals["range_vol"].values.astype(np.float64) if "range_vol" in signals.columns \
              else np.full(len(signals), np.nan)
    atr_col = signals["atr_14"].values.astype(np.float64) if "atr_14" in signals.columns \
              else np.full(len(signals), np.nan)
    is_sh   = signals["direction"].values == "short"
    eb      = signals["_bar_idx"].values.astype(np.int32)

    sl_arr, tp_arr = _levels(ep, rh, rl, rvol, atr_col, is_sh, risk_mode, sl_k, tp_mode, tp_k)

    n        = len(signals)
    outcomes = np.full(n, "timeout", dtype=object)
    exit_px  = ep.copy()
    mae_pts  = np.zeros(n, np.float64)
    mfe_pts  = np.zeros(n, np.float64)
    dur_bars = np.zeros(n, np.int32)

    for i in range(n):
        s = int(eb[i]) + 1
        e_ = min(s + max_bars, N)
        if s >= N:
            continue
        fh, fl, fc = highs[s:e_], lows[s:e_], closes[s:e_]
        if not len(fh):
            continue
        shrt = bool(is_sh[i])
        sl, tp = sl_arr[i], tp_arr[i]

        sl_hit = fh >= sl if shrt else fl <= sl
        tp_hit = fl <= tp if shrt else fh >= tp
        f_sl   = int(np.argmax(sl_hit)) if sl_hit.any() else len(fh)
        f_tp   = int(np.argmax(tp_hit)) if tp_hit.any() else len(fh)

        if not sl_hit.any() and not tp_hit.any():
            xl, outcomes[i], exit_px[i] = len(fh)-1, "timeout", fc[-1]
        elif f_sl <= f_tp:
            xl, outcomes[i], exit_px[i] = f_sl, "sl_hit", sl
        else:
            xl, outcomes[i], exit_px[i] = f_tp, "tp_hit", tp

        dur_bars[i] = xl + 1
        th, tl = fh[:xl+1], fl[:xl+1]
        if shrt:
            mfe_pts[i] = max(0.0, ep[i] - float(np.min(tl)))
            mae_pts[i] = max(0.0, float(np.max(th)) - ep[i])
        else:
            mfe_pts[i] = max(0.0, float(np.max(th)) - ep[i])
            mae_pts[i] = max(0.0, ep[i] - float(np.min(tl)))

    res = signals.copy()
    res["sl_price"]   = sl_arr
    res["tp_price"]   = tp_arr
    res["outcome"]    = outcomes
    res["exit_price"] = exit_px
    res["mae"]        = mae_pts
    res["mfe"]        = mfe_pts
    res["duration"]   = dur_bars
    res["tp_hit"]     = outcomes == "tp_hit"
    res["mae_pct"]    = mae_pts / ep * 100
    res["mfe_pct"]    = mfe_pts / ep * 100
    return res


# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=16)
def aggregate_heatmap(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    agg = (trades.groupby(["_weekday_name","range_label"])
           .agg(n=("tp_hit","count"), n_tp=("tp_hit","sum"),
                avg_mae=("mae_pct","mean"), avg_mfe=("mfe_pct","mean"))
           .reset_index())
    agg["strike_rate"] = (agg["n_tp"] / agg["n"] * 100).round(1)
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# HEATMAP FIGURE
# ──────────────────────────────────────────────────────────────────────────────
def fig_heatmap(agg: pd.DataFrame) -> go.Figure:
    days  = [d for d in DAY_ORDER if d in agg["_weekday_name"].values]
    pivot = (agg.pivot(index="range_label", columns="_weekday_name", values="strike_rate")
               .reindex(columns=days).sort_index())
    p_mae = (agg.pivot(index="range_label", columns="_weekday_name", values="avg_mae")
               .reindex(columns=days).reindex(pivot.index))
    p_mfe = (agg.pivot(index="range_label", columns="_weekday_name", values="avg_mfe")
               .reindex(columns=days).reindex(pivot.index))
    p_n   = (agg.pivot(index="range_label", columns="_weekday_name", values="n")
               .reindex(columns=days).reindex(pivot.index))

    z      = pivot.values
    labels = pivot.index.tolist()
    nrows  = len(labels)

    # Strike rate text on each cell
    cell_text = [[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in z]

    # Rich hover
    hover = []
    for ri, lbl in enumerate(labels):
        row = []
        for di, d in enumerate(days):
            sr  = z[ri, di]
            if np.isnan(sr):
                row.append(f"<b>{d[:3]} · {lbl}</b><br>No data")
            else:
                clr = C["green"] if sr >= 50 else C["red"]
                ic  = "▲" if sr >= 50 else "▼"
                row.append(
                    f"<b>{d[:3]} · {lbl}</b><br>"
                    f"<span style='color:{clr}'>{ic} {sr:.1f}% TP hit</span><br>"
                    f"Trades: {int(p_n.values[ri,di])}<br>"
                    f"<span style='color:{C['red']}'>Avg MAE {p_mae.values[ri,di]:.3f}%</span><br>"
                    f"<span style='color:{C['green']}'>Avg MFE {p_mfe.values[ri,di]:.3f}%</span>"
                )
        hover.append(row)

    colorscale = [
        [0.00,"#2a0808"],[0.30,"#5a1010"],[0.45,"#8a2800"],
        [0.50,"#1a1a20"],[0.55,"#0a2a18"],[0.70,"#006030"],
        [1.00, C["green"]],
    ]
    cell_h   = max(18, min(32, 600 // max(nrows, 1)))
    fig_h    = min(900, cell_h * nrows + 130)
    txt_size = max(7, min(10, cell_h - 8))

    fig = go.Figure(go.Heatmap(
        z=z, x=days, y=labels,
        text=cell_text, customdata=hover,
        texttemplate="%{text}",
        textfont=dict(size=txt_size, color="rgba(242,242,244,0.88)"),
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale,
        zmid=50, zmin=0, zmax=100,
        xgap=2, ygap=1,
        colorbar=dict(
            title=dict(text="TP hit %", font=dict(color=C["t2"], size=10)),
            tickfont=dict(color=C["t2"], size=9),
            bgcolor=C["bg0"], bordercolor=C["border"],
            thickness=10, len=0.8,
            tickvals=[0,25,50,75,100],
            ticktext=["0%","25%","50%","75%","100%"],
        ),
    ))
    fig.update_layout(**_layout(
        height=fig_h,
        dragmode=False,                  # no pan/drag
        title=dict(text="TP Hit Rate — click a cell to see MAE/MFE breakdown",
                   font=dict(size=11, color=C["t1"]), x=0.01),
        xaxis=dict(side="top", fixedrange=True,
                   tickfont=dict(size=11, color=C["t1"]),
                   **{k:v for k,v in AXIS.items() if k!="tickfont"}),
        yaxis=dict(autorange="reversed", fixedrange=True,
                   tickfont=dict(size=9, color=C["t0"]),
                   **{k:v for k,v in AXIS.items() if k!="tickfont"}),
        margin=dict(l=90, r=20, t=60, b=20),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ANALYTICS FIGURES
# ──────────────────────────────────────────────────────────────────────────────

def fig_mae_mfe(trades: pd.DataFrame, label: str = "") -> go.Figure:
    fig = make_subplots(1, 2,
        subplot_titles=["MAE  (adverse excursion)", "MFE  (favorable excursion)"],
        horizontal_spacing=0.08)
    mv, fv = trades["mae_pct"].dropna(), trades["mfe_pct"].dropna()
    fig.add_trace(go.Histogram(x=mv, name="MAE", marker_color=C["red"],
        opacity=0.75, nbinsx=50,
        hovertemplate="MAE %{x:.3f}%<br>Count %{y}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Histogram(x=fv, name="MFE", marker_color=C["green"],
        opacity=0.75, nbinsx=50,
        hovertemplate="MFE %{x:.3f}%<br>Count %{y}<extra></extra>"), row=1, col=2)
    for p, a in [(25,"p25"),(50,"p50"),(75,"p75")]:
        if len(mv):
            fig.add_vline(x=float(np.percentile(mv,p)), row=1, col=1,
                line=dict(color=C["red"],dash="dash",width=1),
                annotation=dict(text=a, font=dict(color=C["red"],size=8), showarrow=False))
        if len(fv):
            fig.add_vline(x=float(np.percentile(fv,p)), row=1, col=2,
                line=dict(color=C["green"],dash="dash",width=1),
                annotation=dict(text=a, font=dict(color=C["green"],size=8), showarrow=False))
    ax = dict(gridcolor=C["border"],linecolor=C["border"],tickfont=dict(size=9,color=C["t2"]),zeroline=False)
    fig.update_layout(**_layout(
        title=dict(text=f"MAE / MFE Distributions{('  ·  '+label) if label else ''}",
                   font=dict(size=11,color=C["t1"]),x=0.01),
        height=295, showlegend=False,
        xaxis=ax, yaxis=ax, xaxis2=ax, yaxis2=ax,
        margin=dict(l=50,r=20,t=50,b=30),
    ))
    fig.update_annotations(font_size=9)
    return fig


def fig_scatter(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for oc, clr, op, lbl in [
        ("tp_hit", C["green"], .55, "TP hit"),
        ("sl_hit", C["red"],   .55, "SL hit"),
        ("timeout",C["yellow"],.35, "Timeout"),
    ]:
        sub = trades[trades["outcome"] == oc]
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["mae_pct"], y=sub["mfe_pct"], mode="markers", name=lbl,
            marker=dict(color=clr, size=3.5, opacity=op, line=dict(width=0)),
            hovertemplate=f"<b>{lbl}</b><br>MAE %{{x:.3f}}%<br>MFE %{{y:.3f}}%<extra></extra>",
        ))
    fig.update_layout(**_layout(
        title=dict(text="MAE vs MFE — each dot is one trade",
                   font=dict(size=11,color=C["t1"]),x=0.01),
        height=310,
        xaxis=dict(title="MAE %",**AXIS), yaxis=dict(title="MFE %",**AXIS),
        legend=dict(font=dict(size=9,color=C["t2"]),bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=55,r=20,t=50,b=40),
    ))
    return fig


def fig_regime(trades: pd.DataFrame, feature: str) -> go.Figure:
    """
    Bins trades by feature value (12 equal-width bins) and shows:
    - Top panel : avg MAE (red) and avg MFE (green) bars per bin
    - Bottom panel: MFE/MAE ratio line — tells you which regime is structurally cleanest
    """
    sub = trades.dropna(subset=[feature]).copy()
    if len(sub) < 10:
        return go.Figure()
    sub["_bin"] = pd.cut(sub[feature], bins=12)
    agg = (sub.groupby("_bin", observed=True)
             .agg(n=("tp_hit","count"), mae=("mae_pct","mean"), mfe=("mfe_pct","mean"))
             .reset_index())
    agg = agg[agg["n"] >= 3]
    agg["mid"]   = agg["_bin"].apply(lambda x: float(x.mid))
    agg["ratio"] = (agg["mfe"] / agg["mae"].replace(0, np.nan)).fillna(0)

    fig = make_subplots(2, 1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f"Avg MAE & MFE  by  {feature}", "MFE / MAE ratio  (>1 = favorable)"],
        row_heights=[0.65, 0.35])

    fig.add_trace(go.Bar(x=agg["mid"], y=agg["mae"], name="Avg MAE",
        marker_color=C["red"], opacity=0.8,
        customdata=agg["n"],
        hovertemplate=f"{feature}=%{{x:.2f}}<br>MAE %{{y:.3f}}%<br>n=%{{customdata}}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Bar(x=agg["mid"], y=agg["mfe"], name="Avg MFE",
        marker_color=C["green"], opacity=0.8,
        customdata=agg["n"],
        hovertemplate=f"{feature}=%{{x:.2f}}<br>MFE %{{y:.3f}}%<br>n=%{{customdata}}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=agg["mid"], y=agg["ratio"],
        mode="lines+markers", name="MFE/MAE",
        line=dict(color=C["blue"],width=2), marker=dict(size=5,color=C["blue"]),
        hovertemplate=f"{feature}=%{{x:.2f}}<br>ratio %{{y:.2f}}<extra></extra>"),
        row=2, col=1)
    fig.add_hline(y=1.0, row=2, col=1, line=dict(color=C["t3"],dash="dot",width=1))

    ax = dict(gridcolor=C["border"],linecolor=C["border"],tickfont=dict(size=9,color=C["t2"]),zeroline=False)
    fig.update_layout(**_layout(
        height=370, barmode="group",
        title=dict(text=f"Regime Impact — {feature}",font=dict(size=11,color=C["t1"]),x=0.01),
        legend=dict(font=dict(size=9,color=C["t2"]),bgcolor="rgba(0,0,0,0)"),
        xaxis=ax, yaxis=dict(title="%",**ax),
        xaxis2=dict(title=feature,**ax), yaxis2=dict(title="ratio",**ax),
        margin=dict(l=55,r=20,t=50,b=40),
    ))
    fig.update_annotations(font_size=9)
    return fig


def _pct_table(trades: pd.DataFrame) -> str:
    mv, fv = trades["mae_pct"].dropna(), trades["mfe_pct"].dropna()
    rows = ""
    for p in [10,25,50,75,90,95]:
        rows += (f"<tr>"
                 f"<td style='color:{C['t2']};padding:5px 14px'>p{p}</td>"
                 f"<td style='color:{C['red']};padding:5px 14px;text-align:right'>"
                 f"{np.percentile(mv,p):.4f}%" if len(mv) else "—"
                 f"</td>"
                 f"<td style='color:{C['green']};padding:5px 14px;text-align:right'>"
                 f"{np.percentile(fv,p):.4f}%" if len(fv) else "—"
                 f"</td></tr>")
    # rebuild correctly
    rows = ""
    for p in [10,25,50,75,90,95]:
        mv_ = f"{np.percentile(mv,p):.4f}%" if len(mv) else "—"
        fv_ = f"{np.percentile(fv,p):.4f}%" if len(fv) else "—"
        rows += (f"<tr>"
                 f"<td style='color:{C['t2']};padding:5px 14px'>p{p}</td>"
                 f"<td style='color:{C['red']};padding:5px 14px;text-align:right'>{mv_}</td>"
                 f"<td style='color:{C['green']};padding:5px 14px;text-align:right'>{fv_}</td>"
                 f"</tr>")
    return (f"<table style='width:100%;background:{C['bg2']};border:1px solid {C['border']};"
            f"border-radius:6px;border-collapse:collapse;font-family:Geist Mono,monospace;font-size:.78rem'>"
            f"<thead><tr style='border-bottom:1px solid {C['border']}'>"
            f"<th style='color:{C['t3']};padding:7px 14px;text-align:left;letter-spacing:.10em'>PCTILE</th>"
            f"<th style='color:{C['red']};padding:7px 14px;text-align:right'>MAE %</th>"
            f"<th style='color:{C['green']};padding:7px 14px;text-align:right'>MFE %</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>")


def kpi(label: str, value: str, variant: str = "") -> str:
    return (f"<div class='kpi-card'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value {variant}'>{value}</div>"
            f"</div>")


# ──────────────────────────────────────────────────────────────────────────────
# ANALYTICS PANEL
# ──────────────────────────────────────────────────────────────────────────────
def render_analytics(trades: pd.DataFrame, label: str = "") -> None:
    if trades.empty:
        st.warning("No trades for this selection.")
        return

    n      = len(trades)
    n_tp   = int(trades["tp_hit"].sum())
    n_sl   = int((trades["outcome"] == "sl_hit").sum())
    n_to   = int((trades["outcome"] == "timeout").sum())
    tp_r   = n_tp / n * 100 if n else 0
    avg_mae = float(trades["mae_pct"].mean())
    avg_mfe = float(trades["mfe_pct"].mean())
    ratio   = avg_mfe / avg_mae if avg_mae > 0 else 0.0

    cs = st.columns(7)
    for col, lbl, val, var in [
        (cs[0], "Trades",   f"{n:,}",          ""),
        (cs[1], "TP Hit",   f"{tp_r:.1f}%",    "green" if tp_r>=50 else "red"),
        (cs[2], "TP / SL",  f"{n_tp} / {n_sl}",""),
        (cs[3], "Timeout",  f"{n_to:,}",       "yellow"),
        (cs[4], "Avg MAE",  f"{avg_mae:.3f}%", "red"),
        (cs[5], "Avg MFE",  f"{avg_mfe:.3f}%", "green"),
        (cs[6], "MFE/MAE",  f"{ratio:.2f}",    "green" if ratio>=1 else "red"),
    ]:
        col.markdown(kpi(lbl, val, var), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _k = label.replace(" ","_").replace("·","").replace(":","").replace("–","_")
    feat_cols = [c for c in ["adx_14","atr_14","zscore_20"] if c in trades.columns]

    t1, t2, t3, t4 = st.tabs(["📊  MAE / MFE","🔵  Scatter","🔬  Regime Impact","🗂  Trades"])

    with t1:
        ca, cb = st.columns([3, 2])
        with ca:
            st.plotly_chart(fig_mae_mfe(trades, label), use_container_width=True, key=f"mf_{_k}")
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(_pct_table(trades), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if ratio >= 1.5:
                msg = f"MFE/MAE {ratio:.2f} — price moves further toward target than against on average"
                st.markdown(f'<span class="badge badge-green">{msg}</span>', unsafe_allow_html=True)
            elif ratio < 1.0:
                msg = f"MFE/MAE {ratio:.2f} — adverse moves outpace favorable on average"
                st.markdown(f'<span class="badge badge-red">{msg}</span>', unsafe_allow_html=True)

    with t2:
        st.plotly_chart(fig_scatter(trades), use_container_width=True, key=f"sc_{_k}")

    with t3:
        if feat_cols:
            sel = st.selectbox("Feature", feat_cols, key=f"feat_{_k}",
                               help="Shows how avg MAE/MFE and their ratio shift across regime bins")
            sub = trades.dropna(subset=[sel])
            if len(sub) < 10:
                st.info(f"Not enough trades with {sel} data to bin.")
            else:
                st.plotly_chart(fig_regime(sub, sel), use_container_width=True, key=f"rg_{_k}")
                st.markdown(
                    f'<div style="font-size:.70rem;color:{C["t2"]};margin-top:-6px">'
                    f'Bins with &lt;3 trades hidden. '
                    f'Blue line above dashed = regime where setup is structurally cleaner (MFE outpaces MAE).</div>',
                    unsafe_allow_html=True)
        else:
            st.info("No feature columns found (adx_14, atr_14, zscore_20). "
                    "Add them to your dataset to unlock regime analysis.")

    with t4:
        show = [c for c in [
            "range_label","_weekday_name","direction",
            "entry_price","sl_price","tp_price","exit_price",
            "outcome","mae_pct","mfe_pct","duration",
        ] if c in trades.columns]
        st.dataframe(
            trades[show].rename(columns={"_weekday_name":"weekday","range_label":"range"})
                        .round(5).head(2000),
            use_container_width=True, height=300)


# ──────────────────────────────────────────────────────────────────────────────
# HEATMAP + ANALYTICS FRAGMENT
# Only this section rerenders when a cell is clicked — main page stays still.
# ──────────────────────────────────────────────────────────────────────────────
@st.fragment
def heatmap_section(ft: pd.DataFrame, p: dict) -> None:

    if p["custom_range"]:
        st.markdown(
            f'<div class="section-label">Custom Range Analytics '
            f'<span class="badge badge-green" style="margin-left:8px">'
            f'{p["c_start"]} – {p["c_end"]}</span></div>',
            unsafe_allow_html=True)
        render_analytics(ft, f"{p['c_start']}–{p['c_end']}")
        return

    # badges
    n_ft   = len(ft)
    tp_r   = ft["tp_hit"].mean()*100 if n_ft else 0
    sr_cls = "badge-green" if tp_r >= 50 else "badge-red"
    rm_cls = "badge-yellow" if p["risk_mode"]=="atr" else "badge-dim"
    rm_lbl = "ATR Risk" if p["risk_mode"]=="atr" else "Fixed % Risk"
    st.markdown(
        f'<span class="badge badge-dim" style="margin-right:8px">{n_ft:,} trades</span>'
        f'<span class="badge {sr_cls}" style="margin-right:8px">{tp_r:.1f}% TP hit</span>'
        f'<span class="badge {rm_cls}">{rm_lbl}</span>',
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">TP Hit Rate Heatmap</div>', unsafe_allow_html=True)

    agg = aggregate_heatmap(ft)
    if agg.empty:
        st.warning("No data — check direction filter."); return

    event = st.plotly_chart(
        fig_heatmap(agg),
        use_container_width=True,
        on_select="rerun",        # fragment reruns on click, not full page
        key="main_heatmap",
        config={"displayModeBar": False, "scrollZoom": False},
    )

    # ── Robust click extraction ───────────────────────────────────────
    sel_day = sel_range = None
    try:
        pts = event.selection.points  # type: ignore[union-attr]
        if pts:
            sel_day   = pts[0].get("x")
            sel_range = pts[0].get("y")
    except Exception:
        pass

    if sel_day and sel_range:
        st.session_state["selected_cell"] = (sel_day, sel_range)

    cell = st.session_state.get("selected_cell")

    # clear button
    if cell:
        c1, c2 = st.columns([9, 1])
        with c1:
            d, rl = cell
            st.markdown(
                f'<div class="cell-banner">'
                f'<span class="badge badge-green">{d}</span>'
                f'<span style="color:{C["t0"]};font-weight:600">{rl}</span>'
                f'<span style="color:{C["t2"]}"> — click another cell to switch</span>'
                f'</div>', unsafe_allow_html=True)
        with c2:
            if st.button("✕", key="clr"):
                st.session_state["selected_cell"] = None
                st.rerun(scope="fragment")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Render analytics ──────────────────────────────────────────────
    if cell:
        d, rl = cell
        cell_trades = ft[(ft["_weekday_name"] == d) & (ft["range_label"] == rl)]
        st.markdown(
            f'<div class="section-label">Analytics — {d} · {rl}</div>',
            unsafe_allow_html=True)
        render_analytics(cell_trades, f"{d} · {rl}")
    else:
        st.markdown('<div class="section-label">Global Analytics</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:.73rem;color:{C["t2"]};margin-bottom:14px">'
            f'Click any cell to drill into that window. Showing aggregate below.</div>',
            unsafe_allow_html=True)
        render_analytics(ft, "All Filtered Trades")


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:18px 0 20px;border-bottom:1px solid {C['border']};margin-bottom:14px">
          <div style="font-size:.60rem;color:{C['t3']};letter-spacing:.14em;text-transform:uppercase;margin-bottom:5px">Strategy Engine</div>
          <div style="font-size:.95rem;font-weight:600;color:{C['t0']}">⚡ Fake Breakout</div>
          <div style="font-size:.62rem;color:{C['t3']};margin-top:3px">MAE · MFE · Regime Analyzer</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
        src = st.radio("_src", ["Upload File","Local Path"], label_visibility="collapsed")
        uploaded, local_path = None, ""
        if src == "Upload File":
            uploaded = st.file_uploader("_up", type=["csv","parquet"], label_visibility="collapsed")
        else:
            local_path = st.text_input("_lp", placeholder="/data/nq_1m.parquet",
                                       label_visibility="collapsed")
        years = st.slider("Years of data", 1, 10, 3)

        st.markdown('<div class="section-label">Range</div>', unsafe_allow_html=True)
        tf = st.selectbox("Range TF", ["1m","5m","15m","1h"], index=1,
                          help="Consolidation window. Breakout trigger = first 1-min bar that breaks out.")
        st.markdown(
            f'<div style="font-size:.62rem;color:{C["t3"]};margin:-4px 0 10px">'
            f'Trigger: <span style="color:{C["green"]}">1-min close</span></div>',
            unsafe_allow_html=True)
        custom = st.checkbox("Custom Window")
        cs = ce = ""
        if custom:
            ca, cb = st.columns(2)
            cs = ca.text_input("Start","09:30",key="cs")
            ce = cb.text_input("End",  "09:35",key="ce")

        st.markdown('<div class="section-label">Risk</div>', unsafe_allow_html=True)
        risk_lbl  = st.selectbox("Risk Mode", ["Fixed %","ATR-Based"])
        risk_mode = "atr" if risk_lbl=="ATR-Based" else "fixed_pct"
        sl_k = st.number_input(
            "SL mult (×ATR)" if risk_mode=="atr" else "Stop Loss %",
            0.05, 10.0, 1.5 if risk_mode=="atr" else 0.5, 0.05, format="%.2f")
        tp_lbl  = st.selectbox("TP Mode",
                               ["Opposite Side of Range","Fixed %","Realized Vol of Range"])
        tp_mode = {"Opposite Side of Range":"range","Fixed %":"fixed_pct",
                   "Realized Vol of Range":"vol"}[tp_lbl]
        tp_k = 1.0
        if tp_mode != "range":
            tp_k = st.number_input("TP mult" if tp_mode=="vol" else "TP %",
                                   0.05, 10.0, 1.0, 0.05, format="%.2f")

        st.markdown('<div class="section-label">Filter</div>', unsafe_allow_html=True)
        direction = st.multiselect("Direction", ["short","long"], default=["short","long"])

        st.markdown("---")
        run = st.button("⚡  Run Backtest", use_container_width=True)

        return dict(src=src, uploaded=uploaded, local_path=local_path, years=years,
                    tf=tf, custom_range=custom, c_start=cs, c_end=ce,
                    risk_mode=risk_mode, sl_k=sl_k, tp_mode=tp_mode, tp_k=tp_k,
                    direction=direction, run=run)


# ──────────────────────────────────────────────────────────────────────────────
# WELCOME
# ──────────────────────────────────────────────────────────────────────────────
def render_welcome() -> None:
    st.markdown(f"""
    <div style="margin:48px auto;max-width:640px;text-align:center">
      <div style="font-size:2.5rem;margin-bottom:12px">⚡</div>
      <div style="font-size:1.1rem;font-weight:600;color:{C['t0']};margin-bottom:10px">Fake Breakout Backtest Engine</div>
      <div style="font-size:.78rem;color:{C['t2']};line-height:1.9;margin-bottom:26px">
        Find repeatable time-of-day setups by measuring how far price moves toward<br>
        your target (MFE) versus against you (MAE) after a fake breakout.<br>
        No P&amp;L tracking. Just structure.
      </div>
      <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-bottom:32px">
        <span class="badge badge-green">1-min Breakout Trigger</span>
        <span class="badge badge-green">MAE · MFE · Regime Analysis</span>
        <span class="badge badge-dim">ATR / Vol Risk</span>
        <span class="badge badge-dim">Parquet / CSV · 200MB+</span>
      </div>
      <div style="background:{C['bg2']};border:1px solid {C['border']};border-radius:8px;padding:18px 22px;text-align:left">
        <div style="font-size:.58rem;color:{C['t3']};letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px">Expected Schema</div>
        <code style="font-size:.70rem;color:{C['t2']};line-height:1.9;font-family:'Geist Mono',monospace;white-space:pre"><span style="color:{C['t3']}">timestamp            open      high      low       close   volume  adx_14  atr_14  zscore_20</span>
<span style="color:{C['green']}">2022-01-03T09:29:00</span>  <span style="color:{C['t1']}">1828.72   1829.19   1828.72   1829.07  25      31.1    0.3     2.84</span></code>
      </div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    st.markdown(f"""
    <div class="top-header">
      <div class="pulse"></div>
      <div>
        <div class="title">Fake Breakout Backtest</div>
        <div class="sub">MAE · MFE · Regime Discovery — 1-min Trigger Engine</div>
      </div>
      <span class="badge badge-green" style="margin-left:auto">LIVE</span>
    </div>""", unsafe_allow_html=True)

    for k, v in [("trades",None),("df",None),("selected_cell",None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    p = render_sidebar()

    # load
    df_raw = None
    if p["src"] == "Upload File" and p["uploaded"]:
        f = p["uploaded"]
        with st.spinner(f"Loading {f.name} …"):
            try:
                rb = f.read()
                df_raw = pd.read_parquet(io.BytesIO(rb)) if f.name.endswith(".parquet") else _load_csv(rb)
            except Exception as e:
                st.error(f"Load error: {e}"); return
    elif p["src"] == "Local Path" and p["local_path"].strip():
        path = Path(p["local_path"].strip())
        if not path.exists():
            st.error(f"Not found: {path}"); return
        with st.spinner(f"Loading {path.name} …"):
            try:
                df_raw = _load_parquet(str(path)) if str(path).endswith(".parquet") else _load_csv(path.read_bytes())
            except Exception as e:
                st.error(f"Load error: {e}"); return

    if df_raw is None:
        render_welcome(); return

    with st.spinner("Preprocessing …"):
        try:
            df = preprocess(df_raw, p["years"])
        except Exception as e:
            st.error(f"Preprocessing error: {e}"); return
    st.session_state["df"] = df

    # summary
    mem   = df.memory_usage(deep=True).sum() / 1024**2
    ndays = df["_date"].nunique()
    cols  = st.columns(5)
    for col, lbl, val in [
        (cols[0],"Bars",f"{len(df):,}"), (cols[1],"Days",f"{ndays:,}"),
        (cols[2],"From",df.index[0].strftime("%Y-%m-%d")),
        (cols[3],"To",  df.index[-1].strftime("%Y-%m-%d")),
        (cols[4],"RAM", f"{mem:.1f} MB"),
    ]:
        col.markdown(kpi(lbl, val), unsafe_allow_html=True)

    feat_ok = [c for c in ["adx_14","atr_14","zscore_20"] if c in df.columns]
    if feat_ok:
        st.markdown(
            '<div style="margin-top:8px">'
            + "".join(f'<span class="badge badge-green" style="margin-right:6px">{f}</span>' for f in feat_ok)
            + '</div>', unsafe_allow_html=True)
    if p["risk_mode"]=="atr" and "atr_14" not in df.columns:
        st.warning("⚠️ ATR-Based risk selected but atr_14 not in dataset — falling back to fixed %.")

    st.markdown("<br>", unsafe_allow_html=True)

    # run
    if p["run"]:
        st.session_state["selected_cell"] = None
        tf_min = TF_MINUTES[p["tf"]]
        cs = p["c_start"] if p["custom_range"] else ""
        ce = p["c_end"]   if p["custom_range"] else ""
        with st.spinner("Detecting breakouts on 1-min bars …"):
            signals = compute_signals(df, tf_min, cs, ce)
        if signals.empty:
            st.warning("No signals found."); st.session_state["trades"] = pd.DataFrame(); return
        prog = st.progress(0, text=f"Simulating {len(signals):,} trades …")
        trades = simulate_trades(df, signals,
                                 risk_mode=p["risk_mode"], sl_k=p["sl_k"],
                                 tp_mode=p["tp_mode"],     tp_k=p["tp_k"])
        prog.progress(100, text="Done."); prog.empty()
        st.session_state["trades"] = trades

    if st.session_state["trades"] is None:
        st.info("Configure parameters and click **⚡ Run Backtest**."); return

    trades_all = st.session_state["trades"]
    if trades_all.empty:
        st.warning("No trades generated."); return

    ft = trades_all.copy()
    if p["direction"]:
        ft = ft[ft["direction"].isin(p["direction"])]
    if ft.empty:
        st.warning("No trades after direction filter."); return

    heatmap_section(ft, p)


if __name__ == "__main__":
    main()
