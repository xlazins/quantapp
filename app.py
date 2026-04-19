# ============================================================
#  MAE / MFE PIPELINE — Streamlit Application
#  Run:  streamlit run pipeline.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io, requests, warnings
from datetime import date, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── PAGE ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAE / MFE PIPELINE",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

*, *::before, *::after {
  font-family: 'Geist Mono', 'JetBrains Mono', ui-monospace,
               'Cascadia Code', 'Courier New', monospace !important;
  box-sizing: border-box;
}
html, body, .stApp { background:#000 !important; color:#e0e0e0 !important; }
.block-container { padding-top:1.2rem !important; max-width:1500px !important; }
#MainMenu, footer, header { visibility:hidden !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background:#070707 !important;
  border-right:1px solid #1d1d1d !important;
}
section[data-testid="stSidebar"] * { color:#e0e0e0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap:0 !important; background:#000 !important;
  border-bottom:1px solid #222 !important; overflow-x:auto !important;
}
.stTabs [data-baseweb="tab"] {
  background:#000 !important; color:#3a3a3a !important;
  border:1px solid #1a1a1a !important; border-bottom:none !important;
  padding:5px 13px !important; font-size:9.5px !important;
  letter-spacing:.12em !important; text-transform:uppercase !important;
  white-space:nowrap !important; margin-right:2px !important;
}
.stTabs [aria-selected="true"] {
  background:#e0e0e0 !important; color:#000 !important;
  border-color:#e0e0e0 !important;
}
.stTabs [data-baseweb="tab-panel"] { background:#000 !important; padding:0 !important; }

/* Buttons */
.stButton > button {
  background:#e0e0e0 !important; color:#000 !important; border:none !important;
  font-size:10px !important; font-weight:700 !important;
  letter-spacing:.15em !important; text-transform:uppercase !important;
  padding:7px 18px !important; border-radius:0 !important;
}
.stButton > button:hover { background:#bbb !important; }

/* Widgets */
.stSelectbox label, .stNumberInput label, .stSlider label,
.stDateInput label, .stTextInput label, .stRadio label,
.stCheckbox label { font-size:9.5px !important; color:#444 !important;
  letter-spacing:.1em !important; text-transform:uppercase !important; }

/* Progress */
.stProgress > div > div { background:#e0e0e0 !important; }

/* Metrics */
[data-testid="stMetric"] {
  background:#080808 !important; border:1px solid #1d1d1d !important; padding:12px !important;
}
[data-testid="stMetricLabel"] { font-size:9px !important; color:#444 !important; }
[data-testid="stMetricValue"] { font-size:18px !important; color:#e0e0e0 !important; }
[data-testid="stMetricDelta"] { font-size:10px !important; }

/* Tables */
table { font-size:10.5px !important; width:100% !important; border-collapse:collapse !important; }
th { background:#0a0a0a !important; color:#444 !important;
     border:1px solid #1d1d1d !important; padding:7px !important;
     font-size:9px !important; letter-spacing:.1em !important; text-transform:uppercase !important; }
td { background:#000 !important; color:#e0e0e0 !important;
     border:1px solid #111 !important; padding:5px 8px !important; }

/* Alerts */
.stAlert { background:#0a0a0a !important; border:1px solid #1d1d1d !important;
           color:#888 !important; border-radius:0 !important; font-size:10.5px !important; }
hr { border-color:#1a1a1a !important; margin:10px 0 !important; }

/* Custom */
.sec { font-size:9px; color:#3d3d3d; letter-spacing:.2em; text-transform:uppercase;
       padding-bottom:7px; border-bottom:1px solid #1a1a1a; margin-top:20px; margin-bottom:14px; }
.kpi { border:1px solid #1d1d1d; background:#060606; padding:14px; }
.kpi-lbl { font-size:9px; color:#444; letter-spacing:.15em; text-transform:uppercase; margin-bottom:5px; }
.kpi-val { font-size:20px; color:#e0e0e0; font-weight:700; }
.kpi-sub { font-size:9px; color:#3d3d3d; margin-top:4px; }
.step-box { border:1px solid #1a1a1a; padding:11px 14px; margin-bottom:5px; background:#050505; }
.step-num { font-size:9px; color:#333; letter-spacing:.18em; margin-bottom:3px; }
.step-ttl { font-size:12.5px; color:#e0e0e0; margin-bottom:3px; }
.step-dsc { font-size:9.5px; color:#4d4d4d; }
.pass { display:inline-block; background:#e0e0e0; color:#000; padding:1px 8px;
        font-size:9px; font-weight:700; letter-spacing:.1em; }
.fail { display:inline-block; background:#1d1d1d; color:#555; padding:1px 8px;
        font-size:9px; letter-spacing:.1em; }
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB STYLE ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#000",  "axes.facecolor":"#060606",
    "axes.edgecolor":"#1d1d1d", "axes.labelcolor":"#555",
    "text.color":"#e0e0e0",     "xtick.color":"#444",
    "ytick.color":"#444",       "grid.color":"#111",
    "grid.linestyle":"-",       "grid.linewidth":.5,
    "font.family":"monospace",  "font.size":8.5,
    "axes.titlesize":9.5,       "axes.titlecolor":"#aaa",
    "legend.facecolor":"#060606","legend.edgecolor":"#1d1d1d",
    "legend.fontsize":8,        "figure.dpi":110,
    "savefig.facecolor":"#000", "lines.linewidth":1.2,
})


# ════════════════════════════════════════════════════════════
#  DATA LAYER
# ════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def fetch_api(years: tuple, path: str, tf: str) -> pd.DataFrame:
    base = "https://data-api.londonstrategicedge.com/download/candles"
    dfs = []
    for yr in years:
        url = f"{base}/{path}/{tf}/{yr}.csv.gz"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                dfs.append(pd.read_csv(io.BytesIO(r.content), compression="gzip"))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

@st.cache_data(show_spinner=False)
def parse_upload(raw: bytes, name: str) -> pd.DataFrame:
    kw = {"compression":"gzip"} if name.endswith(".gz") else {}
    return pd.read_csv(io.BytesIO(raw), **kw)

def standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    ts = next((c for c in df.columns if "time" in c or "date" in c), df.columns[0])
    df = df.rename(columns={ts:"ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    return df[["ts","open","high","low","close","volume"]].dropna(subset=["open"])

def date_slice(df: pd.DataFrame, s: date, e: date) -> pd.DataFrame:
    m = (df["ts"].dt.date >= s) & (df["ts"].dt.date <= e)
    return df[m].reset_index(drop=True)


# ════════════════════════════════════════════════════════════
#  PIPELINE COMPUTATIONS
# ════════════════════════════════════════════════════════════

def compute_box(df: pd.DataFrame, hr: int, mins: int) -> pd.DataFrame:
    """High/low of first `mins` minutes of hour `hr` UTC per calendar date."""
    d = df.copy()
    d["_dt"] = d["ts"].dt.date
    d["_h"]  = d["ts"].dt.hour
    d["_m"]  = d["ts"].dt.minute
    sub = d[(d["_h"] == hr) & (d["_m"] < mins)]
    g = sub.groupby("_dt").agg(
        box_high=("high","max"), box_low=("low","min"),
        box_open=("open","first"),
    ).reset_index().rename(columns={"_dt":"date"})
    g["box_range"] = (g["box_high"] - g["box_low"]) * 10_000   # pips
    return g.reset_index(drop=True)

def compute_measurements(df: pd.DataFrame, box: pd.DataFrame,
                          hr: int, mins: int, direction: str,
                          holds: list) -> pd.DataFrame:
    """
    For each day and hold period, measure MAE and MFE (pips)
    from box_high (short) and/or box_low (long).
    """
    idx = df.set_index("ts").sort_index()
    pip = 0.0001
    rows = []

    for _, row in box.iterrows():
        d = row["date"]
        bh, bl = row["box_high"], row["box_low"]
        box_end = pd.Timestamp(str(d), tz="UTC") + pd.Timedelta(hours=hr, minutes=mins)

        # daily range for regime labelling
        day_mask = idx.index.date == d
        dr = (idx.loc[day_mask,"high"].max() - idx.loc[day_mask,"low"].min()) * 10_000 \
             if day_mask.any() else np.nan

        for hold in holds:
            hold_end = box_end + pd.Timedelta(hours=hold)
            try:
                w = idx.loc[box_end:hold_end]
            except Exception:
                continue
            if len(w) < 2:
                continue
            wh, wl = w["high"].max(), w["low"].min()

            dirs = ["short","long"] if direction == "both" else [direction]
            for dr_dir in dirs:
                if dr_dir == "short":
                    mae = max(wh - bh, 0) / pip
                    mfe = max(bh - wl, 0) / pip
                else:
                    mae = max(bl - wl, 0) / pip
                    mfe = max(wh - bl, 0) / pip
                rows.append({"date":d,"direction":dr_dir,"hold":hold,
                              "mae":mae,"mfe":mfe,
                              "box_range":row["box_range"],"daily_range":dr})
    return pd.DataFrame(rows)

def add_regime(meas: pd.DataFrame, thresh: float = 1.5) -> pd.DataFrame:
    meas = meas.copy()
    dr = (meas[["date","daily_range"]].drop_duplicates("date")
          .set_index("date")["daily_range"].sort_index())
    roll = dr.rolling(10, min_periods=3).median()
    ratio = dr / roll
    meas["range_ratio"] = meas["date"].map(ratio)
    meas["regime"] = meas["range_ratio"].apply(
        lambda x: "volatile" if pd.notna(x) and x >= thresh else "calm")
    return meas

def pct_table(s: pd.Series, ps=(10,25,50,75,90,95)) -> dict:
    s = s.dropna()
    return {f"P{p}": float(np.percentile(s, p)) for p in ps} if len(s) > 0 else {}

def compute_rho(meas: pd.DataFrame, direction: str, dates: list) -> dict:
    m = meas[(meas["direction"] == direction) & (meas["date"].isin(dates))]
    holds = sorted(m["hold"].unique())
    mae_m = [m[m["hold"]==h]["mae"].median() for h in holds]
    mfe_m = [m[m["hold"]==h]["mfe"].median() for h in holds]
    if len(mae_m) < 2 or mae_m[0] == 0 or mfe_m[0] == 0:
        return {}
    mg = mae_m[-1] / mae_m[0]
    fg = mfe_m[-1] / mfe_m[0] if mfe_m[0] > 0 else 1
    rho = mg / fg if fg > 0 else 1
    return {"rho":rho,"mae_growth":mg,"mfe_growth":fg,
            "mae_medians":mae_m,"mfe_medians":mfe_m,"holds":holds,
            "edge":"WIN-RATE" if rho > 1 else "MOMENTUM"}

def simulate(meas: pd.DataFrame, stop: float, target: float, queen: float,
             spread: float) -> pd.DataFrame:
    """Convert MAE/MFE measurements into simulated trade P&L."""
    rows = []
    for _, r in meas.iterrows():
        mae, mfe = r["mae"], r["mfe"]
        stopped = mae >= stop
        hit_tgt = mfe >= target
        hit_q   = mfe >= queen

        if stopped and hit_tgt:
            # Conservative: stop first if MAE ratio > MFE ratio
            pnl    = -stop if (mae/stop) > (mfe/target) else target
            outcome = "loss" if pnl < 0 else "win"
        elif stopped:
            pnl, outcome = -stop, "loss"
        elif hit_tgt:
            pnl, outcome = target, "win"
        else:
            # Held to close: use remaining MFE minus proportional adverse
            pnl = mfe * 0.35 - mae * 0.65
            outcome = "scratch"

        pnl -= spread   # cost per trade (spread)
        rows.append({"date":r["date"],"direction":r["direction"],
                     "hold":r["hold"],"regime":r.get("regime","all"),
                     "mae":mae,"mfe":mfe,"pnl":pnl,"outcome":outcome})
    return pd.DataFrame(rows)

def grade(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {}
    wins   = trades[trades["outcome"] == "win"]
    losses = trades[trades["outcome"] == "loss"]
    wr     = len(wins) / n
    lr     = len(losses) / n
    aw     = float(wins["pnl"].mean())   if len(wins)   > 0 else 0
    al     = float(losses["pnl"].mean()) if len(losses) > 0 else 0

    ev = wr * aw + lr * al
    tw = wins["pnl"].sum()   if len(wins)   > 0 else 0
    tl = losses["pnl"].sum() if len(losses) > 0 else 0
    pf = tw / abs(tl) if tl != 0 else (np.inf if tw > 0 else 0)

    streak = int(np.log(n) / np.log(1/lr)) if 0 < lr < 1 else n
    dd     = streak * abs(al)

    pnls   = trades["pnl"]
    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0
    sqn    = float(pnls.mean() / pnls.std() * np.sqrt(n))   if pnls.std() > 0 else 0
    ror    = float((lr / wr) ** (abs(al) / max(aw, 0.01)))  if wr > 0 and lr > 0 and aw > 0 else 1.0
    ror    = min(max(ror, 0), 1)
    kelly  = max(0, min(ev / aw if aw > 0 else 0, 1))

    return {"n":n,"win_rate":wr,"loss_rate":lr,
            "avg_win":aw,"avg_loss":abs(al),
            "ev":ev,"profit_factor":pf,
            "max_streak":streak,"max_dd":dd,
            "sharpe":sharpe,"sqn":sqn,
            "risk_of_ruin":ror,"kelly":kelly}

def run_gates(meas: pd.DataFrame, trades: pd.DataFrame, g: dict,
              n_combos: int, stop: float, target: float, queen: float,
              spread: float) -> dict:
    gates = {}

    # G0 — Correctness
    g0 = g.get("n",0) >= 100 and g.get("profit_factor",0) > 1.0
    gates["G0"] = {"pass":g0,"critical":True,
                   "label":"GATE 0  —  CORRECTNESS",
                   "detail":f"N={g.get('n',0)} (need 100+)  |  PF={g.get('profit_factor',0):.2f} (need >1.0)"}

    # G2 — Perturbation (5-pip shift on stop and target)
    pv = []
    for ds, dt in [(-5,0),(5,0),(0,-5),(0,5)]:
        t2 = simulate(meas, max(stop+ds, 1), max(target+dt, 1), queen, spread)
        pv.append(grade(t2).get("profit_factor",0) > 1.0)
    gates["G2"] = {"pass":sum(pv) >= 3,"critical":False,
                   "label":"GATE 2  —  PERTURBATION",
                   "detail":f"{sum(pv)}/4 perturbed variants profitable"}

    # G4 — Deflated Sharpe
    sh = g.get("sharpe",0)
    n  = max(g.get("n",1), 1)
    penalty = np.sqrt(2 * np.log(max(n_combos,1))) / np.sqrt(n)
    dsh = sh - penalty
    gates["G4"] = {"pass":dsh > 0 and sh > 0.3,"critical":True,
                   "label":"GATE 4  —  STATISTICAL SIGNIFICANCE",
                   "detail":f"Raw Sharpe={sh:.3f}  |  Deflated Sharpe={dsh:.3f}  |  n_combos={n_combos}"}

    # G5 — Cost survival
    gates["G5"] = {"pass":g.get("ev",0) > spread * 1.2,"critical":True,
                   "label":"GATE 5  —  COST SURVIVAL",
                   "detail":f"EV={g.get('ev',0):.2f} pips vs spread={spread:.1f} pips"}

    # G7 — Regime-aware walk-forward
    N  = len(trades)
    sp = int(N * 0.7)
    tr = trades.iloc[:sp-5]
    te = trades.iloc[sp+5:]
    sub = []
    for rg in ["calm","volatile"]:
        tr_r = tr[tr["regime"] == rg]
        te_r = te[te["regime"] == rg]
        if len(tr_r) >= 20 and len(te_r) >= 10:
            sub.append(grade(te_r).get("profit_factor",0) > 1.0)
    if len(te) >= 20:
        sub.append(grade(te).get("profit_factor",0) > 1.0)
    g7 = (sum(sub) >= max(1, round(len(sub)*0.6))) if sub else False
    gates["G7"] = {"pass":g7,"critical":False,
                   "label":"GATE 7  —  REGIME WALK-FORWARD",
                   "detail":f"{sum(sub)}/{len(sub)} walk-forward sub-tests passed"}

    # G8 — Monte Carlo block bootstrap
    pnls = trades["pnl"].values
    bs = max(5, int(np.sqrt(len(pnls))))
    mc_w = 0
    for _ in range(2000):
        blocks, idx = [], 0
        while idx < len(pnls):
            s = np.random.randint(0, max(1, len(pnls)-bs))
            blocks.extend(pnls[s:s+bs].tolist()); idx += bs
        if np.sum(blocks[:len(pnls)]) > 0:
            mc_w += 1
    mc = mc_w / 2000
    gates["G8"] = {"pass":mc >= 0.60,"critical":False,
                   "label":"GATE 8  —  MONTE CARLO",
                   "detail":f"{mc:.1%} of 2,000 shuffled sequences profitable (need 60%+)"}

    return gates


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("#### PIPELINE CONFIG")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sec">Data Source</div>', unsafe_allow_html=True)
    source = st.radio("", ["API — London Strategic Edge","Upload CSV / GZ"],
                      label_visibility="collapsed")

    if "API" in source:
        st.markdown('<div class="sec">API Settings</div>', unsafe_allow_html=True)
        asset   = st.selectbox("Asset Class", ["forex","crypto","indices","stocks","etfs"])
        sym     = st.text_input("Symbol", "eur_usd", help="e.g. eur_usd  btc_usd  aapl")
        tf      = st.selectbox("Timeframe", ["1m","5m","15m","1h"])
        c1,c2   = st.columns(2)
        yr_s    = c1.number_input("From",2019,2025,2020,1)
        yr_e    = c2.number_input("To",  2019,2025,2024,1)
        sym_path = f"{asset}/{sym}"
        years    = tuple(range(int(yr_s), int(yr_e)+1))
        upload   = None
    else:
        upload   = st.file_uploader("File", type=["csv","gz"])
        tf       = "1m"; sym_path = "uploaded"; years = ()

    st.markdown('<div class="sec">Fixed Constant</div>', unsafe_allow_html=True)
    sess_hr   = st.slider("Session Hour (UTC)", 0, 23, 8)
    box_mins  = st.slider("Box Duration (min)",  1, 60,  5)
    direction = st.selectbox("Direction", ["both","short","long"])

    st.markdown('<div class="sec">Forward Test</div>', unsafe_allow_html=True)
    use_fwd  = st.checkbox("Enable Forward Test", True)
    fwd_cut  = st.date_input("Cut-off Date", value=date(2024,7,1)) if use_fwd else None

    st.markdown('<div class="sec">Risk</div>', unsafe_allow_html=True)
    spread   = st.number_input("Spread (pips)", 0.1, 10.0, 1.2, 0.1)

    st.markdown("<hr>", unsafe_allow_html=True)
    load_btn = st.button("LOAD DATA")

# ── SESSION STATE ────────────────────────────────────────────
for k in ["df","box","meas","derive_res","best","val_res"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── LOAD ─────────────────────────────────────────────────────
if load_btn:
    with st.spinner("Loading…"):
        if "API" in source:
            raw = fetch_api(years, sym_path, tf)
        elif upload is not None:
            raw = parse_upload(upload.read(), upload.name)
        else:
            raw = pd.DataFrame()

    if raw.empty:
        st.sidebar.error("No data loaded.")
    else:
        st.session_state["df"] = standardise(raw)
        for k in ["box","meas","derive_res","best","val_res"]:
            st.session_state[k] = None
        st.sidebar.success(f"Loaded {len(st.session_state['df']):,} rows")

# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='border-bottom:1px solid #1a1a1a;padding-bottom:14px;margin-bottom:6px;'>
  <span style='font-size:21px;font-weight:700;letter-spacing:.04em;'>MAE / MFE PIPELINE</span>
  <span style='font-size:9.5px;color:#3a3a3a;letter-spacing:.15em;margin-left:18px;'>
    FIXED CONSTANT  ·  RAW MEASUREMENT  ·  MARKOV  ·  REGIME  ·  DERIVE  ·  VALIDATE  ·  FORWARD
  </span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
tabs = st.tabs([
    "OVERVIEW",
    "DATA",
    "STEPS 3-4  MEASUREMENT",
    "STEP 5  MARKOV",
    "STEP 6  REGIME",
    "STEP 7  DERIVE",
    "STEP 8  VALIDATE",
    "FORWARD TEST",
])


# ─────────────────────────────────────────────────────────────
#  TAB 0 — OVERVIEW
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="sec">The Eight-Step Pipeline</div>', unsafe_allow_html=True)
    for num, title, desc in [
        ("01","FIXED CONSTANT",
         "A deterministic price level. Same data in, same number out. Known before decisions are made."),
        ("02","RAW MEASUREMENT",
         "For every instance of the level, measure MAE and MFE across 1h, 2h, 3h holds. No filters."),
        ("03","MARKOV CHAIN",
         "Rho = MAE growth / MFE growth. Below 1 = momentum. Above 1 = win-rate. Drives exit strategy."),
        ("04","REGIME SWITCHING",
         "Split 1,250 measurements by volatility regime. Same level, different edge in calm vs volatile."),
        ("05","DERIVE",
         "Test 300+ stop/target/queen combinations. Grade each with ten metrics per regime."),
        ("06","VALIDATE",
         "Seven-gate stress test. Perturbation, deflated Sharpe, cost survival, walk-forward, Monte Carlo."),
        ("07","FORWARD TEST",
         "Apply validated parameters to held-out recent data before a single dollar of capital is at risk."),
    ]:
        st.markdown(f"""
        <div class="step-box">
          <div class="step-num">STEP {num}</div>
          <div class="step-ttl">{title}</div>
          <div class="step-dsc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Core Concepts</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="kpi"><div class="kpi-lbl">MAE</div>
          <div class="kpi-val">Adverse</div>
          <div class="kpi-sub">Maximum adverse excursion from the fixed constant. How far price moves against you before the trade resolves.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="kpi"><div class="kpi-lbl">MFE</div>
          <div class="kpi-val">Favorable</div>
          <div class="kpi-sub">Maximum favorable excursion. How far price moves in your direction within the hold period.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="kpi"><div class="kpi-lbl">RHO</div>
          <div class="kpi-val">Edge Type</div>
          <div class="kpi-sub">MAE growth / MFE growth across hold periods. Determines whether you trade momentum or win-rate.</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TAB 1 — DATA
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    df = st.session_state.get("df")
    if df is None:
        st.info("Configure the sidebar and click LOAD DATA.")
    else:
        max_date = df["ts"].dt.date.max()
        min_date = df["ts"].dt.date.min()
        train_end = (fwd_cut - timedelta(days=1)) if fwd_cut else max_date
        train_df  = date_slice(df, min_date, train_end)
        fwd_df    = date_slice(df, fwd_cut, max_date) if fwd_cut else pd.DataFrame()

        st.markdown('<div class="sec">Dataset Summary</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Train Rows", f"{len(train_df):,}")
        c3.metric("Forward Rows", f"{len(fwd_df):,}")
        c4.metric("Date Span", f"{min_date}  to  {max_date}")

        st.markdown('<div class="sec">Sample Rows</div>', unsafe_allow_html=True)
        st.dataframe(df.head(60), use_container_width=True, height=280)

        st.markdown('<div class="sec">Fixed Constant Preview</div>', unsafe_allow_html=True)
        if st.button("COMPUTE BOX LEVELS"):
            with st.spinner("Computing…"):
                b = compute_box(train_df, sess_hr, box_mins)
                st.session_state["box"] = b

        box = st.session_state.get("box")
        if box is not None:
            c1,c2,c3 = st.columns(3)
            c1.metric("Trading Days",  f"{len(box):,}")
            c2.metric("Avg Box Range", f"{box['box_range'].mean():.1f} pips")
            c3.metric("Max Box Range", f"{box['box_range'].max():.1f} pips")

            # Box range distribution
            fig, ax = plt.subplots(figsize=(10,3.5))
            ax.hist(box["box_range"].clip(upper=box["box_range"].quantile(0.98)),
                    bins=60, color="#e0e0e0", edgecolor="none", alpha=0.9)
            ax.axvline(box["box_range"].median(), color="#666", linestyle="--",
                       linewidth=1, label=f"Median {box['box_range'].median():.1f}")
            ax.set_xlabel("Box Range (pips)")
            ax.set_title(f"Fixed Constant — Box Range Distribution  |  {sess_hr:02d}:{box_mins:02d} UTC")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.dataframe(box.tail(30), use_container_width=True, height=250)


# ─────────────────────────────────────────────────────────────
#  TAB 2 — MEASUREMENT
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    df  = st.session_state.get("df")
    box = st.session_state.get("box")

    if df is None:
        st.info("Load data first.")
    elif box is None:
        st.info("Compute box levels in the DATA tab.")
    else:
        train_end = (fwd_cut - timedelta(days=1)) if fwd_cut else df["ts"].dt.date.max()
        train_df  = date_slice(df, df["ts"].dt.date.min(), train_end)

        if st.button("RUN MEASUREMENT"):
            prog = st.progress(0, "Measuring MAE / MFE…")
            meas = compute_measurements(train_df, box, sess_hr, box_mins,
                                         direction, [1,2,3])
            prog.progress(70, "Labelling regimes…")
            meas = add_regime(meas)
            st.session_state["meas"] = meas
            prog.progress(100, "Done.")
            prog.empty()

        meas = st.session_state.get("meas")
        if meas is None:
            st.info("Click RUN MEASUREMENT.")
        elif len(meas) == 0:
            st.warning("No measurements found. Check session hour and box duration settings.")
        else:
            dirs = sorted(meas["direction"].unique())
            d_sel = st.selectbox("Direction", dirs)
            m = meas[meas["direction"] == d_sel]
            all_dates = sorted(m["date"].unique())
            n = len(all_dates)

            win = {
                "5Y": all_dates,
                "1Y": all_dates[-252:] if n >= 252 else all_dates,
                "90D": all_dates[-63:]  if n >= 63  else all_dates,
            }

            # ── Percentile tables ──
            st.markdown('<div class="sec">Percentile Tables — MAE and MFE (pips)</div>',
                        unsafe_allow_html=True)
            for hold in [1,2,3]:
                st.markdown(f"**{hold}H Hold**")
                pct_rows = []
                for wname, wdates in win.items():
                    hw = m[(m["hold"]==hold) & (m["date"].isin(wdates))]
                    if len(hw) < 10: continue
                    row = {"Window":wname, "N":len(hw)}
                    for metric in ["mae","mfe"]:
                        p = pct_table(hw[metric])
                        for k,v in p.items():
                            row[f"{metric.upper()} {k}"] = f"{v:.1f}"
                    pct_rows.append(row)
                if pct_rows:
                    st.dataframe(pd.DataFrame(pct_rows).set_index("Window"),
                                 use_container_width=True)

            # ── Distributions ──
            st.markdown('<div class="sec">Distributions</div>', unsafe_allow_html=True)
            h_sel = st.select_slider("Hold Period", [1,2,3], value=1)
            hd = m[m["hold"] == h_sel]
            if len(hd) > 5:
                fig, axes = plt.subplots(1,2,figsize=(13,4))
                for ax, col, lbl in zip(axes, ["mae","mfe"], ["MAE","MFE"]):
                    data = hd[col].dropna()
                    clip = np.percentile(data, 97)
                    ax.hist(data[data <= clip*1.2], bins=55, color="#e0e0e0",
                            edgecolor="none", alpha=0.88)
                    for p,ls in [(50,"--"),(90,":")]:
                        v = np.percentile(data,p)
                        ax.axvline(v, color="#666", linewidth=1, linestyle=ls,
                                   label=f"P{p} = {v:.1f}")
                    ax.set_title(f"{lbl}  —  {h_sel}H  —  {d_sel.upper()}")
                    ax.set_xlabel("pips"); ax.legend(); ax.grid(True,alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            # ── Box range vs MAE scatter ──
            st.markdown('<div class="sec">Box Range vs 1H MAE</div>', unsafe_allow_html=True)
            h1 = m[m["hold"]==1]
            if len(h1) > 10:
                fig2, ax2 = plt.subplots(figsize=(10,4))
                ax2.scatter(h1["box_range"], h1["mae"], s=7, alpha=0.25, color="#e0e0e0")
                v = h1[["box_range","mae"]].dropna()
                if len(v) > 20:
                    z = np.polyfit(v["box_range"], v["mae"], 1)
                    xr = np.linspace(v["box_range"].quantile(.02), v["box_range"].quantile(.95), 100)
                    ax2.plot(xr, np.poly1d(z)(xr), color="#888", linewidth=1.5, label="OLS trend")
                ax2.set_xlabel("Box Range (pips)"); ax2.set_ylabel("MAE (pips)")
                ax2.set_title("Box Range as MAE Predictor")
                ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)


# ─────────────────────────────────────────────────────────────
#  TAB 3 — MARKOV
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    meas = st.session_state.get("meas")
    if meas is None:
        st.info("Complete the Measurement step.")
    else:
        st.markdown('<div class="sec">Rho Classification Across Lookback Windows</div>',
                    unsafe_allow_html=True)
        dirs = sorted(meas["direction"].unique())

        for dr in dirs:
            m = meas[meas["direction"] == dr]
            all_dates = sorted(m["date"].unique())
            n = len(all_dates)
            win = {
                "Five Year":  all_dates,
                "One Year":   all_dates[-252:] if n >= 252 else all_dates,
                "Ninety Day": all_dates[-63:]  if n >= 63  else all_dates,
            }
            st.markdown(f"**Direction: {dr.upper()}**")
            rho_rows = []
            rho_vals  = {}
            for wname, wdates in win.items():
                r = compute_rho(meas, dr, wdates)
                if not r: continue
                rho_rows.append({
                    "Window": wname, "N": len([d for d in all_dates if d in wdates]),
                    "MAE P50 1H": f"{r['mae_medians'][0]:.1f}",
                    "MAE P50 3H": f"{r['mae_medians'][-1]:.1f}",
                    "MFE P50 1H": f"{r['mfe_medians'][0]:.1f}",
                    "MFE P50 3H": f"{r['mfe_medians'][-1]:.1f}",
                    "MAE Growth": f"{r['mae_growth']:.2f}x",
                    "MFE Growth": f"{r['mfe_growth']:.2f}x",
                    "Rho": f"{r['rho']:.3f}",
                    "Edge": r["edge"],
                })
                rho_vals[wname] = r

            if rho_rows:
                st.dataframe(pd.DataFrame(rho_rows).set_index("Window"),
                             use_container_width=True)

            # Growth chart
            if rho_vals:
                fig, ax = plt.subplots(figsize=(10,3.8))
                colors = {"Five Year":"#e0e0e0","One Year":"#777","Ninety Day":"#333"}
                for wname, r in rho_vals.items():
                    hs = r["holds"]
                    ax.plot(hs, r["mae_medians"], color=colors[wname], linestyle="-",
                            linewidth=1.4, label=f"{wname} MAE")
                    ax.plot(hs, r["mfe_medians"], color=colors[wname], linestyle="--",
                            linewidth=1.4, label=f"{wname} MFE")
                ax.axhline(0, color="#333", linewidth=.5)
                ax.set_xticks([1,2,3])
                ax.set_xlabel("Hold Period (hours)"); ax.set_ylabel("Median (pips)")
                ax.set_title(f"MAE / MFE Growth  —  {dr.upper()}")
                ax.legend(loc="upper left", fontsize=7); ax.grid(True, alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                # 1Y rho callout
                r1y = rho_vals.get("One Year", list(rho_vals.values())[-1])
                st.markdown(f"""
                <div class="kpi" style="margin-top:14px;max-width:340px;">
                  <div class="kpi-lbl">One-Year Rho  —  {dr.upper()}</div>
                  <div class="kpi-val">{r1y['rho']:.3f}</div>
                  <div class="kpi-sub">{r1y['edge']}  &mdash;  {"hold to close, no target" if r1y['edge']=="WIN-RATE" else "use take-profit target"}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TAB 4 — REGIME
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    meas = st.session_state.get("meas")
    if meas is None:
        st.info("Complete the Measurement step.")
    else:
        st.markdown('<div class="sec">Regime Distribution</div>', unsafe_allow_html=True)
        rc = meas.drop_duplicates("date")["regime"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Calm Days",    rc.get("calm", 0))
        c2.metric("Volatile Days", rc.get("volatile", 0))
        c3.metric("Volatile %",
                  f"{rc.get('volatile',0)/(rc.sum() or 1)*100:.1f}%")

        dr_opts = sorted(meas["direction"].unique())
        dr = st.selectbox("Direction", dr_opts, key="rg_dir")
        m  = meas[meas["direction"] == dr]
        all_dates = sorted(m["date"].unique())
        n = len(all_dates)
        oney = all_dates[-252:] if n >= 252 else all_dates
        m1y  = m[m["date"].isin(oney)]

        st.markdown('<div class="sec">Regime-Split Metrics — 1Y Window</div>',
                    unsafe_allow_html=True)
        reg_rows = []
        for rg in ["calm","volatile"]:
            for hold in [1,2,3]:
                hw = m1y[(m1y["regime"]==rg) & (m1y["hold"]==hold)]
                if len(hw) < 10: continue
                # compute rho vs 1h baseline
                h1 = m1y[(m1y["regime"]==rg) & (m1y["hold"]==1)]
                if hold > 1 and len(h1) > 0 and h1["mae"].median() > 0:
                    rho = (hw["mae"].median() / h1["mae"].median()) / \
                          (hw["mfe"].median() / max(h1["mfe"].median(), 0.001))
                    rho_s = f"{rho:.3f}"
                else:
                    rho_s = "—"
                reg_rows.append({
                    "Regime": rg.upper(), "Hold": f"{hold}H", "N": len(hw),
                    "MAE P50": f"{hw['mae'].median():.1f}",
                    "MAE P90": f"{np.percentile(hw['mae'],90):.1f}",
                    "MFE P50": f"{hw['mfe'].median():.1f}",
                    "MFE P90": f"{np.percentile(hw['mfe'],90):.1f}",
                    "Rho*":     rho_s,
                })
        if reg_rows:
            st.dataframe(pd.DataFrame(reg_rows), use_container_width=True)

        # Distribution comparison
        st.markdown('<div class="sec">MAE / MFE Distribution by Regime — 1H Hold</div>',
                    unsafe_allow_html=True)
        h_data = m1y[m1y["hold"]==1]
        if len(h_data) > 10:
            fig, axes = plt.subplots(1,2,figsize=(13,4))
            for ax, col, lbl in zip(axes, ["mae","mfe"], ["MAE","MFE"]):
                for rg, clr, ls in [("calm","#e0e0e0","-"),("volatile","#777","--")]:
                    d = h_data[h_data["regime"]==rg][col].dropna()
                    if len(d) < 5: continue
                    clip = np.percentile(d, 96)
                    ax.hist(d[d<=clip*1.3], bins=40, color=clr, alpha=0.55,
                            edgecolor="none", label=rg, density=True)
                ax.set_title(f"{lbl}  by Regime  —  1H  —  {dr.upper()}")
                ax.set_xlabel("pips"); ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  TAB 5 — DERIVE
# ─────────────────────────────────────────────────────────────
with tabs[5]:
    meas = st.session_state.get("meas")
    if meas is None:
        st.info("Complete the Measurement step.")
    else:
        st.markdown('<div class="sec">Combination Testing</div>', unsafe_allow_html=True)

        dr_opts = sorted(meas["direction"].unique())
        d_col, h_col = st.columns(2)
        der_dir  = d_col.selectbox("Direction", dr_opts, key="der_dir")
        der_hold = h_col.select_slider("Hold Period", [1,2,3], value=1, key="der_hold")

        m = meas[(meas["direction"]==der_dir) & (meas["hold"]==der_hold)]
        if len(m) < 50:
            st.warning("Not enough measurements for reliable derive step (need 50+).")
        else:
            mae_v = m["mae"].dropna(); mfe_v = m["mfe"].dropna()
            stops   = [(p, float(np.percentile(mae_v,p))) for p in [30,50,70,90]]
            targets = [(p, float(np.percentile(mfe_v,p))) for p in [30,50,70,90,95]]
            queens  = [(p, float(np.percentile(mfe_v,p))) for p in [10,30,50]]
            n_combos = len(stops)*len(targets)*len(queens)

            st.markdown(f"""
            <div class="step-box" style="margin-bottom:14px;">
              <div class="step-num">COMBINATION SPACE</div>
              <div class="step-ttl">{n_combos} combinations  &times;  {len(m)} measurements</div>
              <div class="step-dsc">
                Stops: MAE P30 / P50 / P70 / P90  |
                Targets: MFE P30 / P50 / P70 / P90 / P95  |
                Queen: MFE P10 / P30 / P50
              </div>
            </div>""", unsafe_allow_html=True)

            if st.button("RUN DERIVE"):
                results = []
                prog = st.progress(0, "Testing…")
                total = n_combos; done = 0
                for sp, sv in stops:
                    for tp, tv in targets:
                        for qp, qv in queens:
                            tr = simulate(m, sv, tv, qv, spread)
                            g  = grade(tr)
                            if g:
                                results.append({
                                    "Stop":   f"MAE P{sp}",
                                    "Target": f"MFE P{tp}",
                                    "Queen":  f"MFE P{qp}",
                                    "Stop pips":   f"{sv:.1f}",
                                    "Tgt pips":    f"{tv:.1f}",
                                    "N":           g["n"],
                                    "Win Rate":    f"{g['win_rate']:.1%}",
                                    "EV pips":     f"{g['ev']:.2f}",
                                    "Prof Factor": f"{g['profit_factor']:.2f}",
                                    "Sharpe":      f"{g['sharpe']:.2f}",
                                    "SQN":         f"{g['sqn']:.2f}",
                                    "Max DD":      f"{g['max_dd']:.1f}",
                                    "Kelly":       f"{g['kelly']:.3f}",
                                    "_pf":    g["profit_factor"],
                                    "_ev":    g["ev"],
                                    "_stop":  sv,
                                    "_tgt":   tv,
                                    "_queen": qv,
                                    "_sp":    sp, "_tp":    tp,
                                    "_grade": g,
                                })
                            done += 1
                            prog.progress(done/total)
                prog.empty()
                if results:
                    rdf = pd.DataFrame(results).sort_values("_pf", ascending=False)
                    st.session_state["derive_res"] = rdf
                    best = rdf.iloc[0]
                    st.session_state["best"] = {
                        "stop":  best["_stop"], "target": best["_tgt"],
                        "queen": best["_queen"], "grade": best["_grade"],
                        "direction": der_dir, "hold": der_hold,
                        "stop_pct": best["_sp"], "tgt_pct": best["_tp"],
                    }

            rdf = st.session_state.get("derive_res")
            if rdf is not None and len(rdf) > 0:
                show_cols = ["Stop","Target","Queen","Stop pips","Tgt pips",
                             "N","Win Rate","EV pips","Prof Factor",
                             "Sharpe","SQN","Max DD","Kelly"]
                st.markdown('<div class="sec">Results — Top 30 by Profit Factor</div>',
                            unsafe_allow_html=True)
                st.dataframe(rdf[show_cols].head(30), use_container_width=True, height=380)

                best = rdf.iloc[0]
                st.markdown(f"""
                <div class="kpi" style="margin-top:14px;">
                  <div class="kpi-lbl">Best Combination</div>
                  <div class="kpi-val">{best['Stop pips']} / {best['Tgt pips']} pips</div>
                  <div class="kpi-sub">
                    {best['Stop']} stop  &middot;  {best['Target']} target  &middot;  {best['Queen']} queen
                    &mdash;  Win Rate {best['Win Rate']}  &middot;  PF {best['Prof Factor']}  &middot;  EV {best['EV pips']} pips
                  </div>
                </div>""", unsafe_allow_html=True)

                # PF Heatmap
                st.markdown('<div class="sec">Profit Factor Heatmap — Stop vs Target</div>',
                            unsafe_allow_html=True)
                hm = (rdf.groupby(["Stop","Target"])["_pf"]
                        .mean().reset_index()
                        .pivot(index="Stop", columns="Target", values="_pf"))
                if not hm.empty:
                    fig, ax = plt.subplots(figsize=(11,4))
                    vals = hm.values.clip(0, 3)
                    im = ax.imshow(vals, cmap="Greys", aspect="auto", vmin=0, vmax=2.5)
                    ax.set_xticks(range(len(hm.columns)))
                    ax.set_xticklabels(hm.columns, rotation=45, ha="right", fontsize=8)
                    ax.set_yticks(range(len(hm.index)))
                    ax.set_yticklabels(hm.index, fontsize=8)
                    ax.set_xlabel("Take Profit"); ax.set_ylabel("Stop Loss")
                    ax.set_title("Mean Profit Factor  (brighter = higher)")
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    for i in range(len(hm.index)):
                        for j in range(len(hm.columns)):
                            v = vals[i,j]
                            if not np.isnan(v):
                                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                                        fontsize=7.5,
                                        color="black" if v > 1.2 else "#888")
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  TAB 6 — VALIDATE
# ─────────────────────────────────────────────────────────────
with tabs[6]:
    best = st.session_state.get("best")
    meas = st.session_state.get("meas")

    if best is None:
        st.info("Complete the Derive step first.")
    else:
        bc = best
        m  = meas[(meas["direction"]==bc["direction"]) & (meas["hold"]==bc["hold"])]

        st.markdown(f"""
        <div class="step-box" style="margin-bottom:14px;">
          <div class="step-num">VALIDATING COMBINATION</div>
          <div class="step-ttl">Stop {bc['stop']:.1f} pips  &middot;  Target {bc['target']:.1f} pips  &middot;  Queen {bc['queen']:.1f} pips</div>
          <div class="step-dsc">
            Direction {bc['direction'].upper()}  &middot;  {bc['hold']}H hold  &middot;
            Stop = MAE P{bc['stop_pct']}  &middot;  Target = MFE P{bc['tgt_pct']}
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("RUN VALIDATION"):
            with st.spinner("Running 7-gate stress test…"):
                trades = simulate(m, bc["stop"], bc["target"], bc["queen"], spread)
                g = grade(trades)
                rdf = st.session_state.get("derive_res")
                n_combos = len(rdf) if rdf is not None else 60
                gates = run_gates(m, trades, g, n_combos,
                                  bc["stop"], bc["target"], bc["queen"], spread)
                st.session_state["val_res"] = {"trades":trades,"grade":g,"gates":gates}

        val = st.session_state.get("val_res")
        if val is not None:
            g     = val["grade"]
            gates = val["gates"]
            trades= val["trades"]

            # Grade card
            st.markdown('<div class="sec">Grade Card</div>', unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("N Trades",      g["n"])
            c2.metric("Win Rate",      f"{g['win_rate']:.1%}")
            c3.metric("EV (pips)",     f"{g['ev']:.2f}")
            c4.metric("Profit Factor", f"{g['profit_factor']:.2f}")
            c5,c6,c7,c8 = st.columns(4)
            c5.metric("Sharpe",        f"{g['sharpe']:.2f}")
            c6.metric("SQN",           f"{g['sqn']:.2f}")
            c7.metric("Max DD pips",   f"{g['max_dd']:.1f}")
            c8.metric("Kelly",         f"{g['kelly']:.3f}")

            # Gates
            st.markdown('<div class="sec">Seven-Gate Stress Test</div>', unsafe_allow_html=True)
            passed = 0; crit_fail = False
            for gid, gdata in gates.items():
                ok   = gdata["pass"]
                crit = gdata["critical"]
                if ok: passed += 1
                if crit and not ok: crit_fail = True
                badge = '<span class="pass">PASS</span>' if ok else '<span class="fail">FAIL</span>'
                crit_tag = ' <span style="font-size:9px;color:#333;">[CRITICAL]</span>' if crit else ""
                st.markdown(f"""
                <div class="step-box">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:11.5px;">{gdata['label']}{crit_tag}</span>
                    {badge}
                  </div>
                  <div style="font-size:9.5px;color:#4d4d4d;margin-top:5px;">{gdata['detail']}</div>
                </div>""", unsafe_allow_html=True)

            # Verdict
            ok_color = "#e0e0e0" if not crit_fail else "#555"
            verdict  = f"DEPLOY — {passed}/{len(gates)} gates passed" if not crit_fail \
                       else f"REJECT — critical gate failed  ({passed}/{len(gates)} passed)"
            st.markdown(f"""
            <div style="border:1px solid {'#e0e0e0' if not crit_fail else '#2a2a2a'};
                        padding:16px;margin-top:14px;background:#050505;">
              <div style="font-size:9px;color:#3d3d3d;margin-bottom:5px;">VERDICT</div>
              <div style="font-size:17px;font-weight:700;color:{ok_color};">{verdict}</div>
            </div>""", unsafe_allow_html=True)

            # Equity curve
            st.markdown('<div class="sec">Equity Curve</div>', unsafe_allow_html=True)
            ts = trades.sort_values("date").copy()
            ts["cum"] = ts["pnl"].cumsum()
            fig, ax = plt.subplots(figsize=(13,4))
            ax.plot(range(len(ts)), ts["cum"], color="#e0e0e0", linewidth=1.2)
            ax.fill_between(range(len(ts)), 0, ts["cum"],
                            where=ts["cum"]>=0, alpha=0.08, color="#e0e0e0")
            ax.fill_between(range(len(ts)), 0, ts["cum"],
                            where=ts["cum"]<0,  alpha=0.15, color="#888")
            ax.axhline(0, color="#2a2a2a", linewidth=.6)
            ax.set_xlabel("Trade Number"); ax.set_ylabel("Cumulative P&L (pips)")
            ax.set_title("Equity Curve — Validated Combination")
            ax.grid(True, alpha=0.25)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            # Rolling win rate
            st.markdown('<div class="sec">Rolling Win Rate (30-trade window)</div>',
                        unsafe_allow_html=True)
            ts["win_i"] = (ts["outcome"]=="win").astype(int)
            ts["roll_wr"] = ts["win_i"].rolling(30, min_periods=10).mean()
            fig2, ax2 = plt.subplots(figsize=(13,3))
            ax2.plot(range(len(ts)), ts["roll_wr"]*100, color="#e0e0e0", linewidth=1)
            ax2.axhline(g["win_rate"]*100, color="#555", linewidth=1, linestyle="--",
                        label=f"Overall {g['win_rate']:.1%}")
            ax2.set_ylim(0,100); ax2.set_xlabel("Trade"); ax2.set_ylabel("Win Rate %")
            ax2.set_title("Rolling Win Rate"); ax2.legend()
            ax2.grid(True, alpha=0.25)
            plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)


# ─────────────────────────────────────────────────────────────
#  TAB 7 — FORWARD TEST
# ─────────────────────────────────────────────────────────────
with tabs[7]:
    df   = st.session_state.get("df")
    best = st.session_state.get("best")

    if not use_fwd:
        st.info("Enable the Forward Test toggle in the sidebar.")
    elif df is None:
        st.info("Load data first.")
    elif best is None:
        st.info("Complete Derive and Validate steps first.")
    else:
        bc = best
        max_date = df["ts"].dt.date.max()

        st.markdown(f"""
        <div class="step-box" style="margin-bottom:14px;">
          <div class="step-num">FORWARD TEST CONFIGURATION</div>
          <div class="step-ttl">Out-of-sample verification</div>
          <div class="step-dsc">
            Training period: {df['ts'].dt.date.min()}  to  {fwd_cut - timedelta(days=1)}<br>
            Forward period:  {fwd_cut}  to  {max_date}<br>
            Stop {bc['stop']:.1f} pips  &middot;  Target {bc['target']:.1f} pips  &middot;  Queen {bc['queen']:.1f} pips
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("RUN FORWARD TEST"):
            with st.spinner("Running forward test…"):
                fwd_df = date_slice(df, fwd_cut, max_date)
                if len(fwd_df) < 200:
                    st.warning(f"Only {len(fwd_df)} rows in forward period.")

                fwd_box  = compute_box(fwd_df, sess_hr, box_mins)
                fwd_meas = compute_measurements(fwd_df, fwd_box, sess_hr, box_mins,
                                                 bc["direction"], [bc["hold"]])
                fwd_meas = add_regime(fwd_meas)
                fwd_meas = fwd_meas[fwd_meas["direction"]==bc["direction"]]

                fwd_trades = simulate(fwd_meas, bc["stop"], bc["target"],
                                      bc["queen"], spread)
                fwd_grade  = grade(fwd_trades)

                train_grade = bc["grade"]

                # Comparison
                st.markdown('<div class="sec">In-Sample vs Out-of-Sample</div>',
                            unsafe_allow_html=True)
                cols = st.columns(5)
                for col, metric, lbl, fmt in zip(cols,
                    ["n","win_rate","ev","profit_factor","sharpe"],
                    ["N","Win Rate","EV pips","Prof Factor","Sharpe"],
                    ["{:.0f}","{:.1%}","{:.2f}","{:.2f}","{:.2f}"]):
                    iv = train_grade.get(metric,0)
                    ov = fwd_grade.get(metric,0)
                    col.metric(f"{lbl}", fmt.format(ov),
                               delta=fmt.format(ov-iv))

                # Equity curves side by side
                st.markdown('<div class="sec">Equity Curves</div>', unsafe_allow_html=True)
                val_res = st.session_state.get("val_res")
                fig, axes = plt.subplots(1,2,figsize=(14,4))

                for ax, tr, title in zip(
                    axes,
                    [val_res["trades"] if val_res else pd.DataFrame(), fwd_trades],
                    ["In-Sample (Training)", "Out-of-Sample (Forward)"]
                ):
                    if len(tr) == 0:
                        ax.set_title(title + " — No Data"); continue
                    ts = tr.sort_values("date").copy()
                    ts["cum"] = ts["pnl"].cumsum()
                    ax.plot(range(len(ts)), ts["cum"], color="#e0e0e0", linewidth=1.2)
                    ax.fill_between(range(len(ts)), 0, ts["cum"],
                                    where=ts["cum"]>=0, alpha=0.08, color="#e0e0e0")
                    ax.fill_between(range(len(ts)), 0, ts["cum"],
                                    where=ts["cum"]<0,  alpha=0.15, color="#888")
                    ax.axhline(0, color="#2a2a2a", linewidth=.6)
                    ax.set_title(title); ax.set_xlabel("Trade")
                    ax.set_ylabel("Cumulative P&L (pips)")
                    ax.grid(True, alpha=0.25)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                # Regime breakdown
                st.markdown('<div class="sec">Forward Period — Regime Breakdown</div>',
                            unsafe_allow_html=True)
                rg_rows = []
                for rg in ["calm","volatile"]:
                    rt = fwd_trades[fwd_trades["regime"]==rg]
                    if len(rt) < 5: continue
                    rg_g = grade(rt)
                    rg_rows.append({
                        "Regime": rg.upper(), "N": rg_g["n"],
                        "Win Rate": f"{rg_g['win_rate']:.1%}",
                        "EV pips": f"{rg_g['ev']:.2f}",
                        "Prof Factor": f"{rg_g['profit_factor']:.2f}",
                        "Sharpe": f"{rg_g['sharpe']:.2f}",
                    })
                if rg_rows:
                    st.dataframe(pd.DataFrame(rg_rows), use_container_width=True)

                # Win rate comparison chart
                st.markdown('<div class="sec">Box Range vs MAE — Forward Period</div>',
                            unsafe_allow_html=True)
                h1f = fwd_meas[fwd_meas["hold"]==bc["hold"]]
                if len(h1f) > 10:
                    fig3, ax3 = plt.subplots(figsize=(10,3.5))
                    ax3.scatter(h1f["box_range"], h1f["mae"], s=7, alpha=0.3, color="#e0e0e0")
                    if len(h1f) > 20:
                        v = h1f[["box_range","mae"]].dropna()
                        z = np.polyfit(v["box_range"], v["mae"], 1)
                        xr = np.linspace(v["box_range"].quantile(.02),
                                         v["box_range"].quantile(.95), 100)
                        ax3.plot(xr, np.poly1d(z)(xr), color="#888",
                                 linewidth=1.5, label="OLS trend")
                    ax3.set_xlabel("Box Range (pips)"); ax3.set_ylabel("MAE (pips)")
                    ax3.set_title("Forward — Box Range vs MAE")
                    ax3.legend(); ax3.grid(True, alpha=0.25)
                    plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)
