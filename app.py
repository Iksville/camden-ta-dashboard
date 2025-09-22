import re
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Camden TA Insights", layout="wide")
st.title("Camden Temporary Accommodation – Interactive Insights")

#Load Data
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_borough_snapshot(path: Path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace("\n", " ") for c in df.columns]
    # find borough col
    borough_col = next((c for c in df.columns if c in ["borough","area","la","local authority","local_authority"]), df.columns[0])
    # find metric col
    metric_col = next((c for c in df.columns if any(k in c for k in ["per 1,000","per 1000","rate","proportion","percent","percentage","value"])), df.columns[1])
    df = df.rename(columns={borough_col:"borough", metric_col:"rate"})
    df["borough"] = df["borough"].astype(str).str.strip().str.title()
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    return df.dropna(subset=["borough","rate"])

@st.cache_data
def load_london_timeseries(path: Path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace("\n"," ") for c in df.columns]
    # find a period column
    period_col = next((c for c in df.columns if any(k in c for k in ["period","date","year","quarter","time"])), df.columns[0])
    df = df.rename(columns={period_col:"period_raw"})
    # parse year/quarter
    def parse_period(x):
        s = str(x)
        y = None; q = None
        my = re.search(r"(20\d{2})", s)
        if my: y = int(my.group(1))
        mq = re.search(r"Q([1-4])", s, re.I)
        if mq: q = int(mq.group(1))
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.notna(dt) and y is None: y = dt.year
        return y, q
    yq = df["period_raw"].apply(parse_period)
    df["year"] = yq.apply(lambda t: t[0])
    df["quarter"] = yq.apply(lambda t: t[1])

    # numeric measures only
    non_measures = {"period_raw","year","quarter"}
    measures = []
    for c in df.columns:
        if c in non_measures: continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[c]): measures.append(c)
    if not measures:
        raise ValueError("No numeric columns detected in time-series file.")

    tidy = df.melt(id_vars=["year","quarter"], value_vars=measures,
                   var_name="type", value_name="value").dropna(subset=["year","value"])
    tidy["type"] = tidy["type"].str.replace("_"," ").str.title()
    tidy["period"] = tidy.apply(lambda r: f"{int(r['year'])} Q{int(r['quarter'])}" if pd.notna(r['quarter']) else str(int(r['year'])), axis=1)
    return tidy

# Graceful error if file missing
missing = []
bpath = DATA_DIR / "borough_ta_2024.csv"
tpath = DATA_DIR / "london_ta_types_2002_2025Q1.csv"
if not bpath.exists(): missing.append(bpath.name)
if not tpath.exists(): missing.append(tpath.name)
if missing:
    st.error(f"Missing data files in /data: {', '.join(missing)}")
    st.stop()

borough_df = load_borough_snapshot(bpath)
ts_tidy = load_london_timeseries(tpath)

#Sidebar Filters
st.sidebar.header("Filters")
boroughs = sorted(borough_df["borough"].unique())
focus_borough = st.sidebar.selectbox("Focus borough", options=boroughs,
                                     index=(boroughs.index("Camden") if "Camden" in boroughs else 0))

year_min, year_max = int(ts_tidy["year"].min()), int(ts_tidy["year"].max())
year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max,
                               value=(max(year_min, year_max-6), year_max))
types = sorted(ts_tidy["type"].unique())
selected_types = st.sidebar.multiselect("Accommodation types", options=types, default=types[:3])
stacked = st.sidebar.checkbox("Show stacked area", value=True)

#KPIs
ranked = borough_df.sort_values("rate", ascending=False).reset_index(drop=True)
if focus_borough not in ranked["borough"].values:
    st.warning(f"{focus_borough} not in snapshot file.")
else:
    row = ranked[ranked["borough"] == focus_borough].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{focus_borough} rank (1=highest rate)", f"{int(row.name)+1} / {len(ranked)}")
    col2.metric("Rate (per 1,000 households)", f"{row['rate']:,.2f}")
    london_avg = borough_df["rate"].mean()
    delta = ((row["rate"] - london_avg) / london_avg * 100) if london_avg else 0
    col3.metric("vs London average", f"{delta:+.1f}%")

st.caption("Source: DLUHC statutory homelessness tables; London Datastore extracts.")
st.markdown("---")

#Chart 1: Borough ranking
st.subheader("Borough snapshot – latest rate of households in temporary accommodation")
rank_plot_df = ranked.copy()
rank_plot_df["highlight"] = rank_plot_df["borough"].eq(focus_borough).map({True: focus_borough, False: "Other boroughs"})
fig_rank = px.bar(rank_plot_df,
                  x="rate", y="borough",
                  orientation="h",
                  color="highlight",
                  category_orders={"borough": rank_plot_df["borough"][::-1].tolist()},
                  labels={"rate":"Rate (per 1,000 households)", "borough":"Borough"})
fig_rank.update_layout(height=720, legend_title=None, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_rank, use_container_width=True)

st.markdown("---")

#Chart 2: Time series
st.subheader("Temporary accommodation over time – London totals by accommodation type")
mask = (ts_tidy["year"].between(year_range[0], year_range[1])) & (ts_tidy["type"].isin(selected_types))
ts_filtered = ts_tidy[mask].sort_values(["year","quarter"]).assign(x=lambda d: d["period"])

if stacked:
    fig_ts = px.area(ts_filtered, x="x", y="value", color="type", labels={"x":"Period","value":"Households"})
else:
    fig_ts = px.line(ts_filtered, x="x", y="value", color="type", markers=True, labels={"x":"Period","value":"Households"})
fig_ts.update_layout(height=520, legend_title=None, margin=dict(l=10, r=10, t=30, b=10), xaxis_tickangle=-30)
st.plotly_chart(fig_ts, use_container_width=True)

#Explanation
st.markdown("---")
st.subheader("Story")

# PPossible pull dynamic figures already calculated above
try:
    focus_rank = int(row.name) + 1            
    focus_rate = float(row["rate"])
    london_avg = float(london_avg)
    vs_london_pct = ((focus_rate - london_avg) / london_avg) * 100 if london_avg else None
except Exception:
    focus_rank = None
    focus_rate = None
    vs_london_pct = None

tab1, tab2, tab3, tab4 = st.tabs([
    "Why the trend matters",
    "Camden vs London",
    "Implications for services",
    "Possible AI/LLM applications"
])

with tab1:
    st.markdown(f"""
**Why this trend matters**

- **Budget pressure**: Temporary Accommodation (TA) is one of the fastest-growing cost lines for London boroughs. Even small % changes translate into **material £ impacts** across a year.
- **Family stability**: Rising TA correlates with increased placements for **households with children**, affecting schooling continuity and wellbeing.
- **System flow**: TA is a bottleneck indicator — it reflects earlier upstream issues (evictions, affordability) and downstream capacity (move-on options). Tracking it helps **prioritise the right levers**.
- **Operational planning**: Understanding the **timing** (which quarters/years spike) supports staff rostering, procurement cycles, and early communication with providers.
""")

with tab2:
    # Data Awareness
    bullets = []
    if focus_rank is not None:
        bullets.append(f"- **Rank**: {focus_borough} is currently **{focus_rank}** out of {len(ranked)} London boroughs (1 = highest TA rate).")
    if focus_rate is not None:
        bullets.append(f"- **Current rate**: {focus_rate:,.2f} households in TA **per 1,000 households** in {focus_borough}.")
    if vs_london_pct is not None:
        sign = "above" if vs_london_pct >= 0 else "below"
        bullets.append(f"- **Vs London average**: **{abs(vs_london_pct):.1f}% {sign}** the London mean.")
    if not bullets:
        bullets.append("- **Camden vs London** comparison available when data loads (see KPIs above).")

    st.markdown("**Where Camden sits vs London**")
    st.markdown("\n".join(bullets))
    st.markdown("""
**What to watch in the chart**
- Are spikes concentrated in particular **quarters** (e.g., Q4/Q1)?  
- Which **accommodation types** (e.g., nightly-paid, PSL, B&B) are driving increases?  
- Does the profile suggest **procurement** challenges (e.g., more nightly-paid) or **case complexity** (longer stays, fewer move-ons)?
""")

with tab3:
    st.markdown("""
**What this means for planning & service delivery**

1) **Procurement & contracts**
   - Use trend windows to negotiate capacity ahead of expected peaks.
   - Shift mix away from **nightly-paid** where possible; target **longer-term, lower-unit-cost** supply.

2) **Prevention targeting**
   - Focus upstream on cohorts most at risk (e.g., **families with arrears** or **Section 21 notices**).
   - Partner with advice services to **triage earlier** and reduce flow into TA.

3) **Move-on pipeline**
   - Track time-in-TA and unblock **move-on pathways** (PRS incentives, supported housing throughput).
   - Align lettings, voids, and landlord engagement to reduce **average TA duration**.

4) **Operational resilience**
   - Plan **staffing and budget** around periods with historically higher demand.
   - Use quarterly indicators as **early warnings** for finance and cabinet reports.

> The dashboard gives managers a quick read on *where pressure is coming from* and *which levers to pull* next.
""")

with tab4:
    st.markdown("""
**Where AI/LLMs can help right now**

1) **Demand Forecasting (next 3–12 months)**
   - Train a lightweight model on historic TA counts by quarter + macro signals (evictions, rent inflation, UC changes).
   - Output: **scenario forecasts** with confidence bands for finance planning and procurement.

2) **Smart Placement & Matching**
   - Use rules + ML to **match households to properties** (distance to school/work, accessibility, cost ceiling).
   - Prioritise options that **reduce churn** and **minimise nightly-paid usage**.

3) **Automated Narrative Summaries**
   - Generate **plain-English summaries** each quarter for cabinet packs:
     > “TA rose by X% this quarter, driven by Y type; projected spend variance £Z. Recommended actions: …”
   - Cuts manual drafting time and ensures **consistent, evidence-based messaging**.

4) **Early-Warning Signals**
   - Classify inbound **letters/emails** (e.g., Section 21, arrears) and triage to prevention teams.
   - Set alerts when risk indicators breach thresholds (e.g., **spike in B&B usage**).

**Implementation notes**
- Start with **secure, in-house data**; no PII to external services without DPIA.
- Use small, practical models (Prophet/XGBoost) + **LLMs for summarisation**.
- Keep an auditable trail: **data → features → forecast → decision**.
""")
