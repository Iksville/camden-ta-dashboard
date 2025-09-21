import re
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Camden TA Insights", layout="wide")
st.title("Camden Temporary Accommodation – Interactive Insights")

# ---------- Load data ----------
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

# Graceful error if files missing
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

# ---------- Sidebar ----------
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

# ---------- KPIs ----------
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

# ---------- Chart 1: Borough ranking ----------
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

# ---------- Chart 2: Time series ----------
st.subheader("Temporary accommodation over time – London totals by accommodation type")
mask = (ts_tidy["year"].between(year_range[0], year_range[1])) & (ts_tidy["type"].isin(selected_types))
ts_filtered = ts_tidy[mask].sort_values(["year","quarter"]).assign(x=lambda d: d["period"])

if stacked:
    fig_ts = px.area(ts_filtered, x="x", y="value", color="type", labels={"x":"Period","value":"Households"})
else:
    fig_ts = px.line(ts_filtered, x="x", y="value", color="type", markers=True, labels={"x":"Period","value":"Households"})
fig_ts.update_layout(height=520, legend_title=None, margin=dict(l=10, r=10, t=30, b=10), xaxis_tickangle=-30)
st.plotly_chart(fig_ts, use_container_width=True)

with st.expander("About this demo"):
    st.markdown("""
- **Goal**: demonstrate analysis, coding and storytelling on a Camden-relevant topic.
- Sidebar lets you explore periods and accommodation types; Camden is highlighted in the ranking.
- Built with **Python, pandas, Plotly, Streamlit**.
""")
