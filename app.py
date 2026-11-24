from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finance_engine import (
    calculate_summary,
    detect_anomalies,
    forecast_expenses,
    generate_recommendations,
    monthly_cashflow,
    normalize_transactions,
)

st.set_page_config(page_title="FinSight AI", page_icon="💹", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#e8edf7; --muted:#8d9ab1; --panel:#111b2c; --line:#243249; --green:#35d39a; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(circle at 80% 0%, #152b3a 0, #0a1220 38%, #080f1b 100%); color:var(--ink); }
.block-container { max-width:1440px; padding-top:2.2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.04em; }
[data-testid="stSidebar"] { background:#0c1625; border-right:1px solid #1e2b40; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#142238,#101a2a); border:1px solid #26364e; padding:18px 20px; border-radius:16px; }
[data-testid="stMetricLabel"] { color:#9aa9bf; }
[data-testid="stMetricValue"] { color:#f4f7fc; font-family:'Manrope',sans-serif; }
[data-testid="stMetricDelta"] { color:#35d39a; }
div[data-testid="stTabs"] button { color:#aab7cb; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#35d39a; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid #35d39a; background:#35d39a; color:#07151a; font-weight:700; }
div[data-testid="stDataFrame"] { border:1px solid #26364e; border-radius:12px; overflow:hidden; }
.hero { padding:8px 0 22px 0; }
.eyebrow { color:#35d39a; font-size:12px; letter-spacing:.18em; font-weight:700; text-transform:uppercase; }
.hero h1 { font-size:42px; margin:6px 0 4px 0; color:#f4f7fc; }
.hero p { color:#9aa9bf; font-size:15px; margin:0; }
.section-title { color:#eaf0fa; font-size:19px; font-weight:700; margin:10px 0 14px; }
.insight { background:linear-gradient(120deg,#142b31,#122236); border:1px solid #28514c; padding:15px 17px; border-radius:13px; margin-bottom:10px; color:#dcebe9; }
.small-muted { color:#91a0b6; font-size:13px; }
</style>
""", unsafe_allow_html=True)

COLORS = {"income": "#35d39a", "expense": "#ff8b75", "savings": "#7ea7ff"}

@st.cache_data
def load_sample():
    path = Path(__file__).parent / "data" / "sample_transactions.csv"
    return normalize_transactions(pd.read_csv(path))

def money(value: float, currency: str) -> str:
    return f"{value:,.2f} {currency}"

def chart_layout(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aab7cb", family="DM Sans"),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(showgrid=False, linecolor="#26364e"),
        yaxis=dict(gridcolor="#223047", zerolinecolor="#26364e"),
    )
    return fig

st.sidebar.markdown("## 💹 FinSight AI")
st.sidebar.caption("PERSONAL FINANCE INTELLIGENCE")
st.sidebar.divider()
source = st.sidebar.radio("Transaction source", ["Sample data", "Upload CSV"], index=0)
uploaded = None
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload transactions", type=["csv"])
    st.sidebar.caption("Required columns: date, description, amount")
currency = st.sidebar.selectbox("Currency", ["RON", "EUR", "USD", "GBP"], index=0)
st.sidebar.divider()
st.sidebar.markdown("**Your data stays in this session**")
st.sidebar.caption("Upload a CSV to analyze your own transaction history. No account connection is required.")

try:
    if source == "Upload CSV":
        if uploaded is None:
            st.markdown('<div class="hero"><div class="eyebrow">PERSONAL FINANCE OVERVIEW</div><h1>Make every transaction count.</h1><p>Understand your spending, spot patterns, and plan your next move.</p></div>', unsafe_allow_html=True)
            st.info("Upload a transaction CSV in the sidebar to get started, or switch to Sample data to explore the dashboard.")
            st.stop()
        raw = pd.read_csv(uploaded)
        transactions = normalize_transactions(raw)
    else:
        transactions = load_sample()
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

if transactions.empty:
    st.warning("No valid transactions were found. Check the CSV format and try again.")
    st.stop()

st.markdown('<div class="hero"><div class="eyebrow">PERSONAL FINANCE OVERVIEW</div><h1>Make every transaction count.</h1><p>Understand your spending, spot patterns, and plan your next move.</p></div>', unsafe_allow_html=True)

min_date = transactions["date"].min().date()
max_date = transactions["date"].max().date()
with st.expander("Filter your transactions", expanded=False):
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    available_categories = sorted(transactions["category"].unique().tolist())
    selected_categories = st.multiselect("Categories", available_categories, default=available_categories)
    selected_types = st.multiselect("Transaction types", ["income", "expense"], default=["income", "expense"])

filtered = transactions.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[filtered["date"].dt.date.between(start, end)]
if selected_categories:
    filtered = filtered[filtered["category"].isin(selected_categories)]
else:
    filtered = filtered.iloc[0:0]
if selected_types:
    filtered = filtered[filtered["type"].isin(selected_types)]
else:
    filtered = filtered.iloc[0:0]

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

summary = calculate_summary(filtered)
previous_months = monthly_cashflow(filtered)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total income", money(summary["income"], currency))
col2.metric("Total expenses", money(summary["expenses"], currency))
col3.metric("Net savings", money(summary["savings"], currency), delta=f'{summary["savings_rate"]:.1f}% savings rate')
col4.metric("Transactions", f'{summary["transactions"]:,}')

tab_overview, tab_insights, tab_transactions = st.tabs(["Overview", "Insights & forecast", "Transactions"])

with tab_overview:
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown('<div class="section-title">Monthly cash flow</div>', unsafe_allow_html=True)
        if not previous_months.empty:
            flow = previous_months.melt(id_vars="month", value_vars=["income", "expenses"], var_name="type", value_name="amount")
            fig = px.bar(flow, x="month", y="amount", color="type", barmode="group",
                         color_discrete_map=COLORS, labels={"month":"Month","amount":f"Amount ({currency})","type":"Type"})
            st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Where your money goes</div>', unsafe_allow_html=True)
        expense_categories = filtered[filtered["type"] == "expense"].groupby("category", as_index=False)["amount"].sum()
        if not expense_categories.empty:
            fig = px.pie(expense_categories, names="category", values="amount", hole=.68,
                         color_discrete_sequence=["#35d39a","#7ea7ff","#ff8b75","#e6bd69","#a58bff","#55c6d8","#df82b3"])
            fig.update_traces(textposition="inside", textinfo="percent")
            st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
        else:
            st.info("No expense transactions in this period.")
    st.markdown('<div class="section-title">Income, expenses & savings</div>', unsafe_allow_html=True)
    if not previous_months.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=previous_months["month"], y=previous_months["savings"], name="Savings", mode="lines+markers", line=dict(color=COLORS["savings"], width=3), fill="tozeroy", fillcolor="rgba(126,167,255,.08)"))
        fig.add_hline(y=0, line_dash="dot", line_color="#62718a")
        st.plotly_chart(chart_layout(fig, 260), use_container_width=True)

with tab_insights:
    st.markdown('<div class="section-title">Your financial action plan</div>', unsafe_allow_html=True)
    for tip in generate_recommendations(filtered):
        st.markdown(f'<div class="insight">✦ &nbsp; {tip}</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Expense forecast</div>', unsafe_allow_html=True)
        forecast = forecast_expenses(filtered, periods=3)
        if not forecast.empty:
            history = monthly_cashflow(filtered).tail(6)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history["month"], y=history["expenses"], name="Historical", mode="lines+markers", line=dict(color="#35d39a", width=3)))
            fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["forecast"], name="Forecast", mode="lines+markers", line=dict(color="#e6bd69", width=3, dash="dash")))
            st.plotly_chart(chart_layout(fig, 300), use_container_width=True)
            st.caption("Forecasts are estimates based on the transaction history in the selected period.")
    with right:
        st.markdown('<div class="section-title">Unusual transactions</div>', unsafe_allow_html=True)
        anomalies = detect_anomalies(filtered)
        flagged = anomalies[anomalies["anomaly"]]
        if flagged.empty:
            st.success("No unusual expense patterns were detected in this selection.")
        else:
            st.dataframe(flagged[["date", "description", "category", "amount"]].sort_values("date", ascending=False),
                         use_container_width=True, hide_index=True,
                         column_config={"date": st.column_config.DateColumn("Date"), "amount": st.column_config.NumberColumn("Amount", format="%.2f")})
            st.caption("Flagged by an Isolation Forest model based on transaction amounts.")

with tab_transactions:
    st.markdown('<div class="section-title">Transaction history</div>', unsafe_allow_html=True)
    display = filtered.sort_values("date", ascending=False).copy()
    st.dataframe(display[["date", "description", "type", "category", "amount"]],
                 use_container_width=True, hide_index=True,
                 column_config={
                     "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                     "amount": st.column_config.NumberColumn(f"Amount ({currency})", format="%.2f"),
                     "type": st.column_config.TextColumn("Type"),
                 })
    csv_bytes = display.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered transactions", csv_bytes, "finsight-transactions.csv", "text/csv")

report = [
    "FINSIGHT AI — FINANCIAL SUMMARY",
    f"Period: {filtered['date'].min().date()} to {filtered['date'].max().date()}",
    f"Income: {money(summary['income'], currency)}",
    f"Expenses: {money(summary['expenses'], currency)}",
    f"Net savings: {money(summary['savings'], currency)}",
    f"Savings rate: {summary['savings_rate']:.1f}%",
    "",
    "RECOMMENDATIONS",
    *[f"- {tip}" for tip in generate_recommendations(filtered)],
]
st.sidebar.download_button("Download monthly summary", "\n".join(report), "finsight-summary.txt", "text/plain", use_container_width=True)
st.markdown('<p class="small-muted" style="text-align:center;margin-top:42px">FinSight AI · Personal Finance Intelligence</p>', unsafe_allow_html=True)
