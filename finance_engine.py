"""Core data processing and machine-learning utilities for FinSight AI."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

CATEGORIES = [
    "Housing", "Groceries", "Transport", "Dining", "Utilities",
    "Shopping", "Health", "Entertainment", "Travel", "Education",
    "Salary", "Freelance", "Other",
]

KEYWORDS = {
    "Housing": ["rent", "apartment", "mortgage"],
    "Groceries": ["grocery", "groceries", "supermarket", "carrefour", "lidl", "kaufland"],
    "Transport": ["metro", "bus", "train", "uber", "bolt", "taxi", "fuel", "gas station"],
    "Dining": ["restaurant", "coffee", "cafe", "food", "pizza", "dinner", "lunch"],
    "Utilities": ["electricity", "water bill", "internet", "mobile", "phone", "subscription"],
    "Shopping": ["shopping", "clothing", "electronics", "online store", "amazon"],
    "Health": ["pharmacy", "doctor", "gym", "hospital", "medicine"],
    "Entertainment": ["cinema", "movie", "concert", "game", "ticket"],
    "Travel": ["hotel", "flight", "trip", "airbnb", "booking"],
    "Education": ["course", "tuition", "book", "university"],
    "Salary": ["salary", "payroll", "wage"],
    "Freelance": ["freelance", "client payment", "project payment"],
}

def categorize_description(description: str, transaction_type: str = "expense") -> str:
    text = str(description).lower()
    for category, words in KEYWORDS.items():
        if any(word in text for word in words):
            if transaction_type == "income" and category not in {"Salary", "Freelance"}:
                continue
            return category
    return "Other"

def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize uploaded transaction data."""
    df = frame.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "description", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["description"] = df["description"].fillna("Unspecified transaction").astype(str)
    df = df.dropna(subset=["date", "amount"]).copy()
    if "type" not in df:
        df["type"] = np.where(df["amount"] >= 0, "income", "expense")
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["type"] = df["type"].replace({"income": "income", "expense": "expense"})
    df = df[df["type"].isin(["income", "expense"])].copy()
    df["amount"] = df["amount"].abs()
    if "category" not in df:
        df["category"] = ""
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    empty = df["category"].eq("") | df["category"].str.lower().isin(["nan", "none"])
    df.loc[empty, "category"] = [
        categorize_description(desc, kind)
        for desc, kind in zip(df.loc[empty, "description"], df.loc[empty, "type"])
    ]
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "description", "amount", "type", "category", "month"]]

def calculate_summary(df: pd.DataFrame) -> dict:
    income = float(df.loc[df["type"] == "income", "amount"].sum())
    expenses = float(df.loc[df["type"] == "expense", "amount"].sum())
    return {
        "income": income,
        "expenses": expenses,
        "savings": income - expenses,
        "savings_rate": (income - expenses) / income * 100 if income else 0.0,
        "transactions": int(len(df)),
    }

def monthly_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "income", "expenses", "savings"])
    pivot = df.pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0)
    result = pivot.rename(columns={"income": "income", "expense": "expenses"})
    for col in ["income", "expenses"]:
        if col not in result:
            result[col] = 0.0
    result["savings"] = result["income"] - result["expenses"]
    return result.reset_index().sort_values("month")

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    expenses = df[df["type"] == "expense"].copy()
    if len(expenses) < 8 or expenses["amount"].nunique() < 3:
        expenses["anomaly"] = False
        expenses["anomaly_score"] = 0.0
        return expenses
    model = IsolationForest(contamination=min(max(contamination, 0.01), 0.25), random_state=42)
    features = np.log1p(expenses[["amount"]].clip(lower=0))
    predictions = model.fit_predict(features)
    expenses["anomaly"] = predictions == -1
    expenses["anomaly_score"] = -model.decision_function(features)
    return expenses.sort_values("anomaly_score", ascending=False)

def forecast_expenses(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    monthly = monthly_cashflow(df)
    if monthly.empty:
        return pd.DataFrame(columns=["month", "forecast"])
    values = monthly["expenses"].astype(float).to_numpy()
    future_index = np.arange(len(values), len(values) + periods).reshape(-1, 1)
    if len(values) >= 3:
        model = make_pipeline(
            # A compact ensemble keeps forecasts responsive to recent spending.
            RandomForestRegressor(n_estimators=120, max_depth=3, random_state=42)
        )
        x = np.arange(len(values)).reshape(-1, 1)
        model.fit(x, values)
        predictions = model.predict(future_index)
    else:
        model = LinearRegression().fit(np.arange(len(values)).reshape(-1, 1), values)
        predictions = model.predict(future_index)
    last_period = pd.Period(monthly["month"].iloc[-1], freq="M")
    months = [(last_period + i).strftime("%Y-%m") for i in range(1, periods + 1)]
    return pd.DataFrame({"month": months, "forecast": np.maximum(predictions, 0).round(2)})

def generate_recommendations(df: pd.DataFrame) -> list[str]:
    expenses = df[df["type"] == "expense"]
    if expenses.empty:
        return ["Add expense transactions to receive tailored saving recommendations."]
    by_category = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
    total = float(expenses["amount"].sum())
    tips = []
    if not by_category.empty and total > 0:
        category = str(by_category.index[0])
        share = float(by_category.iloc[0] / total * 100)
        if category not in {"Housing", "Utilities"} and share >= 20:
            tips.append(f"{category} represents {share:.0f}% of recorded expenses. Review this category for flexible costs.")
    cashflow = monthly_cashflow(df)
    if len(cashflow) >= 2:
        previous, latest = cashflow.iloc[-2], cashflow.iloc[-1]
        if latest["expenses"] > previous["expenses"] * 1.15:
            tips.append("Expenses rose by more than 15% compared with the previous month. Check recent purchases for one-off costs.")
    summary = calculate_summary(df)
    if summary["income"] > 0 and summary["savings_rate"] < 10:
        tips.append("Your savings rate is below 10%. Try setting an automatic savings target immediately after income arrives.")
    if summary["savings"] < 0:
        tips.append("Recorded expenses exceed income. Prioritize essential bills and set a short-term spending limit.")
    if not tips:
        tips.append("Your recorded cash flow is in a healthy range. Keep tracking consistently and review your largest flexible category each month.")
    return tips[:4]
