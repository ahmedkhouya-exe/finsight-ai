import pandas as pd
import pytest

from finance_engine import (
    calculate_summary,
    categorize_description,
    detect_anomalies,
    forecast_expenses,
    generate_recommendations,
    normalize_transactions,
)

def test_normalize_transactions_and_infer_category():
    raw = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "description": ["Monthly salary", "Carrefour groceries"],
        "amount": [3000, -120],
    })
    result = normalize_transactions(raw)
    assert result["type"].tolist() == ["income", "expense"]
    assert result["amount"].tolist() == [3000, 120]
    assert result["category"].tolist() == ["Salary", "Groceries"]

def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        normalize_transactions(pd.DataFrame({"date": [], "amount": []}))

def test_summary():
    df = normalize_transactions(pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "description": ["Salary", "Rent"],
        "amount": [1000, -400],
    }))
    summary = calculate_summary(df)
    assert summary["income"] == 1000
    assert summary["expenses"] == 400
    assert summary["savings"] == 600
    assert summary["savings_rate"] == 60

def test_categorization():
    assert categorize_description("Metro pass") == "Transport"
    assert categorize_description("Monthly salary", "income") == "Salary"

def test_forecast_returns_requested_months():
    df = normalize_transactions(pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=12, freq="MS").strftime("%Y-%m-%d"),
        "description": ["Rent"] * 12,
        "amount": [-500] * 12,
    }))
    forecast = forecast_expenses(df, periods=2)
    assert len(forecast) == 2
    assert forecast["forecast"].ge(0).all()

def test_anomaly_detector_handles_small_dataset():
    df = normalize_transactions(pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "description": ["Food", "Rent"],
        "amount": [-20, -300],
    }))
    result = detect_anomalies(df)
    assert "anomaly" in result.columns
    assert not result["anomaly"].any()

def test_recommendations_are_generated():
    df = normalize_transactions(pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "description": ["Salary", "Shopping"],
        "amount": [1000, -500],
    }))
    assert generate_recommendations(df)
