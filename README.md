# FinSight AI

A personal finance dashboard that turns transaction history into clear spending insights, forecasts, and practical saving recommendations.

## Features

- Import transactions from CSV or use the included sample dataset
- Track income, expenses, savings, and monthly cash flow
- Explore spending by category and over time
- Automatically categorize common transaction descriptions
- Detect unusual expenses with Isolation Forest
- Forecast upcoming monthly expenses using a time-series regression model
- Generate saving recommendations from spending patterns
- Filter transactions by date, type, and category
- Export filtered transactions and a monthly report

## Preview

Run the application locally to explore the interactive dashboard.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/ahmedkhouya-exe/finsight-ai.git
cd finsight-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch FinSight AI

```bash
streamlit run app.py
```

## CSV format

Upload a CSV file with these columns:

| Column | Description |
|---|---|
| date | Transaction date |
| description | Transaction description |
| amount | Numeric transaction amount |
| type | `income` or `expense` |
| category | Optional category |

Amounts should be positive numbers. Dates should be parseable by Pandas. If `type` is omitted, positive amounts are treated as income and negative amounts as expenses.

An example file is available at `data/sample_transactions.csv`.

## Project structure

```
finsight-ai/
├── app.py
├── finance_engine.py
├── data/
│   └── sample_transactions.csv
├── tests/
│   └── test_finance_engine.py
├── requirements.txt
└── README.md
```

## Run tests

```bash
pytest
```

## Built with

Python · Streamlit · Pandas · Plotly · Scikit-learn · SQLite
