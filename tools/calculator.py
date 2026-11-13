import pandas as pd


def safe_float(value):
    try:
        return float(str(value).replace("₹", "").replace(",", "").replace(" ", ""))
    except Exception:
        return 0.0


def sum_numeric(series):
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def calculate_percent(part, total):
    if total in (None, 0):
        return 0.0
    return (part / total) * 100


def calculate_monthly_totals(df):
    if df.empty:
        return []
    monthly = df.copy()
    monthly["month"] = pd.to_datetime(monthly["date"], errors="coerce").dt.to_period("M").astype(str)
    result = monthly[monthly["transaction_type"] == "expense"].groupby("month", as_index=False)["amount"].sum()
    return result.rename(columns={"amount": "amount"}).to_dict(orient="records")
