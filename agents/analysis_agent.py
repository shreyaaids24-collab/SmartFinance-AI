import pandas as pd


def compute_summary(df):
    if df is None or df.empty:
        return {
            "total_income": 0.0,
            "total_expense": 0.0,
            "net_savings": 0.0,
            "transaction_count": 0,
            "category_breakdown": [],
            "monthly_spend": [],
            "top_expenses": [],
            "average_monthly_expense": 0.0,
        }

    cleaned = df.copy()
    if "is_valid" in cleaned:
        cleaned = cleaned[cleaned["is_valid"].fillna(False)].copy()
    if cleaned.empty:
        return compute_summary(pd.DataFrame())
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce").fillna(0)
    if "category" not in cleaned:
        cleaned["category"] = "Others"
    else:
        cleaned["category"] = cleaned["category"].fillna("Others")

    income = cleaned[cleaned["transaction_type"] == "income"]["amount"].sum()
    expense = cleaned[cleaned["transaction_type"] == "expense"]["amount"].sum()
    net_savings = income - expense

    category_breakdown = (
        cleaned[cleaned["transaction_type"] == "expense"]
        .groupby("category", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "amount"})
        .sort_values("amount", ascending=False)
    )
    category_breakdown["percentage"] = (category_breakdown["amount"] / expense * 100) if expense else 0

    monthly_expenses = cleaned[
        (cleaned["transaction_type"] == "expense") & cleaned["date"].notna()
    ]
    monthly_spend = (
        monthly_expenses
        .assign(month=lambda df_: df_["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["amount"]
        .sum()
        .sort_values("month")
        .rename(columns={"amount": "amount"})
    )

    top_expenses = (
        cleaned[cleaned["transaction_type"] == "expense"]
        .sort_values("amount", ascending=False)
        .head(5)[["description", "amount", "category"]]
        .rename(columns={"description": "description", "amount": "amount", "category": "category"})
    )

    transaction_count = len(cleaned)
    average_monthly_expense = monthly_spend["amount"].mean() if not monthly_spend.empty else 0.0

    return {
        "total_income": float(income),
        "total_expense": float(expense),
        "net_savings": float(net_savings),
        "transaction_count": int(transaction_count),
        "category_breakdown": category_breakdown.to_dict(orient="records"),
        "monthly_spend": monthly_spend.to_dict(orient="records"),
        "top_expenses": top_expenses.to_dict(orient="records"),
        "average_monthly_expense": float(average_monthly_expense),
    }
