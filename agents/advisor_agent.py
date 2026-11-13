import pandas as pd

from utils.formatting import format_inr


def generate_advice(df, llm=None):
    if df is None or df.empty:
        return ["Upload transaction data to generate actionable savings suggestions."]

    usable = df.copy()
    if "is_valid" in usable:
        usable = usable[usable["is_valid"].fillna(False)]
    usable["amount"] = pd.to_numeric(usable["amount"], errors="coerce")
    expenses = usable[
        (usable["transaction_type"] == "expense")
        & usable["amount"].notna()
        & usable["amount"].gt(0)
    ].copy()
    if expenses.empty:
        return ["No expense records found. Add more transactions to begin financial advice."]

    category_totals = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_category = category_totals.index[0] if not category_totals.empty else "Others"
    top_value = float(category_totals.iloc[0]) if not category_totals.empty else 0.0
    total_expense = float(expenses["amount"].sum())
    share = (top_value / total_expense * 100) if total_expense else 0.0

    suggestions = [
        f"Your top spending category is {top_category} at {format_inr(top_value)} "
        f"({share:.1f}% of recorded expenses). Review its largest or recurring "
        "transactions before setting a budget cap.",
        "Track cash flow weekly and move any non-essential subscriptions to a quarterly review cycle.",
        "Use a spending guardrail of 10% to 15% below your current monthly average for discretionary categories.",
    ]

    if llm is not None:
        summary = f"Top category: {top_category}; total expense: {format_inr(total_expense)}"
        advice = llm.invoke(
            "You are a finance coaching assistant. Give three concise, grounded recommendations based on this summary. "
            f"Summary: {summary}"
        )
        content = getattr(advice, "content", str(advice))
        suggestions.append(content)

    return suggestions
