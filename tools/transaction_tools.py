import pandas as pd


def filter_transactions(df, category=None, search_text=None):
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "description", "category", "amount", "type", "transaction_id"])

    filtered = df.copy()
    if category:
        filtered = filtered[filtered["category"].str.contains(category, case=False, na=False)]
    if search_text:
        filtered = filtered[filtered["description"].str.contains(search_text, case=False, na=False)]
    return filtered.reset_index(drop=True)


def update_transaction_category(df, transaction_id, new_category):
    updated = df.copy()
    if "transaction_id" in updated.columns:
        updated.loc[updated["transaction_id"] == transaction_id, "category"] = new_category
    else:
        updated.loc[updated.index == transaction_id, "category"] = new_category
    return updated
