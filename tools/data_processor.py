import pandas as pd


def normalize_transaction_columns(df):
    if df is None or df.empty:
        return df
    columns = {col.lower(): col for col in df.columns}
    rename_map = {}
    for target, source in {
        "date": ["date", "transaction_date", "posted_date"],
        "description": ["description", "narration", "details", "purpose"],
        "amount": ["amount", "value", "transaction_amount"],
        "debit": ["debit", "withdrawal", "dr"],
        "credit": ["credit", "deposit", "cr"],
        "type": ["type", "transaction_type"],
        "balance": ["balance", "closing_balance"],
    }.items():
        for candidate in source:
            if candidate.lower() in columns:
                rename_map[columns[candidate.lower()]] = target
                break
    df = df.rename(columns=rename_map)
    return df


def detect_duplicate_records(df):
    if df.empty:
        return df
    duplicate_mask = df.duplicated(subset=["date", "description", "amount"], keep=False)
    df = df.copy()
    df["duplicate_flag"] = duplicate_mask
    return df


def sanitize_transaction_data(df):
    if df is None or df.empty:
        return df
    cleaned = df.copy()
    cleaned["description"] = cleaned["description"].fillna("Unknown")
    cleaned["date"] = cleaned["date"].fillna("Unknown")
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce").fillna(0)
    cleaned["category"] = cleaned.get("category", "Others")
    return cleaned
