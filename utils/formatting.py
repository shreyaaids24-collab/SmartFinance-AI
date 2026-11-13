import math
from datetime import date, datetime

import pandas as pd


def format_inr(value):
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(amount):
        return "—"

    formatted = f"{abs(amount):.2f}"
    integer, fraction = formatted.split(".")
    if len(integer) > 3:
        last_three = integer[-3:]
        leading = integer[:-3]
        groups = []
        while leading:
            groups.insert(0, leading[-2:])
            leading = leading[:-2]
        integer = ",".join([*groups, last_three])
    sign = "-" if amount < 0 else ""
    return f"{sign}₹{integer}.{fraction}"


def format_statement_date(value):
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime("%d-%b-%Y")
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%d-%b-%Y")


def format_transaction_display(df):
    display = df.copy()
    if "date" in display:
        display["date"] = display["date"].map(format_statement_date)
    for column in ("amount", "debit", "credit", "balance"):
        if column in display:
            display[column] = display[column].map(format_inr)
    return display


def format_summary_display(df):
    display = df.copy()
    for column in ("amount", "total_debit", "total_credit"):
        if column in display:
            display[column] = display[column].map(format_inr)
    return display
