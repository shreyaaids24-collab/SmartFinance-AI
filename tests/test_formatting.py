import pandas as pd

from utils.formatting import (
    format_inr,
    format_statement_date,
    format_transaction_display,
)


def test_format_inr_uses_indian_grouping_and_two_decimal_places():
    assert format_inr(0) == "₹0.00"
    assert format_inr(1234.5) == "₹1,234.50"
    assert format_inr(1234567.89) == "₹12,34,567.89"
    assert format_inr(-5000) == "-₹5,000.00"
    assert format_inr(float("nan")) == "—"


def test_statement_date_and_transaction_display_are_presentation_only():
    transactions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-11-13"),
                "description": "Synthetic debit",
                "amount": 1234567.89,
                "debit": 1234567.89,
                "credit": 0.0,
            }
        ]
    )
    display = format_transaction_display(transactions)

    assert format_statement_date(transactions.loc[0, "date"]) == "13-Nov-2026"
    assert display.loc[0, "date"] == "13-Nov-2026"
    assert display.loc[0, "amount"] == "₹12,34,567.89"
    assert display.loc[0, "debit"] == "₹12,34,567.89"
    assert display.loc[0, "credit"] == "₹0.00"
    assert transactions.loc[0, "amount"] == 1234567.89
    assert transactions.loc[0, "date"] == pd.Timestamp("2026-11-13")
