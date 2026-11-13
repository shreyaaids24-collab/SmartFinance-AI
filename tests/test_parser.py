import pandas as pd

from agents.parser_agent import parse_statement


def test_csv_parser_loads_transactions():
    df = parse_statement("data/sample_transactions.csv")
    assert not df.empty
    assert {"date", "description", "amount"}.issubset(df.columns)
    assert len(df) > 0
    assert df["transaction_id"].nunique() == len(df)


def test_parser_keeps_income_and_expense_columns():
    df = parse_statement("data/sample_transactions.csv")
    assert "transaction_type" in df.columns
    assert set(df["transaction_type"].unique()).issubset({"income", "expense"})
