import pandas as pd

from utils.validators import parse_amount, standardize_dataframe


def test_parse_amount_supports_common_bank_formats():
    assert parse_amount("₹1,234.50") == 1234.5
    assert parse_amount("₹12,34,567.50") == 1234567.5
    assert parse_amount("1,234 DR") == 1234.0
    assert parse_amount("500 CR") == 500.0
    assert parse_amount("(500.00)") == -500.0
    assert pd.isna(parse_amount("not-an-amount"))


def test_standardize_dataframe_uses_debit_credit_and_marks_bad_rows():
    source = pd.DataFrame(
        [
            {"Txn Date": "13/11/2026", "Narration": "Salary", "Deposit": "50000", "Withdrawal": ""},
            {"Txn Date": "2026-11-12", "Narration": "Groceries", "Deposit": "", "Withdrawal": "1,200"},
            {"Txn Date": "bad-date", "Narration": "Invalid amount", "Amount": "unknown"},
        ]
    )

    result = standardize_dataframe(source)

    assert result.loc[0, "transaction_type"] == "income"
    assert result.loc[0, "amount"] == 50000
    assert result.loc[0, "date"] == pd.Timestamp("2026-11-13")
    assert result.loc[1, "transaction_type"] == "expense"
    assert result.loc[1, "amount"] == 1200
    assert bool(result.loc[0, "is_valid"])
    assert bool(result.loc[1, "is_valid"])
    assert not bool(result.loc[2, "is_valid"])
    assert pd.isna(result.loc[2, "amount"])


def test_standardize_dataframe_rejects_unknown_transaction_type():
    result = standardize_dataframe(
        pd.DataFrame(
            [{"date": "2026-11-13", "description": "Fee", "amount": 25, "type": "unknown"}]
        )
    )
    assert not bool(result.loc[0, "is_valid"])


def test_unsigned_amount_without_direction_is_not_guessed_from_description():
    result = standardize_dataframe(
        pd.DataFrame(
            [
                {"date": "2026-11-13", "description": "UPI transfer", "amount": "500"},
                {"date": "2026-11-13", "description": "Mutual fund deposit", "amount": "+500"},
                {"date": "2026-11-13", "description": "Unknown incoming funds", "amount": "500 CR"},
            ]
        )
    )

    assert not bool(result.loc[0, "is_valid"])
    assert pd.isna(result.loc[0, "transaction_type"])
    assert bool(result.loc[1, "is_valid"])
    assert result.loc[1, "transaction_type"] == "income"
    assert bool(result.loc[2, "is_valid"])
    assert result.loc[2, "transaction_type"] == "income"


def test_validation_errors_explain_invalid_fields_and_clear_when_valid():
    result = standardize_dataframe(
        pd.DataFrame(
            [
                {"date": "", "description": "Missing date", "amount": "-10", "type": "DR"},
                {"date": "2026-11-13", "description": "", "amount": "-10", "type": "DR"},
                {
                    "date": "2026-11-13",
                    "description": "Invalid amount",
                    "amount": "not-money",
                    "type": "DR",
                },
                {
                    "date": "2026-11-13",
                    "description": "Invalid type",
                    "amount": "10",
                    "type": "unknown",
                },
                {"date": "2026-11-13", "description": "Valid debit", "amount": "-10", "type": "DR"},
            ]
        )
    )

    assert "transaction date" in result.loc[0, "validation_errors"].lower()
    assert "description" in result.loc[1, "validation_errors"].lower()
    assert "valid number" in result.loc[2, "validation_errors"].lower()
    assert "transaction type" in result.loc[3, "validation_errors"].lower()
    assert bool(result.loc[4, "is_valid"])
    assert result.loc[4, "validation_errors"] == ""


def test_manual_duplicate_flag_override_updates_validation_status():
    source = pd.DataFrame(
        [
            {"date": "2026-11-13", "description": "Same", "amount": "-10", "type": "DR"},
            {"date": "2026-11-13", "description": "Same", "amount": "-10", "type": "DR"},
        ]
    )
    detected = standardize_dataframe(source)
    assert detected["duplicate_flag"].tolist() == [False, True]
    assert "duplicate" in detected.loc[1, "validation_errors"].lower()

    detected.loc[1, "duplicate_flag_override"] = False
    reviewed = standardize_dataframe(detected)
    assert reviewed["duplicate_flag"].tolist() == [False, False]
    assert reviewed["is_valid"].all()
    assert reviewed["validation_errors"].eq("").all()
