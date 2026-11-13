import pandas as pd

from agents.advisor_agent import generate_advice
from agents.analysis_agent import compute_summary
from agents.categorizer_agent import categorize_transactions
from agents.parser_agent import parse_statement


def test_sample_pdf_parser_extracts_valid_transactions():
    transactions = parse_statement("data/sample_statement.pdf")
    assert len(transactions) == 4
    assert transactions["is_valid"].all()


def test_categorizer_assigns_known_categories():
    transactions = pd.DataFrame(
        [
            {"description": "Zomato dinner", "category": "Others"},
            {"description": "Metro card recharge", "category": "Others"},
            {"description": "Unrecognized merchant", "category": "Others"},
            {"description": "Course fee", "category": "Education"},
        ]
    )

    categorized = categorize_transactions(transactions)

    assert categorized["category"].tolist() == ["Food", "Transport", "Others", "Education"]


def test_categorizer_avoids_substring_and_ambiguous_category_guesses():
    transactions = pd.DataFrame(
        [
            {"description": "Workshop payment", "category": "Others"},
            {"description": "Grocery bill", "category": "Others"},
            {"description": "UPI transfer", "category": "Others"},
            {"description": "Unknown merchant 001", "category": "Others"},
            {"description": "Merchant text", "category": "Healthcare"},
        ]
    )

    categorized = categorize_transactions(transactions)

    assert categorized["category"].tolist() == [
        "Others",
        "Others",
        "Transfers",
        "Others",
        "Healthcare",
    ]


def test_analysis_ignores_invalid_rows():
    transactions = pd.DataFrame(
        [
            {"date": "2026-11-01", "description": "Pay", "amount": 1000, "transaction_type": "income", "is_valid": True},
            {"date": "2026-11-02", "description": "Food", "amount": 200, "transaction_type": "expense", "category": "Food", "is_valid": True},
            {"date": "not-a-date", "description": "Invalid", "amount": 9000, "transaction_type": "expense", "category": "Food", "is_valid": False},
        ]
    )

    summary = compute_summary(transactions)
    assert summary["total_income"] == 1000
    assert summary["total_expense"] == 200
    assert summary["transaction_count"] == 2
    assert summary["monthly_spend"] == [{"month": "2026-11", "amount": 200}]


def test_advisor_explains_largest_spending_category():
    transactions = pd.DataFrame(
        [
            {"description": "Dinner", "amount": 300, "transaction_type": "expense", "category": "Food", "is_valid": True},
            {"description": "Train", "amount": 100, "transaction_type": "expense", "category": "Transport", "is_valid": True},
        ]
    )

    advice = generate_advice(transactions)
    assert "Food" in advice[0]
    assert "75.0%" in advice[0]
