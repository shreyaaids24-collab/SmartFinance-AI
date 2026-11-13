from agents.analysis_agent import compute_summary
from agents.parser_agent import parse_statement


def test_compute_summary_returns_values():
    df = parse_statement("data/sample_transactions.csv")
    summary = compute_summary(df)
    assert summary["total_income"] > 0
    assert summary["total_expense"] > 0
    assert summary["net_savings"] == summary["total_income"] - summary["total_expense"]
    assert summary["transaction_count"] == len(df)
    assert len(summary["category_breakdown"]) > 0
