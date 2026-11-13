from collections import Counter

import pandas as pd
import pdfplumber
import pytest

from agents.parser_agent import parse_statement
from utils import pdf_reader
from utils.csv_reader import read_transactions_csv
from utils.pdf_reader import parse_pdf_text_lines, read_transactions_pdf


def test_csv_reader_detects_delimiters_headers_and_quoted_commas(tmp_path):
    examples = [
        (
            "comma.csv",
            "Transaction Date,Narration,Amount,Type\n"
            '13/11/2026,"Cafe, Central","+₹1,234.50",CR\n'
            "14/11/2026,Book shop,-500.00,DR\n",
        ),
        (
            "semi.csv",
            "Posted Date;Particulars;Withdrawal Amount;Deposit Amount;Reference No.\n"
            "13-11-2026;Grocery;1,234.50;-;ref-001\n"
            "14-11-2026;Salary;-;50,000.00;ref-002\n",
        ),
        (
            "tab.csv",
            "Value Date\tDetails\tAmount\tTransaction Type\n"
            "2026-11-13\tMetro\t₹75.00\tExpense\n",
        ),
    ]

    for filename, content in examples:
        path = tmp_path / filename
        path.write_text(content, encoding="utf-8")
        raw = read_transactions_csv(path)
        assert not raw.empty
        normalized = parse_statement(path)
        assert normalized["is_valid"].all()

    parsed = parse_statement(tmp_path / "comma.csv")
    assert parsed.loc[0, "description"] == "Cafe, Central"
    assert parsed.loc[0, "transaction_type"] == "income"
    assert parsed.loc[0, "amount"] == 1234.5
    assert parsed.loc[1, "transaction_type"] == "expense"


def test_csv_reader_supports_cp1252_and_ignores_blank_rows(tmp_path):
    path = tmp_path / "bank.csv"
    path.write_bytes(
        "Date;Description;Amount;Type\n13/11/2026;Café payment;-250;DR\n\n".encode("cp1252")
    )
    parsed = parse_statement(path)
    assert len(parsed) == 1
    assert parsed.loc[0, "description"] == "Café payment"
    assert parsed.loc[0, "transaction_type"] == "expense"


def test_csv_reader_reports_empty_unsupported_and_missing_columns(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_transactions_csv(empty)

    unsupported = tmp_path / "unsupported.csv"
    unsupported.write_text("hello|world\n1|2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Required statement columns not found"):
        read_transactions_csv(unsupported)

    missing_columns = tmp_path / "missing.csv"
    missing_columns.write_text("Date,Description,Merchant\n2026-11-13,Payment,Shop\n")
    with pytest.raises(ValueError, match="Required statement columns not found"):
        parse_statement(missing_columns)


def test_pdf_text_parser_supports_delimited_and_aligned_tables():
    delimited = parse_pdf_text_lines(
        [
            "Transaction Date,Value Date,Description,Reference Number,Withdrawal,Deposit,Balance",
            '13.11.2026,13.11.2026,UPI grocery,ref-1,"1,234.50",-,"10,000.00"',
        ]
    )
    assert len(delimited) == 1
    assert delimited[0]["date"] == "13.11.2026"
    assert delimited[0]["debit"] == "1,234.50"

    quoted = parse_pdf_text_lines(
        [
            "Date  Description  Withdrawal  Deposit  Balance",
            "13/11/2026  Groceries  ₹1,234.50  -  ₹10,000.00",
        ]
    )
    assert len(quoted) == 1
    assert quoted[0]["description"] == "Groceries"
    assert quoted[0]["debit"] == "₹1,234.50"
    assert quoted[0]["credit"] == "-"

    comma_text = parse_pdf_text_lines(
        [
            "Transaction Date,Description,Amount,Type",
            '2026-11-13,"Cafe, Central","-₹75.00",DR',
        ]
    )
    assert comma_text[0]["description"] == "Cafe, Central"
    assert comma_text[0]["amount"] == "-₹75.00"


def test_pdf_positioned_parser_handles_multiline_descriptions_and_columns():
    class FakePage:
        height = 800

        def extract_words(self, keep_blank_chars=False):
            return [
                {"text": "Transaction", "x0": 60, "x1": 108, "top": 30},
                {"text": "Withdrawal", "x0": 399, "x1": 447, "top": 30},
                {"text": "Deposit", "x0": 474, "x1": 504, "top": 30},
                {"text": "Balance", "x0": 532, "x1": 564, "top": 30},
                {"text": "Date", "x0": 74, "x1": 94, "top": 40},
                {"text": "Amount", "x0": 396, "x1": 427, "top": 40},
                {"text": "Amount", "x0": 462, "x1": 493, "top": 40},
                {"text": "1", "x0": 30, "x1": 35, "top": 60},
                {"text": "13.11.2026", "x0": 61, "x1": 104, "top": 60},
                {"text": "1234.50", "x0": 422, "x1": 453, "top": 60},
                {"text": "10000.00", "x0": 526, "x1": 571, "top": 60},
                {"text": "UPI", "x0": 192, "x1": 215, "top": 65},
                {"text": "payment", "x0": 220, "x1": 270, "top": 65},
                {"text": "at", "x0": 192, "x1": 205, "top": 75},
                {"text": "merchant", "x0": 210, "x1": 270, "top": 75},
            ]

    rows = pdf_reader._extract_positioned_transactions(FakePage())
    assert rows == [
        {
            "date": "13.11.2026",
            "description": "UPI payment at merchant",
            "debit": "1234.50",
            "credit": "",
            "balance": "10000.00",
        }
    ]


def test_pdf_reader_handles_multiple_pages_and_clear_error_conditions(tmp_path, monkeypatch):
    class FakePage:
        images = []
        height = 800

        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

        def extract_tables(self):
            return []

        def extract_words(self, keep_blank_chars=False):
            return []

    class FakePDF:
        def __init__(self, pages):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    original_pdf_open = pdfplumber.open
    path = tmp_path / "multi.pdf"
    path.write_bytes(b"synthetic pdf payload")
    monkeypatch.setattr(
        pdf_reader.pdfplumber,
        "open",
        lambda _path: FakePDF(
            [
                FakePage(
                    "Date,Description,Debit,Credit,Balance\n"
                    "13/11/2026,Food order,250.00,,1000.00"
                ),
                FakePage(
                    "Date,Description,Debit,Credit,Balance\n"
                    "14/11/2026,Salary,,5000.00,6000.00"
                ),
            ]
        ),
    )
    parsed = parse_statement(path)
    assert len(parsed) == 2
    assert parsed["is_valid"].all()
    assert Counter(parsed["transaction_type"]) == {"expense": 1, "income": 1}

    class TablePage(FakePage):
        def extract_tables(self):
            return [[
                ["Value Date", "Reference Number", "Particulars", "Credit", "Balance", "Debit"],
                ["13.11.2026", "ref-1", "Interest credit", "₹25.00", "1000.00", "-"],
            ]]

    monkeypatch.setattr(
        pdf_reader.pdfplumber,
        "open",
        lambda _path: FakePDF([TablePage("")]),
    )
    table_path = tmp_path / "table.pdf"
    table_path.write_bytes(b"synthetic pdf payload")
    table_parsed = parse_statement(table_path)
    assert len(table_parsed) == 1
    assert bool(table_parsed.loc[0, "is_valid"])
    assert table_parsed.loc[0, "transaction_type"] == "income"

    empty_path = tmp_path / "empty.pdf"
    empty_path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_transactions_pdf(empty_path)

    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"not a pdf")
    monkeypatch.setattr(pdf_reader.pdfplumber, "open", original_pdf_open)
    with pytest.raises(ValueError, match="corrupted"):
        read_transactions_pdf(corrupt_path)

    monkeypatch.setattr(
        pdf_reader.pdfplumber,
        "open",
        lambda _path: (_ for _ in ()).throw(Exception("password required")),
    )
    with pytest.raises(ValueError, match="password-protected"):
        read_transactions_pdf(path)

    class ScannedPage(FakePage):
        images = [object()]

        def extract_text(self):
            return ""

    monkeypatch.setattr(
        pdf_reader.pdfplumber,
        "open",
        lambda _path: FakePDF([ScannedPage("")]),
    )
    with pytest.raises(ValueError, match="OCR"):
        read_transactions_pdf(path)


def test_duplicate_csv_rows_are_flagged_and_excluded_from_totals(tmp_path):
    path = tmp_path / "duplicates.csv"
    path.write_text(
        "Date,Description,Amount,Type,Reference Number\n"
        "13/11/2026,Same debit,-100,DR,ref-1\n"
        "13/11/2026,Same debit,-100,DR,ref-1\n"
        "13/11/2026,Same debit,-100,DR,ref-2\n",
        encoding="utf-8",
    )
    parsed = parse_statement(path)
    assert parsed["duplicate_flag"].tolist() == [False, True, False]
    assert parsed["is_valid"].tolist() == [True, False, True]


def test_pdf_text_parser_handles_date_periods_and_month_names():
    rows = parse_pdf_text_lines(
        [
            "Date  Description  Debit  Credit  Balance",
            "13.11.2026  Grocery store  250.00  -  1000.00",
        ]
    )
    assert rows
