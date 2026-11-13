import csv
import re
from pathlib import Path

import pandas as pd
import pdfplumber

from utils.validators import canonical_column_name, normalize_column_name, parse_amount


DATE_PATTERN = (
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
)
DATE_PREFIX = re.compile(rf"^\s*({DATE_PATTERN})\b", re.IGNORECASE)
AMOUNT_TOKEN = re.compile(
    r"(?<![\w/])(?:₹|INR|Rs\.?)?\s*\(?[+-]?(?:\d{1,3}(?:,\d{2})*,\d{3}|\d+)"
    r"(?:\.\d+)?\)?(?:\s*(?:CR|DR))?(?![\w/])",
    re.IGNORECASE,
)


def _canonical_headers(cells):
    date_columns = [
        index
        for index, cell in enumerate(cells)
        if canonical_column_name(cell or "") == "date"
    ]
    preferred_date = next(
        (
            index
            for index in date_columns
            if normalize_column_name(cells[index] or "") not in {"value_date"}
        ),
        date_columns[0] if date_columns else None,
    )
    headers = []
    for index, cell in enumerate(cells):
        canonical = canonical_column_name(cell or "")
        if canonical == "date" and index != preferred_date:
            canonical = "value_date"
        headers.append(canonical)
    has_date = "date" in headers
    has_description = "description" in headers
    has_amount = bool(set(headers).intersection({"amount", "debit", "credit"}))
    return headers if has_date and has_description and has_amount else None


def _table_rows(table):
    if not table:
        return []
    for header_index, header_row in enumerate(table[:4]):
        headers = _canonical_headers(header_row)
        if headers is None:
            continue
        rows = []
        for values in table[header_index + 1 :]:
            if not values or all(value is None or not str(value).strip() for value in values):
                continue
            row = {
                headers[index]: value
                for index, value in enumerate(values[: len(headers)])
                if index < len(headers) and headers[index]
            }
            if len(row) >= 2:
                rows.append(row)
        return rows
    return []


def _parse_delimited_lines(lines):
    rows = []
    active_headers = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        delimiter = next((candidate for candidate in (",", "\t", ";", "|") if candidate in stripped), None)
        if delimiter:
            try:
                cells = next(csv.reader([stripped], delimiter=delimiter, skipinitialspace=True))
            except csv.Error:
                continue
            headers = _canonical_headers(cells)
            if headers:
                active_headers = headers
                continue
            if active_headers and len(cells) >= len(active_headers):
                row = {
                    active_headers[index]: value.strip()
                    for index, value in enumerate(cells[: len(active_headers)])
                    if active_headers[index]
                }
                rows.append(row)
                continue

        cells = re.split(r"\s{2,}", stripped)
        headers = _canonical_headers(cells)
        if headers:
            active_headers = headers
            continue
        if active_headers and len(cells) >= len(active_headers):
            rows.append(
                {
                    active_headers[index]: value.strip()
                    for index, value in enumerate(cells[: len(active_headers)])
                    if active_headers[index]
                }
            )
    return rows


def _suffix_amounts(text):
    matches = list(AMOUNT_TOKEN.finditer(text))
    if not matches:
        return [], text
    suffix_matches = []
    expected_end = len(text.rstrip())
    for match in reversed(matches):
        if text[match.end() : expected_end].strip():
            break
        suffix_matches.append(match)
        expected_end = match.start()
    suffix_matches.reverse()
    if not suffix_matches:
        return [], text
    values = [match.group(0).strip() for match in suffix_matches]
    description = text[: suffix_matches[0].start()].strip(" ,;\t")
    return values, description


def _parse_freeform_line(line):
    date_match = DATE_PREFIX.match(line)
    if not date_match:
        return None
    date = date_match.group(1).strip()
    remainder = line[date_match.end() :].strip(" ,;\t")

    second_date_match = DATE_PREFIX.match(remainder)
    if second_date_match:
        remainder = remainder[second_date_match.end() :].strip(" ,;\t")

    values, description = _suffix_amounts(remainder)
    if not values or not description:
        return None
    if len(values) >= 3:
        return {
            "date": date,
            "description": description,
            "debit": values[-3],
            "credit": values[-2],
            "balance": values[-1],
        }
    if len(values) == 2:
        return {
            "date": date,
            "description": description,
            "debit": values[-2],
            "credit": values[-1],
        }
    return {"date": date, "description": description, "amount": values[0]}


def _extract_positioned_transactions(page):
    words = page.extract_words(keep_blank_chars=False)
    if not words:
        return []

    def centers_for(label):
        return [
            (word["x0"] + word["x1"]) / 2
            for word in words
            if word["text"].strip().lower() == label
        ]

    date_header_words = [
        word for word in words if word["text"].strip().lower() == "date"
    ]
    debit_centers = centers_for("withdrawal") or centers_for("debit")
    credit_centers = centers_for("deposit") or centers_for("credit")
    balance_centers = centers_for("balance")
    if not date_header_words or not debit_centers or not credit_centers:
        return []

    related_date_headers = [
        header
        for header in date_header_words
        if any(
            word["text"].strip().lower() in {"transaction", "value", "posted"}
            and word["x0"] < header["x0"]
            and abs(word["top"] - header["top"]) <= 25
            for word in words
        )
    ]
    date_header = min(
        related_date_headers or date_header_words,
        key=lambda word: word["top"],
    )
    date_header_center = (date_header["x0"] + date_header["x1"]) / 2
    debit_center = debit_centers[0]
    credit_center = credit_centers[0]
    balance_center = balance_centers[0] if balance_centers else float("inf")
    amount_header_left_edges = [
        word["x0"]
        for word in words
        if word["text"].strip().lower() == "amount"
    ]
    description_right_edge = (
        min(amount_header_left_edges) - 3
        if amount_header_left_edges
        else min(debit_center, credit_center) - 8
    )
    description_left_edge = date_header_center + (
        min(debit_center, credit_center) - date_header_center
    ) * 0.25
    date_tolerance = max(
        abs(min(debit_center, credit_center) - date_header_center) * 0.12,
        10,
    )
    date_words = [
        word
        for word in words
        if DATE_PREFIX.fullmatch(word["text"].strip())
        and abs((word["x0"] + word["x1"]) / 2 - date_header_center) <= date_tolerance
    ]
    date_words.sort(key=lambda word: word["top"])
    if not date_words:
        return []

    row_spacings = [
        following["top"] - current["top"]
        for current, following in zip(date_words, date_words[1:])
        if following["top"] > current["top"]
    ]
    default_row_span = (
        float(pd.Series(row_spacings).median())
        if row_spacings
        else page.height * 0.1
    )
    parsed_rows = []
    for row_index, date_word in enumerate(date_words):
        row_top = date_word["top"]
        row_bottom = (
            date_words[row_index + 1]["top"]
            if row_index + 1 < len(date_words)
            else row_top + default_row_span
        )
        same_line = [
            word for word in words if abs(word["top"] - row_top) <= 2 and word is not date_word
        ]
        numeric_words = []
        for word in same_line:
            value = parse_amount(word["text"].strip())
            if not pd.isna(value) and not DATE_PREFIX.fullmatch(word["text"].strip()):
                numeric_words.append(word)

        def find_column_value(center, competing_centers=()):
            if not numeric_words:
                return None
            candidate = min(
                numeric_words,
                key=lambda word: abs((word["x0"] + word["x1"]) / 2 - center),
            )
            candidate_center = (candidate["x0"] + candidate["x1"]) / 2
            distance = abs(candidate_center - center)
            max_distance = abs(debit_center - credit_center) / 2 + 5
            if distance > max_distance or any(
                abs(candidate_center - other_center) < distance
                for other_center in competing_centers
            ):
                return None
            return candidate["text"]

        debit_value = find_column_value(debit_center, (credit_center, balance_center))
        credit_value = find_column_value(credit_center, (debit_center, balance_center))
        if debit_value is None and credit_value is None:
            continue

        description_words = [
            word
            for word in words
            if row_top < word["top"] < min(row_bottom, page.height - 20)
            and word["x0"] >= description_left_edge
            and word["x1"] <= description_right_edge
        ]
        description_lines = {}
        for word in description_words:
            description_lines.setdefault(round(word["top"]), []).append(word)
        description = " ".join(
            " ".join(word["text"] for word in sorted(line_words, key=lambda item: item["x0"]))
            for _, line_words in sorted(description_lines.items())
        ).strip()

        row = {
            "date": date_word["text"],
            "description": description,
            "debit": debit_value or "",
            "credit": credit_value or "",
        }
        balance_value = (
            find_column_value(balance_center, (debit_center, credit_center))
            if balance_centers
            else None
        )
        if balance_value is not None:
            row["balance"] = balance_value
        parsed_rows.append(row)
    return parsed_rows


def parse_pdf_text_lines(lines):
    """Parse synthetic/extracted PDF text with delimited, aligned, or free-form rows."""
    lines = list(lines)
    rows = _parse_delimited_lines(lines)
    if rows:
        return rows
    return [row for line in lines if (row := _parse_freeform_line(line)) is not None]


def extract_rows_from_pdf_pdfplumber(file_path):
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("The uploaded PDF is empty.")

    rows = []
    saw_text = False
    saw_images = False
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                saw_text = saw_text or bool(page_text.strip())
                saw_images = saw_images or bool(page.images)
                page_rows = []
                for table in page.extract_tables() or []:
                    page_rows.extend(_table_rows(table))
                if not page_rows:
                    page_rows = parse_pdf_text_lines(page_text.splitlines())
                if not page_rows:
                    page_rows = _extract_positioned_transactions(page)
                rows.extend(page_rows)
    except Exception as exc:
        if "password" in str(exc).lower() or "password" in type(exc).__name__.lower():
            raise ValueError("This PDF is password-protected. Remove its password and upload it again.") from exc
        raise ValueError("The PDF could not be opened. It may be corrupted or not a valid PDF.") from exc

    if not saw_text and saw_images:
        raise ValueError("This PDF appears to be scanned/image-only. OCR is required before it can be parsed.")
    if not rows:
        if not saw_text:
            raise ValueError("No readable transaction text was found in the PDF.")
        raise ValueError(
            "No transaction rows could be recognized. This text-based PDF layout may not be supported."
        )
    return rows


def read_transactions_pdf(file_path):
    return pd.DataFrame(extract_rows_from_pdf_pdfplumber(file_path))
