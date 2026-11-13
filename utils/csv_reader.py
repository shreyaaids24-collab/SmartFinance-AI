import csv
from io import StringIO
from pathlib import Path

import pandas as pd

from utils.validators import canonical_column_name


DELIMITERS = (",", ";", "\t", "|")
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def _read_bytes(source):
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return Path(source).read_bytes()


def _decode_csv(content):
    for encoding in ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The CSV encoding is not supported. Save it as UTF-8 and try again.")


def _header_is_supported(header):
    names = {canonical_column_name(value.strip()) for value in header if value.strip()}
    return (
        "date" in names
        and "description" in names
        and bool(names.intersection({"amount", "debit", "credit"}))
    )


def _detect_delimiter(text):
    sample = "\n".join(line for line in text.splitlines() if line.strip())[:8192]
    if not sample:
        raise ValueError("The uploaded CSV is empty.")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS))
        candidates = [dialect.delimiter]
    except csv.Error:
        candidates = []

    candidates.extend(
        sorted(
            DELIMITERS,
            key=lambda delimiter: sum(line.count(delimiter) for line in sample.splitlines()),
            reverse=True,
        )
    )
    seen = set()
    for delimiter in candidates:
        if delimiter in seen:
            continue
        seen.add(delimiter)
        try:
            header = next(csv.reader([sample.splitlines()[0]], delimiter=delimiter))
        except (csv.Error, StopIteration):
            continue
        if _header_is_supported(header):
            return delimiter
    available = [
        delimiter
        for delimiter in DELIMITERS
        if any(line.count(delimiter) for line in sample.splitlines())
    ]
    if available:
        return max(
            available,
            key=lambda delimiter: sum(line.count(delimiter) for line in sample.splitlines()),
        )
    raise ValueError(
        "Could not recognize the CSV delimiter. Use comma, semicolon, tab, or pipe-separated columns."
    )


def read_transactions_csv(file_path):
    content = _read_bytes(file_path)
    if not content or not content.strip():
        raise ValueError("The uploaded CSV is empty.")
    text = _decode_csv(content)
    delimiter = _detect_delimiter(text)
    try:
        df = pd.read_csv(
            StringIO(text),
            sep=delimiter,
            engine="python",
            dtype="string",
            skip_blank_lines=True,
            skipinitialspace=True,
            on_bad_lines="error",
        )
    except (pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("The CSV does not contain a transaction header and data.") from exc
    except (pd.errors.ParserError, csv.Error) as exc:
        raise ValueError("The CSV rows could not be read. Check the delimiter and quoted fields.") from exc

    if not _header_is_supported(df.columns):
        raise ValueError(
            "Required statement columns not found: transaction date, description or narration, "
            "and amount, debit, or credit."
        )
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No transaction rows were found in the CSV.")
    return df
