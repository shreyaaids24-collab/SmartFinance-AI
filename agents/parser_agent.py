import os
from pathlib import Path

from utils.csv_reader import read_transactions_csv
from utils.pdf_reader import read_transactions_pdf
from utils.validators import standardize_dataframe


def parse_statement(file_path):
    path = Path(file_path)
    ext = os.path.splitext(path.name)[1].lower()
    if ext == ".csv":
        df = read_transactions_csv(path)
    elif ext == ".pdf":
        df = read_transactions_pdf(path)
    else:
        raise ValueError("Unsupported file type. Upload a PDF or CSV statement.")

    standardized = standardize_dataframe(df)
    return standardized


def validate_transactions(df):
    df = df.copy()
    invalid_rows = df[~df["is_valid"].fillna(False)].copy()
    duplicates = df[df["duplicate_flag"]].copy()
    return {
        "valid_count": int(df["is_valid"].fillna(False).sum()),
        "invalid_count": int((~df["is_valid"].fillna(False)).sum()),
        "duplicate_count": int(df["duplicate_flag"].sum()),
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicates,
    }
