import re

import pandas as pd


COLUMN_ALIASES = {
    "date": (
        "date", "transaction_date", "transaction_date_time", "posted_date", "post_date",
        "value_date", "txn_date",
    ),
    "description": (
        "description", "narration", "details", "purpose", "particulars",
        "transaction_details", "transaction_description", "remarks",
    ),
    "amount": ("amount", "value", "transaction_amount", "txn_amount"),
    "debit": ("debit", "withdrawal", "withdrawal_amount", "debit_amount", "dr"),
    "credit": ("credit", "deposit", "deposit_amount", "credit_amount", "cr"),
    "transaction_type": (
        "transaction_type", "type", "dr_cr", "credit_debit", "debit_credit",
    ),
    "reference_number": (
        "reference_number", "reference_no", "ref_number", "ref_no", "cheque_number",
        "check_number", "transaction_reference", "utr", "utr_number",
    ),
    "balance": ("balance", "closing_balance", "available_balance"),
    "category": ("category",),
}

INCOME_TYPES = {"income", "credit", "cr", "deposit", "salary", "interest"}
EXPENSE_TYPES = {"expense", "debit", "dr", "withdrawal", "payment", "purchase"}


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def canonical_column_name(name):
    normalized = normalize_column_name(name)
    if normalized in {"transaction_date", "value_date", "txn_date", "posted_date", "post_date"}:
        return "date"
    if normalized in {"transaction_date_time"}:
        return "date"
    if normalized in {"narration", "details", "purpose", "particulars", "remarks"}:
        return "description"
    if normalized in {"transaction_details", "transaction_description"}:
        return "description"
    if normalized in {"withdrawal", "withdrawal_amount", "debit_amount", "dr"} or normalized.startswith(
        "debit_"
    ):
        return "debit"
    if normalized in {"deposit", "deposit_amount", "credit_amount", "cr"} or normalized.startswith(
        "credit_"
    ):
        return "credit"
    if normalized in {"transaction_amount", "txn_amount", "value"}:
        return "amount"
    if normalized.startswith("amount_") or normalized.endswith("_amount"):
        return "amount"
    if normalized in {"type", "dr_cr", "credit_debit", "debit_credit"}:
        return "transaction_type"
    if normalized in {"reference_no", "ref_number", "ref_no", "cheque_number", "check_number"}:
        return "reference_number"
    if normalized in {"transaction_reference", "utr", "utr_number"}:
        return "reference_number"
    if normalized in {"closing_balance", "available_balance"}:
        return "balance"
    for canonical, aliases in COLUMN_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def parse_amount(value):
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text in {"-", "--", "—", "–", "N/A", "n/a"}:
        return float("nan")

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = re.sub(r"^(?:INR|Rs\.?)\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("₹", "").replace(" ", "")
    suffix = re.search(r"(CR|DR)$", text, flags=re.IGNORECASE)
    if suffix:
        text = text[: suffix.start()]
    integer_pattern = r"(?:\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})*|\d+)"
    if not re.fullmatch(rf"[+-]?{integer_pattern}(?:\.\d+)?", text):
        return float("nan")

    try:
        amount = float(text.replace(",", ""))
    except ValueError:
        return float("nan")
    return -abs(amount) if negative else amount


def _column_for(df, target):
    for candidate in COLUMN_ALIASES[target]:
        if candidate in df.columns:
            return candidate
    return None


def _numeric_column(df, name, missing_zero=False):
    if name not in df:
        return pd.Series(0.0, index=df.index)
    parsed = df[name].map(parse_amount)
    return parsed.mask(~_has_amount_value(df[name]), 0.0) if missing_zero else parsed


def _has_amount_value(series):
    text = series.astype("string").str.strip()
    return series.notna() & ~text.isin({"", "-", "--", "—", "–", "N/A", "n/a"})


def _amount_direction(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if re.search(r"CR$", text):
        return "income"
    if re.search(r"DR$", text):
        return "expense"
    if text.startswith("(") or text.startswith("-"):
        return "expense"
    if text.startswith("+"):
        return "income"
    return None


def standardize_dataframe(df):
    if df is None:
        raise ValueError("Transaction data is missing.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Transaction data must be a pandas DataFrame.")
    if df.empty and len(df.columns) == 0:
        raise ValueError("No transaction columns were found in the statement.")

    standardized = df.copy()
    standardized.columns = [normalize_column_name(column) for column in standardized.columns]
    rename_map = {}
    for column in standardized.columns:
        canonical = canonical_column_name(column)
        if canonical != column and canonical not in standardized.columns:
            rename_map[column] = canonical
    standardized = standardized.rename(columns=rename_map)

    missing_columns = []
    if "date" not in standardized:
        missing_columns.append("transaction date")
    if "description" not in standardized:
        missing_columns.append("description or narration")
    if not any(column in standardized for column in ("amount", "debit", "credit")):
        missing_columns.append("amount, debit, or credit")
    if missing_columns:
        raise ValueError("Required statement columns not found: " + ", ".join(missing_columns) + ".")

    date_text = standardized["date"].astype("string").str.strip()
    iso_dates = date_text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").fillna(False)
    parsed_dates = pd.Series(pd.NaT, index=standardized.index)
    parsed_dates.loc[iso_dates] = pd.to_datetime(
        standardized.loc[iso_dates, "date"],
        errors="coerce",
        dayfirst=False,
        format="mixed",
    )
    parsed_dates.loc[~iso_dates] = pd.to_datetime(
        standardized.loc[~iso_dates, "date"],
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )
    descriptions = standardized["description"].astype("string").str.strip()

    amount_present = "amount" in standardized
    raw_amount = _numeric_column(standardized, "amount")
    debit = _numeric_column(standardized, "debit", missing_zero=True)
    credit = _numeric_column(standardized, "credit", missing_zero=True)
    if amount_present:
        amount_supplied = _has_amount_value(standardized["amount"])
        amount_parse_valid = raw_amount.notna() | ~amount_supplied
    else:
        amount_supplied = pd.Series(False, index=standardized.index)
        amount_parse_valid = pd.Series(True, index=standardized.index)

    debit_supplied = "debit" in standardized
    credit_supplied = "credit" in standardized
    debit_has_value = (
        _has_amount_value(standardized["debit"])
        if debit_supplied
        else pd.Series(False, index=standardized.index)
    )
    credit_has_value = (
        _has_amount_value(standardized["credit"])
        if credit_supplied
        else pd.Series(False, index=standardized.index)
    )
    debit_parse_valid = debit.notna() | ~debit_has_value
    credit_parse_valid = credit.notna() | ~credit_has_value
    debit_values = debit.fillna(0.0).abs()
    credit_values = credit.fillna(0.0).abs()
    amount = raw_amount.copy()
    if not amount_present:
        amount = credit_values - debit_values
    else:
        amount.loc[amount.isna() & ~amount_supplied] = (
            credit_values - debit_values
        ).loc[amount.isna() & ~amount_supplied]
    amount = amount.abs()

    raw_type = standardized["transaction_type"] if "transaction_type" in standardized else None
    transaction_type = pd.Series(pd.NA, index=standardized.index, dtype="string")
    has_explicit_type = pd.Series(False, index=standardized.index)
    if raw_type is not None:
        normalized_type = raw_type.astype("string").str.strip().str.lower()
        transaction_type.loc[normalized_type.isin(INCOME_TYPES)] = "income"
        transaction_type.loc[normalized_type.isin(EXPENSE_TYPES)] = "expense"
        has_explicit_type = raw_type.notna() & normalized_type.ne("")
        invalid_explicit_type = has_explicit_type & ~normalized_type.isin(
            INCOME_TYPES | EXPENSE_TYPES
        )
        transaction_type.loc[invalid_explicit_type] = normalized_type.loc[
            invalid_explicit_type
        ]

    for index in standardized.index:
        if has_explicit_type.loc[index]:
            continue
        if credit_values.loc[index] > 0 and debit_values.loc[index] == 0:
            transaction_type.loc[index] = "income"
        elif debit_values.loc[index] > 0 and credit_values.loc[index] == 0:
            transaction_type.loc[index] = "expense"
        elif debit_values.loc[index] == 0 and credit_values.loc[index] == 0:
            if amount_present:
                transaction_type.loc[index] = _amount_direction(standardized.at[index, "amount"])

    standardized["date"] = parsed_dates
    standardized["description"] = descriptions
    standardized["amount"] = amount
    standardized["debit"] = debit_values
    standardized["credit"] = credit_values
    standardized["transaction_type"] = transaction_type
    standardized["category"] = standardized.get(
        "category", pd.Series("Others", index=standardized.index)
    ).fillna("Others")
    if "transaction_id" not in standardized or standardized["transaction_id"].isna().any() or standardized[
        "transaction_id"
    ].astype(str).duplicated().any():
        standardized["transaction_id"] = [f"txn_{idx + 1:04d}" for idx in range(len(standardized))]

    errors = []
    explicit_direction_conflict = pd.Series(False, index=standardized.index)
    for index, row in standardized.iterrows():
        reasons = []
        if pd.isna(parsed_dates.loc[index]):
            reasons.append("Missing or invalid transaction date")
        if pd.isna(descriptions.loc[index]) or not descriptions.loc[index]:
            reasons.append("Missing transaction description")
        if not amount_parse_valid.loc[index]:
            reasons.append("Amount is not a valid number")
        elif not amount_supplied.loc[index] and amount_present:
            if pd.isna(amount.loc[index]) or amount.loc[index] <= 0:
                reasons.append("Missing transaction amount")
        elif pd.isna(amount.loc[index]):
            reasons.append("Missing transaction amount")
        elif amount.loc[index] <= 0:
            reasons.append("Transaction amount must be greater than zero")
        if not debit_parse_valid.loc[index] or not credit_parse_valid.loc[index]:
            reasons.append("Debit or credit value is not a valid number")
        if debit_values.loc[index] > 0 and credit_values.loc[index] > 0:
            reasons.append("Both debit and credit are populated; transaction direction is ambiguous")
            explicit_direction_conflict.loc[index] = True
        if has_explicit_type.loc[index] and transaction_type.loc[index] not in {"income", "expense"}:
            reasons.append("Unrecognized transaction type")
        elif transaction_type.loc[index] not in {"income", "expense"}:
            reasons.append("Could not determine whether the transaction is income or expense")
        errors.append(reasons)

    reference = standardized.get("reference_number", pd.Series("", index=standardized.index))
    duplicate_key = pd.DataFrame(
        {
            "date": parsed_dates,
            "description": descriptions.str.lower(),
            "amount": amount,
            "transaction_type": transaction_type,
            "reference_number": reference.fillna("").astype(str).str.strip(),
        },
        index=standardized.index,
    )
    duplicate_key["reference_number"] = duplicate_key["reference_number"].where(
        duplicate_key["reference_number"].ne(""),
        "",
    )
    duplicate_mask = duplicate_key.duplicated(keep="first")
    if "duplicate_flag_override" in standardized:
        overrides = standardized["duplicate_flag_override"].map(
            lambda value: (
                pd.NA
                if pd.isna(value)
                else str(value).strip().lower() in {"true", "1", "yes"}
                if str(value).strip().lower() in {"true", "false", "1", "0", "yes", "no"}
                else pd.NA
            )
        ).astype("boolean")
        duplicate_mask = overrides.fillna(duplicate_mask).astype(bool)
    for position, duplicate in enumerate(duplicate_mask):
        if duplicate:
            errors[position].append("Flagged as a possible duplicate")

    standardized["duplicate_flag"] = duplicate_mask
    standardized["validation_errors"] = ["; ".join(reason) for reason in errors]
    standardized["is_valid"] = standardized["validation_errors"].eq("")
    return standardized
