import re

import pandas as pd

CATEGORY_KEYWORDS = {
    "Food": ["zomato", "swiggy", "restaurant", "cafe", "grocer", "grocery", "milk", "food", "coffee"],
    "Shopping": ["amazon", "flipkart", "myntra", "shop", "retail", "marketplace"],
    "Transport": ["uber", "ola", "metro", "fuel", "petrol", "diesel", "bus", "train"],
    "Bills": ["electricity", "water", "internet", "phone", "utility", "bill"],
    "Entertainment": ["netflix", "spotify", "movie", "cinema", "games", "subscription"],
    "Healthcare": ["pharmacy", "apollo", "clinic", "hospital", "medical", "doctor"],
    "Education": ["udemy", "coursera", "college", "school", "fees", "library", "course"],
    "Salary": ["salary", "payroll", "employer", "income", "freelance"],
    "Transfers": ["transfer", "wallet transfer", "upi transfer", "bank transfer"],
    "Investments": ["mutual fund", "sip", "investment", "stock", "brokerage"],
    "Rent": ["rent"],
}
VALID_CATEGORIES = {
    category.lower(): category for category in (*CATEGORY_KEYWORDS, "Others")
}


def _category_matches(description):
    if pd.isna(description):
        return set()
    normalized = str(description).lower()
    matches = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            escaped_keyword = re.escape(keyword).replace("\\ ", r"\s+")
            pattern = rf"(?<!\w){escaped_keyword}(?!\w)"
            if re.search(pattern, normalized):
                matches.add(category)
                break
    return matches


def categorize_transactions(df, llm=None):
    if df is None or df.empty:
        return df

    categorized = df.copy()
    existing_categories = categorized.get(
        "category", pd.Series("Others", index=categorized.index)
    )
    normalized_categories = (
        existing_categories.fillna("Others")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(VALID_CATEGORIES)
        .fillna("Others")
    )
    categorized["category"] = normalized_categories
    for index, row in categorized.iterrows():
        if normalized_categories.loc[index] != "Others":
            continue
        matches = _category_matches(row.get("description", ""))
        if len(matches) == 1:
            categorized.at[index, "category"] = matches.pop()

    if llm is not None:
        try:
            for index, row in categorized.iterrows():
                if row["category"] == "Others":
                    prompt = (
                        f"Classify '{row['description']}' only when clearly supported by the merchant or "
                        "description. If unclear or ambiguous, reply Others. Choose one of: "
                        "Food, Shopping, Transport, Bills, Entertainment, Healthcare, Education, Rent, "
                        "Salary, Transfers, Investments, Others."
                    )
                    result = llm.invoke(prompt)
                    content = getattr(result, "content", str(result)).strip()
                    category = VALID_CATEGORIES.get(content.lower())
                    if category:
                        categorized.at[index, "category"] = category
        except Exception:
            return categorized
    return categorized
