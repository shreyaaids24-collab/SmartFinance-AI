import re

from core.prompts import INTENT_PROMPT
from core.llm import get_llm


def detect_intent(question, llm=None):
    lowered = question.lower()
    keywords = [
        ("category_analysis", ["food", "shopping", "transport", "bills", "entertainment", "category"]),
        ("spending_analysis", ["how much did i spend", "total spend", "spending", "expenditure", "spent this month"]),
        ("transaction_lookup", ["transaction", "show my transactions", "from amazon", "search", "lookup", "spend on"]),
        ("monthly_comparison", ["compare this month", "last month", "monthly comparison", "vs last month", "month over month"]),
        ("savings_guidance", ["save money", "how can i save", "savings", "reduce spending", "budget"]),
        ("overspending_detection", ["overspending", "over budget", "too much on", "exceeding", "overspend"]),
        ("general_finance", ["financial help", "advice", "summary", "income", "expenses", "cash flow"]),
    ]

    for intent, patterns in keywords:
        if any(pattern in lowered for pattern in patterns):
            return {"intent": intent, "confidence": 0.9}

    if llm is not None:
        try:
            result = llm.invoke(INTENT_PROMPT.format(question=question))
            value = getattr(result, "content", str(result)).strip().lower()
            if value in {"transaction_lookup", "spending_analysis", "category_analysis", "monthly_comparison", "savings_guidance", "overspending_detection", "general_finance"}:
                return {"intent": value, "confidence": 0.85}
        except Exception:
            pass

    return {"intent": "general_finance", "confidence": 0.5}


def route_question(intent_payload):
    intent = intent_payload.get("intent", "general_finance")
    route_map = {
        "transaction_lookup": "analysis",
        "spending_analysis": "analysis",
        "category_analysis": "analysis",
        "monthly_comparison": "analysis",
        "savings_guidance": "advisor",
        "overspending_detection": "advisor",
        "general_finance": "advisor",
    }
    return route_map.get(intent, "analysis")
