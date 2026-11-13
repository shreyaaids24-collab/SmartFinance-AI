import logging

from rag.vector_store import add_knowledge_documents, query_collection
from utils.formatting import format_inr


logger = logging.getLogger(__name__)

FINANCE_GUIDANCE = [
    "Food expenses can be reduced by planning meals and reviewing frequent delivery orders.",
    "Set a realistic monthly spending cap for discretionary shopping and entertainment.",
    "Transfers between a person's own accounts are not consumption expenses and should be reviewed separately.",
    "Recurring subscriptions can contribute to overspending; review them periodically and cancel unused services.",
    "Compare monthly category totals and focus on the largest flexible expense when setting a savings goal.",
    "Keep an emergency fund and avoid treating short-term spending estimates as personalized investment advice.",
]


def _summary_documents(summary):
    if not summary:
        return []
    documents = [
        {
            "source": "sanitized_financial_summary",
            "text": (
                f"Recorded financial summary: income {format_inr(summary.get('total_income', 0))}, "
                f"expenses {format_inr(summary.get('total_expense', 0))}, "
                f"net savings {format_inr(summary.get('net_savings', 0))} across "
                f"{int(summary.get('transaction_count', 0))} valid transactions."
            ),
        }
    ]
    for row in summary.get("category_breakdown", []):
        category = str(row.get("category", "Others"))
        amount = float(row.get("amount", 0))
        percentage = float(row.get("percentage", 0))
        documents.append(
            {
                "source": "sanitized_category_summary",
                "text": f"Recorded spending summary: {category} expenses total {format_inr(amount)}, "
                f"which is {percentage:.1f}% of recorded expenses.",
            }
        )
    for row in summary.get("monthly_spend", []):
        month = str(row.get("month", ""))
        amount = float(row.get("amount", 0))
        documents.append(
            {
                "source": "sanitized_monthly_summary",
                "text": f"Recorded monthly expenses for {month}: {format_inr(amount)}.",
            }
        )
    return documents


def get_relevant_context(question, summary=None):
    summary_documents = _summary_documents(summary)
    guidance_documents = [
        {"source": "finance_guidance", "text": text}
        for text in FINANCE_GUIDANCE
    ]

    try:
        add_knowledge_documents(guidance_documents)
        result = query_collection(
            question,
            n_results=3,
            where={"source": "finance_guidance"},
        )
        retrieved = result.get("documents", [[]])
        retrieved_text = retrieved[0] if retrieved and retrieved[0] else []
        retrieved_text.extend(
            document["text"]
            for document in summary_documents
            if document["text"] not in retrieved_text
        )
        if retrieved_text:
            return "\n".join(text for text in retrieved_text if text)
    except Exception:
        logger.warning("ChromaDB retrieval failed; using bundled finance guidance.", exc_info=True)
    return "\n".join(
        [*FINANCE_GUIDANCE, *(document["text"] for document in summary_documents)]
    )
