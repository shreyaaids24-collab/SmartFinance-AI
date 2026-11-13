from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a careful financial analyst. Use only the supplied transaction data and verified budget guidance. If the information is missing, say so honestly. Do not fabricate financial facts. Format monetary values in Indian Rupees (₹) with Indian comma grouping and two decimal places; format dates as DD-MMM-YYYY."""

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Classify this user request into one of: transaction_lookup, spending_analysis, category_analysis, monthly_comparison, savings_guidance, overspending_detection, general_finance. Reply with only the classification name. User request: {question}",
        ),
    ]
)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Use the transaction summary and retrieved context to answer the user question. If calculations are needed, prefer exact numbers from the supplied summary. \nQuestion: {question}\nSummary: {summary}\nContext: {context}",
        ),
    ]
)

ADVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Generate practical, non-fabricated savings recommendations for the user. Base each recommendation on the actual spending data and retrieved context. \nQuestion: {question}\nSummary: {summary}\nContext: {context}",
        ),
    ]
)

PARSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Review these extracted transactions and highlight duplicate or invalid records. Return a concise validation summary.\nTransactions:\n{transactions}",
        ),
    ]
)
