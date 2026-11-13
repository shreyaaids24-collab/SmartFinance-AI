from core.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL
from utils.formatting import format_inr, format_statement_date


def _rule_based_response(route, summary, context):
    if "advice" in summary:
        advice = summary["advice"]
        details = "\n".join(f"- {item}" for item in advice)
    elif route == "analysis":
        categories = summary.get("category_breakdown", [])
        monthly = summary.get("monthly_spend", [])
        details = (
            f"Recorded income: {format_inr(summary.get('total_income', 0))}\n"
            f"Recorded expenses: {format_inr(summary.get('total_expense', 0))}\n"
            f"Net savings: {format_inr(summary.get('net_savings', 0))}\n"
            f"Transactions included: {summary.get('transaction_count', 0)}"
        )
        if categories:
            details += "\nTop expense categories: " + ", ".join(
                f"{item['category']} {format_inr(item['amount'])}"
                for item in categories[:3]
            )
        if monthly:
            details += "\nMonthly expenses: " + ", ".join(
                f"{item['month']} {format_inr(item['amount'])}" for item in monthly
            )
        if summary.get("records"):
            details += "\nRecent matching transactions: " + "; ".join(
                f"{format_statement_date(item.get('date', ''))} "
                f"{item.get('description', '')} "
                f"{format_inr(item.get('amount', 0))} "
                f"({item.get('category', 'Others')})"
                for item in summary["records"]
            )
    else:
        details = (
            f"Recorded income: {format_inr(summary.get('total_income', 0))}; "
            f"recorded expenses: {format_inr(summary.get('total_expense', 0))}; "
            f"net savings: {format_inr(summary.get('net_savings', 0))}."
        )
    reference = context.strip() if context else "No additional reference guidance was retrieved."
    return f"Rule-based response (no AI model configured).\n{details}\n\nRetrieved reference context:\n{reference}"


def get_llm():
    provider = LLM_PROVIDER

    if provider == "openai" and OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
        except Exception:
            return None

    if provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        except Exception:
            return None

    return None


def build_llm_response(user_query, route, summary, context, llm=None):
    if llm is None:
        return _rule_based_response(route, summary, context)

    try:
        from core.prompts import ANALYSIS_PROMPT, ADVISOR_PROMPT

        prompt = ANALYSIS_PROMPT if route in {"analysis", "transaction_lookup", "category_analysis", "monthly_comparison"} else ADVISOR_PROMPT
        response = prompt.invoke({"question": user_query, "summary": summary, "context": context})
        final_response = llm.invoke(response)
        content = final_response.content if hasattr(final_response, "content") else str(final_response)
        return f"AI-generated response:\n{content}"
    except Exception as exc:
        fallback = _rule_based_response(route, summary, context)
        return f"AI request failed ({type(exc).__name__}); showing a rule-based response instead.\n{fallback}"
