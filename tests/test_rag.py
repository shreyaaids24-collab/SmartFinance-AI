from types import SimpleNamespace

from rag import vector_store
from rag.retriever import get_relevant_context
from core.llm import build_llm_response


def test_retriever_indexes_and_returns_guidance_and_sanitized_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "CHROMA_DIR", tmp_path / "chroma")
    summary = {
        "total_income": 5000.0,
        "total_expense": 5000.0,
        "net_savings": 0.0,
        "transaction_count": 8,
        "category_breakdown": [
            {"category": "Shopping", "amount": 2500.0, "percentage": 50.0}
        ],
        "monthly_spend": [{"month": "2026-11", "amount": 5000.0}],
    }

    context = get_relevant_context("How can I reduce shopping spending?", summary=summary)

    assert context
    assert "spending" in context.lower() or "shopping" in context.lower()
    assert "income ₹5,000.00, expenses ₹5,000.00" in context
    _, collection = vector_store.build_or_get_collection()
    stored = collection.get(where={"source": "finance_guidance"})
    assert len(stored["documents"]) > 0
    assert all("₹5,000.00" not in document for document in stored["documents"])


def test_rule_based_assistant_includes_metrics_and_retrieved_context():
    response = build_llm_response(
        user_query="How much did I spend?",
        route="analysis",
        summary={
            "total_income": 1000,
            "total_expense": 250,
            "net_savings": 750,
            "transaction_count": 2,
            "category_breakdown": [],
            "monthly_spend": [],
        },
        context="Review recurring subscriptions.",
        llm=None,
    )

    assert response.startswith("Rule-based response")
    assert "₹250.00" in response
    assert "Review recurring subscriptions." in response


def test_ai_assistant_prompt_contains_retrieved_context():
    class FakeLLM:
        prompt = ""

        def invoke(self, prompt):
            self.prompt = prompt.to_string()
            return SimpleNamespace(content="Use the provided spending summary.")

    llm = FakeLLM()
    response = build_llm_response(
        user_query="How can I save?",
        route="advisor",
        summary={"advice": ["Review recurring costs."]},
        context="Retrieved advice: review subscriptions.",
        llm=llm,
    )

    assert response.startswith("AI-generated response:")
    assert "Retrieved advice: review subscriptions." in llm.prompt
