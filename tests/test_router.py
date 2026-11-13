from core.intent_router import detect_intent, route_question


def test_detect_spending_intent():
    result = detect_intent("How much did I spend on food this month?")
    assert result["intent"] in {"spending_analysis", "category_analysis"}


def test_route_question_for_savings():
    route = route_question({"intent": "savings_guidance"})
    assert route == "advisor"
