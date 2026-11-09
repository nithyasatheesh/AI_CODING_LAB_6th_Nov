import json, re

ALLOWED = {"Billing", "Fraud", "Digital Banking", "Loans", "Deposits"}
PRIORITIES = {"High", "Medium", "Low"}
FALLBACK = {"category": "Unknown", "priority": "Low"}


def parse_response(text: str) -> dict:
    """Extract {category, priority} from an LLM reply; return a safe fallback if invalid."""
    try:
        # Grab the first {...} block, even inside ```json fences or extra text
        match = re.search(r"\{.*?\}", text, re.S)
        if not match:
            return dict(FALLBACK)
        data = json.loads(match.group(0))
        category = str(data.get("category", "")).strip()
        priority = str(data.get("priority", "")).strip().title()
        if category not in ALLOWED:
            return dict(FALLBACK)
        if priority not in PRIORITIES:
            priority = "Low"
        return {"category": category, "priority": priority}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return dict(FALLBACK)


# Tests
print(parse_response('{"category": "Fraud", "priority": "High"}'))
print(parse_response('Sure!\n```json\n{"category": "Loans", "priority": "Medium"}\n```\nHope this helps'))
print(parse_response('I am not sure'))
print(parse_response('{"category": "Pizza", "priority": "High"}'))
print(parse_response('{"category": "Billing", "priority": "urgent"}'))
print(parse_response('{broken json'))
