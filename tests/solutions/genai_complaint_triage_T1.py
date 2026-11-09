SYSTEM_PROMPT = """You are a complaint triage assistant for a retail bank.
Classify each customer complaint into exactly one category and one priority.

Allowed categories: Billing, Fraud, Digital Banking, Loans, Deposits
Allowed priorities: High, Medium, Low

Priority rules:
- High: suspected fraud, unauthorised transactions, money lost or debited wrongly
- Medium: incorrect charges, delays or service issues with money involved but not lost
- Low: app, login or information issues with no money at risk

Reply with JSON only, no extra text, in this format:
{"category": "<one category>", "priority": "<one priority>"}"""


def build_prompt(complaint: str) -> str:
    return f"""Classify the complaint below.
Choose the category from: Billing, Fraud, Digital Banking, Loans, Deposits.
Choose the priority from: High, Medium, Low.
Return JSON only with keys "category" and "priority".

Example:
Complaint: Someone used my card abroad without my permission.
Answer: {{"category": "Fraud", "priority": "High"}}

Complaint: {complaint}
Answer:"""


print(build_prompt("My debit card was charged twice"))
