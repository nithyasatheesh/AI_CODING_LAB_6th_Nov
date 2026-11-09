import pandas as pd

complaints = pd.read_csv("data/complaints.csv")

rows = []
for _, r in complaints.iterrows():
    reply = ask_llm(build_prompt(r["complaint_text"]), system=SYSTEM_PROMPT)
    parsed = parse_response(reply)
    rows.append({
        "complaint_text": r["complaint_text"],
        "predicted_category": parsed["category"],
        "predicted_priority": parsed["priority"],
        "true_category": r["true_category"],
        "true_priority": r["true_priority"],
    })

predictions = pd.DataFrame(rows)
accuracy = (predictions["predicted_category"] == predictions["true_category"]).mean()

print(predictions[["predicted_category", "true_category", "predicted_priority", "true_priority"]])
print(f"\nCategory accuracy: {accuracy:.0%}")

wrong = predictions[predictions["predicted_category"] != predictions["true_category"]]
print("\nMisclassified complaints:")
print(wrong[["complaint_text", "predicted_category", "true_category"]] if len(wrong) else "None")

print("\nProduction monitoring: send every High priority and every 'Unknown' to a human agent; "
      "log prompts, replies and latency; mask account numbers and names before calling the LLM; "
      "re-check accuracy weekly on a labelled sample to catch drift.")
