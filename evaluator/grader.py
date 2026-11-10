"""
Rubric grader.

Two kinds of rubric criteria:
  * method = "auto" -> deterministic Python check; full points if it passes, else 0.
  * method = "llm"  -> scored by an LLM judge (OpenAI, Groq, Ollama or any OpenAI-compatible
                       endpoint). If no LLM is configured, a transparent rule-based estimate is used.
"""
from __future__ import annotations

import json
import re

JUDGE_SYSTEM = """You are a strict but fair evaluator for a corporate Python / Data Science / GenAI hands-on assessment.
Score ONLY the rubric criteria given to you. Base your judgement on the code, its output and any errors.
Never award points for work that is not present. Partial credit is allowed (any number between 0 and the max, one decimal).
Respond with valid JSON only."""

JUDGE_TEMPLATE = """## Task
{instruction}

## Rubric criteria to score (LLM-judged)
{criteria}

## Automatic checks already run (for context)
{auto}

## Participant code
```python
{code}
```

## Program output (truncated)
```
{stdout}
```

## Runtime error
{error}

Return JSON exactly in this shape:
{{
  "criteria": [{{"criterion": "<name exactly as given>", "score": <number>, "justification": "<one or two sentences>"}}],
  "overall_feedback": "<3-4 sentences of constructive feedback addressed to the participant>",
  "improvements": ["<specific improvement>", "<specific improvement>"]
}}"""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}


def _llm_judge(task: dict, llm_criteria: list[dict], code: str, run: dict, auto_rows: list[dict],
               cfg: dict) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"] or "ollama", base_url=cfg.get("base_url") or None)
    prompt = JUDGE_TEMPLATE.format(
        instruction=task["instruction"],
        criteria="\n".join(f"- {c['criterion']} (max {c['points']} pts): {c.get('guide', '')}" for c in llm_criteria),
        auto="\n".join(f"- {r['criterion']}: {'PASS' if r['score'] else 'FAIL'}" for r in auto_rows) or "None",
        code=code[:6000],
        stdout=(run.get("stdout") or "")[:3000] or "(no output)",
        error=run.get("error") or "None",
    )
    kwargs = dict(model=cfg["model"], temperature=0,
                  messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}])
    try:
        resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:  # some open-source servers don't support response_format
        resp = client.chat.completions.create(**kwargs)
    return _extract_json(resp.choices[0].message.content)


def _own_code(task: dict, code: str) -> str:
    """The participant's code without the unchanged starter-code lines (so hints in comments earn nothing)."""
    starter = {ln.strip() for ln in task.get("starter_code", "").splitlines() if ln.strip()}
    return "\n".join(ln for ln in code.splitlines() if ln.strip() not in starter)


def _rule_based(task: dict, llm_criteria: list[dict], code: str, run: dict, auto_rows: list[dict]) -> dict:
    """Offline scoring when no LLM judge is configured.

    A criterion with "evidence" (a list of regular expressions) earns points in proportion to how many patterns are
    found in the participant's own code and printed output; a runtime error halves it. Criteria without evidence
    get half marks and are flagged as not verified.
    """
    ran_ok = not run.get("error")
    text = _own_code(task, code) + "\n" + (run.get("stdout") or "")
    rows, missing = [], []
    for c in llm_criteria:
        pats = c.get("evidence") or []
        if not code.strip():
            score, why = 0.0, "No code submitted."
        elif pats:
            hits = [bool(re.search(p, text, re.I | re.S)) for p in pats]
            ratio = sum(hits) / len(hits)
            score = c["points"] * ratio * (1.0 if ran_ok else 0.5)
            if ratio == 1:
                why = "Evidence found in your code/output." + ("" if ran_ok else " Halved: the code raised an error.")
            else:
                why = "Not found in your code or output." if ratio == 0 else "Partly found in your code/output."
                missing.append(c["criterion"])
        else:
            score, why = c["points"] * (0.5 if ran_ok else 0.25), "Not verified offline (no AI judge configured)."
        rows.append({"criterion": c["criterion"], "score": round(score, 1), "justification": why})
    failed_auto = [r["criterion"] for r in auto_rows if not r["score"]]
    return {
        "criteria": rows,
        "overall_feedback": ("Code ran successfully." if ran_ok else "Your code raised an error — fix it and compare with the task requirements.")
                            + " Scored offline from the evidence in your code and output; an AI judge gives detailed feedback.",
        "improvements": (failed_auto + missing)[:4],
    }


def grade(task: dict, code: str, run: dict, cfg: dict) -> dict:
    rubric = task["rubric"]
    check_map = {c["criterion"]: c for c in run.get("checks", [])}

    auto_rows = []
    for c in rubric:
        if c.get("method") == "auto":
            chk = check_map.get(c["criterion"], {"passed": False, "error": "not run"})
            auto_rows.append({"criterion": c["criterion"], "method": "Auto check", "max": c["points"],
                              "score": c["points"] if chk["passed"] else 0,
                              "justification": "Passed ✔" if chk["passed"] else f"Failed ✘ {chk.get('error', '')}".strip()})

    llm_criteria = [c for c in rubric if c.get("method") != "auto"]
    judge, judge_label = {}, "Rule-based estimate"
    if llm_criteria:
        if cfg.get("provider") != "None (rule-based)" and (cfg.get("api_key") or cfg.get("provider") == "Ollama (local)"):
            try:
                judge = _llm_judge(task, llm_criteria, code, run, auto_rows, cfg)
                judge_label = f"LLM judge ({cfg['model']})"
            except Exception as e:
                judge = _rule_based(task, llm_criteria, code, run, auto_rows)
                judge["overall_feedback"] = f"⚠️ LLM judge unavailable ({type(e).__name__}). " + judge["overall_feedback"]
        else:
            judge = _rule_based(task, llm_criteria, code, run, auto_rows)

    by_name = {j.get("criterion"): j for j in judge.get("criteria", [])}
    llm_rows = []
    for c in llm_criteria:
        j = by_name.get(c["criterion"], {})
        try:
            s = float(j.get("score", 0))
        except (TypeError, ValueError):
            s = 0.0
        llm_rows.append({"criterion": c["criterion"], "method": judge_label, "max": c["points"],
                         "score": round(max(0.0, min(float(c["points"]), s)), 1),
                         "justification": j.get("justification", "Not scored")})

    rows = auto_rows + llm_rows
    order = {c["criterion"]: i for i, c in enumerate(rubric)}
    rows.sort(key=lambda r: order.get(r["criterion"], 99))
    score = round(sum(r["score"] for r in rows), 1)
    max_score = sum(c["points"] for c in rubric)
    return {"rows": rows, "score": score, "max_score": max_score,
            "feedback": judge.get("overall_feedback", "All criteria are automatically checked." if not llm_criteria else ""),
            "improvements": judge.get("improvements", [])}
