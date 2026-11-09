"""Every task's reference solution must pass all automatic checks (catches broken rubrics or runner changes)."""
import json
from pathlib import Path

import pytest

from evaluator.grader import grade
from evaluator.runner import run_code

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = sorted((ROOT / "problems").glob("*.json"))
RULE_BASED = {"provider": "None (rule-based)", "model": "", "base_url": "", "api_key": ""}


@pytest.mark.parametrize("problem_file", PROBLEMS, ids=lambda p: p.stem)
def test_reference_solutions_pass_all_auto_checks(problem_file):
    problem = json.loads(problem_file.read_text(encoding="utf-8"))
    prior = []
    for task in problem["tasks"]:
        sol = ROOT / "tests" / "solutions" / f"{problem['id']}_{task['id']}.py"
        if not sol.exists():
            pytest.skip(f"no reference solution for {problem['id']} {task['id']}")
        code = sol.read_text(encoding="utf-8")
        checks = [{"criterion": c["criterion"], "check": c["check"]} for c in task["rubric"] if c.get("method") == "auto"]
        res = run_code(code, prior, checks, timeout=120)
        assert not res["error"], f"{task['id']} raised:\n{res['error']}"
        failed = [c for c in res["checks"] if not c["passed"]]
        assert not failed, f"{task['id']} failed auto checks: {failed}"
        g = grade(task, code, res, RULE_BASED)
        assert g["score"] >= 0.8 * g["max_score"], f"{task['id']} scored only {g['score']}/{g['max_score']}"
        prior.append(code)


def test_problem_files_are_valid():
    for f in PROBLEMS:
        p = json.loads(f.read_text(encoding="utf-8"))
        assert {"id", "title", "overview", "tasks"} <= p.keys(), f.name
        for t in p["tasks"]:
            assert t["rubric"], f"{f.name} {t['id']} has no rubric"
            for c in t["rubric"]:
                assert c["points"] > 0
                if c.get("method") == "auto":
                    compile(c["check"], "<check>", "eval")  # syntax error in a check would silently score 0
