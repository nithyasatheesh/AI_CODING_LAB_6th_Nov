"""
Sandboxed code runner.

Each participant submission runs in a separate Python subprocess with a timeout.
Code from previously submitted tasks is replayed first (silently), so variables
like `df`, `X_train` or `model` carry over from task to task — exactly like a notebook.

After the participant's code runs, the rubric's automatic checks (Python expressions)
are evaluated inside the same namespace.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "<<<EVAL_RESULT>>>"
MAX_STDOUT = 8000

HARNESS = r'''
import json, sys, io, os, contextlib, traceback, re
import warnings; warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

payload = json.load(open(sys.argv[1], encoding="utf-8"))
figdir = payload["figdir"]
figs = []
_replaying = [False]

def _capture_figures(*args, **kwargs):
    if _replaying[0]:
        plt.close("all")
        return
    for num in plt.get_fignums():
        path = os.path.join(figdir, f"fig_{len(figs)}.png")
        plt.figure(num).savefig(path, bbox_inches="tight", dpi=90)
        figs.append(path)
    plt.close("all")

plt.show = _capture_figures

# ---------- LLM helper available to participants (GenAI tasks) ----------
def ask_llm(prompt, system=None, temperature=0.0):
    """Call the configured LLM. Falls back to a keyword-based mock when no API key is set."""
    key = os.environ.get("PARTICIPANT_LLM_API_KEY", "")
    if key:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=os.environ.get("PARTICIPANT_LLM_BASE_URL") or None)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = client.chat.completions.create(model=os.environ.get("PARTICIPANT_LLM_MODEL", "gpt-4o-mini"),
                                           messages=msgs, temperature=temperature)
        return r.choices[0].message.content
    # ---- mock LLM (deterministic, so training works offline) ----
    # Use the LAST "Complaint: ..." line, so category lists and few-shot examples are ignored
    found = re.findall(r'complaint\s*(?:text)?\s*[:=]\s*["\']?([^\n]+)', prompt, re.I)
    found = [f for f in found if len(f.strip()) > 3]
    body = (found[-1] if found else prompt.strip().splitlines()[-1] if prompt.strip() else "").lower()
    rules = [("Fraud", ["without my knowledge", "otp", "unknown", "fraud", "unauthori"]),
             ("Loans", ["loan", "emi"]),
             ("Digital Banking", ["app", "net banking", "password", "login"]),
             ("Deposits", ["fd", "deposit", "interest"]),
             ("Billing", ["charged", "fee", "reverse"])]
    cat = next((c for c, kws in rules if any(k in body for k in kws)), "Billing")
    pri = "High" if cat == "Fraud" or "twice this month" in body else ("Low" if cat == "Digital Banking" else "Medium")
    return "Sure! Here is the classification:\n```json\n" + json.dumps({"category": cat, "priority": pri}) + "\n```"

# ---------- helpers that rubric checks can call (flexible, so different-but-correct answers pass) ----------
def _close(value, truth, tol=0.006):
    """True if value matches truth, allowing rounding and 0-1 vs 0-100 (percent) scales."""
    try:
        v, t = float(value), float(truth)
    except Exception:
        return False
    return abs(v - t) <= tol or abs(v - 100 * t) <= 100 * tol

def _model(preferred="model"):
    """The participant's fitted classifier: `model` if present, else any fitted object with predict_proba."""
    def fitted(v):
        try:
            return hasattr(v, "predict_proba") and not isinstance(v, type) and hasattr(v, "classes_")
        except Exception:
            return False
    m = ns.get(preferred)
    if fitted(m):
        return m
    return next((v for k, v in list(ns.items()) if not k.startswith("_") and fitted(v)), None)

def _auc_consistent(auc_value, y_name="y_test", tol=0.006):
    """True if auc_value equals the ROC-AUC of the participant's model on ANY test matrix in their workspace
    (X_test, a scaled copy, a selected-feature copy...). Allows rounding."""
    from sklearn.metrics import roc_auc_score
    y = ns.get(y_name)
    m = _model()
    if y is None or m is None:
        return False
    try:
        auc_value = float(auc_value)
    except Exception:
        return False
    for k, v in list(ns.items()):
        if k.startswith("_") or not hasattr(v, "shape") or len(getattr(v, "shape", ())) != 2:
            continue
        if v.shape[0] != len(y):
            continue
        try:
            if abs(roc_auc_score(y, m.predict_proba(v)[:, 1]) - auc_value) <= tol:
                return True
        except Exception:
            continue
    return False

ns = {"__name__": "__main__", "ask_llm": ask_llm,
      "_close": _close, "_model": _model, "_auc_consistent": _auc_consistent}

# ---------- replay earlier submitted tasks ----------
prior_error = None
_replaying[0] = True
for i, code in enumerate(payload["prior"]):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            exec(compile(code, f"<task_{i+1}>", "exec"), ns)
        except Exception as e:
            prior_error = f"Earlier task {i+1} raised {type(e).__name__}: {e}"
    plt.close("all")
_replaying[0] = False

# ---------- run current code ----------
out = io.StringIO()
error = None
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
    try:
        exec(compile(payload["code"], "<your_code>", "exec"), ns)
    except Exception:
        tb = traceback.format_exc()
        error = tb[tb.find('File "<your_code>"'):] if '<your_code>' in tb else tb
_capture_figures()
ns["_n_figures"] = len(figs)

# ---------- automatic rubric checks ----------
checks = []
for c in payload["checks"]:
    try:
        ok = bool(eval(c["check"], ns))
        msg = ""
    except Exception as e:
        ok, msg = False, f"{type(e).__name__}: {e}"
    checks.append({"criterion": c["criterion"], "passed": ok, "error": msg})

print("<<<EVAL_RESULT>>>" + json.dumps({
    "stdout": out.getvalue(), "error": error, "prior_error": prior_error,
    "figures": figs, "checks": checks}))
'''


def run_code(code: str, prior_codes: list[str], checks: list[dict], timeout: int = 60,
             llm_env: dict | None = None) -> dict:
    """Execute participant code and return stdout, error, figures and auto-check results."""
    figdir = tempfile.mkdtemp(prefix="evalfigs_")
    payload = {"code": code, "prior": prior_codes, "checks": checks, "figdir": figdir}

    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "payload.json"
        harness_path = Path(tmp) / "harness.py"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        harness_path.write_text(HARNESS, encoding="utf-8")

        env = {k: v for k, v in os.environ.items()
               if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
        env["MPLBACKEND"] = "Agg"
        env.update(llm_env or {})

        # Working folder where both "data/loan_data.csv" and plain "loan_data.csv" resolve.
        workdir = Path(tmp) / "work"
        workdir.mkdir()
        data_dir = ROOT / "data"
        if data_dir.exists():
            for src in [data_dir, *data_dir.iterdir()]:
                dst = workdir / src.name
                if dst.exists():
                    continue
                try:
                    dst.symlink_to(src, target_is_directory=src.is_dir())
                except OSError:  # symlinks not allowed (e.g. Windows) → copy instead
                    import shutil
                    (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)

        try:
            proc = subprocess.run([sys.executable, str(harness_path), str(payload_path)],
                                  cwd=str(workdir), capture_output=True, text=True,
                                  timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"stdout": "", "error": f"⏱️ Timed out after {timeout}s. Check for infinite loops or very slow code.",
                    "prior_error": None, "figures": [],
                    "checks": [{"criterion": c["criterion"], "passed": False, "error": "timeout"} for c in checks]}

    raw = proc.stdout
    if MARKER in raw:
        before, _, js = raw.rpartition(MARKER)
        result = json.loads(js.strip())
        result["stdout"] = (before + result["stdout"])[:MAX_STDOUT]
    else:  # process crashed (e.g. sys.exit, segfault)
        result = {"stdout": raw[:MAX_STDOUT], "error": proc.stderr[-3000:] or "Process exited unexpectedly.",
                  "prior_error": None, "figures": [],
                  "checks": [{"criterion": c["criterion"], "passed": False, "error": "not run"} for c in checks]}
    return result
