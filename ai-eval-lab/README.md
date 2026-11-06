# 🧪 AI Eval Lab

A browser-based hands-on assessment platform for **Python, EDA, Machine Learning and GenAI** tasks.
Participants read the problem, see the rubric, write code task-by-task, submit, and get instant
rubric-based scores with AI feedback.

| Tab | What participants see |
|---|---|
| 📄 **Problem Statement** | Business context, dataset dictionary, data preview, download |
| 📏 **Rubric** | Every criterion per task, its points, and whether it is auto-checked or AI-judged |
| 💻 **Code** | Task-by-task editor (tasks unlock one by one), ▶ Run to test, ✅ Submit to grade |
| 📊 **Evaluation Results** | Total score, band, per-task breakdown, AI feedback, output & charts, CSV/JSON report, leaderboard |

## Roles: participant and trainer

Choose **Sign in as** at the top of the sidebar.

| | Participant | Trainer (password) |
|---|---|---|
| Sees | Problem, Rubric, Code, own Evaluation Results | **Trainer Dashboard**, Problem, Rubric |
| Other people's results | Never (leaderboard only if the trainer turns it on) | Everyone: scores, time, status, code, feedback |
| Settings | None | LLM judge, timeout, copy-paste lock, time-up lock, leaderboard, problem upload, reset a participant |
| Identity | Name + Email/ID + a PIN they create; the PIN is needed to resume | `TRAINER_PASSWORD` from Secrets |

**Trainer Dashboard** (auto-refreshes every 30 s): participants, in progress, completed, average and top score;
a live table with each person's status, current task, score, time taken and time left; average score and time per
task; score distribution; the 10 criteria participants struggle with most; a drill-down into any participant's code,
output and feedback; CSV downloads.

Trainer settings are server-wide: they apply to every participant immediately (copy-paste changes apply after the
participant refreshes the page).

## How evaluation works

Each rubric criterion is one of two types:

- **`auto`** — a Python expression run against the participant's variables after their code executes
  (e.g. `df_clean.isna().sum().sum() == 0`). Deterministic, full points or zero.
- **`llm`** — an LLM judge reads the task, code, output and errors, and awards partial credit with a justification.

Code runs in a **separate Python process with a timeout**. Code from earlier submitted tasks is replayed
first, so variables (`df`, `X_train`, `model`…) carry over like a notebook. Charts from `plt.show()` are captured.

### LLM options (sidebar → ⚙️ Evaluator settings)

| Provider | Cost | Secret name |
|---|---|---|
| OpenAI (`gpt-4o-mini` default) | Paid, low | `OPENAI_API_KEY` |
| Groq — open-source Llama models | Free tier | `GROQ_API_KEY` |
| Ollama — fully local open-source | Free | none (local runs only) |
| Any OpenAI-compatible API (OpenRouter, Together, vLLM) | Varies | `LLM_API_KEY` |
| None | Free | Rule-based estimate, no AI feedback |

GenAI tasks can call `ask_llm(prompt, system=None)`. It uses the configured LLM, or a deterministic
**offline mock** if no key is set — so the GenAI pack works even without an API key.

## Run locally

```bash
git clone https://github.com/<you>/ai-eval-lab.git
cd ai-eval-lab
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your key
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud (via GitHub)

1. Create a new GitHub repo and push this folder (the `.gitignore` keeps `secrets.toml` out).
2. Go to **share.streamlit.io** → **Create app** → pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings → Secrets**, paste `TRAINER_PASSWORD = "..."` plus e.g. `GROQ_API_KEY = "gsk_..."` or `OPENAI_API_KEY = "sk-..."`.
4. Deploy, and share the URL with participants.

> ℹ️ Streamlit Cloud storage is temporary — the leaderboard resets when the app restarts.
> Ask participants to download their JSON report at the end, or connect a Google Sheet / database for persistence.

## Add your own problem

Drop a JSON file into `problems/` (or upload it from **🧑‍🏫 Trainer tools** in the sidebar):

```json
{
  "id": "unique_id",
  "title": "Problem title",
  "domain": "Insurance · Claims",
  "level": "Beginner",
  "duration_minutes": 60,
  "max_attempts": 3,
  "overview": "Markdown business context shown in the Problem tab",
  "dataset": {"path": "data/your_file.csv", "description": "…", "columns": {"col": "meaning"}},
  "tasks": [
    {
      "id": "T1",
      "title": "Load the data",
      "instruction": "Markdown instructions. Name the variables you will check, e.g. **`df`**.",
      "starter_code": "import pandas as pd\n",
      "rubric": [
        {"criterion": "df loaded", "points": 3, "method": "auto", "check": "df.shape[0] == 1000"},
        {"criterion": "Code clarity", "points": 2, "method": "llm", "guide": "What earns full marks"}
      ]
    }
  ]
}
```

**Tips for `check` expressions:** they run in the participant's namespace, so reference the variable
names you asked for in the instruction. `_n_figures` holds the number of charts shown. Use
`__import__('pandas')` if you need a module the participant might not have imported.

**Flexible check helpers** (so different-but-correct answers still pass):

| Helper | Use it for |
|---|---|
| `_close(value, truth)` | A metric the participant computed; allows rounding and 0–1 vs % scales |
| `_model()` | The participant's fitted classifier, even if not named `model` |
| `_auc_consistent(auc)` | AUC matches their model on any test matrix (scaled copies included) |

Check that the metric is **computed correctly**, and use a **threshold** (`auc >= 0.70`) rather than an exact value
for quality, so any valid model or LLM earns the points.

## Included problem packs

- **Retail Loan Default — EDA & ML** (5 tasks, 50 pts): load, clean, visualise, feature-engineer, model with AUC ≥ 0.70
- **GenAI — Customer Complaint Triage** (3 tasks, 30 pts): prompt design, robust JSON parser, pipeline accuracy & monitoring

## Security note

Participant code runs as Python on the server with a timeout and API keys removed from its environment.
That is fine for trusted cohorts (corporate training). For open public use, run inside a container sandbox.

## Project layout

```
app.py                 Streamlit UI (4 tabs)
evaluator/runner.py    Subprocess code execution, chart capture, auto checks, ask_llm helper
evaluator/grader.py    Rubric scoring: auto checks + LLM judge (OpenAI-compatible) + fallback
problems/*.json        Problem packs (statement, tasks, rubric)
data/                  Datasets
```
