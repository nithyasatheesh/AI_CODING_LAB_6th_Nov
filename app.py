"""
AI Eval Lab — hands-on coding assessment platform for Python, EDA, ML and GenAI.

Tabs: Problem Statement · Rubric · Code · Evaluation Results
Run locally:   streamlit run app.py
Deploy:        push to GitHub → share.streamlit.io → New app → app.py
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from evaluator.grader import grade
from evaluator.runner import run_code
from evaluator.storage import make_storage
from evaluator.report import build_excel, build_html

try:
    from streamlit_ace import st_ace
    HAS_ACE = True
except ImportError:
    HAS_ACE = False

ROOT = Path(__file__).parent
PROBLEM_DIR = ROOT / "problems"
RESULTS_DIR = ROOT / "results"
AUTOSAVE_EVERY_S = 15

PROVIDERS = {
    "OpenAI": {"base_url": "", "model": "gpt-4o-mini", "secret": "OPENAI_API_KEY"},
    "Groq (open-source Llama)": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "secret": "GROQ_API_KEY"},
    "Ollama (local)": {"base_url": "http://localhost:11434/v1", "model": "llama3.1", "secret": ""},
    "Custom OpenAI-compatible": {"base_url": "", "model": "", "secret": "LLM_API_KEY"},
    "None (rule-based)": {"base_url": "", "model": "", "secret": ""},
}

def _setting(name: str, default: str = "") -> str:
    """Reads a setting from Streamlit secrets or environment variables (Docker)."""
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    return str(v or os.environ.get(name, "") or default)


APP_NAME = _setting("APP_NAME", "AI Eval Lab")
APP_TAGLINE = _setting("APP_TAGLINE", "Hands-on Python · EDA · ML · GenAI assessment")
if _setting("OLLAMA_BASE_URL"):
    PROVIDERS["Ollama (local)"].update(base_url=_setting("OLLAMA_BASE_URL"),
                                       model=_setting("OLLAMA_MODEL", PROVIDERS["Ollama (local)"]["model"]))
st.set_page_config(page_title=APP_NAME, page_icon="🧪", layout="wide",
                   menu_items={"Get help": None, "Report a bug": None, "About": f"{APP_NAME} — hands-on assessment"})

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-left: 2rem; padding-right: 2rem; max-width: 1800px;}
.task-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:1rem 1.2rem; margin-bottom:.8rem;
            background: rgba(37,99,235,.04);}
.chip {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; margin-right:6px;}
.chip-auto {background:#dcfce7; color:#166534;}
.chip-llm {background:#e0e7ff; color:#3730a3;}
.chip-meta {background:rgba(128,128,128,.15);}
.hero h1 {margin-bottom:.1rem;}
.hero p {opacity:.75; margin-top:0;}
div[data-testid="stMetricValue"] {font-size:1.8rem;}
</style>
""", unsafe_allow_html=True)


# ───────────────────────── helpers ─────────────────────────
def secret(name: str) -> str:
    return _setting(name) if name else ""


TZ = ZoneInfo(secret("APP_TIMEZONE") or "Asia/Kolkata")


def fmt_ts(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, TZ).strftime("%d %b %Y, %I:%M:%S %p") if ts else "–"


def fmt_clock(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, TZ).strftime("%I:%M:%S %p") if ts else "–"


def fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "–"
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


@st.cache_data
def load_builtin_problems() -> dict:
    probs = {}
    for f in sorted(PROBLEM_DIR.glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        probs[p["id"]] = p
    return probs


def task_points(task: dict) -> float:
    return sum(c["points"] for c in task["rubric"])


def subs_for(pid: str) -> dict:
    return st.session_state.setdefault("subs", {}).setdefault(pid, {})


def prog_for(pid: str) -> dict:
    return st.session_state.setdefault("progress", {}).setdefault(
        pid, {"started_at": None, "drafts": {}, "task_started": {}, "saved_at": None})


# ---------- saving / restoring progress (PostgreSQL when DATABASE_URL is set, else local files) ----------
@st.cache_resource
def get_storage():
    return make_storage(secret("DATABASE_URL"), ROOT)


STORE, STORE_WARNING = get_storage()


def _encode_subs(subs: dict) -> dict:
    out = {}
    for tid, s in subs.items():
        s2 = dict(s)
        run = {k: v for k, v in s2.get("run", {}).items() if k not in ("fig_bytes", "figures")}
        run["fig_b64"] = [base64.b64encode(b).decode() for b in s2.get("run", {}).get("fig_bytes", [])]
        s2["run"] = run
        out[tid] = s2
    return out


def _decode_subs(subs: dict) -> dict:
    for s in subs.values():
        run = s.get("run", {})
        run["fig_bytes"] = [base64.b64decode(x) for x in run.pop("fig_b64", [])]
    return subs


def pin_hash(pkey: str, pin: str) -> str:
    return hashlib.sha256(f"{pkey}:{pin}".encode()).hexdigest() if pin else ""


def save_progress(pkey: str, pid: str, name: str, email: str) -> bool:
    if not pkey:
        return False
    prog = prog_for(pid)
    prog["saved_at"] = time.time()
    return STORE.put(pkey, pid, {"name": name, "email": email, "problem": pid,
                                 "pin_hash": st.session_state.get("pin_hash", ""), "progress": prog,
                                 "subs": _encode_subs(subs_for(pid))})


def restore_progress(pkey: str, pid: str) -> str:
    """Returns 'new' (no saved work), 'ok' (restored or already loaded), 'need_pin' or 'wrong_pin'."""
    my_hash = st.session_state.get("pin_hash", "")
    cache_key = f"access_{pkey}_{pid}_{my_hash}"
    if st.session_state.get(cache_key) in ("ok", "new") and (prog_for(pid)["started_at"] or
                                                             st.session_state[cache_key] == "new"):
        return st.session_state[cache_key]  # already checked this session — no database round-trip
    try:
        data = STORE.get(pkey, pid)
    except Exception:
        return "new"
    if data is None:
        st.session_state[cache_key] = "new"
        return "new"
    if not my_hash:
        return "need_pin"
    if not hmac.compare_digest(data.get("pin_hash", ""), my_hash):
        return "wrong_pin"
    if not subs_for(pid) and not prog_for(pid)["started_at"]:
        st.session_state["progress"][pid] = data["progress"]
        st.session_state["subs"][pid] = _decode_subs(data["subs"])
        st.session_state["just_restored"] = True
    st.session_state[cache_key] = "ok"
    return "ok"


def all_participants() -> pd.DataFrame:
    """Summary of every participant's saved work (trainer view)."""
    rows = []
    for d in STORE.list():
        subs = d.get("subs", {})
        prob = problems.get(d.get("problem"), {})
        n_tasks = len(prob.get("tasks", [])) or "?"
        rows.append({"Name": d.get("name"), "Email / ID": d.get("email"), "Problem": prob.get("title", d.get("problem")),
                     "Tasks submitted": f"{len(subs)}/{n_tasks}",
                     "Score": round(sum(s["grade"]["score"] for s in subs.values()), 1),
                     "_ref": (d["_key"], d["_pid"])})
    return pd.DataFrame(rows)


# ---------- run queue: limits how many participants' code runs at the same time ----------
@st.cache_resource
def run_slots() -> dict:
    try:
        n = int(secret("MAX_PARALLEL_RUNS") or 4)
    except ValueError:
        n = 4
    return {"sem": threading.Semaphore(max(1, n)), "waiting": 0, "lock": threading.Lock(), "size": n}


def run_in_queue(**kwargs) -> dict:
    slots = run_slots()
    with slots["lock"]:
        slots["waiting"] += 1
        ahead = max(0, slots["waiting"] - slots["size"])
    try:
        msg = "Running your code…" if ahead == 0 else f"Server busy — you are in the queue ({ahead} ahead)…"
        with st.spinner(msg):
            if not slots["sem"].acquire(timeout=180):
                return {"stdout": "", "error": "⏳ The server is very busy. Please wait a moment and click Run again.",
                        "prior_error": None, "fig_bytes": [], "figures": [],
                        "checks": [{"criterion": c["criterion"], "passed": False, "error": "busy"}
                                   for c in kwargs.get("checks", [])]}
            try:
                return run_code(**kwargs)
            finally:
                slots["sem"].release()
    finally:
        with slots["lock"]:
            slots["waiting"] -= 1


# ---------- live timer (runs in the browser, ticks every second) ----------
def live_timer(end_ts: float, task_start_ts: float | None = None, task_label: str = ""):
    task_box = f"""
      <div class="box"><div class="lbl">🕒 Time on {task_label}</div><div id="task" class="val">--:--</div></div>""" \
        if task_start_ts else ""
    html = f"""
    <style>
      body {{margin:0; font-family:"Source Sans Pro", system-ui, sans-serif;}}
      .wrap {{display:flex; gap:10px;}}
      .box {{flex:1; border-radius:10px; padding:6px 12px; background:#eff6ff; border:1px solid #bfdbfe;}}
      .lbl {{font-size:12px; color:#475569;}}
      .val {{font-size:24px; font-weight:700; color:#1e3a8a; font-variant-numeric:tabular-nums;}}
      .warn {{background:#fff7ed; border-color:#fdba74;}} .warn .val {{color:#c2410c;}}
      .over {{background:#fef2f2; border-color:#fca5a5;}} .over .val {{color:#b91c1c;}}
    </style>
    <div class="wrap">
      <div class="box" id="remBox"><div class="lbl">⏳ Time remaining</div><div id="rem" class="val">--:--</div></div>
      {task_box}
    </div>
    <script>
      const end = {end_ts * 1000:.0f}, ts = {f"{task_start_ts * 1000:.0f}" if task_start_ts else "null"};
      const pad = n => String(n).padStart(2, "0");
      const fmt = ms => {{ let s = Math.max(0, Math.floor(ms / 1000)); const h = Math.floor(s / 3600); s %= 3600;
                          return (h ? h + ":" : "") + pad(Math.floor(s / 60)) + ":" + pad(s % 60); }};
      function tick() {{
        const now = Date.now(), r = end - now, box = document.getElementById("remBox");
        if (r <= 0) {{ document.getElementById("rem").textContent = "Time's up"; box.className = "box over"; }}
        else {{ document.getElementById("rem").textContent = fmt(r); box.className = r < 300000 ? "box warn" : "box"; }}
        if (ts) document.getElementById("task").textContent = fmt(now - ts);
      }}
      tick(); setInterval(tick, 1000);
    </script>"""
    if hasattr(st, "iframe"):  # Streamlit ≥ 1.52
        st.iframe(html, height=72)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=72)


# ---------- copy / paste lock (deterrent for assessments) ----------
_GUARD_JS = r"""
(function () {
  const W = window, D = document;
  if (W.__evalLabGuard) return;
  W.__evalLabGuard = true;

  let last = 0;
  function notify() {
    const now = Date.now();
    if (now - last < 1500) return;
    last = now;
    const n = D.createElement("div");
    n.textContent = "🚫 Copy & paste is disabled during this assessment — please type your code.";
    n.style.cssText = "position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:999999;" +
      "background:#b91c1c;color:#fff;padding:10px 18px;border-radius:10px;font:600 14px sans-serif;" +
      "box-shadow:0 6px 20px rgba(0,0,0,.25)";
    D.body.appendChild(n);
    setTimeout(() => n.remove(), 2600);
  }
  function block(e) {
    e.preventDefault(); e.stopPropagation();
    if (e.stopImmediatePropagation) e.stopImmediatePropagation();
    if (e.type !== "dragover") notify();
    return false;
  }
  function isShortcut(e) {
    const k = (e.key || "").toLowerCase();
    return ((e.ctrlKey || e.metaKey) && ["c", "v", "x"].includes(k)) ||
           (e.key === "Insert" && (e.shiftKey || e.ctrlKey));
  }
  const inSidebar = t => t && t.closest && t.closest('[data-testid="stSidebar"]');

  // Code editor (Ace runs inside its own iframe): block every clipboard route
  function guardEditor(doc) {
    if (!doc || doc.__evalLabGuard) return;
    doc.__evalLabGuard = true;
    ["paste", "copy", "cut", "drop", "dragover", "contextmenu"].forEach(ev => doc.addEventListener(ev, block, true));
    doc.addEventListener("keydown", e => { if (isShortcut(e)) block(e); }, true);
  }

  // Main page: no copying task text; no pasting into the fallback text editor. Sidebar stays usable.
  ["copy", "cut"].forEach(ev => D.addEventListener(ev, e => { if (!inSidebar(e.target)) block(e); }, true));
  D.addEventListener("paste", e => {
    const t = e.target;
    if (!inSidebar(t) && t.closest && t.closest('[data-testid="stTextArea"]')) block(e);
  }, true);
  D.addEventListener("contextmenu", e => { if (!inSidebar(e.target)) block(e); }, true);
  const st = D.createElement("style");
  st.textContent = '[data-testid="stMain"] {-webkit-user-select:none; user-select:none;}' +
                   '[data-testid="stMain"] input, [data-testid="stMain"] textarea {-webkit-user-select:text; user-select:text;}';
  D.head.appendChild(st);

  function scan() {
    D.querySelectorAll("iframe").forEach(f => {
      const id = (f.getAttribute("src") || "") + " " + (f.getAttribute("title") || "");
      if (!id.includes("streamlit_ace")) return;
      try {
        guardEditor(f.contentDocument);
        if (!f.__evalLabHooked) { f.__evalLabHooked = true; f.addEventListener("load", () => guardEditor(f.contentDocument)); }
      } catch (err) {}
    });
  }
  scan();
  setInterval(scan, 700);
})();
"""


def copy_paste_guard():
    """Installs the lock into the main app page (runs once; survives reruns)."""
    code = json.dumps(_GUARD_JS)
    html = f"""<script>
      try {{
        const P = window.parent;
        if (!P.__evalLabGuard) {{
          const s = P.document.createElement("script");
          s.textContent = {code};
          P.document.head.appendChild(s);
        }}
      }} catch (e) {{ console.warn("copy/paste guard not installed", e); }}
    </script>"""
    if hasattr(st, "iframe"):
        st.iframe(html, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=0)


# ---------- server-wide settings: set by the trainer, apply to every participant ----------
@st.cache_resource
def shared_settings() -> dict:
    if secret("OLLAMA_BASE_URL"):  # self-hosted open-source LLM (e.g. the Ollama service in docker-compose)
        prov = "Ollama (local)"
    elif secret("GROQ_API_KEY"):
        prov = "Groq (open-source Llama)"
    elif secret("OPENAI_API_KEY"):
        prov = "OpenAI"
    elif secret("LLM_API_KEY"):
        prov = "Custom OpenAI-compatible"
    else:
        prov = "None (rule-based)"
    return {"provider": prov, "model": PROVIDERS[prov]["model"], "base_url": PROVIDERS[prov]["base_url"],
            "api_key": "", "share_llm": True, "timeout": 60, "lock_on_timeout": True,
            "block_copy_paste": True, "show_leaderboard": False, "custom_problems": {}}


S = shared_settings()
TRAINER_PASSWORD = secret("TRAINER_PASSWORD")

# ───────────────────────── sidebar ─────────────────────────
problems = dict(load_builtin_problems())
problems.update(S["custom_problems"])

with st.sidebar:
    st.markdown(f"## 🧪 {APP_NAME}")
    st.caption(APP_TAGLINE)
    role = st.segmented_control("Sign in as", ["👤 Participant", "🧑‍🏫 Trainer"], default="👤 Participant",
                                key="role", width="stretch") or "👤 Participant"
    is_trainer = False

    if role == "🧑‍🏫 Trainer":
        if not TRAINER_PASSWORD:
            st.warning("Trainer access is not set up. Add `TRAINER_PASSWORD = \"...\"` in the app's Secrets "
                       "(Manage app → ⋮ → Settings → Secrets).")
        else:
            tpw = st.text_input("Trainer password", type="password", key="trainer_pw")
            is_trainer = bool(tpw) and hmac.compare_digest(tpw, TRAINER_PASSWORD)
            if tpw and not is_trainer:
                st.error("Incorrect password.")
        name = email = pin = pkey = ""
        st.session_state["pin_hash"] = ""
    else:
        st.markdown("#### 👤 Participant")
        name = st.text_input("Full name", key="p_name", placeholder="Required")
        email = st.text_input("Email / Employee ID", key="p_email", placeholder="Required — used to resume")
        pin = st.text_input("Create your own PIN (any 4+ characters)", key="p_pin", type="password",
                            placeholder="e.g. 4821 — remember it",
                            help="Make up any PIN, like a password. It protects your work: you need the same "
                                 "Email / ID and PIN to continue after a refresh. Forgot it? Ask the trainer to reset.")
        pkey = slug(email)
        st.session_state["pin_hash"] = pin_hash(pkey, pin) if len(pin) >= 4 else ""

    st.markdown("#### 📚 Assessment")
    pid = st.selectbox("Problem", list(problems), format_func=lambda k: problems[k]["title"])
    problem = problems[pid]
    tasks = problem["tasks"]
    duration_min = problem.get("duration_minutes", 60)
    total_max = sum(task_points(t) for t in tasks)

    if role == "👤 Participant":
        access = restore_progress(pkey, pid) if pkey else "new"
        subs = subs_for(pid)
        prog = prog_for(pid)
        started_at = prog.get("started_at")
        end_ts = started_at + duration_min * 60 if started_at else None

        if st.session_state.pop("just_restored", False):
            st.toast("Welcome back — your saved work has been restored.", icon="💾")
        if access == "need_pin":
            st.info("This ID already has saved work. Enter your PIN to resume.")
        elif access == "wrong_pin":
            st.error("Wrong PIN for this ID. Enter the PIN you created, or ask the trainer to reset it.")

        ready = bool(name.strip()) and bool(pkey) and len(pin) >= 4 and access in ("new", "ok")
        if not started_at:
            st.info(f"⏱️ You have **{duration_min} minutes**. The timer starts when you click **Start**.")
            if st.button("▶ Start assessment", type="primary", width="stretch", disabled=not ready):
                prog["started_at"] = time.time()
                prog["task_started"][tasks[0]["id"]] = prog["started_at"]
                save_progress(pkey, pid, name, email)
                st.rerun()
            if not ready and access in ("new", "ok"):
                missing = [lbl for ok, lbl in ((name.strip(), "Full name"), (pkey, "Email / ID"),
                                               (len(pin) >= 4, "a PIN of 4+ characters")) if not ok]
                st.caption("To start, fill in: **" + ", ".join(missing) + "** — press **Enter** after typing each one.")
        else:
            live_timer(end_ts)
            st.caption(f"Started at {fmt_clock(started_at)} · ends at {fmt_clock(end_ts)}")
            if prog.get("saved_at"):
                st.caption(f"💾 Last saved {fmt_clock(prog['saved_at'])}")

        done = sum(1 for t in tasks if t["id"] in subs)
        total_score = sum(subs[t["id"]]["grade"]["score"] for t in tasks if t["id"] in subs)
        st.progress(done / len(tasks), text=f"{done}/{len(tasks)} tasks submitted")
        st.metric("Score so far", f"{total_score:g} / {total_max:g}")
    else:
        subs, prog, started_at, end_ts, done, total_score = {}, prog_for(pid), None, None, 0, 0

    if is_trainer:
        st.success("Trainer mode — settings apply to all participants.", icon="🔓")
        if STORE.persistent:
            st.caption(f"💾 Storage: **{STORE.name}** — results are kept permanently.")
        else:
            st.warning(STORE_WARNING or "💾 Storage: temporary files. Results are lost when the app restarts. "
                       "Add `DATABASE_URL` in Secrets for permanent storage.")
        with st.expander("⚙️ Evaluator settings", expanded=False):
            provider = st.selectbox("LLM judge", list(PROVIDERS), index=list(PROVIDERS).index(S["provider"]))
            if provider != S["provider"]:
                S.update(provider=provider, model=PROVIDERS[provider]["model"], base_url=PROVIDERS[provider]["base_url"])
            S["model"] = st.text_input("Model", value=S["model"], disabled=provider == "None (rule-based)")
            S["base_url"] = st.text_input("Base URL", value=S["base_url"],
                                          disabled=provider in ("OpenAI", "None (rule-based)"))
            if secret(PROVIDERS[provider]["secret"]):
                st.success(f"Using `{PROVIDERS[provider]['secret']}` from secrets", icon="🔐")
            elif provider not in ("None (rule-based)", "Ollama (local)"):
                S["api_key"] = st.text_input("API key (kept in memory only)", value=S["api_key"], type="password")
            S["share_llm"] = st.checkbox("Let participant code use this LLM via `ask_llm()`", value=S["share_llm"],
                                         help="Off = participants' ask_llm() uses an offline mock LLM.")
            S["timeout"] = st.slider("Execution timeout (s)", 10, 180, S["timeout"])

        with st.expander("🛡️ Assessment rules", expanded=False):
            S["lock_on_timeout"] = st.checkbox("Block submissions when time is up", value=S["lock_on_timeout"])
            S["block_copy_paste"] = st.checkbox("Disable copy & paste for participants", value=S["block_copy_paste"],
                                                help="Participants who already have the page open get the change "
                                                     "after a page refresh.")
            S["show_leaderboard"] = st.checkbox("Show leaderboard to participants", value=S["show_leaderboard"])

        with st.expander("📦 Problems & participants", expanded=False):
            up = st.file_uploader("Upload a problem pack (.json)", type="json")
            if up:
                try:
                    newp = json.loads(up.getvalue())
                    S["custom_problems"][newp["id"]] = newp
                    st.success(f"Loaded for everyone: {newp['title']}")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
            people = all_participants()
            if len(people):
                refs = list(people["_ref"])
                pick = st.selectbox("Reset a participant", range(len(refs)),
                                    format_func=lambda i: " · ".join(str(people.iloc[i][c]) for c in ("Name", "Email / ID", "Problem")))
                if st.button("🗑️ Reset selected participant", width="stretch"):
                    STORE.delete(*refs[pick])
                    st.success("Reset. They can start again with a new PIN.")
            else:
                st.caption("No participants yet.")

provider, model, base_url = S["provider"], S["model"], S["base_url"]
api_key = secret(PROVIDERS[provider]["secret"]) or S["api_key"]
share_llm, timeout = S["share_llm"], S["timeout"]
lock_on_timeout, block_copy_paste = S["lock_on_timeout"], S["block_copy_paste"]

time_up = bool(end_ts) and time.time() >= end_ts
if block_copy_paste and not is_trainer:
    copy_paste_guard()
judge_cfg = {"provider": provider, "model": model, "base_url": base_url, "api_key": api_key}
participant_llm_env = {}
if share_llm and provider != "None (rule-based)" and (api_key or provider == "Ollama (local)"):
    participant_llm_env = {"PARTICIPANT_LLM_API_KEY": api_key or "ollama",
                           "PARTICIPANT_LLM_BASE_URL": base_url, "PARTICIPANT_LLM_MODEL": model}

# ───────────────────────── header ─────────────────────────
st.markdown(f"<div class='hero'><h1>{problem['title']}</h1>"
            f"<p>{problem.get('domain', '')} · {problem.get('level', '')} · ⏱️ {duration_min} min · "
            f"{len(tasks)} tasks · {total_max:g} points</p></div>", unsafe_allow_html=True)

PARTICIPANT_VIEW = role == "👤 Participant"
if is_trainer:
    tab_dash, tab_problem, tab_rubric = st.tabs(["🧑‍🏫 Trainer Dashboard", "📄 Problem Statement", "📏 Rubric"])
elif PARTICIPANT_VIEW:
    tab_problem, tab_rubric, tab_code, tab_results = st.tabs(
        ["📄 Problem Statement", "📏 Rubric", "💻 Code", "📊 Evaluation Results"])
else:
    st.info("🔒 Enter the trainer password in the sidebar to open the Trainer Dashboard.")
    tab_problem, tab_rubric = st.tabs(["📄 Problem Statement", "📏 Rubric"])

# ───────────────────────── 1. Problem statement ─────────────────────────
with tab_problem:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(problem["overview"])
        st.markdown("### Tasks")
        for i, t in enumerate(tasks, 1):
            status = "✅" if t["id"] in subs else "⬜"
            st.markdown(f"{status} **Task {i} — {t['title']}** · {task_points(t):g} pts")
    with right:
        ds = problem.get("dataset")
        if ds:
            st.markdown(f"### 🗂️ Dataset\n`{ds['path']}` — {ds.get('description', '')}")
            if ds.get("columns"):
                st.dataframe(pd.DataFrame(ds["columns"].items(), columns=["Column", "Description"]),
                             hide_index=True, width="stretch")
            fp = ROOT / ds["path"]
            if fp.exists():
                st.markdown("**Preview**")
                st.dataframe(pd.read_csv(fp).head(8), width="stretch", hide_index=True)
                st.download_button("⬇️ Download dataset", fp.read_bytes(), file_name=fp.name, width="stretch")

# ───────────────────────── 2. Rubric ─────────────────────────
with tab_rubric:
    st.markdown("Every task is scored on the criteria below. "
                "<span class='chip chip-auto'>AUTO</span> criteria are checked by running your code; "
                "<span class='chip chip-llm'>AI JUDGE</span> criteria are scored by an LLM evaluator that reads your code and output.",
                unsafe_allow_html=True)
    summary = pd.DataFrame([{"Task": f"Task {i} — {t['title']}", "Criteria": len(t["rubric"]),
                             "Auto pts": sum(c["points"] for c in t["rubric"] if c.get("method") == "auto"),
                             "AI-judge pts": sum(c["points"] for c in t["rubric"] if c.get("method") != "auto"),
                             "Total": task_points(t)} for i, t in enumerate(tasks, 1)])
    st.dataframe(summary, hide_index=True, width="stretch")
    for i, t in enumerate(tasks, 1):
        with st.expander(f"Task {i} — {t['title']}  ·  {task_points(t):g} pts", expanded=(i == 1)):
            st.dataframe(pd.DataFrame([{
                "Criterion": c["criterion"], "Points": c["points"],
                "Evaluated by": "✅ Auto check" if c.get("method") == "auto" else "🤖 AI judge",
                "What earns the points": c.get("guide", "Passes the automatic test")} for c in t["rubric"]]),
                hide_index=True, width="stretch",
                column_config={"What earns the points": st.column_config.TextColumn(width="large")})

if PARTICIPANT_VIEW:
    # ───────────────────────── 3. Code ─────────────────────────
    with tab_code:
        if not started_at:
            st.info("👈 In the sidebar, enter your **name**, **Email / ID** and **make up a PIN** "
                        "(any 4+ characters you'll remember), then click **▶ Start assessment**. "
                        f"Your {duration_min}-minute timer begins then. Read the Problem and Rubric tabs first if you like.")
        else:
            def label(i: int) -> str:
                t = tasks[i]
                if t["id"] in subs:
                    icon = "✅"
                elif i == 0 or tasks[i - 1]["id"] in subs:
                    icon = "✏️"
                else:
                    icon = "🔒"
                return f"{icon} Task {i + 1}"

            first_open = next((i for i, t in enumerate(tasks) if t["id"] not in subs), len(tasks) - 1)
            if f"advance_{pid}" in st.session_state:  # move to the next task after a submit
                st.session_state[f"task_sel_{pid}"] = st.session_state.pop(f"advance_{pid}")
            st.session_state.setdefault(f"task_sel_{pid}", first_open)
            idx = st.radio("Task", range(len(tasks)), format_func=label,
                           horizontal=True, label_visibility="collapsed", key=f"task_sel_{pid}")
            task = tasks[idx]
            tid = task["id"]
            locked = idx > 0 and tasks[idx - 1]["id"] not in subs
            sub = subs.get(tid)
            max_att = problem.get("max_attempts", 3)
            attempts = sub["attempts"] if sub else 0

            if locked:
                st.warning(f"🔒 Submit **Task {idx}** first — tasks unlock one by one.")
            else:
                if not sub:
                    prog["task_started"].setdefault(tid, time.time())
                live_timer(end_ts, None if sub else prog["task_started"].get(tid), f"Task {idx + 1}")
                if time_up:
                    st.error("⏰ Time is up." + (" Submissions are closed — your saved work is kept."
                                                 if lock_on_timeout else " Late submissions are allowed by the trainer."))

                lay1, lay2, _ = st.columns([1.2, 2, 3])
                wide = lay1.toggle("↔️ Full-width editor", value=True, key="wide_editor",
                                   help="Show the task above the editor so the editor uses the whole width.")
                editor_h = lay2.select_slider("Editor height", options=[450, 600, 750, 900], value=600,
                                              format_func=lambda h: {450: "Small", 600: "Medium", 750: "Large", 900: "X-Large"}[h],
                                              key="editor_height")
                if wide:
                    c1 = st.expander(f"📋 Task {idx + 1} instructions — {task['title']}", expanded=True)
                    c2 = st.container()
                else:
                    c1, c2 = st.columns([1, 2], gap="large")
                with c1:
                    st.markdown(f"<div class='task-card'><b>Task {idx + 1} of {len(tasks)}</b><br>"
                                f"<span style='font-size:1.25rem;font-weight:700'>{task['title']}</span><br>"
                                f"<span class='chip chip-meta'>{task_points(task):g} pts</span>"
                                f"<span class='chip chip-meta'>Attempts {attempts}/{max_att}</span></div>",
                                unsafe_allow_html=True)
                    st.markdown(task["instruction"])
                    if idx > 0:
                        st.caption("💡 Variables from your earlier submitted tasks (e.g. `df`) are available here.")
                    if sub:
                        st.success(f"Submitted at {fmt_clock(sub.get('submitted_ts'))} · "
                                   f"time taken **{fmt_dur(sub.get('time_taken_s'))}** · "
                                   f"scored **{sub['grade']['score']:g} / {sub['grade']['max_score']:g}**. "
                                   "See the **📊 Evaluation Results** tab.")

                with c2:
                    reset_n = st.session_state.get(f"reset_{pid}_{tid}", 0)
                    initial = prog["drafts"].get(tid) or (sub["code"] if sub else task.get("starter_code", ""))
                    if block_copy_paste:
                        st.caption("✍️ Copy & paste is disabled — type your code.")
                    if HAS_ACE:
                        code = st_ace(value=initial, language="python", theme="tomorrow_night", font_size=14,
                                      height=editor_h, tab_size=4, wrap=True, show_gutter=True, auto_update=True,
                                      key=f"ace_{pid}_{tid}_{reset_n}")
                    else:
                        code = st.text_area("Your code", value=initial, height=editor_h, key=f"ta_{pid}_{tid}_{reset_n}")

                    # auto-save the draft every few seconds while typing
                    if code != prog["drafts"].get(tid):
                        prog["drafts"][tid] = code
                        if time.time() - (prog.get("saved_at") or 0) > AUTOSAVE_EVERY_S:
                            save_progress(pkey, pid, name, email)

                    b1, b2, b3, b4 = st.columns(4)
                    run_clicked = b1.button("▶ Run", width="stretch", help="Test your code — not graded")
                    save_clicked = b2.button("💾 Save", width="stretch", help="Save your work without submitting")
                    blocked = time_up and lock_on_timeout
                    can_submit = bool(name.strip()) and attempts < max_att and not blocked
                    submit_clicked = b3.button("✅ Submit", type="primary", width="stretch", disabled=not can_submit)
                    if b4.button("↺ Reset", width="stretch", help="Restore the starter code"):
                        st.session_state[f"reset_{pid}_{tid}"] = reset_n + 1
                        prog["drafts"][tid] = task.get("starter_code", "")
                        save_progress(pkey, pid, name, email)
                        st.rerun()
                    if attempts >= max_att:
                        st.caption("No attempts left for this task.")
                    if prog.get("saved_at"):
                        st.caption(f"💾 Auto-saved every {AUTOSAVE_EVERY_S}s · last saved {fmt_clock(prog['saved_at'])}")

                    if save_clicked:
                        ok = save_progress(pkey, pid, name, email)
                        st.toast(f"Saved at {fmt_clock(prog['saved_at'])}" if ok else "Couldn't save — check your name/ID.",
                                 icon="💾" if ok else "⚠️")

                    prior = [subs[t["id"]]["code"] for t in tasks[:idx] if t["id"] in subs]
                    checks = [{"criterion": c["criterion"], "check": c["check"]}
                              for c in task["rubric"] if c.get("method") == "auto"]

                    def show_run(res: dict):
                        if res.get("prior_error"):
                            st.warning(f"⚠️ {res['prior_error']}")
                        if res.get("stdout", "").strip():
                            st.markdown("**Output**")
                            st.code(res["stdout"], language="text")
                        if res.get("error"):
                            st.error("**Error**")
                            st.code(res["error"], language="text")
                        for fig in res.get("fig_bytes", []):
                            st.image(fig, width="stretch")
                        if not res.get("error") and not res.get("stdout", "").strip() and not res.get("fig_bytes"):
                            st.info("Code ran with no output. Use `print()` or `plt.show()` to display results.")

                    def execute() -> dict:
                        return run_in_queue(code=code, prior_codes=prior, checks=checks, timeout=timeout,
                                            llm_env=participant_llm_env)

                    if run_clicked:
                        st.session_state[f"last_run_{pid}_{tid}"] = execute()
                        save_progress(pkey, pid, name, email)

                    if submit_clicked:
                        res = execute()
                        with st.spinner("Evaluating against the rubric…"):
                            g = grade(task, code, res, judge_cfg)
                        now = time.time()
                        taken = now - prog["task_started"].get(tid, started_at)
                        subs[tid] = {"code": code, "run": res, "grade": g, "attempts": attempts + 1,
                                     "submitted_ts": now, "submitted_at": fmt_ts(now), "time_taken_s": round(taken)}
                        if idx + 1 < len(tasks):  # next task's clock starts now, and the editor moves to it
                            prog["task_started"].setdefault(tasks[idx + 1]["id"], now)
                            st.session_state[f"advance_{pid}"] = idx + 1
                        save_progress(pkey, pid, name, email)
                        STORE.log_submission({"submitted_at": fmt_ts(now), "name": name, "email": email, "problem": pid,
                                          "task": tid, "score": g["score"], "max": g["max_score"],
                                          "attempt": attempts + 1, "time_taken_s": round(taken)})
                        st.session_state[f"last_run_{pid}_{tid}"] = res
                        st.toast(f"Task {idx + 1} submitted & saved — {g['score']:g}/{g['max_score']:g} "
                                 f"in {fmt_dur(taken)}", icon="🎉")
                        st.rerun()

                    last = st.session_state.get(f"last_run_{pid}_{tid}")
                    if last:
                        st.divider()
                        show_run(last)

    # ───────────────────────── 4. Evaluation results ─────────────────────────
    with tab_results:
        if not subs:
            st.info("No submissions yet. Complete a task in the **💻 Code** tab and press **Submit**.")
        else:
            pct = 100 * total_score / total_max if total_max else 0
            grade_band = ("🏆 Excellent" if pct >= 85 else "👍 Proficient" if pct >= 70
                          else "📈 Developing" if pct >= 50 else "🌱 Beginner")
            total_time = sum(s.get("time_taken_s") or 0 for s in subs.values())
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total score", f"{total_score:g} / {total_max:g}")
            m2.metric("Percentage", f"{pct:.0f}%")
            m3.metric("Tasks submitted", f"{done} / {len(tasks)}")
            m4.metric("Time taken", fmt_dur(total_time))
            m5.metric("Band", grade_band)

            timeline = pd.DataFrame([{
                "Task": f"Task {i} — {t['title']}",
                "Score": f"{subs[t['id']]['grade']['score']:g} / {task_points(t):g}" if t["id"] in subs else "–",
                "Time taken": fmt_dur(subs[t["id"]].get("time_taken_s")) if t["id"] in subs else "–",
                "Submitted at": fmt_clock(subs[t["id"]].get("submitted_ts")) if t["id"] in subs else "Not submitted",
                "Attempts": subs[t["id"]]["attempts"] if t["id"] in subs else 0,
            } for i, t in enumerate(tasks, 1)])
            st.dataframe(timeline, hide_index=True, width="stretch")

            chart_df = pd.DataFrame([{"Task": f"T{i}", "Score": subs[t["id"]]["grade"]["score"] if t["id"] in subs else 0,
                                      "Remaining": task_points(t) - (subs[t["id"]]["grade"]["score"] if t["id"] in subs else 0)}
                                     for i, t in enumerate(tasks, 1)]).set_index("Task")
            st.bar_chart(chart_df, color=["#2563eb", "#cbd5e1"], height=240, horizontal=True)

            report_rows = []
            for i, t in enumerate(tasks, 1):
                s = subs.get(t["id"])
                if not s:
                    continue
                g = s["grade"]
                for r in g["rows"]:
                    report_rows.append({"task": f"Task {i}", "title": t["title"], **r,
                                        "time_taken": fmt_dur(s.get("time_taken_s")), "submitted_at": s.get("submitted_at")})
                with st.expander(f"Task {i} — {t['title']}  ·  {g['score']:g} / {g['max_score']:g}  ·  "
                                 f"⏱️ {fmt_dur(s.get('time_taken_s'))}  ·  attempt {s['attempts']}",
                                 expanded=(i == done)):
                    st.dataframe(pd.DataFrame(g["rows"]).rename(columns={
                        "criterion": "Criterion", "method": "Evaluated by", "max": "Max", "score": "Score",
                        "justification": "Justification"})[["Criterion", "Score", "Max", "Evaluated by", "Justification"]],
                        hide_index=True, width="stretch",
                        column_config={"Justification": st.column_config.TextColumn(width="large")})
                    if g.get("feedback"):
                        st.markdown(f"**🧠 Feedback:** {g['feedback']}")
                    if g.get("improvements"):
                        st.markdown("**🔧 Improve next time:**\n" + "\n".join(f"- {x}" for x in g["improvements"]))
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.markdown("**Submitted code**")
                        st.code(s["code"], language="python")
                    with cc2:
                        st.markdown("**Output**")
                        run = s["run"]
                        st.code((run.get("stdout") or "(no output)")[:3000], language="text")
                        if run.get("error"):
                            st.code(run["error"], language="text")
                        for fig in run.get("fig_bytes", []):
                            st.image(fig, width="stretch")

            report = {"participant": name, "email": email, "problem": problem["title"], "score": total_score,
                      "max_score": total_max, "percentage": round(pct, 1),
                      "started_at": fmt_ts(started_at), "total_time": fmt_dur(total_time),
                      "generated_at": fmt_ts(time.time()),
                      "tasks": {tid_: {k: v for k, v in s.items() if k != "run"} for tid_, s in subs.items()}}
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download report (CSV)", pd.DataFrame(report_rows).to_csv(index=False),
                               file_name=f"{pid}_{pkey or 'participant'}_report.csv", width="stretch")
            d2.download_button("⬇️ Download report (JSON)", json.dumps(report, indent=2, default=str),
                               file_name=f"{pid}_{pkey or 'participant'}_report.json", width="stretch")

        if S["show_leaderboard"]:
            with st.expander("🏁 Leaderboard"):
                lb = STORE.submissions()
                lb = lb[lb["problem"] == pid]
                if len(lb):
                    best = lb.groupby(["name", "task"]).last().reset_index()
                    board = (best.groupby("name").agg(score=("score", "sum"), tasks=("task", "nunique"),
                                                      time_s=("time_taken_s", "sum"))
                             .sort_values(["score", "time_s"], ascending=[False, True]).reset_index())
                    board["time taken"] = board.pop("time_s").map(fmt_dur)
                    board.insert(0, "Rank", range(1, len(board) + 1))
                    st.dataframe(board, hide_index=True, width="stretch")
                else:
                    st.caption("No submissions for this problem yet.")

# ───────────────────────── 5. Trainer dashboard ─────────────────────────
def load_records(problem_id: str) -> list[dict]:
    recs = STORE.list(problem_id)
    for d in recs:
        d["_file"] = d["_key"]
    return recs


def trainer_dashboard():
    recs = load_records(pid)
    now = time.time()
    if not recs:
        st.info("No participants have started this problem yet. Share the app link — people appear here "
                "as soon as they click **Start assessment**.")
        return

    people, task_rows, crit_rows = [], [], []
    for d in recs:
        sb, pr = d.get("subs", {}), d.get("progress", {})
        st_at = pr.get("started_at")
        end = st_at + duration_min * 60 if st_at else None
        n = len(sb)
        score = sum(x["grade"]["score"] for x in sb.values())
        status = ("✅ Completed" if n == len(tasks) else "⏰ Time up" if end and now >= end
                  else "🟢 In progress" if st_at else "⚪ Not started")
        on_task = next((f"Task {i}" for i, t in enumerate(tasks, 1) if t["id"] not in sb), "—")
        people.append({"Name": d.get("name"), "Email / ID": d.get("email"), "Status": status,
                       "Tasks done": f"{n}/{len(tasks)}", "Working on": on_task if status == "🟢 In progress" else "—",
                       "Score": round(score, 1), "Score %": round(100 * score / total_max, 1) if total_max else 0,
                       "Time taken": fmt_dur(sum(x.get("time_taken_s") or 0 for x in sb.values())),
                       "Time left": fmt_dur(max(0, end - now)) if end and status == "🟢 In progress" else "—",
                       "Started": fmt_clock(st_at), "Last activity": fmt_clock(pr.get("saved_at")), "_file": d["_file"]})
        for i, t in enumerate(tasks, 1):
            x = sb.get(t["id"])
            if not x:
                continue
            g = x["grade"]
            task_rows.append({"Task": f"T{i} {t['title']}", "pct": 100 * g["score"] / g["max_score"] if g["max_score"] else 0,
                              "min": (x.get("time_taken_s") or 0) / 60, "attempts": x.get("attempts", 1)})
            for r in g["rows"]:
                crit_rows.append({"Task": f"T{i}", "Criterion": r["criterion"], "Evaluated by": r["method"],
                                  "pct": 100 * r["score"] / r["max"] if r["max"] else 0,
                                  "full": r["score"] >= r["max"] - 1e-9,
                                  "Name": d.get("name"), "Email / ID": d.get("email"), "Score": r["score"], "Max": r["max"]})

    dfp = pd.DataFrame(people).sort_values(["Score", "Name"], ascending=[False, True])
    completed = dfp[dfp["Status"] == "✅ Completed"]
    m = st.columns(5)
    m[0].metric("Participants", len(dfp))
    m[1].metric("In progress", int((dfp["Status"] == "🟢 In progress").sum()))
    m[2].metric("Completed", len(completed))
    m[3].metric("Average score", f"{dfp['Score %'].mean():.0f}%")
    m[4].metric("Top score", f"{dfp['Score'].max():g} / {total_max:g}")

    st.markdown("##### 👥 Participants")
    st.dataframe(dfp.drop(columns="_file"), hide_index=True, width="stretch",
                 column_config={"Score %": st.column_config.ProgressColumn("Score %", min_value=0, max_value=100,
                                                                           format="%.0f%%")})

    by_task = pd.DataFrame(columns=["Avg score %", "Avg time (min)", "Submissions", "Avg attempts"])
    bins = pd.cut(dfp["Score %"], bins=[-0.1, 20, 40, 60, 80, 100],
                  labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
    dist = bins.value_counts().sort_index().rename("Participants")
    hard = pd.DataFrame(columns=["Task", "Criterion", "Evaluated by", "Avg points earned %", "Got full marks %", "Attempts"])
    if task_rows:
        dft = pd.DataFrame(task_rows)
        by_task = dft.groupby("Task").agg(**{"Avg score %": ("pct", "mean"), "Avg time (min)": ("min", "mean"),
                                             "Submissions": ("pct", "size"), "Avg attempts": ("attempts", "mean")})
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 📊 Average score by task (%)")
            st.bar_chart(by_task["Avg score %"], height=260, color="#2a78d6", horizontal=True)
        with c2:
            st.markdown("##### ⏱️ Average time per task (min)")
            st.bar_chart(by_task["Avg time (min)"], height=260, color="#2a78d6", horizontal=True)
        with c3:
            st.markdown("##### 📈 Score distribution")
            st.bar_chart(dist, height=260, color="#2a78d6")

        st.markdown("##### 🎯 Where participants struggle")
        dfc = pd.DataFrame(crit_rows)
        hard = (dfc.groupby(["Task", "Criterion", "Evaluated by"])
                .agg(**{"Avg points earned %": ("pct", "mean"), "Got full marks %": ("full", lambda x: 100 * x.mean()),
                        "Attempts": ("pct", "size")})
                .reset_index().sort_values("Avg points earned %").head(10))
        st.dataframe(hard, hide_index=True, width="stretch",
                     column_config={"Avg points earned %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                                    "Got full marks %": st.column_config.NumberColumn(format="%.0f%%")})
        st.caption("Lowest-scoring criteria first — good topics to re-teach or clarify in the task wording.")

    st.markdown("##### 🔎 Participant details")
    pick = st.selectbox("Choose a participant", dfp["_file"],
                        format_func=lambda f: " · ".join(str(v) for v in dfp.loc[dfp["_file"] == f, ["Name", "Email / ID", "Status"]].iloc[0]))
    rec = next(d for d in recs if d["_file"] == pick)
    for i, t in enumerate(tasks, 1):
        x = rec.get("subs", {}).get(t["id"])
        draft = rec.get("progress", {}).get("drafts", {}).get(t["id"])
        if x:
            g = x["grade"]
            with st.expander(f"Task {i} — {t['title']} · {g['score']:g}/{g['max_score']:g} · "
                             f"⏱️ {fmt_dur(x.get('time_taken_s'))} · attempt {x.get('attempts', 1)}"):
                st.dataframe(pd.DataFrame(g["rows"])[["criterion", "score", "max", "method", "justification"]],
                             hide_index=True, width="stretch")
                if g.get("feedback"):
                    st.markdown(f"**Feedback:** {g['feedback']}")
                a, b = st.columns(2)
                a.code(x["code"], language="python")
                b.code((x.get("run", {}).get("stdout") or "(no output)")[:3000], language="text")
                for fig in x.get("run", {}).get("fig_b64", []):
                    b.image(base64.b64decode(fig), width="stretch")
        elif draft:
            with st.expander(f"Task {i} — {t['title']} · not submitted (draft)"):
                st.code(draft, language="python")

    st.markdown("##### ⬇️ Download dashboard")
    log = STORE.submissions()
    stamp = datetime.now(TZ)
    meta = {"app_name": APP_NAME, "problem": problem["title"], "domain": problem.get("domain", ""),
            "generated": stamp.strftime("%d %b %Y, %I:%M %p"), "n_tasks": len(tasks), "total_max": total_max}
    people_out = dfp.drop(columns="_file")
    crit_out = pd.DataFrame(crit_rows).drop(columns=["pct", "full"], errors="ignore")
    log_out = log[log["problem"] == pid] if len(log) and "problem" in log.columns else log
    fname = f"{slug(APP_NAME)}_{pid}_{stamp.strftime('%Y%m%d_%H%M')}"
    r1, r2 = st.columns(2)
    try:
        r1.download_button("📊 Dashboard — Excel (.xlsx)", build_excel(meta, people_out, by_task, dist, hard, crit_out, log_out),
                           file_name=f"{fname}.xlsx", type="primary", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           help="Headline numbers and charts, plus a sheet for every table")
    except Exception as e:
        r1.warning(f"Excel report unavailable ({type(e).__name__}).")
    r2.download_button("🖨️ Dashboard — printable report (HTML → PDF)", build_html(meta, people_out, by_task, dist, hard),
                       file_name=f"{fname}.html", mime="text/html", width="stretch",
                       help="Open the file and click 'Print / Save as PDF'")
    st.caption("Raw data (CSV)")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Participant summary (CSV)", dfp.drop(columns="_file").to_csv(index=False),
                       file_name=f"{pid}_summary.csv", width="stretch")
    if crit_rows:
        d2.download_button("Criterion-level scores (CSV)", pd.DataFrame(crit_rows).to_csv(index=False),
                           file_name=f"{pid}_criteria.csv", width="stretch")
    if len(log):
        d3.download_button("Submission log (CSV)", log.to_csv(index=False),
                           file_name="submission_log.csv", width="stretch")


if is_trainer:
    with tab_dash:
        h1, h2, h3 = st.columns([3, 1, 1])
        h1.caption(f"Live view of **{problem['title']}** · {fmt_clock(time.time())}")
        auto = h2.toggle("Auto-refresh (30s)", value=True)
        if h3.button("🔄 Refresh now", width="stretch"):
            st.rerun()
        st.fragment(run_every=30 if auto else None)(trainer_dashboard)()
