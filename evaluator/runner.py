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

payload = json.load(open(sys.argv[1], encoding="utf-8"))
figdir = payload["figdir"]

# ---------- resource limits for this process ----------
try:
    import resource
    # Address-space cap: the participant's budget plus 1 GB headroom that maths libraries (OpenBLAS) reserve
    mem = (int(payload.get("max_memory_mb", 1536)) + 1024) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))
    cpu = int(payload.get("timeout", 60)) + 5
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
except Exception:
    pass

# Pre-load common libraries before the sandbox switches on
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Other libraries (pandas, scikit-learn, ...) load only when the participant's code imports them,
# so a quick Run stays quick. The sandbox allows reading installed packages.
try:
    import numpy  # noqa: F401  (tiny and used by almost every task)
except Exception:
    pass
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
# The API key never enters this process: a separate helper process holds it and we talk to it over a pipe.
# On Windows (local testing only) pipes can't be handed to a child process, so the call is made directly.
_llm_pipe = payload.get("llm_fds")
_llm_out = os.fdopen(_llm_pipe[0], "w", encoding="utf-8") if _llm_pipe else None
_llm_in = os.fdopen(_llm_pipe[1], "r", encoding="utf-8") if _llm_pipe else None
_llm_direct = None
_net_ok = [False]
if not _llm_pipe and os.environ.get("EVAL_LLM_KEY"):
    try:
        from openai import OpenAI as _OpenAI
        _llm_direct = (_OpenAI(api_key=os.environ.pop("EVAL_LLM_KEY"),
                               base_url=os.environ.pop("EVAL_LLM_BASE_URL", "") or None),
                       os.environ.pop("EVAL_LLM_MODEL", "") or "gpt-4o-mini")
    except Exception:
        _llm_direct = None

def ask_llm(prompt, system=None, temperature=0.0):
    """Call the configured LLM. Falls back to a keyword-based mock when no API key is set."""
    if _llm_out is not None:
        _llm_out.write(json.dumps({"prompt": str(prompt), "system": system, "temperature": temperature}) + "\n")
        _llm_out.flush()
        reply = json.loads(_llm_in.readline() or '{"error": "LLM helper stopped"}')
        if "error" in reply:
            raise RuntimeError("LLM call failed: " + reply["error"])
        return reply["text"]
    if _llm_direct is not None:
        client, model = _llm_direct
        msgs = ([{"role": "system", "content": str(system)}] if system else []) + \
               [{"role": "user", "content": str(prompt)[:20000]}]
        _net_ok[0] = True
        try:
            out = client.chat.completions.create(model=model, messages=msgs, temperature=float(temperature or 0))
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {type(e).__name__}: {e}"[:300]) from None
        finally:
            _net_ok[0] = False
        return out.choices[0].message.content or ""
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

# ---------- sandbox: block file, process and network access outside the participant's own space ----------
def _install_sandbox(own_dirs, data_dir, deny_dirs):
    import site, tempfile
    def real(p):
        return os.path.normcase(os.path.realpath(p))
    own = tuple(real(d) for d in own_dirs)
    data = real(data_dir) if data_dir else None
    deny = tuple(real(d) for d in deny_dirs if d)
    sys_dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix, "/usr", "/lib", "/lib64", "/opt/conda",
                "/etc/localtime", "/etc/timezone", "/etc/ssl", "/etc/fonts", "/etc/matplotlibrc", "/etc/ld.so.cache",
                "/proc/self", "/proc/cpuinfo", "/proc/meminfo", "/proc/stat", "/sys/devices/system/cpu",
                "/sys/fs/cgroup", "/dev/null", "/dev/urandom", "/dev/random", "/dev/shm"}
    if os.name == "nt":  # Windows fonts and time-zone data
        windir = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
        sys_dirs.update({os.path.join(windir, "Fonts"), "NUL", "nul"})
    try:
        sys_dirs.update(site.getsitepackages())
    except Exception:
        pass
    sys_dirs = tuple(real(d) for d in sys_dirs if d)
    blocked_events = {"os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.kill",
                      "os.killpg", "subprocess.Popen", "pty.spawn", "socket.connect", "socket.bind", "socket.sendto",
                      "socket.sendmsg", "socket.getaddrinfo", "socket.gethostbyname", "gc.get_objects",
                      "gc.get_referrers", "gc.get_referents", "sys._current_frames", "sys.remote_exec",
                      "ctypes.string_at", "ctypes.wstring_at", "os.chdir", "webbrowser.open"}
    path_events_read = {"open", "os.listdir", "os.scandir", "glob.glob", "os.chdir", "shutil.copyfile"}
    path_events_write = {"os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace", "os.mkdir", "os.chmod",
                         "os.chown", "os.symlink", "os.link", "shutil.rmtree", "shutil.move", "os.truncate", "os.utime"}

    proc_ok = ("/proc/self", f"/proc/{os.getpid()}", "/proc/thread-self", "/proc/cpuinfo", "/proc/meminfo", "/proc/stat")

    sep = os.sep

    def under(p, roots):
        return any(p == r or p.startswith(r.rstrip(sep) + sep) for r in roots)

    # Python's own libraries may sit inside the app folder (e.g. a "venv" folder next to app.py): allow those.
    sys_inside_app = tuple(d for d in sys_dirs if under(d, deny))

    def can_read(p):
        if data and under(p, (data,)):
            return True
        parts = set(p.replace("\\", "/").split("/"))
        if ".streamlit" in parts or ".git" in parts or p.endswith("secrets.toml") or p.endswith(".env"):
            return False
        if under(p, sys_inside_app):
            return True
        if under(p, deny):
            return False
        if os.name != "nt" and p.startswith("/proc/") and not under(p, proc_ok):
            return False
        return under(p, own) or under(p, sys_dirs)

    def can_write(p):
        return under(p, own) and not (data and under(p, (data,)))

    def deny_msg(what):
        raise PermissionError(f"🔒 Not allowed in this assessment: {what}")

    def to_path(x):
        if isinstance(x, int) or x is None:
            return None
        try:
            return real(os.fsdecode(x))
        except Exception:
            return None

    def hook(event, args):
        if _net_ok[0] and (event.startswith("socket.") or event in ("open", "ctypes.dlopen", "ctypes.dlsym")):
            return  # inside ask_llm() on Windows: let the LLM client reach its API
        if event in blocked_events or event.startswith("os.exec") or event.startswith("os.spawn"):
            if event.startswith("socket"):
                deny_msg("network access (use ask_llm() for LLM calls)")
            if event in ("subprocess.Popen", "os.system", "os.fork", "os.posix_spawn") or "exec" in event or "spawn" in event:
                deny_msg("starting other programs or processes")
            deny_msg(event)
        if event == "ctypes.dlsym":  # maths libraries look up their own functions; block OS-level ones
            name = args[1] if len(args) > 1 else ""
            name = name.decode() if isinstance(name, bytes) else str(name)
            if re.fullmatch(r"_*(system|popen|exec[lv]?p?e?|fexecve|fork|vfork|clone3?|posix_spawnp?|open(at)?|fopen|"
                            r"creat|dlopen|dlmopen|dlsym|syscall|mmap|mprotect|ptrace|kill|socket|connect|unlink(at)?|"
                            r"rename(at)?|chmod|setuid|getenv|secure_getenv|environ|read|write|pread|pwrite)(64)?",
                            name):
                deny_msg("calling operating-system functions")
            return
        if event == "ctypes.dlopen":
            name = args[0] if args else None
            n = name.replace("\\", "/").lower() if isinstance(name, str) else ""
            if not (n and ("site-packages" in n or ".libs/" in n or "dist-packages" in n)):
                deny_msg("loading system libraries")
            return
        if event in path_events_read or event in path_events_write:
            if not args:
                return
            paths = [args[0]] + ([args[1]] if event in ("os.rename", "os.replace", "os.symlink", "os.link",
                                                       "shutil.move", "shutil.copyfile") and len(args) > 1 else [])
            writing = event in path_events_write
            if event == "open" and len(args) > 1:
                mode, flags = args[1], (args[2] if len(args) > 2 else 0)
                if (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                   (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                    writing = True
            for raw in paths:
                p = to_path(raw)
                if p is None:
                    continue
                if writing and not can_write(p):
                    deny_msg(f"writing outside your work folder ({os.fsdecode(raw)})")
                if not writing and not can_read(p):
                    deny_msg(f"reading files outside the data folder ({os.fsdecode(raw)})")

    sys.addaudithook(hook)


ns = {"__name__": "__main__", "ask_llm": ask_llm,
      "_close": _close, "_model": _model, "_auc_consistent": _auc_consistent}

if payload.get("sandbox", True):
    _install_sandbox(payload["own_dirs"], payload.get("data_dir"), payload.get("deny_dirs", []))

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
ns["_stdout"] = out.getvalue()

def _printed(pattern, flags=re.I):
    """True if the participant's printed output matches the regular expression."""
    return re.search(pattern, ns["_stdout"], flags) is not None

ns["_printed"] = _printed

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


LLM_HELPER = r"""
import sys, os, json
from openai import OpenAI
client = OpenAI(api_key=os.environ["LLM_KEY"], base_url=os.environ.get("LLM_BASE_URL") or None)
model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
limit = int(os.environ.get("LLM_MAX_CALLS", "60"))
for n, line in enumerate(sys.stdin):
    try:
        if n >= limit:
            raise RuntimeError(f"limit of {limit} LLM calls per run reached")
        r = json.loads(line)
        msgs = ([{"role": "system", "content": str(r["system"])}] if r.get("system") else []) + \
               [{"role": "user", "content": str(r["prompt"])[:20000]}]
        out = client.chat.completions.create(model=model, messages=msgs, temperature=float(r.get("temperature") or 0))
        print(json.dumps({"text": out.choices[0].message.content or ""}), flush=True)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"[:300]}), flush=True)
"""


def _deny_dirs() -> list[str]:
    """App folders participant code must never read (code, secrets, saved results)."""
    deny = [str(ROOT)]
    parent = ROOT.parent
    if len(parent.parts) > 2 and ((parent / ".git").exists() or "/mount/src" in str(parent)):
        deny.append(str(parent))  # repository root on Streamlit Cloud
    return deny


def run_code(code: str, prior_codes: list[str], checks: list[dict], timeout: int = 60,
             llm_env: dict | None = None, sandbox: bool = True, max_memory_mb: int = 1536) -> dict:
    """Execute participant code in an isolated subprocess and return stdout, error, figures and check results."""
    with tempfile.TemporaryDirectory(prefix="evalrun_", ignore_cleanup_errors=True) as tmp:
        tmp_p = Path(tmp)
        workdir, figdir, scratch = tmp_p / "work", tmp_p / "figs", tmp_p / "tmp"
        for d in (workdir, figdir, scratch):
            d.mkdir()

        # Working folder where both "data/loan_data.csv" and plain "loan_data.csv" resolve.
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

        payload = {"code": code, "prior": prior_codes, "checks": checks, "figdir": str(figdir),
                   "sandbox": sandbox, "own_dirs": [tmp], "data_dir": str(data_dir), "deny_dirs": _deny_dirs(),
                   "timeout": timeout, "max_memory_mb": max_memory_mb}

        # Clean environment: no secrets, private temp/home folders, single-threaded maths libraries.
        env = {k: v for k, v in os.environ.items()
               if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "DATABASE", "DSN"))}
        env.update({"MPLBACKEND": "Agg", "TMPDIR": str(scratch), "TEMP": str(scratch), "TMP": str(scratch),
                    "HOME": str(scratch), "MPLCONFIGDIR": str(scratch), "PYTHONDONTWRITEBYTECODE": "1",
                    "JOBLIB_MULTIPROCESSING": "0", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2", "PYTHONHASHSEED": "0"})

        # LLM helper process holds the API key; participant code only gets a pipe to it.
        helper = None
        pass_fds = ()
        llm_env = llm_env or {}
        use_pipe = os.name != "nt" and not os.environ.get("EVAL_FORCE_DIRECT_LLM")
        if llm_env.get("PARTICIPANT_LLM_API_KEY") and not use_pipe:
            env.update({"EVAL_LLM_KEY": llm_env["PARTICIPANT_LLM_API_KEY"],
                        "EVAL_LLM_BASE_URL": llm_env.get("PARTICIPANT_LLM_BASE_URL", ""),
                        "EVAL_LLM_MODEL": llm_env.get("PARTICIPANT_LLM_MODEL", "")})
            for k in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
                if os.environ.get(k):
                    env[k] = os.environ[k]
        elif llm_env.get("PARTICIPANT_LLM_API_KEY"):
            h_env = {k: v for k, v in env.items()}
            h_env.update({"LLM_KEY": llm_env["PARTICIPANT_LLM_API_KEY"],
                          "LLM_BASE_URL": llm_env.get("PARTICIPANT_LLM_BASE_URL", ""),
                          "LLM_MODEL": llm_env.get("PARTICIPANT_LLM_MODEL", "")})
            for k in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
                if os.environ.get(k):
                    h_env[k] = os.environ[k]
            helper = subprocess.Popen([sys.executable, "-I", "-c", LLM_HELPER], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=h_env, text=True)
            pass_fds = (helper.stdin.fileno(), helper.stdout.fileno())
            payload["llm_fds"] = list(pass_fds)

        payload_path = tmp_p / "payload.json"
        harness_path = tmp_p / "harness.py"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        harness_path.write_text(HARNESS, encoding="utf-8")

        try:
            proc = subprocess.run([sys.executable, "-I", str(harness_path), str(payload_path)],
                                  cwd=str(workdir), capture_output=True, text=True,
                                  timeout=timeout, env=env, **({"pass_fds": pass_fds} if pass_fds else {}))
        except subprocess.TimeoutExpired:
            return {"stdout": "", "error": f"⏱️ Timed out after {timeout}s. Check for infinite loops or very slow code.",
                    "prior_error": None, "figures": [], "fig_bytes": [],
                    "checks": [{"criterion": c["criterion"], "passed": False, "error": "timeout"} for c in checks]}
        finally:
            if helper:
                helper.kill()
                helper.wait()

        raw = proc.stdout
        if MARKER in raw:
            before, _, js = raw.rpartition(MARKER)
            result = json.loads(js.strip())
            result["stdout"] = (before + result["stdout"])[:MAX_STDOUT]
        else:  # process crashed (memory limit, sys.exit, segfault)
            err = proc.stderr[-3000:] or "Process exited unexpectedly."
            if "MemoryError" in err or proc.returncode in (-9, 137):
                err = f"💥 Your code used more than {max_memory_mb} MB of memory and was stopped.\n\n" + err[-1500:]
            result = {"stdout": raw[:MAX_STDOUT], "error": err, "prior_error": None, "figures": [],
                      "checks": [{"criterion": c["criterion"], "passed": False, "error": "not run"} for c in checks]}
        result["fig_bytes"] = [Path(p).read_bytes() for p in result.get("figures", []) if Path(p).exists()]
        result["figures"] = []
    return result
