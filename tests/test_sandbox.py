"""Participant code must not reach app files, secrets, other participants, processes or the network."""
import pytest

from evaluator.runner import ROOT, run_code

BLOCKED = {
    "read app code": f"print(open(r'{ROOT / 'app.py'}').read())",
    "read via relative path": "print(open('../../../../../../' + __import__('os').getcwd().split('/')[1]).read())",
    "list app folder": f"import os; print(os.listdir(r'{ROOT}'))",
    "read problems": f"import glob; print(glob.glob(r'{ROOT / 'problems'}/*'))",
    "parent process env": "import os; print(open(f'/proc/{os.getppid()}/environ').read())",
    "shell command": "import os; os.system('id')",
    "subprocess": "import subprocess; subprocess.run(['id'])",
    "network": "import urllib.request; urllib.request.urlopen('http://example.com')",
    "load libc": "import ctypes; ctypes.CDLL(None)",
    "overwrite dataset": "open('data/loan_data.csv', 'w').write('x')",
    "memory scan": "import gc; gc.get_objects()",
}

ALLOWED = {
    "read dataset (both paths)": "import pandas as pd; print(pd.read_csv('loan_data.csv').shape, pd.read_csv('data/loan_data.csv').shape)",
    "write own file": "open('out.txt', 'w').write('hi'); print(open('out.txt').read())",
    "temp files": "import tempfile; f = tempfile.NamedTemporaryFile(delete=False); f.write(b'x'); f.close(); print('ok')",
    "parallel model": "import numpy as np; from sklearn.ensemble import RandomForestClassifier as R; X=np.random.rand(200,3); R(20, n_jobs=-1).fit(X, X[:,0]>.5); print('ok')",
    "hist gradient boosting + kmeans": "import numpy as np; from sklearn.ensemble import HistGradientBoostingClassifier as H; from sklearn.cluster import KMeans; X=np.random.rand(200,3); H().fit(X, X[:,0]>.5); KMeans(3, n_init=2).fit(X); print('ok')",
    "chart": "import matplotlib.pyplot as plt; plt.plot([1,2]); plt.show(); print('ok')",
    "mock llm": "print(ask_llm('Complaint: my card was charged twice'))",
}


@pytest.mark.parametrize("name", BLOCKED)
def test_dangerous_code_is_blocked(name):
    res = run_code(BLOCKED[name], [], [], timeout=60)
    assert res["error"], f"'{name}' was NOT blocked. Output: {res['stdout'][:200]}"


@pytest.mark.parametrize("name", ALLOWED)
def test_normal_code_still_works(name):
    res = run_code(ALLOWED[name], [], [], timeout=90)
    assert not res["error"], f"'{name}' failed:\n{res['error']}"


def test_chart_is_captured():
    res = run_code("import matplotlib.pyplot as plt; plt.plot([1,2]); plt.show()", [], [], timeout=60)
    assert len(res["fig_bytes"]) == 1 and res["fig_bytes"][0][:4] == b"\x89PNG"


def test_infinite_loop_times_out():
    res = run_code("while True: pass", [], [], timeout=3)
    assert "Timed out" in res["error"]


def test_memory_limit():
    res = run_code("x = bytearray(4 * 1024**3)", [], [], timeout=30, max_memory_mb=1024)
    assert res["error"] and "Memory" in res["error"]
