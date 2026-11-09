"""
Storage for participant progress and the submission log.

* PostgresStorage — used when DATABASE_URL is set (Supabase, Neon, or your own PostgreSQL).
  Data survives app restarts and redeploys, and is never on the app's disk where code could reach it.
* FileStorage     — fallback: JSON/CSV files under results/ (lost when Streamlit Cloud restarts the app).

Both expose the same methods, so the app does not care which one is active.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

SUBMISSION_COLUMNS = ["submitted_at", "name", "email", "problem", "task", "score", "max", "attempt", "time_taken_s"]


class FileStorage:
    name = "Local files (temporary)"
    persistent = False

    def __init__(self, root: Path):
        self.dir = root / "results" / "progress"
        self.log = root / "results" / "leaderboard_v2.csv"

    def _f(self, pkey: str, pid: str) -> Path:
        return self.dir / f"{pkey}__{pid}.json"

    def get(self, pkey: str, pid: str) -> dict | None:
        f = self._f(pkey, pid)
        try:
            return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        except Exception:
            return None

    def put(self, pkey: str, pid: str, record: dict) -> bool:
        """Saves unless the ID already belongs to someone with a different PIN."""
        existing = self.get(pkey, pid)
        if existing is not None and existing.get("pin_hash", "") != record.get("pin_hash", ""):
            return False
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self._f(pkey, pid).with_suffix(".tmp")
            tmp.write_text(json.dumps(record, default=str), encoding="utf-8")
            tmp.replace(self._f(pkey, pid))  # atomic swap: never a half-written file
            return True
        except Exception:
            return False

    def delete(self, pkey: str, pid: str):
        self._f(pkey, pid).unlink(missing_ok=True)

    def list(self, pid: str | None = None) -> list[dict]:
        out = []
        if not self.dir.exists():
            return out
        for f in sorted(self.dir.glob(f"*__{pid}.json" if pid else "*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                d["_key"] = f.stem.rsplit("__", 1)[0]
                d["_pid"] = f.stem.rsplit("__", 1)[1]
                out.append(d)
            except Exception:
                pass
        return out

    def log_submission(self, row: dict):
        try:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{c: row.get(c) for c in SUBMISSION_COLUMNS}]).to_csv(
                self.log, mode="a", header=not self.log.exists(), index=False)
        except Exception:
            pass

    def submissions(self) -> pd.DataFrame:
        if self.log.exists():
            try:
                return pd.read_csv(self.log)
            except Exception:
                pass
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)


class PostgresStorage:
    name = "PostgreSQL database"
    persistent = True

    def __init__(self, url: str):
        import psycopg  # noqa: F401  (fail early with a clear error if the driver is missing)
        self.url = url
        self._lock = threading.Lock()
        self._conn = None
        self._exec("""
            CREATE TABLE IF NOT EXISTS eval_progress (
                pkey TEXT NOT NULL, problem TEXT NOT NULL, data JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (pkey, problem));
            CREATE TABLE IF NOT EXISTS eval_submissions (
                id BIGSERIAL PRIMARY KEY, submitted_at TEXT, name TEXT, email TEXT, problem TEXT, task TEXT,
                score DOUBLE PRECISION, max DOUBLE PRECISION, attempt INT, time_taken_s INT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now());
            CREATE INDEX IF NOT EXISTS eval_submissions_problem ON eval_submissions (problem);
        """)

    def _connect(self):
        import psycopg
        # prepare_threshold=None keeps it compatible with connection poolers (Supabase, Neon, PgBouncer)
        self._conn = psycopg.connect(self.url, autocommit=True, connect_timeout=10, prepare_threshold=None)

    def _exec(self, sql: str, params=None, fetch: bool = False):
        with self._lock:
            for attempt in (1, 2):  # reconnect once if the connection dropped
                try:
                    if self._conn is None or self._conn.closed:
                        self._connect()
                    cur = self._conn.execute(sql, params)
                    return (cur.fetchall() if fetch else cur.rowcount)
                except Exception:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    if attempt == 2:
                        raise

    def get(self, pkey: str, pid: str) -> dict | None:
        rows = self._exec("SELECT data FROM eval_progress WHERE pkey=%s AND problem=%s", (pkey, pid), fetch=True)
        return rows[0][0] if rows else None

    def put(self, pkey: str, pid: str, record: dict) -> bool:
        """Atomic upsert that refuses to overwrite an ID owned by a different PIN."""
        try:
            n = self._exec("""
                INSERT INTO eval_progress (pkey, problem, data) VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (pkey, problem) DO UPDATE SET data = EXCLUDED.data, updated_at = now()
                WHERE eval_progress.data->>'pin_hash' = EXCLUDED.data->>'pin_hash'
            """, (pkey, pid, json.dumps(record, default=str)))
            return n > 0
        except Exception:
            return False

    def delete(self, pkey: str, pid: str):
        self._exec("DELETE FROM eval_progress WHERE pkey=%s AND problem=%s", (pkey, pid))

    def list(self, pid: str | None = None) -> list[dict]:
        if pid:
            rows = self._exec("SELECT pkey, problem, data FROM eval_progress WHERE problem=%s ORDER BY pkey",
                              (pid,), fetch=True)
        else:
            rows = self._exec("SELECT pkey, problem, data FROM eval_progress ORDER BY problem, pkey", fetch=True)
        out = []
        for pkey, prob, data in rows:
            data["_key"], data["_pid"] = pkey, prob
            out.append(data)
        return out

    def log_submission(self, row: dict):
        try:
            self._exec(f"INSERT INTO eval_submissions ({', '.join(SUBMISSION_COLUMNS)}) "
                       f"VALUES ({', '.join(['%s'] * len(SUBMISSION_COLUMNS))})",
                       tuple(row.get(c) for c in SUBMISSION_COLUMNS))
        except Exception:
            pass

    def submissions(self) -> pd.DataFrame:
        rows = self._exec(f"SELECT {', '.join(SUBMISSION_COLUMNS)} FROM eval_submissions ORDER BY id", fetch=True)
        return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def make_storage(database_url: str, root: Path):
    """PostgreSQL if a DATABASE_URL is configured and reachable, else local files. Returns (storage, warning)."""
    if database_url:
        try:
            return PostgresStorage(database_url), None
        except Exception as e:
            return FileStorage(root), f"Database unavailable ({type(e).__name__}) — using temporary local files."
    return FileStorage(root), None
