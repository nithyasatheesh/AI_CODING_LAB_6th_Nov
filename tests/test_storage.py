"""Progress storage: PIN protection and round-trips, for files and (when available) PostgreSQL."""
import os

import pytest

from evaluator.storage import FileStorage, PostgresStorage


def backends(tmp_path):
    out = [FileStorage(tmp_path)]
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        db = PostgresStorage(url)
        db._exec("TRUNCATE eval_progress, eval_submissions")
        out.append(db)
    return out


def test_save_load_and_pin_protection(tmp_path):
    for store in backends(tmp_path):
        rec = {"name": "Asha", "email": "a@b.c", "problem": "p1", "pin_hash": "H1", "progress": {"started_at": 1},
               "subs": {"T1": {"grade": {"score": 5}}}}
        assert store.put("asha", "p1", rec), store.name
        assert store.get("asha", "p1")["subs"]["T1"]["grade"]["score"] == 5
        intruder = dict(rec, pin_hash="H2", subs={})
        assert not store.put("asha", "p1", intruder), f"{store.name}: overwrite with wrong PIN must fail"
        assert store.get("asha", "p1")["subs"], f"{store.name}: data must be unchanged"
        assert store.put("asha", "p1", dict(rec, subs={"T1": {}, "T2": {}})), "owner can update"
        assert [d["_key"] for d in store.list("p1")] == ["asha"]
        store.delete("asha", "p1")
        assert store.get("asha", "p1") is None


def test_submission_log(tmp_path):
    for store in backends(tmp_path):
        store.log_submission({"submitted_at": "now", "name": "A", "email": "a", "problem": "p1", "task": "T1",
                              "score": 7.5, "max": 10, "attempt": 1, "time_taken_s": 60})
        df = store.submissions()
        assert len(df) == 1 and float(df.iloc[0]["score"]) == 7.5, store.name
