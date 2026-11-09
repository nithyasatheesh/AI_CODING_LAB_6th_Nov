"""End-to-end smoke test of the Streamlit app: participant submits a task, trainer sees it on the dashboard."""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
PID = "loan_default_eda_ml"


@pytest.fixture(autouse=True)
def secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_PASSWORD", "test-pass")
    if os.environ.get("TEST_DATABASE_URL"):
        monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])


def test_participant_submit_and_trainer_dashboard():
    code = (ROOT / "tests" / "solutions" / f"{PID}_T1.py").read_text()
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.text_input(key="p_name").set_value("CI Bot").run()
    at.text_input(key="p_email").set_value("ci-bot@test").run()
    at.text_input(key="p_pin").set_value("4321").run()
    next(b for b in at.button if "Start" in b.label).click().run()
    at.session_state["progress"][PID]["drafts"]["T1"] = code
    at.run()
    next(b for b in at.button if "Submit" in b.label).click().run()
    assert not at.exception
    assert at.session_state["subs"][PID]["T1"]["grade"]["score"] > 0

    # someone else using the same ID with another PIN is blocked
    other = AppTest.from_file(APP, default_timeout=60).run()
    other.text_input(key="p_name").set_value("Intruder").run()
    other.text_input(key="p_email").set_value("ci-bot@test").run()
    other.text_input(key="p_pin").set_value("9999").run()
    assert any("Wrong PIN" in e.value for e in other.error)
    assert not other.session_state["subs"].get(PID)

    # trainer
    tr = AppTest.from_file(APP, default_timeout=60).run()
    tr.segmented_control(key="role").set_value("🧑‍🏫 Trainer").run()
    tr.text_input(key="trainer_pw").set_value("test-pass").run()
    assert not tr.exception
    assert tr.tabs[0].label.endswith("Trainer Dashboard")
    assert int(tr.metric[0].value) >= 1
