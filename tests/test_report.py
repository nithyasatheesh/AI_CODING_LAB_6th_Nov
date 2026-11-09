"""Dashboard downloads: the Excel workbook and HTML report build correctly (also with no submissions yet)."""
import io

import openpyxl
import pandas as pd

from evaluator.report import build_excel, build_html

META = {"app_name": "Test Lab", "problem": "Loan Default", "domain": "Banking", "generated": "now",
        "n_tasks": 2, "total_max": 20}
PEOPLE = pd.DataFrame([{"Name": "Asha", "Email / ID": "a@b.c", "Status": "✅ Completed", "Tasks done": "2/2",
                        "Score": 18.0, "Score %": 90.0, "Time taken": "5m 00s"},
                       {"Name": "Ravi <script>", "Email / ID": "r@b.c", "Status": "🟢 In progress", "Tasks done": "1/2",
                        "Score": 6.0, "Score %": 30.0, "Time taken": "2m 00s"}])
BY_TASK = pd.DataFrame({"Avg score %": [80.0, 50.0], "Avg time (min)": [0.5, 4.2], "Submissions": [2, 1],
                        "Avg attempts": [1.0, 1.5]}, index=pd.Index(["T1 Load", "T2 Clean"], name="Task"))
DIST = pd.Series([0, 1, 0, 0, 1], index=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"], name="Participants")
HARD = pd.DataFrame([{"Task": "T2", "Criterion": "All rows retained", "Evaluated by": "Auto check",
                      "Avg points earned %": 50.0, "Got full marks %": 50.0, "Attempts": 2}])
EMPTY_TASK = pd.DataFrame(columns=["Avg score %", "Avg time (min)", "Submissions", "Avg attempts"])
EMPTY_HARD = pd.DataFrame(columns=list(HARD.columns))


def test_excel_has_dashboard_charts_and_sheets():
    wb = openpyxl.load_workbook(io.BytesIO(build_excel(META, PEOPLE, BY_TASK, DIST, HARD, pd.DataFrame(), pd.DataFrame())))
    assert wb.sheetnames[0] == "Dashboard"
    assert {"Participants", "By task", "Where they struggle", "Criterion scores", "Submission log"} <= set(wb.sheetnames)
    assert len(wb["Dashboard"]._charts) == 3
    assert wb["Participants"]["C2"].value == "Completed"  # emoji stripped


def test_excel_with_no_submissions():
    build_excel(META, PEOPLE, EMPTY_TASK, DIST, EMPTY_HARD, pd.DataFrame(), pd.DataFrame())


def test_html_report_is_escaped_and_complete():
    page = build_html(META, PEOPLE, BY_TASK, DIST, HARD)
    assert "<script>" not in page.split("<body>")[1]  # participant text is escaped
    assert "Ravi &lt;script&gt;" in page
    assert page.count("<svg") == 3 and "Print / Save as PDF" in page
    assert "30s" in page and "4.2 min" in page
    build_html(META, PEOPLE, EMPTY_TASK, DIST, EMPTY_HARD)
