"""
Downloadable batch reports for the Trainer Dashboard.

build_excel(...) -> .xlsx bytes : Dashboard sheet (headline numbers + native Excel charts) and one sheet per table
build_html(...)  -> HTML string : a printable, self-contained snapshot of the dashboard (Print → Save as PDF)
"""
from __future__ import annotations

import html
import io
import re

import pandas as pd

BLUE = "2a78d6"          # single-series chart colour (reference palette, slot 1)
INK, INK_2, MUTED, GRID = "0b0b0b", "52514e", "8a8984", "e6e5e0"

_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿⬀-⯿️]+\s*")


def plain(text) -> str:
    """Status labels without emoji, for Excel and print."""
    return _EMOJI.sub("", str(text)).strip()


def kpis(dfp: pd.DataFrame, total_max: float) -> list[tuple[str, str]]:
    status = dfp["Status"].map(plain)
    return [("Participants", f"{len(dfp)}"),
            ("Completed", f"{int((status == 'Completed').sum())}"),
            ("In progress", f"{int((status == 'In progress').sum())}"),
            ("Average score", f"{dfp['Score %'].mean():.0f}%" if len(dfp) else "–"),
            ("Top score", f"{dfp['Score'].max():g} / {total_max:g}" if len(dfp) else "–")]


# ───────────────────────────── Excel ─────────────────────────────
def build_excel(meta: dict, dfp: pd.DataFrame, by_task: pd.DataFrame, dist: pd.Series,
                hard: pd.DataFrame, crit: pd.DataFrame, log: pd.DataFrame) -> bytes:
    from openpyxl.chart import BarChart, Reference
    from openpyxl.chart.label import DataLabelList
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    people = dfp.copy()
    people["Status"] = people["Status"].map(plain)
    task_tbl = by_task.reset_index().round({"Avg score %": 1, "Avg time (min)": 1, "Avg attempts": 2})
    dist_tbl = dist.rename_axis("Score band").reset_index(name="Participants")
    hard_tbl = hard.round({"Avg points earned %": 1, "Got full marks %": 1})

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame().to_excel(xw, sheet_name="Dashboard")
        people.to_excel(xw, sheet_name="Participants", index=False)
        task_tbl.to_excel(xw, sheet_name="By task", index=False)
        hard_tbl.to_excel(xw, sheet_name="Where they struggle", index=False)
        crit.to_excel(xw, sheet_name="Criterion scores", index=False)
        log.to_excel(xw, sheet_name="Submission log", index=False)
        wb = xw.book

        head_font = Font(bold=True, color="FFFFFF")
        head_fill = PatternFill("solid", fgColor=BLUE)
        thin = Side(style="thin", color=GRID)

        def style_table(ws, n_cols, n_rows, header_row=1, first_col=1):
            for c in range(first_col, first_col + n_cols):
                cell = ws.cell(row=header_row, column=c)
                cell.font, cell.fill = head_font, head_fill
                cell.alignment = Alignment(vertical="center", wrap_text=True)
            for r in range(header_row + 1, header_row + 1 + n_rows):
                for c in range(first_col, first_col + n_cols):
                    ws.cell(row=r, column=c).border = Border(bottom=thin)

        def autosize(ws, min_w=8, max_w=60):
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=min_w)
                ws.column_dimensions[get_column_letter(col[0].column)].width = max(min_w, min(max_w, width + 2))

        for name, df in (("Participants", people), ("By task", task_tbl), ("Where they struggle", hard_tbl),
                         ("Criterion scores", crit), ("Submission log", log)):
            ws = wb[name]
            style_table(ws, len(df.columns), len(df))
            ws.freeze_panes = "A2"
            autosize(ws)
            if len(df):
                ws.auto_filter.ref = ws.dimensions

        # ---------- Dashboard sheet ----------
        ws = wb["Dashboard"]
        ws.sheet_view.showGridLines = False
        ws["B2"] = meta["app_name"]
        ws["B2"].font = Font(size=18, bold=True, color=INK)
        ws["B3"] = f"{meta['problem']} · {meta.get('domain', '')}"
        ws["B3"].font = Font(size=12, color=INK_2)
        ws["B4"] = f"Generated {meta['generated']} · {meta['n_tasks']} tasks · {meta['total_max']:g} points"
        ws["B4"].font = Font(size=10, color=MUTED)

        for i, (label, value) in enumerate(kpis(dfp, meta["total_max"])):
            col = 2 + i * 2
            ws.cell(row=6, column=col, value=label).font = Font(size=10, color=INK_2)
            ws.cell(row=7, column=col, value=value).font = Font(size=20, bold=True, color=INK)
        for c in range(2, 12):
            ws.column_dimensions[get_column_letter(c)].width = 14

        def bar(title, sheet, cat_col, val_col, n, y_title, anchor, number_format="0"):
            ch = BarChart()
            ch.type, ch.style, ch.title = "bar", 10, title
            ch.legend = None
            ch.y_axis.title, ch.y_axis.majorGridlines = y_title, None
            ch.y_axis.delete = False
            ch.x_axis.delete = False
            data = Reference(sheet, min_col=val_col, min_row=1, max_row=n + 1)
            cats = Reference(sheet, min_col=cat_col, min_row=2, max_row=n + 1)
            ch.add_data(data, titles_from_data=True)
            ch.set_categories(cats)
            s = ch.series[0]
            s.graphicalProperties.solidFill = BLUE
            s.graphicalProperties.line.solidFill = BLUE
            s.dLbls = DataLabelList()
            s.dLbls.showVal, s.dLbls.numFmt = True, number_format
            s.dLbls.showSerName = s.dLbls.showCatName = s.dLbls.showLegendKey = s.dLbls.showPercent = False
            ch.x_axis.scaling.orientation = "maxMin"  # first task at the top
            ch.gapWidth = 60
            ch.height, ch.width = 7.5, 15
            ws.add_chart(ch, anchor)

        if len(task_tbl):
            bt = wb["By task"]
            bar("Average score by task (%)", bt, 1, 2, len(task_tbl), "%", "B10", "0")
            bar("Average time per task (min)", bt, 1, 3, len(task_tbl), "minutes", "J10", "0.0")
        ds = wb.create_sheet("Score bands")
        ds.append(["Score band", "Participants"])
        for row in dist_tbl.itertuples(index=False):
            ds.append(list(row))
        style_table(ds, 2, len(dist_tbl))
        autosize(ds)
        bar("Participants by score band", ds, 1, 2, len(dist_tbl), "participants", "B26", "0")

        ws.cell(row=42, column=2, value="Where participants struggle (lowest-scoring criteria first)").font = \
            Font(size=12, bold=True, color=INK)
        for j, h in enumerate(hard_tbl.columns):
            ws.cell(row=43, column=2 + j, value=h)
        for i, row in enumerate(hard_tbl.itertuples(index=False)):
            for j, v in enumerate(row):
                ws.cell(row=44 + i, column=2 + j, value=v)
        style_table(ws, len(hard_tbl.columns), len(hard_tbl), header_row=43, first_col=2)
        ws.column_dimensions["C"].width = 40
        wb.move_sheet("Score bands", offset=-(len(wb.sheetnames) - 3))
    return buf.getvalue()


# ───────────────────────────── HTML ─────────────────────────────
def _fmt_minutes(v: float) -> str:
    return f"{v * 60:.0f}s" if v < 1 else f"{v:.1f} min"


def _svg_bars(labels: list[str], values: list[float], unit: str, fmt: str = "{:.0f}", fmt_fn=None) -> str:
    """Horizontal single-series bar chart as inline SVG (one hue, value labels, hover titles)."""
    if not values:
        return "<p class='muted'>No data yet.</p>"
    row_h, gap, label_w, w = 24, 8, 150, 380
    plot_w = w - label_w - 58
    vmax = max(max(values), 1e-9)
    h = len(values) * (row_h + gap) + 6
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" class="chart">']
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = i * (row_h + gap) + 3
        bw = max(2.0, plot_w * v / vmax)
        lab_s = html.escape(lab if len(lab) <= 22 else lab[:21] + "…")
        val_s = fmt_fn(v) if fmt_fn else fmt.format(v) + unit
        out.append(f'<g><title>{html.escape(lab)}: {val_s}</title>'
                   f'<text x="{label_w - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" class="lbl">{lab_s}</text>'
                   f'<rect x="{label_w}" y="{y}" width="{bw:.1f}" height="{row_h}" rx="4" class="bar"/>'
                   f'<text x="{label_w + bw + 6}" y="{y + row_h / 2 + 4}" class="val">{html.escape(val_s)}</text></g>')
    out.append(f'<line x1="{label_w}" x2="{label_w}" y1="0" y2="{h}" class="axis"/></svg>')
    return "".join(out)


def _table(df: pd.DataFrame, pct_cols=(), max_rows=None) -> str:
    if df.empty:
        return "<p class='muted'>No data yet.</p>"
    rows = df if max_rows is None else df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in rows.columns)
    body = []
    for r in rows.itertuples(index=False):
        cells = []
        for c, v in zip(rows.columns, r):
            if c in pct_cols and isinstance(v, (int, float)):
                cells.append(f'<td class="num"><span class="pbar"><i style="width:{max(0, min(100, v)):.0f}%"></i></span>'
                             f'{v:.0f}%</td>')
            elif isinstance(v, (int, float)):
                cells.append(f'<td class="num">{v:g}</td>')
            else:
                cells.append(f"<td>{html.escape(plain(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_html(meta: dict, dfp: pd.DataFrame, by_task: pd.DataFrame, dist: pd.Series, hard: pd.DataFrame) -> str:
    tiles = "".join(f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'
                    for k, v in kpis(dfp, meta["total_max"]))
    t_labels = list(by_task.index) if len(by_task) else []
    score_chart = _svg_bars(t_labels, list(by_task["Avg score %"]) if len(by_task) else [], "%")
    time_chart = _svg_bars(t_labels, list(by_task["Avg time (min)"]) if len(by_task) else [], "", fmt_fn=_fmt_minutes)
    dist_chart = _svg_bars([str(x) for x in dist.index], [float(x) for x in dist.values], "", "{:.0f}")
    people = dfp.drop(columns=[c for c in ("_file",) if c in dfp.columns])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(meta['app_name'])} — Batch Report</title>
<style>
:root {{ --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --grid:#e6e5e0;
        --bar:#2a78d6; --track:#e8eef8; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{
  --surface:#1a1a19; --card:#232322; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e86; --grid:#3a3936; --bar:#3987e5; --track:#2b3442; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--surface); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:28px 16px 48px; }}
h1 {{ font-size:26px; margin:0 0 2px; }} h2 {{ font-size:16px; margin:28px 0 10px; }}
.sub {{ color:var(--ink2); margin:0; }} .muted {{ color:var(--muted); }}
.bar-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap; }}
button {{ font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--grid); background:var(--card); color:var(--ink); cursor:pointer; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-top:20px; }}
.tile {{ background:var(--card); border:1px solid var(--grid); border-radius:12px; padding:12px 14px; }}
.tile .k {{ color:var(--ink2); font-size:12px; }} .tile .v {{ font-size:24px; font-weight:700; }}
.grid3 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }}
.card {{ background:var(--card); border:1px solid var(--grid); border-radius:12px; padding:12px 14px; }}
.card h3 {{ font-size:13px; margin:0 0 8px; color:var(--ink2); font-weight:600; }}
.chart .bar {{ fill:var(--bar); }} .chart .lbl {{ fill:var(--ink2); font-size:12px; }}
.chart .val {{ fill:var(--ink); font-size:12px; font-variant-numeric:tabular-nums; }} .chart .axis {{ stroke:var(--grid); }}
.chart g:hover .bar {{ opacity:.8; }}
.scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--grid); border-radius:12px; overflow:hidden; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
th {{ font-size:12px; color:var(--ink2); background:var(--surface); }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.pbar {{ display:inline-block; width:60px; height:6px; background:var(--track); border-radius:3px; margin-right:8px; vertical-align:middle; }}
.pbar i {{ display:block; height:100%; background:var(--bar); border-radius:3px; }}
footer {{ margin-top:28px; color:var(--muted); font-size:12px; }}
@media print {{ button {{ display:none; }} body {{ background:#fff; }} .card, .tile, table {{ break-inside:avoid; }}
  main {{ padding:0; }} th, td {{ white-space:normal; }} }}
</style></head><body><main>
<div class="bar-top"><div><h1>{html.escape(meta['app_name'])}</h1>
<p class="sub">{html.escape(meta['problem'])} · {html.escape(meta.get('domain', ''))}</p>
<p class="muted">Batch report · generated {html.escape(meta['generated'])} · {meta['n_tasks']} tasks · {meta['total_max']:g} points</p></div>
<button onclick="window.print()">Print / Save as PDF</button></div>
<div class="tiles">{tiles}</div>
<h2>Performance by task</h2>
<div class="grid3">
<div class="card"><h3>Average score by task (%)</h3>{score_chart}</div>
<div class="card"><h3>Average time per task (minutes)</h3>{time_chart}</div>
<div class="card"><h3>Participants by score band</h3>{dist_chart}</div>
</div>
<h2>Where participants struggle</h2>
<p class="muted">Lowest-scoring rubric criteria first: topics to re-teach or clarify.</p>
<div class="scroll">{_table(hard, pct_cols=("Avg points earned %", "Got full marks %"))}</div>
<h2>Participants</h2>
<div class="scroll">{_table(people, pct_cols=("Score %",))}</div>
<footer>{html.escape(meta['app_name'])} · confidential: participant results</footer>
</main></body></html>"""
