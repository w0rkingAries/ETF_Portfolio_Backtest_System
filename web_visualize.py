from __future__ import annotations

import csv
import json
import math
from pathlib import Path


CS_REPORT_PATH = Path("report/cross_sectional_momentum")
TS_REPORT_PATH = Path("report/time_series_momentum")
CS_STRATEGY_PATH = Path("data/strategy/cross_sectional_momentum")
TS_STRATEGY_PATH = Path("data/strategy/time_series_momentum")
OUTPUT_PATH = Path("report/dashboard.html")


def _to_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


def _read_csv_rows(path: Path, max_rows: int | None = None) -> tuple[list[str], list[dict]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            rows = []
            for row in reader:
                clean = {}
                for key, value in row.items():
                    key = key if key not in {None, ""} else "index"
                    parsed = _to_float(value)
                    clean[key] = parsed if parsed is not None else value
                rows.append(clean)
                if max_rows is not None and len(rows) >= max_rows:
                    break
            return [c if c else "index" for c in columns], rows
    except Exception:
        return [], []


def _as_records(path: Path, max_rows: int = 80) -> dict:
    columns, rows = _read_csv_rows(path, max_rows=max_rows)
    if not columns:
        return {"columns": [], "rows": []}
    return {"columns": columns, "rows": rows}


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _summarize_returns(path: Path) -> tuple[list[dict], dict]:
    columns, rows = _read_csv_rows(path)
    if not columns or not rows:
        return [], {"dates": [], "series": []}

    summary = []
    chart_series = []
    date_col = columns[0]

    for col in columns[1:]:
        dated_returns = []
        for row in rows:
            value = row.get(col)
            if isinstance(value, (int, float)):
                dated_returns.append((str(row.get(date_col, "")), float(value)))

        if not dated_returns:
            continue

        returns = [value for _, value in dated_returns]
        total_growth = 1.0
        growth_points = []
        peak = 1.0
        max_drawdown = 0.0

        for date, value in dated_returns:
            total_growth *= 1.0 + value
            peak = max(peak, total_growth)
            max_drawdown = min(max_drawdown, total_growth / peak - 1.0)
            growth_points.append({"date": date, "value": total_growth})

        ann_return = total_growth ** (252 / len(returns)) - 1.0 if total_growth > 0 else None
        std = _sample_std(returns)
        ann_vol = std * math.sqrt(252) if std is not None else None
        sharpe = ann_return / ann_vol if ann_return is not None and ann_vol not in {None, 0} else None

        summary.append({
            "strategy": col,
            "annual_return": ann_return,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        })

        chart_series.append({
            "name": col,
            "points": growth_points,
        })

    summary = sorted(
        summary,
        key=lambda row: row["sharpe"] if row["sharpe"] is not None else -999,
        reverse=True,
    )

    return summary, {"series": chart_series}


def _read_text(path: Path, max_chars: int = 20000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""

    return text[:max_chars]


def _collect_profile(name: str, report_path: Path, strategy_path: Path) -> dict:
    report_csvs = sorted(report_path.glob("*.csv"))
    report_txts = sorted(report_path.glob("*.txt"))
    strategy_csvs = sorted(strategy_path.glob("*/*.csv"))

    return_files = [
        p for p in strategy_csvs
        if p.name in {"strategy_returns.csv", "ts_strategy_returns.csv"}
    ]
    if not return_files:
        return_files = [p for p in strategy_csvs if "strategy_returns" in p.name]

    return_summary = []
    return_charts = []
    for path in return_files:
        summary, chart = _summarize_returns(path)
        return_summary.append({
            "file": str(path),
            "summary": summary,
        })
        return_charts.append({
            "file": str(path),
            "chart": chart,
        })

    tables = []
    for path in report_csvs:
        tables.append({
            "name": path.name,
            "path": str(path),
            "data": _as_records(path),
        })

    texts = []
    for path in report_txts:
        texts.append({
            "name": path.name,
            "path": str(path),
            "text": _read_text(path),
        })

    return {
        "name": name,
        "report_path": str(report_path),
        "strategy_path": str(strategy_path),
        "return_summary": return_summary,
        "return_charts": return_charts,
        "tables": tables,
        "texts": texts,
    }


def _html_template(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ETF Momentum Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d7dce2;
      --accent: #2563eb;
      --accent-soft: #e8f0ff;
      --good: #057a55;
      --bad: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
    h3 {{ margin: 18px 0 10px; font-size: 16px; }}
    .muted {{ color: var(--muted); }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 22px 24px 48px; }}
    .tabs {{ display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 7px;
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 600;
    }}
    button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow: hidden;
    }}
    .card-title {{ font-size: 13px; color: var(--muted); margin-bottom: 4px; }}
    .card-value {{ font-size: 24px; font-weight: 750; }}
    .chart-wrap {{ height: 360px; }}
    svg {{ width: 100%; height: 100%; display: block; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 10px; font-size: 12px; color: var(--muted); }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      background: var(--c);
    }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f2f4f7; color: #344054; position: sticky; top: 0; }}
    .table-scroll {{ overflow: auto; max-height: 520px; border: 1px solid var(--line); border-radius: 7px; }}
    pre {{
      margin: 0;
      padding: 12px;
      overflow: auto;
      max-height: 540px;
      background: #101828;
      color: #f8fafc;
      border-radius: 7px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .section {{ display: none; }}
    .section.active {{ display: block; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .file-label {{ color: var(--muted); font-size: 12px; margin: -4px 0 10px; overflow-wrap: anywhere; }}
    @media (max-width: 900px) {{
      .grid, .two-col {{ grid-template-columns: 1fr; }}
      main {{ padding: 16px; }}
      header {{ padding: 22px 18px 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ETF Momentum Dashboard</h1>
    <div class="muted">Generated from cross-sectional and time-series latest outputs.</div>
  </header>
  <main>
    <div class="tabs" id="tabs"></div>
    <div id="content"></div>
  </main>
  <script>
    const DATA = {data};
    const colors = ["#2563eb", "#059669", "#dc6803", "#7c3aed", "#be123c", "#0891b2", "#4d7c0f", "#9333ea", "#0f766e"];

    function fmt(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      if (typeof value === "number") return Math.abs(value) < 10 ? value.toFixed(3) : value.toFixed(1);
      return value;
    }}

    function table(data) {{
      if (!data || !data.columns || data.columns.length === 0) return '<div class="muted">No data</div>';
      const head = data.columns.map(c => `<th>${{c}}</th>`).join("");
      const body = data.rows.map(row => `<tr>${{data.columns.map(c => `<td>${{fmt(row[c])}}</td>`).join("")}}</tr>`).join("");
      return `<div class="table-scroll"><table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table></div>`;
    }}

    function summaryTable(rows) {{
      return table({{
        columns: ["strategy", "annual_return", "annual_vol", "sharpe", "max_drawdown"],
        rows: rows || []
      }});
    }}

    function drawChart(el, chart) {{
      const series = (chart && chart.series || []).filter(s => s.points && s.points.length);
      if (!series.length) {{
        el.innerHTML = '<div class="muted">No return series found.</div>';
        return;
      }}
      const maxSeries = series.slice(0, 9);
      const all = maxSeries.flatMap(s => s.points.map(p => p.value));
      const minY = Math.min(...all);
      const maxY = Math.max(...all);
      const w = 1000, h = 330, pad = 34;
      const n = Math.max(...maxSeries.map(s => s.points.length));
      const y = v => h - pad - ((v - minY) / ((maxY - minY) || 1)) * (h - pad * 2);
      const x = i => pad + (i / Math.max(n - 1, 1)) * (w - pad * 2);
      const paths = maxSeries.map((s, idx) => {{
        const d = s.points.map((p, i) => `${{i === 0 ? "M" : "L"}}${{x(i).toFixed(1)}},${{y(p.value).toFixed(1)}}`).join(" ");
        return `<path d="${{d}}" fill="none" stroke="${{colors[idx % colors.length]}}" stroke-width="2"/>`;
      }}).join("");
      const grid = [0, .25, .5, .75, 1].map(t => {{
        const yy = pad + t * (h - pad * 2);
        const val = maxY - t * (maxY - minY);
        return `<line x1="${{pad}}" x2="${{w-pad}}" y1="${{yy}}" y2="${{yy}}" stroke="#e5e7eb"/><text x="4" y="${{yy+4}}" font-size="11" fill="#667085">${{val.toFixed(2)}}</text>`;
      }}).join("");
      el.innerHTML = `<svg viewBox="0 0 ${{w}} ${{h}}" preserveAspectRatio="none">${{grid}}${{paths}}</svg>` +
        `<div class="legend">${{maxSeries.map((s, i) => `<span style="--c:${{colors[i % colors.length]}}">${{s.name}}</span>`).join("")}}</div>`;
    }}

    function renderProfile(profile, idx) {{
      const returnBlocks = profile.return_summary.map((item, i) => `
        <div class="panel">
          <h3>Return Summary</h3>
          <div class="file-label">${{item.file}}</div>
          ${{summaryTable(item.summary)}}
        </div>
        <div class="panel">
          <h3>Equity Curve</h3>
          <div class="chart-wrap" id="chart-${{idx}}-${{i}}"></div>
        </div>
      `).join("");

      const tables = profile.tables.map(item => `
        <div class="panel">
          <h3>${{item.name}}</h3>
          <div class="file-label">${{item.path}}</div>
          ${{table(item.data)}}
        </div>
      `).join("");

      const texts = profile.texts.map(item => `
        <div class="panel">
          <h3>${{item.name}}</h3>
          <div class="file-label">${{item.path}}</div>
          <pre>${{item.text.replace(/[&<>]/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[ch]))}}</pre>
        </div>
      `).join("");

      return `
        <section class="section ${{idx === 0 ? "active" : ""}}" id="section-${{idx}}">
          <div class="grid">
            <div class="panel"><div class="card-title">Report Path</div><div class="card-value">${{profile.tables.length}}</div><div class="muted">structured CSV tables</div></div>
            <div class="panel"><div class="card-title">Strategy Path</div><div class="card-value">${{profile.return_summary.length}}</div><div class="muted">return files</div></div>
            <div class="panel"><div class="card-title">Text Reports</div><div class="card-value">${{profile.texts.length}}</div><div class="muted">research report files</div></div>
          </div>
          <h2>Strategy Performance</h2>
          <div class="two-col">${{returnBlocks || '<div class="panel muted">No strategy return files found.</div>'}}</div>
          <h2>Structured Reports</h2>
          <div class="two-col">${{tables || '<div class="panel muted">No report CSV files found.</div>'}}</div>
          <h2>Text Reports</h2>
          <div>${{texts || '<div class="panel muted">No report text found.</div>'}}</div>
        </section>
      `;
    }}

    function init() {{
      const tabs = document.getElementById("tabs");
      const content = document.getElementById("content");
      tabs.innerHTML = DATA.profiles.map((p, i) => `<button class="${{i === 0 ? "active" : ""}}" data-tab="${{i}}">${{p.name}}</button>`).join("");
      content.innerHTML = DATA.profiles.map(renderProfile).join("");
      DATA.profiles.forEach((profile, idx) => {{
        profile.return_charts.forEach((item, i) => drawChart(document.getElementById(`chart-${{idx}}-${{i}}`), item.chart));
      }});
      tabs.addEventListener("click", event => {{
        const btn = event.target.closest("button");
        if (!btn) return;
        document.querySelectorAll("button[data-tab]").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`section-${{btn.dataset.tab}}`).classList.add("active");
      }});
    }}
    init();
  </script>
</body>
</html>"""


def build_dashboard(output_path: Path = OUTPUT_PATH) -> Path:
    payload = {
        "profiles": [
            _collect_profile("Cross Sectional Momentum", CS_REPORT_PATH, CS_STRATEGY_PATH),
            _collect_profile("Time Series Momentum", TS_REPORT_PATH, TS_STRATEGY_PATH),
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_template(payload), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = build_dashboard()
    print(f"Dashboard saved to: {path}")
