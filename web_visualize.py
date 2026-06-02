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


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = [column if column else "index" for column in reader.fieldnames or []]
            rows = []
            for row in reader:
                clean = {}
                for key, value in row.items():
                    key = key if key not in {None, ""} else "index"
                    parsed = _to_float(value)
                    clean[key] = parsed if parsed is not None else value
                rows.append(clean)
            return columns, rows
    except Exception:
        return [], []


def _as_records(path: Path) -> dict:
    columns, rows = _read_csv_rows(path)
    return {
        "columns": columns,
        "rows": rows,
        "file": str(path),
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _summarize_returns(path: Path) -> tuple[list[dict], list[dict]]:
    columns, rows = _read_csv_rows(path)
    if not columns or not rows:
        return [], []

    summary = []
    chart_series = []
    date_col = columns[0]

    for column in columns[1:]:
        dated_returns = [
            (str(row.get(date_col, "")), float(row[column]))
            for row in rows
            if isinstance(row.get(column), (int, float))
        ]
        if not dated_returns:
            continue

        returns = [value for _, value in dated_returns]
        growth = 1.0
        peak = 1.0
        max_drawdown = 0.0
        points = []

        for date, value in dated_returns:
            growth *= 1.0 + value
            peak = max(peak, growth)
            max_drawdown = min(max_drawdown, growth / peak - 1.0)
            points.append({"date": date, "value": growth})

        annual_return = growth ** (252 / len(returns)) - 1.0 if growth > 0 else None
        std = _sample_std(returns)
        annual_vol = std * math.sqrt(252) if std is not None else None
        sharpe = annual_return / annual_vol if annual_return is not None and annual_vol else None

        summary.append({
            "strategy": column,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        })
        chart_series.append({"name": column, "points": points})

    summary.sort(key=lambda row: row["sharpe"] if row["sharpe"] is not None else -999, reverse=True)
    return summary, chart_series


def _find_report(report_path: Path, suffix: str) -> Path | None:
    matches = sorted(report_path.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _collect_profile(name: str, report_path: Path, strategy_path: Path) -> dict:
    strategy_files = sorted(strategy_path.glob("*/*.csv"))
    return_files = [
        path for path in strategy_files
        if path.name in {"strategy_returns.csv", "ts_strategy_returns.csv"}
    ]
    return_path = return_files[0] if return_files else None
    return_summary, chart_series = _summarize_returns(return_path) if return_path else ([], [])

    regime_path = _find_report(report_path, "metrics_by_regime.csv")
    cost_path = _find_report(report_path, "risk_cost_summary.csv")
    oos_path = _find_report(report_path, "Rolling_OOS_summary.csv")
    trade_path = _find_report(report_path, "trade_stats_by_regime.csv")

    return {
        "name": name,
        "return_file": str(return_path) if return_path else "",
        "return_summary": return_summary,
        "chart_series": chart_series,
        "regime_metrics": _as_records(regime_path) if regime_path else {"columns": [], "rows": [], "file": ""},
        "cost_summary": _as_records(cost_path) if cost_path else {"columns": [], "rows": [], "file": ""},
        "oos_summary": _as_records(oos_path) if oos_path else {"columns": [], "rows": [], "file": ""},
        "trade_stats": _as_records(trade_path) if trade_path else {"columns": [], "rows": [], "file": ""},
        "conclusions": {
            section: {
                "file": str(path) if path else "",
                "text": _read_text(path) if path else "",
            }
            for section, path in {
                "overview": _find_report(report_path, "Overview_Conclusion.txt"),
                "regime": _find_report(report_path, "Regime_Conclusion.txt"),
                "cost": _find_report(report_path, "Cost_Conclusion.txt"),
            }.items()
        },
    }


def _html_template(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ETF Momentum Dashboard</title>
  <style>
    :root {
      --bg: #f5f7f9;
      --panel: #ffffff;
      --ink: #17212b;
      --muted: #667085;
      --line: #d8dee6;
      --soft: #edf2f6;
      --accent: #1769aa;
      --accent-soft: #e7f1fa;
      --good: #067647;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 20px 28px;
    }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    h2 { margin: 0 0 4px; font-size: 19px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 14px; letter-spacing: 0; }
    p { margin: 0; }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 7px;
      padding: 8px 11px;
      cursor: pointer;
      font-weight: 650;
    }
    button.active { border-color: var(--accent); background: var(--accent); color: white; }
    main { max-width: 1480px; margin: 0 auto; padding: 18px 22px 46px; }
    .muted, .file-label { color: var(--muted); }
    .file-label { margin: -4px 0 10px; font-size: 11px; overflow-wrap: anywhere; }
    .profile-tabs, .view-tabs, .selector-list {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
    }
    .profile-tabs { margin-bottom: 14px; }
    .profile, .view { display: none; }
    .profile.active, .view.active { display: block; }
    .view-tabs {
      position: sticky;
      top: 0;
      z-index: 4;
      margin-bottom: 16px;
      padding: 9px 0;
      background: rgba(245, 247, 249, 0.95);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(6px);
    }
    .intro { margin: 8px 0 14px; }
    .intro p { max-width: 760px; color: var(--muted); font-size: 13px; }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 11px;
      margin-bottom: 14px;
    }
    .kpi-section .kpi-grid { margin-bottom: 14px; }
    .section-subtitle { margin: 0 0 8px; font-size: 14px; }
    .kpi, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .kpi { padding: 13px; min-height: 105px; }
    .kpi-label { color: var(--muted); font-size: 11px; margin-bottom: 7px; }
    .kpi-value { font-size: 21px; font-weight: 760; overflow-wrap: anywhere; }
    .kpi-note { margin-top: 5px; color: var(--muted); font-size: 11px; }
    .panel { padding: 14px; overflow: hidden; }
    .stack { display: grid; gap: 13px; }
    .two-col { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(320px, .75fr); gap: 13px; }
    .panel-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
    .chart-wrap { min-height: 365px; margin-top: 8px; }
    svg { width: 100%; height: 330px; display: block; }
    .legend { display: flex; flex-wrap: wrap; gap: 7px 13px; margin-top: 7px; color: var(--muted); font-size: 11px; }
    .legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 5px; border-radius: 50%; background: var(--c); }
    details { margin-top: 10px; }
    summary { width: fit-content; color: var(--accent); cursor: pointer; font-size: 12px; font-weight: 700; }
    .selector-list { margin-top: 10px; }
    .selector {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 5px 7px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      font-size: 11px;
    }
    .selector input { accent-color: var(--accent); }
    .selector-message { min-height: 17px; margin-top: 6px; color: var(--warn); font-size: 11px; }
    .conclusion-text { max-width: 800px; margin: 8px 0 0; white-space: pre-wrap; color: #000; font: inherit; font-size: 15px; line-height: 1.6; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { position: sticky; top: 0; background: #f1f4f7; color: #344054; }
    .table-scroll { max-height: 515px; overflow: auto; border: 1px solid var(--line); border-radius: 7px; }
    .empty { padding: 16px; border: 1px dashed var(--line); border-radius: 7px; color: var(--muted); background: var(--panel); }
    @media (max-width: 980px) {
      header { display: block; padding: 18px; }
      main { padding: 14px; }
      .kpi-grid, .two-col { grid-template-columns: 1fr; }
      .chart-wrap { min-height: 295px; }
      svg { height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>ETF Momentum Dashboard</h1>
      <p class="muted">Strategy comparison, regime behavior, and cost-aware performance.</p>
    </div>
  </header>
  <main>
    <div class="profile-tabs" id="profile-tabs"></div>
    <div id="content"></div>
  </main>
  <script>
    const DATA = __DATA__;
    const COLORS = ["#1769aa", "#067647", "#dc6803", "#7c3aed", "#be123c", "#0891b2", "#4d7c0f", "#9333ea"];
    const DEFAULT_SELECTED = 5;
    const MAX_SELECTED = 10;

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));
    }

    function fmt(value, kind = "") {
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      if (typeof value !== "number") return escapeHtml(value);
      if (kind === "percent") return `${(value * 100).toFixed(2)}%`;
      if (kind === "ratio") return value.toFixed(2);
      if(Math.abs(value) == 0) return value;
      if (Math.abs(value) < .01) return value.toFixed(4);
      if (Math.abs(value) < 1 || !Number.isInteger(value)) return value.toFixed(3);
      return value;
    }

    function isBenchmark(row) {
      return String(row.strategy || "").startsWith("Benchmark(");
    }

    function metricKind(column) {
      if (column.includes("return") || column.includes("vol") || column.includes("mdd") || column.includes("drawdown") || column.includes("cost") || column.includes("drag") || column.includes("turnover") || column.includes("VaR")) return "percent";
      if (column.includes("sharpe") || column.includes("ratio") || column.includes("factor")) return "ratio";
      return "";
    }

    function table(data, columns) {
      const rows = data && data.rows || [];
      const chosen = (columns || []).filter(column => (data.columns || []).includes(column));
      if (!chosen.length || !rows.length) return '<div class="empty">No data available.</div>';
      const head = chosen.map(column => `<th>${escapeHtml(column)}</th>`).join("");
      const body = rows.map(row => `
        <tr>
          ${chosen.map(column => `<td>${fmt(row[column], metricKind(column))}</td>`).join("")}
        </tr>
      `).join("");
      return `<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function returnTable(rows) {
      return table(
        {columns: ["strategy", "annual_return", "annual_vol", "sharpe", "max_drawdown"], rows},
        ["strategy", "annual_return", "annual_vol", "sharpe", "max_drawdown"]
      );
    }

    function strategies(profile) {
      return (profile.return_summary || []).filter(row => !isBenchmark(row));
    }

    function bestStrategy(profile) {
      return strategies(profile)[0] || {};
    }

    function costRows(profile) {
      return profile.cost_summary && profile.cost_summary.rows || [];
    }

    function costFor(profile, strategyName) {
      return costRows(profile).find(row => row.strategy === strategyName) || {};
    }

    function maxCostDrag(profile) {
      return [...costRows(profile)].sort((a, b) => (b.return_drag ?? -999) - (a.return_drag ?? -999))[0] || {};
    }

    function regimeDistribution(profile) {
      const rows = profile.regime_metrics && profile.regime_metrics.rows || [];
      const firstStrategy = rows[0] && rows[0].strategy;
      const counts = new Map(
        rows
          .filter(row => row.strategy === firstStrategy)
          .map(row => [row.regime, row.count || 0])
      );
      const regimes = ["Bull Low Vol", "Bull High Vol", "Bear Low Vol", "Bear High Vol"];
      const total = regimes.reduce((sum, regime) => sum + (counts.get(regime) || 0), 0);
      return regimes.map(regime => ({
        regime,
        count: counts.get(regime) || 0,
        share: total ? (counts.get(regime) || 0) / total : 0,
      }));
    }

    function defaultSeries(profile) {
      const benchmarks = (profile.return_summary || []).filter(isBenchmark).map(row => row.strategy);
      const leaders = strategies(profile).slice(0, DEFAULT_SELECTED).map(row => row.strategy);
      return [...benchmarks, ...leaders].slice(0, DEFAULT_SELECTED);
    }

    function drawChart(el, profile, selectedNames) {
      const wanted = new Set(selectedNames);
      const series = (profile.chart_series || []).filter(item => wanted.has(item.name) && item.points.length);
      if (!series.length) {
        el.innerHTML = '<div class="empty">No return series available.</div>';
        return;
      }

      const values = series.flatMap(item => item.points.map(point => point.value));
      const minY = Math.min(...values);
      const maxY = Math.max(...values);
      const width = 1000, height = 330, pad = 36, bottomPad = 52;
      const count = Math.max(...series.map(item => item.points.length));
      const x = index => pad + (index / Math.max(count - 1, 1)) * (width - pad * 2);
      const y = value => height - bottomPad - ((value - minY) / ((maxY - minY) || 1)) * (height - pad - bottomPad);
      const grid = [0, .25, .5, .75, 1].map(step => {
        const yy = pad + step * (height - pad - bottomPad);
        const label = maxY - step * (maxY - minY);
        return `<line x1="${pad}" x2="${width-pad}" y1="${yy}" y2="${yy}" stroke="#e5e7eb"/><text x="3" y="${yy+4}" font-size="11" fill="#667085">${label.toFixed(2)}</text>`;
      }).join("");
      const referencePoints = series[0].points;
      const tickSteps = [0, .25, .5, .75, 1];
      const xAxis = tickSteps.map(step => {
        const index = Math.round(step * Math.max(referencePoints.length - 1, 0));
        const point = referencePoints[index];
        if (!point) return "";
        const xx = pad + step * (width - pad * 2);
        return `<line x1="${xx}" x2="${xx}" y1="${height-bottomPad}" y2="${height-bottomPad+5}" stroke="#98a2b3"/><text x="${xx}" y="${height-bottomPad+20}" text-anchor="middle" font-size="11" fill="#667085">${escapeHtml(point.date)}</text>`;
      }).join("");
      const paths = series.map((item, index) => {
        const path = item.points.map((point, pointIndex) => `${pointIndex ? "L" : "M"}${x(pointIndex).toFixed(1)},${y(point.value).toFixed(1)}`).join(" ");
        return `<path d="${path}" fill="none" stroke="${COLORS[index % COLORS.length]}" stroke-width="2"/>`;
      }).join("");
      const legend = series.map((item, index) => `<span style="--c:${COLORS[index % COLORS.length]}">${escapeHtml(item.name)}</span>`).join("");
      el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${grid}${xAxis}${paths}</svg><div class="legend">${legend}</div>`;
    }

    function renderSelectors(profile, profileIndex) {
      const defaults = new Set(defaultSeries(profile));
      return (profile.chart_series || []).map(item => `
        <label class="selector">
          <input type="checkbox" data-series="${escapeHtml(item.name)}" data-profile="${profileIndex}" ${defaults.has(item.name) ? "checked" : ""}>
          <span>${escapeHtml(item.name)}</span>
        </label>
      `).join("");
    }

    function renderConclusion(profile, section) {
      const conclusion = profile.conclusions && profile.conclusions[section] || {};
      if (!conclusion.text) return "";
      return `
        <pre class="conclusion-text">${escapeHtml(conclusion.text)}</pre>
      `;
    }

    function renderOverview(profile, profileIndex) {
      const best = bestStrategy(profile);
      const bestCost = costFor(profile, best.strategy);
      const costly = maxCostDrag(profile);
      return `
        <div class="view active" data-view="overview">
          <div class="intro">
            <h2>Overview</h2>
            ${renderConclusion(profile, "overview")}
          </div>
          <div class="kpi-section">
            <h3 class="section-subtitle">Net performance highlights and representative strategy comparisons.</h3>
            <div class="kpi-grid">
              <div class="kpi"><div class="kpi-label">Best strategy by Sharpe</div><div class="kpi-value">${escapeHtml(best.strategy || "N/A")}</div><div class="kpi-note">Sharpe ${fmt(best.sharpe, "ratio")}</div></div>
              <div class="kpi"><div class="kpi-label">Annual return</div><div class="kpi-value">${fmt(best.annual_return, "percent")}</div><div class="kpi-note">${escapeHtml(best.strategy || "")}</div></div>
              <div class="kpi"><div class="kpi-label">Maximum drawdown</div><div class="kpi-value">${fmt(best.max_drawdown, "percent")}</div><div class="kpi-note">${escapeHtml(best.strategy || "")}</div></div>
              <div class="kpi"><div class="kpi-label">Return drag</div><div class="kpi-value">${fmt(bestCost.return_drag, "percent")}</div><div class="kpi-note">${escapeHtml(best.strategy || "")}</div></div>
            </div>
          </div>
          <div class="two-col">
            <div class="panel">
              <div class="panel-head"><div><h3>Equity Curve Comparison</h3><div class="file-label">${escapeHtml(profile.return_file)}</div></div></div>
              <div class="chart-wrap" id="chart-${profileIndex}"></div>
              <details>
                <summary>Compare series</summary>
                <div class="selector-list">${renderSelectors(profile, profileIndex)}</div>
                <div class="selector-message" id="selector-message-${profileIndex}"></div>
              </details>
            </div>
            <div class="panel">
              <h3>Strategy Ranking</h3>
              ${returnTable(profile.return_summary || [])}
            </div>
          </div>
          <div class="panel" style="margin-top:13px">
            <h3>Rolling OOS Summary</h3>
            <div class="file-label">${escapeHtml(profile.oos_summary.file)}</div>
            ${table(profile.oos_summary, ["index", "train_start", "train_end", "test_start", "test_end", "train_mean_ic", "train_ic_ir", "train_pos_ic_ratio", "test_annu_return", "test_annu_vol", "test_sharpe", "benchmark_test_sharpe"])}
          </div>
        </div>
      `;
    }

    function renderRegime(profile) {
      const distribution = regimeDistribution(profile);
      return `
        <div class="view" data-view="regime">
          <div class="intro">
            <h2>Regime</h2>
            ${renderConclusion(profile, "regime")}
          </div>
          <div class="kpi-section">
            <h3 class="section-subtitle">Risk-adjusted performance across bull, bear, high-volatility, and low-volatility conditions.</h3>
            <div class="kpi-grid">
              ${distribution.map(item => `
                <div class="kpi">
                  <div class="kpi-label">${escapeHtml(item.regime)}</div>
                  <div class="kpi-value">${fmt(item.share, "percent")}</div>
                  <div class="kpi-note">${fmt(item.count, "count")} trading days</div>
                </div>
              `).join("")}
            </div>
          </div>
          <div class="panel">
            <h3>Metrics by Regime</h3>
            <div class="file-label">${escapeHtml(profile.regime_metrics.file)}</div>
            ${table(profile.regime_metrics, ["strategy", "regime", "count", "annual_return", "annual_vol", "sharpe", "max_drawdown", "VaR(95%)"])}
          </div>
          <div class="panel" style="margin-top:13px">
            <h3>Trade Stats by Entry Regime</h3>
            <div class="file-label">${escapeHtml(profile.trade_stats.file)}</div>
            ${table(profile.trade_stats, ["strategy", "regime", "count", "win_rate", "avg_win", "avg_loss", "payoff_ratio", "profit_factor"])}
          </div>
        </div>
      `;
    }

    function renderCost(profile) {
      const costly = maxCostDrag(profile);
      return `
        <div class="view" data-view="cost">
          <div class="intro">
            <h2>Cost</h2>
            ${renderConclusion(profile, "cost")}
          </div>
          <div class="kpi-section">
            <h3 class="section-subtitle">Gross-to-net performance drag from turnover and execution assumptions.</h3>
            <div class="kpi-grid">
              <div class="kpi"><div class="kpi-label">Largest return drag</div><div class="kpi-value">${fmt(costly.return_drag, "percent")}</div><div class="kpi-note">${escapeHtml(costly.strategy || "N/A")}</div></div>
              <div class="kpi"><div class="kpi-label">Sharpe drag</div><div class="kpi-value">${fmt(costly.sharpe_drag, "ratio")}</div><div class="kpi-note">${escapeHtml(costly.strategy || "")}</div></div>
              <div class="kpi"><div class="kpi-label">Total transaction cost</div><div class="kpi-value">${fmt(costly.total_transaction_cost, "percent")}</div><div class="kpi-note">${escapeHtml(costly.strategy || "")}</div></div>
              <div class="kpi"><div class="kpi-label">Annualized turnover</div><div class="kpi-value">${fmt(costly.annualized_turnover, "percent")}</div><div class="kpi-note">${escapeHtml(costly.strategy || "")}</div></div>
            </div>
          </div>
          <div class="panel">
            <h3>Risk and Cost Summary</h3>
            <div class="file-label">${escapeHtml(profile.cost_summary.file)}</div>
            ${table(profile.cost_summary, ["strategy", "gross_annual_return", "net_annual_return", "return_drag", "total_transaction_cost", "annualized_turnover"])}
          </div>
        </div>
      `;
    }

    function renderProfile(profile, index) {
      return `
        <section class="profile ${index === 0 ? "active" : ""}" id="profile-${index}">
          <div class="view-tabs">
            <button class="active" data-view-tab="overview">Overview</button>
            <button data-view-tab="regime">Regime</button>
            <button data-view-tab="cost">Cost</button>
          </div>
          ${renderOverview(profile, index)}
          ${renderRegime(profile)}
          ${renderCost(profile)}
        </section>
      `;
    }

    function selectedSeries(profileIndex) {
      return [...document.querySelectorAll(`input[data-profile="${profileIndex}"]:checked`)].map(input => input.dataset.series);
    }

    function refreshChart(profileIndex) {
      drawChart(document.getElementById(`chart-${profileIndex}`), DATA.profiles[profileIndex], selectedSeries(profileIndex));
    }

    function init() {
      const profileTabs = document.getElementById("profile-tabs");
      const content = document.getElementById("content");
      profileTabs.innerHTML = DATA.profiles.map((profile, index) => `<button class="${index === 0 ? "active" : ""}" data-profile-tab="${index}">${escapeHtml(profile.name)}</button>`).join("");
      content.innerHTML = DATA.profiles.map(renderProfile).join("");
      DATA.profiles.forEach((_, index) => refreshChart(index));

      profileTabs.addEventListener("click", event => {
        const button = event.target.closest("button[data-profile-tab]");
        if (!button) return;
        document.querySelectorAll("button[data-profile-tab]").forEach(item => item.classList.toggle("active", item === button));
        document.querySelectorAll(".profile").forEach(profile => profile.classList.remove("active"));
        document.getElementById(`profile-${button.dataset.profileTab}`).classList.add("active");
      });

      content.addEventListener("click", event => {
        const button = event.target.closest("button[data-view-tab]");
        if (!button) return;
        const profile = button.closest(".profile");
        profile.querySelectorAll("button[data-view-tab]").forEach(item => item.classList.toggle("active", item === button));
        profile.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.dataset.view === button.dataset.viewTab));
      });

      content.addEventListener("change", event => {
        const input = event.target.closest("input[data-series]");
        if (!input) return;
        const profileIndex = input.dataset.profile;
        const selected = selectedSeries(profileIndex);
        const message = document.getElementById(`selector-message-${profileIndex}`);
        if (selected.length > MAX_SELECTED) {
          input.checked = false;
          message.textContent = `Select up to ${MAX_SELECTED} series.`;
          return;
        }
        message.textContent = "";
        refreshChart(profileIndex);
      });
    }

    init();
  </script>
</body>
</html>"""
    return template.replace("__DATA__", data)


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
