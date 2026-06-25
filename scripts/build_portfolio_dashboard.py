from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dashboard_data"
DOCS_DIR = ROOT / "docs"
OUT = DOCS_DIR / "index.html"


def records(name: str) -> list[dict]:
    return pd.read_csv(DATA_DIR / name).to_dict(orient="records")


def main() -> None:
    data = {
        "overview": records("overview_metrics.csv"),
        "target": records("target_distribution.csv"),
        "correlations": records("numeric_correlations.csv"),
        "segments": records("segment_default_rates.csv"),
        "bins": records("binned_numeric_default_rates.csv"),
        "models": records("model_cv_comparison.csv"),
        "final": records("final_model_metrics.csv"),
        "importance": records("feature_importance.csv"),
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Loan Default Risk Dashboard</title>
  <style>
    :root {{
      --navy: #15324d;
      --ink: #1f2937;
      --muted: #667085;
      --line: #d9e2ec;
      --soft: #f3f7fb;
      --blue: #4c78a8;
      --orange: #f28e2b;
      --green: #54a24b;
      --red: #b44747;
      --bg: #f7f9fc;
      --white: #fff;
      --shadow: 0 14px 35px rgba(21, 50, 77, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    header {{
      display: grid;
      grid-template-columns: 1.4fr .9fr;
      gap: 24px;
      align-items: end;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      color: var(--navy);
      font-size: clamp(30px, 5vw, 52px);
      letter-spacing: 0;
      line-height: 1.02;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 16px;
      line-height: 1.55;
      max-width: 760px;
    }}
    .badge {{
      justify-self: end;
      background: #eaf2fb;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px 16px;
      color: var(--navy);
      font-weight: 700;
      line-height: 1.35;
      max-width: 320px;
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .kpis {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--white);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
      min-width: 0;
    }}
    .kpi-value {{
      color: var(--navy);
      font-size: 30px;
      line-height: 1;
      font-weight: 800;
      margin-bottom: 8px;
    }}
    .kpi-label {{
      font-size: 13px;
      font-weight: 700;
      color: var(--ink);
    }}
    .kpi-section {{
      margin-top: 5px;
      font-size: 12px;
      color: var(--muted);
    }}
    .two-col {{
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    h2 {{
      margin: 0 0 4px;
      color: var(--navy);
      font-size: 20px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .section-note {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .chart {{
      display: grid;
      gap: 10px;
      min-height: 160px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 145px 1fr 74px;
      gap: 10px;
      align-items: center;
      font-size: 12px;
    }}
    .label {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--ink);
    }}
    .track {{
      height: 18px;
      background: #edf2f7;
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }}
    .bar {{
      height: 100%;
      border-radius: 999px;
      background: var(--blue);
      min-width: 2px;
    }}
    .bar.orange {{ background: var(--orange); }}
    .bar.green {{ background: var(--green); }}
    .bar.red {{ background: var(--red); }}
    .value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-top: 10px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      display: inline-block;
      border-radius: 50%;
      margin-right: 5px;
      background: var(--blue);
    }}
    .dot.orange {{ background: var(--orange); }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 14px;
    }}
    button {{
      border: 1px solid var(--line);
      background: var(--white);
      color: var(--navy);
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 700;
      cursor: pointer;
      font-size: 12px;
    }}
    button.active {{
      background: var(--navy);
      color: var(--white);
      border-color: var(--navy);
    }}
    .model-row {{
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .dual {{
      display: grid;
      gap: 5px;
    }}
    .mini-line {{
      display: grid;
      grid-template-columns: 72px 1fr 46px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }}
    .callout {{
      background: #eaf2fb;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      color: var(--navy);
      font-weight: 700;
      line-height: 1.45;
    }}
    .footer {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }}
    @media (max-width: 850px) {{
      header, .two-col, .kpis {{ grid-template-columns: 1fr; }}
      .badge {{ justify-self: stretch; }}
      .bar-row {{ grid-template-columns: 120px 1fr 64px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div>
        <h1>Loan Default Risk Dashboard</h1>
        <div class="subtitle">Portfolio dashboard built from aggregated loan-risk modeling outputs. It translates EDA, model selection, and final performance into business-facing risk signals.</div>
      </div>
      <div class="badge">Public-safe dashboard data only<br><span style="font-weight:500">No borrower-level records or raw dataset included.</span></div>
    </header>

    <section class="grid kpis" id="kpis"></section>

    <section class="grid two-col">
      <div class="card">
        <h2>Target Distribution</h2>
        <p class="section-note">Defaults are a minority class, so accuracy alone is not enough.</p>
        <div class="chart" id="targetChart"></div>
      </div>
      <div class="card">
        <h2>Numerical Risk Signals</h2>
        <p class="section-note">Affordability and pricing are the strongest numerical signals.</p>
        <div class="chart" id="corrChart"></div>
      </div>
      <div class="card wide">
        <h2>Segment-Level Risk</h2>
        <p class="section-note">Use the filter buttons to compare default rates across categorical borrower and loan segments.</p>
        <div class="tabs" id="segmentTabs"></div>
        <div class="chart" id="segmentChart"></div>
      </div>
      <div class="card">
        <h2>Model Comparison</h2>
        <p class="section-note">Gradient Boosting is selected by cross-validated PR-AUC.</p>
        <div id="modelChart"></div>
        <div class="legend"><span><i class="dot"></i>CV ROC-AUC</span><span><i class="dot orange"></i>CV PR-AUC</span></div>
      </div>
      <div class="card">
        <h2>Top Predictive Drivers</h2>
        <p class="section-note">Permutation importance aligns with EDA: repayment burden, pricing, housing stability, and income.</p>
        <div class="chart" id="importanceChart"></div>
      </div>
      <div class="card wide">
        <h2>Final Test Readout</h2>
        <p class="section-note">The selected model is evaluated once on the untouched holdout test set after threshold tuning.</p>
        <div class="grid kpis" id="finalKpis"></div>
        <div class="callout" id="finalCallout"></div>
      </div>
    </section>

    <div class="footer">Prepared by Juana Zhang. Dashboard uses aggregated CSV tables from the portfolio repository's <code>dashboard_data</code> folder.</div>
  </main>

  <script>
    const DATA = {json.dumps(data)};
    const fmtPct = v => `${{(v * 100).toFixed(1)}}%`;
    const fmt3 = v => Number(v).toFixed(3);
    const nice = s => String(s).replaceAll("_", " ");

    function barRows(el, rows, valueKey, labelKey, opts = {{}}) {{
      const max = opts.max ?? Math.max(...rows.map(d => Math.abs(Number(d[valueKey]))));
      el.innerHTML = rows.map(d => {{
        const raw = Number(d[valueKey]);
        const width = Math.max(1, Math.abs(raw) / max * 100);
        const color = opts.color || (raw < 0 ? "green" : "blue");
        const val = opts.percent ? fmtPct(raw) : fmt3(raw);
        return `<div class="bar-row">
          <div class="label" title="${{d[labelKey]}}">${{nice(d[labelKey])}}</div>
          <div class="track"><div class="bar ${{color}}" style="width:${{width}}%"></div></div>
          <div class="value">${{val}}</div>
        </div>`;
      }}).join("");
    }}

    function renderKpis() {{
      const wanted = ["Records analyzed", "Observed default rate", "Selected model", "Test ROC-AUC", "Test PR-AUC", "Tuned test F1"];
      const rows = DATA.overview.filter(d => wanted.includes(d.metric));
      document.getElementById("kpis").innerHTML = rows.map(d => `<div class="card">
        <div class="kpi-value">${{d.display_value}}</div>
        <div class="kpi-label">${{d.metric}}</div>
        <div class="kpi-section">${{d.section}}</div>
      </div>`).join("");
    }}

    function renderTarget() {{
      barRows(document.getElementById("targetChart"), DATA.target, "share", "loan_status_label", {{percent:true, color:"blue", max:1}});
    }}

    function renderCorrelations() {{
      const rows = DATA.correlations.slice(0, 6);
      barRows(document.getElementById("corrChart"), rows, "correlation_to_default", "feature");
    }}

    function renderSegments(active = "loan_grade") {{
      const dims = [...new Set(DATA.segments.map(d => d.dimension))];
      const tabs = document.getElementById("segmentTabs");
      tabs.innerHTML = dims.map(d => `<button class="${{d === active ? "active" : ""}}" onclick="renderSegments('${{d}}')">${{nice(d)}}</button>`).join("");
      const rows = DATA.segments.filter(d => d.dimension === active).sort((a, b) => b.default_rate - a.default_rate);
      barRows(document.getElementById("segmentChart"), rows, "default_rate", "segment", {{percent:true, color:"blue", max: Math.max(...rows.map(d => Number(d.default_rate))) }});
    }}

    function renderModels() {{
      const html = DATA.models.map(d => `<div class="model-row">
        <strong>${{d.model}}</strong>
        <div class="dual">
          <div class="mini-line"><span>ROC-AUC</span><div class="track"><div class="bar" style="width:${{Number(d.cv_roc_auc_mean) * 100}}%"></div></div><span class="value">${{fmt3(d.cv_roc_auc_mean)}}</span></div>
          <div class="mini-line"><span>PR-AUC</span><div class="track"><div class="bar orange" style="width:${{Number(d.cv_pr_auc_mean) * 100}}%"></div></div><span class="value">${{fmt3(d.cv_pr_auc_mean)}}</span></div>
        </div>
      </div>`).join("");
      document.getElementById("modelChart").innerHTML = html;
    }}

    function renderImportance() {{
      const rows = DATA.importance.slice(0, 8);
      barRows(document.getElementById("importanceChart"), rows, "importance", "feature", {{color:"blue"}});
    }}

    function renderFinal() {{
      const final = DATA.final.find(d => d.stage === "Final holdout test");
      const cards = [
        ["Precision", fmtPct(final.precision)],
        ["Recall", fmtPct(final.recall)],
        ["F1", fmt3(final.f1)],
        ["PR-AUC", fmt3(final.pr_auc)]
      ];
      document.getElementById("finalKpis").innerHTML = cards.map(([label, value]) => `<div class="card">
        <div class="kpi-value">${{value}}</div><div class="kpi-label">${{label}}</div><div class="kpi-section">Final holdout test</div>
      </div>`).join("");
      document.getElementById("finalCallout").textContent = `At the tuned threshold, the model identifies ${{Number(final.tp).toLocaleString()}} defaults and misses ${{Number(final.fn).toLocaleString()}}, with ${{Number(final.fp).toLocaleString()}} false positives.`;
    }}

    renderKpis();
    renderTarget();
    renderCorrelations();
    renderSegments();
    renderModels();
    renderImportance();
    renderFinal();
  </script>
</body>
</html>
"""

    DOCS_DIR.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
