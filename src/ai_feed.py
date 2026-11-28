"""Pure-Python static AI/LLM feed generator (llms.txt, llms-full.txt, ai/index.html, robots.txt, sitemap.xml).

Runs alongside scripts/update_live.py so web AI chat fetchers that do not execute client-side
JavaScript can read the complete live regime, Top-60 rankings, Top-30 orders, Paper Trading Bot
ledger, verified 16Y PIT / 2018-26 tables, and specification links via plain HTTP GET without
modifying the dashboard UI.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

SITE_BASE_URL = "https://harjaschadha.github.io/midcap-momentum-index"
RAW_BASE_URL = "https://raw.githubusercontent.com/HarjasChadha/midcap-momentum-index/main"


def _fmt_num(val: Any, dec: int = 2, suffix: str = "") -> str:
    if val is None or isinstance(val, bool):
        return "—"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "—"
    if num != num:
        return "—"
    return f"{num:,.{dec}f}{suffix}"


def _fmt_inr(val: Any, dec: int = 0) -> str:
    if val is None or isinstance(val, bool):
        return "—"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "—"
    if num != num:
        return "—"
    return f"₹{num:,.{dec}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("\n", " ").replace("|", "/") for c in r) + " |")
    return "\n".join(lines)


def _render_llms_txt(live: dict[str, Any], ledger: dict[str, Any] | None) -> str:
    as_of = live.get("as_of_date", "—")
    reg = live.get("regime", {}) or {}
    tg = reg.get("trend_gold_status", {}) or {}
    mf = reg.get("multifactor_blocks", {}) or {}
    acc_1l = ((ledger or {}).get("accounts") or {}).get("R3_1L_SlotCap", {}).get("summary", {})
    acc_10l = ((ledger or {}).get("accounts") or {}).get("R3_10L_Standard", {}).get("summary", {})

    lines = [
        "# DIY Nifty Midcap 150 Momentum Index (V3 / R3)",
        "",
        f"> Plain-text & static AI-readable mirror (as of {as_of}) of the live quantitative dashboard at {SITE_BASE_URL}/. "
        "Personal Quantitative Research & Paper-Trading Sandbox — Not Investment Advice / Not SEBI Registered.",
        "",
        "## Why Read Here",
        f"- The interactive dashboard at {SITE_BASE_URL}/ renders purely via client-side JavaScript (`assets/app.js` fetching `data/live_state.json`).",
        "- Web AI fetchers without JavaScript execution should read `llms-full.txt` or `/ai/` below, which contain 100% of the live numbers, orders, Top-60 rankings, Paper Bot ledgers, and verified backtest tables directly in plain text / semantic HTML.",
        "",
        "## Live Snapshot",
        f"- **As of Date:** {as_of} ({live.get('universe_count', 150)} NSE Nifty Midcap 150 equities + 9 ETFs/benchmarks)",
        f"- **Model Posture (T+1 Close):** `{reg.get('state', '—')}` -> {_fmt_num(reg.get('target_equity_pct'), 1, '%')} Equity / {_fmt_num(reg.get('target_goldbees_pct'), 1, '%')} GOLDBEES / {_fmt_num(reg.get('target_liquidcase_pct'), 1, '%')} LIQUIDCASE",
        f"- **MF4 Composite:** raw `{mf.get('composite_raw')}` / smoothed `{mf.get('composite_smooth')}` (Cautious `< -0.10`, Defensive `< -0.25`, Re-enter `> +0.18`)",
        f"- **TrendGold150 Gate:** active `{reg.get('active_defensive_etf', '—')}` (`GOLDBEES` ₹{_fmt_num(tg.get('gold_price'), 2)} vs EMA150 ₹{_fmt_num(tg.get('gold_ema150'), 2)}, {_fmt_num(tg.get('gold_vs_ema150_pct'), 2, '%')})",
        f"- **Market Internals:** India VIX `{reg.get('india_vix')}` · Breadth > 50-DMA `{_fmt_num(reg.get('breadth_50_sma10_pct'), 2, '%')}` · EW 1Y Drawdown `{_fmt_num(reg.get('ew_dd252_pct'), 2, '%')}`",
        f"- **Paper Bot NAV (`R3_1L_SlotCap`):** {_fmt_inr(acc_1l.get('nav'), 2)} ({_fmt_num(acc_1l.get('total_return_pct'), 2, '%')} | alpha `{_fmt_num(acc_1l.get('alpha_vs_ew_pct'), 2, '%')}` | {acc_1l.get('held_count', 0)} positions)",
        f"- **Paper Bot NAV (`R3_10L_Standard`):** {_fmt_inr(acc_10l.get('nav'), 2)} ({_fmt_num(acc_10l.get('total_return_pct'), 2, '%')} | {acc_10l.get('held_count', 0)} positions)",
        "",
        "## Direct Static Links (Zero JavaScript Required)",
        f"- [Complete Live & Backtest Data Dump (Markdown)]({SITE_BASE_URL}/llms-full.txt): Single-file complete state (`R3/S3/R2/R7/F2/F4` orders, Top-60 stocks, Paper Bot positions & ledger, 16Y PIT + 2018-26 verified tables)",
        f"- [Static No-JS HTML Snapshot]({SITE_BASE_URL}/ai/): Clean semantic HTML page with all tables pre-rendered on GitHub Pages",
        f"- [Complete Mathematical Specification (Markdown)]({RAW_BASE_URL}/MATHEMATICAL_SPECIFICATION.md): Every formula, corporate action rule, Sec 50AA tax model, and parameter provenance",
        f"- [Mathematical Specification PDF]({SITE_BASE_URL}/docs/Momentum_Strategy_Mathematical_Specification.pdf): 14-page typeset audit PDF",
        f"- [Final Strategy Audit Report]({RAW_BASE_URL}/reports/FINAL_STRATEGY_REPORT.md): 16-year PIT empirical verification & stress findings",
        f"- [16-Year Exact Verification JSON]({RAW_BASE_URL}/reports/src_exact_16yr_verification.json): Raw Python FIFO benchmark metrics (`2011-2026`)",
        f"- [Live State JSON]({SITE_BASE_URL}/data/live_state.json): Full live API payload",
        f"- [Paper Bot Ledger JSON]({SITE_BASE_URL}/data/paper_ledger.json): Autonomous FIFO paper-trading ledger",
        f"- [Compact Forward Tape JSON]({SITE_BASE_URL}/data/forward_tape.json): Daily forward replay tape",
        f"- [GitHub Source Repository](https://github.com/HarjasChadha/midcap-momentum-index): Complete Python engine, Numba research suite, and GitHub Actions CI",
        "",
    ]
    return "\n".join(lines)


def _render_llms_full_txt(
    live: dict[str, Any],
    ledger: dict[str, Any] | None,
    tape: dict[str, Any] | None,
) -> str:
    as_of = live.get("as_of_date", "—")
    gen_utc = live.get("generated_at_utc", "—")
    reg = live.get("regime", {}) or {}
    mf = reg.get("multifactor_blocks", {}) or {}
    tg = reg.get("trend_gold_status", {}) or {}
    etfs = reg.get("etfs", {}) or {}
    nse = live.get("nse_mcp_status", {}) or {}

    parts: list[str] = [
        f"# DIY Nifty Midcap 150 Momentum Index — Complete AI Data & Audit Snapshot ({as_of})",
        "",
        f"Generated: `{gen_utc}` | Dashboard: {SITE_BASE_URL}/ | Repo: https://github.com/HarjasChadha/midcap-momentum-index",
        "Disclosure: Personal Quantitative Research & Paper-Trading Sandbox — Not Investment Advice / Not SEBI Registered.",
        "",
        "---",
        "",
        "## 1. Live Market Regime & Multi-Factor Health (MF4)",
        _md_table(
            ["Metric", "Current Value", "Rule / Meaning"],
            [
                ["Data As Of", str(as_of), f"NSE Source: {nse.get('source', '—')} | Bulk quotes: {nse.get('bulk_quotes_fetched', '—')}"],
                ["Model Posture (T+1)", str(reg.get("state", "—")), f"Eq {_fmt_num(reg.get('target_equity_pct'), 1, '%')} / Gold {_fmt_num(reg.get('target_goldbees_pct'), 1, '%')} / Liquid {_fmt_num(reg.get('target_liquidcase_pct'), 1, '%')}"],
                ["MF4 Block 1 (Trend 44%)", str(mf.get("b1_trend", "—")), "EW vs EMA200, EMA126 & Rel Strength vs NIFTYBEES"],
                ["MF4 Block 2 (Vol/VIX 10%)", str(mf.get("b2_vol", "—")), f"21d downside semivol + India VIX ({reg.get('india_vix', '—')})"],
                ["MF4 Block 3 (Breadth 34%)", str(mf.get("b3_breadth", "—")), f"{_fmt_num(reg.get('breadth_50_sma10_pct'), 2, '%')} stocks > 50-DMA + Zweig 21d thrust"],
                ["MF4 Block 4 (DM/Gold 12%)", str(mf.get("b4_macro_gold", "—")), "Daniel-Moskowitz crash flag + Gold 126d risk trend"],
                ["MF4 Composite (Raw / Smooth)", f"{mf.get('composite_raw', '—')} / {mf.get('composite_smooth', '—')}", "8d down / 22d up EWMA; Cautious < -0.10, Defensive < -0.25, Re-enter > +0.18"],
                ["TrendGold150 Gate", str(reg.get("active_defensive_etf", "—")), f"GOLDBEES ₹{_fmt_num(tg.get('gold_price'), 2)} vs EMA150 ₹{_fmt_num(tg.get('gold_ema150'), 2)} ({_fmt_num(tg.get('gold_vs_ema150_pct'), 2, '%')}; +/-2% band)"],
                ["EW Index & 1Y Drawdown", f"{_fmt_num(reg.get('ew_index'), 2)} ({_fmt_num(reg.get('ew_dd252_pct'), 2, '%')} DD)", f"Midcap 150 Index: {reg.get('midcap150_index', '—')} ({_fmt_num(reg.get('midcap150_dd252_pct'), 2, '%')} DD)"],
                ["LIQUIDCASE Live Price", _fmt_inr((etfs.get("LIQUIDCASE") or {}).get("price"), 2), "Overnight liquid ETF sweep destination"],
            ],
        ),
        "",
    ]

    # Regime transitions history
    hist_trig = reg.get("historical_triggers") or []
    if hist_trig:
        parts.append("### Recent Historical MF4 Regime Shifts")
        trig_rows = [
            [
                str(x.get("date", "")),
                str(x.get("event", "")),
                _fmt_num(x.get("equity_pct"), 0, "%"),
                _fmt_num(x.get("ew_dd252_pct"), 2, "%"),
            ]
            for x in hist_trig[-12:]
        ]
        parts.append(_md_table(["Date", "Event", "Target Eq%", "EW 1Y DD%"], trig_rows))
        parts.append("")

    # Section 2: Current R3 1L Order Sheet + All Preset Summaries
    orders_1l = live.get("orders_1L") or {}
    parts.append("## 2. Current ₹1,00,000 Model Order Calculators (`T+1` Execution)")
    preset_summary_rows = []
    for k, box in orders_1l.items():
        if not isinstance(box, dict):
            continue
        preset_summary_rows.append([
            k,
            str(box.get("held_count", 0)),
            _fmt_inr(box.get("deployed_equity_inr"), 2),
            _fmt_num(box.get("deployed_equity_pct"), 1, "%"),
            str(box.get("liquidcase_sweep_units", 0)),
            _fmt_inr(box.get("liquidcase_sweep_inr"), 2),
            _fmt_inr(box.get("idle_cash_inr"), 2),
        ])
    if preset_summary_rows:
        parts.append(_md_table(["Preset Basket", "Stocks", "Equity Deployed", "Eq%", "LIQUIDCASE Units", "Liquid Sweep", "Uninvested Cash"], preset_summary_rows))
        parts.append("")

    r3_orders = (orders_1l.get("R3_Risk9M_SlotCap_LiqSweep") or {}).get("orders") or []
    if r3_orders:
        parts.append("### R3 Optimal (₹1,00,000 SlotCap + LIQUIDCASE Sweep) Target Orders")
        ord_rows = [
            [
                str(r.get("rank", "")),
                str(r.get("symbol", "")),
                str(r.get("tier", "")),
                _fmt_num(r.get("weight_pct"), 2, "%"),
                _fmt_inr(r.get("price"), 2),
                _fmt_inr(r.get("target_inr"), 2),
                str(r.get("shares", 0)),
                _fmt_inr(r.get("invested_inr"), 2),
                _fmt_num(r.get("raw_9m_pct"), 2, "%"),
            ]
            for r in r3_orders
        ]
        parts.append(_md_table(["Rank", "Symbol", "Bucket", "Target Wt", "LTP", "Target ₹", "Qty", "Order Value", "Raw 9M%"], ord_rows))
        parts.append("")

    # Section 3: Top-60 Momentum Rankings
    top60 = live.get("rankings_top60") or []
    if top60:
        parts.append(f"## 3. Top-60 Nifty Midcap 150 Momentum Rankings (As of {as_of})")
        t60_rows = [
            [
                str(r.get("rank_risk9m", "")),
                str(r.get("rank_sortino9m", "")),
                str(r.get("symbol", "")),
                _fmt_inr(r.get("price"), 2),
                f"{r.get('listing_days', '—')}d" + (" (NEW)" if r.get("new_listing") else ""),
                _fmt_num(r.get("risk_9m"), 2),
                _fmt_num(r.get("sortino_9m"), 2),
                _fmt_num(r.get("raw_9m_pct"), 2, "%"),
                _fmt_num(r.get("raw_12m_pct"), 2, "%"),
                "Hold (>50DMA)" if r.get("above_ma50") else "Trim (<50DMA)",
                _fmt_num(r.get("risk_ew_5050_wt_pct"), 2, "%"),
                "Yes" if r.get("risk_affordable_slotcap_1l") else "Skip (>Slot)",
            ]
            for r in top60
        ]
        parts.append(_md_table(
            ["R3 Rank", "S3 Rank", "Symbol", "LTP", "Listed", "Risk9M", "Sortino9M", "Raw 9M%", "Raw 12M%", "50-DMA Signal", "50/50 Wt%", "1L SlotCap"],
            t60_rows,
        ))
        parts.append("")

    # Section 4: Autonomous Paper Trading Bot Ledger
    if ledger and isinstance(ledger.get("accounts"), dict):
        parts.append(f"## 4. Autonomous Paper Trading Bot (`inception={ledger.get('inception_date')}`, `as_of={ledger.get('last_date')}`)")
        acc_rows = []
        for aid, acc in ledger["accounts"].items():
            s = acc.get("summary") or {}
            alloc = s.get("alloc_pct") or {}
            acc_rows.append([
                aid,
                str(acc.get("label", "")),
                _fmt_inr(acc.get("initial_capital"), 0),
                _fmt_inr(s.get("nav_full_tax", s.get("nav")), 2),
                _fmt_inr(s.get("nav_pre_def_tax"), 2),
                _fmt_num(s.get("total_return_pct"), 2, "%"),
                _fmt_num(s.get("alpha_vs_ew_pct"), 2, "%"),
                f"{_fmt_num(alloc.get('equity'), 1, '%')} Eq / {_fmt_num(alloc.get('gold'), 1, '%')} Gold / {_fmt_num(alloc.get('liquid'), 1, '%')} Liq",
                _fmt_inr(s.get("total_fees_cum"), 2),
                str(s.get("held_count", 0)),
                f"{s.get('next_rebalance_in_trading_days', 126)}d",
            ])
        parts.append(_md_table(
            ["Account", "Label", "Start Cap", "NAV (Net Tax)", "NAV (Pre Def-Tax)", "Return%", "Alpha vs EW", "Allocation", "Fees", "Held", "Next Rebal"],
            acc_rows,
        ))
        parts.append("")

        for aid in ("R3_1L_SlotCap", "R3_10L_Standard"):
            acc = (ledger["accounts"] or {}).get(aid)
            if not acc:
                continue
            holdings = acc.get("holdings") or []
            if holdings:
                parts.append(f"### Open FIFO Positions — `{aid}` ({len(holdings)})")
                h_rows = [
                    [
                        str(h.get("symbol", "")),
                        str(h.get("bucket", "")),
                        _fmt_num(h.get("shares"), 0),
                        _fmt_inr(h.get("avg_buy"), 2),
                        _fmt_inr(h.get("live_price"), 2),
                        _fmt_inr(h.get("value"), 2),
                        f"{_fmt_inr(h.get('unrealized_pnl'), 2)} ({_fmt_num(h.get('unrealized_pnl_pct'), 2, '%')})",
                        f"{h.get('days_held', 0)}d ({h.get('ltcg_days_remaining', 365)}d to LTCG)",
                    ]
                    for h in holdings
                ]
                parts.append(_md_table(["Symbol", "Bucket", "Qty", "Avg Buy", "Live", "Value", "Unrealized P&L", "LTCG Timer"], h_rows))
                parts.append("")

            decs = acc.get("decision_log") or []
            if decs:
                parts.append(f"### Recent Bot Decisions — `{aid}`")
                for d in decs[-10:]:
                    parts.append(f"- `{d.get('date')}` **{d.get('event_type')}**: {d.get('summary')}")
                parts.append("")

    # Section 5: Verified 16Y PIT + Live 2018-26 Backtests
    pit_rows = live.get("pit_validation_16yr") or []
    if pit_rows:
        parts.append("## 5. Verified 16-Year Zero-Survivorship PIT Verification (`2011-01-03` to `2026-11-26`, Net Sec 50AA + `15/35 bps`)")
        p_tbl = [
            [
                str(r.get("id", "")),
                str(r.get("strategy", "")),
                _fmt_num(r.get("p1_cagr"), 2, "%"),
                _fmt_num(r.get("p2_cagr"), 2, "%"),
                _fmt_num(r.get("p3_cagr"), 2, "%"),
                _fmt_num(r.get("full_cagr"), 2, "%"),
                _fmt_num(r.get("full_dd"), 2, "%"),
                _fmt_num(r.get("full_cal"), 2),
                _fmt_num(r.get("full_sh"), 2),
            ]
            for r in pit_rows
        ]
        parts.append(_md_table(["ID", "Strategy / Benchmark", "P1 11-16", "P2 16-21", "P3 21-26 (Post-Sel)", "16Y CAGR", "16Y MaxDD", "Calmar", "Sharpe"], p_tbl))
        parts.append("")

    clean_rows = live.get("clean_verification") or []
    if clean_rows:
        parts.append("## 6. Live Synchronized Current-Constituent Panel (`2018-01-02` to current, Survivorship-Biased Disclosure)")
        c_tbl = [
            [
                str(r.get("scenario", r.get("name", ""))),
                _fmt_inr(r.get("final_value"), 0),
                _fmt_num(r.get("cagr_pct"), 2, "%"),
                _fmt_num(r.get("max_dd_pct"), 2, "%"),
                _fmt_num(r.get("calmar"), 2),
                _fmt_num(r.get("sharpe"), 2),
                _fmt_num(r.get("avg_eq_pct"), 1, "%"),
                _fmt_num(r.get("ann_turnover_pct"), 1, "%"),
                _fmt_inr(r.get("total_tax_paid"), 0),
            ]
            for r in clean_rows
        ]
        parts.append(_md_table(["Scenario", "End Value (₹1L Start)", "CAGR%", "MaxDD%", "Calmar", "Sharpe", "Avg Eq%", "Turnover%", "Total Tax"], c_tbl))
        parts.append("")

    # Compact forward tape tail
    tape_days = (tape or {}).get("days") if isinstance(tape, dict) else None
    if isinstance(tape_days, list) and tape_days:
        tail = tape_days[-10:]
        parts.append(f"## 7. Forward Tape Trailing Daily Signals ({tail[0].get('date', '—')} to {tail[-1].get('date', '—')})")
        t_rows = []
        for day in tail:
            rg = day.get("regime") or {}
            t_rows.append([
                str(day.get("date", "—")),
                str(rg.get("posture", "—")),
                _fmt_num((rg.get("target_eq_mf4") or 0) * 100.0, 0, "%"),
                str(rg.get("active_def_etf", "—")),
                _fmt_num(rg.get("ew_index_val"), 2),
            ])
        parts.append(_md_table(["Date", "MF4 Posture", "Target Eq%", "Def ETF", "EW Index"], t_rows))
        parts.append("")

    parts.extend([
        "## 8. Canonical Documentation & Raw Endpoints",
        f"- Full Math Specification (Markdown): {RAW_BASE_URL}/MATHEMATICAL_SPECIFICATION.md",
        f"- Full Math Specification (14-page PDF): {SITE_BASE_URL}/docs/Momentum_Strategy_Mathematical_Specification.pdf",
        f"- Final Research Audit Report: {RAW_BASE_URL}/reports/FINAL_STRATEGY_REPORT.md",
        f"- 16Y Exact Verification JSON: {RAW_BASE_URL}/reports/src_exact_16yr_verification.json",
        f"- Live State JSON: {SITE_BASE_URL}/data/live_state.json",
        f"- Paper Bot Ledger JSON: {SITE_BASE_URL}/data/paper_ledger.json",
        f"- Forward Tape JSON: {SITE_BASE_URL}/data/forward_tape.json",
        "",
    ])
    return "\n".join(parts)


def _md_to_simple_html(md_text: str, title: str, as_of: str) -> str:
    """Convert the generated Markdown feed into a clean static HTML page with zero JS needed."""
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0

    def _inline(s: str) -> str:
        esc_s = html.escape(s)
        esc_s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', esc_s)
        esc_s = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc_s)
        esc_s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", esc_s)
        return esc_s

    while i < len(lines):
        s = lines[i].strip()
        if not s or s == "---":
            i += 1
            continue
        if s.startswith("# "):
            out.append(f"<h1>{_inline(s[2:])}</h1>")
            i += 1
        elif s.startswith("## "):
            out.append(f"<h2>{_inline(s[3:])}</h2>")
            i += 1
        elif s.startswith("### "):
            out.append(f"<h3>{_inline(s[4:])}</h3>")
            i += 1
        elif s.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|") and "-" in lines[i + 1]:
            hdrs = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            body_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                body_rows.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            th = "".join(f"<th>{_inline(h)}</th>" for h in hdrs)
            out.append(f'<div class="tbl-wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>')
        elif s.startswith("- "):
            lis = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                lis.append(f"<li>{_inline(lines[i].strip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(lis) + "</ul>")
        else:
            out.append(f"<p>{_inline(s)}</p>")
            i += 1

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="index, follow">
<title>{html.escape(title)}</title>
<meta name="description" content="Static plain-HTML snapshot as of {html.escape(as_of)} for Nifty Midcap 150 Momentum (R3), Paper Trading Bot ledgers, Top-60 rankings, and 16Y PIT benchmarks.">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #0f172a; max-width: 1120px; margin: 0 auto; padding: 24px 18px 48px; line-height: 1.5; background: #ffffff; }}
  h1 {{ font-size: 1.65rem; border-bottom: 2px solid #059669; padding-bottom: 8px; }}
  h2 {{ font-size: 1.25rem; margin-top: 28px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }}
  h3 {{ font-size: 1.05rem; margin-top: 20px; color: #1e293b; }}
  code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
  .tbl-wrap {{ overflow-x: auto; margin: 10px 0 18px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; white-space: nowrap; }}
  th {{ background: #f8fafc; font-weight: 600; }}
  a {{ color: #047857; }}
</style>
</head>
<body>
<article>
{"".join(out)}
</article>
</body>
</html>
"""


def build_ai_feeds(
    repo_root: Path,
    live_payload: dict[str, Any],
    ledger_payload: dict[str, Any] | None = None,
    tape_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write separate static AI-accessible endpoints without modifying the main website UI."""
    as_of = str(live_payload.get("as_of_date", "2026-11-27"))

    llms_txt = _render_llms_txt(live_payload, ledger_payload)
    llms_full_txt = _render_llms_full_txt(live_payload, ledger_payload, tape_payload)
    ai_html = _md_to_simple_html(
        llms_full_txt,
        f"DIY Nifty Midcap 150 Momentum (R3) — Static AI Snapshot ({as_of})",
        as_of,
    )

    robots_txt = "\n".join([
        "User-agent: *",
        "Allow: /",
        "",
        "User-agent: GPTBot",
        "Allow: /",
        "",
        "User-agent: OAI-SearchBot",
        "Allow: /",
        "",
        "User-agent: ChatGPT-User",
        "Allow: /",
        "",
        "User-agent: ClaudeBot",
        "Allow: /",
        "",
        "User-agent: Claude-Web",
        "Allow: /",
        "",
        "User-agent: PerplexityBot",
        "Allow: /",
        "",
        "User-agent: Google-Extended",
        "Allow: /",
        "",
        f"Sitemap: {SITE_BASE_URL}/sitemap.xml",
        "",
    ])

    urls = [
        f"{SITE_BASE_URL}/",
        f"{SITE_BASE_URL}/llms.txt",
        f"{SITE_BASE_URL}/llms-full.txt",
        f"{SITE_BASE_URL}/ai/",
        f"{SITE_BASE_URL}/docs/Momentum_Strategy_Mathematical_Specification.pdf",
        f"{SITE_BASE_URL}/data/live_state.json",
        f"{SITE_BASE_URL}/data/paper_ledger.json",
        f"{SITE_BASE_URL}/data/forward_tape.json",
    ]
    sitemap_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{html.escape(u)}</loc><lastmod>{html.escape(as_of)}</lastmod></url>\n" for u in urls)
        + "</urlset>\n"
    )

    llms_path = repo_root / "llms.txt"
    llms_full_path = repo_root / "llms-full.txt"
    ai_html_path = repo_root / "ai" / "index.html"
    robots_path = repo_root / "robots.txt"
    sitemap_path = repo_root / "sitemap.xml"

    llms_path.write_text(llms_txt, encoding="utf-8")
    llms_full_path.write_text(llms_full_txt, encoding="utf-8")
    ai_html_path.parent.mkdir(parents=True, exist_ok=True)
    ai_html_path.write_text(ai_html, encoding="utf-8")
    robots_path.write_text(robots_txt, encoding="utf-8")
    sitemap_path.write_text(sitemap_xml, encoding="utf-8")

    return {
        "llms": llms_path,
        "llms_full": llms_full_path,
        "ai_html": ai_html_path,
        "robots": robots_path,
        "sitemap": sitemap_path,
    }
