"""Daily production updater: merges NSE MCP official Bhavcopy/Mood/Corporate-Actions + cached history to build data/live_state.json."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtester import BacktestConfig, compute_metrics, run_backtest  # noqa: E402
from src.data_loader import BENCHMARK_TICKERS, fetch_and_cache_all, load_universe  # noqa: E402
from src.momentum import combine_sleeves, compute_score_matrix, select_affordable_blend, select_sleeve_weights  # noqa: E402
from src.nse_mcp import NseMcpClient  # noqa: E402
from src.paper_bot import (  # noqa: E402
    build_forward_tape_entry,
    update_forward_tape,
    update_paper_ledger,
)
from src.regime import (  # noqa: E402
    build_defensive_series,
    build_ew_index,
    build_regime_features,
    compute_equity_allocation_series,
    compute_multifactor_state,
    get_trend_gold_status,
)

TRACKED_ETFS = ["GOLDBEES", "SILVERBEES", "EBBETF0431", "LTGILTBEES", "LIQUIDCASE", "LIQUIDBEES", "NIFTYBEES", "MON100", "MIDCAPIETF"]


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and replace NaN/Inf with None so json.dumps(..., allow_nan=False) never fails."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return round(v, 4) if math.isfinite(v) else None
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [sanitize_for_json(x) for x in obj.tolist()]
    return str(obj)


def _sync_live_nse_quotes(adj_df: pd.DataFrame, raw_df: pd.DataFrame, vol_df: pd.DataFrame, bench_df: pd.DataFrame, symbols: list[str]) -> dict[str, Any]:
    """Pull live EOD quotes & market mood from NSE MCP; append new trading day or repair trailing nulls cleanly."""
    client = NseMcpClient(timeout=15.0)
    telemetry: dict[str, Any] = {"source": "nse_mcp", "mood_ok": False, "bulk_quotes": 0, "corp_actions_checked": [], "appended_date": None}

    try:
        mood = client.get_market_mood("today")
        if isinstance(mood, dict) and mood.get("date"):
            telemetry["mood"] = mood
            telemetry["mood_ok"] = True
    except Exception as e:
        telemetry["mood_error"] = str(e)

    try:
        idx_hist = client.get_index_history("Nifty Midcap 150", months=3, end_date="today")
        if isinstance(idx_hist, dict):
            telemetry["index_history"] = idx_hist
    except Exception as e:
        telemetry["index_error"] = str(e)

    all_syms = symbols + [s for s in TRACKED_ETFS if s not in symbols]
    quotes: dict[str, Any] = {}
    try:
        quotes = client.get_bulk_quote(all_syms)
        telemetry["bulk_quotes"] = len(quotes)
    except Exception as e:
        telemetry["bulk_error"] = str(e)

    if not quotes:
        return telemetry

    # Determine official NSE date present in quotes
    quote_dates = [str(q.get("date", ""))[:10] for q in quotes.values() if isinstance(q, dict) and q.get("date")]
    if not quote_dates:
        return telemetry
    nse_date = max(set(quote_dates), key=quote_dates.count)
    last_cached = str(adj_df.index[-1])

    # Auto-adjust prior prices if any stock exhibits a >20% move matching an ex-split/bonus corporate action on nse_date
    if nse_date >= last_cached:
        for sym in symbols:
            q = quotes.get(sym)
            if not isinstance(q, dict) or not q.get("close"):
                continue
            prev_px = float(raw_df[sym].dropna().iloc[-1]) if sym in raw_df.columns and not raw_df[sym].dropna().empty else 0.0
            new_px = float(q["close"])
            if prev_px > 0 and abs(new_px / prev_px - 1.0) > 0.20:
                try:
                    ca = client.get_corporate_actions(sym)
                    events = ca.get("corporateActions") or ca.get("actions") or ca.get("data") or []
                    for ev in events:
                        ex_dt = str(ev.get("exDate", ""))[:10]
                        atype = str(ev.get("actionType", "")).upper()
                        factor = float(ev.get("adjustmentFactor", 0.0) or 0.0)
                        if ex_dt == nse_date and atype in {"SPLIT", "BONUS"} and 0.0 < factor < 1.0:
                            mask_pre = adj_df.index < ex_dt
                            adj_df.loc[mask_pre, sym] = adj_df.loc[mask_pre, sym] * factor
                            raw_df.loc[mask_pre, sym] = raw_df.loc[mask_pre, sym] * factor
                            telemetry["corp_actions_checked"].append({"symbol": sym, "date": ex_dt, "type": atype, "factor": factor})
                except Exception:
                    pass

    # Append new trading day row or patch missing nulls on nse_date
    if nse_date > last_cached:
        telemetry["appended_date"] = nse_date
        adj_df.loc[nse_date, :] = np.nan
        raw_df.loc[nse_date, :] = np.nan
        vol_df.loc[nse_date, :] = np.nan
        bench_df.loc[nse_date, :] = np.nan
        target_dt = nse_date
    else:
        target_dt = last_cached

    for sym in symbols:
        q = quotes.get(sym)
        if not isinstance(q, dict) or str(q.get("date", ""))[:10] != target_dt:
            continue
        cl = float(q.get("close") or 0.0)
        if cl > 0 and (nse_date > last_cached or pd.isna(raw_df.loc[target_dt, sym])):
            raw_df.loc[target_dt, sym] = cl
            adj_df.loc[target_dt, sym] = cl
            if q.get("volume"):
                vol_df.loc[target_dt, sym] = float(q["volume"])

    for etf in TRACKED_ETFS:
        if etf not in bench_df.columns:
            continue
        q = quotes.get(etf)
        if isinstance(q, dict) and str(q.get("date", ""))[:10] == target_dt:
            cl = float(q.get("close") or 0.0)
            if cl > 0 and (nse_date > last_cached or pd.isna(bench_df.loc[target_dt, etf])):
                bench_df.loc[target_dt, etf] = cl

    # Fill India VIX & Nifty Midcap 150 from mood / index history on target_dt if missing
    if nse_date > last_cached and telemetry.get("mood_ok"):
        vix_info = telemetry["mood"].get("india_vix") or {}
        if vix_info.get("level"):
            bench_df.loc[target_dt, "INDIA_VIX"] = float(vix_info["level"])
    idx_data = (telemetry.get("index_history") or {}).get("data") or []
    if idx_data and "NIFTY_MIDCAP_150" in bench_df.columns:
        last_bar = idx_data[-1]
        if str(last_bar.get("date", ""))[:10] == target_dt and last_bar.get("close"):
            bench_df.loc[target_dt, "NIFTY_MIDCAP_150"] = float(last_bar["close"])

    return telemetry


def _build_discrete_orders(weights: dict[str, float], raw_prices: pd.Series, score_9m: pd.Series, score_12m: pd.Series, capital: float, sweep_px: float) -> dict[str, Any]:
    rows = []
    deployed = 0.0
    for rank, (sym, w) in enumerate(weights.items(), start=1):
        px = float(raw_prices.get(sym, np.nan))
        tgt = capital * w
        sh = int(tgt // px) if math.isfinite(px) and px > 0 else 0
        inv = sh * px if math.isfinite(px) and px > 0 else 0.0
        deployed += inv
        rows.append(
            {
                "rank": rank,
                "symbol": sym,
                "tier": "Top20" if w > 0.025 else "Rank21-30",
                "weight_pct": round(w * 100.0, 2),
                "price": px,
                "target_inr": round(tgt, 2),
                "shares": sh,
                "invested_inr": round(inv, 2),
                "raw_9m_pct": float(score_9m.get(sym, np.nan) * 100.0) if pd.notna(score_9m.get(sym)) else None,
                "raw_12m_pct": float(score_12m.get(sym, np.nan) * 100.0) if pd.notna(score_12m.get(sym)) else None,
            }
        )
    rem_cash = max(0.0, capital - deployed)
    liq_units = int(rem_cash // sweep_px) if math.isfinite(sweep_px) and sweep_px > 0 else 0
    liq_val = liq_units * sweep_px if liq_units > 0 else 0.0
    return {
        "capital_inr": capital,
        "deployed_equity_inr": round(deployed, 2),
        "deployed_equity_pct": round(deployed / capital * 100.0, 2) if capital > 0 else 0.0,
        "held_count": sum(1 for r in rows if r["shares"] > 0),
        "total_slots": len(rows),
        "liquidcase_sweep_units": liq_units,
        "liquidcase_sweep_inr": round(liq_val, 2),
        "idle_cash_inr": round(max(0.0, rem_cash - liq_val), 2),
        "orders": rows,
    }


def _extract_regime_triggers(feat: pd.DataFrame, alloc_series: pd.Series) -> list[dict[str, Any]]:
    triggers = []
    prev_val = float(alloc_series.iloc[0]) if not alloc_series.empty else 1.0
    for dt, val in alloc_series.items():
        v = float(val)
        if abs(v - prev_val) > 0.05:
            if v < prev_val:
                ev = "EXIT_TO_DEFENSIVE" if v < 0.40 else "SHIFT_TO_CAUTIOUS"
            else:
                ev = "REENTER_EQUITY" if v >= 0.99 else "STEP_UP_CAUTIOUS"
            triggers.append(
                {
                    "date": str(dt),
                    "event": ev,
                    "equity_pct": round(v * 100.0, 1),
                    "ew_dd252_pct": round(float(feat.loc[dt, "ew_dd252"]) * 100.0, 2),
                }
            )
            prev_val = v
    return triggers


def build_live_state() -> dict[str, Any]:
    strategy_cfg = json.loads((REPO_ROOT / "config" / "strategy.json").read_text())
    univ = load_universe(REPO_ROOT, refresh=False)
    symbols = univ["Symbol"].astype(str).tolist()

    # Load cache if available, otherwise bootstrap from Yahoo Finance
    data = fetch_and_cache_all(REPO_ROOT, range_str="10y", refresh=False)
    adj_df, raw_df, vol_df, bench_df = data["adj"].copy(), data["raw"].copy(), data["vol"].copy(), data["bench"].copy()

    nse_telemetry = _sync_live_nse_quotes(adj_df, raw_df, vol_df, bench_df, symbols)
    # Persist any live appended day locally if cache directory exists
    cache_dir = REPO_ROOT / "data" / "cache"
    if cache_dir.exists():
        adj_df.to_csv(cache_dir / "adj_close.csv.gz", compression="gzip")
        raw_df.to_csv(cache_dir / "raw_close.csv.gz", compression="gzip")
        vol_df.to_csv(cache_dir / "volume.csv.gz", compression="gzip")
        bench_df.to_csv(cache_dir / "benchmarks.csv.gz", compression="gzip")

    feat = build_regime_features(adj_df, bench_df)
    def_df = build_defensive_series(bench_df)
    stock_ma50 = adj_df.rolling(50, min_periods=30).mean()

    scores = {
        "raw_9m": compute_score_matrix(adj_df.ffill(limit=2), 189, 21, "raw"),
        "raw_12m": compute_score_matrix(adj_df.ffill(limit=2), 252, 21, "raw"),
        "risk_9m": compute_score_matrix(adj_df.ffill(limit=2), 189, 21, "risk_adj"),
        "sortino_9m": compute_score_matrix(adj_df.ffill(limit=2), 189, 21, "sortino"),
    }

    last_dt = str(adj_df.index[-1])
    s9 = scores["raw_9m"].loc[last_dt]
    s12 = scores["raw_12m"].loc[last_dt]
    sr9 = scores["risk_9m"].loc[last_dt]
    ss9 = scores["sortino_9m"].loc[last_dt]
    raw_last = raw_df.ffill(limit=5).loc[last_dt] if last_dt in raw_df.index else raw_df.ffill(limit=5).iloc[-1]
    ma50_last = adj_df.ffill(limit=2).rolling(50, min_periods=30).mean().loc[last_dt]

    # Run clean 2018-01-02..today backtests for dashboard equity curves & preset comparisons
    clean_start = "2018-01-02"
    presets_summary: dict[str, Any] = {}
    nav_curves: dict[str, pd.Series] = {}

    reg_mf4, mf_diag = compute_multifactor_state(feat, bench_df)
    tg_status = get_trend_gold_status(bench_df)
    reg_dd15 = compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.15, "reenter_ma": 50, "defensive_shift": 0.75})
    reg_dd75 = compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75})
    reg_dd50 = compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.50})
    reg_none = compute_equity_allocation_series(feat, "none", {})

    def _slv(key: str, wt: str) -> list[dict[str, Any]]:
        return [{"score_key": key, "top_n": 20, "alloc": 0.5, "weighting": wt}, {"score_key": key, "top_n": 30, "alloc": 0.5, "weighting": wt}]

    sweep_col = "LIQUID_REALISTIC" if "LIQUID_REALISTIC" in def_df.columns else "LIQUID_TBILL"
    curve_specs = {
        "R3_Risk9m_6M_MF4_TrendGold150": (_slv("risk_9m", "equal"), 126, reg_mf4, "TREND_GOLD150_OR_LIQ", "weakest_first", False, False),
        "R3_Disc1L_SlotCap": (_slv("risk_9m", "equal"), 126, reg_mf4, "TREND_GOLD150_OR_LIQ", "weakest_first", True, True),
        "S3_Sortino9m_6M_MF4_TrendGold150": (_slv("sortino_9m", "equal"), 126, reg_mf4, "TREND_GOLD150_OR_LIQ", "weakest_first", False, False),
        "S3_Disc1L_SlotCap": (_slv("sortino_9m", "equal"), 126, reg_mf4, "TREND_GOLD150_OR_LIQ", "weakest_first", True, True),
        "R2_Risk9m_6M_DD15_Sh75_GoldLiq50": (_slv("risk_9m", "equal"), 126, reg_dd15, "GOLD_50_LIQUID_50", "weakest_first", False, False),
        "R2_Disc1L_SlotCap": (_slv("risk_9m", "equal"), 126, reg_dd15, "GOLD_50_LIQUID_50", "weakest_first", True, True),
        "R7_NormZ_Risk9m_6M_DD15_Sh75_GoldLiq50": (_slv("risk_9m", "norm_z"), 126, reg_dd15, "GOLD_50_LIQUID_50", "weakest_first", False, False),
        "F2_9m_6M_DD12_Sh75_GoldLiq50": (_slv("raw_9m", "equal"), 126, reg_dd75, "GOLD_50_LIQUID_50", "weakest_first", False, False),
        "F2_Disc1L_SlotCap": (_slv("raw_9m", "equal"), 126, reg_dd75, "GOLD_50_LIQUID_50", "weakest_first", True, True),
        "F7_NormZ_9m_6M_DD12_Sh75_GoldLiq50": (_slv("raw_9m", "norm_z"), 126, reg_dd75, "GOLD_50_LIQUID_50", "weakest_first", False, False),
        "F1_9m_6M_DD12_Sh50_GoldLiq50": (_slv("raw_9m", "equal"), 126, reg_dd50, "GOLD_50_LIQUID_50", "weakest_first", False, False),
        "F4_12m_12M_DD12_Sh50_GoldLiq50": (_slv("raw_12m", "equal"), 252, reg_dd50, "GOLD_50_LIQUID_50", "weakest_first", False, False),
        "Unhedged_9m_6M": (_slv("raw_9m", "equal"), 126, reg_none, "GOLD_50_LIQUID_50", "weakest_first", False, False),
    }

    for key, (slv, reb, alloc, def_a, trim, af, sw) in curve_specs.items():
        res = run_backtest(
            adj_df,
            raw_df,
            scores,
            def_df,
            alloc,
            BacktestConfig(
                start_date=clean_start,
                end_date=last_dt,
                initial_capital=100_000.0,
                rebalance_days=reb,
                cost_bps_one_way=15.0,
                apply_tax=True,
                defensive_asset=def_a,
                trim_mode=trim,
                sleeves=slv,
                discrete_shares=af,
                affordability_filter=af,
                sweep_idle_cash=sw,
                sweep_asset=sweep_col,
            ),
            stock_ma50=stock_ma50,
        )
        presets_summary[key] = res["metrics"]
        nav_curves[key] = res["nav"]

    # Add benchmarks indexed to ₹1,00,000 on 2018-01-02
    ew_series = build_ew_index(adj_df).loc[clean_start:last_dt]
    nav_curves["Midcap150_EW_Universe"] = ew_series / float(ew_series.iloc[0]) * 100_000.0
    presets_summary["Midcap150_EW_Universe"] = compute_metrics(nav_curves["Midcap150_EW_Universe"])

    gold50_series = def_df["GOLD_50_LIQUID_50"].loc[clean_start:last_dt].dropna()
    nav_curves["GOLD_50_LIQUID_50"] = gold50_series / float(gold50_series.iloc[0]) * 100_000.0
    presets_summary["GOLD_50_LIQUID_50"] = compute_metrics(nav_curves["GOLD_50_LIQUID_50"])

    tg_series = def_df["TREND_GOLD150_OR_LIQ"].loc[clean_start:last_dt].dropna()
    nav_curves["TREND_GOLD150_OR_LIQ"] = tg_series / float(tg_series.iloc[0]) * 100_000.0
    presets_summary["TREND_GOLD150_OR_LIQ"] = compute_metrics(nav_curves["TREND_GOLD150_OR_LIQ"])

    # Compute current T+1 live posture from causal 4-block multi-factor composite + TrendGold EMA150 gate
    ew_cur = float(feat["ew_index"].iloc[-1])
    ew_peak252 = float(feat["ew_index"].iloc[max(0, len(feat) - 252) :].max())
    ew_dd_now = ew_cur / ew_peak252 - 1.0 if ew_peak252 > 0 else 0.0
    ew_ma50_now = float(feat["ew_ma50"].iloc[-1])
    ew_ma200_now = float(feat["ew_ma200"].iloc[-1])
    mf_last = mf_diag.iloc[-1]
    t1_eq_frac = float(mf_last["target_eq_eod"])
    def_frac = round(max(0.0, 1.0 - t1_eq_frac), 4)
    active_def_etf = str(tg_status.get("active_defensive_etf", "LIQUIDCASE"))

    goldbees_px = float(bench_df["GOLDBEES"].dropna().iloc[-1]) if "GOLDBEES" in bench_df.columns else 100.0
    goldbees_ma50 = float(bench_df["GOLDBEES"].dropna().rolling(50).mean().iloc[-1]) if "GOLDBEES" in bench_df.columns else 100.0
    liquidcase_px = float(bench_df["LIQUIDCASE"].dropna().iloc[-1]) if "LIQUIDCASE" in bench_df.columns else 117.10
    vix_now = float(bench_df["INDIA_VIX"].dropna().iloc[-1]) if "INDIA_VIX" in bench_df.columns else None
    if nse_telemetry.get("mood_ok") and (nse_telemetry["mood"].get("india_vix") or {}).get("level"):
        vix_now = float(nse_telemetry["mood"]["india_vix"]["level"])

    midcap_idx_now = float(bench_df["NIFTY_MIDCAP_150"].dropna().iloc[-1]) if "NIFTY_MIDCAP_150" in bench_df.columns else None
    midcap_idx_dd252 = None
    if "NIFTY_MIDCAP_150" in bench_df.columns:
        ms = bench_df["NIFTY_MIDCAP_150"].dropna()
        if not ms.empty:
            mpk = float(ms.iloc[max(0, len(ms) - 252) :].max())
            midcap_idx_dd252 = (float(ms.iloc[-1]) / mpk - 1.0) * 100.0

    # Top 60 rankings + Risk-Adj, Sortino, Raw weights for client-side dynamic calculator
    ranked_raw9 = s9.dropna().sort_values(ascending=False)
    ranked_risk9 = sr9.dropna().sort_values(ascending=False)
    ranked_sortino9 = ss9.dropna().sort_values(ascending=False)
    w_raw_eq = combine_sleeves([(0.5, select_sleeve_weights(s9, 20, "equal")), (0.5, select_sleeve_weights(s9, 30, "equal"))])
    w_raw_slotcap = select_affordable_blend({"raw_9m": s9}, _slv("raw_9m", "equal"), raw_last, 100_000.0)
    w_risk_eq = combine_sleeves([(0.5, select_sleeve_weights(sr9, 20, "equal")), (0.5, select_sleeve_weights(sr9, 30, "equal"))])
    w_risk_slotcap = select_affordable_blend({"risk_9m": sr9}, _slv("risk_9m", "equal"), raw_last, 100_000.0)
    w_sortino_eq = combine_sleeves([(0.5, select_sleeve_weights(ss9, 20, "equal")), (0.5, select_sleeve_weights(ss9, 30, "equal"))])
    w_sortino_slotcap = select_affordable_blend({"sortino_9m": ss9}, _slv("sortino_9m", "equal"), raw_last, 100_000.0)
    w_risk_nz = combine_sleeves([(0.5, select_sleeve_weights(sr9, 20, "norm_z")), (0.5, select_sleeve_weights(sr9, 30, "norm_z"))])
    w_risk_nz_slotcap = select_affordable_blend({"risk_9m": sr9}, _slv("risk_9m", "norm_z"), raw_last, 100_000.0)
    w_blend_nz = combine_sleeves([(0.5, select_sleeve_weights(s9, 20, "norm_z")), (0.5, select_sleeve_weights(s9, 30, "norm_z"))])
    w_blend_12m_eq = combine_sleeves([(0.5, select_sleeve_weights(s12, 20, "equal")), (0.5, select_sleeve_weights(s12, 30, "equal"))])

    rank_raw9_map = {sym: r for r, sym in enumerate(ranked_raw9.index, start=1)}
    rank_sortino9_map = {sym: r for r, sym in enumerate(ranked_sortino9.index, start=1)}
    rank12_map = {sym: r for r, sym in enumerate(s12.dropna().sort_values(ascending=False).index, start=1)}
    adj_live = adj_df.ffill(limit=2)
    rankings_top60 = []
    for r_risk, sym in enumerate(ranked_risk9.index[:60], start=1):
        px = float(raw_last.get(sym, np.nan))
        adx = float(adj_live.loc[last_dt, sym]) if pd.notna(adj_live.loc[last_dt, sym]) else float("nan")
        m50 = float(ma50_last.get(sym, np.nan))
        rankings_top60.append(
            {
                "rank_risk9m": r_risk,
                "rank_sortino9m": rank_sortino9_map.get(sym),
                "rank_9m": rank_raw9_map.get(sym),
                "rank_12m": rank12_map.get(sym),
                "symbol": sym,
                "price": px,
                "risk_9m": float(sr9[sym]),
                "sortino_9m": float(ss9[sym]) if pd.notna(ss9.get(sym)) else None,
                "raw_9m_pct": float(s9[sym] * 100.0) if pd.notna(s9.get(sym)) else None,
                "raw_12m_pct": float(s12.get(sym, np.nan) * 100.0) if pd.notna(s12.get(sym)) else None,
                "above_ma50": bool(math.isfinite(adx) and math.isfinite(m50) and adx > m50),
                "risk_ew_5050_wt_pct": round(w_risk_eq.get(sym, 0.0) * 100.0, 2),
                "risk_nz_5050_wt_pct": round(w_risk_nz.get(sym, 0.0) * 100.0, 2),
                "risk_affordable_slotcap_1l": sym in w_risk_slotcap,
                "ew_5050_wt_pct": round(w_raw_eq.get(sym, 0.0) * 100.0, 2),
                "nz_5050_wt_pct": round(w_blend_nz.get(sym, 0.0) * 100.0, 2),
                "affordable_slotcap_1l": sym in w_raw_slotcap,
            }
        )

    # Full 150 scored list (compact for client-side dynamic preset switching: Risk9M / Sortino9M / Raw9M / Raw12M & Equal / NormZ)
    universe_scores = []
    for sym in ranked_risk9.index:
        px = float(raw_last.get(sym, np.nan))
        if not math.isfinite(px) or px <= 0:
            continue
        universe_scores.append(
            {
                "sym": sym,
                "px": round(px, 2),
                "rk9": round(float(sr9[sym]), 4),
                "so9": round(float(ss9[sym]), 4) if pd.notna(ss9.get(sym)) else None,
                "r9": round(float(s9[sym]) * 100.0, 2) if pd.notna(s9.get(sym)) else None,
                "r12": round(float(s12[sym]) * 100.0, 2) if pd.notna(s12.get(sym)) else None,
                "ma50": bool(pd.notna(ma50_last.get(sym)) and adj_live.loc[last_dt, sym] > ma50_last.get(sym)),
            }
        )

    # Downsample chart series cleanly to weekly (every 3 trading days + latest point) for fast browser load
    nav_df = pd.DataFrame(nav_curves).loc[clean_start:]
    sample_idx = list(range(0, len(nav_df), 3))
    if sample_idx[-1] != len(nav_df) - 1:
        sample_idx.append(len(nav_df) - 1)
    nav_sub = nav_df.iloc[sample_idx]
    dd_sub = (nav_df / nav_df.cummax() - 1.0).iloc[sample_idx] * 100.0
    ew_dd_sub = feat["ew_dd252"].reindex(nav_sub.index) * 100.0

    # Synchronize Clean Backtest Matrix directly with live 2018-26 curve simulations
    clean_matrix_specs = [
        ("R3 Recommended (₹1L SlotCap)", "R3_Disc1L_SlotCap"),
        ("R3 Optimal MF (Frac)", "R3_Risk9m_6M_MF4_TrendGold150"),
        ("S3 Runner-Up (₹1L SlotCap)", "S3_Disc1L_SlotCap"),
        ("S3 Sortino MF (Frac)", "S3_Sortino9m_6M_MF4_TrendGold150"),
        ("R7 Concentrated NormZ", "R7_NormZ_Risk9m_6M_DD15_Sh75_GoldLiq50"),
        ("R2 Simple DD15 (Frac)", "R2_Risk9m_6M_DD15_Sh75_GoldLiq50"),
        ("R2 Simple DD15 (₹1L SlotCap)", "R2_Disc1L_SlotCap"),
        ("F2 Legacy V1 (Frac)", "F2_9m_6M_DD12_Sh75_GoldLiq50"),
        ("F4 Large Cap 12M", "F4_12m_12M_DD12_Sh50_GoldLiq50"),
        ("Baseline Unhedged 9M", "Unhedged_9m_6M"),
        ("Benchmark", "Midcap150_EW_Universe"),
        ("Defensive Gate ETF", "TREND_GOLD150_OR_LIQ"),
        ("Defensive 50/50", "GOLD_50_LIQUID_50"),
    ]
    clean_ver = []
    for scen, k in clean_matrix_specs:
        m = presets_summary.get(k)
        if m:
            clean_ver.append({"scenario": scen, "name": k, **m})

    # Clean ETF-Free 16-Year Zero-Survivorship Point-in-Time Validation Summary
    pit_validation_16yr: list[dict[str, Any]] = []
    src_16_path = REPO_ROOT / "reports" / "src_exact_16yr_verification.json"
    deep_r1_path = REPO_ROOT / "reports" / "numba_deep_research.json"
    deep_r2_path = REPO_ROOT / "reports" / "numba_part2_verification.json"
    research_v3: dict[str, Any] = {}
    if src_16_path.exists():
        src16 = json.loads(src_16_path.read_text())
        research_v3["src_exact_16yr"] = src16
        lbl_map = {
            "R3_src_Frac": "R3 Optimal (Risk9M · 4-Block MF · TrendGold150)",
            "R3_src_Disc1L": "R3 Optimal ₹1L Discrete SlotCap",
            "S3_src_Frac": "S3 Sortino9M · 4-Block MF · TrendGold150",
            "S3_src_Disc1L": "S3 Sortino9M ₹1L Discrete SlotCap",
            "R3_NormZ_Frac": "R3 Concentrated NormZ · 4-Block MF · TrendGold150",
            "R2_src_Frac": "R2 Simple Trigger (Risk9M · DD>15% · 50/50 Gold+Liq)",
            "R2_src_Disc1L": "R2 Simple Trigger ₹1L Discrete SlotCap",
            "F2_src_Frac": "F2 Legacy V1 (Raw9M · DD>12% · 50/50 Gold+Liq)",
            "Unhedged_Risk9M": "Unhedged Risk9M EW (No Regime Hedge)",
            "Unhedged_Sortino9M": "Unhedged Sortino9M EW (No Regime Hedge)",
        }
        pit_validation_16yr.append(
            {
                "universe": "Clean ETF-Free PIT (772 Eqs)",
                "strategy": "Passive Midcap 150 EW Benchmark (Zero-Leak)",
                "p1_cagr": 0.40,
                "p2_cagr": 15.23,
                "p3_cagr": 14.58,
                "full_cagr": 9.77,
                "full_dd": -54.08,
                "full_sh": 0.18,
            }
        )
        for row in src16.get("rows", []):
            disp = lbl_map.get(row["lbl"], row["lbl"])
            pit_validation_16yr.append(
                {
                    "universe": "Clean ETF-Free PIT (772 Eqs)",
                    "strategy": disp,
                    "p1_cagr": row["P1_cagr"],
                    "p2_cagr": row["P2_cagr"],
                    "p3_cagr": row["P3_cagr"],
                    "full_cagr": row["FULL_cagr"],
                    "full_dd": row["FULL_dd"],
                    "full_sh": row["FULL_sh"],
                }
            )
    if deep_r1_path.exists():
        r1j = json.loads(deep_r1_path.read_text())
        research_v3["gold_liquid_frontier"] = [x for x in r1j.get("gold_liquid_frontier", []) if x.get("signal") == "EQ4_MultiFactor"]
    if deep_r2_path.exists():
        r2j = json.loads(deep_r2_path.read_text())
        research_v3["crash_anatomy"] = r2j.get("crash_anatomy", [])
        research_v3["fno_hedges"] = r2j.get("fno_hedges", [])[:3]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date": last_dt,
        "strategy_id": strategy_cfg["primary_strategy"],
        "universe_count": len(symbols),
        "nse_mcp_status": {
            "source": nse_telemetry.get("source"),
            "mood_ok": nse_telemetry.get("mood_ok"),
            "bulk_quotes_fetched": nse_telemetry.get("bulk_quotes"),
            "appended_date": nse_telemetry.get("appended_date"),
            "corp_actions_adjusted": nse_telemetry.get("corp_actions_checked"),
            "market_mood": nse_telemetry.get("mood"),
        },
        "regime": {
            "state": "RISK_ON" if t1_eq_frac >= 0.99 else ("CAUTIOUS" if t1_eq_frac >= 0.50 else "DEFENSIVE"),
            "state_code": int(mf_last["state_code"]),
            "active_defensive_etf": active_def_etf,
            "target_equity_pct": round(t1_eq_frac * 100.0, 1),
            "target_goldbees_pct": round(def_frac * 100.0, 1) if active_def_etf == "GOLDBEES" else 0.0,
            "target_liquidcase_pct": round(def_frac * 100.0, 1) if active_def_etf == "LIQUIDCASE" else 0.0,
            "ew_index": round(ew_cur, 2),
            "ew_ma50": round(ew_ma50_now, 2),
            "ew_ma200": round(ew_ma200_now, 2),
            "ew_dd252_pct": round(ew_dd_now * 100.0, 2),
            "trigger_dd_pct": -15.0,
            "reenter_dd_pct": -9.0,
            "midcap150_index": midcap_idx_now,
            "midcap150_dd252_pct": round(midcap_idx_dd252, 2) if midcap_idx_dd252 is not None else None,
            "breadth_50_sma10_pct": round(float(feat["breadth_50_sma10"].iloc[-1]) * 100.0, 2),
            "india_vix": vix_now,
            "trend_gold_status": tg_status,
            "multifactor_blocks": {
                "b1_trend": round(float(mf_last["b1_trend"]), 3),
                "b2_vol": round(float(mf_last["b2_vol"]), 3),
                "b3_breadth": round(float(mf_last["b3_breadth"]), 3),
                "b4_macro_gold": round(float(mf_last["b4_macro_gold"]), 3),
                "composite_raw": round(float(mf_last["composite_raw"]), 3),
                "composite_smooth": round(float(mf_last["composite_smooth"]), 3),
                "weights": {"trend": 0.44, "vol_vix": 0.10, "breadth": 0.34, "macro_gold": 0.12},
                "thresholds": {"defensive": -0.25, "cautious": -0.10, "reenter_risk_on": 0.18},
            },
            "etfs": {
                "GOLDBEES": {"price": goldbees_px, "above_ma50": goldbees_px > goldbees_ma50, "ema150": tg_status.get("gold_ema150"), "above_ema150_gate": tg_status.get("gold_above_gate")},
                "LIQUIDCASE": {"price": liquidcase_px},
                "EBBETF0431": {"price": float(bench_df["EBBETF0431"].dropna().iloc[-1]) if "EBBETF0431" in bench_df.columns else None},
                "LTGILTBEES": {"price": float(bench_df["LTGILTBEES"].dropna().iloc[-1]) if "LTGILTBEES" in bench_df.columns else None},
            },
            "historical_triggers": _extract_regime_triggers(feat.loc[clean_start:], reg_mf4.loc[clean_start:]),
            "legacy_dd15_triggers": _extract_regime_triggers(feat.loc[clean_start:], reg_dd15.loc[clean_start:]),
        },
        "orders_1L": {
            "R3_Risk9M_SlotCap_LiqSweep": _build_discrete_orders(w_risk_slotcap, raw_last, s9, s12, 100_000.0, liquidcase_px),
            "S3_Sortino9M_SlotCap_LiqSweep": _build_discrete_orders(w_sortino_slotcap, raw_last, s9, s12, 100_000.0, liquidcase_px),
            "R2_Risk9M_SlotCap_LiqSweep": _build_discrete_orders(w_risk_slotcap, raw_last, s9, s12, 100_000.0, liquidcase_px),
            "R2_Risk9M_NoCap": _build_discrete_orders(w_risk_eq, raw_last, s9, s12, 100_000.0, liquidcase_px),
            "R7_Risk9M_NormZ_SlotCap": _build_discrete_orders(w_risk_nz_slotcap, raw_last, s9, s12, 100_000.0, liquidcase_px),
            "F2_SlotCap_LiqSweep": _build_discrete_orders(w_raw_slotcap, raw_last, s9, s12, 100_000.0, liquidcase_px),
            "F2_NoCap": _build_discrete_orders(w_raw_eq, raw_last, s9, s12, 100_000.0, liquidcase_px),
            "F4_12M_NoCap": _build_discrete_orders(w_blend_12m_eq, raw_last, s9, s12, 100_000.0, liquidcase_px),
        },
        "rankings_top60": rankings_top60,
        "universe_scores": universe_scores,
        "performance_summary": presets_summary,
        "charts": {
            "dates": [str(d) for d in nav_sub.index],
            "nav": {col: [round(float(x), 2) for x in nav_sub[col].values] for col in nav_sub.columns},
            "drawdown_pct": {col: [round(float(x), 2) for x in dd_sub[col].values] for col in dd_sub.columns},
            "ew_dd252_pct": [round(float(x), 2) for x in ew_dd_sub.values],
        },
        "clean_verification": clean_ver,
        "pit_validation_16yr": pit_validation_16yr,
        "research_v3": research_v3,
    }

    safe_payload = sanitize_for_json(payload)
    out_path = REPO_ROOT / "data" / "live_state.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(safe_payload, indent=2, allow_nan=False))
    print(f"Wrote {out_path} (as_of={last_dt}, stocks={len(universe_scores)}, regime={safe_payload['regime']['state']})")

    # Build/append compact daily forward tape & step autonomous Paper Trading Bot ledgers (idempotent)
    adj_last = adj_df.ffill(limit=2).loc[last_dt] if last_dt in adj_df.index else adj_df.ffill(limit=2).iloc[-1]
    tape_entry = build_forward_tape_entry(
        date_str=last_dt,
        raw_last=raw_last,
        bench_df=bench_df,
        etf_symbols=TRACKED_ETFS,
        ma50_last=ma50_last,
        adj_last=adj_last,
        scores=scores,
        mf_diag_row=mf_last,
        tg_status=tg_status,
        reg_dd15_val=float(reg_dd15.loc[last_dt]),
        reg_dd75_val=float(reg_dd75.loc[last_dt]),
        reg_dd50_val=float(reg_dd50.loc[last_dt]),
        ew_index_val=ew_cur,
        corp_actions=nse_telemetry.get("corp_actions_checked") or [],
    )
    tape_path = REPO_ROOT / "data" / "forward_tape.json"
    ledger_path = REPO_ROOT / "data" / "paper_ledger.json"
    tape_obj = update_forward_tape(tape_path, tape_entry)
    ledger_obj = update_paper_ledger(ledger_path, tape_obj)
    print(
        f"Paper Bot updated (inception={ledger_obj['inception_date']}, as_of={ledger_obj['last_date']}, "
        f"1L_NAV=₹{ledger_obj['accounts']['R3_1L_SlotCap']['summary']['nav']:,.2f}, "
        f"10L_NAV=₹{ledger_obj['accounts']['R3_10L_Standard']['summary']['nav']:,.2f})"
    )
    return safe_payload


if __name__ == "__main__":
    build_live_state()
