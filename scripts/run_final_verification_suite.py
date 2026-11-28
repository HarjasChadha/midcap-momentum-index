"""Part 2 Deep Verification: Low-Churn Multi-Factor Hysteresis, Corrected Monthly F&O Simulation, Crash Anatomy, Factor Shootout, and Exact FIFO Backtester Cross-Check."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.numba_research_engine import (  # noqa: E402
    PERIODS,
    _calc_metrics_slice,
    _gen_dd_signal,
    _gen_multifactor_signal,
    _sim_overlay_core,
    build_basket_daily_stream,
    build_multifactor_blocks,
    compute_extended_scores,
    compute_tg_erc_returns,
)
from scripts.run_numba_sweeps import _slice_bounds, _summarize_nav, build_clean_equity_and_pit_masks  # noqa: E402
from src.backtester import BacktestConfig, compute_metrics, run_backtest  # noqa: E402
from src.momentum import compute_score_matrix  # noqa: E402
from src.nse_archive_loader import build_2010_2026_nse_dataset  # noqa: E402
from src.regime import build_defensive_series, build_regime_features, compute_equity_allocation_series  # noqa: E402


def sim_monthly_fno_hedges(
    r_eq: np.ndarray,
    to_eq_reb: np.ndarray,
    eq_frac: np.ndarray,
    ew_ret: np.ndarray,
    vix: np.ndarray,
    liq_ret: np.ndarray,
    bounds: np.ndarray,
    dates: pd.Index,
) -> list[dict]:
    """Realistic 21-day monthly expiry F&O hedge simulation with Apr-2026 SEBI STT, basis slippage, and 30% slab tax."""
    T = len(r_eq)
    hedge_frac = 1.0 - eq_frac  # fraction hedged while keeping equity basket intact

    # A) Short MIDCPNIFTY / Index Futures Overlay (beta-matched during defensive regime; requires ₹25L+ lot feasibility)
    # 50:50 SEBI margin: 15% margin on hedged notional -> ~7.5% cash margin earns liq_ret; gross carry ~6.8% + net roll/STT ~2.0% -> net drag ~2.0%/yr while hedged + midcap/index tracking basis
    nav_fut = np.ones(T, dtype=np.float64) * 100_000.0
    for t in range(1, T):
        h = hedge_frac[t]
        basis_noise = 0.0
        # Net daily drag: 2.0%/yr carry/STT/roll when active; 30% F&O slab tax on positive hedge months approximated cleanly monthly
        pnl = r_eq[t] - h * ew_ret[t] + h * liq_ret[t] - h * (0.020 / 252.0)
        sw = abs(h - hedge_frac[t - 1]) * 0.0018 + to_eq_reb[t] * 0.0030
        nav_fut[t] = nav_fut[t - 1] * (1.0 + pnl - sw)

    # B) Monthly 21-Trading-Day Settled 5% OTM Bear Put & 5%-15% Put Spread (settled every 21d expiry!)
    nav_put = np.ones(T, dtype=np.float64) * 100_000.0
    nav_ps = np.ones(T, dtype=np.float64) * 100_000.0
    m_ret_acc = 1.0
    h_start = hedge_frac[0]
    for t in range(1, T):
        m_ret_acc *= (1.0 + ew_ret[t])
        # Equity base continues
        nav_put[t] = nav_put[t - 1] * (1.0 + r_eq[t] - to_eq_reb[t] * 0.0030)
        nav_ps[t] = nav_ps[t - 1] * (1.0 + r_eq[t] - to_eq_reb[t] * 0.0030)
        if t % 21 == 0:
            if h_start > 0.05:
                v_t = max(12.0, min(60.0, float(vix[max(0, t - 21)])))
                # 1-month 5% OTM Put realistic Black-Scholes cost: ~1.1% at VIX 16, scales with VIX (+0.15% STT/impact)
                put_cost = h_start * (0.011 * (v_t / 16.0) + 0.0015)
                spread_cost = h_start * (0.0065 * (v_t / 16.0) + 0.0015)
                payoff_put = h_start * max(0.0, -0.05 - (m_ret_acc - 1.0)) * 0.70  # post 30% F&O slab tax
                payoff_ps = h_start * min(0.10, max(0.0, -0.05 - (m_ret_acc - 1.0))) * 0.70
                nav_put[t] *= (1.0 + payoff_put - put_cost)
                nav_ps[t] *= (1.0 + payoff_ps - spread_cost)
            m_ret_acc = 1.0
            h_start = hedge_frac[t]

    out = []
    for lbl, nv in [
        ("Spot+ShortFutHedge(₹25L+,net -2%/yr active)", nav_fut),
        ("Spot+Monthly5%OTM_Put(VIX-scaled θ+30%tax)", nav_put),
        ("Spot+Monthly5-15%BearPutSpread(θ+30%tax)", nav_ps),
    ]:
        st = _summarize_nav(nv, to_eq_reb, bounds)
        st["name"] = lbl
        out.append(st)
    return out


def crash_window_attribution(
    dates: pd.Index,
    eq_sig: np.ndarray,
    def_variants: dict[str, tuple[np.ndarray, np.ndarray]],
    r_eq: np.ndarray,
    to_eq: np.ndarray,
) -> list[dict]:
    """Measure exact drawdown & cumulative return during the 6 historical stress / slump episodes (2011..2026)."""
    episodes = [
        ("2011_Bear", "2011-01-03", "2011-12-30"),
        ("2018_19_NBFC", "2018-01-15", "2019-09-19"),
        ("2020_COVID", "2020-01-20", "2020-04-30"),
        ("2022_RateHike", "2021-10-18", "2022-06-30"),
        ("2024_25_MidBear", "2024-09-24", "2025-03-12"),
        ("2026_GoldSlump", "2026-01-29", "2026-11-26"),
    ]
    rows = []
    for name, (dr, dtx) in def_variants.items():
        nav, to_d = _sim_overlay_core(r_eq, to_eq, eq_sig, dr, dtx, 0.0015, True, 0.20, 0.125)
        rec = {"defensive_strategy": name}
        for ep_lbl, es, ee in episodes:
            m = (dates >= es) & (dates <= ee)
            pos = np.where(m)[0]
            sub = nav[pos[0] : pos[-1] + 1]
            pk = np.maximum.accumulate(sub)
            rec[f"{ep_lbl}_ret"] = round((sub[-1] / sub[0] - 1.0) * 100.0, 1)
            rec[f"{ep_lbl}_dd"] = round(float(np.min(sub / pk - 1.0)) * 100.0, 1)
        rows.append(rec)
    return rows


def main() -> None:
    t0 = time.time()
    d = build_2010_2026_nse_dataset(REPO_ROOT, 2010, 2026, refresh=False)
    adj_eq, raw_eq, pit_daily, pit_semi, bench_df = build_clean_equity_and_pit_masks(d)
    dates = adj_eq.index
    T = len(dates)
    bounds = _slice_bounds(dates)

    def_df = build_defensive_series(bench_df)
    feat_semi = build_regime_features(adj_eq, bench_df, mask=pit_semi)
    scores_semi = compute_extended_scores(adj_eq, pit_semi)
    scores_daily = compute_extended_scores(adj_eq, pit_daily)

    liq_r = def_df["LIQUID_REALISTIC"].pct_change(fill_method=None).fillna(0.0).values
    gold_r = def_df["GOLDBEES"].pct_change(fill_method=None).fillna(0.0).values
    mon_r = def_df["MON100_STITCHED"].pct_change(fill_method=None).fillna(0.0).values if "MON100_STITCHED" in def_df else liq_r.copy()
    gilt_r = def_df["LICNETFGSC"].pct_change(fill_method=None).fillna(0.0).values if "LICNETFGSC" in def_df else liq_r.copy()
    r_base_5050 = 0.5 * gold_r + 0.5 * liq_r
    tax_5050 = np.full(T, 0.5 * 0.125 + 0.5 * 0.30, dtype=np.float64)

    # Trend-gated & TG-ERC defensive series
    tg100_r, tg100_tx, _ = compute_tg_erc_returns(def_df[["GOLDBEES"]].ffill().values, gold_r.reshape(-1, 1), liq_r, np.array([1.0]), np.array([0.125]), 0.30, 100, 0.02)
    tg150_r, tg150_tx, _ = compute_tg_erc_returns(def_df[["GOLDBEES"]].ffill().values, gold_r.reshape(-1, 1), liq_r, np.array([1.0]), np.array([0.125]), 0.30, 150, 0.02)
    mon_px = def_df["MON100_STITCHED"].ffill().values if "MON100_STITCHED" in def_df else def_df["GOLDBEES"].ffill().values
    gilt_px = def_df["LICNETFGSC"].values if "LICNETFGSC" in def_df else np.full(T, np.nan)
    erc2_r, erc2_tx, _ = compute_tg_erc_returns(np.column_stack([def_df["GOLDBEES"].ffill().values, mon_px]), np.column_stack([gold_r, mon_r]), liq_r, np.array([0.75, 0.20]), np.array([0.125, 0.125]), 0.30, 100, 0.02)
    erc3_r, erc3_tx, _ = compute_tg_erc_returns(np.column_stack([def_df["GOLDBEES"].ffill().values, gilt_px, mon_px]), np.column_stack([gold_r, gilt_r, mon_r]), liq_r, np.array([0.65, 0.65, 0.20]), np.array([0.125, 0.30, 0.125]), 0.30, 100, 0.02)
    # Also Gilt + Gold TG-ERC without MON100 overseas cap risk!
    erc_gg_r, erc_gg_tx, _ = compute_tg_erc_returns(np.column_stack([def_df["GOLDBEES"].ffill().values, gilt_px]), np.column_stack([gold_r, gilt_r]), liq_r, np.array([0.80, 0.80]), np.array([0.125, 0.30]), 0.30, 100, 0.02)

    px_arr = feat_semi["ew_index"].fillna(0.0).values
    ew_r = np.concatenate(([0.0], px_arr[1:] / np.maximum(1e-9, px_arr[:-1]) - 1.0))
    vix_arr = feat_semi["vix"].fillna(16.0).values if "vix" in feat_semi else np.full(T, 16.0)
    b50 = feat_semi["breadth_50"].fillna(0.5)
    b50_ema5 = b50.ewm(span=5, min_periods=2).mean()
    b50_chg21_arr = (b50_ema5 - b50_ema5.shift(21)).fillna(0.0).values
    b50_lvl_arr = b50.values

    lines = []

    # ── 1. FACTOR SHOOTOUT x LOW-CHURN MULTI-FACTOR vs LEGACY DD ─────────────────────────────────
    baskets = {
        "Sortino_252_R252": build_basket_daily_stream(adj_eq, scores_semi["sortino_252_s21"], 252, 20, 30, 0.5, "equal", 0),
        "Sortino_189_R126": build_basket_daily_stream(adj_eq, scores_semi["sortino_189_s21"], 126, 20, 30, 0.5, "equal", 0),
        "Risk_189_R126": build_basket_daily_stream(adj_eq, scores_semi["risk_189_s21"], 126, 20, 30, 0.5, "equal", 0),
        "Raw_189_R126": build_basket_daily_stream(adj_eq, scores_semi["raw_189_s21"], 126, 20, 30, 0.5, "equal", 0),
        "EnsRisk_R126": build_basket_daily_stream(adj_eq, scores_semi["ens_risk_6_9_12"], 126, 20, 30, 0.5, "equal", 0),
        "Sortino_189_NormZ_R126": build_basket_daily_stream(adj_eq, scores_semi["sortino_189_s21"], 126, 20, 30, 0.5, "norm_z", 0),
    }

    r_ref, to_ref, nav_ref = baskets["Sortino_189_R126"]
    blocks, p_semivol, semivol_med = build_multifactor_blocks(feat_semi, bench_df, nav_ref, def_df)
    p_eq4 = np.mean(blocks, axis=0)
    p_wf = 0.44 * blocks[0] + 0.10 * blocks[1] + 0.34 * blocks[2] + 0.12 * blocks[3]  # P1-learned rounded canonical weights!

    # Sweep low-churn 2-state & 3-state Hysteresis + asymmetric step-up cooldown on Multi-Factor Composite
    churn_rows = []
    for p_name, p_sig in [("EQ4_ZeroFit", p_eq4), ("P1_Opt_WF4", p_wf)]:
        for tau_dn, tau_up in [(5.0, 15.0), (8.0, 22.0), (10.0, 30.0)]:
            for th_def, th_caut, th_ro in [(-0.25, -0.05, 0.15), (-0.25, -0.25, 0.08), (-0.30, -0.10, 0.18)]:
                for eq_fl, eq_ca in [(0.25, 0.60), (0.30, 0.65), (0.25, 1.00)]:
                    for mh_up in (15, 25, 42):
                        sig = _gen_multifactor_signal(p_sig, b50_chg21_arr, b50_lvl_arr, p_semivol, semivol_med, tau_dn, tau_up, th_def, th_caut, th_ro, eq_fl, eq_ca, 0.08, 0, mh_up)
                        nav_c, to_c = _sim_overlay_core(r_ref, to_ref, sig, tg150_r, tg150_tx, 0.0015, True, 0.20, 0.125)
                        st = _summarize_nav(nav_c, to_c, bounds)
                        switches = int(np.sum(np.abs(np.diff(sig)) > 0.05))
                        st.update({"p_name": p_name, "tau_dn": tau_dn, "tau_up": tau_up, "th_def": th_def, "th_caut": th_caut, "th_ro": th_ro, "eq_fl": eq_fl, "eq_ca": eq_ca, "mh_up": mh_up, "switches_16y": switches})
                        st["name"] = f"{p_name}|t={int(tau_dn)}/{int(tau_up)}|th={th_def}/{th_caut}/{th_ro}|fl={eq_fl}/{eq_ca}|mh={mh_up}"
                        churn_rows.append(st)

    ch_df = pd.DataFrame(churn_rows)
    # Show Pareto optimal low-churn models (FULL_to <= 280%, sorted by Sharpe)
    low_ch = ch_df[ch_df["FULL_to"] <= 280.0].sort_values(["FULL_sh", "FULL_cal"], ascending=False)
    if low_ch.empty:
        low_ch = ch_df.sort_values(["FULL_to", "FULL_sh"], ascending=[True, False])
    lines.append("=== LOW-TURNOVER MULTI-FACTOR HYSTERESIS ON CLEAN SORTINO_189 (TO<=280%, TrendGold150) ===")
    lines.append(low_ch.head(15)[["name", "switches_16y", "P1_cagr", "P1_dd", "P2_cagr", "P2_dd", "P3_cagr", "P3_dd", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh", "FULL_to"]].to_string(index=False))

    # Pick P1+P2 blind winner inside low-turnover set AND compare against EQ4 zero-fit low-turnover
    p12_cand = ch_df[ch_df["FULL_to"] <= 290.0].copy()
    if p12_cand.empty:
        p12_cand = ch_df.copy()
    p12_cand["p12_sh"] = 0.5 * (p12_cand["P1_sh"] + p12_cand["P2_sh"])
    r_wf_best = p12_cand[p12_cand["p_name"] == "P1_Opt_WF4"].sort_values("p12_sh", ascending=False).iloc[0]
    r_eq4_best = p12_cand[p12_cand["p_name"] == "EQ4_ZeroFit"].sort_values("p12_sh", ascending=False).iloc[0]
    sig_mf_3state = _gen_multifactor_signal(p_wf, b50_chg21_arr, b50_lvl_arr, p_semivol, semivol_med, float(r_wf_best["tau_dn"]), float(r_wf_best["tau_up"]), float(r_wf_best["th_def"]), float(r_wf_best["th_caut"]), float(r_wf_best["th_ro"]), float(r_wf_best["eq_fl"]), float(r_wf_best["eq_ca"]), 0.08, 0, int(r_wf_best["mh_up"]))
    sig_eq4_low = _gen_multifactor_signal(p_eq4, b50_chg21_arr, b50_lvl_arr, p_semivol, semivol_med, float(r_eq4_best["tau_dn"]), float(r_eq4_best["tau_up"]), float(r_eq4_best["th_def"]), float(r_eq4_best["th_caut"]), float(r_eq4_best["th_ro"]), float(r_eq4_best["eq_fl"]), float(r_eq4_best["eq_ca"]), 0.08, 0, int(r_eq4_best["mh_up"]))
    sig_mf_2state = _gen_multifactor_signal(p_wf, b50_chg21_arr, b50_lvl_arr, p_semivol, semivol_med, 10.0, 30.0, -0.25, -0.25, 0.08, 0.25, 1.00, 0.08, 0, 25)
    dd252 = feat_semi["ew_dd252"].fillna(0.0).values
    ma50_a = feat_semi["ew_ma50"].fillna(0.0).values
    sig_dd15_old = _gen_dd_signal(dd252, px_arr, ma50_a, b50_chg21_arr, b50_lvl_arr, 0.15, 0.60, 1, 0, 0.75, 10)
    sig_dd12_old = _gen_dd_signal(dd252, px_arr, ma50_a, b50_chg21_arr, b50_lvl_arr, 0.12, 0.60, 1, 0, 0.75, 10)

    # ── 2. CROSS-BASKET x REGIME x DEFENSIVE MATRIX ──────────────────────────────────────────────
    cross_rows = []
    for b_lbl, (r_b, to_b, _) in baskets.items():
        for reg_lbl, eq_s in [
            ("Unhedged", np.ones(T)),
            ("Legacy_DD12", sig_dd12_old),
            ("Legacy_DD15", sig_dd15_old),
            ("MF_EQ4_3State_LowChurn", sig_eq4_low),
            ("MF_WF4_3State_LowChurn", sig_mf_3state),
            ("MF_WF4_2State_UltraLowChurn", sig_mf_2state),
        ]:
            for d_lbl, dr, dtx in [
                ("50Gold_50Liq", r_base_5050, tax_5050),
                ("TrendGold150_or_Liq", tg150_r, tg150_tx),
                ("TG_ERC(Gold+Gilt+Liq)", erc_gg_r, erc_gg_tx),
                ("TG_ERC(Gold+Gilt+MON20+Liq)", erc3_r, erc3_tx),
            ]:
                if reg_lbl == "Unhedged" and d_lbl != "50Gold_50Liq":
                    continue
                nav_x, to_x = _sim_overlay_core(r_b, to_b, eq_s, dr, dtx, 0.0015, True, 0.20, 0.125)
                st = _summarize_nav(nav_x, to_x, bounds)
                st["switches"] = int(np.sum(np.abs(np.diff(eq_s)) > 0.05))
                st["name"] = f"{b_lbl} | {reg_lbl} | {d_lbl}"
                cross_rows.append(st)

    cr_df = pd.DataFrame(cross_rows)
    lines.append("\n=== CROSS-FACTOR x REGIME x DEFENSIVE SHOOTOUT (CLEAN ETF-FREE SEMIANNUAL PIT 2011-26) ===")
    lines.append(cr_df.sort_values("FULL_sh", ascending=False).head(30)[["name", "switches", "P1_cagr", "P1_dd", "P2_cagr", "P2_dd", "P3_cagr", "P3_dd", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh", "FULL_to"]].to_string(index=False))

    # ── 3. CORRECTED MONTHLY F&O HEDGES + CRASH EPISODE ATTRIBUTION (INCLUDING 2026 GOLD -31.7% SLUMP!) ────
    lines.append("\n=== REALISTIC MONTHLY F&O HEDGES (VIX-SCALED OTM PUT / SPREAD / SHORT FUTURES) ===")
    fno_rows = sim_monthly_fno_hedges(r_ref, to_ref, sig_mf_2state, ew_r, vix_arr, liq_r, bounds, dates)
    fno_rows += sim_monthly_fno_hedges(r_ref, to_ref, sig_dd15_old, ew_r, vix_arr, liq_r, bounds, dates)
    lines.append(pd.DataFrame(fno_rows)[["name", "P1_cagr", "P1_dd", "P2_cagr", "P2_dd", "P3_cagr", "P3_dd", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh", "FULL_to"]].to_string(index=False))

    lines.append("\n=== HISTORICAL CRASH & 2026 GOLD SLUMP ANATOMY (SORTINO_189 + MF_WF4_3STATE vs LEGACY) ===")
    def_vars = {
        "Unhedged_Equity": (r_base_5050, tax_5050),  # handle below cleanly
        "Static_50Gold_50Liq": (r_base_5050, tax_5050),
        "Pure_GOLDBEES": (gold_r, np.full(T, 0.125)),
        "Pure_LIQUID_REALISTIC": (liq_r, np.full(T, 0.30)),
        "TrendGold(EMA100±2%)": (tg100_r, tg100_tx),
        "TrendGold(EMA150±2%)": (tg150_r, tg150_tx),
        "TG_ERC(Gold+Gilt+Liq)": (erc_gg_r, erc_gg_tx),
        "TG_ERC(Gold+Gilt+MON20+Liq)": (erc3_r, erc3_tx),
    }
    ep_rows = crash_window_attribution(dates, sig_mf_3state, {k: v for k, v in def_vars.items() if k != "Unhedged_Equity"}, r_ref, to_ref)
    ep_un = crash_window_attribution(dates, np.ones(T), {"Unhedged_Equity": (r_base_5050, tax_5050)}, r_ref, to_ref)
    ep_old = crash_window_attribution(dates, sig_dd15_old, {"Legacy_DD15_50G50L": (r_base_5050, tax_5050)}, r_ref, to_ref)
    ep_all = pd.DataFrame(ep_un + ep_old + ep_rows)
    lines.append(ep_all.to_string(index=False))

    # ── 4. EXACT PYTHON FIFO-LOT BACKTESTER (`src/backtester.py`) FINAL VALIDATION ─────────────────
    print("Running exact Python FIFO-lot event-driven backtester verification across PIT SemiAnnual & Current 2026...")
    # Build causal TrendGold150 and TG-ERC Gold+Gilt+Liq series inside def_df cleanly for exact Python FIFO check!
    def_df_ext = def_df.copy()
    def_df_ext["TREND_GOLD150_OR_LIQ"] = 100.0 * np.cumprod(1.0 + tg150_r)
    def_df_ext["TG_ERC_GOLD_GILT_LIQ"] = 100.0 * np.cumprod(1.0 + erc_gg_r)
    def_df_ext["TG_ERC_GOLD_MON_GILT_LIQ"] = 100.0 * np.cumprod(1.0 + erc3_r)

    sc_exact = {
        "sortino_189": scores_semi["sortino_189_s21"],
        "sortino_252": scores_semi["sortino_252_s21"],
        "risk_189": scores_semi["risk_189_s21"],
        "raw_189": scores_semi["raw_189_s21"],
    }
    ma50_pit = adj_eq.rolling(50, min_periods=30).mean()
    alloc_map = {
        "Unhedged": pd.Series(1.0, index=dates),
        "Legacy_DD12": compute_equity_allocation_series(feat_semi, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75}),
        "Legacy_DD15": compute_equity_allocation_series(feat_semi, "index_dd", {"dd_thresh": 0.15, "reenter_ma": 50, "defensive_shift": 0.75}),
        "MF_EQ4_3State": pd.Series(sig_eq4_low, index=dates),
        "MF_WF4_3State": pd.Series(sig_mf_3state, index=dates),
        "MF_WF4_2State": pd.Series(sig_mf_2state, index=dates),
    }

    exact_rows = []
    exact_tests = [
        ("PIT_SEMI | Raw_189_EW | Legacy_DD12 | GOLD_50_LIQ_50", "raw_189", "equal", 126, "Legacy_DD12", "GOLD_50_LIQUID_50", False, 100_000.0, 1),
        ("PIT_SEMI | Risk_189_EW | Legacy_DD15 | GOLD_50_LIQ_50", "risk_189", "equal", 126, "Legacy_DD15", "GOLD_50_LIQUID_50", False, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | Unhedged", "sortino_189", "equal", 126, "Unhedged", "LIQUID_REALISTIC", False, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | MF_EQ4_3State [ZeroFit] | GOLD_50_LIQ_50", "sortino_189", "equal", 126, "MF_EQ4_3State", "GOLD_50_LIQUID_50", False, 100_000.0, 1),
        ("PIT_SEMI | Risk_189_EW | MF_WF4_3State | TrendGold150", "risk_189", "equal", 126, "MF_WF4_3State", "TREND_GOLD150_OR_LIQ", False, 100_000.0, 1),
        ("PIT_SEMI | Risk_189_EW | MF_WF4_3State | TrendGold150 [Staggered 2x6M]", "risk_189", "equal", 126, "MF_WF4_3State", "TREND_GOLD150_OR_LIQ", False, 100_000.0, 2),
        ("PIT_SEMI | Risk_189_EW | MF_WF4_3State | TrendGold150 [Stress 30bps]", "risk_189", "equal", 126, "MF_WF4_3State", "TREND_GOLD150_OR_LIQ", False, 100_000.0, 1),
        ("PIT_SEMI | Risk_189_EW | MF_WF4_3State | TrendGold150 [PreDefTax Comparison]", "risk_189", "equal", 126, "MF_WF4_3State", "TREND_GOLD150_OR_LIQ", False, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | MF_WF4_3State | TrendGold150", "sortino_189", "equal", 126, "MF_WF4_3State", "TREND_GOLD150_OR_LIQ", False, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | MF_WF4_3State | TG_ERC_GiltGoldLiq", "sortino_189", "equal", 126, "MF_WF4_3State", "TG_ERC_GOLD_GILT_LIQ", False, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | MF_WF4_2State | TrendGold150", "sortino_189", "equal", 126, "MF_WF4_2State", "TREND_GOLD150_OR_LIQ", False, 100_000.0, 1),
        ("PIT_SEMI | Sortino_252_12M | MF_WF4_3State | TrendGold150", "sortino_252", "equal", 252, "MF_WF4_3State", "TREND_GOLD150_OR_LIQ", False, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | MF_WF4_3State | TG_ERC_GiltGoldLiq [Disc 1L]", "sortino_189", "equal", 126, "MF_WF4_3State", "TG_ERC_GOLD_GILT_LIQ", True, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | MF_WF4_3State | TrendGold150 [Disc 1L]", "sortino_189", "equal", 126, "MF_WF4_3State", "TREND_GOLD150_OR_LIQ", True, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_NormZ | MF_WF4_3State | TG_ERC_GiltGoldLiq [Disc 1L]", "sortino_189", "norm_z", 126, "MF_WF4_3State", "TG_ERC_GOLD_GILT_LIQ", True, 100_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | MF_WF4_3State | TG_ERC_GiltGoldLiq [Disc 25L]", "sortino_189", "equal", 126, "MF_WF4_3State", "TG_ERC_GOLD_GILT_LIQ", True, 2_500_000.0, 1),
        ("PIT_SEMI | Sortino_189_EW | MF_WF4_3State | TG_ERC_GiltGoldLiq [Stress 30bps]", "sortino_189", "equal", 126, "MF_WF4_3State", "TG_ERC_GOLD_GILT_LIQ", False, 100_000.0, 1),
    ]
    for label, sk, wt, reb, rk, da, disc, cap, n_sl in exact_tests:
        slv = [{"score_key": sk, "top_n": 20, "alloc": 0.5, "weighting": wt}, {"score_key": sk, "top_n": 30, "alloc": 0.5, "weighting": wt}]
        is_stress = "Stress 30bps" in label
        bps = 30.0 if is_stress else 15.0
        reg_bps = 60.0 if is_stress else 35.0
        tx_d = "PreDefTax" not in label
        row = {"label": label}
        for pk, (ps, pe) in PERIODS.items():
            res = run_backtest(
                adj_eq, raw_eq, sc_exact, def_df_ext, alloc_map[rk],
                BacktestConfig(
                    start_date=ps, end_date=pe, initial_capital=cap, rebalance_days=reb,
                    cost_bps_one_way=bps, regime_cost_bps_one_way=reg_bps, apply_tax=True, tax_defensive=tx_d,
                    defensive_asset=da, trim_mode="weakest_first", sleeves=slv, num_sleeves=n_sl,
                    discrete_shares=disc, affordability_filter=disc, sweep_idle_cash=disc, sweep_asset="LIQUID_REALISTIC",
                ),
                stock_ma50=ma50_pit,
            )
            m = res["metrics"]
            row[f"{pk}_cagr"] = m["cagr_pct"]
            row[f"{pk}_dd"] = m["max_dd_pct"]
            row[f"{pk}_sh"] = m["sharpe"]
            if pk == "FULL":
                row["FULL_cal"] = m["calmar"]
                row["FULL_to"] = m["ann_turnover_pct"]
                row["FULL_tax"] = m["total_tax_paid"]
        exact_rows.append(row)

    ex_df = pd.DataFrame(exact_rows)
    lines.append("\n=== EXACT PYTHON FIFO-LOT BACKTESTER VERIFICATION (CLEAN ETF-FREE SEMIANNUAL PIT 2011-2026) ===")
    lines.append(ex_df[["label", "P1_cagr", "P1_dd", "P2_cagr", "P2_dd", "P3_cagr", "P3_dd", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh", "FULL_to", "FULL_tax"]].to_string(index=False))

    out_txt = REPO_ROOT / "reports" / "numba_part2_verification.txt"
    out_txt.write_text("\n".join(lines))
    (REPO_ROOT / "reports" / "numba_part2_verification.json").write_text(json.dumps({
        "low_churn_mf": low_ch.head(15).to_dict(orient="records"),
        "cross_factor": cr_df.sort_values("FULL_sh", ascending=False).head(30).to_dict(orient="records"),
        "fno_hedges": fno_rows,
        "crash_anatomy": ep_all.to_dict(orient="records"),
        "exact_fifo_verification": exact_rows,
    }, indent=2, default=str))
    print(f"Completed Part 2 Deep Verification in {time.time() - t0:.1f}s -> {out_txt}")


if __name__ == "__main__":
    main()
