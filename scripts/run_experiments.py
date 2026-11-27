"""Run comprehensive Midcap Momentum research experiments across all user design questions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtester import BacktestConfig, compute_metrics, run_backtest  # noqa: E402
from src.data_loader import fetch_and_cache_all  # noqa: E402
from src.momentum import compute_score_matrix  # noqa: E402
from src.regime import (  # noqa: E402
    build_defensive_series,
    build_regime_features,
    compute_equity_allocation_series,
)


def _row(label: str, res: dict, extra: dict | None = None) -> dict:
    m = dict(res["metrics"])
    m["strategy"] = label
    if extra:
        m.update(extra)
    return m


def main() -> None:
    data = fetch_and_cache_all(REPO_ROOT, refresh=False)
    adj_df = data["adj"]
    raw_df = data["raw"]
    vol_df = data["vol"]
    bench_df = data["bench"]

    feat = build_regime_features(adj_df, bench_df)
    def_df = build_defensive_series(bench_df)
    stock_ma50 = adj_df.rolling(50, min_periods=30).mean()

    print("Precomputing momentum score matrices...")
    scores = {
        "raw_3m": compute_score_matrix(adj_df, 63, 21, "raw"),
        "raw_6m": compute_score_matrix(adj_df, 126, 21, "raw"),
        "raw_9m": compute_score_matrix(adj_df, 189, 21, "raw"),
        "raw_12m": compute_score_matrix(adj_df, 252, 21, "raw"),
        "raw_6m_noskip": compute_score_matrix(adj_df, 126, 0, "raw"),
        "raw_9m_noskip": compute_score_matrix(adj_df, 189, 0, "raw"),
        "raw_12m_noskip": compute_score_matrix(adj_df, 252, 0, "raw"),
        "risk_3m": compute_score_matrix(adj_df, 63, 21, "risk_adj"),
        "risk_6m": compute_score_matrix(adj_df, 126, 21, "risk_adj"),
        "risk_9m": compute_score_matrix(adj_df, 189, 21, "risk_adj"),
        "risk_12m": compute_score_matrix(adj_df, 252, 21, "risk_adj"),
        "blend_6_12": compute_score_matrix(adj_df, 126, 21, "blend_6_12"),
        "blend_3_6_12": compute_score_matrix(adj_df, 126, 21, "blend_3_6_12"),
        "blend_risk_6_12": compute_score_matrix(adj_df, 126, 21, "blend_risk_6_12"),
    }

    ones = pd.Series(1.0, index=adj_df.index)
    full_start = "2017-12-01"  # leaves 12m + 1m warm-up from 2016-11-28
    end_dt = str(adj_df.index[-1])

    all_results: dict[str, list[dict]] = {}

    # ── EXP 1: PDF Baseline Replication (2020-01-06 -> 2025-12-30) & 10y Benchmark ───────────
    exp1 = []
    for win_name, s_dt, e_dt in [("PDF_2020_2025", "2020-01-06", "2025-12-30"), ("FULL_2017_2026", full_start, end_dt)]:
        ew_sub = feat["ew_index"].loc[(feat.index >= s_dt) & (feat.index <= e_dt)]
        exp1.append({"strategy": f"{win_name} | Universe_EW_150", **compute_metrics(ew_sub)})
        for lbl, sk, n in [
            ("Raw_Top20_9m", "raw_9m", 20),
            ("Raw_Top30_9m", "raw_9m", 30),
            ("Raw_Top50_9m", "raw_9m", 50),
            ("Risk_Top20_9m", "risk_9m", 20),
            ("Risk_Top30_9m", "risk_9m", 30),
            ("Risk_Top50_9m", "risk_9m", 50),
        ]:
            r = run_backtest(
                adj_df, raw_df, scores, def_df, ones,
                BacktestConfig(
                    start_date=s_dt, end_date=e_dt, rebalance_days=189, cost_bps_one_way=0.0,
                    sleeves=[{"score_key": sk, "top_n": n, "alloc": 1.0}],
                ),
                stock_ma50=stock_ma50,
            )
            exp1.append(_row(f"{win_name} | {lbl} (gross)", r))
        # User's 50/50 Raw20 + Raw30
        r50 = run_backtest(
            adj_df, raw_df, scores, def_df, ones,
            BacktestConfig(
                start_date=s_dt, end_date=e_dt, rebalance_days=189, cost_bps_one_way=15.0,
                sleeves=[{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5}],
            ),
            stock_ma50=stock_ma50,
        )
        exp1.append(_row(f"{win_name} | 50k_Raw20+50k_Raw30_9m (15bps)", r50))
    all_results["exp1_baseline"] = exp1

    # ── EXP 2: Rebalance Frequency x Lookback x Friction x Indian Taxes (FULL 9-year window) ─
    exp2 = []
    reb_grid = [(21, "1M"), (42, "2M"), (63, "3M"), (84, "4M"), (126, "6M"), (189, "9M"), (252, "12M")]
    lb_grid = [
        ("raw_6m", "Raw_6M"),
        ("raw_9m", "Raw_9M"),
        ("raw_12m", "Raw_12M"),
        ("risk_6m", "Risk_6M"),
        ("risk_9m", "Risk_9M"),
        ("risk_12m", "Risk_12M"),
        ("blend_6_12", "Blend_6_12"),
        ("blend_risk_6_12", "BlendRisk_6_12"),
        ("raw_9m_noskip", "Raw_9M_NoSkip"),
    ]
    for sk, sk_lbl in lb_grid:
        for reb_d, reb_lbl in reb_grid:
            for apply_tax in (False, True):
                r = run_backtest(
                    adj_df, raw_df, scores, def_df, ones,
                    BacktestConfig(
                        start_date=full_start, end_date=end_dt, rebalance_days=reb_d,
                        cost_bps_one_way=15.0, apply_tax=apply_tax,
                        sleeves=[{"score_key": sk, "top_n": 25, "alloc": 1.0}],
                    ),
                    stock_ma50=stock_ma50,
                )
                exp2.append(_row(
                    f"{sk_lbl}_Top25_Reb{reb_lbl}",
                    r,
                    {"score": sk_lbl, "rebalance": reb_lbl, "tax_mode": "post_tax" if apply_tax else "pre_tax"},
                ))
    # Also test Rank-Buffer hysteresis on top frequencies (Top 25 / Keep Top 40, Top 30 / Keep Top 45)
    for sk, sk_lbl in [("raw_9m", "Raw_9M"), ("risk_9m", "Risk_9M"), ("blend_6_12", "Blend_6_12"), ("blend_risk_6_12", "BlendRisk_6_12")]:
        for reb_d, reb_lbl in [(63, "3M"), (126, "6M"), (189, "9M")]:
            for apply_tax in (False, True):
                r = run_backtest(
                    adj_df, raw_df, scores, def_df, ones,
                    BacktestConfig(
                        start_date=full_start, end_date=end_dt, rebalance_days=reb_d,
                        cost_bps_one_way=15.0, apply_tax=apply_tax,
                        sleeves=[{"score_key": sk, "top_n": 25, "buffer_n": 40, "alloc": 1.0}],
                    ),
                    stock_ma50=stock_ma50,
                )
                exp2.append(_row(
                    f"{sk_lbl}_Top25_Buf40_Reb{reb_lbl}",
                    r,
                    {"score": sk_lbl + "+Buf40", "rebalance": reb_lbl, "tax_mode": "post_tax" if apply_tax else "pre_tax"},
                ))
    all_results["exp2_rebalance_lookback"] = exp2

    # ── EXP 3: Portfolio Splits (50k Raw20 + 50k Raw30 vs alternatives + Discrete ₹1L) ──────
    exp3 = []
    split_configs = [
        ("Raw20 (100%)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 1.0}], 1, 126, False),
        ("Raw30 (100%)", [{"score_key": "raw_9m", "top_n": 30, "alloc": 1.0}], 1, 126, False),
        ("Risk30 (100%)", [{"score_key": "risk_9m", "top_n": 30, "alloc": 1.0}], 1, 126, False),
        ("BlendRisk25_Buf40 (100%)", [{"score_key": "blend_risk_6_12", "top_n": 25, "buffer_n": 40, "alloc": 1.0}], 1, 126, False),
        ("50k Raw20 + 50k Raw30 (Sync 6M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5}], 1, 126, False),
        ("50k Raw20 + 50k Raw30 (Sync 9M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5}], 1, 189, False),
        ("50k Raw20 + 50k Raw30 (Staggered 2x6M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5}], 2, 126, False),
        ("50k Raw20 + 50k Risk30 (Sync 6M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5}, {"score_key": "risk_9m", "top_n": 30, "alloc": 0.5}], 1, 126, False),
        ("50k Blend6-12_20 + 50k BlendRisk_30 (6M)", [{"score_key": "blend_6_12", "top_n": 20, "alloc": 0.5}, {"score_key": "blend_risk_6_12", "top_n": 30, "alloc": 0.5}], 1, 126, False),
        ("50k Raw20 + 50k Raw30 InvVol (6M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5, "weighting": "inv_vol"}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5, "weighting": "inv_vol"}], 1, 126, False),
        ("50k Raw20 + 50k Raw30 NormZ (6M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5, "weighting": "norm_z"}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5, "weighting": "norm_z"}], 1, 126, False),
        ("Discrete ₹1L: Raw20 (6M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 1.0}], 1, 126, True),
        ("Discrete ₹1L: Raw30 (6M)", [{"score_key": "raw_9m", "top_n": 30, "alloc": 1.0}], 1, 126, True),
        ("Discrete ₹1L: 50k Raw20+50k Raw30 (6M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5}], 1, 126, True),
        ("Discrete ₹1L: 50k Raw20+50k Risk30 (6M)", [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5}, {"score_key": "risk_9m", "top_n": 30, "alloc": 0.5}], 1, 126, True),
    ]
    for lbl, slv, n_stagger, reb_d, disc in split_configs:
        for tax_flag in (False, True):
            r = run_backtest(
                adj_df, raw_df, scores, def_df, ones,
                BacktestConfig(
                    start_date=full_start, end_date=end_dt, rebalance_days=reb_d, num_sleeves=n_stagger,
                    cost_bps_one_way=15.0, apply_tax=tax_flag, discrete_shares=disc, sleeves=slv,
                ),
                stock_ma50=stock_ma50,
            )
            exp3.append(_row(lbl, r, {"tax_mode": "post_tax" if tax_flag else "pre_tax"}))
    all_results["exp3_splits_and_discrete"] = exp3

    # ── EXP 4: Market-Fall Regime Signals Across 20-30 Stocks ───────────────────────────────
    base_sleeves = [
        {"score_key": "raw_9m", "top_n": 20, "alloc": 0.5},
        {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5},
    ]
    alt_sleeves = [
        {"score_key": "blend_risk_6_12", "top_n": 25, "buffer_n": 40, "alloc": 1.0},
    ]
    regime_tests = [
        ("Unhedged (None)", "none", {}, "none", {}),
        ("Index < 200DMA (100% Gold)", "index_ma", {"ma_len": 200}, "none", {}),
        ("Index MA200 Band -3%/+1% (100% Gold)", "index_ma_band", {"ma_len": 200, "exit_band": 0.97, "enter_band": 1.01, "defensive_shift": 1.0}, "none", {}),
        ("Index MA200 Band -3%/+1% (50% Gold)", "index_ma_band", {"ma_len": 200, "exit_band": 0.97, "enter_band": 1.01, "defensive_shift": 0.5}, "none", {}),
        ("Index < 100DMA (100% Gold)", "index_ma", {"ma_len": 100}, "none", {}),
        ("Index Dual MA 50<200 (100% Gold)", "dual_ma", {}, "none", {}),
        ("Index DD > 12% (100% Gold)", "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50}, "none", {}),
        ("Index DD > 12% (50% Gold)", "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.5}, "none", {}),
        ("Breadth50 < 30% / > 45% (100% Gold)", "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.30, "enter_above": 0.45}, "none", {}),
        ("Breadth50 < 35% / > 50% (100% Gold)", "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.35, "enter_above": 0.50}, "none", {}),
        ("Breadth200 < 40% / > 55% (100% Gold)", "breadth_hysteresis", {"col": "breadth_200_sma10", "exit_below": 0.40, "enter_above": 0.55}, "none", {}),
        ("VIX > 24 / < 20 (50% Gold)", "vix_filter", {"vix_exit": 24, "vix_enter": 20, "defensive_shift": 0.5}, "none", {}),
        ("Composite 4-Score Tiered (100/60/25/0%)", "tiered_composite", {}, "none", {}),
        ("Composite Confirmed (10d smooth, 21d hold)", "tiered_composite", {"smooth_days": 10, "min_hold_days": 21}, "none", {}),
        ("Portfolio NAV < 50DMA (100% Gold)", "none", {}, "port_ma", {"ma_len": 50}),
        ("Portfolio NAV < 100DMA (100% Gold)", "none", {}, "port_ma", {"ma_len": 100}),
        ("Portfolio DD > 12% (100% Gold)", "none", {}, "port_dd", {"dd_thresh": 0.12, "reenter_ma": 50}),
        ("Portfolio DD > 12% (50% Gold)", "none", {}, "port_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.5}),
        ("Portfolio Holdings Breadth <35% / >50%", "none", {}, "port_breadth", {"exit_below": 0.35, "enter_above": 0.50}),
        ("Per-Stock Trailing Stop 20%", "none", {}, "stock_stop", {"stop_pct": 0.20}),
        ("Per-Stock Trailing Stop 25%", "none", {}, "stock_stop", {"stop_pct": 0.25}),
        ("Breadth50 (<30/>45) + Stock Stop 25%", "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.30, "enter_above": 0.45}, "stock_stop", {"stop_pct": 0.25}),
    ]
    exp4 = []
    for sl_name, sl_cfg in [("50kRaw20+50kRaw30_6M", base_sleeves), ("BlendRisk25_Buf40_6M", alt_sleeves)]:
        for r_lbl, ext_rule, ext_p, int_rule, int_p in regime_tests:
            alloc_s = compute_equity_allocation_series(feat, ext_rule, ext_p)
            r = run_backtest(
                adj_df, raw_df, scores, def_df, alloc_s,
                BacktestConfig(
                    start_date=full_start, end_date=end_dt, rebalance_days=126,
                    cost_bps_one_way=15.0, apply_tax=True,
                    defensive_asset="GOLD_OR_LIQUID",
                    internal_rule=int_rule, internal_params=int_p,
                    sleeves=sl_cfg,
                ),
                stock_ma50=stock_ma50,
            )
            exp4.append(_row(f"{sl_name} | {r_lbl}", r, {"basket": sl_name, "regime": r_lbl}))
    all_results["exp4_regime_signals"] = exp4

    # ── EXP 5: Defensive Asset Shootout, Shift Size, & Trim Mode ────────────────────────────
    exp5 = []
    # 5a) Full 2017-2026 & 2022-2026 (to include SILVERBEES & EBBETF0431) under Breadth50 (<35/>50) and Tiered Composite
    alloc_b50_full = compute_equity_allocation_series(feat, "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.35, "enter_above": 0.50})
    alloc_comp_full = compute_equity_allocation_series(feat, "tiered_composite", {})

    for def_name in ["GOLDBEES", "LIQUID_TBILL", "GOLD_OR_LIQUID", "GOLD_50_LIQUID_50", "NIFTYBEES", "MON100"]:
        for reg_name, alloc_s in [("Breadth50", alloc_b50_full), ("CompositeTiered", alloc_comp_full)]:
            for trim_mode in (["pro_rata", "weakest_first"] if reg_name == "CompositeTiered" else ["pro_rata"]):
                r = run_backtest(
                    adj_df, raw_df, scores, def_df, alloc_s,
                    BacktestConfig(
                        start_date=full_start, end_date=end_dt, rebalance_days=126,
                        cost_bps_one_way=15.0, apply_tax=True,
                        defensive_asset=def_name, trim_mode=trim_mode, sleeves=base_sleeves,
                    ),
                    stock_ma50=stock_ma50,
                )
                exp5.append(_row(f"2017-26 | {reg_name} | {def_name} | {trim_mode}", r, {"window": "2017-26", "defensive": def_name, "regime": reg_name, "trim": trim_mode}))

    # 5b) Partial shift amounts under Breadth50
    for shift_amt in (0.25, 0.50, 0.75, 1.0):
        for trim_mode in ("pro_rata", "weakest_first"):
            if shift_amt == 1.0 and trim_mode == "weakest_first":
                continue
            alloc_s = compute_equity_allocation_series(feat, "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.35, "enter_above": 0.50, "defensive_shift": shift_amt})
            r = run_backtest(
                adj_df, raw_df, scores, def_df, alloc_s,
                BacktestConfig(
                    start_date=full_start, end_date=end_dt, rebalance_days=126,
                    cost_bps_one_way=15.0, apply_tax=True,
                    defensive_asset="GOLD_OR_LIQUID", trim_mode=trim_mode, sleeves=base_sleeves,
                ),
                stock_ma50=stock_ma50,
            )
            exp5.append(_row(f"Shift {int(shift_amt*100)}% | {trim_mode}", r, {"window": "2017-26", "defensive": "GOLD_OR_LIQUID", "shift": shift_amt, "trim": trim_mode}))

    # 5c) Common 2022-03..2026-11 window comparing SILVERBEES, EBBETF0431, LTGILTBEES, GOLDBEES, GOLD_OR_LIQUID
    for def_name in ["GOLDBEES", "SILVERBEES", "EBBETF0431", "LTGILTBEES", "LIQUID_TBILL", "GOLD_OR_LIQUID", "TREND_GOLD_GILT_LIQ"]:
        r = run_backtest(
            adj_df, raw_df, scores, def_df, alloc_b50_full,
            BacktestConfig(
                start_date="2022-03-01", end_date=end_dt, rebalance_days=126,
                cost_bps_one_way=15.0, apply_tax=True,
                defensive_asset=def_name, sleeves=base_sleeves,
            ),
            stock_ma50=stock_ma50,
        )
        exp5.append(_row(f"2022-26 | Breadth50 | {def_name}", r, {"window": "2022-26", "defensive": def_name}))
    all_results["exp5_defensives_and_sizing"] = exp5

    # ── EXP 6: Walk-Forward Regimes + 2x Stress Cost + Start-Date Sensitivity ───────────────
    exp6 = []
    wf_windows = [
        ("WF1_2018_2020_Bear_Covid", "2017-12-01", "2020-12-31"),
        ("WF2_2021_2023_Bull", "2021-01-01", "2023-12-31"),
        ("WF3_2024_2026_Peak_Corr", "2024-01-01", end_dt),
    ]
    candidates = [
        ("Baseline_50kRaw20+30_9M_Unhedged", base_sleeves, 189, "none", {}, "LIQUID_TBILL", 15.0),
        ("50kRaw20+30_6M_Breadth50_GoldOrLiq", base_sleeves, 126, "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.35, "enter_above": 0.50}, "GOLD_OR_LIQUID", 15.0),
        ("50kRaw20+30_6M_Composite_GoldOrLiq", base_sleeves, 126, "tiered_composite", {}, "GOLD_OR_LIQUID", 15.0),
        ("BlendRisk25_Buf40_6M_Breadth50_GoldOrLiq", alt_sleeves, 126, "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.35, "enter_above": 0.50}, "GOLD_OR_LIQUID", 15.0),
        ("BlendRisk25_Buf40_6M_Composite_GoldOrLiq", alt_sleeves, 126, "tiered_composite", {}, "GOLD_OR_LIQUID", 15.0),
        ("STRESS_2xCost_30bps_BlendRisk25_Comp", alt_sleeves, 126, "tiered_composite", {}, "GOLD_OR_LIQUID", 30.0),
    ]
    for cname, slv, reb_d, erule, ep, d_asset, bps in candidates:
        alloc_s = compute_equity_allocation_series(feat, erule, ep)
        for wname, s_d, e_d in wf_windows:
            r = run_backtest(
                adj_df, raw_df, scores, def_df, alloc_s,
                BacktestConfig(
                    start_date=s_d, end_date=e_d, rebalance_days=reb_d,
                    cost_bps_one_way=bps, apply_tax=True, defensive_asset=d_asset, sleeves=slv,
                ),
                stock_ma50=stock_ma50,
            )
            exp6.append(_row(f"{wname} | {cname}", r, {"window": wname, "candidate": cname}))
    all_results["exp6_walk_forward_stress"] = exp6

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Saved full experiment matrix -> {out_path}")

    # Print crisp executive summary tables
    for sec, rows in all_results.items():
        print(f"\n================ {sec.upper()} ================")
        df_s = pd.DataFrame(rows)
        cols = [c for c in ["strategy", "cagr_pct", "max_dd_pct", "calmar", "sharpe", "sortino", "ann_turnover_pct", "avg_eq_pct", "avg_holdings", "win_years_pct"] if c in df_s.columns]
        print(df_s[cols].to_string(index=False))


if __name__ == "__main__":
    main()
