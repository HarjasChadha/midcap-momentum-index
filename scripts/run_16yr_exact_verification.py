"""Exact 16-Year Zero-Survivorship PIT Verification (`src/backtester.py` FIFO engine).

Computes full-period (2011-01-03..2026-11-26) + sub-period P1/P2/P3 exact FIFO performance across:
- Primary R3 / S3 / R2 / F2 presets (tax_defensive=True & 35 bps risk-off regime slippage defaults)
- Staggered 2-Tranche 6M sleeve (num_sleeves=2, 50% every 3M)
- 30 bps (2x friction) cost stress test (60 bps regime shift impact)
- Pre-defensive-tax comparison column (tax_defensive=False)
- Delisting/halted stock write-down stress tests (delist_haircut_pct = 0%, 15%, 25%)
- Zero-fit ablation baselines (EQ4 = [0.25]*4 equal weights + 50/50 Gold/Liquid & 100% Liquid)
- Official NSE benchmarks (NIFTY_MIDCAP_100 full 16Y, Stitched M100->M150, NIFTY_MIDCAP_150 2019-26, NIFTYBEES, PIT EW)
- 6-phase rebalance sensitivity (0..105d offsets) for single-tranche vs staggered 2-tranche
- Production `src.momentum` vs Numba research score parity check (< 1e-9)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.numba_research_engine import PERIODS, compute_extended_scores  # noqa: E402
from scripts.run_numba_sweeps import build_clean_equity_and_pit_masks  # noqa: E402
from src.backtester import BacktestConfig, compute_metrics, run_backtest  # noqa: E402
from src.momentum import compute_score_matrix  # noqa: E402
from src.nse_archive_loader import build_2010_2026_nse_dataset  # noqa: E402
from src.prowess_loader import discover_cmie_prowess_export  # noqa: E402
from src.regime import (  # noqa: E402
    build_defensive_series,
    build_regime_features,
    compute_equity_allocation_series,
    compute_multifactor_state,
)


def _compute_benchmark_row(lbl: str, universe: str, series: pd.Series, periods: dict[str, tuple[str, str]]) -> dict[str, Any]:
    row: dict[str, Any] = {"lbl": lbl, "universe": universe}
    s_clean = series.dropna()
    for pk, (ps, pe) in periods.items():
        sub = s_clean.loc[ps:pe]
        if len(sub) < 30:
            row[f"{pk}_cagr"] = None
            row[f"{pk}_dd"] = None
            row[f"{pk}_sh"] = None
            row[f"{pk}_cal"] = None
            continue
        m = compute_metrics(sub)
        row[f"{pk}_cagr"] = m["cagr_pct"]
        row[f"{pk}_dd"] = m["max_dd_pct"]
        row[f"{pk}_sh"] = m["sharpe"]
        if pk == "FULL":
            row["FULL_cal"] = m["calmar"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact 16-year FIFO verification via src/backtester.py")
    parser.add_argument("--prowess-file", default=None, help="Optional CMIE Prowessdx export path")
    args = parser.parse_args()

    t0 = time.time()
    d = build_2010_2026_nse_dataset(REPO_ROOT, 2010, 2026, refresh=False)
    prowess_used = Path(args.prowess_file) if args.prowess_file else discover_cmie_prowess_export(REPO_ROOT)
    adj_eq, raw_eq, _, pit_semi, bench_df = build_clean_equity_and_pit_masks(d, prowess_file=prowess_used)
    dates = adj_eq.index

    # Strictly causal lagged semiannual universe mask
    pit_lag = pit_semi.shift(1, fill_value=False)
    def_df = build_defensive_series(bench_df)
    feat_semi = build_regime_features(adj_eq, bench_df, mask=pit_semi)
    ma50_pit = adj_eq.rolling(50, min_periods=30).mean()

    # Production scores masked strictly by lagged PIT constituents
    sc_prod = {
        "raw_189": compute_score_matrix(adj_eq, 189, 21, "raw").where(pit_lag),
        "risk_189": compute_score_matrix(adj_eq, 189, 21, "risk_adj").where(pit_lag),
        "sortino_189": compute_score_matrix(adj_eq, 189, 21, "sortino").where(pit_lag),
        "raw_252": compute_score_matrix(adj_eq, 252, 21, "raw").where(pit_lag),
        "sortino_252": compute_score_matrix(adj_eq, 252, 21, "sortino").where(pit_lag),
    }

    # Verify < 1e-9 parity between production `src.momentum` and research `compute_extended_scores`
    sc_res = compute_extended_scores(adj_eq, pit_semi)
    parity_diffs = {}
    for k_prod, k_res in [("raw_189", "raw_189_s21"), ("risk_189", "risk_189_s21"), ("sortino_189", "sortino_189_s21")]:
        a = sc_prod[k_prod].values
        b = sc_res[k_res].values
        valid = np.isfinite(a) & np.isfinite(b)
        max_diff = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0
        same_nan = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
        parity_diffs[k_prod] = {"max_abs_diff": max_diff, "exact_nan_mask_match": same_nan}
        if max_diff >= 1e-9 or not same_nan:
            raise AssertionError(f"Prod vs Research score mismatch for {k_prod}: diff={max_diff}, same_nan={same_nan}")

    reg_mf4, _ = compute_multifactor_state(feat_semi, bench_df)
    reg_eq4, _ = compute_multifactor_state(feat_semi, bench_df, weights=(0.25, 0.25, 0.25, 0.25))
    reg_dd15 = compute_equity_allocation_series(feat_semi, "index_dd", {"dd_thresh": 0.15, "reenter_ma": 50, "defensive_shift": 0.75})
    reg_dd75 = compute_equity_allocation_series(feat_semi, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75})
    reg_none = pd.Series(1.0, index=dates)

    def _slv(sk: str, wt: str) -> list[dict[str, Any]]:
        return [{"score_key": sk, "top_n": 20, "alloc": 0.5, "weighting": wt}, {"score_key": sk, "top_n": 30, "alloc": 0.5, "weighting": wt}]

    # Specifications:
    # (lbl, sk, wt, alloc_series, def_asset, disc_1l, cfg_overrides)
    specs = [
        ("R3_src_Frac", "risk_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", False, {}),
        ("R3_Staggered_2x6M", "risk_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", False, {"num_sleeves": 2}),
        ("R3_Stress_30bps_2xCost", "risk_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", False, {"cost_bps_one_way": 30.0, "regime_cost_bps_one_way": 60.0}),
        ("R3_PreDefTax_Comparison", "risk_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", False, {"tax_defensive": False}),
        ("R3_Delist_Haircut_15pct", "risk_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", False, {"delist_haircut_pct": 0.15}),
        ("R3_Delist_Haircut_25pct", "risk_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", False, {"delist_haircut_pct": 0.25}),
        ("R3_src_Disc1L", "risk_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", True, {}),
        ("S3_src_Frac", "sortino_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", False, {}),
        ("S3_src_Disc1L", "sortino_189", "equal", reg_mf4, "TREND_GOLD150_OR_LIQ", True, {}),
        ("R3_NormZ_Frac", "risk_189", "norm_z", reg_mf4, "TREND_GOLD150_OR_LIQ", False, {}),
        ("R2_src_Frac", "risk_189", "equal", reg_dd15, "GOLD_50_LIQUID_50", False, {}),
        ("R2_src_Disc1L", "risk_189", "equal", reg_dd15, "GOLD_50_LIQUID_50", True, {}),
        ("F2_src_Frac", "raw_189", "equal", reg_dd75, "GOLD_50_LIQUID_50", False, {}),
        ("ZeroFit_EQ4_GoldLiq50", "risk_189", "equal", reg_eq4, "GOLD_50_LIQUID_50", False, {}),
        ("ZeroFit_EQ4_Liquid", "risk_189", "equal", reg_eq4, "LIQUID_REALISTIC", False, {}),
        ("Unhedged_Risk9M", "risk_189", "equal", reg_none, "LIQUID_REALISTIC", False, {}),
        ("Unhedged_Sortino9M", "sortino_189", "equal", reg_none, "LIQUID_REALISTIC", False, {}),
    ]

    rows: list[dict[str, Any]] = []
    for lbl, sk, wt, alloc_s, da, disc, ov in specs:
        rec: dict[str, Any] = {"lbl": lbl}
        for pk, (ps, pe) in PERIODS.items():
            res = run_backtest(
                adj_eq,
                raw_eq,
                sc_prod,
                def_df,
                alloc_s,
                BacktestConfig(
                    start_date=ps,
                    end_date=pe,
                    initial_capital=100_000.0,
                    rebalance_days=126,
                    cost_bps_one_way=float(ov.get("cost_bps_one_way", 15.0)),
                    regime_cost_bps_one_way=float(ov.get("regime_cost_bps_one_way", 35.0)),
                    apply_tax=True,
                    tax_defensive=bool(ov.get("tax_defensive", True)),
                    delist_haircut_pct=float(ov.get("delist_haircut_pct", 0.0)),
                    defensive_asset=da,
                    trim_mode="weakest_first",
                    sleeves=_slv(sk, wt),
                    num_sleeves=int(ov.get("num_sleeves", 1)),
                    discrete_shares=disc,
                    affordability_filter=disc,
                    sweep_idle_cash=disc,
                    sweep_asset="LIQUID_REALISTIC",
                ),
                stock_ma50=ma50_pit,
            )
            m = res["metrics"]
            rec[f"{pk}_cagr"] = m["cagr_pct"]
            rec[f"{pk}_dd"] = m["max_dd_pct"]
            rec[f"{pk}_sh"] = m["sharpe"]
            if pk == "FULL":
                rec["FULL_cal"] = m["calmar"]
                rec["FULL_to"] = m["ann_turnover_pct"]
                rec["FULL_eq"] = m["avg_eq_pct"]
                rec["FULL_tax"] = m["total_tax_paid"]
                rec["FULL_def_tax"] = m["defensive_tax_paid"]
                rec["FULL_fees"] = m["total_fees_paid"]
                rec["delisted_write_downs"] = m["delisted_write_downs"]
        rows.append(rec)

    # Benchmarks row list
    pit_label = f"CMIE Prowess PIT ({adj_eq.shape[1]} Eqs)" if prowess_used else f"Clean ETF-Free PIT ({adj_eq.shape[1]} Eqs)"
    benchmarks: list[dict[str, Any]] = [
        _compute_benchmark_row("PIT_EW_Benchmark", pit_label, feat_semi["ew_index"], PERIODS),
    ]
    if "NIFTY_MIDCAP_100" in def_df.columns:
        benchmarks.append(_compute_benchmark_row("NIFTY_MIDCAP_100", "Official NSE Index (16Y)", def_df["NIFTY_MIDCAP_100"], PERIODS))
    if "NIFTY_MIDCAP_100" in def_df.columns and "NIFTY_MIDCAP_150" in def_df.columns:
        m100_r = def_df["NIFTY_MIDCAP_100"].pct_change(fill_method=None).fillna(0.0)
        m150_r = def_df["NIFTY_MIDCAP_150"].pct_change(fill_method=None)
        st_r = np.where(m150_r.notna().values, m150_r.fillna(0.0).values, m100_r.values)
        st_s = pd.Series(100.0 * np.cumprod(1.0 + st_r), index=dates)
        benchmarks.append(_compute_benchmark_row("NIFTY_MIDCAP_100_150_STITCHED", "Official Stitched M100->M150 (16Y)", st_s, PERIODS))
    if "NIFTY_MIDCAP_150" in def_df.columns:
        m150_clean = def_df["NIFTY_MIDCAP_150"].dropna()
        if not m150_clean.empty:
            p3_only = compute_metrics(m150_clean.loc[PERIODS["P3"][0] : PERIODS["P3"][1]])
            since_2019 = compute_metrics(m150_clean)
            benchmarks.append(
                {
                    "lbl": "NIFTY_MIDCAP_150_2019_26",
                    "universe": f"Official NSE Midcap 150 ({m150_clean.index[0]}..{m150_clean.index[-1]})",
                    "P1_cagr": None,
                    "P2_cagr": None,
                    "P3_cagr": p3_only["cagr_pct"],
                    "FULL_cagr": since_2019["cagr_pct"],
                    "FULL_dd": since_2019["max_dd_pct"],
                    "FULL_sh": since_2019["sharpe"],
                    "FULL_cal": since_2019["calmar"],
                }
            )
    if "NIFTYBEES" in def_df.columns:
        benchmarks.append(_compute_benchmark_row("NIFTYBEES", "Official Nifty 50 ETF (16Y)", def_df["NIFTYBEES"], PERIODS))

    # 6-phase sensitivity across offsets [0, 21, 42, 63, 84, 105] + Staggered 2x6M
    phase_spread: dict[str, Any] = {}
    phase_specs = [
        ("R3", "risk_189", reg_mf4, "TREND_GOLD150_OR_LIQ", 1),
        ("R3_Staggered_2x6M", "risk_189", reg_mf4, "TREND_GOLD150_OR_LIQ", 2),
        ("S3", "sortino_189", reg_mf4, "TREND_GOLD150_OR_LIQ", 1),
        ("R2", "risk_189", reg_dd15, "GOLD_50_LIQUID_50", 1),
    ]
    ps_full, pe_full = PERIODS["FULL"]
    for pname, sk, alloc_s, da, n_sl in phase_specs:
        cagrs = []
        for off in (0, 21, 42, 63, 84, 105):
            r_ph = run_backtest(
                adj_eq,
                raw_eq,
                sc_prod,
                def_df,
                alloc_s,
                BacktestConfig(
                    start_date=ps_full,
                    end_date=pe_full,
                    initial_capital=100_000.0,
                    rebalance_days=126,
                    phase_offset_days=off,
                    cost_bps_one_way=15.0,
                    regime_cost_bps_one_way=35.0,
                    apply_tax=True,
                    tax_defensive=True,
                    defensive_asset=da,
                    trim_mode="weakest_first",
                    sleeves=_slv(sk, "equal"),
                    num_sleeves=n_sl,
                    sweep_asset="LIQUID_REALISTIC",
                ),
                stock_ma50=ma50_pit,
            )
            cagrs.append(r_ph["metrics"]["cagr_pct"])
        phase_spread[pname] = {
            "cagrs": cagrs,
            "mean": round(float(np.mean(cagrs)), 2),
            "min": round(float(np.min(cagrs)), 2),
            "max": round(float(np.max(cagrs)), 2),
            "spread": round(float(max(cagrs) - min(cagrs)), 2),
        }

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "pit_universe": pit_label,
        "p3_disclosure": "P3 (2021-07-01..2026-11-26) is Post-Selection / Partially In-Sample for TrendGold150",
        "score_parity_vs_research": parity_diffs,
        "defaults": {
            "tax_defensive": True,
            "cost_bps_one_way": 15.0,
            "regime_cost_bps_one_way": 35.0,
            "delist_haircut_pct": 0.0,
            "cash_benchmark": "LIQUID_REALISTIC",
            "trading_days_per_year": 252,
        },
        "benchmarks": benchmarks,
        "rows": rows,
        "phase_spread": phase_spread,
    }
    out_path = REPO_ROOT / "reports" / "src_exact_16yr_verification.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path} in {payload['elapsed_sec']}s ({len(rows)} rows, {len(benchmarks)} benchmarks)")


if __name__ == "__main__":
    main()
