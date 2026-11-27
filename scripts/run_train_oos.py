"""Strict Two-Period Train vs Out-of-Sample (OOS) Validation Split to audit overfitting cleanly without peeking."""

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
from src.regime import build_defensive_series, build_ew_index, build_regime_features, compute_equity_allocation_series  # noqa: E402


def _eval_grid(adj_df, raw_df, scores, def_df, feat, stock_ma50, start_dt: str, end_dt: str) -> list[dict]:
    regimes = {
        "Unhedged": (compute_equity_allocation_series(feat, "none", {}), "GOLD_50_LIQUID_50", "weakest_first"),
        "DD10_Sh50_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.10, "reenter_ma": 50, "defensive_shift": 0.50}), "GOLD_50_LIQUID_50", "weakest_first"),
        "DD12_Sh50_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.50}), "GOLD_50_LIQUID_50", "weakest_first"),
        "DD12_Sh75_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75}), "GOLD_50_LIQUID_50", "weakest_first"),
        "DD12_Sh75_GOL": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75}), "GOLD_OR_LIQUID", "weakest_first"),
        "DD12_Sh75_Gold": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75}), "GOLDBEES", "weakest_first"),
        "DD15_Sh75_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.15, "reenter_ma": 50, "defensive_shift": 0.75}), "GOLD_50_LIQUID_50", "weakest_first"),
        "MA200Band_Sh60_GL50": (compute_equity_allocation_series(feat, "index_ma_band", {"ma_len": 200, "exit_band": 0.03, "enter_band": 0.01, "defensive_shift": 0.60}), "GOLD_50_LIQUID_50", "weakest_first"),
        "Breadth35_Sh50_GOL": (compute_equity_allocation_series(feat, "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.35, "enter_above": 0.50, "defensive_shift": 0.50}), "GOLD_OR_LIQUID", "weakest_first"),
        "VIX24_Sh50_GOL": (compute_equity_allocation_series(feat, "vix_filter", {"exit_above": 24.0, "enter_below": 20.0, "defensive_shift": 0.50}), "GOLD_OR_LIQUID", "weakest_first"),
    }

    baskets = [
        ("Raw_6M_5050_EQ", "raw_6m", "equal"),
        ("Raw_9M_5050_EQ", "raw_9m", "equal"),
        ("Raw_12M_5050_EQ", "raw_12m", "equal"),
        ("Raw_9M_5050_NZ", "raw_9m", "norm_z"),
        ("Risk_9M_5050_EQ", "risk_9m", "equal"),
    ]
    rebs = [("3M", 63), ("6M", 126), ("12M", 252)]

    rows = []
    for b_name, skey, wt in baskets:
        sleeves = [{"score_key": skey, "top_n": 20, "alloc": 0.5, "weighting": wt}, {"score_key": skey, "top_n": 30, "alloc": 0.5, "weighting": wt}]
        for r_name, r_days in rebs:
            # Test key regimes on all 5x3 grid = 150 total parameter combinations!
            for reg_name, (alloc, def_a, trim) in regimes.items():
                res = run_backtest(
                    adj_df, raw_df, scores, def_df, alloc,
                    BacktestConfig(start_date=start_dt, end_date=end_dt, rebalance_days=r_days, cost_bps_one_way=15.0, apply_tax=True, defensive_asset=def_a, trim_mode=trim, sleeves=sleeves),
                    stock_ma50=stock_ma50,
                )
                m = res["metrics"]
                rows.append({
                    "config_id": f"{b_name} | {r_name} | {reg_name}",
                    "basket": b_name,
                    "reb": r_name,
                    "regime": reg_name,
                    "cagr": m["cagr_pct"],
                    "max_dd": m["max_dd_pct"],
                    "calmar": m["calmar"],
                    "sharpe": m["sharpe"],
                    "sortino": m["sortino"],
                    "turnover": m["ann_turnover_pct"],
                })
    return rows


def main() -> None:
    data = fetch_and_cache_all(REPO_ROOT, refresh=False)
    adj_df, raw_df, bench_df = data["adj"], data["raw"], data["bench"]
    feat = build_regime_features(adj_df, bench_df)
    def_df = build_defensive_series(bench_df)
    stock_ma50 = adj_df.rolling(50, min_periods=30).mean()

    scores = {
        "raw_6m": compute_score_matrix(adj_df, 126, 21, "raw"),
        "raw_9m": compute_score_matrix(adj_df, 189, 21, "raw"),
        "raw_12m": compute_score_matrix(adj_df, 252, 21, "raw"),
        "risk_9m": compute_score_matrix(adj_df, 189, 21, "risk_adj"),
    }

    # Strict chronological split (~4.5y Train Optimization -> ~4.4y Blind OOS Overfitting Check)
    is_start, is_end = "2018-01-02", "2022-06-30"
    oos_start, oos_end = "2022-07-01", "2026-11-26"

    print(f"Running 150-strategy Optimization Grid on IN-SAMPLE ({is_start}..{is_end})...")
    is_rows = _eval_grid(adj_df, raw_df, scores, def_df, feat, stock_ma50, is_start, is_end)
    df_is = pd.DataFrame(is_rows).add_prefix("is_").rename(columns={"is_config_id": "config_id", "is_basket": "basket", "is_reb": "reb", "is_regime": "regime"})

    print(f"Running exact same 150-strategy Grid BLINDLY on OUT-OF-SAMPLE ({oos_start}..{oos_end})...")
    oos_rows = _eval_grid(adj_df, raw_df, scores, def_df, feat, stock_ma50, oos_start, oos_end)
    df_oos = pd.DataFrame(oos_rows)[["config_id", "cagr", "max_dd", "calmar", "sharpe", "sortino", "turnover"]].add_prefix("oos_").rename(columns={"oos_config_id": "config_id"})

    merged = df_is.merge(df_oos, on="config_id")
    merged["sharpe_ratio_oos_is"] = (merged["oos_sharpe"] / merged["is_sharpe"].replace(0, np.nan)).round(2)
    merged["is_sharpe_rank"] = merged["is_sharpe"].rank(ascending=False).astype(int)
    merged["is_calmar_rank"] = merged["is_calmar"].rank(ascending=False).astype(int)
    merged["oos_sharpe_rank"] = merged["oos_sharpe"].rank(ascending=False).astype(int)
    merged["oos_calmar_rank"] = merged["oos_calmar"].rank(ascending=False).astype(int)

    rho_sharpe = merged["is_sharpe"].rank().corr(merged["oos_sharpe"].rank())
    rho_calmar = merged["is_calmar"].rank().corr(merged["oos_calmar"].rank())
    rho_cagr = merged["is_cagr"].rank().corr(merged["oos_cagr"].rank())

    by_regime = merged.groupby("regime")[["is_cagr", "is_max_dd", "is_calmar", "is_sharpe", "oos_cagr", "oos_max_dd", "oos_calmar", "oos_sharpe"]].mean().round(2).sort_values("is_calmar", ascending=False)
    by_basket = merged.groupby("basket")[["is_cagr", "is_max_dd", "is_calmar", "is_sharpe", "oos_cagr", "oos_max_dd", "oos_calmar", "oos_sharpe"]].mean().round(2).sort_values("is_calmar", ascending=False)
    by_reb = merged.groupby("reb")[["is_cagr", "is_max_dd", "is_calmar", "is_sharpe", "oos_cagr", "oos_max_dd", "oos_calmar", "oos_sharpe"]].mean().round(2).sort_values("is_calmar", ascending=False)

    # Sort purely by IN-SAMPLE Calmar (top picks selected WITHOUT seeing OOS) vs Top-10 IS Sharpe!
    top_is_calmar = merged.sort_values("is_calmar", ascending=False).head(15)
    top_is_sharpe = merged.sort_values("is_sharpe", ascending=False).head(15)
    f2_row = merged[merged["config_id"] == "Raw_9M_5050_EQ | 6M | DD12_Sh75_GL50"]
    unhedged_row = merged[merged["basket"].eq("Raw_9M_5050_EQ") & merged["reb"].eq("6M") & merged["regime"].eq("Unhedged")]

    ew_is = compute_metrics(build_ew_index(adj_df).loc[is_start:is_end])
    ew_oos = compute_metrics(build_ew_index(adj_df).loc[oos_start:oos_end])

    summary_lines = [
        "=== STRICT TRAIN (2018-01-02..2022-06-30) VS BLIND OUT-OF-SAMPLE (2022-07-01..2026-11-26) ===",
        f"Grid size: {len(merged)} parameter configs (5 baskets x 3 rebal x 10 regimes)",
        f"Spearman Rank Correlation (IS -> OOS across all 150 configs): Sharpe rho={rho_sharpe:.3f} | Calmar rho={rho_calmar:.3f} | CAGR rho={rho_cagr:.3f}",
        f"Top-10 IS Calmar Mean OOS Sharpe={merged.sort_values('is_calmar', ascending=False).head(10)['oos_sharpe'].mean():.2f} (IS Mean={merged.sort_values('is_calmar', ascending=False).head(10)['is_sharpe'].mean():.2f})",
        f"Bottom-10 IS Calmar Mean OOS Sharpe={merged.sort_values('is_calmar', ascending=True).head(10)['oos_sharpe'].mean():.2f}",
        f"Benchmark Midcap150_EW_Universe: IS CAGR={ew_is['cagr_pct']}% DD={ew_is['max_dd_pct']}% Sh={ew_is['sharpe']} | OOS CAGR={ew_oos['cagr_pct']}% DD={ew_oos['max_dd_pct']}% Sh={ew_oos['sharpe']}",
        "",
        "--- REGIME FAMILY AVERAGES (SELECTED BY IS CALMAR -> BLIND OOS CHECK) ---",
        by_regime.to_string(),
        "",
        "--- BASKET / FACTOR AVERAGES (IS -> OOS) ---",
        by_basket.to_string(),
        "",
        "--- REBALANCE CADENCE AVERAGES (IS -> OOS) ---",
        by_reb.to_string(),
        "",
        "--- TOP 15 SELECTED BLINDLY BY IN-SAMPLE CALMAR (AND THEIR OOS VALIDATION) ---",
        top_is_calmar[["config_id", "is_cagr", "is_max_dd", "is_calmar", "is_sharpe", "oos_cagr", "oos_max_dd", "oos_calmar", "oos_sharpe", "oos_calmar_rank"]].to_string(index=False),
        "",
        "--- TOP 15 SELECTED BLINDLY BY IN-SAMPLE SHARPE (AND THEIR OOS VALIDATION) ---",
        top_is_sharpe[["config_id", "is_cagr", "is_max_dd", "is_calmar", "is_sharpe", "oos_cagr", "oos_max_dd", "oos_calmar", "oos_sharpe", "oos_sharpe_rank"]].to_string(index=False),
        "",
        "--- TRACKED LOCKED F2 & UNHEDGED BASELINE ---",
        pd.concat([f2_row, unhedged_row])[["config_id", "is_cagr", "is_max_dd", "is_calmar", "is_sharpe", "is_calmar_rank", "oos_cagr", "oos_max_dd", "oos_calmar", "oos_sharpe", "oos_calmar_rank"]].to_string(index=False),
    ]
    report = "\n".join(summary_lines)
    (REPO_ROOT / "reports" / "train_oos_validation.txt").write_text(report)
    (REPO_ROOT / "reports" / "train_oos_validation.json").write_text(merged.to_json(orient="records", indent=2))
    print(report)


if __name__ == "__main__":
    main()
