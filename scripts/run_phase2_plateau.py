"""Phase 2: Parameter plateau, start-date offset, discrete ₹1L + regime, and walk-forward robustness sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtester import BacktestConfig, run_backtest  # noqa: E402
from src.data_loader import fetch_and_cache_all  # noqa: E402
from src.momentum import compute_score_matrix  # noqa: E402
from src.regime import (  # noqa: E402
    build_defensive_series,
    build_regime_features,
    compute_equity_allocation_series,
)


def main() -> None:
    data = fetch_and_cache_all(REPO_ROOT, refresh=False)
    adj_df, raw_df, bench_df = data["adj"], data["raw"], data["bench"]
    feat = build_regime_features(adj_df, bench_df)
    def_df = build_defensive_series(bench_df)
    stock_ma50 = adj_df.rolling(50, min_periods=30).mean()

    scores = {
        "raw_9m": compute_score_matrix(adj_df, 189, 21, "raw"),
        "raw_12m": compute_score_matrix(adj_df, 252, 21, "raw"),
        "risk_9m": compute_score_matrix(adj_df, 189, 21, "risk_adj"),
        "blend_6_12": compute_score_matrix(adj_df, 126, 21, "blend_6_12"),
    }

    full_start = "2017-12-01"
    end_dt = str(adj_df.index[-1])

    sleeves_5050 = [
        {"score_key": "raw_9m", "top_n": 20, "alloc": 0.5},
        {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5},
    ]
    sleeves_5050_12m = [
        {"score_key": "raw_12m", "top_n": 20, "alloc": 0.5},
        {"score_key": "raw_12m", "top_n": 30, "alloc": 0.5},
    ]
    sleeves_5050_normz = [
        {"score_key": "raw_9m", "top_n": 20, "alloc": 0.5, "weighting": "norm_z"},
        {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5, "weighting": "norm_z"},
    ]
    sleeves_raw25_buf = [
        {"score_key": "raw_12m", "top_n": 25, "buffer_n": 40, "alloc": 1.0},
    ]

    rows = []

    # 1. Plateau over Index DD threshold (10%, 12%, 14%, 15%) x shift (50%, 75%, 100%) x defensive
    for dd_th in (0.10, 0.12, 0.14, 0.15):
        for sh in (0.50, 0.75, 1.0):
            for d_asset in ("GOLD_50_LIQUID_50", "GOLD_OR_LIQUID", "GOLDBEES"):
                alloc = compute_equity_allocation_series(
                    feat, "index_dd", {"dd_thresh": dd_th, "reenter_ma": 50, "defensive_shift": sh}
                )
                for trim in (["weakest_first"] if sh < 1.0 else ["pro_rata"]):
                    r = run_backtest(
                        adj_df, raw_df, scores, def_df, alloc,
                        BacktestConfig(
                            start_date=full_start, end_date=end_dt, rebalance_days=126,
                            cost_bps_one_way=15.0, apply_tax=True, defensive_asset=d_asset,
                            trim_mode=trim, sleeves=sleeves_5050,
                        ),
                        stock_ma50=stock_ma50,
                    )
                    rows.append({"group": "DD_PLATEAU", "label": f"DD>{int(dd_th*100)}%|Shift{int(sh*100)}%|{d_asset}|{trim}", **r["metrics"]})

    # 2. Compare Rebalance (6M, 9M, 12M) x Basket (50k Raw20+30 vs 12M vs NormZ vs Raw25+Buf40) under top protection (Index DD>12% Shift 75% & Shift 50% & MA200 Band)
    protections = [
        ("Unhedged", "none", {}, "LIQUID_TBILL", "pro_rata"),
        ("DD12_Shift50_WeakFirst_GoldLiq50", "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.50}, "GOLD_50_LIQUID_50", "weakest_first"),
        ("DD12_Shift75_WeakFirst_GoldLiq50", "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75}, "GOLD_50_LIQUID_50", "weakest_first"),
        ("DD12_Shift100_GoldOrLiq", "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 1.00}, "GOLD_OR_LIQUID", "pro_rata"),
        ("MA200Band_Shift60_WeakFirst_GoldLiq50", "index_ma_band", {"ma_len": 200, "exit_band": 0.97, "enter_band": 1.01, "defensive_shift": 0.60}, "GOLD_50_LIQUID_50", "weakest_first"),
    ]
    baskets = [
        ("50kRaw20+30_9mLB_Reb6M", sleeves_5050, 126),
        ("50kRaw20+30_9mLB_Reb9M", sleeves_5050, 189),
        ("50kRaw20+30_9mLB_Reb12M", sleeves_5050, 252),
        ("50kRaw20+30_12mLB_Reb6M", sleeves_5050_12m, 126),
        ("50kRaw20+30_12mLB_Reb12M", sleeves_5050_12m, 252),
        ("50kRaw20+30_NormZ_Reb6M", sleeves_5050_normz, 126),
        ("Raw25_12mLB_Buf40_Reb6M", sleeves_raw25_buf, 126),
        ("Raw25_12mLB_Buf40_Reb12M", sleeves_raw25_buf, 252),
    ]
    for b_lbl, slv, reb_d in baskets:
        for p_lbl, r_name, r_params, d_asset, trim in protections:
            alloc = compute_equity_allocation_series(feat, r_name, r_params)
            r = run_backtest(
                adj_df, raw_df, scores, def_df, alloc,
                BacktestConfig(
                    start_date=full_start, end_date=end_dt, rebalance_days=reb_d,
                    cost_bps_one_way=15.0, apply_tax=True, defensive_asset=d_asset,
                    trim_mode=trim, sleeves=slv,
                ),
                stock_ma50=stock_ma50,
            )
            rows.append({"group": "BASKET_x_REGIME", "label": f"{b_lbl} | {p_lbl}", **r["metrics"]})

    # 3. Start-Date Offset Sensitivity (+0d, +21d, +42d, +63d) + Discrete ₹1L / ₹2L + 2x Cost Stress + Walk-Forward on top candidates
    finalists = [
        ("F1_50kRaw20+30_9mLB_Reb6M_DD12_Shift50_GoldLiq50", sleeves_5050, 126, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.50}, "GOLD_50_LIQUID_50", "weakest_first"),
        ("F2_50kRaw20+30_9mLB_Reb6M_DD12_Shift75_GoldLiq50", sleeves_5050, 126, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75}, "GOLD_50_LIQUID_50", "weakest_first"),
        ("F3_50kRaw20+30_9mLB_Reb6M_DD12_Shift100_GoldOrLiq", sleeves_5050, 126, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 1.00}, "GOLD_OR_LIQUID", "pro_rata"),
        ("F4_50kRaw20+30_12mLB_Reb12M_DD12_Shift50_GoldLiq50", sleeves_5050_12m, 252, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.50}, "GOLD_50_LIQUID_50", "weakest_first"),
        ("F5_50kRaw20+30_12mLB_Reb12M_DD12_Shift100_GoldOrLiq", sleeves_5050_12m, 252, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 1.00}, "GOLD_OR_LIQUID", "pro_rata"),
        ("F6_50kRaw20+30_NormZ_Reb6M_DD12_Shift50_GoldLiq50", sleeves_5050_normz, 126, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.50}, "GOLD_50_LIQUID_50", "weakest_first"),
    ]
    start_dates = ["2017-12-01", "2018-01-01", "2018-02-01", "2018-03-01"]
    for f_lbl, slv, reb_d, r_name, r_params, d_asset, trim in finalists:
        alloc = compute_equity_allocation_series(feat, r_name, r_params)
        for s_d in start_dates:
            r = run_backtest(
                adj_df, raw_df, scores, def_df, alloc,
                BacktestConfig(
                    start_date=s_d, end_date=end_dt, rebalance_days=reb_d,
                    cost_bps_one_way=15.0, apply_tax=True, defensive_asset=d_asset,
                    trim_mode=trim, sleeves=slv,
                ),
                stock_ma50=stock_ma50,
            )
            rows.append({"group": "START_OFFSET", "label": f"{f_lbl} | start={s_d}", **r["metrics"]})

        # Discrete ₹1L, ₹2L, and 2x Cost Stress
        for cap, disc, bps, tag in [
            (100_000.0, True, 15.0, "Discrete_1L"),
            (200_000.0, True, 15.0, "Discrete_2L"),
            (100_000.0, False, 30.0, "Stress_2xCost_30bps"),
        ]:
            r = run_backtest(
                adj_df, raw_df, scores, def_df, alloc,
                BacktestConfig(
                    start_date=full_start, end_date=end_dt, initial_capital=cap,
                    rebalance_days=reb_d, cost_bps_one_way=bps, apply_tax=True,
                    defensive_asset=d_asset, trim_mode=trim, discrete_shares=disc, sleeves=slv,
                ),
                stock_ma50=stock_ma50,
            )
            rows.append({"group": "EXEC_AND_STRESS", "label": f"{f_lbl} | {tag}", **r["metrics"]})

        # Walk forward 3 periods
        for wname, s_d, e_d in [("WF1_2018_20", "2017-12-01", "2020-12-31"), ("WF2_2021_23", "2021-01-01", "2023-12-31"), ("WF3_2024_26", "2024-01-01", end_dt)]:
            r = run_backtest(
                adj_df, raw_df, scores, def_df, alloc,
                BacktestConfig(
                    start_date=s_d, end_date=e_d, rebalance_days=reb_d,
                    cost_bps_one_way=15.0, apply_tax=True, defensive_asset=d_asset,
                    trim_mode=trim, sleeves=slv,
                ),
                stock_ma50=stock_ma50,
            )
            rows.append({"group": "WALK_FORWARD", "label": f"{f_lbl} | {wname}", **r["metrics"]})

    out_json = REPO_ROOT / "reports" / "phase2_plateau.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str))

    df = pd.DataFrame(rows)
    cols = ["group", "label", "cagr_pct", "max_dd_pct", "calmar", "sharpe", "sortino", "ann_turnover_pct", "avg_eq_pct", "win_years_pct"]
    out_txt = REPO_ROOT / "reports" / "phase2_plateau.txt"
    out_txt.write_text(df[cols].to_string(index=False))
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
