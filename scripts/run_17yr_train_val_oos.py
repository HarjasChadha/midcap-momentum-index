"""17-Year Official NSE Archive (2010-2026) Multi-Period Optimization & Out-of-Sample Validation (Survivorship-Free PIT vs Current 150)."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtester import BacktestConfig, compute_metrics, run_backtest  # noqa: E402
from src.data_loader import load_universe  # noqa: E402
from src.momentum import compute_score_matrix  # noqa: E402
from src.nse_archive_loader import build_2010_2026_nse_dataset  # noqa: E402
from src.regime import build_defensive_series, build_ew_index, build_regime_features, compute_equity_allocation_series  # noqa: E402

ETFS = {"GOLDBEES", "LIQUIDBEES", "LIQUIDCASE", "NIFTYBEES", "MON100", "SILVERBEES", "EBBETF0431", "LTGILTBEES"}

PERIODS = {
    "P1_Train_2011_16": ("2011-01-03", "2016-05-31"),
    "P2_Val1_2016_21": ("2016-06-01", "2021-06-30"),
    "P3_Val2_2021_26": ("2021-07-01", "2026-11-26"),
    "FULL_2011_26": ("2011-01-03", "2026-11-26"),
}

_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    global _CTX
    _CTX = ctx


def _run_one_task(task: tuple[str, str, str, int, str, str, str]) -> dict:
    label, b_name, skey, wt, r_name, r_days, reg_name = task
    sleeves = [{"score_key": skey, "top_n": 20, "alloc": 0.5, "weighting": wt}, {"score_key": skey, "top_n": 30, "alloc": 0.5, "weighting": wt}]
    alloc, def_a, trim = _CTX["regimes"][reg_name]
    cid = f"{b_name} | {r_name} | {reg_name}"
    rec = {"universe": label, "config_id": cid, "basket": b_name, "reb": r_name, "regime": reg_name}
    for p_key, (ps, pe) in PERIODS.items():
        res = run_backtest(
            _CTX["adj"],
            _CTX["raw"],
            _CTX["scores"],
            _CTX["def_df"],
            alloc,
            BacktestConfig(start_date=ps, end_date=pe, rebalance_days=r_days, cost_bps_one_way=15.0, apply_tax=True, defensive_asset=def_a, trim_mode=trim, sleeves=sleeves),
            stock_ma50=_CTX["ma50"],
        )
        m = res["metrics"]
        rec[f"{p_key}_cagr"] = m["cagr_pct"]
        rec[f"{p_key}_dd"] = m["max_dd_pct"]
        rec[f"{p_key}_cal"] = m["calmar"]
        rec[f"{p_key}_sh"] = m["sharpe"]
    return rec


def build_pit_midcap_mask(adj_df: pd.DataFrame, raw_df: pd.DataFrame, tv_df: pd.DataFrame) -> pd.DataFrame:
    """Zero-survivorship point-in-time Midcap 150 proxy: Ranks 81..230 by trailing 126d median NSE Rupee turnover (raw >= Rs 20, active >= 90d)."""
    med_tv = tv_df.rolling(126, min_periods=60).median()
    active = adj_df.rolling(126, min_periods=90).count() >= 90
    price_ok = raw_df >= 20.0
    eligible_tv = med_tv.where(active & price_ok)
    ranks = eligible_tv.rank(axis=1, ascending=False, method="first")
    return (ranks >= 81) & (ranks <= 230)


def eval_periods_parallel(
    label: str,
    adj_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    pit_mask: pd.DataFrame | None = None,
) -> dict:
    feat = build_regime_features(adj_df, bench_df, mask=pit_mask)
    def_df = build_defensive_series(bench_df)
    stock_ma50 = adj_df.rolling(50, min_periods=30).mean()

    raw_scores = {
        "raw_6m": compute_score_matrix(adj_df, 126, 21, "raw"),
        "raw_9m": compute_score_matrix(adj_df, 189, 21, "raw"),
        "raw_12m": compute_score_matrix(adj_df, 252, 21, "raw"),
        "risk_9m": compute_score_matrix(adj_df, 189, 21, "risk_adj"),
    }
    z_raw = raw_scores["raw_9m"].sub(raw_scores["raw_9m"].mean(axis=1), axis=0).div(raw_scores["raw_9m"].std(axis=1).replace(0, np.nan), axis=0)
    z_risk = raw_scores["risk_9m"].sub(raw_scores["risk_9m"].mean(axis=1), axis=0).div(raw_scores["risk_9m"].std(axis=1).replace(0, np.nan), axis=0)
    raw_scores["hybrid_9m"] = 0.5 * z_raw + 0.5 * z_risk

    if pit_mask is not None:
        m = pit_mask.reindex_like(adj_df).fillna(False)
        scores = {k: v.where(m) for k, v in raw_scores.items()}
    else:
        scores = raw_scores

    regimes = {
        "Unhedged": (compute_equity_allocation_series(feat, "none", {}), "GOLDBEES", "weakest_first"),
        "DD10_Sh75_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.10, "reenter_ma": 50, "defensive_shift": 0.75}), "GOLD_50_LIQUID_50", "weakest_first"),
        "DD12_Sh50_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.50}), "GOLD_50_LIQUID_50", "weakest_first"),
        "DD12_Sh75_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75}), "GOLD_50_LIQUID_50", "weakest_first"),
        "DD14_Sh75_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.14, "reenter_ma": 50, "defensive_shift": 0.75}), "GOLD_50_LIQUID_50", "weakest_first"),
        "DD15_Sh75_GL50": (compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.15, "reenter_ma": 50, "defensive_shift": 0.75}), "GOLD_50_LIQUID_50", "weakest_first"),
        "MA200Band_Sh75_GL50": (compute_equity_allocation_series(feat, "index_ma_band", {"ma_len": 200, "exit_band": 0.97, "enter_band": 1.01, "defensive_shift": 0.75}), "GOLD_50_LIQUID_50", "weakest_first"),
        "Breadth35_Sh50_GOL": (compute_equity_allocation_series(feat, "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.35, "enter_above": 0.50, "defensive_shift": 0.50}), "GOLD_OR_LIQUID", "weakest_first"),
    }
    baskets = [
        ("Raw_6M_EW", "raw_6m", "equal"),
        ("Raw_9M_EW", "raw_9m", "equal"),
        ("Raw_12M_EW", "raw_12m", "equal"),
        ("Raw_9M_NormZ", "raw_9m", "norm_z"),
        ("Risk_9M_EW", "risk_9m", "equal"),
        ("Hybrid_9M_EW", "hybrid_9m", "equal"),
    ]
    rebs = [("6M", 126), ("12M", 252)]

    tasks = [(label, b_name, skey, wt, r_name, r_days, reg_name) for b_name, skey, wt in baskets for r_name, r_days in rebs for reg_name in regimes]
    ctx = {"adj": adj_df, "raw": raw_df, "scores": scores, "def_df": def_df, "ma50": stock_ma50, "regimes": regimes}
    n_workers = min(6, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        records = list(pool.map(_run_one_task, tasks, chunksize=4))

    ew_full = build_ew_index(adj_df, mask=pit_mask)
    bench_metrics = {p_key: compute_metrics(ew_full.loc[ps:pe]) for p_key, (ps, pe) in PERIODS.items()}
    return {"records": records, "ew_benchmarks": bench_metrics}


def main() -> None:
    d = build_2010_2026_nse_dataset(REPO_ROOT, 2010, 2026, refresh=False)
    adj_all, raw_all, tv_all, bench_df = d["adj"], d["raw"], d["trdval"], d["bench"]

    stock_cols = [c for c in adj_all.columns if c not in ETFS]
    adj_eq, raw_eq, tv_eq = adj_all[stock_cols], raw_all[stock_cols], tv_all[stock_cols]

    print("Building Zero-Survivorship Point-in-Time Midcap Mask (Ranks 81..230 trailing 126d NSE Rupee Turnover)...")
    pit_mask = build_pit_midcap_mask(adj_eq, raw_eq, tv_eq)
    pit_cols = list(pit_mask.columns[pit_mask.any(axis=0)])
    print(f"Eligible PIT stocks across 2010-2026: {len(pit_cols)}")

    res_pit = eval_periods_parallel("PIT_NO_SURVIVOR_150", adj_eq[pit_cols], raw_eq[pit_cols], bench_df, pit_mask=pit_mask[pit_cols])
    curr_syms = [s for s in load_universe(REPO_ROOT)["Symbol"].astype(str) if s in adj_eq.columns]
    res_curr = eval_periods_parallel("CURRENT_2026_150", adj_eq[curr_syms], raw_eq[curr_syms], bench_df, pit_mask=None)

    all_recs = res_pit["records"] + res_curr["records"]
    df = pd.DataFrame(all_recs)
    (REPO_ROOT / "reports" / "nse_17yr_train_val_oos.json").write_text(json.dumps({"records": all_recs, "benchmarks": {"PIT": res_pit["ew_benchmarks"], "CURR": res_curr["ew_benchmarks"]}}, indent=2))

    lines = []
    for u_label, b_info in [("PIT_NO_SURVIVOR_150", res_pit["ew_benchmarks"]), ("CURRENT_2026_150", res_curr["ew_benchmarks"])]:
        sub = df[df["universe"] == u_label].copy()
        r12 = sub["P1_Train_2011_16_sh"].rank().corr(sub["P2_Val1_2016_21_sh"].rank())
        r13 = sub["P1_Train_2011_16_sh"].rank().corr(sub["P3_Val2_2021_26_sh"].rank())
        c12 = sub["P1_Train_2011_16_cal"].rank().corr(sub["P2_Val1_2016_21_cal"].rank())
        c13 = sub["P1_Train_2011_16_cal"].rank().corr(sub["P3_Val2_2021_26_cal"].rank())

        lines.append(f"==================== {u_label} (96 configs across 16 yrs: 2011-2026) ====================")
        lines.append("EW Benchmark: " + " | ".join(f"{k}: {v['cagr_pct']}% DD={v['max_dd_pct']}% Sh={v['sharpe']}" for k, v in b_info.items()))
        lines.append(f"Spearman Sharpe P1->P2={r12:.3f}, P1->P3={r13:.3f} | Calmar P1->P2={c12:.3f}, P1->P3={c13:.3f}")
        cols = ["config_id", "P1_Train_2011_16_cagr", "P1_Train_2011_16_dd", "P1_Train_2011_16_cal", "P2_Val1_2016_21_cagr", "P2_Val1_2016_21_dd", "P2_Val1_2016_21_cal", "P3_Val2_2021_26_cagr", "P3_Val2_2021_26_dd", "P3_Val2_2021_26_cal", "FULL_2011_26_cagr", "FULL_2011_26_dd", "FULL_2011_26_sh"]
        lines.append("--- Top 15 Chosen BLINDLY by Period-1 Train (2011-16) Calmar -> Evaluated on P2 (2016-21) & P3 (2021-26) ---")
        lines.append(sub.sort_values("P1_Train_2011_16_cal", ascending=False).head(15)[cols].to_string(index=False))
        lines.append("--- Regime Averages Across P1 -> P2 -> P3 -> Full ---")
        lines.append(sub.groupby("regime")[["P1_Train_2011_16_cal", "P2_Val1_2016_21_cal", "P3_Val2_2021_26_cal", "FULL_2011_26_cagr", "FULL_2011_26_dd", "FULL_2011_26_sh"]].mean().round(2).sort_values("P1_Train_2011_16_cal", ascending=False).to_string())
        lines.append("--- Basket Averages Across P1 -> P2 -> P3 -> Full ---")
        lines.append(sub.groupby("basket")[["P1_Train_2011_16_cal", "P2_Val1_2016_21_cal", "P3_Val2_2021_26_cal", "FULL_2011_26_cagr", "FULL_2011_26_dd", "FULL_2011_26_sh"]].mean().round(2).sort_values("P1_Train_2011_16_cal", ascending=False).to_string())
        lines.append("--- Rebalance Averages Across P1 -> P2 -> P3 -> Full ---")
        lines.append(sub.groupby("reb")[["P1_Train_2011_16_cal", "P2_Val1_2016_21_cal", "P3_Val2_2021_26_cal", "FULL_2011_26_cagr", "FULL_2011_26_dd", "FULL_2011_26_sh"]].mean().round(2).to_string())
        f2_and_base = sub[sub["config_id"].isin([
            "Raw_9M_EW | 6M | Unhedged",
            "Raw_9M_EW | 6M | DD12_Sh75_GL50",
            "Raw_9M_EW | 6M | DD14_Sh75_GL50",
            "Raw_9M_EW | 6M | DD15_Sh75_GL50",
            "Raw_9M_EW | 6M | MA200Band_Sh75_GL50",
            "Hybrid_9M_EW | 6M | DD14_Sh75_GL50",
            "Hybrid_9M_EW | 6M | DD15_Sh75_GL50",
            "Risk_9M_EW | 6M | DD15_Sh75_GL50",
            "Risk_9M_EW | 12M | DD15_Sh75_GL50",
        ])]
        lines.append("--- Key Candidates ---")
        lines.append(f2_and_base[cols].to_string(index=False))
        lines.append("")

    report = "\n".join(lines)
    (REPO_ROOT / "reports" / "nse_17yr_train_val_oos.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
