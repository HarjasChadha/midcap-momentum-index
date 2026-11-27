"""Clean 2018-01-01+ finalist verification: eliminates 12M warm-up bias and tests discrete SlotCap + LIQUID sweep + defensive tax sensitivity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtester import BacktestConfig, compute_metrics, run_backtest  # noqa: E402
from src.data_loader import fetch_and_cache_all  # noqa: E402
from src.momentum import compute_score_matrix  # noqa: E402
from src.nse_mcp import NseMcpClient  # noqa: E402
from src.regime import build_defensive_series, build_ew_index, build_regime_features, compute_equity_allocation_series  # noqa: E402


def _patch_tail_nulls(adj_df: pd.DataFrame, raw_df: pd.DataFrame, vol_df: pd.DataFrame) -> None:
    tail_nulls = list(adj_df.columns[adj_df.iloc[-1].isna()])
    if not tail_nulls:
        return
    last_dt = str(adj_df.index[-1])
    quotes = NseMcpClient().get_bulk_quote(tail_nulls)
    changed = False
    for sym in tail_nulls:
        q = quotes.get(sym)
        if isinstance(q, dict) and str(q.get("date", "")).startswith(last_dt):
            cl = float(q.get("close", 0.0))
            if cl > 0:
                adj_df.loc[last_dt, sym] = cl
                raw_df.loc[last_dt, sym] = cl
                if q.get("volume"):
                    vol_df.loc[last_dt, sym] = float(q["volume"])
                changed = True
    if changed:
        cdir = REPO_ROOT / "data" / "cache"
        adj_df.to_csv(cdir / "adj_close.csv.gz", compression="gzip")
        raw_df.to_csv(cdir / "raw_close.csv.gz", compression="gzip")
        vol_df.to_csv(cdir / "volume.csv.gz", compression="gzip")


def main() -> None:
    data = fetch_and_cache_all(REPO_ROOT, refresh=False)
    adj_df, raw_df, vol_df, bench_df = data["adj"], data["raw"], data["vol"], data["bench"]
    _patch_tail_nulls(adj_df, raw_df, vol_df)

    feat = build_regime_features(adj_df, bench_df)
    def_df = build_defensive_series(bench_df)
    stock_ma50 = adj_df.rolling(50, min_periods=30).mean()

    scores = {
        "raw_9m": compute_score_matrix(adj_df, 189, 21, "raw"),
        "raw_12m": compute_score_matrix(adj_df, 252, 21, "raw"),
    }

    s_20_30_9m = [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5, "weighting": "equal"}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5, "weighting": "equal"}]
    s_20_30_9m_nz = [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5, "weighting": "norm_z"}, {"score_key": "raw_9m", "top_n": 30, "alloc": 0.5, "weighting": "norm_z"}]
    s_20_30_12m = [{"score_key": "raw_12m", "top_n": 20, "alloc": 0.5, "weighting": "equal"}, {"score_key": "raw_12m", "top_n": 30, "alloc": 0.5, "weighting": "equal"}]

    reg_none = compute_equity_allocation_series(feat, "none", {})
    reg_dd50 = compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.50})
    reg_dd75 = compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 0.75})
    reg_dd100 = compute_equity_allocation_series(feat, "index_dd", {"dd_thresh": 0.12, "reenter_ma": 50, "defensive_shift": 1.00})

    clean_start, clean_end = "2018-01-02", "2026-11-26"

    specs = {
        "B0_Unhedged_9m_6M": (s_20_30_9m, 126, reg_none, "GOLD_50_LIQUID_50", "weakest_first"),
        "B0_Unhedged_12m_12M": (s_20_30_12m, 252, reg_none, "GOLD_50_LIQUID_50", "weakest_first"),
        "F1_9m_6M_DD12_Sh50_GoldLiq50": (s_20_30_9m, 126, reg_dd50, "GOLD_50_LIQUID_50", "weakest_first"),
        "F2_9m_6M_DD12_Sh75_GoldLiq50": (s_20_30_9m, 126, reg_dd75, "GOLD_50_LIQUID_50", "weakest_first"),
        "F2b_9m_6M_DD12_Sh75_GoldOrLiq": (s_20_30_9m, 126, reg_dd75, "GOLD_OR_LIQUID", "weakest_first"),
        "F3_9m_6M_DD12_Sh100_GoldOrLiq": (s_20_30_9m, 126, reg_dd100, "GOLD_OR_LIQUID", "pro_rata"),
        "F4_12m_12M_DD12_Sh50_GoldLiq50": (s_20_30_12m, 252, reg_dd50, "GOLD_50_LIQUID_50", "weakest_first"),
        "F6_NormZ_9m_6M_DD12_Sh50_GoldLiq50": (s_20_30_9m_nz, 126, reg_dd50, "GOLD_50_LIQUID_50", "weakest_first"),
        "F7_NormZ_9m_6M_DD12_Sh75_GoldLiq50": (s_20_30_9m_nz, 126, reg_dd75, "GOLD_50_LIQUID_50", "weakest_first"),
    }

    rows = []
    for name, (slv, reb, alloc, def_a, trim) in specs.items():
        r = run_backtest(adj_df, raw_df, scores, def_df, alloc, BacktestConfig(start_date=clean_start, end_date=clean_end, rebalance_days=reb, cost_bps_one_way=15.0, apply_tax=True, defensive_asset=def_a, trim_mode=trim, sleeves=slv), stock_ma50)
        rows.append({"scenario": "Clean_2018_26_15bps", "name": name, **r["metrics"]})

        if name in {"F1_9m_6M_DD12_Sh50_GoldLiq50", "F2_9m_6M_DD12_Sh75_GoldLiq50", "F2b_9m_6M_DD12_Sh75_GoldOrLiq", "F6_NormZ_9m_6M_DD12_Sh50_GoldLiq50", "F7_NormZ_9m_6M_DD12_Sh75_GoldLiq50"}:
            # Tax defensive test
            r_td = run_backtest(adj_df, raw_df, scores, def_df, alloc, BacktestConfig(start_date=clean_start, end_date=clean_end, rebalance_days=reb, cost_bps_one_way=15.0, apply_tax=True, tax_defensive=True, defensive_asset=def_a, trim_mode=trim, sleeves=slv), stock_ma50)
            rows.append({"scenario": "TaxDef_15bps", "name": name, **r_td["metrics"]})

            # Discrete 1L variants
            for label, cap, af, sw in [
                ("Disc1L_NoCap_Cash0", 100_000.0, False, False),
                ("Disc1L_SlotCap_LiqSweep", 100_000.0, True, True),
                ("Disc2L_SlotCap_LiqSweep", 200_000.0, True, True),
                ("Disc5L_NoCap_LiqSweep", 500_000.0, False, True),
            ]:
                rd = run_backtest(adj_df, raw_df, scores, def_df, alloc, BacktestConfig(start_date=clean_start, end_date=clean_end, initial_capital=cap, rebalance_days=reb, cost_bps_one_way=15.0, apply_tax=True, defensive_asset=def_a, trim_mode=trim, sleeves=slv, discrete_shares=True, affordability_filter=af, sweep_idle_cash=sw, sweep_asset="LIQUID_TBILL"), stock_ma50)
                rows.append({"scenario": label, "name": name, **rd["metrics"]})

    # Clean Benchmarks 2018-01-01..2026-11-26
    ew = build_ew_index(adj_df).loc[clean_start:clean_end]
    rows.append({"scenario": "Benchmark", "name": "Midcap150_EW_Universe", **compute_metrics(ew)})
    for col in ("GOLDBEES", "GOLD_50_LIQUID_50", "GOLD_OR_LIQUID", "LIQUID_TBILL", "NIFTYBEES"):
        bs = def_df[col].loc[clean_start:clean_end].dropna()
        rows.append({"scenario": "Benchmark", "name": f"Def_{col}", **compute_metrics(bs)})

    df = pd.DataFrame(rows)
    cols = ["scenario", "name", "cagr_pct", "max_dd_pct", "calmar", "sharpe", "sortino", "ann_turnover_pct", "avg_eq_pct", "avg_holdings", "win_years_pct", "total_tax_paid"]
    out_txt = df[cols].to_string(index=False)
    (REPO_ROOT / "reports" / "clean_verification.txt").write_text(out_txt)
    (REPO_ROOT / "reports" / "clean_verification.json").write_text(json.dumps(rows, indent=2, default=str))
    print(out_txt)


if __name__ == "__main__":
    main()
