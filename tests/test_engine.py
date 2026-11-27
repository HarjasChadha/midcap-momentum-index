import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.backtester import BacktestConfig, compute_metrics, run_backtest  # noqa: E402
from src.data_loader import _repair_transient_glitches  # noqa: E402
from src.momentum import combine_sleeves, compute_score_matrix, normalize_z_pdf, select_affordable_blend, select_sleeve_weights  # noqa: E402
from src.nse_archive_loader import parse_bc_purpose_factor  # noqa: E402
from src.nse_mcp import CMMKT_MCP_URL, NseMcpClient  # noqa: E402
from src.prowess_loader import build_point_in_time_midcap150_mask, load_prowess_file, prowess_to_panels  # noqa: E402
from src.regime import (  # noqa: E402
    build_defensive_series,
    build_regime_features,
    compute_equity_allocation_series,
    compute_multifactor_state,
    get_trend_gold_status,
)


def _synthetic_market(days: int = 360, n_stocks: int = 40):
    idx = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2022-01-03", periods=days)]
    rng = np.random.default_rng(42)
    cols = [f"STK{i:02d}" for i in range(n_stocks)]
    drifts = np.linspace(-0.0002, 0.0018, n_stocks)
    rets = rng.normal(drifts, 0.015, size=(days, n_stocks))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    adj_df = pd.DataFrame(prices, index=idx, columns=cols)
    raw_df = adj_df.copy()
    bench = pd.DataFrame(
        {
            "GOLDBEES": 50.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.007, days)),
            "LIQUIDBEES": np.full(days, 1000.0),
            "INDIA_VIX": np.full(days, 15.0),
        },
        index=idx,
    )
    return adj_df, raw_df, bench


class EngineTests(unittest.TestCase):
    def test_repair_transient_glitches_only_fixes_snapbacks(self):
        idx = [f"2023-01-0{i}" for i in range(1, 8)]
        df = pd.DataFrame({"GOLD": [100.0, 101.0, 1.01, 1.02, 101.5, 102.0, 103.0], "REAL": [100.0, 101.0, 30.0, 29.0, 28.0, 27.0, 26.0]}, index=idx)
        rep = _repair_transient_glitches(df)
        self.assertGreater(rep.loc["2023-01-03", "GOLD"], 90.0)
        self.assertGreater(rep.loc["2023-01-04", "GOLD"], 90.0)
        self.assertAlmostEqual(rep.loc["2023-01-03", "REAL"], 30.0)

    def test_momentum_scoring_and_50_50_split(self):
        adj_df, _, _ = _synthetic_market()
        scores = compute_score_matrix(adj_df, lookback_days=126, skip_days=21, mode="raw")
        last_row = scores.iloc[-1]
        w20 = select_sleeve_weights(last_row, 20, "equal")
        w30 = select_sleeve_weights(last_row, 30, "equal")
        self.assertEqual(len(w20), 20)
        self.assertEqual(len(w30), 30)
        self.assertAlmostEqual(sum(w20.values()), 1.0, places=6)

        blended = combine_sleeves([(0.5, w20), (0.5, w30)])
        self.assertEqual(len(blended), 30)
        self.assertAlmostEqual(sum(blended.values()), 1.0, places=6)
        # Top-20 stocks get 0.5/20 + 0.5/30 = 1/24; Ranks 21-30 get 0.5/30 = 1/60
        top_sym = next(iter(w20))
        rank25_sym = [s for s in w30 if s not in w20][0]
        self.assertAlmostEqual(blended[top_sym], 1.0 / 24.0, places=6)
        self.assertAlmostEqual(blended[rank25_sym], 1.0 / 60.0, places=6)

    def test_rank_buffer_reduces_churn(self):
        adj_df, _, _ = _synthetic_market()
        scores = compute_score_matrix(adj_df, lookback_days=63, skip_days=21, mode="raw")
        r0 = scores.iloc[-20]
        r1 = scores.iloc[-1]
        w0 = select_sleeve_weights(r0, 15, "equal")
        w_nobuf = select_sleeve_weights(r1, 15, "equal")
        w_buf = select_sleeve_weights(r1, 15, "equal", prev_holdings=set(w0), buffer_n=25)
        self.assertEqual(len(w_buf), 15)
        self.assertGreaterEqual(len(set(w0) & set(w_buf)), len(set(w0) & set(w_nobuf)))

    def test_normalize_z_pdf(self):
        nz = normalize_z_pdf(pd.Series({"A": 1.5, "B": -1.0}))
        self.assertAlmostEqual(nz["A"], 2.5)
        self.assertAlmostEqual(nz["B"], 0.5)

    def test_zero_lookahead_and_tax_drag(self):
        adj_df, raw_df, bench = _synthetic_market(days=400, n_stocks=35)
        feat = build_regime_features(adj_df, bench)
        def_df = build_defensive_series(bench)
        scores = {"raw_6m": compute_score_matrix(adj_df, 126, 21, "raw")}
        alloc = compute_equity_allocation_series(feat, "breadth_hysteresis", {"col": "breadth_50_sma10", "exit_below": 0.35, "enter_above": 0.50})
        start_dt = adj_df.index[180]
        end_dt = adj_df.index[-1]

        res_notax = run_backtest(
            adj_df, raw_df, scores, def_df, alloc,
            BacktestConfig(start_date=start_dt, end_date=end_dt, rebalance_days=42, apply_tax=False, sleeves=[{"score_key": "raw_6m", "top_n": 15, "alloc": 1.0}]),
        )
        res_tax = run_backtest(
            adj_df, raw_df, scores, def_df, alloc,
            BacktestConfig(start_date=start_dt, end_date=end_dt, rebalance_days=42, apply_tax=True, sleeves=[{"score_key": "raw_6m", "top_n": 15, "alloc": 1.0}]),
        )
        self.assertGreater(len(res_notax["nav"]), 150)
        self.assertLessEqual(res_tax["metrics"]["final_value"], res_notax["metrics"]["final_value"] + 1e-6)

        # Discrete integer shares test
        res_disc = run_backtest(
            adj_df, raw_df, scores, def_df, alloc,
            BacktestConfig(start_date=start_dt, end_date=end_dt, rebalance_days=63, discrete_shares=True, sleeves=[{"score_key": "raw_6m", "top_n": 15, "alloc": 1.0}]),
        )
        self.assertGreater(res_disc["metrics"]["final_value"], 0)
        # Should NOT trade every single day from rounding residual when no regime shift occurs
        ones = pd.Series(1.0, index=adj_df.index)
        res_disc_nh = run_backtest(
            adj_df, raw_df, scores, def_df, ones,
            BacktestConfig(start_date=start_dt, end_date=end_dt, rebalance_days=63, discrete_shares=True, sleeves=[{"score_key": "raw_6m", "top_n": 15, "alloc": 1.0}]),
        )
        self.assertLessEqual(int((res_disc_nh["turnover"] > 0).sum()), 6)

    def test_nse_mcp_client_sse_parsing(self):
        client = NseMcpClient()
        sample = 'event:message\ndata:{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\\"ok\\": true}"}]}}'
        parsed = client._parse_rpc_body(sample)
        self.assertEqual(parsed["id"], 2)
        cmmkt = client._cmmkt()
        self.assertEqual(cmmkt.endpoint, CMMKT_MCP_URL)

    def test_sequential_slot_cap_substitutes_unaffordable_stocks(self):
        # Top stock costs ₹15,000 (> ₹25k budget * w), remaining stocks are affordable ₹500
        scores = pd.Series({f"S{i:02d}": 100.0 - i for i in range(15)})
        prices = pd.Series({f"S{i:02d}": (15000.0 if i == 0 else 500.0) for i in range(15)})
        sleeves = [{"score_key": "raw_9m", "top_n": 10, "alloc": 1.0, "weighting": "equal"}]
        w = select_affordable_blend({"raw_9m": scores}, sleeves, prices, capital_inr=25_000.0)
        self.assertEqual(len(w), 10)
        self.assertNotIn("S00", w)
        self.assertIn("S10", w)

    def test_sweep_idle_cash_and_defensive_tax(self):
        adj_df, raw_df, bench = _synthetic_market(days=400, n_stocks=35)
        raw_df = raw_df * 30.0  # Make stocks ~₹3,000+ so 1L discrete leaves meaningful uninvested cash
        feat = build_regime_features(adj_df, bench)
        def_df = build_defensive_series(bench)
        scores = {"raw_6m": compute_score_matrix(adj_df, 126, 21, "raw")}
        ones = pd.Series(1.0, index=adj_df.index)
        start_dt, end_dt = adj_df.index[180], adj_df.index[-1]
        r_no_sw = run_backtest(adj_df, raw_df, scores, def_df, ones, BacktestConfig(start_date=start_dt, end_date=end_dt, rebalance_days=63, discrete_shares=True, sweep_idle_cash=False, sleeves=[{"score_key": "raw_6m", "top_n": 15, "alloc": 1.0}]))
        r_sw = run_backtest(adj_df, raw_df, scores, def_df, ones, BacktestConfig(start_date=start_dt, end_date=end_dt, rebalance_days=63, discrete_shares=True, sweep_idle_cash=True, sweep_asset="LIQUID_TBILL", sleeves=[{"score_key": "raw_6m", "top_n": 15, "alloc": 1.0}]))
        self.assertGreater(r_sw["metrics"]["final_value"], r_no_sw["metrics"]["final_value"])

    def test_live_state_and_config_strict_json(self):
        cfg = json.loads((REPO_ROOT / "config" / "strategy.json").read_text())
        self.assertEqual(cfg["primary_strategy"], "R3_Risk9m_6M_MF4_TrendGold150")
        live_path = REPO_ROOT / "data" / "live_state.json"
        if live_path.exists():
            def _reject_nan(val):
                raise ValueError(f"Forbidden JSON constant: {val}")
            data = json.loads(live_path.read_text(), parse_constant=_reject_nan)
            self.assertEqual(data["universe_count"], 150)
            self.assertIn(data["regime"]["state"], {"RISK_ON", "CAUTIOUS", "DEFENSIVE"})
            self.assertIn("multifactor_blocks", data["regime"])
            self.assertIn("trend_gold_status", data["regime"])
            self.assertGreaterEqual(len(data["rankings_top60"]), 50)
            self.assertEqual(data["orders_1L"]["R3_Risk9M_SlotCap_LiqSweep"]["held_count"], 30)
            self.assertEqual(data["orders_1L"]["S3_Sortino9M_SlotCap_LiqSweep"]["held_count"], 30)
            self.assertEqual(data["orders_1L"]["R2_Risk9M_SlotCap_LiqSweep"]["held_count"], 30)
            self.assertEqual(data["orders_1L"]["F2_SlotCap_LiqSweep"]["held_count"], 30)
            self.assertIn("rk9", data["universe_scores"][0])
            self.assertIn("so9", data["universe_scores"][0])

    def test_prowess_loader_and_point_in_time_mask(self):
        import tempfile
        sample = (
            "Company Name|NSE Symbol|Date|NSE Closing Price|NSE Adjusted Closing Price|NSE Market Capitalisation\n"
            "Alpha Ltd|ALPHA|2024-01-02|100.0|100.0|50000\n"
            "Alpha Ltd|ALPHA|2024-01-03|105.0|105.0|52500\n"
            "Beta Ltd|BETA|2024-01-02|400.0|200.0|20000\n"
            "Beta Ltd|BETA|2024-01-03|204.0|204.0|20400\n"
        )
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "prowess_sample.txt"
            f.write_text(sample)
            raw_tbl = load_prowess_file(f)
            panels = prowess_to_panels(raw_tbl)
            self.assertAlmostEqual(float(panels["raw"].loc["2024-01-02", "BETA"]), 400.0)
            self.assertAlmostEqual(float(panels["adj"].loc["2024-01-02", "BETA"]), 200.0)
            # Point-in-time rank masking test (300 ranks: indices 100..249 kept)
            idx = pd.date_range("2024-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
            mcap = pd.DataFrame({f"C{i:03d}": np.full(len(idx), 1000.0 - i) for i in range(320)}, index=idx)
            mask = build_point_in_time_midcap150_mask(mcap, min_stocks_for_pit=300)
            self.assertFalse(bool(mask.iloc[-1]["C000"]))  # Large cap Rank 1 excluded
            self.assertTrue(bool(mask.iloc[-1]["C150"]))   # Midcap Rank 151 included
            self.assertFalse(bool(mask.iloc[-1]["C280"]))  # Small cap Rank 281 excluded

    def test_nse_bc_corporate_action_factor_parser(self):
        self.assertAlmostEqual(parse_bc_purpose_factor("FV SPLT FRM RS 10 TO RS 2")[0], 0.2)
        self.assertAlmostEqual(parse_bc_purpose_factor("BONUS 3:1")[0], 0.25)
        self.assertAlmostEqual(parse_bc_purpose_factor("BON 1:2/SPLIT RS.10TORS.2")[0], (2.0 / 3.0) * 0.2)
        self.assertAlmostEqual(parse_bc_purpose_factor("AGM/DIV-FINRS 22+SPLRS 10")[0], 1.0)
        self.assertAlmostEqual(parse_bc_purpose_factor("DIV- FIN RS.6.5+SPL RS.60")[0], 1.0)
        self.assertAlmostEqual(parse_bc_purpose_factor("SCH OF AGMT-BONUS DEB 1:1")[0], 1.0)

    def test_sortino_score_penalizes_downside_more_than_upside(self):
        days = 240
        idx = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-01-02", periods=days)]
        r_clean = np.full(days, 0.0018)
        r_clean[50] = 0.04  # Upside spike only
        r_down = np.full(days, 0.0024)
        r_down[60:80:2] = -0.025  # Downside tail shocks
        r_down[61:81:2] = 0.021
        p_clean = 100.0 * np.cumprod(1.0 + r_clean)
        p_down = 100.0 * np.cumprod(1.0 + r_down)
        # Equalize total 189d lookback price gain at t[-22]
        scale = p_clean[-22] / p_down[-22]
        p_down = p_down * scale
        df = pd.DataFrame({"CLEAN": p_clean, "DOWN": p_down}, index=idx)
        so = compute_score_matrix(df, 189, 21, "sortino").iloc[-1]
        self.assertGreater(float(so["CLEAN"]), float(so["DOWN"]))

    def test_trend_gold_ema150_hysteresis_and_series(self):
        idx = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2022-01-03", periods=320)]
        gold = np.concatenate([np.linspace(50.0, 90.0, 220), np.linspace(89.0, 55.0, 100)])
        bench = pd.DataFrame({
            "GOLDBEES": gold,
            "LIQUIDBEES": 100.0 * (1.06 ** (np.arange(320) / 252.0)),
            "LIQUIDCASE": 100.0 * (1.06 ** (np.arange(320) / 252.0)),
        }, index=idx)
        st_top = get_trend_gold_status(bench.iloc[:220])
        self.assertTrue(st_top["gold_above_gate"])
        self.assertEqual(st_top["active_defensive_etf"], "GOLDBEES")
        st_end = get_trend_gold_status(bench)
        self.assertFalse(st_end["gold_above_gate"])
        self.assertEqual(st_end["active_defensive_etf"], "LIQUIDCASE")
        defs = build_defensive_series(bench)
        self.assertIn("TREND_GOLD150_OR_LIQ", defs.columns)
        # During the final Gold slump window, TREND_GOLD150_OR_LIQ holds steady or earns cash instead of dropping with Gold
        self.assertGreater(float(defs["TREND_GOLD150_OR_LIQ"].iloc[-1] / defs["TREND_GOLD150_OR_LIQ"].iloc[260]), 0.99)

    def test_multifactor_wf4_is_causal_and_no_future_leak(self):
        adj_df, _, bench = _synthetic_market(days=420, n_stocks=35)
        feat = build_regime_features(adj_df, bench)
        alloc_full, _ = compute_multifactor_state(feat, bench)
        alloc_trim, _ = compute_multifactor_state(feat.iloc[:-10], bench.iloc[:-10])
        pd.testing.assert_series_equal(alloc_full.iloc[:-10], alloc_trim, check_names=False)
        alloc_via_rule = compute_equity_allocation_series(feat, "multifactor_wf4", {"bench_df": bench})
        pd.testing.assert_series_equal(alloc_full, alloc_via_rule, check_names=False)
        self.assertTrue(set(np.unique(alloc_full.values)).issubset({0.25, 0.60, 1.0}))


if __name__ == "__main__":
    unittest.main()
