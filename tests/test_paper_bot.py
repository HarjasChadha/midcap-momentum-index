import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.paper_bot import (  # noqa: E402
    build_forward_tape_entry,
    compute_target_weights,
    initialize_account_state,
    run_synthetic_steps,
    step_portfolio,
    update_forward_tape,
    update_paper_ledger,
)


def build_synthetic_parity_tape() -> list[dict]:
    """Build a deterministic 10-day synthetic tape exercising buys, splits, regime shifts, and ETF flips."""
    syms = [f"STK{i:02d}" for i in range(1, 36)]
    dates = [f"2026-11-{d:02d}" for d in range(2, 12)]  # 10 consecutive dates for clean readability
    days: list[dict] = []

    for idx, dt in enumerate(dates):
        prices = {}
        rankings = {}
        for rank_i, sym in enumerate(syms):
            # Base price grows gently; STK01 splits 1:2 on Day 5 (index 4)
            base_px = 200.0 + rank_i * 15.0 + idx * (2.0 if rank_i < 20 else -0.5)
            if sym == "STK01" and idx >= 4:
                base_px = base_px * 0.5
            # Make STK35 very expensive on SlotCap day-0 so SlotCap skips it at 1L
            if sym == "STK35":
                base_px = 18000.0
            prices[sym] = round(base_px, 2)

            # On day 8 (index 7), flip score ranking to prepare T+1 signal if needed
            rk9 = round(4.0 - rank_i * 0.08 + (0.05 if idx >= 7 else 0.0), 4)
            rankings[sym] = {
                "rk9": rk9,
                "so9": round(rk9 * 1.4, 4),
                "r9": round((40.0 - rank_i) + idx * 0.3, 2),
                "r12": round((50.0 - rank_i) + idx * 0.3, 2),
                "a50": 1 if rank_i < 25 else 0,
            }

        prices["GOLDBEES"] = round(100.0 + idx * 1.5, 2)
        prices["LIQUIDCASE"] = round(117.0 + idx * 0.03, 2)

        # Day 0-1: RISK_ON (100% Eq), GOLD
        # Day 2-4: Signal observed at Day 2 close drops to CAUTIOUS (60% Eq) -> executes Day 3
        # Day 5: GOLD gate flips GOLDBEES -> LIQUIDCASE -> executes Day 6
        # Day 7: Recovery signal back to RISK_ON (100% Eq) -> executes Day 8
        if idx < 2:
            eq_mf4, mf4_s, posture = 1.0, 0.28, "RISK_ON"
        elif idx < 7:
            eq_mf4, mf4_s, posture = 0.60, -0.14, "CAUTIOUS"
        else:
            eq_mf4, mf4_s, posture = 1.0, 0.24, "RISK_ON"

        active_def = "GOLDBEES" if idx < 5 else "LIQUIDCASE"
        ca = [{"symbol": "STK01", "type": "SPLIT", "factor": 0.5}] if idx == 4 else []

        days.append(
            {
                "date": dt,
                "prices": prices,
                "corp_actions": ca,
                "regime": {
                    "mf4_score": mf4_s,
                    "posture": posture,
                    "target_eq_mf4": eq_mf4,
                    "target_eq_dd15": eq_mf4,
                    "target_eq_dd75": eq_mf4,
                    "target_eq_dd50": eq_mf4,
                    "active_def_etf": active_def,
                    "gold_price": prices["GOLDBEES"],
                    "gold_ema150": 98.0 if active_def == "GOLDBEES" else 120.0,
                    "gold_vs_ema150_pct": 3.5 if active_def == "GOLDBEES" else -8.0,
                    "ew_index_val": round(1000.0 + idx * 6.0, 2),
                },
                "rankings": rankings,
            }
        )
    return days


class PaperBotTests(unittest.TestCase):
    def test_inception_slotcap_and_liquid_sweep(self):
        tape = build_synthetic_parity_tape()
        st = run_synthetic_steps([tape[0]], initial_capital=100_000.0, preset="R3", slot_cap=True, liquid_sweep=True)
        h0 = st["daily_history"][0]
        # 0.15% one-way friction on Day 0 deployment leaves ~99,850
        self.assertLess(h0["nav"], 100_000.0)
        self.assertGreater(h0["nav"], 99_750.0)
        self.assertEqual(len([h for h in st["holdings"] if h["asset_class"] == "EQUITY"]), 30)
        self.assertIn("LIQUIDCASE", {h["symbol"] for h in st["holdings"]})

    def test_t_plus_one_causality_and_split_and_gate_flip(self):
        tape = build_synthetic_parity_tape()
        st = run_synthetic_steps(tape, initial_capital=1_000_000.0, preset="R3", slot_cap=False, liquid_sweep=True)
        decisions = [d["event_type"] for d in st["decision_log"]]
        self.assertIn("INCEPTION_BUY", decisions)
        self.assertIn("REGIME_SHIFT", decisions)
        self.assertIn("CORPORATE_ACTION", decisions)
        self.assertIn("DEFENSIVE_GATE_FLIP", decisions)

        # Verify Regime Shift executes strictly T+1 (signal observed idx=2 -> executed idx=3 date '2026-11-05')
        reg_dates = [d["date"] for d in st["decision_log"] if d["event_type"] == "REGIME_SHIFT"]
        self.assertEqual(reg_dates[0], tape[3]["date"])
        self.assertEqual(reg_dates[1], tape[8]["date"])

        # Verify Gate Flip executes strictly T+1 (idx=5 close -> idx=6 date '2026-11-08')
        gate_dates = [d["date"] for d in st["decision_log"] if d["event_type"] == "DEFENSIVE_GATE_FLIP"]
        self.assertEqual(gate_dates, [tape[6]["date"]])

        # Verify Split adjusted STK01 shares without artificial 50% NAV crash on split day (idx=4)
        nav_pre = st["daily_history"][3]["nav"]
        nav_post = st["daily_history"][4]["nav"]
        self.assertLess(abs(nav_post / nav_pre - 1.0), 0.03)

        # Verify defensive ETF tax is deducted by default (tax_defensive=True)
        self.assertTrue(st.get("tax_defensive", True))
        self.assertGreater(st["tax_state"]["estimated_def_tax_cum"], 0.0)
        self.assertGreater(st["tax_state"]["realized_def_tax_cum"], 0.0)
        last_h = st["daily_history"][-1]
        self.assertAlmostEqual(last_h["nav_full_tax"], last_h["nav"], places=4)
        self.assertGreater(last_h["nav_pre_def_tax"], last_h["nav"])

        # Verify optional pre-defensive-tax comparison mode (tax_defensive=False)
        st_pre = run_synthetic_steps(tape, initial_capital=1_000_000.0, preset="R3", slot_cap=False, liquid_sweep=True, tax_defensive=False)
        last_pre = st_pre["daily_history"][-1]
        self.assertLess(last_pre["nav_full_tax"], last_pre["nav"])
        self.assertAlmostEqual(last_pre["nav_pre_def_tax"], last_pre["nav"], places=4)

        # Verify 35 bps regime-cut cost rate vs standard 15 bps sale
        reg_sell = next(t for t in st["trade_log"] if t["event_type"] == "REGIME_SHIFT" and t["side"] == "SELL")
        self.assertAlmostEqual(reg_sell["friction"], reg_sell["notional"] * 0.0035, places=2)

    def test_126_day_scheduled_rebalance_trigger(self):
        import datetime as _dt

        base = build_synthetic_parity_tape()[0]
        # Build 128 days where top rank completely flips on day 125 close
        days = []
        for i in range(128):
            d = json.loads(json.dumps(base))
            d["date"] = (_dt.date(2026, 1, 1) + _dt.timedelta(days=i)).isoformat()
            if i >= 125:
                # Reverse ranks strictly on day 125+ so day 126 T+1 signal picks bottom stocks
                for sym, row in d["rankings"].items():
                    row["rk9"] = -row["rk9"]
            days.append(d)

        st = run_synthetic_steps(days, initial_capital=500_000.0, preset="R3", slot_cap=False, liquid_sweep=True)
        reb = [d for d in st["decision_log"] if d["event_type"] == "REBALANCE_6M"]
        self.assertEqual(len(reb), 1)
        self.assertEqual(reb[0]["date"], days[126]["date"])
        # Held equities after rebalance must include STK35 (which flipped from rank 35 to rank 1)
        self.assertIn("STK35", {h["symbol"] for h in st["holdings"]})

    def test_idempotency_byte_identical_persistence(self):
        tape_days = build_synthetic_parity_tape()[:2]
        with tempfile.TemporaryDirectory() as tmp:
            t_path = Path(tmp) / "forward_tape.json"
            l_path = Path(tmp) / "paper_ledger.json"

            t1 = update_forward_tape(t_path, tape_days[0])
            update_paper_ledger(l_path, t1)
            bytes_tape_1 = t_path.read_bytes()
            bytes_ledger_1 = l_path.read_bytes()

            # Re-running with same date must leave both JSON files byte-identical
            t2 = update_forward_tape(t_path, tape_days[0])
            update_paper_ledger(l_path, t2)
            self.assertEqual(t_path.read_bytes(), bytes_tape_1)
            self.assertEqual(l_path.read_bytes(), bytes_ledger_1)

            # Advancing to Day 2 increments trading_days to 2
            t3 = update_forward_tape(t_path, tape_days[1])
            l3 = update_paper_ledger(l_path, t3)
            self.assertEqual(len(t3["days"]), 2)
            self.assertEqual(l3["accounts"]["R3_1L_SlotCap"]["summary"]["trading_days"], 2)

    def test_fifo_stcg_ltcg_exemption_and_shield_deferral(self):
        base = build_synthetic_parity_tape()[0]
        # User holds STK35 (outside top 30) bought 350 days before Day 0 at cost 10,000 (price 18,000 -> +80% gain)
        day0 = json.loads(json.dumps(base))
        day0["date"] = "2026-11-20"
        day1 = json.loads(json.dumps(base))
        day1["date"] = "2026-12-06"  # 366 days after 2025-12-05 -> LTCG matured!

        lots = [{"symbol": "STK34", "shares": 20, "buy_date": "2025-12-05", "buy_price": 100.0}]
        st = run_synthetic_steps(
            [day0, day1],
            initial_capital=250_000.0,
            preset="R3",
            slot_cap=False,
            liquid_sweep=True,
            respect_ltcg_shield=True,
            initial_lots=lots,
            starting_cash=100_000.0,
        )
        # Day 0 held STK34 shielded (days_held = 350 in 330..364 range)
        d0_sells = [t for t in st["trade_log"] if t["date"] == "2026-11-20" and t["symbol"] == "STK34" and t["side"] == "SELL"]
        self.assertEqual(len(d0_sells), 0)
        # Day 1 matured (>=365d) -> LTCG_SHIELD_MATURED sold STK34 under EQ_LTCG within ₹1.25L exempt!
        d1_sells = [t for t in st["trade_log"] if t["date"] == "2026-12-06" and t["symbol"] == "STK34" and t["side"] == "SELL"]
        self.assertEqual(len(d1_sells), 1)
        self.assertEqual(d1_sells[0]["event_type"], "LTCG_SHIELD_MATURED")
        self.assertEqual(d1_sells[0]["tax_type"], "EQ_LTCG")
        self.assertAlmostEqual(d1_sells[0]["tax_charged"], 0.0, places=2)  # gain < 1.25L FY exempt

    def test_stcg_and_loss_carryforward(self):
        day0 = build_synthetic_parity_tape()[0]
        day0["date"] = "2026-11-27"
        # Hold STK34 at loss (bought 50 days ago @ 1000, now ~695 -> STCL ~305/sh * 10 = 3050)
        # AND STK33 at gain (bought 40 days ago @ 100, now ~680 -> STCG ~580/sh * 10 = 5800)
        # Alphabetical sell order in rebalance: STK33 sold BEFORE STK34 OR STK34 loss carried forward into next sell!
        lots = [
            {"symbol": "STK32", "shares": 10, "buy_date": "2026-10-01", "buy_price": 1000.0},
            {"symbol": "STK33", "shares": 10, "buy_date": "2026-10-01", "buy_price": 100.0},
        ]
        st = run_synthetic_steps(
            [day0],
            initial_capital=100_000.0,
            preset="R3",
            slot_cap=False,
            liquid_sweep=True,
            initial_lots=lots,
            starting_cash=50_000.0,
        )
        # STK32 sells first alphabetically -> creates STCL carryforward; STK33 gain offsets against STCL before 20% tax!
        s32 = next(t for t in st["trade_log"] if t["symbol"] == "STK32" and t["side"] == "SELL")
        s33 = next(t for t in st["trade_log"] if t["symbol"] == "STK33" and t["side"] == "SELL")
        self.assertLess(s32["realized_pnl"], 0)
        expected_net = s33["realized_pnl"] + s32["realized_pnl"]
        self.assertAlmostEqual(s33["tax_charged"], max(0.0, expected_net) * 0.20, places=2)

    def test_halted_mtm_uses_last_valid_price_and_singular_day_grammar(self):
        import datetime as _dt

        tape = build_synthetic_parity_tape()
        d0 = json.loads(json.dumps(tape[0]))
        d0["date"] = "2026-11-02"
        # Deploy at initial price of STK01 (200) -> Day 1 rallies to 340 -> Day 2 STK01 missing (halted)
        d1 = json.loads(json.dumps(d0))
        d1["date"] = "2026-11-03"
        d1["prices"]["STK01"] = 340.0
        d2 = json.loads(json.dumps(d1))
        d2["date"] = "2026-11-04"
        del d2["prices"]["STK01"]

        st = run_synthetic_steps([d0, d1, d2], initial_capital=1_000_000.0, preset="R3", slot_cap=False, liquid_sweep=True)
        stk01_row = next(h for h in st["holdings"] if h["symbol"] == "STK01")
        # MTM on halted day MUST hold last valid price 340.0, never revert to buy price 200.0
        self.assertAlmostEqual(stk01_row["live_price"], 340.0, places=2)

        # Verify singular "1 trading day" grammar when next_rebalance_in_trading_days == 1 (day 125)
        days126 = []
        for i in range(126):
            d = json.loads(json.dumps(d0))
            d["date"] = (_dt.date(2026, 1, 1) + _dt.timedelta(days=i)).isoformat()
            days126.append(d)
        st126 = run_synthetic_steps(days126, initial_capital=250_000.0, preset="R3", slot_cap=False, liquid_sweep=True)
        last_hold = [x for x in st126["decision_log"] if x["event_type"] == "HOLD_CHECK"][-1]
        self.assertIn("1 trading day;", last_hold["summary"])

    def test_committed_json_files_valid_and_nan_free(self):
        def _reject_nan(val):
            raise ValueError(f"Invalid non-finite JSON token: {val}")

        tape_path = REPO_ROOT / "data" / "forward_tape.json"
        ledger_path = REPO_ROOT / "data" / "paper_ledger.json"
        if tape_path.exists():
            tape = json.loads(tape_path.read_text(), parse_constant=_reject_nan)
            self.assertGreaterEqual(len(tape["days"]), 1)
            for k, v in tape["days"][0]["prices"].items():
                self.assertTrue(math.isfinite(v) and v > 0, msg=k)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(), parse_constant=_reject_nan)
            self.assertIn("R3_1L_SlotCap", ledger["accounts"])
            self.assertIn("R3_10L_Standard", ledger["accounts"])

    def test_ai_feed_generation_and_static_endpoints(self):
        from src.ai_feed import build_ai_feeds

        live_path = REPO_ROOT / "data" / "live_state.json"
        ledger_path = REPO_ROOT / "data" / "paper_ledger.json"
        tape_path = REPO_ROOT / "data" / "forward_tape.json"
        if not (live_path.exists() and ledger_path.exists() and tape_path.exists()):
            self.skipTest("Live data payloads absent")

        live = json.loads(live_path.read_text())
        ledger = json.loads(ledger_path.read_text())
        tape = json.loads(tape_path.read_text())

        with tempfile.TemporaryDirectory() as tmpd:
            out = build_ai_feeds(Path(tmpd), live, ledger, tape)
            for k in ("llms", "llms_full", "ai_html", "robots", "sitemap"):
                self.assertTrue(out[k].exists(), msg=f"Missing AI endpoint {k}")
            llms_txt = out["llms"].read_text(encoding="utf-8")
            full_txt = out["llms_full"].read_text(encoding="utf-8")
            ai_html = out["ai_html"].read_text(encoding="utf-8")
            self.assertIn(live["as_of_date"], llms_txt)
            self.assertIn("R3_1L_SlotCap", full_txt)
            self.assertIn("Top-60 Nifty Midcap 150", full_txt)
            self.assertIn("15.61%", full_txt)
            self.assertNotIn("NaN", full_txt)
            self.assertIn("<article>", ai_html)
            self.assertIn("GPTBot", out["robots"].read_text(encoding="utf-8"))
            self.assertIn("llms-full.txt", out["sitemap"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
