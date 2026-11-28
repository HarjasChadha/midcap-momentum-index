"""Autonomous Paper Trading Bot, Forward Tape builder, and Indian FIFO tax engine.

Production module (pure Python + stdlib; zero numba/scipy imports) designed to maintain
1:1 deterministic parity with the browser JS Forward Replay Engine in `index.html`.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

COST_RATE = 0.0015  # 15 bps one-way brokerage + STT + slippage
MIN_TRADE_DELTA_INR = 500.0

LTCG_DAYS = 365
LTCG_EXEMPTION_INR = 125_000.0
STCG_RATE = 0.20
LTCG_RATE = 0.125
LIQUID_SLAB_RATE = 0.30
GOLD_STCG_RATE = 0.30
GOLD_LTCG_RATE = 0.125

ETF_SYMBOLS = [
    "GOLDBEES",
    "SILVERBEES",
    "EBBETF0431",
    "LTGILTBEES",
    "LIQUIDCASE",
    "LIQUIDBEES",
    "NIFTYBEES",
    "MON100",
    "MIDCAPIETF",
]
GOLD_ETFS = {"GOLDBEES", "SILVERBEES"}
LIQUID_ETFS = {"LIQUIDCASE", "LIQUIDBEES"}
ETF_SET = set(ETF_SYMBOLS)

PRESET_RULES: dict[str, dict[str, Any]] = {
    "R3": {
        "id": "R3",
        "label": "R3 · Risk-Adj 9M + MF4 + TrendGold150",
        "score_key": "rk9",
        "weighting": "equal",
        "rebalance_days": 126,
        "regime_key": "target_eq_mf4",
        "def_mode": "TREND_GOLD150_OR_LIQ",
        "trim_mode": "weakest_first",
    },
    "S3": {
        "id": "S3",
        "label": "S3 · Sortino 9M + MF4 + TrendGold150",
        "score_key": "so9",
        "weighting": "equal",
        "rebalance_days": 126,
        "regime_key": "target_eq_mf4",
        "def_mode": "TREND_GOLD150_OR_LIQ",
        "trim_mode": "weakest_first",
    },
    "R2": {
        "id": "R2",
        "label": "R2 · Risk-Adj 9M + DD15 + 50/50 Gold/Liq",
        "score_key": "rk9",
        "weighting": "equal",
        "rebalance_days": 126,
        "regime_key": "target_eq_dd15",
        "def_mode": "GOLD_50_LIQUID_50",
        "trim_mode": "weakest_first",
    },
    "R7": {
        "id": "R7",
        "label": "R7 · NormZ Risk-Adj 9M + DD15 + 50/50",
        "score_key": "rk9",
        "weighting": "norm_z",
        "rebalance_days": 126,
        "regime_key": "target_eq_dd15",
        "def_mode": "GOLD_50_LIQUID_50",
        "trim_mode": "weakest_first",
    },
    "F2": {
        "id": "F2",
        "label": "F2 · Legacy Raw 9M + DD12 + 50/50",
        "score_key": "r9",
        "weighting": "equal",
        "rebalance_days": 126,
        "regime_key": "target_eq_dd75",
        "def_mode": "GOLD_50_LIQUID_50",
        "trim_mode": "weakest_first",
    },
    "F4": {
        "id": "F4",
        "label": "F4 · Raw 12M 12M-Reb + DD12 + 50/50",
        "score_key": "r12",
        "weighting": "equal",
        "rebalance_days": 252,
        "regime_key": "target_eq_dd50",
        "def_mode": "GOLD_50_LIQUID_50",
        "trim_mode": "weakest_first",
    },
}


def sanitize_json(obj: Any, decimals: int = 4) -> Any:
    """Recursively convert floats/None into strict NaN-free JSON values."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return round(obj, decimals) if math.isfinite(obj) else None
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        return sanitize_json(obj.item(), decimals=decimals)
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v, decimals=decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(x, decimals=decimals) for x in obj]
    return str(obj)


def _finite(val: Any) -> float | None:
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _days_between(start_date: str, end_date: str) -> int:
    try:
        return (date.fromisoformat(end_date[:10]) - date.fromisoformat(start_date[:10])).days
    except ValueError:
        return 0


def _fy_for(date_str: str) -> str:
    y = int(date_str[:4])
    m = int(date_str[5:7])
    return str(y if m >= 4 else y - 1)


def _asset_class(symbol: str) -> str:
    if symbol in GOLD_ETFS:
        return "GOLD_ETF"
    if symbol in LIQUID_ETFS:
        return "LIQUID_ETF"
    if symbol in ETF_SET:
        return "OTHER_ETF"
    return "EQUITY"


def _compute_sleeve_weights(
    ordered_syms: list[str],
    scores_map: dict[str, float],
    all_valid_syms: list[str],
    top_n: int,
    weighting: str,
) -> dict[str, float]:
    selected = ordered_syms[:top_n]
    if not selected:
        return {}
    if weighting == "norm_z":
        vals = [scores_map[s] for s in all_valid_syms]
        n = len(vals)
        mean = sum(vals) / n if n > 0 else 0.0
        var = sum((x - mean) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var) if var > 0 else 0.0
        raw_w: dict[str, float] = {}
        for sym in selected:
            z = (scores_map[sym] - mean) / std if std > 1e-12 else 0.0
            raw_w[sym] = (1.0 + z) if z >= 0.0 else (1.0 / (1.0 - z))
        tot = sum(raw_w.values())
        return {sym: (w / tot if tot > 0 else 1.0 / len(selected)) for sym, w in raw_w.items()}
    w_eq = 1.0 / len(selected)
    return {sym: w_eq for sym in selected}


def compute_target_weights(
    rankings: dict[str, dict[str, Any]],
    prices: dict[str, Any],
    preset: str,
    equity_budget: float,
    slot_cap: bool = False,
) -> dict[str, float]:
    """Compute Top-20/Top-30 blended weights (exact parity with src/momentum.py)."""
    rule = PRESET_RULES[preset]
    score_key = rule["score_key"]
    weighting = rule["weighting"]

    base_scores: dict[str, float] = {}
    base_prices: dict[str, float] = {}
    for sym, row in (rankings or {}).items():
        if sym in ETF_SET:
            continue
        sc = _finite(row.get(score_key) if isinstance(row, dict) else None)
        px = _finite((prices or {}).get(sym))
        if sc is not None and px is not None and px > 0:
            base_scores[sym] = sc
            base_prices[sym] = px

    if not base_scores:
        return {}

    ordered_all = sorted(base_scores.keys(), key=lambda s: (-base_scores[s], s))
    excluded: set[str] = set()
    max_iters = 60 if slot_cap and equity_budget > 0 else 1

    final_weights: dict[str, float] = {}
    for _ in range(max_iters):
        eligible = [s for s in ordered_all if s not in excluded]
        if not eligible:
            break
        w20 = _compute_sleeve_weights(eligible, base_scores, eligible, 20, weighting)
        w30 = _compute_sleeve_weights(eligible, base_scores, eligible, 30, weighting)
        blend: dict[str, float] = {}
        for sym in eligible[:30]:
            w = 0.5 * w20.get(sym, 0.0) + 0.5 * w30.get(sym, 0.0)
            if w > 0:
                blend[sym] = w
        tot = sum(blend.values())
        if tot > 0:
            blend = {sym: w / tot for sym, w in blend.items()}
        final_weights = blend
        if not slot_cap or equity_budget <= 0:
            break
        unaffordable = [sym for sym, w in blend.items() if base_prices[sym] > equity_budget * w]
        if not unaffordable or len(eligible) <= 30:
            break
        drop_sym = max(unaffordable, key=lambda s: (base_prices[s], s))
        excluded.add(drop_sym)

    return final_weights


def _resolve_target_allocation(regime: dict[str, Any], preset: str, unhedged: bool = False) -> tuple[float, float, float]:
    """Return (eq_frac, gold_frac, liquid_frac) target fractions for a preset and regime dict."""
    if unhedged:
        return 1.0, 0.0, 0.0
    rule = PRESET_RULES[preset]
    eq_frac = _finite(regime.get(rule["regime_key"]))
    if eq_frac is None:
        eq_frac = _finite(regime.get("target_eq_mf4"))
    if eq_frac is None:
        eq_frac = 1.0
    eq_frac = max(0.0, min(1.0, eq_frac))
    def_frac = max(0.0, 1.0 - eq_frac)
    if def_frac <= 1e-6:
        return 1.0, 0.0, 0.0
    if rule["def_mode"] == "GOLD_50_LIQUID_50":
        return eq_frac, 0.5 * def_frac, 0.5 * def_frac
    active_def = str(regime.get("active_def_etf", "LIQUIDCASE"))
    if active_def == "GOLDBEES":
        return eq_frac, def_frac, 0.0
    return eq_frac, 0.0, def_frac


def _posture_label(eq_frac: float) -> str:
    if eq_frac >= 0.99:
        return "RISK_ON"
    if eq_frac >= 0.50:
        return "CAUTIOUS"
    return "DEFENSIVE"


def _compute_positions_summary(lots: list[dict[str, Any]], prices: dict[str, Any], as_of_date: str) -> dict[str, Any]:
    by_sym: dict[str, dict[str, Any]] = {}
    eq_val = 0.0
    gold_val = 0.0
    liq_val = 0.0
    for lot in lots:
        sh = float(lot["shares"])
        if sh <= 1e-6:
            continue
        sym = str(lot["symbol"])
        px = _finite((prices or {}).get(sym))
        if px is None or px <= 0:
            px = float(lot["buy_price"])
        mv = sh * px
        ac = _asset_class(sym)
        if ac == "GOLD_ETF":
            gold_val += mv
        elif ac == "LIQUID_ETF":
            liq_val += mv
        else:
            eq_val += mv
        entry = by_sym.setdefault(
            sym,
            {
                "symbol": sym,
                "asset_class": ac,
                "shares": 0.0,
                "cost_basis_total": 0.0,
                "market_value": 0.0,
                "oldest_buy_date": lot["buy_date"],
                "newest_buy_date": lot["buy_date"],
                "shielded_shares": 0.0,
            },
        )
        entry["shares"] += sh
        entry["cost_basis_total"] += sh * float(lot["buy_price"])
        entry["market_value"] += mv
        if lot["buy_date"] < entry["oldest_buy_date"]:
            entry["oldest_buy_date"] = lot["buy_date"]
        if lot["buy_date"] > entry["newest_buy_date"]:
            entry["newest_buy_date"] = lot["buy_date"]
        if lot.get("shield_defer"):
            entry["shielded_shares"] += sh
    return {
        "by_sym": by_sym,
        "equity_val": eq_val,
        "gold_val": gold_val,
        "liquid_val": liq_val,
    }


def _build_holdings_rows(state: dict[str, Any], prices: dict[str, Any], as_of_date: str) -> list[dict[str, Any]]:
    pos = _compute_positions_summary(state["lots"], prices, as_of_date)
    basket = state.get("basket_weights") or {}
    rows: list[dict[str, Any]] = []
    for sym, info in pos["by_sym"].items():
        sh = info["shares"]
        avg_buy = info["cost_basis_total"] / sh if sh > 0 else 0.0
        px = _finite((prices or {}).get(sym)) or avg_buy
        mv = sh * px
        pnl = mv - info["cost_basis_total"]
        pnl_pct = (pnl / info["cost_basis_total"] * 100.0) if info["cost_basis_total"] > 0 else 0.0
        days_held = _days_between(info["oldest_buy_date"], as_of_date)
        ac = info["asset_class"]
        if ac in {"GOLD_ETF", "LIQUID_ETF"}:
            bucket = "Defensive"
        else:
            w = basket.get(sym, 0.0)
            bucket = "Top 20" if w > 0.025 else ("Ranks 21-30" if w > 0 else "Legacy / Shield")
        ltcg_rem = max(0, LTCG_DAYS - days_held)
        rows.append(
            {
                "symbol": sym,
                "asset_class": ac,
                "bucket": bucket,
                "shares": round(sh, 4),
                "avg_buy": round(avg_buy, 2),
                "live_price": round(px, 2),
                "value": round(mv, 2),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "oldest_buy_date": info["oldest_buy_date"],
                "days_held": days_held,
                "ltcg_days_remaining": ltcg_rem,
                "ltcg_eligible": days_held >= LTCG_DAYS,
                "shield_deferred": info["shielded_shares"] > 1e-6,
            }
        )
    rows.sort(key=lambda r: (0 if r["asset_class"] == "EQUITY" else 1, -r["value"], r["symbol"]))
    return rows


def _apply_fifo_sell(
    state: dict[str, Any],
    sym: str,
    sell_qty: float,
    exec_px: float,
    exec_date: str,
    event_type: str,
    reason: str,
    allow_shielded: bool = True,
) -> float:
    """Sell up to sell_qty of `sym` using FIFO, recording tax, fees, and trades. Returns gross shares sold."""
    if sell_qty <= 1e-6 or exec_px <= 0:
        return 0.0

    lots = state["lots"]
    remaining = float(sell_qty)
    fy = _fy_for(exec_date)
    fy_map = state["tax_state"]["fy"].setdefault(fy, {"ltcg_exempt_used": 0.0, "stcg_tax": 0.0, "ltcg_tax": 0.0, "stcg_pnl": 0.0, "ltcg_pnl": 0.0})

    sold_total = 0.0
    total_pnl = 0.0
    weighted_days = 0.0
    eq_tax_before = float(state["tax_state"]["realized_eq_tax_cum"])
    def_tax_before = float(state["tax_state"]["estimated_def_tax_cum"])
    new_lots: list[dict[str, Any]] = []
    ac = _asset_class(sym)

    for lot in lots:
        if lot["symbol"] != sym or remaining <= 1e-6 or (not allow_shielded and lot.get("shield_defer")):
            new_lots.append(lot)
            continue
        lot_sh = float(lot["shares"])
        take = min(lot_sh, remaining)
        if take > 1e-6:
            days_held = _days_between(lot["buy_date"], exec_date)
            lot_pnl = (exec_px - float(lot["buy_price"])) * take
            sold_total += take
            total_pnl += lot_pnl
            weighted_days += days_held * take
            remaining -= take
            left = lot_sh - take

            # Record tax per lot slice according to asset class & Indian tax rules
            if ac == "EQUITY":
                if days_held < LTCG_DAYS:
                    fy_map["stcg_pnl"] += lot_pnl
                    if lot_pnl < 0:
                        state["tax_state"]["stcl_carry"] += abs(lot_pnl)
                    else:
                        offset = min(lot_pnl, state["tax_state"]["stcl_carry"])
                        state["tax_state"]["stcl_carry"] -= offset
                        taxable = lot_pnl - offset
                        tax = taxable * STCG_RATE
                        fy_map["stcg_tax"] += tax
                        state["tax_state"]["realized_eq_tax_cum"] += tax
                        state["cash"] -= tax
                else:
                    fy_map["ltcg_pnl"] += lot_pnl
                    if lot_pnl < 0:
                        state["tax_state"]["ltcl_carry"] += abs(lot_pnl)
                    else:
                        off_lt = min(lot_pnl, state["tax_state"]["ltcl_carry"])
                        state["tax_state"]["ltcl_carry"] -= off_lt
                        rem_gain = lot_pnl - off_lt
                        off_st = min(rem_gain, state["tax_state"]["stcl_carry"])
                        state["tax_state"]["stcl_carry"] -= off_st
                        rem_gain -= off_st
                        ex_left = max(0.0, LTCG_EXEMPTION_INR - fy_map["ltcg_exempt_used"])
                        ex_used = min(rem_gain, ex_left)
                        fy_map["ltcg_exempt_used"] += ex_used
                        taxable = max(0.0, rem_gain - ex_used)
                        tax = taxable * LTCG_RATE
                        fy_map["ltcg_tax"] += tax
                        state["tax_state"]["realized_eq_tax_cum"] += tax
                        state["cash"] -= tax
            elif ac == "LIQUID_ETF":
                def_tax = max(0.0, lot_pnl) * LIQUID_SLAB_RATE
                state["tax_state"]["estimated_def_tax_cum"] += def_tax
            else:  # GOLD_ETF / OTHER_ETF
                rate = GOLD_LTCG_RATE if days_held >= LTCG_DAYS else GOLD_STCG_RATE
                def_tax = max(0.0, lot_pnl) * rate
                state["tax_state"]["estimated_def_tax_cum"] += def_tax

            if left > 1e-6:
                upd = dict(lot)
                upd["shares"] = round(left, 6)
                new_lots.append(upd)
        else:
            new_lots.append(lot)

    if sold_total <= 1e-6:
        return 0.0

    state["lots"] = new_lots
    gross = sold_total * exec_px
    fee = gross * COST_RATE
    state["cash"] += gross - fee
    state["total_fees_cum"] += fee

    avg_days = int(round(weighted_days / sold_total))
    tax_type = "EQ_LTCG" if (ac == "EQUITY" and avg_days >= LTCG_DAYS) else (
        "EQ_STCG" if ac == "EQUITY" else ("DEBT_50AA_30%" if ac == "LIQUID_ETF" else ("GOLD_12.5%" if avg_days >= LTCG_DAYS else "GOLD_30%"))
    )
    # Approximate trade-level tax delta by comparing pre/post isn't needed if summed cleanly
    eq_tax_delta = float(state["tax_state"]["realized_eq_tax_cum"]) - eq_tax_before
    def_tax_delta = float(state["tax_state"]["estimated_def_tax_cum"]) - def_tax_before
    state["trade_log"].append(
        sanitize_json(
            {
                "date": exec_date,
                "event_type": event_type,
                "symbol": sym,
                "side": "SELL",
                "qty": round(sold_total, 4),
                "price": exec_px,
                "notional": gross,
                "friction": fee,
                "holding_days": avg_days,
                "realized_pnl": total_pnl,
                "tax_type": tax_type,
                "tax_charged": eq_tax_delta,
                "def_tax_est": def_tax_delta,
                "reason": reason,
            },
            decimals=4,
        )
    )
    return sold_total


def _buy_lot(
    state: dict[str, Any],
    sym: str,
    want_shares: int,
    exec_px: float,
    exec_date: str,
    event_type: str,
    reason: str,
) -> int:
    if want_shares <= 0 or exec_px <= 0:
        return 0
    unit_cost = exec_px * (1.0 + COST_RATE)
    affordable = int(max(0.0, state["cash"]) // unit_cost)
    qty = min(int(want_shares), affordable)
    if qty <= 0:
        return 0
    gross = qty * exec_px
    fee = gross * COST_RATE
    state["cash"] -= gross + fee
    state["total_fees_cum"] += fee
    state["_lot_seq"] += 1
    state["lots"].append(
        {
            "id": f"L{state['_lot_seq']}",
            "symbol": sym,
            "asset_class": _asset_class(sym),
            "shares": float(qty),
            "buy_date": exec_date,
            "buy_price": round(float(exec_px), 4),
            "shield_defer": False,
        }
    )
    state["trade_log"].append(
        sanitize_json(
            {
                "date": exec_date,
                "event_type": event_type,
                "symbol": sym,
                "side": "BUY",
                "qty": qty,
                "price": exec_px,
                "notional": gross,
                "friction": fee,
                "holding_days": 0,
                "realized_pnl": 0.0,
                "tax_type": "OPEN_LOT",
                "tax_charged": 0.0,
                "reason": reason,
            },
            decimals=4,
        )
    )
    return qty


def _sync_defensive_and_sweep(
    state: dict[str, Any],
    prices: dict[str, Any],
    exec_date: str,
    target_gold_inr: float,
    target_liq_inr: float,
    event_type: str,
    reason: str,
    liquid_sweep: bool,
) -> None:
    """Rebalance GOLDBEES & LIQUIDCASE (and sweep leftover cash to LIQUIDCASE if enabled)."""
    pos = _compute_positions_summary(state["lots"], prices, exec_date)["by_sym"]
    gold_px = _finite((prices or {}).get("GOLDBEES")) or 0.0
    liq_px = _finite((prices or {}).get("LIQUIDCASE")) or 0.0

    # First sell any excess GOLD or LIQUID so cash is freed
    cur_gold_sh = pos.get("GOLDBEES", {}).get("shares", 0.0)
    cur_gold_val = cur_gold_sh * gold_px
    tgt_gold_sh = int(target_gold_inr // gold_px) if (gold_px > 0 and target_gold_inr > MIN_TRADE_DELTA_INR) else 0
    if cur_gold_sh > 0 and (tgt_gold_sh == 0 or (cur_gold_val - target_gold_inr) > MIN_TRADE_DELTA_INR):
        sell_sh = cur_gold_sh if tgt_gold_sh == 0 else min(cur_gold_sh, math.ceil((cur_gold_val - target_gold_inr) / gold_px))
        if sell_sh > 0:
            _apply_fifo_sell(state, "GOLDBEES", sell_sh, gold_px, exec_date, event_type, reason)

    cur_liq_sh = pos.get("LIQUIDCASE", {}).get("shares", 0.0)
    cur_liq_val = cur_liq_sh * liq_px
    # Total desired liquid includes explicit defensive liquid allocation + idle cash reserve if sweep
    if not liquid_sweep:
        tgt_liq_sh = int(target_liq_inr // liq_px) if (liq_px > 0 and target_liq_inr > MIN_TRADE_DELTA_INR) else 0
        if cur_liq_sh > 0 and (tgt_liq_sh == 0 or (cur_liq_val - target_liq_inr) > MIN_TRADE_DELTA_INR):
            sell_sh = cur_liq_sh if tgt_liq_sh == 0 else min(cur_liq_sh, math.ceil((cur_liq_val - target_liq_inr) / liq_px))
            if sell_sh > 0:
                _apply_fifo_sell(state, "LIQUIDCASE", sell_sh, liq_px, exec_date, event_type, reason)
    else:
        # When cash is negative (e.g. from tax) or we need cash for GOLDBEES buy, trim LIQUIDCASE if needed
        need_cash = max(0.0, (target_gold_inr - cur_gold_val) * (1.0 + COST_RATE) - state["cash"])
        if cur_liq_sh > 0 and liq_px > 0 and need_cash > MIN_TRADE_DELTA_INR:
            sell_sh = min(cur_liq_sh, math.ceil(need_cash / (liq_px * (1.0 - COST_RATE))))
            if sell_sh > 0:
                _apply_fifo_sell(state, "LIQUIDCASE", sell_sh, liq_px, exec_date, event_type, reason)

    # Buy GOLDBEES up to target_gold_inr
    pos2 = _compute_positions_summary(state["lots"], prices, exec_date)["by_sym"]
    cur_gold_val2 = pos2.get("GOLDBEES", {}).get("shares", 0.0) * gold_px
    if gold_px > 0 and (target_gold_inr - cur_gold_val2) > MIN_TRADE_DELTA_INR:
        want_sh = int((target_gold_inr - cur_gold_val2) // (gold_px * (1.0 + COST_RATE)))
        if want_sh > 0:
            _buy_lot(state, "GOLDBEES", want_sh, gold_px, exec_date, event_type, reason)

    # Sweep residual cash (or buy target LIQUIDCASE allocation) into LIQUIDCASE
    if liq_px > 0:
        if liquid_sweep:
            if state["cash"] >= liq_px * (1.0 + COST_RATE):
                want_sh = int(state["cash"] // (liq_px * (1.0 + COST_RATE)))
                if want_sh > 0:
                    _buy_lot(state, "LIQUIDCASE", want_sh, liq_px, exec_date, event_type, f"{reason} + LIQUIDCASE sweep")
        else:
            pos3 = _compute_positions_summary(state["lots"], prices, exec_date)["by_sym"]
            cur_liq_val3 = pos3.get("LIQUIDCASE", {}).get("shares", 0.0) * liq_px
            if (target_liq_inr - cur_liq_val3) > MIN_TRADE_DELTA_INR:
                want_sh = int((target_liq_inr - cur_liq_val3) // (liq_px * (1.0 + COST_RATE)))
                if want_sh > 0:
                    _buy_lot(state, "LIQUIDCASE", want_sh, liq_px, exec_date, event_type, reason)


def _settle_negative_cash(state: dict[str, Any], prices: dict[str, Any], exec_date: str) -> None:
    """If tax deducted cash slightly below zero, liquidate LIQUIDCASE / GOLDBEES or weakest stock."""
    if state["cash"] >= 0:
        return
    for def_sym in ("LIQUIDCASE", "LIQUIDBEES", "GOLDBEES"):
        if state["cash"] >= 0:
            break
        px = _finite((prices or {}).get(def_sym)) or 0.0
        held = sum(l["shares"] for l in state["lots"] if l["symbol"] == def_sym)
        if held > 0 and px > 0:
            need = min(held, math.ceil(abs(state["cash"]) / (px * (1.0 - COST_RATE))))
            _apply_fifo_sell(state, def_sym, need, px, exec_date, "TAX_SETTLE", "Cover realized FIFO tax liability")


def _rebalance_to_basket(
    state: dict[str, Any],
    target_weights: dict[str, float],
    target_eq_inr: float,
    prices: dict[str, Any],
    exec_date: str,
    event_type: str,
    reason: str,
    respect_ltcg_shield: bool = False,
) -> None:
    """Sell excess / non-target equities first (honoring LTCG shield if enabled), then buy underweights."""
    pos = _compute_positions_summary(state["lots"], prices, exec_date)["by_sym"]
    held_eq_syms = [s for s, info in pos.items() if info["asset_class"] == "EQUITY"]

    # Step 1: sells (full exits of non-target or trims of overweights)
    for sym in sorted(held_eq_syms):
        px = _finite((prices or {}).get(sym)) or pos[sym]["cost_basis_total"] / max(1e-6, pos[sym]["shares"])
        cur_sh = pos[sym]["shares"]
        cur_val = cur_sh * px
        tgt_w = target_weights.get(sym, 0.0)
        tgt_val = target_eq_inr * tgt_w
        if tgt_w <= 0:
            sell_want = cur_sh
        elif (cur_val - tgt_val) > MIN_TRADE_DELTA_INR and px > 0:
            tgt_sh = int(tgt_val // px)
            sell_want = max(0.0, cur_sh - tgt_sh)
        else:
            sell_want = 0.0
        if sell_want <= 1e-6:
            continue

        # Respect 330-364d LTCG shield if flagged
        if respect_ltcg_shield:
            allow_now = 0.0
            for lot in state["lots"]:
                if lot["symbol"] != sym:
                    continue
                d_held = _days_between(lot["buy_date"], exec_date)
                if 330 <= d_held < LTCG_DAYS:
                    lot["shield_defer"] = True
                else:
                    allow_now += lot["shares"]
            sell_qty = min(sell_want, allow_now)
            if sell_qty > 1e-6:
                _apply_fifo_sell(state, sym, sell_qty, px, exec_date, event_type, reason, allow_shielded=False)
        else:
            _apply_fifo_sell(state, sym, sell_want, px, exec_date, event_type, reason, allow_shielded=True)

    # Liquidate defensive ETFs down if needed before equity buys? Handled in caller when regime adds eq.
    # Step 2: buys into underweighted target stocks (ordered highest weight -> symbol)
    pos_after_sell = _compute_positions_summary(state["lots"], prices, exec_date)["by_sym"]
    buy_list = sorted(target_weights.keys(), key=lambda s: (-target_weights[s], s))
    for sym in buy_list:
        px = _finite((prices or {}).get(sym))
        if px is None or px <= 0:
            continue
        cur_val = pos_after_sell.get(sym, {}).get("shares", 0.0) * px
        tgt_val = target_eq_inr * target_weights[sym]
        diff = tgt_val - cur_val
        if diff > MIN_TRADE_DELTA_INR:
            want_sh = int(diff // (px * (1.0 + COST_RATE)))
            if want_sh > 0:
                _buy_lot(state, sym, want_sh, px, exec_date, event_type, reason)


def initialize_account_state(
    account_id: str,
    label: str,
    initial_capital: float,
    preset: str = "R3",
    slot_cap: bool = True,
    liquid_sweep: bool = True,
    respect_ltcg_shield: bool = False,
    unhedged: bool = False,
    do_nothing: bool = False,
    initial_lots: list[dict[str, Any]] | None = None,
    starting_cash: float | None = None,
) -> dict[str, Any]:
    """Create a fresh account / tracked view state container."""
    lots_copy: list[dict[str, Any]] = []
    seq = 0
    for lot in initial_lots or []:
        seq += 1
        sym = str(lot["symbol"]).upper().strip()
        sh = float(lot["shares"])
        if sh <= 0:
            continue
        lots_copy.append(
            {
                "id": f"L{seq}",
                "symbol": sym,
                "asset_class": _asset_class(sym),
                "shares": round(sh, 6),
                "buy_date": str(lot["buy_date"])[:10],
                "buy_price": round(float(lot["buy_price"]), 4),
                "shield_defer": bool(lot.get("shield_defer", False)),
            }
        )
    return {
        "account_id": account_id,
        "label": label,
        "preset": preset,
        "slot_cap": bool(slot_cap),
        "liquid_sweep": bool(liquid_sweep),
        "respect_ltcg_shield": bool(respect_ltcg_shield),
        "unhedged": bool(unhedged),
        "do_nothing": bool(do_nothing),
        "initial_capital": round(float(initial_capital), 2),
        "inception_date": None,
        "last_synced_date": None,
        "status": "ACTIVE",
        "ended_date": None,
        "cash": float(initial_capital if starting_cash is None else starting_cash),
        "total_fees_cum": 0.0,
        "peak_nav": 0.0,
        "trading_days_count": 0,
        "trading_days_since_rebalance": 0,
        "last_rebalance_date": None,
        "active_eq_frac": 1.0,
        "active_def_etf": "LIQUIDCASE",
        "basket_weights": {},
        "_lot_seq": seq,
        "lots": lots_copy,
        "tax_state": {
            "fy": {},
            "stcl_carry": 0.0,
            "ltcl_carry": 0.0,
            "realized_eq_tax_cum": 0.0,
            "estimated_def_tax_cum": 0.0,
        },
        "ew_inception_val": None,
        "daily_history": [],
        "decision_log": [],
        "trade_log": [],
        "holdings": [],
    }


def step_portfolio(
    state: dict[str, Any],
    tape_entry: dict[str, Any],
    prev_tape_entry: dict[str, Any] | None = None,
    unhedged_nav_today: float | None = None,
    bench_ew_nav_today: float | None = None,
) -> dict[str, Any]:
    """Advance `state` by one trading day using strict T+1 causality (day-0 deploys immediately).

    Idempotent if `tape_entry["date"] <= state["last_synced_date"]`.
    """
    exec_date = str(tape_entry["date"])
    if state.get("status") == "ENDED":
        return state
    if state.get("last_synced_date") and exec_date <= state["last_synced_date"]:
        return state

    prices = tape_entry.get("prices") or {}
    regime_today = tape_entry.get("regime") or {}
    preset = state["preset"]
    rule = PRESET_RULES[preset]
    reb_period = int(rule["rebalance_days"])
    unhedged = bool(state.get("unhedged", False))
    do_nothing = bool(state.get("do_nothing", False))

    if state["inception_date"] is None:
        state["inception_date"] = exec_date
        ew_val = _finite(regime_today.get("ew_index_val"))
        state["ew_inception_val"] = ew_val if (ew_val and ew_val > 0) else 100.0

    # Step 1: Adjust FIFO lots for any ex-date SPLIT / BONUS corporate actions on exec_date
    ca_events_logged: list[str] = []
    for ev in tape_entry.get("corp_actions") or []:
        sym = str(ev.get("symbol", "")).upper()
        factor = _finite(ev.get("factor"))
        atype = str(ev.get("type", "SPLIT")).upper()
        if not sym or factor is None or factor <= 0 or abs(factor - 1.0) < 1e-6:
            continue
        touched = False
        for lot in state["lots"]:
            if lot["symbol"] == sym and lot["shares"] > 0:
                lot["shares"] = round(float(lot["shares"]) / factor, 6)
                lot["buy_price"] = round(float(lot["buy_price"]) * factor, 4)
                touched = True
        if touched:
            mult = round(1.0 / factor, 4)
            msg = f"{atype} on {sym} (factor {factor}): FIFO lot shares multiplied by {mult}x and cost basis adjusted cleanly."
            ca_events_logged.append(f"{atype} {sym} {mult}x")
            state["decision_log"].append({"date": exec_date, "event_type": "CORPORATE_ACTION", "summary": msg})
            state["trade_log"].append(
                {
                    "date": exec_date,
                    "event_type": "CORPORATE_ACTION",
                    "symbol": sym,
                    "side": "ADJUST",
                    "qty": mult,
                    "price": _finite(prices.get(sym)),
                    "notional": 0.0,
                    "friction": 0.0,
                    "holding_days": 0,
                    "realized_pnl": 0.0,
                    "tax_type": "NONE",
                    "tax_charged": 0.0,
                    "reason": msg,
                }
            )

    pos_start = _compute_positions_summary(state["lots"], prices, exec_date)
    pre_trade_nav = state["cash"] + pos_start["equity_val"] + pos_start["gold_val"] + pos_start["liquid_val"]

    is_inception = prev_tape_entry is None and len(state["daily_history"]) == 0
    action_tags: list[str] = list(ca_events_logged)

    # Strictly causal T+1: Day >= 1 observes prev_tape_entry (Day t-1 close), Day 0 observes tape_entry
    sig_entry = tape_entry if is_inception else prev_tape_entry
    assert sig_entry is not None
    sig_regime = sig_entry.get("regime") or {}
    sig_rankings = sig_entry.get("rankings") or {}

    eq_frac, gold_frac, liq_frac = _resolve_target_allocation(sig_regime, preset, unhedged=unhedged)
    sig_def_etf = "50_50" if rule["def_mode"] == "GOLD_50_LIQUID_50" else ("GOLDBEES" if gold_frac > liq_frac else "LIQUIDCASE")

    if do_nothing:
        state["trading_days_count"] += 1
        if is_inception:
            state["last_rebalance_date"] = exec_date
            state["decision_log"].append(
                {
                    "date": exec_date,
                    "event_type": "INCEPTION_HOLD",
                    "summary": "Started Frozen Do-Nothing baseline holding Day-0 universe lots untouched (only corporate-action lot splits adjust automatically).",
                }
            )
            action_tags.append("INCEPTION_HOLD")
        else:
            state["decision_log"].append(
                {
                    "date": exec_date,
                    "event_type": "HOLD_CHECK",
                    "summary": f"Frozen Do-Nothing baseline mark-to-market ({exec_date}); zero trades executed.",
                }
            )
    elif is_inception:
        # Day-0 Initial Deployment / Rebalance
        state["trading_days_count"] = 1
        state["trading_days_since_rebalance"] = 0
        state["last_rebalance_date"] = exec_date
        eq_budget = pre_trade_nav * eq_frac
        target_w = compute_target_weights(sig_rankings, prices, preset, eq_budget, slot_cap=state["slot_cap"])
        state["basket_weights"] = dict(target_w)
        evt = "INCEPTION_REBALANCE" if state["lots"] else "INCEPTION_BUY"
        reason = (
            f"Day-0 deployment under {preset} ({'SlotCap ON' if state['slot_cap'] else 'Standard'}): "
            f"{_posture_label(eq_frac)} posture ({eq_frac*100:.0f}% Eq / {(1-eq_frac)*100:.0f}% "
            f"{sig_def_etf if gold_frac + liq_frac > 0 else 'Def'}) across {len(target_w)} slots."
        )
        _rebalance_to_basket(
            state,
            target_w,
            eq_budget,
            prices,
            exec_date,
            evt,
            reason,
            respect_ltcg_shield=state["respect_ltcg_shield"],
        )
        # Recompute post-eq NAV so gold/liquid targets remain aligned
        pos_mid = _compute_positions_summary(state["lots"], prices, exec_date)
        mid_nav = state["cash"] + pos_mid["equity_val"] + pos_mid["gold_val"] + pos_mid["liquid_val"]
        _sync_defensive_and_sweep(
            state,
            prices,
            exec_date,
            mid_nav * gold_frac,
            mid_nav * liq_frac,
            evt,
            reason,
            liquid_sweep=state["liquid_sweep"],
        )
        _settle_negative_cash(state, prices, exec_date)
        state["active_eq_frac"] = eq_frac
        state["active_def_etf"] = sig_def_etf
        state["decision_log"].append({"date": exec_date, "event_type": evt, "summary": reason})
        action_tags.append(evt)
    else:
        # Subsequent Day T+1 transitions
        state["trading_days_count"] += 1
        state["trading_days_since_rebalance"] += 1
        is_rebalance_due = state["trading_days_since_rebalance"] >= reb_period

        # 1. Check any 330-364d shielded lots that just crossed >= 365 days today!
        matured_syms: list[str] = []
        if state["respect_ltcg_shield"]:
            target_set = set(state.get("basket_weights") or {})
            for sym in sorted({l["symbol"] for l in state["lots"] if l.get("shield_defer")}):
                if sym in target_set:
                    for lot in state["lots"]:
                        if lot["symbol"] == sym:
                            lot["shield_defer"] = False
                    continue
                px = _finite(prices.get(sym)) or 0.0
                if px <= 0:
                    continue
                matured_qty = sum(
                    l["shares"]
                    for l in state["lots"]
                    if l["symbol"] == sym and l.get("shield_defer") and _days_between(l["buy_date"], exec_date) >= LTCG_DAYS
                )
                if matured_qty > 1e-6:
                    for lot in state["lots"]:
                        if lot["symbol"] == sym and _days_between(lot["buy_date"], exec_date) >= LTCG_DAYS:
                            lot["shield_defer"] = False
                    msg = f"LTCG Shield matured (>=365d) on {sym}: selling deferred lot cleanly at LTCG rate."
                    _apply_fifo_sell(state, sym, matured_qty, px, exec_date, "LTCG_SHIELD_MATURED", msg, allow_shielded=False)
                    matured_syms.append(sym)
                    state["decision_log"].append({"date": exec_date, "event_type": "LTCG_SHIELD_MATURED", "summary": msg})
            if matured_syms:
                action_tags.append("LTCG_SHIELD_MATURED")
                if not is_rebalance_due:
                    target_eq_inr = pre_trade_nav * eq_frac
                    if eq_frac > 0 and state.get("basket_weights"):
                        _rebalance_to_basket(
                            state,
                            state["basket_weights"],
                            target_eq_inr,
                            prices,
                            exec_date,
                            "LTCG_SHIELD_MATURED",
                            "Reallocate matured LTCG proceeds into target equity basket",
                            respect_ltcg_shield=state["respect_ltcg_shield"],
                        )
                    pos_mid = _compute_positions_summary(state["lots"], prices, exec_date)
                    mid_nav = state["cash"] + pos_mid["equity_val"] + pos_mid["gold_val"] + pos_mid["liquid_val"]
                    _sync_defensive_and_sweep(
                        state,
                        prices,
                        exec_date,
                        mid_nav * gold_frac,
                        mid_nav * liq_frac,
                        "LTCG_SHIELD_MATURED",
                        "Sweep LTCG matured residual proceeds",
                        liquid_sweep=state["liquid_sweep"],
                    )
                    _settle_negative_cash(state, prices, exec_date)

        eq_shifted = abs(eq_frac - float(state["active_eq_frac"])) > 1e-6
        def_flipped = (sig_def_etf != state["active_def_etf"]) and (1.0 - eq_frac > 1e-6)

        if is_rebalance_due:
            evt = "REBALANCE_6M" if reb_period <= 150 else "REBALANCE_12M"
            state["trading_days_since_rebalance"] = 0
            state["last_rebalance_date"] = exec_date
            eq_budget = pre_trade_nav * eq_frac
            new_w = compute_target_weights(sig_rankings, prices, preset, eq_budget, slot_cap=state["slot_cap"])
            state["basket_weights"] = dict(new_w)
            reason = (
                f"Scheduled {reb_period}d rebalance ({evt}): ranked Top 30 via {preset} signals "
                f"from {sig_entry['date']}; deployed {eq_frac*100:.0f}% Eq ({len(new_w)} slots)."
            )
            # If increasing equity, trim defensive first
            if gold_frac + liq_frac < (1.0 - state["active_eq_frac"]) - 1e-6:
                _sync_defensive_and_sweep(state, prices, exec_date, pre_trade_nav * gold_frac, pre_trade_nav * liq_frac, evt, reason, liquid_sweep=False)
            _rebalance_to_basket(
                state,
                new_w,
                eq_budget,
                prices,
                exec_date,
                evt,
                reason,
                respect_ltcg_shield=state["respect_ltcg_shield"],
            )
            pos_mid = _compute_positions_summary(state["lots"], prices, exec_date)
            mid_nav = state["cash"] + pos_mid["equity_val"] + pos_mid["gold_val"] + pos_mid["liquid_val"]
            _sync_defensive_and_sweep(state, prices, exec_date, mid_nav * gold_frac, mid_nav * liq_frac, evt, reason, liquid_sweep=state["liquid_sweep"])
            _settle_negative_cash(state, prices, exec_date)
            state["active_eq_frac"] = eq_frac
            state["active_def_etf"] = sig_def_etf
            state["decision_log"].append({"date": exec_date, "event_type": evt, "summary": reason})
            action_tags.append(evt)

        elif eq_shifted:
            old_pct = round(state["active_eq_frac"] * 100.0)
            new_pct = round(eq_frac * 100.0)
            evt = "REGIME_SHIFT"
            score_key = rule["score_key"]
            mf4_s = _finite(sig_regime.get("mf4_score"))
            if eq_frac < state["active_eq_frac"]:
                # Cut equity exposure: weakest_first trimming across held equity positions
                pos_cur = _compute_positions_summary(state["lots"], prices, exec_date)
                cur_nav = state["cash"] + pos_cur["equity_val"] + pos_cur["gold_val"] + pos_cur["liquid_val"]
                target_eq_inr = cur_nav * eq_frac
                need_free = max(0.0, pos_cur["equity_val"] - target_eq_inr)
                held_eq = [s for s, info in pos_cur["by_sym"].items() if info["asset_class"] == "EQUITY"]
                # Weakest score first; missing score = -1e9
                held_eq.sort(key=lambda s: (_finite((sig_rankings.get(s) or {}).get(score_key)) or -1e9, s))
                trimmed_syms: list[str] = []
                reason_base = (
                    f"{'MF4 ' + f'{mf4_s:+.2f} ' if mf4_s is not None else ''}stepped exposure "
                    f"{_posture_label(state['active_eq_frac'])} ({old_pct}%) -> {_posture_label(eq_frac)} ({new_pct}%); "
                    f"weakest-first trim into {sig_def_etf}."
                )
                freed = 0.0
                for sym in held_eq:
                    if freed >= need_free - MIN_TRADE_DELTA_INR:
                        break
                    px = _finite(prices.get(sym)) or 0.0
                    cur_sh = pos_cur["by_sym"][sym]["shares"]
                    if px <= 0 or cur_sh <= 0:
                        continue
                    rem_need = need_free - freed
                    want_sh = cur_sh if cur_sh * px <= rem_need + MIN_TRADE_DELTA_INR else max(1.0, math.floor(rem_need / px))
                    sold = _apply_fifo_sell(
                        state,
                        sym,
                        min(cur_sh, want_sh),
                        px,
                        exec_date,
                        evt,
                        reason_base,
                        allow_shielded=not state["respect_ltcg_shield"],
                    )
                    if sold > 0:
                        freed += sold * px
                        trimmed_syms.append(sym)
                pos_mid = _compute_positions_summary(state["lots"], prices, exec_date)
                mid_nav = state["cash"] + pos_mid["equity_val"] + pos_mid["gold_val"] + pos_mid["liquid_val"]
                _sync_defensive_and_sweep(state, prices, exec_date, mid_nav * gold_frac, mid_nav * liq_frac, evt, reason_base, liquid_sweep=state["liquid_sweep"])
                msg = reason_base + (f" Trimmed: {', '.join(trimmed_syms)}." if trimmed_syms else "")
            else:
                # Increase equity exposure: free defensive ETFs and re-deploy into current basket
                reason_base = (
                    f"{'MF4 ' + f'{mf4_s:+.2f} ' if mf4_s is not None else ''}stepped exposure UP "
                    f"{_posture_label(state['active_eq_frac'])} ({old_pct}%) -> {_posture_label(eq_frac)} ({new_pct}%); "
                    f"rotates defensive back into target equity basket."
                )
                target_eq_inr = pre_trade_nav * eq_frac
                _sync_defensive_and_sweep(state, prices, exec_date, pre_trade_nav * gold_frac, pre_trade_nav * liq_frac, evt, reason_base, liquid_sweep=False)
                if not state.get("basket_weights"):
                    state["basket_weights"] = compute_target_weights(sig_rankings, prices, preset, target_eq_inr, slot_cap=state["slot_cap"])
                _rebalance_to_basket(state, state["basket_weights"], target_eq_inr, prices, exec_date, evt, reason_base, respect_ltcg_shield=state["respect_ltcg_shield"])
                if state["liquid_sweep"]:
                    pos_mid = _compute_positions_summary(state["lots"], prices, exec_date)
                    mid_nav = state["cash"] + pos_mid["equity_val"] + pos_mid["gold_val"] + pos_mid["liquid_val"]
                    _sync_defensive_and_sweep(state, prices, exec_date, mid_nav * gold_frac, mid_nav * liq_frac, evt, reason_base, liquid_sweep=True)
                msg = reason_base
            _settle_negative_cash(state, prices, exec_date)
            state["active_eq_frac"] = eq_frac
            state["active_def_etf"] = sig_def_etf
            state["decision_log"].append({"date": exec_date, "event_type": evt, "summary": msg})
            action_tags.append(evt)

        elif def_flipped:
            evt = "DEFENSIVE_GATE_FLIP"
            g_pct = _finite(sig_regime.get("gold_vs_ema150_pct"))
            reason = (
                f"TrendGold150 gate flipped defensive sleeve from {state['active_def_etf']} -> {sig_def_etf} "
                f"(GOLDBEES {f'{g_pct:+.2f}%' if g_pct is not None else ''} vs EMA150; 0.15% switching fee)."
            )
            _sync_defensive_and_sweep(state, prices, exec_date, pre_trade_nav * gold_frac, pre_trade_nav * liq_frac, evt, reason, liquid_sweep=state["liquid_sweep"])
            _settle_negative_cash(state, prices, exec_date)
            state["active_def_etf"] = sig_def_etf
            state["decision_log"].append({"date": exec_date, "event_type": evt, "summary": reason})
            action_tags.append(evt)

        else:
            # Quiet day: record daily hold verification
            days_left = max(0, reb_period - state["trading_days_since_rebalance"])
            mf4_s = _finite(sig_regime.get("mf4_score"))
            msg = (
                f"Hold {_posture_label(eq_frac)} ({eq_frac*100:.0f}% Eq); "
                f"{'MF4 smooth ' + f'{mf4_s:+.2f}; ' if mf4_s is not None else ''}"
                f"next rebalance in {days_left} trading days; defensive={sig_def_etf}."
            )
            state["decision_log"].append({"date": exec_date, "event_type": "HOLD_CHECK", "summary": msg})

    # End-of-day Valuation & History Snapshot
    pos_end = _compute_positions_summary(state["lots"], prices, exec_date)
    nav_end = state["cash"] + pos_end["equity_val"] + pos_end["gold_val"] + pos_end["liquid_val"]
    state["peak_nav"] = max(float(state.get("peak_nav") or 0.0), nav_end)
    dd_pct = ((nav_end / state["peak_nav"] - 1.0) * 100.0) if state["peak_nav"] > 0 else 0.0
    def_tax_cum = float(state["tax_state"]["estimated_def_tax_cum"])
    eq_tax_cum = float(state["tax_state"]["realized_eq_tax_cum"])
    nav_full_tax = max(0.0, nav_end - def_tax_cum)

    if bench_ew_nav_today is None:
        ew_cur = _finite(regime_today.get("ew_index_val")) or state["ew_inception_val"] or 100.0
        ew_base = float(state.get("ew_inception_val") or ew_cur or 100.0)
        bench_ew_nav_today = state["initial_capital"] * (ew_cur / ew_base) if ew_base > 0 else state["initial_capital"]

    state["holdings"] = _build_holdings_rows(state, prices, exec_date)
    state["last_synced_date"] = exec_date

    eq_share_pct = (pos_end["equity_val"] / nav_end * 100.0) if nav_end > 0 else 0.0
    mf4_today = _finite(regime_today.get("mf4_score"))
    state["daily_history"].append(
        sanitize_json(
            {
                "date": exec_date,
                "nav": nav_end,
                "nav_full_tax": nav_full_tax,
                "cash": state["cash"],
                "equity_val": pos_end["equity_val"],
                "gold_val": pos_end["gold_val"],
                "liquid_val": pos_end["liquid_val"],
                "eq_pct": round(eq_share_pct, 2),
                "posture": str(regime_today.get("posture") or _posture_label(eq_frac)),
                "mf4_score": mf4_today,
                "active_def_etf": str(regime_today.get("active_def_etf") or state["active_def_etf"]),
                "benchmark_ew_nav": bench_ew_nav_today,
                "unhedged_nav": unhedged_nav_today,
                "drawdown_pct": dd_pct,
                "realized_tax_cum": eq_tax_cum,
                "estimated_def_tax_cum": def_tax_cum,
                "total_fees_cum": state["total_fees_cum"],
                "action_summary": ", ".join(action_tags) if action_tags else "HOLD_CHECK",
            },
            decimals=4,
        )
    )
    return state


def run_synthetic_steps(
    tape_entries: list[dict[str, Any]],
    *,
    account_id: str = "SYNTH",
    label: str = "Synthetic Test",
    initial_capital: float = 100_000.0,
    preset: str = "R3",
    slot_cap: bool = True,
    liquid_sweep: bool = True,
    respect_ltcg_shield: bool = False,
    unhedged: bool = False,
    do_nothing: bool = False,
    initial_lots: list[dict[str, Any]] | None = None,
    starting_cash: float | None = None,
) -> dict[str, Any]:
    """Run an entire tape slice deterministically (used for unit tests & JS parity tests)."""
    state = initialize_account_state(
        account_id=account_id,
        label=label,
        initial_capital=initial_capital,
        preset=preset,
        slot_cap=slot_cap,
        liquid_sweep=liquid_sweep,
        respect_ltcg_shield=respect_ltcg_shield,
        unhedged=unhedged,
        do_nothing=do_nothing,
        initial_lots=initial_lots,
        starting_cash=starting_cash,
    )
    for idx, day in enumerate(tape_entries):
        prev_day = tape_entries[idx - 1] if idx > 0 else None
        step_portfolio(state, day, prev_day)
    return state


def build_forward_tape_entry(
    date_str: str,
    raw_last: Any,
    bench_df: Any,
    etf_symbols: list[str],
    ma50_last: Any,
    adj_last: Any,
    scores: dict[str, Any],
    mf_diag_row: Any,
    tg_status: dict[str, Any],
    reg_dd15_val: float,
    reg_dd75_val: float,
    reg_dd50_val: float,
    ew_index_val: float,
    corp_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one compact daily forward-tape step record from live market dataframes."""
    prices: dict[str, float] = {}
    for sym, val in raw_last.items():
        v = _finite(val)
        if v is not None and v > 0:
            prices[str(sym)] = round(v, 2)
    for etf in etf_symbols:
        if hasattr(bench_df, "columns") and etf in bench_df.columns:
            col = bench_df[etf].dropna()
            if not col.empty:
                v = _finite(col.iloc[-1])
                if v is not None and v > 0:
                    prices[etf] = round(v, 2)

    s9 = scores["raw_9m"].loc[date_str]
    s12 = scores["raw_12m"].loc[date_str]
    sr9 = scores["risk_9m"].loc[date_str]
    ss9 = scores["sortino_9m"].loc[date_str]

    rankings: dict[str, dict[str, Any]] = {}
    for sym in prices:
        if sym in ETF_SET:
            continue
        rk9 = _finite(sr9.get(sym))
        so9 = _finite(ss9.get(sym))
        r9 = _finite(s9.get(sym))
        r12 = _finite(s12.get(sym))
        if rk9 is None and so9 is None and r9 is None and r12 is None:
            continue
        a_px = _finite(adj_last.get(sym)) if adj_last is not None else None
        m50 = _finite(ma50_last.get(sym)) if ma50_last is not None else None
        rankings[sym] = sanitize_json(
            {
                "rk9": round(rk9, 4) if rk9 is not None else None,
                "so9": round(so9, 4) if so9 is not None else None,
                "r9": round(r9 * 100.0, 2) if r9 is not None else None,
                "r12": round(r12 * 100.0, 2) if r12 is not None else None,
                "a50": 1 if (a_px is not None and m50 is not None and a_px > m50) else 0,
            },
            decimals=4,
        )

    t_mf4 = _finite(mf_diag_row["target_eq_eod"]) if "target_eq_eod" in mf_diag_row else 1.0
    t_mf4 = 1.0 if t_mf4 is None else t_mf4
    mf4_score = _finite(mf_diag_row["composite_smooth"]) if "composite_smooth" in mf_diag_row else 0.0

    ca_clean: list[dict[str, Any]] = []
    for ev in corp_actions or []:
        if str(ev.get("date", ""))[:10] == date_str[:10]:
            f = _finite(ev.get("factor"))
            if f is not None and f > 0:
                ca_clean.append({"symbol": str(ev["symbol"]), "type": str(ev.get("type", "SPLIT")).upper(), "factor": round(f, 6)})

    return sanitize_json(
        {
            "date": date_str,
            "prices": prices,
            "corp_actions": ca_clean,
            "regime": {
                "mf4_score": round(mf4_score or 0.0, 3),
                "posture": _posture_label(t_mf4),
                "target_eq_mf4": round(t_mf4, 4),
                "target_eq_dd15": round(float(reg_dd15_val), 4),
                "target_eq_dd75": round(float(reg_dd75_val), 4),
                "target_eq_dd50": round(float(reg_dd50_val), 4),
                "active_def_etf": str(tg_status.get("active_defensive_etf", "LIQUIDCASE")),
                "gold_price": tg_status.get("gold_price"),
                "gold_ema150": tg_status.get("gold_ema150"),
                "gold_vs_ema150_pct": tg_status.get("gold_vs_ema150_pct"),
                "ew_index_val": round(float(ew_index_val), 2),
            },
            "rankings": rankings,
        },
        decimals=4,
    )


def update_forward_tape(tape_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append `entry` to forward_tape.json idempotently (no rewrite if date already present)."""
    target_dt = str(entry["date"])
    if tape_path.exists():
        tape = json.loads(tape_path.read_text())
        days = tape.get("days") or []
        if any(str(d.get("date")) == target_dt for d in days):
            return tape
        days.append(entry)
        days.sort(key=lambda d: str(d.get("date")))
        tape["days"] = days
        tape["last_date"] = str(days[-1]["date"])
    else:
        tape = {
            "schema_version": 1,
            "inception_date": target_dt,
            "last_date": target_dt,
            "etf_symbols": list(ETF_SYMBOLS),
            "days": [entry],
        }
    safe = sanitize_json(tape, decimals=4)
    tape_path.parent.mkdir(parents=True, exist_ok=True)
    tape_path.write_text(json.dumps(safe, separators=(",", ":"), allow_nan=False))
    return safe


def _summarize_account(acc: dict[str, Any], ew_series_val: float | None = None, unhedged_val: float | None = None) -> dict[str, Any]:
    hist = acc.get("daily_history") or []
    last = hist[-1] if hist else {}
    cap = float(acc.get("initial_capital") or 100_000.0)
    cur_nav = float(last.get("nav") or cap)
    cur_nav_ft = float(last.get("nav_full_tax") or cur_nav)
    ew_nav = float(last.get("benchmark_ew_nav") or ew_series_val or cap)
    unh_nav = float(last.get("unhedged_nav") or unhedged_val or cap)

    tot_ret = (cur_nav / cap - 1.0) * 100.0 if cap > 0 else 0.0
    tot_ret_ft = (cur_nav_ft / cap - 1.0) * 100.0 if cap > 0 else 0.0
    ew_ret = (ew_nav / cap - 1.0) * 100.0 if cap > 0 else 0.0
    unh_ret = (unh_nav / cap - 1.0) * 100.0 if cap > 0 else 0.0
    max_dd = min([float(h.get("drawdown_pct") or 0.0) for h in hist] or [0.0])

    reb_days = int(PRESET_RULES[acc["preset"]]["rebalance_days"])
    days_since = int(acc.get("trading_days_since_rebalance") or 0)
    next_reb_days = max(0, reb_days - days_since)
    eq_val = float(last.get("equity_val") or 0.0)
    gold_val = float(last.get("gold_val") or 0.0)
    liq_val = float(last.get("liquid_val") or 0.0)
    cash_val = float(last.get("cash") or 0.0)

    return {
        "inception_date": acc.get("inception_date"),
        "as_of_date": acc.get("last_synced_date"),
        "trading_days": len(hist),
        "days_since_rebalance": days_since,
        "next_rebalance_in_trading_days": next_reb_days,
        "nav": round(cur_nav, 2),
        "nav_full_tax": round(cur_nav_ft, 2),
        "total_return_pct": round(tot_ret, 2),
        "total_return_full_tax_pct": round(tot_ret_ft, 2),
        "benchmark_ew_nav": round(ew_nav, 2),
        "benchmark_ew_return_pct": round(ew_ret, 2),
        "unhedged_nav": round(unh_nav, 2),
        "unhedged_return_pct": round(unh_ret, 2),
        "alpha_vs_ew_pct": round(tot_ret - ew_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "alloc_pct": {
            "equity": round(eq_val / cur_nav * 100.0, 1) if cur_nav > 0 else 0.0,
            "gold": round(gold_val / cur_nav * 100.0, 1) if cur_nav > 0 else 0.0,
            "liquid": round(liq_val / cur_nav * 100.0, 1) if cur_nav > 0 else 0.0,
            "cash": round(cash_val / cur_nav * 100.0, 1) if cur_nav > 0 else 0.0,
        },
        "total_fees_cum": round(float(acc.get("total_fees_cum") or 0.0), 2),
        "realized_eq_tax_cum": round(float(acc["tax_state"]["realized_eq_tax_cum"]), 2),
        "estimated_def_tax_cum": round(float(acc["tax_state"]["estimated_def_tax_cum"]), 2),
        "held_count": len(acc.get("holdings") or []),
    }


def update_paper_ledger(ledger_path: Path, tape: dict[str, Any]) -> dict[str, Any]:
    """Update official autonomous paper-trading ledger (idempotent per date)."""
    days: list[dict[str, Any]] = tape.get("days") or []
    if not days:
        raise ValueError("forward_tape has zero trading days")

    latest_date = str(days[-1]["date"])
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("last_date") == latest_date:
            # Strictly idempotent: return exact existing ledger without rewriting file
            return ledger
        accounts = ledger["accounts"]
        baselines = ledger["baselines"]
    else:
        inception = str(days[0]["date"])
        accounts = {
            "R3_1L_SlotCap": initialize_account_state(
                account_id="R3_1L_SlotCap",
                label="R3 · ₹1L (SlotCap + Sweep)",
                initial_capital=100_000.0,
                preset="R3",
                slot_cap=True,
                liquid_sweep=True,
            ),
            "R3_10L_Standard": initialize_account_state(
                account_id="R3_10L_Standard",
                label="R3 · ₹10L (Standard)",
                initial_capital=1_000_000.0,
                preset="R3",
                slot_cap=False,
                liquid_sweep=True,
            ),
        }
        baselines = {
            "Unhedged_Risk9M_10L": initialize_account_state(
                account_id="Unhedged_Risk9M_10L",
                label="Unhedged Risk9M (No Hedge)",
                initial_capital=1_000_000.0,
                preset="R3",
                slot_cap=False,
                liquid_sweep=True,
                unhedged=True,
            ),
        }
        ledger = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inception_date": inception,
            "last_date": None,
            "accounts": accounts,
            "baselines": baselines,
        }

    # Step unhedged baseline and both official bot accounts across any unseen tape days
    unh_acc = baselines["Unhedged_Risk9M_10L"]
    for idx, day in enumerate(days):
        dt = str(day["date"])
        prev_day = days[idx - 1] if idx > 0 else None
        if not unh_acc.get("last_synced_date") or dt > unh_acc["last_synced_date"]:
            step_portfolio(unh_acc, day, prev_day)
        unh_day_row = next((h for h in unh_acc["daily_history"] if h["date"] == dt), None)
        unh_nav_10l = float(unh_day_row["nav"]) if unh_day_row else 1_000_000.0

        ew_val = _finite((day.get("regime") or {}).get("ew_index_val")) or 100.0
        for acc_id, acc in accounts.items():
            if not acc.get("last_synced_date") or dt > acc["last_synced_date"]:
                cap = float(acc["initial_capital"])
                ew_base = float(acc.get("ew_inception_val") or ew_val or 100.0)
                bench_ew_nav = cap * (ew_val / ew_base) if ew_base > 0 else cap
                unh_scaled = cap * (unh_nav_10l / 1_000_000.0)
                step_portfolio(acc, day, prev_day, unhedged_nav_today=unh_scaled, bench_ew_nav_today=bench_ew_nav)

    for acc in accounts.values():
        acc["summary"] = _summarize_account(acc)
    ledger["last_date"] = latest_date
    ledger["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    safe = sanitize_json(ledger, decimals=4)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(safe, indent=2, allow_nan=False))
    return safe
