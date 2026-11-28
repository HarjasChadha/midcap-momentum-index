"""Event-driven portfolio backtester for DIY Midcap Momentum with frictions, Indian taxes, and discrete-share execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.momentum import combine_sleeves, select_affordable_blend, select_sleeve_weights


@dataclass
class BacktestConfig:
    start_date: str = "2017-12-01"
    end_date: str = "2026-11-26"
    initial_capital: float = 100_000.0
    rebalance_days: int = 189  # ~9 months default (21 trading days/month)
    num_sleeves: int = 1  # >1 enables staggered multi-tranche rebalancing
    phase_offset_days: int = 0  # phase shift (0..rebalance_days-1) for rebalance timing sensitivity
    sleeves: list[dict[str, Any]] | None = None  # e.g. [{"score_key": "raw_9m", "top_n": 20, "alloc": 0.5, ...}]
    cost_bps_one_way: float = 15.0  # 0.10% STT + ~0.035% charges + ~0.015% slippage
    apply_tax: bool = False  # Indian STCG 20% (<365d) / LTCG 12.5% (>=365d)
    stcg_rate: float = 0.20
    ltcg_rate: float = 0.125
    ltcg_annual_exempt: float = 125_000.0
    defensive_asset: str = "GOLDBEES"
    regime_rule: str = "none"  # static alloc column or internal rule
    internal_rule: str = "none"  # "none" | "port_ma" | "port_dd" | "port_breadth" | "stock_stop"
    internal_params: dict[str, Any] | None = None
    trim_mode: str = "pro_rata"  # "pro_rata" | "weakest_first"
    discrete_shares: bool = False  # integer shares on raw_close
    affordability_filter: bool = False  # sequential slot-cap substitution when discrete_shares=True
    sweep_idle_cash: bool = False  # sweep uninvested discrete cash into sweep_asset (e.g. LIQUIDCASE)
    sweep_asset: str = "LIQUID_REALISTIC"
    tax_defensive: bool = True  # Sec 50AA 30% slab tax on liquid/debt & 12.5%/30% on Gold ETF switches
    regime_cost_bps_one_way: float = 35.0  # elevated slippage on DEFENSIVE/CAUTIOUS risk-off liquidations
    delist_haircut_pct: float = 0.0  # >0 forces write-down/exit if held stock vanishes past ffill limit=5


def compute_metrics(nav: pd.Series, turnover_series: pd.Series | None = None, total_tax: float = 0.0, rf_annual: float = 0.0625) -> dict[str, Any]:
    s = nav.dropna()
    if len(s) < 20:
        return {"error": "insufficient data"}
    rets = s.pct_change().dropna()
    # Use true calendar years when index is ISO dates (falls back to trading days/252 for non-date indices)
    try:
        cal_days = (pd.Timestamp(str(s.index[-1])) - pd.Timestamp(str(s.index[0]))).days
        yrs = cal_days / 365.2425 if cal_days > 20 else len(s) / 252.0
    except Exception:
        yrs = len(s) / 252.0
    days_per_yr = len(s) / yrs if yrs > 0 else 252.0
    cagr = (float(s.iloc[-1]) / float(s.iloc[0])) ** (1.0 / yrs) - 1.0
    ann_vol = float(rets.std() * math.sqrt(days_per_yr))
    dd_series = s / s.cummax() - 1.0
    max_dd = float(dd_series.min())
    calmar = cagr / abs(max_dd) if max_dd < -1e-9 else float("inf")
    sharpe = (cagr - rf_annual) / ann_vol if ann_vol > 1e-9 else 0.0
    downside = rets[rets < 0]
    down_vol = float(downside.std() * math.sqrt(days_per_yr)) if len(downside) > 2 else ann_vol
    sortino = (cagr - rf_annual) / down_vol if down_vol > 1e-9 else 0.0
    ulcer = float(math.sqrt(np.mean((dd_series.values * 100.0) ** 2)))

    # Longest drawdown recovery in trading days
    in_dd = dd_series < -1e-6
    longest_dd, cur_dd = 0, 0
    for flag in in_dd:
        cur_dd = cur_dd + 1 if flag else 0
        if cur_dd > longest_dd:
            longest_dd = cur_dd

    # Calendar year returns
    yearly = {}
    s_dt = s.copy()
    years = sorted({d[:4] for d in s_dt.index})
    prev_val = float(s_dt.iloc[0])
    for yr in years:
        sub = s_dt.loc[s_dt.index.str.startswith(yr)]
        if len(sub) >= 5:
            end_v = float(sub.iloc[-1])
            yearly[yr] = round((end_v / prev_val - 1.0) * 100.0, 2)
            prev_val = end_v

    ann_turnover = float(turnover_series.sum() / yrs) if turnover_series is not None else 0.0
    pos_years = sum(1 for v in yearly.values() if v > 0)
    win_years_pct = (pos_years / len(yearly) * 100.0) if yearly else 0.0

    return {
        "start_date": str(s.index[0]),
        "end_date": str(s.index[-1]),
        "years": round(yrs, 2),
        "final_value": round(float(s.iloc[-1]), 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "calmar": round(calmar, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "ann_vol_pct": round(ann_vol * 100.0, 2),
        "ulcer_index": round(ulcer, 2),
        "longest_dd_days": int(longest_dd),
        "ann_turnover_pct": round(ann_turnover * 100.0, 1),
        "win_years_pct": round(win_years_pct, 1),
        "total_tax_paid": round(total_tax, 2),
        "yearly_returns_pct": yearly,
    }


def _compute_target_weights_for_sleeve(
    date_str: str,
    score_matrices: dict[str, pd.DataFrame],
    vol_63: pd.DataFrame,
    sleeve_cfg: dict[str, Any],
    prev_holdings: set[str],
) -> dict[str, float]:
    smat = score_matrices[sleeve_cfg["score_key"]]
    if date_str not in smat.index:
        return {}
    srow = smat.loc[date_str]
    vrow = vol_63.loc[date_str] if date_str in vol_63.index else None
    return select_sleeve_weights(
        srow,
        top_n=int(sleeve_cfg["top_n"]),
        weighting=sleeve_cfg.get("weighting", "equal"),
        vol_row=vrow,
        prev_holdings=prev_holdings,
        buffer_n=sleeve_cfg.get("buffer_n"),
    )


def run_backtest(
    adj_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    score_matrices: dict[str, pd.DataFrame],
    def_df: pd.DataFrame,
    eq_alloc_series: pd.Series,
    cfg: BacktestConfig,
    stock_ma50: pd.DataFrame | None = None,
    vol_63: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run daily zero-look-ahead backtest across cfg.start_date..cfg.end_date."""
    mask = (adj_df.index >= cfg.start_date) & (adj_df.index <= cfg.end_date)
    dates = list(adj_df.index[mask])
    if len(dates) < 30:
        raise ValueError("Backtest date range too short")

    adj = adj_df.loc[dates].ffill(limit=5)
    raw = raw_df.loc[dates].ffill(limit=5)
    def_series = def_df[cfg.defensive_asset].reindex(dates).ffill(limit=5)
    sweep_col = cfg.sweep_asset if cfg.sweep_asset in def_df.columns else cfg.defensive_asset
    sweep_series = def_df[sweep_col].reindex(dates).ffill(limit=5)
    eq_alloc = eq_alloc_series.reindex(dates).fillna(1.0).clip(0.0, 1.0)
    sleeves_cfg = cfg.sleeves or [{"score_key": "raw_9m", "top_n": 30, "alloc": 1.0, "weighting": "equal"}]
    need_vol = any(s.get("weighting") == "inv_vol" for s in sleeves_cfg)
    if vol_63 is None:
        vol_63 = (adj_df.pct_change(fill_method=None).rolling(63, min_periods=20).std().reindex(dates) * math.sqrt(252.0)) if need_vol else pd.DataFrame(index=dates)
    else:
        vol_63 = vol_63.reindex(dates)

    first_idx_pos = adj_df.index.get_loc(dates[0])
    first_prev_date = str(adj_df.index[first_idx_pos - 1]) if isinstance(first_idx_pos, int) and first_idx_pos > 0 else dates[0]

    if stock_ma50 is None:
        stock_ma50 = adj_df.rolling(50, min_periods=30).mean().reindex(dates)
    else:
        stock_ma50 = stock_ma50.reindex(dates)

    sleeves = cfg.sleeves or [{"score_key": "raw_9m", "top_n": 30, "alloc": 1.0, "weighting": "equal"}]
    n_sl = max(1, cfg.num_sleeves)
    reb = max(1, cfg.rebalance_days)
    base_off = int(cfg.phase_offset_days) % reb
    sleeve_schedule_offsets = [(base_off + round(k * reb / n_sl)) % reb for k in range(n_sl)]
    sleeve_targets: list[dict[str, float]] = [{} for _ in range(n_sl)]

    # Portfolio holdings state: symbol -> list of lots [shares, buy_adj_price, buy_date]
    lots: dict[str, list[list[Any]]] = {}
    last_valid_px: dict[str, float] = {}
    delisted_written_down: set[str] = set()
    delisted_write_downs = 0
    regime_shift_trades = 0
    def_lots: list[list[Any]] = []  # [units, cost_p, buy_date]
    sweep_lots: list[list[Any]] = []
    tg_active_series = (
        def_df["TREND_GOLD150_ACTIVE"].reindex(dates).ffill().fillna(0.0)
        if "TREND_GOLD150_ACTIVE" in def_df.columns
        else pd.Series(0.0, index=dates)
    )
    prev_tg_gold: bool | None = bool(tg_active_series.iloc[0] > 0.5) if len(dates) else False
    cash = cfg.initial_capital
    stopped_out: set[str] = set()
    stock_peaks: dict[str, float] = {}

    cost_rate = cfg.cost_bps_one_way / 10_000.0
    regime_cost_rate = cfg.regime_cost_bps_one_way / 10_000.0
    total_tax = 0.0
    total_def_tax = 0.0
    total_fees_paid = 0.0
    fy_ltcg_gains: dict[int, float] = {}
    fy_stcg_carry: float = 0.0
    fy_ltcg_carry: float = 0.0
    prev_fy: int | None = None

    # Synthetic unhedged equity-basket NAV (shadow-tracked even while in defensive asset)
    synth_nav = 100.0
    synth_hist: list[float] = []
    internal_eq_cap = 1.0
    prev_target_eq_frac = 1.0
    nav_list = []
    turnover_list = []
    eq_wt_list = []
    def_wt_list = []
    avg_holdings_list = []

    ip = cfg.internal_params or {}

    def _fy_for(dstr: str) -> int:
        y, m = int(dstr[:4]), int(dstr[5:7])
        return y if m >= 4 else y - 1

    def _sell_symbol_fraction(sym: str, target_shares_to_sell: float, sell_price: float, dstr: str) -> float:
        nonlocal total_tax, fy_stcg_carry, fy_ltcg_carry
        if sym not in lots or target_shares_to_sell <= 1e-12:
            return 0.0
        sold_notional = 0.0
        rem = target_shares_to_sell
        sell_dt = pd.Timestamp(dstr)
        fy = _fy_for(dstr)
        new_lots = []
        for sh, cost_p, buy_dt_str in lots[sym]:
            if rem <= 1e-12:
                new_lots.append([sh, cost_p, buy_dt_str])
                continue
            take = min(sh, rem)
            rem -= take
            gross = take * sell_price
            sold_notional += gross
            gain = take * (sell_price - cost_p)
            if cfg.apply_tax:
                days_held = (sell_dt - pd.Timestamp(buy_dt_str)).days
                if days_held >= 365:
                    net_g = gain + fy_ltcg_carry
                    if net_g < 0:
                        fy_ltcg_carry = net_g
                    else:
                        fy_ltcg_carry = 0.0
                        used_ex = fy_ltcg_gains.get(fy, 0.0)
                        taxable_before = max(0.0, used_ex - cfg.ltcg_annual_exempt)
                        fy_ltcg_gains[fy] = used_ex + net_g
                        taxable_after = max(0.0, fy_ltcg_gains[fy] - cfg.ltcg_annual_exempt)
                        tax = max(0.0, taxable_after - taxable_before) * cfg.ltcg_rate
                        total_tax += tax
                        sold_notional -= tax
                else:
                    net_g = gain + fy_stcg_carry
                    if net_g < 0:
                        fy_stcg_carry = net_g
                    else:
                        fy_stcg_carry = 0.0
                        tax = net_g * cfg.stcg_rate
                        total_tax += tax
                        sold_notional -= tax
            left = sh - take
            if left > 1e-10:
                new_lots.append([left, cost_p, buy_dt_str])
        if new_lots:
            lots[sym] = new_lots
        else:
            lots.pop(sym, None)
        return max(0.0, sold_notional)

    def _def_tax_rate_for(asset_name: str, is_tg_gold: bool, days_d: int) -> float:
        if asset_name == "TREND_GOLD150_OR_LIQ":
            return (cfg.ltcg_rate if days_d >= 365 else 0.30) if is_tg_gold else 0.30
        if asset_name == "GOLD_50_LIQUID_50":
            g_r = cfg.ltcg_rate if days_d >= 365 else 0.30
            return 0.5 * g_r + 0.5 * 0.30
        if any(k in asset_name for k in ("GOLD", "SILVER", "NIFTY", "MON100")):
            return cfg.ltcg_rate if days_d >= 365 else 0.30
        return 0.30  # Sec 50AA debt/liquid ETF slab rate

    def _sell_fifo_etf_lots(lot_list: list[list[Any]], sell_u: float, cur_p: float, dstr: str, asset_name: str, is_tg_gold: bool, apply_fee_rate: float) -> float:
        nonlocal total_tax, total_def_tax, total_fees_paid
        if sell_u <= 1e-12 or not lot_list or cur_p <= 0:
            return 0.0
        rem = sell_u
        gross_tot = 0.0
        sell_dt = pd.Timestamp(dstr)
        kept: list[list[Any]] = []
        for u, cp, bdt in lot_list:
            if rem <= 1e-12:
                kept.append([u, cp, bdt])
                continue
            take = min(u, rem)
            rem -= take
            gross = take * cur_p
            gross_tot += gross
            gain = take * (cur_p - cp)
            if cfg.apply_tax and cfg.tax_defensive and gain > 0:
                days_d = (sell_dt - pd.Timestamp(bdt)).days
                tx = gain * _def_tax_rate_for(asset_name, is_tg_gold, days_d)
                total_tax += tx
                total_def_tax += tx
                gross_tot -= tx
            left = u - take
            if left > 1e-10:
                kept.append([left, cp, bdt])
        lot_list[:] = kept
        fee = (sell_u * cur_p) * apply_fee_rate
        total_fees_paid += fee
        return max(0.0, gross_tot - fee)

    prev_prices: pd.Series | None = None

    for i, dstr in enumerate(dates):
        fy_cur = _fy_for(dstr)
        if prev_fy is not None and fy_cur != prev_fy:
            # New Indian fiscal year: LTCG exemption resets (unused carry-forward losses remain)
            pass
        prev_fy = fy_cur

        p_row = adj.loc[dstr]
        raw_row = raw.loc[dstr]
        adj_row = adj.loc[dstr]
        def_p = float(def_series.iloc[i])
        sweep_p = float(sweep_series.iloc[i])
        cur_tg_gold = bool(tg_active_series.iloc[i] > 0.5)

        # If TrendGold150 rotated internally between GOLDBEES and LIQUID_REALISTIC, realize accrued FIFO defensive tax cleanly
        if cfg.defensive_asset == "TREND_GOLD150_OR_LIQ" and i > 0 and def_lots and prev_tg_gold is not None and cur_tg_gold != prev_tg_gold:
            tot_u = sum(l[0] for l in def_lots)
            if cfg.apply_tax and cfg.tax_defensive and tot_u > 1e-9:
                sell_dt_tg = pd.Timestamp(dstr)
                for u, cp, bdt in def_lots:
                    gain_tg = u * (def_p - cp)
                    if gain_tg > 0:
                        d_tg = (sell_dt_tg - pd.Timestamp(bdt)).days
                        tx_tg = gain_tg * _def_tax_rate_for("TREND_GOLD150_OR_LIQ", prev_tg_gold, d_tg)
                        total_tax += tx_tg
                        total_def_tax += tx_tg
                        cash -= tx_tg
                def_lots = [[tot_u, def_p, dstr]]
        prev_tg_gold = cur_tg_gold

        # Handle halted/delisted held symbols (>5d halt past ffill limit=5): NEVER revert to original purchase cost!
        for sym in list(lots.keys()):
            px = float(p_row.get(sym, np.nan))
            if math.isfinite(px) and px > 0:
                last_valid_px[sym] = px
            else:
                base_px = last_valid_px.get(sym, 0.0)
                if cfg.delist_haircut_pct > 0 and sym not in delisted_written_down:
                    delisted_written_down.add(sym)
                    delisted_write_downs += 1
                    write_px = max(0.0, base_px * (1.0 - cfg.delist_haircut_pct))
                    tot_sh = sum(l[0] for l in lots[sym])
                    if write_px > 0:
                        gross_p = _sell_symbol_fraction(sym, tot_sh, write_px, dstr)
                        fee_d = (tot_sh * write_px) * cost_rate
                        total_fees_paid += fee_d
                        cash += max(0.0, gross_p - fee_d)
                    else:
                        lots.pop(sym, None)
                    stock_peaks.pop(sym, None)

        # Mark-to-market total equity using last valid traded price (never purchase cost)
        eq_val_before = 0.0
        for sym, sym_lots in lots.items():
            px = float(p_row.get(sym, np.nan))
            mtm_px = px if (math.isfinite(px) and px > 0) else last_valid_px.get(sym, 0.0)
            eq_val_before += sum(l[0] for l in sym_lots) * mtm_px

        def_units = sum(l[0] for l in def_lots)
        sweep_units = sum(l[0] for l in sweep_lots)
        def_val_before = def_units * def_p
        sweep_val_before = sweep_units * sweep_p
        total_val = eq_val_before + def_val_before + sweep_val_before + cash

        # Update synthetic unhedged basket NAV (shadow-tracks base_eq_weights even if hedged out!)
        shadow_weights = combine_sleeves([(1.0 / n_sl, st) for st in sleeve_targets if st])
        if i > 0 and prev_prices is not None and shadow_weights:
            w_ret, w_sum = 0.0, 0.0
            for sym, w_s in shadow_weights.items():
                p0 = float(prev_prices.get(sym, np.nan))
                p1 = float(adj_row.get(sym, np.nan))
                if math.isfinite(p0) and p0 > 0 and math.isfinite(p1) and p1 > 0:
                    w_ret += w_s * (p1 / p0 - 1.0)
                    w_sum += w_s
            if w_sum > 0:
                synth_nav *= 1.0 + (w_ret / w_sum)
        synth_hist.append(synth_nav)

        # Update trailing peaks on held stocks (using yesterday's prices for zero look-ahead stop check)
        if i > 0 and prev_prices is not None:
            for sym in lots:
                px_prev = float(prev_prices.get(sym, np.nan))
                if math.isfinite(px_prev) and px_prev > 0:
                    stock_peaks[sym] = max(stock_peaks.get(sym, px_prev), px_prev)

        # Evaluate internal regime signal using ONLY info up to i-1 (prev_prices / synth_hist[:-1])
        if cfg.internal_rule == "none" or i < 20:
            internal_eq_cap = 1.0
        elif cfg.internal_rule == "port_ma":
            ma_lb = int(ip.get("ma_len", 50))
            if len(synth_hist) > ma_lb:
                prev_synth = synth_hist[-2]
                ma_val = float(np.mean(synth_hist[-1 - ma_lb : -1]))
                internal_eq_cap = 1.0 if prev_synth > ma_val else (1.0 - float(ip.get("defensive_shift", 1.0)))
            else:
                internal_eq_cap = 1.0
        elif cfg.internal_rule == "port_dd":
            exit_dd = -abs(float(ip.get("dd_thresh", 0.12)))
            ma_lb = int(ip.get("reenter_ma", 50))
            if len(synth_hist) > 2:
                prev_synth = synth_hist[-2]
                peak_252 = float(np.max(synth_hist[max(0, len(synth_hist) - 253) : -1]))
                cur_dd = prev_synth / peak_252 - 1.0 if peak_252 > 0 else 0.0
                ma_val = float(np.mean(synth_hist[max(0, len(synth_hist) - 1 - ma_lb) : -1]))
                if internal_eq_cap > 0.5 and cur_dd < exit_dd:
                    internal_eq_cap = 1.0 - float(ip.get("defensive_shift", 1.0))
                elif internal_eq_cap < 0.99 and prev_synth > ma_val and cur_dd > exit_dd * 0.6:
                    internal_eq_cap = 1.0
        elif cfg.internal_rule == "port_breadth":
            check_syms = shadow_weights.keys() if shadow_weights else lots.keys()
            if check_syms and i > 0:
                prev_dt = dates[i - 1]
                above = 0
                cnt = 0
                for sym in check_syms:
                    px_p = float(adj.loc[prev_dt, sym])
                    ma_p = float(stock_ma50.loc[prev_dt, sym])
                    if math.isfinite(px_p) and math.isfinite(ma_p):
                        cnt += 1
                        if px_p > ma_p:
                            above += 1
                frac = above / cnt if cnt else 1.0
                if internal_eq_cap > 0.5 and frac < float(ip.get("exit_below", 0.35)):
                    internal_eq_cap = 1.0 - float(ip.get("defensive_shift", 1.0))
                elif internal_eq_cap < 0.99 and frac > float(ip.get("enter_above", 0.50)):
                    internal_eq_cap = 1.0
        elif cfg.internal_rule == "stock_stop":
            # Per-stock trailing stop evaluated at i-1 close
            stop_pct = abs(float(ip.get("stop_pct", 0.20)))
            if i > 0:
                prev_dt = dates[i - 1]
                for sym in list(lots.keys()):
                    px_p = float(adj.loc[prev_dt, sym])
                    pk = stock_peaks.get(sym, px_p)
                    if math.isfinite(px_p) and pk > 0 and (px_p / pk - 1.0) < -stop_pct:
                        stopped_out.add(sym)

        # Check which sleeves rebalance today (signal computed on close i-1, executed on close i -> zero look-ahead!)
        rebalanced_today = False
        score_date = dates[i - 1] if i > 0 else first_prev_date
        for s_idx, off in enumerate(sleeve_schedule_offsets):
            if i == 0 or (i - off) % reb == 0 and i >= off:
                # Aggregate multiple sub-strategies if configured inside each scheduled sleeve
                prev_set = set(sleeve_targets[s_idx].keys())
                if cfg.discrete_shares and cfg.affordability_filter:
                    raw_score_row = raw_df.loc[score_date] if score_date in raw_df.index else raw_row
                    score_rows = {k: v.loc[score_date] for k, v in score_matrices.items() if score_date in v.index}
                    vrow = vol_63.loc[score_date] if score_date in vol_63.index else None
                    sleeve_budget = (total_val / n_sl) * max(0.50, float(eq_alloc.iloc[i]))
                    sleeve_targets[s_idx] = select_affordable_blend(
                        score_rows, sleeves, raw_score_row, sleeve_budget, vrow, prev_set
                    )
                else:
                    sub_pairs = []
                    for sc in sleeves:
                        sub_w = _compute_target_weights_for_sleeve(score_date, score_matrices, vol_63, sc, prev_set)
                        sub_pairs.append((float(sc.get("alloc", 1.0)), sub_w))
                    sleeve_targets[s_idx] = combine_sleeves(sub_pairs)
                rebalanced_today = True
                if i == 0 or n_sl == 1:
                    stopped_out.clear()
                else:
                    # Only clear stops for symbols re-selected by this sleeve
                    for sym in list(stopped_out):
                        if sym in sleeve_targets[s_idx]:
                            stopped_out.discard(sym)

        base_eq_weights = combine_sleeves([(1.0 / n_sl, st) for st in sleeve_targets if st])

        # Apply stopped-out exclusions (stopped cash moves into defensive asset)
        if stopped_out and base_eq_weights:
            active_eq_weights = {k: v for k, v in base_eq_weights.items() if k not in stopped_out}
            active_sum = sum(active_eq_weights.values())
        else:
            active_eq_weights = dict(base_eq_weights)
            active_sum = sum(active_eq_weights.values())

        target_eq_frac = min(float(eq_alloc.iloc[i]), internal_eq_cap) * active_sum
        target_eq_frac = max(0.0, min(1.0, target_eq_frac))

        # Build desired rupee weights per symbol
        desired_weights: dict[str, float] = {}
        if active_eq_weights and target_eq_frac > 0:
            if cfg.trim_mode == "weakest_first" and target_eq_frac < active_sum - 1e-6 and i > 0:
                # Keep strongest momentum stocks fully weighted; trim weakest first!
                smat0 = score_matrices[sleeves[0]["score_key"]]
                srow_prev = smat0.loc[score_date] if score_date in smat0.index else pd.Series(dtype=float)
                def _score_val(sym_k: str) -> float:
                    v = float(srow_prev.get(sym_k, -999.0))
                    return v if math.isfinite(v) else -999.0
                ordered = sorted(active_eq_weights.keys(), key=_score_val, reverse=True)
                rem_budget = target_eq_frac
                for sym in ordered:
                    w_full = active_eq_weights[sym]
                    alloc_s = min(w_full, rem_budget)
                    if alloc_s > 1e-6:
                        desired_weights[sym] = alloc_s
                        rem_budget -= alloc_s
            else:
                scale = target_eq_frac / active_sum if active_sum > 0 else 0.0
                desired_weights = {sym: w * scale for sym, w in active_eq_weights.items()}

        # Rebalance trades only when rebalance day, target regime allocation shifted > 2%, or new stop-out fired
        regime_delta = abs(target_eq_frac - prev_target_eq_frac) > 0.02
        has_stopped_held = any(sym in lots for sym in stopped_out)
        traded_notional = 0.0

        if rebalanced_today or regime_delta or has_stopped_held or i == 0:
            is_risk_off_shift = (not rebalanced_today) and regime_delta and (target_eq_frac < prev_target_eq_frac - 1e-6)
            active_cost_rate = regime_cost_rate if is_risk_off_shift else cost_rate
            if is_risk_off_shift:
                regime_shift_trades += 1
            prev_target_eq_frac = target_eq_frac
            # Target rupee values
            target_rupees = {sym: total_val * w for sym, w in desired_weights.items()}
            target_def_rupees = max(0.0, total_val - sum(target_rupees.values()))

            # 1) Execute Sells first (for reduced/exited equity positions)
            for sym in list(lots.keys()):
                px = float(p_row.get(sym, np.nan))
                rpx = float(raw_row.get(sym, px))
                if not math.isfinite(px) or px <= 0:
                    continue
                cur_sh = sum(l[0] for l in lots[sym])
                tgt_r = target_rupees.get(sym, 0.0)
                if cfg.discrete_shares and math.isfinite(rpx) and rpx > 0:
                    n_raw = int(tgt_r // rpx)
                    tgt_sh = (n_raw * rpx) / px
                else:
                    tgt_sh = tgt_r / px
                if tgt_sh < cur_sh - 1e-9:
                    sell_sh = cur_sh - tgt_sh
                    gross_proceeds = _sell_symbol_fraction(sym, sell_sh, px, dstr)
                    fee = (sell_sh * px) * active_cost_rate
                    total_fees_paid += fee
                    traded_notional += sell_sh * px
                    cash += gross_proceeds - fee
                    if sym not in target_rupees:
                        stock_peaks.pop(sym, None)

            # Defensive sell if reducing defensive allocation
            if def_val_before > target_def_rupees + 1e-6:
                sell_def_val = def_val_before - target_def_rupees
                sell_def_u = min(def_units, sell_def_val / def_p)
                if sell_def_u > 1e-9:
                    traded_notional += sell_def_u * def_p
                    cash += _sell_fifo_etf_lots(def_lots, sell_def_u, def_p, dstr, cfg.defensive_asset, cur_tg_gold, active_cost_rate)
                    def_units = sum(l[0] for l in def_lots)

            def _sell_sweep_if_needed(shortfall: float) -> None:
                nonlocal cash, sweep_units, traded_notional
                if not cfg.sweep_idle_cash or sweep_units <= 0 or shortfall <= 0 or sweep_p <= 0:
                    return
                need_u = math.ceil(shortfall / (sweep_p * (1.0 - active_cost_rate))) if cfg.discrete_shares else shortfall / (sweep_p * (1.0 - active_cost_rate))
                sell_sw = min(sweep_units, float(need_u))
                if sell_sw > 1e-9:
                    traded_notional += sell_sw * sweep_p
                    cash += _sell_fifo_etf_lots(sweep_lots, sell_sw, sweep_p, dstr, sweep_col, False, active_cost_rate)
                    sweep_units = sum(l[0] for l in sweep_lots)

            # 2) Execute Buys for equity
            for sym, tgt_r in target_rupees.items():
                px = float(p_row.get(sym, np.nan))
                rpx = float(raw_row.get(sym, px))
                if not math.isfinite(px) or px <= 0:
                    continue
                cur_sh = sum(l[0] for l in lots.get(sym, []))
                if cfg.discrete_shares and math.isfinite(rpx) and rpx > 0:
                    n_raw = int(tgt_r // rpx)
                    tgt_sh = (n_raw * rpx) / px
                else:
                    tgt_sh = tgt_r / px
                if tgt_sh > cur_sh + 1e-9:
                    buy_sh = tgt_sh - cur_sh
                    needed = buy_sh * px * (1.0 + active_cost_rate)
                    if needed > cash:
                        _sell_sweep_if_needed(needed - cash)
                    if needed > cash:
                        if cfg.discrete_shares and math.isfinite(rpx) and rpx > 0:
                            affordable_raw = int(cash // (rpx * (1.0 + active_cost_rate)))
                            buy_sh = (affordable_raw * rpx) / px
                        else:
                            buy_sh = cash / (px * (1.0 + active_cost_rate))
                        needed = buy_sh * px * (1.0 + active_cost_rate)
                    if buy_sh > 1e-9:
                        fee_b = buy_sh * px * active_cost_rate
                        total_fees_paid += fee_b
                        cash -= needed
                        traded_notional += buy_sh * px
                        lots.setdefault(sym, []).append([float(buy_sh), px, dstr])
                        last_valid_px[sym] = px
                        adj_px = float(adj_row.get(sym, px))
                        if math.isfinite(adj_px) and adj_px > 0:
                            stock_peaks[sym] = max(stock_peaks.get(sym, adj_px), adj_px)

            # 3) Park remaining cash into defensive asset if defensive allocation > 0 (or non-discrete residual)
            if target_def_rupees > def_val_before + cash:
                _sell_sweep_if_needed(target_def_rupees - def_val_before - cash)
            if not cfg.discrete_shares and cash > 1.0 and target_def_rupees > 1.0:
                buy_def_val = cash / (1.0 + active_cost_rate)
                add_u = buy_def_val / def_p
                fee_db = buy_def_val * active_cost_rate
                total_fees_paid += fee_db
                def_lots.append([float(add_u), def_p, dstr])
                traded_notional += buy_def_val
                cash = 0.0
            elif cfg.discrete_shares and target_def_rupees > def_p and cash > def_p * (1.0 + active_cost_rate):
                buy_def_u = int(min(cash, target_def_rupees) // (def_p * (1.0 + active_cost_rate)))
                if buy_def_u > 0:
                    fee_db = buy_def_u * def_p * active_cost_rate
                    total_fees_paid += fee_db
                    cash -= buy_def_u * def_p + fee_db
                    def_lots.append([float(buy_def_u), def_p, dstr])
                    traded_notional += buy_def_u * def_p

            # 4) Sweep any residual uninvested discrete cash into LIQUID_REALISTIC / LIQUIDCASE if enabled
            if cfg.discrete_shares and cfg.sweep_idle_cash and sweep_p > 0 and cash > sweep_p * (1.0 + active_cost_rate):
                buy_sw_u = int(cash // (sweep_p * (1.0 + active_cost_rate)))
                if buy_sw_u > 0:
                    fee_sb = buy_sw_u * sweep_p * active_cost_rate
                    total_fees_paid += fee_sb
                    cash -= buy_sw_u * sweep_p + fee_sb
                    sweep_lots.append([float(buy_sw_u), sweep_p, dstr])
                    traded_notional += buy_sw_u * sweep_p

        # End-of-day mark to market (strictly last valid traded close; NEVER purchase lot cost)
        eq_val_after = 0.0
        for sym, sym_lots in lots.items():
            px = float(p_row.get(sym, np.nan))
            if math.isfinite(px) and px > 0:
                last_valid_px[sym] = px
                mtm_px = px
            else:
                mtm_px = last_valid_px.get(sym, 0.0)
            eq_val_after += sum(l[0] for l in sym_lots) * mtm_px
        def_units_end = sum(l[0] for l in def_lots)
        sweep_units_end = sum(l[0] for l in sweep_lots)
        def_val_after = def_units_end * def_p + sweep_units_end * sweep_p
        nav_after = eq_val_after + def_val_after + cash

        nav_list.append(nav_after)
        turnover_list.append((traded_notional / 2.0) / nav_after if nav_after > 0 else 0.0)
        eq_wt_list.append(eq_val_after / nav_after if nav_after > 0 else 0.0)
        def_wt_list.append(def_val_after / nav_after if nav_after > 0 else 0.0)
        avg_holdings_list.append(len(lots))

        prev_prices = adj_row

    nav_s = pd.Series(nav_list, index=dates)
    to_s = pd.Series(turnover_list, index=dates)
    metrics = compute_metrics(nav_s, to_s, total_tax=total_tax)
    metrics["avg_eq_pct"] = round(float(np.mean(eq_wt_list) * 100.0), 1)
    metrics["avg_def_pct"] = round(float(np.mean(def_wt_list) * 100.0), 1)
    metrics["avg_holdings"] = round(float(np.mean(avg_holdings_list)), 1)
    metrics["total_fees_paid"] = round(total_fees_paid, 2)
    metrics["defensive_tax_paid"] = round(total_def_tax, 2)
    metrics["delisted_write_downs"] = int(delisted_write_downs)
    metrics["regime_shift_trades"] = int(regime_shift_trades)
    return {"nav": nav_s, "turnover": to_s, "eq_wt": pd.Series(eq_wt_list, index=dates), "metrics": metrics}
