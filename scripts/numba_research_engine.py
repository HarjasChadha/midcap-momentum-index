"""Massive Numba-accelerated Multi-Period Optimization & Validation Engine (2010-2026 Zero-Leak NSE PIT).

Tests and eliminates arbitrary constants across:
1. Equity Factor, Lookback, Skip, TopN1+TopN2 Split, Rebalance Cadence & Rank Buffer (ETF-Free PIT)
2. Exhaustive Drawdown Trigger Surface (dd_win x dd_thresh x shift x plateau neighborhood)
3. Re-Entry Mechanics (reenter_dd_ratio x MA_len x confirm_days x breadth_thrust_override x staggered scale-in)
4. Orthogonal 4-Block Multi-Factor Regime Mixture + SciPy Out-of-Sample Walk-Forward Weight Optimization + Dual Gates + Barroso Semivol
5. Exhaustive Hedging Frontier: w_gold in [0..1], Trend-Gated Gold, Gold+Gilt+MON100 Risk-Parity (TG-ERC), Dual-Mom, LowVol30 Equity, Bear-Vol, and F&O Futures/Put Drag Model
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.optimize import minimize

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nse_archive_loader import build_2010_2026_nse_dataset  # noqa: E402
from src.regime import build_defensive_series, build_regime_features  # noqa: E402

PERIODS = {
    "P1": ("2011-01-03", "2016-05-31"),
    "P2": ("2016-06-01", "2021-06-30"),
    "P3": ("2021-07-01", "2026-11-26"),
    "FULL": ("2011-01-03", "2026-11-26"),
}


# ── Numba Core Kernels ──────────────────────────────────────────────────────────────────────────────

@njit
def _calc_metrics_slice(nav: np.ndarray, to_arr: np.ndarray, s_idx: int, e_idx: int, rf: float = 0.0625) -> tuple[float, float, float, float, float]:
    """Return (cagr_pct, max_dd_pct, calmar, sharpe, ann_to_pct) over [s_idx, e_idx]."""
    n = e_idx - s_idx + 1
    if n < 30:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    yrs = n / 247.0  # NSE averages ~247 trading days per calendar year
    total = nav[e_idx] / nav[s_idx]
    cagr = (total ** (1.0 / yrs) - 1.0) if total > 0 else -0.99

    peak = nav[s_idx]
    max_dd = 0.0
    sum_r = 0.0
    sum_r2 = 0.0
    sum_to = 0.0
    cnt = n - 1
    for i in range(s_idx + 1, e_idx + 1):
        v = nav[i]
        if v > peak:
            peak = v
        dd = v / peak - 1.0
        if dd < max_dd:
            max_dd = dd
        r = v / nav[i - 1] - 1.0
        sum_r += r
        sum_r2 += r * r
        sum_to += to_arr[i]

    mean_r = sum_r / cnt
    var_r = max(1e-12, (sum_r2 / cnt) - mean_r * mean_r)
    ann_vol = math.sqrt(var_r * 247.0)
    calmar = cagr / abs(max_dd) if max_dd < -1e-6 else 0.0
    sharpe = (cagr - rf) / ann_vol if ann_vol > 1e-6 else 0.0
    ann_to = (sum_to / yrs) * 100.0
    return cagr * 100.0, max_dd * 100.0, calmar, sharpe, ann_to


@njit
def _sim_overlay_core(
    r_eq: np.ndarray,
    to_eq_reb: np.ndarray,
    eq_frac: np.ndarray,
    def_ret: np.ndarray,
    def_tax_rate: np.ndarray,  # daily blended defensive tax rate [0.125..0.30]
    cost_rate: float = 0.0015,
    apply_tax: bool = True,
    stcg_rate: float = 0.20,
    ltcg_rate: float = 0.125,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast tax-and-friction-aware daily simulation from precomputed equity basket stream + defensive asset stream."""
    T = len(r_eq)
    nav = np.ones(T, dtype=np.float64) * 100_000.0
    to_daily = np.zeros(T, dtype=np.float64)
    w_prev = eq_frac[0]
    eq_cost_basis = 1.0
    eq_idx = 1.0
    hold_days = 0.0
    def_basis = 1.0
    def_idx = 1.0

    for t in range(1, T):
        v_prev = nav[t - 1]
        w_cur = eq_frac[t]
        dw = w_cur - w_prev
        trade_notional = 0.0

        # 1. Rebalance / Regime Switch Frictions & Tax Drag on Close t (target w_cur comes from t-1 signal!)
        eq_idx *= (1.0 + r_eq[t])
        def_idx *= (1.0 + def_ret[t])
        gross_ret = w_prev * r_eq[t] + (1.0 - w_prev) * def_ret[t]
        v_mtm = v_prev * (1.0 + gross_ret)

        if abs(dw) > 0.005:
            # Regime switch equity <-> defensive
            sell_eq = max(0.0, -dw)
            sell_def = max(0.0, dw)
            trade_notional += abs(dw)  # half 2-way per rebalance definition -> |dw| equity + |dw| def = 2|dw|, /2 = |dw|
            fric = 2.0 * abs(dw) * cost_rate
            tax_drag = 0.0
            if apply_tax and sell_eq > 1e-6 and eq_idx > eq_cost_basis:
                gain_frac = (eq_idx - eq_cost_basis) / eq_idx
                t_rate = ltcg_rate if hold_days >= 365.0 else stcg_rate
                tax_drag += sell_eq * gain_frac * t_rate * 0.85  # partial exemption/loss-offset factor calibrated vs FIFO
            if apply_tax and sell_def > 1e-6 and def_idx > def_basis:
                gain_frac_d = (def_idx - def_basis) / def_idx
                tax_drag += sell_def * gain_frac_d * def_tax_rate[t]
            if dw > 0.0:
                # Reset weighted equity basis & holding clock proportionally on inflows
                in_w = dw / max(1e-6, w_cur)
                eq_cost_basis = (1.0 - in_w) * eq_cost_basis + in_w * eq_idx
                hold_days *= (1.0 - in_w)
            elif w_cur < 0.01:
                hold_days = 0.0
            if dw < 0.0:
                in_d = (-dw) / max(1e-6, 1.0 - w_cur)
                def_basis = (1.0 - in_d) * def_basis + in_d * def_idx
            v_mtm *= max(0.01, 1.0 - fric - tax_drag)

        # 2. Scheduled Equity Rebalance Friction & Tax Drag
        reb_f = to_eq_reb[t]
        if reb_f > 1e-6 and w_cur > 1e-4:
            eff_to = w_cur * reb_f  # fraction of total portfolio rotated in equity rebalance (2-way sum ~ 2*eff_to)
            trade_notional += eff_to
            fric_r = 2.0 * eff_to * cost_rate
            tax_r = 0.0
            if apply_tax and eq_idx > eq_cost_basis:
                gain_frac = (eq_idx - eq_cost_basis) / eq_idx
                t_rate = ltcg_rate if hold_days >= 365.0 else stcg_rate
                tax_r = eff_to * gain_frac * t_rate * 0.80
            eq_cost_basis = (1.0 - reb_f) * eq_cost_basis + reb_f * eq_idx
            hold_days = (1.0 - reb_f) * hold_days
            v_mtm *= max(0.01, 1.0 - fric_r - tax_r)

        hold_days += 1.47  # calendar days per NSE trading day
        nav[t] = v_mtm
        to_daily[t] = trade_notional
        w_prev = w_cur

    return nav, to_daily


@njit
def _gen_dd_signal(
    dd_arr: np.ndarray,
    px_arr: np.ndarray,
    ma_arr: np.ndarray,
    b50_chg21: np.ndarray,
    b50_lvl: np.ndarray,
    dd_thresh: float,
    reenter_ratio: float,
    confirm_days: int,
    thrust_on: int,
    shift_pct: float,
    min_hold: int,
) -> np.ndarray:
    """Strictly causal DD signal shifted +1d inside."""
    T = len(dd_arr)
    raw = np.ones(T, dtype=np.float64)
    def_active = False
    above_cnt = 0
    last_sw = -999
    for t in range(T):
        px = px_arr[t]
        ma = ma_arr[t]
        if np.isfinite(px) and np.isfinite(ma) and px > ma:
            above_cnt += 1
        else:
            above_cnt = 0
        thrust = (thrust_on == 1) and np.isfinite(b50_chg21[t]) and b50_chg21[t] >= 0.25 and b50_lvl[t] >= 0.50

        if not def_active:
            if np.isfinite(dd_arr[t]) and dd_arr[t] < -dd_thresh and (t - last_sw) >= min_hold:
                def_active = True
                last_sw = t
        else:
            if (t - last_sw) >= min_hold:
                dd_ok = True if reenter_ratio >= 0.999 else (np.isfinite(dd_arr[t]) and dd_arr[t] > -dd_thresh * reenter_ratio)
                ma_ok = (above_cnt >= confirm_days)
                if (dd_ok and ma_ok) or (thrust and ma_ok):
                    def_active = False
                    last_sw = t
        raw[t] = 1.0 - shift_pct if def_active else 1.0

    # Shift +1d for T+1 execution (zero look-ahead)
    out = np.ones(T, dtype=np.float64)
    for t in range(1, T):
        out[t] = raw[t - 1]
    return out


@njit(parallel=True)
def _batch_dd_reentry_grid(
    r_eq: np.ndarray,
    to_eq_reb: np.ndarray,
    def_ret: np.ndarray,
    def_tax: np.ndarray,
    dd_mat: np.ndarray,       # (W_dd, T)
    px_arr: np.ndarray,       # (T,)
    ma_mat: np.ndarray,       # (W_ma, T)
    b50_chg21: np.ndarray,
    b50_lvl: np.ndarray,
    params: np.ndarray,       # (N_cfg, 7): [w_dd_idx, dd_th, r_ratio, ma_idx, conf, thrust, shift, min_hold] -> cols 8
    idx_bounds: np.ndarray,   # (4, 2) start/end indices for P1, P2, P3, FULL
) -> np.ndarray:
    N = params.shape[0]
    # cols: P1(cagr,dd,cal,sh), P2(...), P3(...), FULL(cagr,dd,cal,sh,to)
    res = np.zeros((N, 17), dtype=np.float64)
    for k in prange(N):
        w_dd = int(params[k, 0])
        dd_th = params[k, 1]
        r_ratio = params[k, 2]
        w_ma = int(params[k, 3])
        conf = int(params[k, 4])
        thrust = int(params[k, 5])
        shift = params[k, 6]
        mhold = int(params[k, 7])
        eq_f = _gen_dd_signal(dd_mat[w_dd], px_arr, ma_mat[w_ma], b50_chg21, b50_lvl, dd_th, r_ratio, conf, thrust, shift, mhold)
        nav, to_d = _sim_overlay_core(r_eq, to_eq_reb, eq_f, def_ret, def_tax, 0.0015, True, 0.20, 0.125)
        for p in range(4):
            c, d, cal, sh, to_a = _calc_metrics_slice(nav, to_d, idx_bounds[p, 0], idx_bounds[p, 1], 0.0625)
            res[k, p * 4 + 0] = c
            res[k, p * 4 + 1] = d
            res[k, p * 4 + 2] = cal
            res[k, p * 4 + 3] = sh
            if p == 3:
                res[k, 16] = to_a
    return res


@njit
def _gen_multifactor_signal(
    p_comp: np.ndarray,
    b50_chg21: np.ndarray,
    b50_lvl: np.ndarray,
    port_semivol: np.ndarray,
    semivol_med: np.ndarray,
    tau_down: float,
    tau_up: float,
    th_def: float,
    th_caut: float,
    th_riskon: float,
    eq_floor: float,
    eq_caut: float,
    delta_band: float,
    use_bear_vol: int,
    min_hold_up: int = 0,
) -> np.ndarray:
    """Asymmetric EWMA + 3-state Schmitt-trigger hysteresis + asymmetric step-up cooldown (T+1 lagged)."""
    T = len(p_comp)
    raw = np.ones(T, dtype=np.float64)
    p_sm = 0.0
    state = 2
    cur_eq = 1.0
    last_dn_t = -999
    a_dn = 2.0 / (tau_down + 1.0)
    a_up = 2.0 / (tau_up + 1.0)
    for t in range(T):
        pt = p_comp[t] if np.isfinite(p_comp[t]) else p_sm
        thrust = np.isfinite(b50_chg21[t]) and (b50_chg21[t] >= 0.25) and (b50_lvl[t] >= 0.50)
        alpha = a_dn if (pt < p_sm or thrust) else a_up
        p_sm = alpha * pt + (1.0 - alpha) * p_sm

        can_up = (t - last_dn_t) >= min_hold_up or thrust
        if state == 2:
            if p_sm < th_def:
                state = 0
                last_dn_t = t
            elif p_sm < th_caut:
                state = 1
                last_dn_t = t
        elif state == 1:
            if p_sm < th_def:
                state = 0
                last_dn_t = t
            elif can_up and (p_sm > th_riskon or (thrust and p_sm > th_caut)):
                state = 2
        else:
            if can_up and (p_sm > (th_def + 0.15) or (thrust and p_sm > (th_def + 0.05))):
                state = 1

        w_base = 1.0 if state == 2 else (eq_caut if state == 1 else eq_floor)
        if use_bear_vol == 1 and state < 2 and np.isfinite(port_semivol[t]) and port_semivol[t] > 0.05:
            w_vol = min(1.0, max(eq_floor, semivol_med[t] / port_semivol[t]))
            w_raw = min(w_base, w_vol)
        else:
            w_raw = w_base

        if abs(w_raw - cur_eq) >= delta_band or (state == 0 and cur_eq > eq_floor + 0.01 and abs(w_raw - eq_floor) < 0.02):
            cur_eq = w_raw
        raw[t] = cur_eq

    out = np.ones(T, dtype=np.float64)
    for t in range(1, T):
        out[t] = raw[t - 1]
    return out


# ── Factor & Basket Precomputations ──────────────────────────────────────────────────────────────────

def compute_extended_scores(adj_eq: pd.DataFrame, pit_mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute canonical equity momentum & low-vol scores masked cleanly to lagged ETF-free PIT universe."""
    rets = adj_eq.pct_change(fill_method=None)
    # Clip extreme single-day corporate-action/data glitches when computing vol/FIP stats (already CA cleaned)
    rets_clip = rets.clip(-0.40, 0.60)
    down_rets2 = np.minimum(rets_clip, 0.0) ** 2
    lag_m = pit_mask.shift(1, fill_value=False)

    out: dict[str, pd.DataFrame] = {}
    for lb in (63, 84, 105, 126, 147, 168, 189, 210, 231, 252):
        for skip in (0, 10, 21):
            if lb not in (126, 189, 252) and skip != 21:
                continue
            p_end = adj_eq.shift(skip) if skip > 0 else adj_eq
            p_start = adj_eq.shift(skip + lb)
            min_obs = max(20, int(0.85 * lb))
            obs = adj_eq.rolling(lb + skip, min_periods=min_obs).count()
            raw_ret = ((p_end / p_start) - 1.0).where(adj_eq.notna() & p_end.notna() & p_start.notna() & (obs >= min_obs))
            out[f"raw_{lb}_s{skip}"] = raw_ret.where(lag_m)

            if skip in (10, 21) or lb in (126, 189, 252):
                vol = rets_clip.rolling(lb, min_periods=max(20, int(0.8 * lb))).std()
                sdev = np.sqrt(down_rets2.rolling(lb, min_periods=max(20, int(0.8 * lb))).mean() * 247.0)
                if skip > 0:
                    vol = vol.shift(skip)
                    sdev = sdev.shift(skip)
                ann_vol = (vol * math.sqrt(247.0)).clip(lower=0.08)
                sdev_clip = sdev.clip(lower=0.06)
                out[f"risk_{lb}_s{skip}"] = (raw_ret / ann_vol).where(lag_m)
                out[f"sortino_{lb}_s{skip}"] = (raw_ret / sdev_clip).where(lag_m)

            if skip == 21:
                # FIP (Frog-in-the-Pan consistency): sign(ret) -> raw * (%pos - %neg)
                pos_f = (rets_clip > 0.0005).rolling(lb, min_periods=max(20, int(0.8 * lb))).mean()
                neg_f = (rets_clip < -0.0005).rolling(lb, min_periods=max(20, int(0.8 * lb))).mean()
                if skip > 0:
                    pos_f = pos_f.shift(skip)
                    neg_f = neg_f.shift(skip)
                fip = raw_ret * (pos_f - neg_f).clip(lower=0.0)
                out[f"fip_{lb}_s{skip}"] = fip.where(lag_m)

    # Hybrids & Ensembles at standard skip=21
    for tag, cols_w in [
        ("hybrid_raw_risk_189", [("raw_189_s21", 0.5), ("risk_189_s21", 0.5)]),
        ("hybrid_raw_fip_189", [("raw_189_s21", 0.5), ("fip_189_s21", 0.5)]),
        ("ens_risk_6_9_12", [("risk_126_s21", 1/3), ("risk_189_s21", 1/3), ("risk_252_s21", 1/3)]),
        ("ens_raw_6_9_12", [("raw_126_s21", 1/3), ("raw_189_s21", 1/3), ("raw_252_s21", 1/3)]),
    ]:
        z_acc = None
        for c, wt in cols_w:
            df_c = out[c]
            z_c = df_c.sub(df_c.mean(axis=1), axis=0).div(df_c.std(axis=1).replace(0, np.nan), axis=0)
            z_acc = z_c * wt if z_acc is None else z_acc + z_c * wt
        out[tag] = z_acc.where(lag_m)

    # Internal Low-Vol Equity Defense Factor (negative 63d realized volatility -> highest score = lowest vol!)
    v63 = (rets_clip.rolling(63, min_periods=30).std() * math.sqrt(247.0)).clip(lower=0.08)
    out["lowvol_63"] = (-v63).where(lag_m)
    return out


def build_basket_daily_stream(
    adj_eq: pd.DataFrame,
    scores_df: pd.DataFrame,
    reb_days: int = 126,
    n_core: int = 20,
    n_broad: int = 30,
    w_core: float = 0.5,
    weighting: str = "equal",
    buffer_n: int = 0,
    delist_haircut: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact zero-look-ahead gross equity basket stream (T,), rebalance 1-way turnover (T,), and T-1 NAV series."""
    dates = adj_eq.index
    T = len(dates)
    adj_np = adj_eq.ffill(limit=5).values
    is_valid = adj_eq.notna().values  # True only when stock actually traded (delist detection when False for >5d)
    sc_np = scores_df.values
    # Precompute row means/stds for NormZ weighting
    r_daily = np.zeros(T, dtype=np.float64)
    to_reb = np.zeros(T, dtype=np.float64)

    n_cols = adj_np.shape[1]
    w_cur = np.zeros(n_cols, dtype=np.float64)
    held_idx = np.empty(0, dtype=np.int64)
    missing_streak = np.zeros(n_cols, dtype=np.int32)

    for t in range(T):
        if t == 0:
            pass
        else:
            # Update missing streak and compute daily portfolio return from t-1 -> t
            port_r = 0.0
            w_alive = 0.0
            delist_exit = 0.0
            for j in held_idx:
                if is_valid[t, j]:
                    missing_streak[j] = 0
                else:
                    missing_streak[j] += 1
                p0 = adj_np[t - 1, j]
                p1 = adj_np[t, j]
                if np.isfinite(p0) and p0 > 0 and np.isfinite(p1) and p1 > 0:
                    port_r += w_cur[j] * (p1 / p0 - 1.0)
                    w_alive += w_cur[j]
                elif w_cur[j] > 0:
                    # Frozen/delisted past limit=5: apply delisting gap haircut cleanly
                    port_r += w_cur[j] * (-delist_haircut)
                    delist_exit += w_cur[j]
                    w_cur[j] = 0.0
            r_daily[t] = port_r
            if delist_exit > 0:
                to_reb[t] += delist_exit * 0.5
                s_w = np.sum(w_cur)
                if s_w > 0:
                    w_cur /= s_w

        # Update weights on schedule t % reb_days == 0 using score observed strictly at t-1!
        sc_t = t - 1 if t > 0 else 0
        if t == 0 or (t % reb_days == 0):
            row = sc_np[sc_t]
            valid_idx = np.where(np.isfinite(row) & np.isfinite(adj_np[t]) & (adj_np[t] > 0))[0]
            if len(valid_idx) >= 5:
                vals = row[valid_idx]
                order = np.argsort(-vals)
                sorted_cols = valid_idx[order]

                def _pick_n(n_pick: int, prev_set: set[int]) -> np.ndarray:
                    if buffer_n > n_pick and len(prev_set) > 0:
                        elig = sorted_cols[: min(len(sorted_cols), buffer_n)]
                        kept = [int(c) for c in elig if int(c) in prev_set][:n_pick]
                        if len(kept) < n_pick:
                            kset = set(kept)
                            for c in elig:
                                if int(c) not in kset:
                                    kept.append(int(c))
                                    kset.add(int(c))
                                    if len(kept) == n_pick:
                                        break
                        return np.array(kept, dtype=np.int64)
                    return sorted_cols[: min(len(sorted_cols), n_pick)]

                prev_set = set(int(x) for x in held_idx)
                p1_idx = _pick_n(n_core, prev_set) if w_core > 0 else np.empty(0, dtype=np.int64)
                p2_idx = _pick_n(n_broad, prev_set) if (1.0 - w_core) > 0 else np.empty(0, dtype=np.int64)

                w_new = np.zeros(n_cols, dtype=np.float64)
                if weighting == "norm_z":
                    mu = float(np.mean(vals))
                    sd = max(1e-8, float(np.std(vals)))
                    for sl_idx, alloc_s in [(p1_idx, w_core), (p2_idx, 1.0 - w_core)]:
                        if len(sl_idx) > 0 and alloc_s > 0:
                            z = (row[sl_idx] - mu) / sd
                            nz = np.where(z >= 0, 1.0 + z, 1.0 / (1.0 - z))
                            nz /= np.sum(nz)
                            w_new[sl_idx] += alloc_s * nz
                else:
                    for sl_idx, alloc_s in [(p1_idx, w_core), (p2_idx, 1.0 - w_core)]:
                        if len(sl_idx) > 0 and alloc_s > 0:
                            w_new[sl_idx] += alloc_s / len(sl_idx)

                s_tot = np.sum(w_new)
                if s_tot > 0:
                    w_new /= s_tot
                    if t > 0:
                        to_reb[t] += 0.5 * float(np.sum(np.abs(w_new - w_cur)))
                    w_cur = w_new
                    held_idx = np.where(w_cur > 1e-9)[0]

    nav_gross = np.cumprod(1.0 + r_daily) * 100.0
    return r_daily, to_reb, nav_gross


def causal_mad_squash(series: pd.Series, win: int = 756, min_p: int = 126) -> np.ndarray:
    """Zero-look-ahead causal rolling median/MAD normalization -> tanh(z/2) in [-1, 1]."""
    arr = series.values.astype(np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    for t in range(len(arr)):
        v = arr[t]
        if not math.isfinite(v):
            out[t] = out[t - 1] if t > 0 else 0.0
            continue
        s0 = max(0, t - win + 1)
        sub = arr[s0 : t + 1]
        sub = sub[np.isfinite(sub)]
        if len(sub) < min_p:
            out[t] = 0.0
            continue
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) * 1.4826
        z = (v - med) / max(mad, 1e-6)
        out[t] = math.tanh(0.5 * z)
    return out


def build_multifactor_blocks(
    feat: pd.DataFrame,
    bench_df: pd.DataFrame,
    port_nav: np.ndarray,
    def_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact orthogonal 4-block matrix (4, T), downside semivol (T,), and causal median semivol (T,)."""
    ew = feat["ew_index"]
    ew_r = ew.pct_change(fill_method=None).fillna(0.0)
    ew_ema200 = ew.ewm(span=200, min_periods=60).mean()
    ew_v63 = np.sqrt(247.0 * (ew_r ** 2).ewm(span=63, min_periods=20).mean()).clip(lower=0.08)
    t_ew = ((ew / ew_ema200) - 1.0) / ew_v63

    p_s = pd.Series(port_nav, index=feat.index)
    p_r = p_s.pct_change(fill_method=None).fillna(0.0)
    p_ema126 = p_s.ewm(span=126, min_periods=40).mean()
    p_v63 = np.sqrt(247.0 * (p_r ** 2).ewm(span=63, min_periods=20).mean()).clip(lower=0.08)
    t_port = ((p_s / p_ema126) - 1.0) / p_v63

    nif = def_df["NIFTYBEES"].reindex(feat.index).ffill(limit=5)
    rel = ew / nif
    rel_r = rel.pct_change(fill_method=None).fillna(0.0)
    rel_ema126 = rel.ewm(span=126, min_periods=40).mean()
    rel_v63 = np.sqrt(247.0 * (rel_r ** 2).ewm(span=63, min_periods=20).mean()).clip(lower=0.04)
    t_rel = ((rel / rel_ema126) - 1.0) / rel_v63

    b1 = (causal_mad_squash(t_ew) + causal_mad_squash(t_port) + causal_mad_squash(t_rel)) / 3.0

    # Block 2: Downside portfolio semivol + India VIX spike ratio (negated: +1=calm, -1=stress)
    p_neg2 = np.minimum(p_r, 0.0) ** 2
    p_semivol = np.sqrt(247.0 * p_neg2.rolling(21, min_periods=10).mean()).fillna(0.125)
    semivol_med = p_semivol.rolling(756, min_periods=63).median().fillna(0.125).values
    vix = feat["vix"] if "vix" in feat else pd.Series(16.0, index=feat.index)
    vix_ema63 = vix.ewm(span=63, min_periods=20).mean()
    vix_spike = np.log((vix / vix_ema63).clip(lower=0.2))
    b2 = -(causal_mad_squash(np.log(p_semivol.clip(lower=0.04))) + causal_mad_squash(vix_spike)) / 2.0

    # Block 3: Breadth level & 21d Zweig thrust
    b50 = feat["breadth_50"].fillna(0.5)
    b50_ema5 = b50.ewm(span=5, min_periods=2).mean()
    b50_chg21 = b50_ema5 - b50_ema5.shift(21)
    b3 = (causal_mad_squash(b50) + causal_mad_squash(b50_chg21.fillna(0.0))) / 2.0

    # Block 4: Daniel-Moskowitz Crash State + Gold Safe-Haven Trend (negated)
    dm_raw = np.where(t_ew.values < 0.0, np.maximum(0.0, causal_mad_squash(np.log(p_semivol.clip(lower=0.04)))), 0.0)
    gold = def_df["GOLDBEES"].reindex(feat.index).ffill(limit=5)
    g_r = gold.pct_change(fill_method=None).fillna(0.0)
    g_ema126 = gold.ewm(span=126, min_periods=40).mean()
    g_v63 = np.sqrt(247.0 * (g_r ** 2).ewm(span=63, min_periods=20).mean()).clip(lower=0.06)
    t_gold = ((gold / g_ema126) - 1.0) / g_v63
    b4 = -(dm_raw + causal_mad_squash(t_gold)) / 2.0

    blocks = np.vstack([b1, b2, b3, b4])
    return blocks, p_semivol.values.astype(np.float64), semivol_med.astype(np.float64)


# ── Trend-Gated ERC Defensive Asset Weights (Numba) ─────────────────────────────────────────────────

@njit
def _erc_coord(cov: np.ndarray, n_iter: int = 20) -> np.ndarray:
    m = cov.shape[0]
    y = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-10))
    y /= np.sum(y)
    b = 1.0 / m
    for _ in range(n_iter):
        for j in range(m):
            c_j = 0.0
            for k in range(m):
                if k != j:
                    c_j += cov[j, k] * y[k]
            disc = c_j * c_j + 4.0 * cov[j, j] * b
            y[j] = (-c_j + math.sqrt(max(disc, 1e-16))) / (2.0 * cov[j, j])
        s = np.sum(y)
        if s > 0.0:
            y /= s
    return y


@njit
def compute_tg_erc_returns(
    def_prices: np.ndarray,  # (T, M) risky assets [Gold, Gilt/MON100...]
    def_rets: np.ndarray,    # (T, M)
    liq_ret: np.ndarray,     # (T,) risk-free sink return
    def_caps: np.ndarray,    # (M,)
    def_tax_r: np.ndarray,   # (M,) asset tax rates
    liq_tax: float = 0.30,
    ema_len: int = 100,
    band_pct: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal Trend-Gated Equal-Risk-Contribution portfolio + LIQUID sink (zero look-ahead T+1 lagged weights)."""
    T, M = def_prices.shape
    port_ret = np.zeros(T, dtype=np.float64)
    port_tax = np.zeros(T, dtype=np.float64)
    w_hist = np.zeros((T, M + 1), dtype=np.float64)

    # Causal EMA computation
    ema = np.zeros((T, M), dtype=np.float64)
    alpha = 2.0 / (ema_len + 1.0)
    for j in range(M):
        started = False
        val = 0.0
        for t in range(T):
            px = def_prices[t, j]
            if np.isfinite(px) and px > 0:
                val = px if not started else (alpha * px + (1.0 - alpha) * val)
                started = True
                ema[t, j] = val
            else:
                ema[t, j] = np.nan

    g_gate = np.zeros(M, dtype=np.float64)
    w_prev = np.zeros(M + 1, dtype=np.float64)
    w_prev[M] = 1.0

    for t in range(T):
        # Compute today's return using weights decided yesterday (w_prev)!
        r_t = w_prev[M] * liq_ret[t]
        tx_t = w_prev[M] * liq_tax
        for j in range(M):
            rj = def_rets[t, j] if np.isfinite(def_rets[t, j]) else liq_ret[t]
            r_t += w_prev[j] * rj
            tx_t += w_prev[j] * def_tax_r[j]
        port_ret[t] = r_t
        port_tax[t] = tx_t
        for j in range(M + 1):
            w_hist[t, j] = w_prev[j]

        # Decide end-of-day t target weights for t+1
        active_cnt = 0
        for j in range(M):
            px, em = def_prices[t, j], ema[t, j]
            if np.isfinite(px) and np.isfinite(em) and em > 0:
                if px > (1.0 + band_pct) * em:
                    g_gate[j] = 1.0
                elif px < (1.0 - band_pct) * em:
                    g_gate[j] = 0.0
                active_cnt += 1
            else:
                g_gate[j] = 0.0

        w_next = np.zeros(M + 1, dtype=np.float64)
        if t >= 63 and active_cnt > 0:
            all_avail = True
            for j in range(M):
                for k in range(t - 62, t + 1):
                    if not np.isfinite(def_rets[k, j]):
                        all_avail = False
                        break
                if not all_avail:
                    break
            if all_avail:
                cov = np.zeros((M, M), dtype=np.float64)
                means = np.zeros(M, dtype=np.float64)
                for j in range(M):
                    s = 0.0
                    for k in range(t - 62, t + 1):
                        s += def_rets[k, j]
                    means[j] = s / 63.0
                for j in range(M):
                    for l in range(j, M):
                        c = 0.0
                        for k in range(t - 62, t + 1):
                            c += (def_rets[k, j] - means[j]) * (def_rets[k, l] - means[l])
                        val = (c / 62.0) * 247.0
                        if j == l:
                            cov[j, l] = max(val, 1e-6)
                        else:
                            cov[j, l] = 0.75 * val
                            cov[l, j] = 0.75 * val
                w_erc = _erc_coord(cov, 20)
                used = 0.0
                for j in range(M):
                    wj = min(def_caps[j], g_gate[j] * w_erc[j])
                    w_next[j] = wj
                    used += wj
                w_next[M] = max(0.0, 1.0 - used)
            else:
                # Fallback when Gilt or young ETF hasn't started yet: trend-gate GOLDBEES (idx 0) only!
                w0 = min(def_caps[0], g_gate[0])
                w_next[0] = w0
                w_next[M] = 1.0 - w0
        else:
            w_next[M] = 1.0

        # Switch cost inside defensive basket when >3% rebalance
        sw_cost = 0.5 * float(np.sum(np.abs(w_next - w_prev))) * 0.0015 if t > 0 else 0.0
        if np.sum(np.abs(w_next - w_prev)) > 0.03:
            port_ret[t] -= sw_cost
            w_prev = w_next

    return port_ret, port_tax, w_hist
