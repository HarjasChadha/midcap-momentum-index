"""Market regime, universe breadth, portfolio-internal risk signals, and defensive asset series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_ew_index(adj_df: pd.DataFrame, base: float = 100.0) -> pd.Series:
    """Equal-weighted daily index of active Nifty Midcap 150 constituents (survivorship-aware on IPO dates)."""
    daily_rets = adj_df.pct_change(fill_method=None)
    ew_ret = daily_rets.mean(axis=1, skipna=True).fillna(0.0)
    return base * (1.0 + ew_ret).cumprod()


def build_defensive_series(bench_df: pd.DataFrame, liquid_annual_rate: float = 0.0625) -> pd.DataFrame:
    """Construct clean daily NAV series (base 100) for all defensive asset candidates."""
    idx = bench_df.index
    out = pd.DataFrame(index=idx)

    # 1. Synthetic Indian Liquid / T-Bill (6.25% p.a. daily compounding — accurate proxy across 10y)
    daily_liq = (1.0 + liquid_annual_rate) ** (1.0 / 252.0) - 1.0
    out["LIQUID_TBILL"] = 100.0 * ((1.0 + daily_liq) ** np.arange(len(idx)))

    # 2. Clean ETFs aligned to common index (forward-filling rare single-day gaps)
    for col in ("GOLDBEES", "SILVERBEES", "EBBETF0431", "LTGILTBEES", "LIQUIDCASE", "NIFTYBEES", "MON100"):
        if col in bench_df.columns:
            s = bench_df[col].ffill()
            first = s.dropna()
            if not first.empty:
                out[col] = s / first.iloc[0] * 100.0

    # 3. Dynamic Trend-Filtered Gold vs Liquid (hold Gold when GOLDBEES > 50DMA at t-1, else Liquid)
    if "GOLDBEES" in out.columns:
        g = out["GOLDBEES"]
        g_ma50 = g.rolling(50, min_periods=20).mean()
        g_ret = g.pct_change().fillna(0.0)
        in_gold = (g > g_ma50).shift(1, fill_value=False).astype(bool)
        sw = (in_gold != in_gold.shift(1, fill_value=False)).astype(float) * 0.0010  # 10 bps switch friction
        dyn_ret = np.where(in_gold, g_ret, daily_liq) - sw.values
        out["GOLD_OR_LIQUID"] = 100.0 * np.cumprod(1.0 + dyn_ret)

        # 50/50 static Gold + Liquid
        mix_ret = 0.5 * g_ret + 0.5 * daily_liq
        out["GOLD_50_LIQUID_50"] = 100.0 * np.cumprod(1.0 + mix_ret)

    # 4. Trend-Filtered Gold vs Gilt vs Liquid (if LTGILTBEES > MA50 when Gold < MA50, else Liquid)
    if "GOLDBEES" in out.columns and "LTGILTBEES" in out.columns:
        g = out["GOLDBEES"]
        g_ma = g.rolling(50, min_periods=20).mean()
        g_ret = g.pct_change().fillna(0.0)
        lt = out["LTGILTBEES"]
        lt_ma = lt.rolling(50, min_periods=20).mean()
        lt_ret = lt.pct_change().fillna(0.0)
        use_g = (g > g_ma).shift(1, fill_value=False).astype(bool)
        use_lt = ((g <= g_ma) & (lt > lt_ma)).shift(1, fill_value=False).astype(bool)
        choice = np.where(use_g, 1, np.where(use_lt, 2, 0))
        sw2 = np.concatenate(([0.0], (choice[1:] != choice[:-1]).astype(float) * 0.0010))
        rot_ret = np.where(use_g, g_ret, np.where(use_lt, lt_ret, daily_liq)) - sw2
        out["TREND_GOLD_GILT_LIQ"] = 100.0 * np.cumprod(1.0 + rot_ret)

    return out


def build_regime_features(adj_df: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
    """Compute universe breadth, index trend/drawdown, and VIX features aligned at close t."""
    feat = pd.DataFrame(index=adj_df.index)

    # Equal-weight Midcap 150 universe index (full 10y coverage)
    ew = build_ew_index(adj_df)
    feat["ew_index"] = ew
    for lb in (20, 50, 100, 150, 200):
        feat[f"ew_ma{lb}"] = ew.rolling(lb, min_periods=max(10, lb // 2)).mean()
    feat["ew_dd252"] = ew / ew.rolling(252, min_periods=20).max() - 1.0

    # Official index where available (2019+)
    if "NIFTY_MIDCAP_150" in bench_df.columns:
        idx_s = bench_df["NIFTY_MIDCAP_150"].ffill()
        feat["idx_price"] = idx_s
        feat["idx_ma50"] = idx_s.rolling(50, min_periods=20).mean()
        feat["idx_ma200"] = idx_s.rolling(200, min_periods=50).mean()
        feat["idx_dd252"] = idx_s / idx_s.rolling(252, min_periods=20).max() - 1.0

    # Universe Breadth: % of active Midcap 150 constituents above their 50-DMA & 200-DMA
    ma50_stocks = adj_df.rolling(50, min_periods=35).mean()
    ma200_stocks = adj_df.rolling(200, min_periods=120).mean()
    active_50 = (adj_df.notna() & ma50_stocks.notna())
    active_200 = (adj_df.notna() & ma200_stocks.notna())

    b50 = (adj_df > ma50_stocks).where(active_50).mean(axis=1)
    b200 = (adj_df > ma200_stocks).where(active_200).mean(axis=1)
    feat["breadth_50"] = b50
    feat["breadth_50_sma10"] = b50.rolling(10, min_periods=3).mean()
    feat["breadth_200"] = b200
    feat["breadth_200_sma10"] = b200.rolling(10, min_periods=3).mean()

    # India VIX
    if "INDIA_VIX" in bench_df.columns:
        vix = bench_df["INDIA_VIX"].ffill()
        feat["vix"] = vix
        feat["vix_ma20"] = vix.rolling(20, min_periods=5).mean()
        feat["vix_ratio"] = vix / feat["vix_ma20"].replace(0.0, np.nan)

    return feat


def compute_equity_allocation_series(
    feat: pd.DataFrame,
    rule: str = "none",
    params: dict | None = None,
) -> pd.Series:
    """Return target equity fraction [0.0..1.0] observed at close t-1 (shifted +1 so zero look-ahead)."""
    params = params or {}
    idx = feat.index

    if rule == "none":
        return pd.Series(1.0, index=idx)

    if rule == "index_ma":
        lb = int(params.get("ma_len", 200))
        col = f"ew_ma{lb}"
        sig = (feat["ew_index"] > feat[col]).astype(float)
        sig[feat[col].isna()] = 1.0
        shift_pct = float(params.get("defensive_shift", 1.0))
        alloc = 1.0 - (1.0 - sig) * shift_pct
        return alloc.shift(1).fillna(1.0)

    if rule == "dual_ma":
        sig = (feat["ew_ma50"] > feat["ew_ma200"]).astype(float)
        sig[feat["ew_ma200"].isna()] = 1.0
        shift_pct = float(params.get("defensive_shift", 1.0))
        return (1.0 - (1.0 - sig) * shift_pct).shift(1).fillna(1.0)

    if rule == "index_dd":
        exit_dd = -abs(float(params.get("dd_thresh", 0.12)))
        reenter_above_ma = int(params.get("reenter_ma", 50))
        ma_col = f"ew_ma{reenter_above_ma}"
        state = np.ones(len(idx), dtype=float)
        eq = 1.0
        dds = feat["ew_dd252"].values
        prices = feat["ew_index"].values
        mas = feat[ma_col].values
        for i in range(len(idx)):
            if eq > 0.5 and np.isfinite(dds[i]) and dds[i] < exit_dd:
                eq = 1.0 - float(params.get("defensive_shift", 1.0))
            elif eq < 0.99 and np.isfinite(mas[i]) and prices[i] > mas[i] and dds[i] > exit_dd * 0.6:
                eq = 1.0
            state[i] = eq
        return pd.Series(state, index=idx).shift(1).fillna(1.0)

    if rule == "breadth_hysteresis":
        col = params.get("col", "breadth_50_sma10")
        low = float(params.get("exit_below", 0.30))
        high = float(params.get("enter_above", 0.45))
        shift_pct = float(params.get("defensive_shift", 1.0))
        vals = feat[col].values
        state = np.ones(len(idx), dtype=float)
        eq = 1.0
        for i, v in enumerate(vals):
            if np.isfinite(v):
                if eq > 0.5 and v < low:
                    eq = 1.0 - shift_pct
                elif eq < 0.99 and v > high:
                    eq = 1.0
            state[i] = eq
        return pd.Series(state, index=idx).shift(1).fillna(1.0)

    if rule == "vix_filter":
        vix_exit = float(params.get("vix_exit", 24.0))
        vix_enter = float(params.get("vix_enter", 20.0))
        shift_pct = float(params.get("defensive_shift", 0.5))
        vals = feat["vix"].values if "vix" in feat else np.full(len(idx), 15.0)
        state = np.ones(len(idx), dtype=float)
        eq = 1.0
        for i, v in enumerate(vals):
            if np.isfinite(v):
                if eq > 0.5 and v > vix_exit:
                    eq = 1.0 - shift_pct
                elif eq < 0.99 and v < vix_enter:
                    eq = 1.0
            state[i] = eq
        return pd.Series(state, index=idx).shift(1).fillna(1.0)

    if rule == "index_ma_band":
        lb = int(params.get("ma_len", 200))
        exit_band = float(params.get("exit_band", 0.97))
        enter_band = float(params.get("enter_band", 1.01))
        shift_pct = float(params.get("defensive_shift", 1.0))
        prices = feat["ew_index"].values
        mas = feat[f"ew_ma{lb}"].values
        state = np.ones(len(idx), dtype=float)
        eq = 1.0
        for i, (px, ma) in enumerate(zip(prices, mas)):
            if np.isfinite(px) and np.isfinite(ma):
                if eq > 0.5 and px < ma * exit_band:
                    eq = 1.0 - shift_pct
                elif eq < 0.99 and px > ma * enter_band:
                    eq = 1.0
            state[i] = eq
        return pd.Series(state, index=idx).shift(1).fillna(1.0)

    if rule == "tiered_composite":
        # 4-signal posture score [0..4] -> smooth allocation mapping (exposure-coach style)
        c1 = (feat["ew_index"] > feat["ew_ma200"]).fillna(True).astype(int)
        c2 = (feat["ew_index"] > feat["ew_ma50"]).fillna(True).astype(int)
        c3 = (feat["breadth_50_sma10"] >= float(params.get("b50_mid", 0.40))).fillna(True).astype(int)
        c4 = (feat["vix"] < float(params.get("vix_cap", 22.0))).fillna(True).astype(int) if "vix" in feat else 1
        score = c1 + c2 + c3 + c4
        if params.get("smooth_days", 0) > 1:
            score = score.rolling(int(params["smooth_days"]), min_periods=1).median().round().astype(int)
        mapping = params.get("mapping", {4: 1.0, 3: 1.0, 2: 0.60, 1: 0.25, 0: 0.0})
        alloc = score.map(mapping).astype(float)
        if params.get("min_hold_days", 0) > 1:
            mh = int(params["min_hold_days"])
            vals = alloc.values.copy()
            last_change = -mh
            cur = vals[0]
            for i in range(len(vals)):
                if abs(vals[i] - cur) > 0.05 and (i - last_change) >= mh:
                    cur = vals[i]
                    last_change = i
                vals[i] = cur
            alloc = pd.Series(vals, index=idx)
        return alloc.shift(1).fillna(1.0)

    raise ValueError(f"Unknown regime rule: {rule}")
