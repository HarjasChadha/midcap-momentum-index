"""Market regime, universe breadth, portfolio-internal risk signals, and defensive asset series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_ew_index(adj_df: pd.DataFrame, base: float = 100.0, mask: pd.DataFrame | None = None) -> pd.Series:
    """Equal-weighted daily index of active Nifty Midcap 150 constituents (strictly lagged T-1 mask, zero look-ahead)."""
    daily_rets = adj_df.pct_change(fill_method=None)
    if mask is not None:
        lagged_mask = mask.reindex_like(daily_rets).fillna(False).shift(1, fill_value=False)
        daily_rets = daily_rets.where(lagged_mask)
    ew_ret = daily_rets.mean(axis=1, skipna=True).fillna(0.0)
    return base * (1.0 + ew_ret).cumprod()


# Historical Indian 91-Day T-Bill / Overnight Repo Proxy (post-expense net yield by macro regime)
INDIA_TBILL_REGIME_YIELD = {
    2010: 0.0550, 2011: 0.0775, 2012: 0.0815, 2013: 0.0815, 2014: 0.0840,
    2015: 0.0750, 2016: 0.0660, 2017: 0.0615, 2018: 0.0640, 2019: 0.0565,
    2020: 0.0365, 2021: 0.0345, 2022: 0.0510, 2023: 0.0675, 2024: 0.0660,
    2025: 0.0615, 2026: 0.0575,
}


def build_defensive_series(bench_df: pd.DataFrame, liquid_annual_rate: float = 0.0625) -> pd.DataFrame:
    """Construct clean daily NAV series (base 100) for all defensive asset candidates."""
    idx = bench_df.index
    out = pd.DataFrame(index=idx)

    # 1. Synthetic constant Liquid / T-Bill (for unit tests & baseline comparisons)
    daily_liq = (1.0 + liquid_annual_rate) ** (1.0 / 252.0) - 1.0
    out["LIQUID_TBILL"] = 100.0 * ((1.0 + daily_liq) ** np.arange(len(idx)))

    # 1b. Realistic Regime-Varying India 91D T-Bill + Actual LIQUIDCASE (2024+)
    tbill_rates = np.array([INDIA_TBILL_REGIME_YIELD.get(int(str(d)[:4]), liquid_annual_rate) if str(d)[:4].isdigit() else liquid_annual_rate for d in idx], dtype=float)
    tbill_daily = (1.0 + tbill_rates) ** (1.0 / 252.0) - 1.0
    if "LIQUIDCASE" in bench_df.columns:
        lc_ret = bench_df["LIQUIDCASE"].pct_change(fill_method=None)
        tbill_daily = np.where(lc_ret.notna().values, lc_ret.fillna(0.0).values, tbill_daily)
    tbill_daily[0] = 0.0
    out["LIQUID_REALISTIC"] = 100.0 * np.cumprod(1.0 + tbill_daily)

    # 2. Clean ETFs & Indices aligned to common index (strictly NO backward look-ahead bfill!)
    for col in (
        "GOLDBEES", "SILVERBEES", "EBBETF0430", "EBBETF0431", "LTGILTBEES", "LICNETFGSC",
        "LIQUIDCASE", "NIFTYBEES", "JUNIORBEES", "BANKBEES", "HNGSNGBEES", "MON100",
        "NIFTY_MIDCAP_100", "NIFTY_MIDCAP_150",
    ):
        if col in bench_df.columns:
            s = bench_df[col].ffill(limit=5)
            first = s.dropna()
            if not first.empty:
                out[col] = s / first.iloc[0] * 100.0

    # 2b. Stitched N100 (2011-2021) -> MON100 (2021-2026) continuous INR Nasdaq 100
    if "N100" in bench_df.columns and "MON100" in bench_df.columns:
        n100_s = bench_df["N100"].ffill(limit=5) * 0.1
        mon_s = bench_df["MON100"].ffill(limit=5)
        stitched = mon_s.combine_first(n100_s)
        first = stitched.dropna()
        if not first.empty:
            out["MON100_STITCHED"] = stitched / first.iloc[0] * 100.0

    # 3. Dynamic Trend-Filtered Gold vs Liquid (unified LIQUID_REALISTIC across ALL presets!)
    liq_ref = out["LIQUID_REALISTIC"]
    liq_ref_ret = liq_ref.pct_change(fill_method=None).fillna(0.0).values

    if "GOLDBEES" in out.columns:
        g = out["GOLDBEES"]
        g_ma50 = g.rolling(50, min_periods=20).mean()
        g_ret = g.pct_change(fill_method=None).fillna(0.0)
        in_gold = (g > g_ma50).shift(1, fill_value=False).astype(bool)
        sw = (in_gold != in_gold.shift(1, fill_value=False)).astype(float) * 0.0010  # 10 bps switch friction
        dyn_ret = np.where(in_gold, g_ret.values, liq_ref_ret) - sw.values
        out["GOLD_OR_LIQUID"] = 100.0 * np.cumprod(1.0 + dyn_ret)

        # 50/50 static Gold + LIQUID_REALISTIC
        mix_ret = 0.5 * g_ret.values + 0.5 * liq_ref_ret
        out["GOLD_50_LIQUID_50"] = 100.0 * np.cumprod(1.0 + mix_ret)

        # Research-validated Trend-Gated Gold (EMA150 +/-2% hysteresis band -> LIQUID_REALISTIC)
        g_arr = g.ffill(limit=5).values
        g_ema150 = g.ffill(limit=5).ewm(span=150, min_periods=30).mean().values
        in_gold_150 = np.zeros(len(idx), dtype=bool)
        state_g = False
        for t in range(len(idx)):
            px, em = g_arr[t], g_ema150[t]
            if np.isfinite(px) and np.isfinite(em) and em > 0:
                if not state_g and px > 1.02 * em:
                    state_g = True
                elif state_g and px < 0.98 * em:
                    state_g = False
            in_gold_150[t] = state_g
        lag_g150 = np.concatenate(([False], in_gold_150[:-1]))
        sw_g150 = np.concatenate(([0.0], (lag_g150[1:] != lag_g150[:-1]).astype(float) * 0.0015))
        tg150_ret = np.where(lag_g150, g_ret.values, liq_ref_ret) - sw_g150
        out["TREND_GOLD150_OR_LIQ"] = 100.0 * np.cumprod(1.0 + tg150_ret)
        out["TREND_GOLD150_ACTIVE"] = lag_g150.astype(float)
        out["TG70_LIQ30"] = 100.0 * np.cumprod(1.0 + (0.70 * tg150_ret + 0.30 * liq_ref_ret))

    # 4. Trend-Filtered Gold vs Gilt vs LIQUID_REALISTIC
    if "GOLDBEES" in out.columns and "LTGILTBEES" in out.columns:
        g = out["GOLDBEES"]
        g_ma = g.rolling(50, min_periods=20).mean()
        g_ret = g.pct_change(fill_method=None).fillna(0.0)
        lt = out["LTGILTBEES"]
        lt_ma = lt.rolling(50, min_periods=20).mean()
        lt_ret = lt.pct_change(fill_method=None).fillna(0.0)
        use_g = (g > g_ma).shift(1, fill_value=False).astype(bool)
        use_lt = ((g <= g_ma) & (lt > lt_ma)).shift(1, fill_value=False).astype(bool)
        choice = np.where(use_g, 1, np.where(use_lt, 2, 0))
        sw2 = np.concatenate(([0.0], (choice[1:] != choice[:-1]).astype(float) * 0.0010))
        rot_ret = np.where(use_g, g_ret.values, np.where(use_lt, lt_ret.values, liq_ref_ret)) - sw2
        out["TREND_GOLD_GILT_LIQ"] = 100.0 * np.cumprod(1.0 + rot_ret)

    return out


def build_regime_features(adj_df: pd.DataFrame, bench_df: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute universe breadth, index trend/drawdown, and VIX features aligned at close t (zero look-ahead)."""
    feat = pd.DataFrame(index=adj_df.index)

    # Equal-weight Midcap 150 universe index (uses strictly lagged mask inside build_ew_index)
    ew = build_ew_index(adj_df, mask=mask)
    feat["ew_index"] = ew
    for lb in (20, 50, 100, 150, 200):
        feat[f"ew_ma{lb}"] = ew.rolling(lb, min_periods=max(10, lb // 2)).mean()
    feat["ew_dd252"] = ew / ew.rolling(252, min_periods=20).max() - 1.0
    feat["ew_dd126"] = ew / ew.rolling(126, min_periods=20).max() - 1.0
    feat["ew_dd_ath"] = ew / ew.cummax() - 1.0

    # Official index where available
    if "NIFTY_MIDCAP_150" in bench_df.columns:
        idx_s = bench_df["NIFTY_MIDCAP_150"].ffill(limit=5)
        feat["idx_price"] = idx_s
        feat["idx_ma50"] = idx_s.rolling(50, min_periods=20).mean()
        feat["idx_ma200"] = idx_s.rolling(200, min_periods=50).mean()
        feat["idx_dd252"] = idx_s / idx_s.rolling(252, min_periods=20).max() - 1.0

    # Universe Breadth: % of active constituents (lagged T-1 mask) above their 50-DMA & 200-DMA
    ma50_stocks = adj_df.rolling(50, min_periods=35).mean()
    ma200_stocks = adj_df.rolling(200, min_periods=120).mean()
    active_50 = (adj_df.notna() & ma50_stocks.notna())
    active_200 = (adj_df.notna() & ma200_stocks.notna())
    if mask is not None:
        m_aligned = mask.reindex_like(adj_df).fillna(False).shift(1, fill_value=False)
        active_50 = active_50 & m_aligned
        active_200 = active_200 & m_aligned

    b50 = (adj_df > ma50_stocks).where(active_50).mean(axis=1)
    b200 = (adj_df > ma200_stocks).where(active_200).mean(axis=1)
    feat["breadth_50"] = b50
    feat["breadth_50_sma10"] = b50.rolling(10, min_periods=3).mean()
    feat["breadth_200"] = b200
    feat["breadth_200_sma10"] = b200.rolling(10, min_periods=3).mean()

    # India VIX
    if "INDIA_VIX" in bench_df.columns:
        vix = bench_df["INDIA_VIX"].ffill(limit=5)
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
        sig = (feat["ew_index"] > feat[col]).where(feat[col].notna(), True).astype(float)
        shift_pct = float(params.get("defensive_shift", 1.0))
        alloc = 1.0 - (1.0 - sig) * shift_pct
        return alloc.shift(1).fillna(1.0)

    if rule == "dual_ma":
        sig = (feat["ew_ma50"] > feat["ew_ma200"]).where(feat["ew_ma200"].notna(), True).astype(float)
        shift_pct = float(params.get("defensive_shift", 1.0))
        return (1.0 - (1.0 - sig) * shift_pct).shift(1).fillna(1.0)

    if rule == "index_dd":
        exit_dd = -abs(float(params.get("dd_thresh", 0.12)))
        reenter_ratio = float(params.get("reenter_dd_ratio", 0.6))
        reenter_above_ma = int(params.get("reenter_ma", 50))
        dd_col = params.get("dd_col", "ew_dd252")
        shift_pct = float(params.get("defensive_shift", 1.0))
        ma_col = f"ew_ma{reenter_above_ma}"
        state = np.ones(len(idx), dtype=float)
        def_active = False
        dds = feat[dd_col].values
        prices = feat["ew_index"].values
        mas = feat[ma_col].values
        for i in range(len(idx)):
            if not def_active and np.isfinite(dds[i]) and dds[i] < exit_dd:
                def_active = True
            elif def_active and np.isfinite(mas[i]) and prices[i] > mas[i] and dds[i] > exit_dd * reenter_ratio:
                def_active = False
            state[i] = 1.0 - shift_pct if def_active else 1.0
        return pd.Series(state, index=idx).shift(1).fillna(1.0)

    if rule == "breadth_hysteresis":
        col = params.get("col", "breadth_50_sma10")
        low = float(params.get("exit_below", 0.30))
        high = float(params.get("enter_above", 0.45))
        shift_pct = float(params.get("defensive_shift", 1.0))
        vals = feat[col].values
        state = np.ones(len(idx), dtype=float)
        def_active = False
        for i, v in enumerate(vals):
            if np.isfinite(v):
                if not def_active and v < low:
                    def_active = True
                elif def_active and v > high:
                    def_active = False
            state[i] = 1.0 - shift_pct if def_active else 1.0
        return pd.Series(state, index=idx).shift(1).fillna(1.0)

    if rule == "vix_filter":
        vix_exit = float(params.get("vix_exit", 24.0))
        vix_enter = float(params.get("vix_enter", 20.0))
        shift_pct = float(params.get("defensive_shift", 0.5))
        vals = feat["vix"].values if "vix" in feat else np.full(len(idx), 15.0)
        state = np.ones(len(idx), dtype=float)
        def_active = False
        for i, v in enumerate(vals):
            if np.isfinite(v):
                if not def_active and v > vix_exit:
                    def_active = True
                elif def_active and v < vix_enter:
                    def_active = False
            state[i] = 1.0 - shift_pct if def_active else 1.0
        return pd.Series(state, index=idx).shift(1).fillna(1.0)

    if rule == "index_ma_band":
        lb = int(params.get("ma_len", 200))
        exit_band = float(params.get("exit_band", 0.97))
        enter_band = float(params.get("enter_band", 1.01))
        shift_pct = float(params.get("defensive_shift", 1.0))
        prices = feat["ew_index"].values
        mas = feat[f"ew_ma{lb}"].values
        state = np.ones(len(idx), dtype=float)
        def_active = False
        for i, (px, ma) in enumerate(zip(prices, mas)):
            if np.isfinite(px) and np.isfinite(ma):
                if not def_active and px < ma * exit_band:
                    def_active = True
                elif def_active and px > ma * enter_band:
                    def_active = False
            state[i] = 1.0 - shift_pct if def_active else 1.0
        return pd.Series(state, index=idx).shift(1).fillna(1.0)

    if rule == "tiered_composite":
        # 4-signal posture score [0..4] -> smooth allocation mapping (exposure-coach style, safe warmup)
        c1 = (feat["ew_index"] > feat["ew_ma200"]).where(feat["ew_ma200"].notna(), True).astype(int)
        c2 = (feat["ew_index"] > feat["ew_ma50"]).where(feat["ew_ma50"].notna(), True).astype(int)
        bcol = feat["breadth_50_sma10"]
        c3 = (bcol >= float(params.get("b50_mid", 0.40))).where(bcol.notna(), True).astype(int)
        c4 = (feat["vix"] < float(params.get("vix_cap", 22.0))).where(feat["vix"].notna(), True).astype(int) if "vix" in feat else 1
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

    if rule == "multifactor_wf4":
        bench_df = params.get("bench_df")
        if bench_df is None:
            raise ValueError("multifactor_wf4 requires params['bench_df']")
        alloc_s, _ = compute_multifactor_state(feat, bench_df, weights=params.get("weights"))
        return alloc_s

    raise ValueError(f"Unknown regime rule: {rule}")


def _causal_mad_squash(series: pd.Series, win: int = 756, min_p: int = 126) -> np.ndarray:
    """Strictly causal rolling median/MAD normalization -> tanh(z/2) in [-1, +1]."""
    arr = pd.to_numeric(series, errors="coerce").values.astype(np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    for t in range(len(arr)):
        v = arr[t]
        if not np.isfinite(v):
            out[t] = out[t - 1] if t > 0 else 0.0
            continue
        sub = arr[max(0, t - win + 1) : t + 1]
        sub = sub[np.isfinite(sub)]
        if len(sub) < min_p:
            out[t] = 0.0
            continue
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) * 1.4826
        out[t] = float(np.tanh(0.5 * (v - med) / max(mad, 1e-6)))
    return out


def get_trend_gold_status(bench_df: pd.DataFrame, ema_len: int = 150, band: float = 0.02) -> dict[str, object]:
    """Return latest GOLDBEES vs EMA(150) +-2% hysteresis gate and active defensive ETF ('GOLDBEES' or 'LIQUIDCASE')."""
    if "GOLDBEES" not in bench_df.columns or bench_df["GOLDBEES"].dropna().empty:
        return {"gold_price": None, "gold_ema150": None, "gold_above_gate": False, "active_defensive_etf": "LIQUIDCASE"}
    g = pd.to_numeric(bench_df["GOLDBEES"], errors="coerce").ffill(limit=5)
    g_ema = g.ewm(span=ema_len, min_periods=30).mean()
    g_arr, em_arr = g.values, g_ema.values
    active = False
    for px, em in zip(g_arr, em_arr):
        if np.isfinite(px) and np.isfinite(em) and em > 0:
            if not active and px > (1.0 + band) * em:
                active = True
            elif active and px < (1.0 - band) * em:
                active = False
    px_now = float(g.dropna().iloc[-1])
    em_now = float(g_ema.dropna().iloc[-1])
    return {
        "gold_price": round(px_now, 2),
        "gold_ema150": round(em_now, 2),
        "gold_vs_ema150_pct": round((px_now / em_now - 1.0) * 100.0, 2) if em_now > 0 else 0.0,
        "gold_above_gate": bool(active),
        "active_defensive_etf": "GOLDBEES" if active else "LIQUIDCASE",
    }


def compute_multifactor_state(
    feat: pd.DataFrame,
    bench_df: pd.DataFrame,
    weights: tuple[float, float, float, float] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Zero-look-ahead 4-block causal orthogonal regime system (weights optimized out-of-sample on P1 2011-16).

    Blocks:
      B1 Trend & Appetite (0.44): EW/EMA200, EW/EMA126, Rel(EW/NIFTYBEES)/EMA126
      B2 Stress Vol & VIX (0.10): Negative 21d downside semivol + VIX/EMA63 spike
      B3 Market Breadth   (0.34): % > 50DMA + 21d Zweig breadth thrust
      B4 Crash & Gold     (0.12): Daniel-Moskowitz negative-trend crash risk + Gold/EMA126
    """
    w = weights or (0.44, 0.10, 0.34, 0.12)
    idx = feat.index
    ew = pd.to_numeric(feat["ew_index"], errors="coerce").ffill()
    ew_r = ew.pct_change(fill_method=None).fillna(0.0)
    ew_ema200 = ew.ewm(span=200, min_periods=60).mean()
    ew_ema126 = ew.ewm(span=126, min_periods=40).mean()
    ew_v63 = np.sqrt(252.0 * (ew_r ** 2).ewm(span=63, min_periods=20).mean()).clip(lower=0.08)
    t_ew200 = ((ew / ew_ema200) - 1.0) / ew_v63
    t_ew126 = ((ew / ew_ema126) - 1.0) / ew_v63

    nif_col = "NIFTYBEES" if "NIFTYBEES" in bench_df.columns else None
    if nif_col:
        nif = pd.to_numeric(bench_df[nif_col], errors="coerce").reindex(idx).ffill()
        rel = ew / nif.clip(lower=1e-4)
        rel_r = rel.pct_change(fill_method=None).fillna(0.0)
        rel_ema126 = rel.ewm(span=126, min_periods=40).mean()
        rel_v63 = np.sqrt(252.0 * (rel_r ** 2).ewm(span=63, min_periods=20).mean()).clip(lower=0.04)
        t_rel = ((rel / rel_ema126) - 1.0) / rel_v63
        b1 = (_causal_mad_squash(t_ew200) + _causal_mad_squash(t_ew126) + _causal_mad_squash(t_rel)) / 3.0
    else:
        b1 = 0.5 * (_causal_mad_squash(t_ew200) + _causal_mad_squash(t_ew126))

    neg2 = np.minimum(ew_r, 0.0) ** 2
    semivol21 = np.sqrt(252.0 * neg2.rolling(21, min_periods=10).mean()).fillna(0.125)
    vix = pd.to_numeric(feat["vix"], errors="coerce").ffill() if "vix" in feat else pd.Series(16.0, index=idx)
    vix_spike = np.log((vix / vix.ewm(span=63, min_periods=20).mean()).clip(lower=0.2))
    b2 = -0.5 * (_causal_mad_squash(np.log(semivol21.clip(lower=0.04))) + _causal_mad_squash(vix_spike))

    b50 = pd.to_numeric(feat["breadth_50"], errors="coerce").fillna(0.5)
    b50_e = b50.ewm(span=5, min_periods=2).mean()
    b50_chg21 = (b50_e - b50_e.shift(21)).fillna(0.0)
    b3 = 0.5 * (_causal_mad_squash(b50) + _causal_mad_squash(b50_chg21))

    dm_raw = np.where(t_ew200.values < 0.0, np.maximum(0.0, _causal_mad_squash(np.log(semivol21.clip(lower=0.04)))), 0.0)
    if "GOLDBEES" in bench_df.columns:
        gold = pd.to_numeric(bench_df["GOLDBEES"], errors="coerce").reindex(idx).ffill()
        g_r = gold.pct_change(fill_method=None).fillna(0.0)
        g_ema126 = gold.ewm(span=126, min_periods=40).mean()
        g_v63 = np.sqrt(252.0 * (g_r ** 2).ewm(span=63, min_periods=20).mean()).clip(lower=0.06)
        t_gold = ((gold / g_ema126) - 1.0) / g_v63
        b4 = -0.5 * (dm_raw + _causal_mad_squash(t_gold))
    else:
        b4 = -dm_raw

    p_comp = w[0] * b1 + w[1] * b2 + w[2] * b3 + w[3] * b4
    b50_chg_arr, b50_lvl_arr = b50_chg21.values, b50.values

    # Asymmetric EWMA (8d down / 22d up) + 3-state Schmitt hysteresis + 25d upward cooldown
    a_dn, a_up = 2.0 / (8.0 + 1.0), 2.0 / (22.0 + 1.0)
    p_sm = 0.0
    state = 2
    last_dn_t = -999
    cur_eq = 1.0
    T = len(idx)
    raw_eq = np.ones(T, dtype=np.float64)
    sm_arr = np.zeros(T, dtype=np.float64)
    st_arr = np.full(T, 2, dtype=np.int32)
    for t in range(T):
        pt = float(p_comp[t]) if np.isfinite(p_comp[t]) else p_sm
        thrust = bool(np.isfinite(b50_chg_arr[t]) and b50_chg_arr[t] >= 0.25 and b50_lvl_arr[t] >= 0.50)
        alpha = a_dn if (pt < p_sm or thrust) else a_up
        p_sm = alpha * pt + (1.0 - alpha) * p_sm
        can_up = (t - last_dn_t) >= 25 or thrust
        if state == 2:
            if p_sm < -0.25:
                state, last_dn_t = 0, t
            elif p_sm < -0.10:
                state, last_dn_t = 1, t
        elif state == 1:
            if p_sm < -0.25:
                state, last_dn_t = 0, t
            elif can_up and (p_sm > 0.18 or (thrust and p_sm > -0.10)):
                state = 2
        else:
            if can_up and (p_sm > -0.10 or (thrust and p_sm > -0.20)):
                state = 1
        w_raw = 1.0 if state == 2 else (0.60 if state == 1 else 0.25)
        if abs(w_raw - cur_eq) >= 0.08 or (state == 0 and cur_eq > 0.26):
            cur_eq = w_raw
        raw_eq[t] = cur_eq
        sm_arr[t] = p_sm
        st_arr[t] = state

    alloc_lagged = pd.Series(raw_eq, index=idx).shift(1).fillna(1.0)
    diag = pd.DataFrame(
        {
            "b1_trend": b1,
            "b2_vol": b2,
            "b3_breadth": b3,
            "b4_macro_gold": b4,
            "composite_raw": p_comp,
            "composite_smooth": sm_arr,
            "state_code": st_arr,
            "target_eq_eod": raw_eq,
        },
        index=idx,
    )
    return alloc_lagged, diag
