"""Momentum scoring, ranking, and portfolio weight construction for Nifty Midcap 150."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def normalize_z_pdf(z_series: pd.Series) -> pd.Series:
    """Exact PDF normalization: (1 + z) if z >= 0 else 1 / (1 - z)."""
    s = z_series.dropna()
    return s.where(s >= 0, 1.0 / (1.0 - s)).where(s < 0, 1.0 + s)


def compute_score_matrix(
    adj_df: pd.DataFrame,
    lookback_days: int = 189,
    skip_days: int = 21,
    mode: str = "raw",
    vol_df: pd.DataFrame | None = None,
    raw_df: pd.DataFrame | None = None,
    min_turnover_inr: float = 0.0,
) -> pd.DataFrame:
    """Compute cross-sectional momentum scores aligned to each trading date t (no look-ahead)."""
    p_end = adj_df.shift(skip_days) if skip_days > 0 else adj_df
    daily_rets = adj_df.pct_change(fill_method=None)

    def _raw_ret(lb: int) -> pd.DataFrame:
        p_start = adj_df.shift(skip_days + lb)
        ret = (p_end / p_start) - 1.0
        # Require active stock on day t and sufficient lookback coverage
        min_obs = max(20, int(0.85 * lb))
        obs = adj_df.rolling(lb + skip_days, min_periods=min_obs).count()
        valid = adj_df.notna() & p_end.notna() & p_start.notna() & (obs >= min_obs)
        return ret.where(valid)

    def _risk_adj(lb: int) -> pd.DataFrame:
        ret = _raw_ret(lb)
        vol = daily_rets.rolling(lb, min_periods=max(20, int(0.8 * lb))).std()
        if skip_days > 0:
            vol = vol.shift(skip_days)
        ann_vol = (vol * math.sqrt(252.0)).replace(0.0, np.nan)
        return ret / ann_vol

    if mode == "raw":
        scores = _raw_ret(lookback_days)
    elif mode == "risk_adj":
        scores = _risk_adj(lookback_days)
    elif mode == "blend_6_12":
        scores = 0.5 * _zscore_rows(_raw_ret(126)) + 0.5 * _zscore_rows(_raw_ret(252))
    elif mode == "blend_3_6_12":
        scores = (
            _zscore_rows(_raw_ret(63)) + _zscore_rows(_raw_ret(126)) + _zscore_rows(_raw_ret(252))
        ) / 3.0
    elif mode == "blend_risk_6_12":
        scores = 0.5 * _zscore_rows(_risk_adj(126)) + 0.5 * _zscore_rows(_risk_adj(252))
    else:
        raise ValueError(f"Unknown momentum mode: {mode}")

    if min_turnover_inr > 0 and vol_df is not None and raw_df is not None:
        turnover = (raw_df * vol_df).rolling(20, min_periods=10).median()
        scores = scores.where(turnover >= min_turnover_inr)

    return scores


def select_sleeve_weights(
    score_row: pd.Series,
    top_n: int,
    weighting: str = "equal",
    vol_row: pd.Series | None = None,
    prev_holdings: set[str] | None = None,
    buffer_n: int | None = None,
    excluded: set[str] | None = None,
) -> dict[str, float]:
    """Select Top-N stocks (with optional rank buffer/exclusions) and assign sleeve weights summing to 1.0."""
    valid = score_row.dropna()
    if excluded:
        valid = valid.drop(index=[s for s in excluded if s in valid.index], errors="ignore")
    valid = valid.sort_values(ascending=False)
    if valid.empty:
        return {}

    if buffer_n and buffer_n > top_n and prev_holdings:
        # Keep prior holdings still ranked inside Top buffer_n, then fill remaining slots by rank
        ranked_syms = list(valid.index)
        eligible_set = set(ranked_syms[:buffer_n])
        kept = [s for s in ranked_syms[:buffer_n] if s in prev_holdings][:top_n]
        kept_set = set(kept)
        needed = top_n - len(kept)
        if needed > 0:
            additions = [s for s in ranked_syms if s not in kept_set and s in eligible_set][:needed]
            chosen = kept + additions
        else:
            chosen = kept
    else:
        chosen = list(valid.index[:top_n])

    if not chosen:
        return {}

    if weighting == "equal":
        w = 1.0 / len(chosen)
        return {s: w for s in chosen}

    if weighting == "inv_vol" and vol_row is not None:
        vols = vol_row.reindex(chosen).replace(0.0, np.nan)
        med = vols.median()
        if not np.isfinite(med) or med <= 0:
            w = 1.0 / len(chosen)
            return {s: w for s in chosen}
        vols = vols.fillna(med).clip(lower=1e-4)
        inv = 1.0 / vols
        inv = inv / inv.sum()
        return {s: float(inv[s]) for s in chosen}

    if weighting == "norm_z":
        # Compute z-score across whole eligible universe, then normalize per PDF Appendix
        mu = float(valid.mean())
        sd = float(valid.std())
        if sd <= 1e-12 or not math.isfinite(sd):
            w = 1.0 / len(chosen)
            return {s: w for s in chosen}
        z = (valid.loc[chosen] - mu) / sd
        nz = normalize_z_pdf(z)
        nz = nz / nz.sum()
        return {s: float(nz[s]) for s in chosen}

    raise ValueError(f"Unknown weighting mode: {weighting}")


def combine_sleeves(sleeves: list[tuple[float, Mapping[str, float]]]) -> dict[str, float]:
    """Blend multiple weighted stock sleeves (e.g. 50% Raw20 + 50% Raw30) into unified weights."""
    out: dict[str, float] = {}
    total_alloc = 0.0
    for alloc, wmap in sleeves:
        if not wmap or alloc <= 0:
            continue
        total_alloc += alloc
        for sym, w in wmap.items():
            out[sym] = out.get(sym, 0.0) + alloc * w
    if total_alloc <= 0:
        return {}
    if abs(total_alloc - 1.0) > 1e-9:
        out = {k: v / total_alloc for k, v in out.items()}
    return out


def select_affordable_blend(
    score_rows: Mapping[str, pd.Series],
    sleeves_cfg: list[dict],
    raw_price_row: pd.Series | None,
    capital_inr: float,
    vol_row: pd.Series | None = None,
    prev_holdings: set[str] | None = None,
) -> dict[str, float]:
    """Sequential slot-cap filter: iteratively drop highest-weight unaffordable tickers so remaining ranks slide into freed slots."""
    banned: set[str] = set()
    for _ in range(60):
        sub_pairs = []
        for sc in sleeves_cfg:
            srow = score_rows.get(sc["score_key"])
            if srow is None:
                continue
            w = select_sleeve_weights(
                srow,
                top_n=int(sc["top_n"]),
                weighting=sc.get("weighting", "equal"),
                vol_row=vol_row,
                prev_holdings=prev_holdings,
                buffer_n=sc.get("buffer_n"),
                excluded=banned,
            )
            sub_pairs.append((float(sc.get("alloc", 1.0)), w))
        blended = combine_sleeves(sub_pairs)
        if not blended or raw_price_row is None or capital_inr <= 0:
            return blended
        drop_sym = None
        for sym, w in blended.items():
            px = float(raw_price_row.get(sym, float("nan")))
            if not math.isfinite(px) or px <= 0 or px > capital_inr * w:
                drop_sym = sym
                break
        if drop_sym is None:
            return blended
        banned.add(drop_sym)
    return blended
