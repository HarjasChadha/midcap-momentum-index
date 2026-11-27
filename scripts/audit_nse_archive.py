"""Audit 2010-2026 NSE PR archive data quality: verify split/bonus continuity and flag any unexplained extreme jumps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nse_archive_loader import build_2010_2026_nse_dataset  # noqa: E402


def main() -> None:
    d = build_2010_2026_nse_dataset(REPO_ROOT, 2010, 2026, refresh=False)
    adj = d["adj"]
    raw = d["raw"]
    bench = d["bench"]
    bc_log = d["bc_log"]

    print(f"Dataset shape: {adj.shape} ({adj.index[0]} -> {adj.index[-1]})")
    print(f"Corporate actions applied: {len(bc_log)}")
    print("Benchmark coverage:")
    for col in bench.columns:
        s = bench[col].dropna()
        print(f"  {col:18s}: {len(s):4d} days | {s.index[0]} -> {s.index[-1]} | first={s.iloc[0]:.2f} last={s.iloc[-1]:.2f}")

    # Check known split/bonus test stocks
    for sym, dt_around in [("BANKBARODA", "2015-01"), ("MOTILALOFS", "2024-06"), ("RELAXO", "2015-06"), ("GOLDBEES", "2019-12")]:
        src = bench[sym] if sym in bench.columns else adj[sym]
        sub = src.loc[src.index.str.startswith(dt_around)].dropna()
        r = sub.pct_change().dropna()
        print(f"  {sym} ({dt_around}) max abs daily ret = {float(r.abs().max() * 100.0):.2f}%")

    # Flag any single-day adj returns > +120% or < -55% among liquid (>= Rs 1 Cr/day) stocks
    tv = d["trdval"]
    liq_mask = tv.rolling(20, min_periods=5).median() >= 1e7  # Rs 1 Cr/day
    rets = adj.pct_change(fill_method=None)
    big = ((rets > 1.20) | (rets < -0.55)) & liq_mask
    hits = []
    for dt in big.index[big.any(axis=1)]:
        for sym in big.columns[big.loc[dt]]:
            hits.append((dt, sym, round(float(rets.loc[dt, sym]) * 100.0, 2)))
    print(f"Liquid extreme single-day moves (>120% or <-55%): {len(hits)}")
    for item in hits[:30]:
        print(" ", item)


if __name__ == "__main__":
    main()
