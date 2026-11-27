"""Data loader and quality validator for Nifty Midcap 150 constituents and defensive ETFs."""

from __future__ import annotations

import csv
import datetime
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytz
import requests

IST = pytz.timezone("Asia/Kolkata")
USER_AGENT = "Mozilla/5.0"

MIDCAP150_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv"

BENCHMARK_TICKERS = {
    "NIFTY_MIDCAP_150": "NIFTYMIDCAP150.NS",
    "MIDCAPIETF": "MIDCAPIETF.NS",
    "GOLDBEES": "GOLDBEES.NS",
    "SILVERBEES": "SILVERBEES.NS",
    "LIQUIDBEES": "LIQUIDBEES.NS",
    "EBBETF0431": "EBBETF0431.NS",
    "LTGILTBEES": "LTGILTBEES.NS",
    "LIQUIDCASE": "LIQUIDCASE.NS",
    "NIFTYBEES": "NIFTYBEES.NS",
    "MON100": "MON100.NS",
    "INDIA_VIX": "^INDIAVIX",
}


def _repair_transient_glitches(df: pd.DataFrame) -> pd.DataFrame:
    """Repair 1-3 day Yahoo ex-split/data glitches (>60% drop rebounding within 5 days)."""
    out = df.copy()
    for col in out.columns:
        s = out[col].dropna()
        if len(s) < 4:
            continue
        vals = s.values.copy()
        idx = s.index
        i = 1
        while i < len(vals) - 1:
            day_ret = vals[i] / vals[i - 1] - 1.0
            if day_ret < -0.60:
                # Look ahead up to 5 bars to see if price snaps back to prior level
                for j in range(i + 1, min(len(vals), i + 6)):
                    if abs(vals[j] / vals[i - 1] - 1.0) < 0.20:
                        for k in range(i, j):
                            out.loc[idx[k], col] = vals[i - 1]
                            vals[k] = vals[i - 1]
                        i = j
                        break
            i += 1
    return out


def load_universe(repo_root: Path, refresh: bool = False) -> pd.DataFrame:
    universe_path = repo_root / "data" / "universe" / "midcap150.csv"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not universe_path.exists():
        r = requests.get(MIDCAP150_CSV_URL, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(r.text)))
        if len(rows) < 100:
            raise ValueError(f"Unexpected Midcap 150 CSV row count: {len(rows)}")
        universe_path.write_text(r.text)
    return pd.read_csv(universe_path)


def fetch_yahoo_series(yahoo_symbol: str, range_str: str = "10y") -> dict[str, pd.Series]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(yahoo_symbol, safe='')}?range={range_str}&interval=1d"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    payload = r.json()
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        raise ValueError(f"No Yahoo chart data for {yahoo_symbol}: {(payload.get('chart') or {}).get('error')}")

    res = results[0]
    meta = res.get("meta") or {}
    timestamps = res.get("timestamp") or []
    quote_obj = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote_obj.get("close") or []
    volumes = quote_obj.get("volume") or [None] * len(timestamps)
    adj_list = ((res.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or closes

    adj_map: dict[str, float] = {}
    raw_map: dict[str, float] = {}
    vol_map: dict[str, float] = {}
    for ts, raw_c, adj_c, vol in zip(timestamps, closes, adj_list, volumes):
        dt = datetime.datetime.fromtimestamp(ts, IST).strftime("%Y-%m-%d")
        if raw_c is not None and math.isfinite(raw_c) and raw_c > 0:
            raw_map[dt] = float(raw_c)
        val_adj = adj_c if (adj_c is not None and math.isfinite(adj_c) and adj_c > 0) else raw_c
        if val_adj is not None and math.isfinite(val_adj) and val_adj > 0:
            adj_map[dt] = float(val_adj)
        if vol is not None and math.isfinite(vol) and vol >= 0:
            vol_map[dt] = float(vol)

    # Catch today's unfinalized daily bar via regularMarketPrice
    rmp = meta.get("regularMarketPrice")
    rmt = meta.get("regularMarketTime")
    if rmp is not None and math.isfinite(rmp) and rmp > 0 and rmt is not None:
        mdt = datetime.datetime.fromtimestamp(rmt, IST).strftime("%Y-%m-%d")
        raw_map.setdefault(mdt, float(rmp))
        adj_map.setdefault(mdt, float(rmp))

    return {
        "adj": pd.Series(adj_map, dtype=np.float64),
        "raw": pd.Series(raw_map, dtype=np.float64),
        "vol": pd.Series(vol_map, dtype=np.float64),
    }


def fetch_and_cache_all(repo_root: str | Path, range_str: str = "10y", refresh: bool = False) -> dict[str, pd.DataFrame]:
    repo_root = Path(repo_root)
    cache_dir = repo_root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "adj": cache_dir / "adj_close.csv.gz",
        "raw": cache_dir / "raw_close.csv.gz",
        "vol": cache_dir / "volume.csv.gz",
        "bench": cache_dir / "benchmarks.csv.gz",
    }
    audit_path = cache_dir / "data_audit.json"

    if not refresh and all(p.exists() for p in files.values()) and audit_path.exists():
        return {
            "adj": pd.read_csv(files["adj"], index_col=0),
            "raw": pd.read_csv(files["raw"], index_col=0),
            "vol": pd.read_csv(files["vol"], index_col=0),
            "bench": pd.read_csv(files["bench"], index_col=0),
            "audit": json.loads(audit_path.read_text()),
        }

    univ = load_universe(repo_root, refresh=refresh)
    symbols = univ["Symbol"].astype(str).tolist()

    adj_cols, raw_cols, vol_cols = {}, {}, {}
    failures: dict[str, str] = {}

    def _job(sym: str) -> tuple[str, dict[str, pd.Series]]:
        return sym, fetch_yahoo_series(f"{sym}.NS", range_str=range_str)

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(_job, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                _, data = fut.result()
                adj_cols[sym] = data["adj"]
                raw_cols[sym] = data["raw"]
                vol_cols[sym] = data["vol"]
            except Exception as e:
                failures[sym] = str(e)

    bench_cols = {}
    for label, ysym in BENCHMARK_TICKERS.items():
        try:
            bench_cols[label] = fetch_yahoo_series(ysym, range_str=range_str)["adj"]
        except Exception as e:
            failures[f"BENCH:{label}"] = str(e)

    adj_df = pd.DataFrame(adj_cols).sort_index()
    raw_df = pd.DataFrame(raw_cols).sort_index()
    vol_df = pd.DataFrame(vol_cols).sort_index()
    bench_df = pd.DataFrame(bench_cols).sort_index()

    # Align stocks & benchmarks to NSE trading days where index or majority of stocks traded
    valid_days = adj_df.index[adj_df.notna().sum(axis=1) >= max(20, int(0.3 * len(adj_df.columns)))]
    adj_df = adj_df.loc[valid_days]
    raw_df = raw_df.reindex(valid_days)
    vol_df = vol_df.reindex(valid_days)
    bench_df = bench_df.reindex(valid_days)

    # Verified via NSE MCP Bhavcopy + Corporate Actions:
    # 1) Yahoo Finance applied MOTILALOFS 3:1 bonus (0.25 factor) starting 2024-01-01 instead of pre-2024-01-01.
    if "MOTILALOFS" in adj_df.columns:
        pre_2024 = adj_df.index < "2024-01-01"
        adj_df.loc[pre_2024, "MOTILALOFS"] = adj_df.loc[pre_2024, "MOTILALOFS"] * 0.25
        raw_df.loc[pre_2024, "MOTILALOFS"] = raw_df.loc[pre_2024, "MOTILALOFS"] * 0.25
    # 2) PATANJALI pre-2020-01-27 was pre-NCLT Ruchi Soya capital reduction (-95% relist jump).
    if "PATANJALI" in adj_df.columns:
        pre_relist = adj_df.index < "2020-01-27"
        adj_df.loc[pre_relist, "PATANJALI"] = np.nan
        raw_df.loc[pre_relist, "PATANJALI"] = np.nan

    bench_df = _repair_transient_glitches(bench_df)
    adj_df = _repair_transient_glitches(adj_df)

    # Backfill any trailing single-day Yahoo nulls (e.g., EMBASSY REIT series) via official NSE MCP Bhavcopy
    tail_nulls = list(adj_df.columns[adj_df.iloc[-1].isna()])
    if tail_nulls:
        try:
            from src.nse_mcp import NseMcpClient

            last_dt = str(adj_df.index[-1])
            quotes = NseMcpClient().get_bulk_quote(tail_nulls)
            for sym in tail_nulls:
                q = quotes.get(sym)
                if isinstance(q, dict) and str(q.get("date", "")).startswith(last_dt):
                    cl = float(q.get("close", 0.0))
                    if cl > 0:
                        adj_df.loc[last_dt, sym] = cl
                        raw_df.loc[last_dt, sym] = cl
                        if q.get("volume"):
                            vol_df.loc[last_dt, sym] = float(q["volume"])
        except Exception:
            pass

    # Check extreme single-day moves (>60% drop/jump in adj close) for data quality log
    rets = adj_df.pct_change(fill_method=None)
    outliers = []
    bad_mask = (rets > 0.65) | (rets < -0.50)
    for dt in bad_mask.index[bad_mask.any(axis=1)]:
        for col in bad_mask.columns[bad_mask.loc[dt]]:
            outliers.append({"date": dt, "symbol": col, "ret": round(float(rets.loc[dt, col]), 4)})

    audit = {
        "universe_count": len(symbols),
        "fetched_stocks": len(adj_df.columns),
        "failures": failures,
        "trading_days": len(adj_df),
        "start_date": str(adj_df.index[0]),
        "end_date": str(adj_df.index[-1]),
        "coverage_by_year": {
            str(yr): int(adj_df.loc[adj_df.index.str.startswith(str(yr))].notna().any(axis=0).sum())
            for yr in sorted({d[:4] for d in adj_df.index})
        },
        "extreme_daily_moves_count": len(outliers),
        "extreme_daily_moves_sample": outliers[:20],
    }

    adj_df.to_csv(files["adj"], compression="gzip")
    raw_df.to_csv(files["raw"], compression="gzip")
    vol_df.to_csv(files["vol"], compression="gzip")
    bench_df.to_csv(files["bench"], compression="gzip")
    audit_path.write_text(json.dumps(audit, indent=2))

    return {"adj": adj_df, "raw": raw_df, "vol": vol_df, "bench": bench_df, "audit": audit}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = fetch_and_cache_all(root, refresh=True)
    print(json.dumps(out["audit"], indent=2))
