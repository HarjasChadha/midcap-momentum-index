"""Ingestion and point-in-time survivorship-free universe reconstruction from CMIE Prowessdx exports."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _detect_sep(sample: str) -> str:
    if "|" in sample:
        return "|"
    if "\t" in sample:
        return "\t"
    return ","


def load_prowess_file(path: str | Path) -> pd.DataFrame:
    """Read CMIE Prowessdx export (.zip containing pipe/tab/csv .txt, or raw .txt/.csv) into a tidy long DataFrame."""
    p = Path(path)
    frames: list[pd.DataFrame] = []

    def _read_text(raw_bytes: bytes) -> pd.DataFrame:
        text = raw_bytes.decode("utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return pd.DataFrame()
        sep = _detect_sep("\n".join(lines[:5]))
        return pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, low_memory=False)

    col_map: dict[str, str] = {}
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith((".txt", ".csv")) and "map" in name.lower():
                    mdf = _read_text(zf.read(name))
                    if len(mdf.columns) >= 2:
                        for _, row in mdf.iterrows():
                            k, v = str(row.iloc[0]).strip(), str(row.iloc[1]).strip()
                            if k and v and v.lower() != "nan":
                                col_map[k] = v
            for name in zf.namelist():
                if name.lower().endswith((".txt", ".csv")) and "readme" not in name.lower() and "map" not in name.lower():
                    df = _read_text(zf.read(name))
                    if not df.empty:
                        frames.append(df)
    else:
        frames.append(_read_text(p.read_bytes()))

    if not frames:
        raise ValueError(f"No readable tabular data found in {p}")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [col_map.get(str(c).strip(), str(c).strip()) for c in df.columns]
    return df


def _find_col(cols: list[str], *candidates: str) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        for lc, orig in lower_map.items():
            if cand in lc:
                return orig
    return None


def prowess_to_panels(
    raw_df: pd.DataFrame,
    exchange_prefer: str = "nse",
) -> dict[str, pd.DataFrame]:
    """Convert long CMIE Prowessdx table into aligned (Date x Symbol) panels: raw, adj, mcap, vol."""
    cols = list(raw_df.columns)
    sym_col = _find_col(cols, "nse symbol", "symbol", "co_code", "company name")
    dt_col = _find_col(cols, "date", "co_stkdate")
    adj_col = _find_col(cols, f"{exchange_prefer} adjusted closing", "adjusted closing price", "adj closing", "adj_close")
    close_col = _find_col(cols, f"{exchange_prefer} closing price", "closing price", "close")
    mcap_col = _find_col(cols, f"{exchange_prefer} market cap", "market capitalisation", "market cap", "mcap")
    vol_col = _find_col(cols, f"{exchange_prefer} traded", "no. of shares traded", "volume", "shares traded")
    ret_col = _find_col(cols, f"{exchange_prefer} returns", "total return", "daily returns")
    af_col = _find_col(cols, "adjustment factor", "adj factor", "nse adjustment factor")

    if not sym_col or not dt_col or (not close_col and not adj_col):
        raise ValueError(f"Could not detect required Symbol/Date/Price columns in headers: {cols}")

    work = raw_df.copy()
    work["_dt"] = pd.to_datetime(work[dt_col], errors="coerce").dt.strftime("%Y-%m-%d")
    work["_sym"] = work[sym_col].astype(str).str.strip().str.upper()
    work = work.dropna(subset=["_dt", "_sym"])
    work = work[work["_sym"].ne("") & work["_sym"].ne("NAN")]

    def _pivot(col: str | None) -> pd.DataFrame | None:
        if not col:
            return None
        vals = pd.to_numeric(work[col], errors="coerce")
        sub = work.assign(_val=vals).dropna(subset=["_val"])
        if sub.empty:
            return None
        return sub.pivot_table(index="_dt", columns="_sym", values="_val", aggfunc="last").sort_index()

    raw_px = _pivot(close_col)
    adj_px = _pivot(adj_col)
    mcap = _pivot(mcap_col)
    vol = _pivot(vol_col)

    # Reconstruct adjusted price from adjustment factor or daily total returns if explicit adj column absent
    if adj_px is None and raw_px is not None:
        if af_col:
            af = _pivot(af_col)
            if af is not None:
                # Backward adjustment factor per split/bonus date when factor provided
                adj_px = (raw_px * af.reindex_like(raw_px).ffill().fillna(1.0))
        elif ret_col:
            rets = _pivot(ret_col)
            if rets is not None:
                # Convert % returns if median > 0.5 vs decimal returns
                med_abs = float(rets.abs().stack().median() or 0.0)
                r_dec = rets / 100.0 if med_abs > 0.35 else rets
                adj_px = 100.0 * (1.0 + r_dec.fillna(0.0)).cumprod()
        if adj_px is None:
            adj_px = raw_px.copy()
    if raw_px is None and adj_px is not None:
        raw_px = adj_px.copy()

    return {"adj": adj_px, "raw": raw_px, "mcap": mcap, "vol": vol}


def build_point_in_time_midcap150_mask(
    mcap_df: pd.DataFrame,
    min_stocks_for_pit: int = 300,
    largecap_cutoff: int = 100,
    midcap_end: int = 250,
) -> pd.DataFrame:
    """Reconstruct SEBI/NSE point-in-time Nifty Midcap 150 eligibility mask (Ranks 101..250 by trailing 126d avg MCap)."""
    smooth_mcap = mcap_df.rolling(126, min_periods=20).mean()
    ranks = smooth_mcap.rank(axis=1, ascending=False, method="first")
    valid_counts = smooth_mcap.notna().sum(axis=1)
    mask = (ranks > largecap_cutoff) & (ranks <= midcap_end)
    # Require broad listed universe on that day before applying rank slicing
    return mask.where(valid_counts >= min_stocks_for_pit, True)


def discover_cmie_prowess_export(repo_root: str | Path) -> Path | None:
    """Return path to a dropped CMIE Prowessdx OHLC/MCap export (.zip/.txt/.csv) inside data/cmie/, if present."""
    cmie_dir = Path(repo_root) / "data" / "cmie"
    if not cmie_dir.exists():
        return None
    for ext in ("*.zip", "*.txt", "*.csv"):
        for p in sorted(cmie_dir.glob(ext)):
            if p.name in {"nse_midcap150_symbols.txt", "nse_midcap150_plus_etfs.txt"}:
                continue
            return p
    return None


def load_cmie_prowess_pit_dataset(
    repo_root: str | Path,
    prowess_path: str | Path | None = None,
    reconstitute_semiannual: bool = True,
) -> dict[str, pd.DataFrame] | None:
    """Drop-in CMIE Prowessdx reconstitution returning {adj, raw, mcap, vol, pit_mask} for Ranks 101..250 by MCap."""
    src_path = Path(prowess_path) if prowess_path else discover_cmie_prowess_export(repo_root)
    if src_path is None or not src_path.exists():
        return None
    raw_tbl = load_prowess_file(src_path)
    panels = prowess_to_panels(raw_tbl)
    adj, raw, mcap = panels["adj"], panels["raw"], panels["mcap"]
    if mcap is None or adj is None:
        raise ValueError(f"CMIE Prowess export {src_path} is missing market-capitalisation or price columns")
    pit_daily = build_point_in_time_midcap150_mask(mcap.reindex_like(adj))
    if not reconstitute_semiannual:
        return {**panels, "pit_mask": pit_daily}

    pit_semi = pd.DataFrame(False, index=adj.index, columns=adj.columns)
    cur_sel = pd.Series(False, index=adj.columns)
    prev_ym = ""
    for dt in adj.index:
        m = int(str(dt)[5:7])
        ym = str(dt)[:7]
        if m in (3, 9) and ym != prev_ym and pit_daily.loc[dt].any():
            prev_ym = ym
            cur_sel = pit_daily.loc[dt].copy()
        elif cur_sel.any():
            cur_sel = cur_sel & adj.loc[dt].ffill(limit=5).notna()
        pit_semi.loc[dt] = cur_sel
    return {**panels, "pit_daily": pit_daily, "pit_mask": pit_semi}
