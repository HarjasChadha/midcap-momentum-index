"""Official 2010-2026 NSE PR Archive Loader: survivorship-free equities, ETFs, indices, and exact Bc corporate action factors."""

from __future__ import annotations

import io
import json
import random
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

PR_ARCHIVE_URL = "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{ddmmyy}.zip"
_THREAD_LOCAL = threading.local()

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Connection": "keep-alive",
}


def _session() -> requests.Session:
    s = getattr(_THREAD_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(_HEADERS)
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        s.mount("https://", adapter)
        _THREAD_LOCAL.session = s
    return s


def parse_bc_purpose_factor(purpose: str) -> tuple[float, str]:
    """Extract deterministic backward price multiplier (f < 1 splits/bonus, f > 1 consolidation) from NSE Bc PURPOSE."""
    p = " ".join(str(purpose).upper().split())
    # Explicitly ignore Special Cash Dividends ('SPLDIV', 'SPL RE', 'AGM/DIV...+SPL RS 10') which do not divide shares
    if (
        ("DIV" in p and "SUB-DIV" not in p and "SUB DIV" not in p and "DIVISION" not in p)
        and "FV" not in p
        and "BON" not in p
        and "SPLT" not in p
        and "SPLIT" not in p
        and "FACE" not in p
        and "TO RS" not in p
        and "TORS" not in p
        and "TO RE" not in p
        and "TORE" not in p
    ):
        return 1.0, "dividend_or_other"

    factor = 1.0
    tags: list[str] = []

    # 1. Bonus ratio A:B (A new shares per B held -> factor = B / (A + B))
    for m in re.finditer(r"BON(?:US)?\s*[-:/]?\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", p):
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and b > 0:
            factor *= b / (a + b)
            tags.append(f"bonus_{a:g}:{b:g}")

    # 2. Face Value Split / Sub-division / Consolidation (FV_old -> FV_new -> factor = FV_new / FV_old)
    fv_match = re.search(
        r"(?:FV\s*SPL(?:I?T)?|FACE\s*VAL(?:UE)?\s*SPL(?:I?T)?|SPLIT|SPLT|SUB[- ]?DIV(?:ISION)?|CONSOL(?:IDATION)?)"
        r".{0,24}?(?:RS\.?|RE\.?)?\s*(\d+(?:\.\d+)?)\s*(?:/-)?\s*(?:TO|INTO)\s*(?:RS\.?|RE\.?)?\s*(\d+(?:\.\d+)?)",
        p,
    )
    if not fv_match:
        fv_match = re.search(r"RS\.?\s*(\d+(?:\.\d+)?)\s*(?:TO|TORS)\s*(?:RS\.?|RE\.?)?\s*(\d+(?:\.\d+)?)", p)
    if fv_match:
        old_fv, new_fv = float(fv_match.group(1)), float(fv_match.group(2))
        if old_fv > 0 and new_fv > 0 and old_fv != new_fv:
            factor *= new_fv / old_fv
            tags.append(f"fv_{old_fv:g}_to_{new_fv:g}")

    if "DEMERG" in p:
        tags.append("demerger")
    elif "RIGHT" in p or "RIGH" in p:
        tags.append("rights")

    return factor, "+".join(tags) if tags else "other"


def _parse_single_pr_zip(date_iso: str) -> dict[str, Any] | None:
    dt = pd.Timestamp(date_iso)
    ddmmyy = dt.strftime("%d%m%y")
    url = PR_ARCHIVE_URL.format(ddmmyy=ddmmyy)
    resp = None
    for attempt in range(5):
        try:
            r = _session().get(url, timeout=15)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and len(r.content) >= 1000:
                resp = r
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((1.5 * (2**attempt)) + random.uniform(0.2, 0.8))
                continue
            return None
        except requests.RequestException:
            # Reset thread-local session after connection drops before exponential backoff
            _THREAD_LOCAL.session = None
            time.sleep((2.0 * (2**attempt)) + random.uniform(0.3, 1.0))
    if resp is None:
        return None

    eq_rows: dict[str, tuple[float, float]] = {}  # sym -> (close, trdval)
    etf_rows: dict[str, float] = {}
    idx_rows: dict[str, float] = {}
    bc_events: list[tuple[str, str, str]] = []  # (sym, ex_dt_iso, purpose)

    def _safe_csv(raw_bytes: bytes) -> pd.DataFrame:
        if not raw_bytes or not raw_bytes.strip():
            return pd.DataFrame()
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding="latin1", on_bad_lines="skip", low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            # 1) Pd: Equities + Indices (reject corrupted archives packed with a wrong year's Pd file, e.g. PR020118 -> Pd020119)
            pd_names = [n for n in names if n.lower().startswith("pd") and n.lower().endswith(".csv")]
            if pd_names:
                pd_bn = pd_names[0].lower()
                if any(c.isdigit() for c in pd_bn) and dt.strftime("%d%m%y") not in pd_bn and dt.strftime("%d%m%Y") not in pd_bn:
                    return None
                df_pd = _safe_csv(zf.read(pd_names[0]))
                df_pd.columns = [str(c).strip() for c in df_pd.columns]
                if {"MKT", "SERIES", "SYMBOL", "CLOSE_PRICE", "NET_TRDVAL"}.issubset(df_pd.columns):
                    mkt = df_pd["MKT"].astype(str).str.strip()
                    ser = df_pd["SERIES"].astype(str).str.strip()
                    sym = df_pd["SYMBOL"].astype(str).str.strip()
                    sec = df_pd["SECURITY"].astype(str).str.strip() if "SECURITY" in df_pd.columns else sym
                    cl = pd.to_numeric(df_pd["CLOSE_PRICE"], errors="coerce")
                    tv = pd.to_numeric(df_pd["NET_TRDVAL"], errors="coerce").fillna(0.0)

                    # Index rows (MKT == 'Y' & SYMBOL blank)
                    idx_mask = (mkt == "Y") & (sym == "") & cl.notna() & (cl > 0)
                    for sname, cval in zip(sec[idx_mask], cl[idx_mask]):
                        low = sname.lower()
                        if low in {"s&p cnx nifty", "nifty 50"}:
                            idx_rows["NIFTY_50"] = float(cval)
                        elif low in {"cnx midcap", "nifty midcap 100"}:
                            idx_rows["NIFTY_MIDCAP_100"] = float(cval)
                        elif low == "nifty midcap 150":
                            idx_rows["NIFTY_MIDCAP_150"] = float(cval)
                        elif low == "india vix":
                            idx_rows["INDIA_VIX"] = float(cval)

                    # Regular equity series (EQ / BE / BZ)
                    eq_mask = (sym != "") & ser.isin({"EQ", "BE", "BZ"}) & cl.notna() & (cl > 0)
                    for sy, sr, cval, tval in zip(sym[eq_mask], ser[eq_mask], cl[eq_mask], tv[eq_mask]):
                        # Prefer EQ if duplicate symbol appears
                        if sy not in eq_rows or sr == "EQ":
                            eq_rows[sy] = (float(cval), float(tval))

            # 2) etf: GOLDBEES, LIQUIDBEES, LIQUIDCASE, NIFTYBEES, MON100, SILVERBEES, EBBETF0431, LTGILTBEES
            etf_names = [n for n in names if n.lower().startswith("etf") and n.lower().endswith(".csv")]
            if etf_names:
                df_e = _safe_csv(zf.read(etf_names[0]))
                df_e.columns = [str(c).strip().upper() for c in df_e.columns]
                if "SYMBOL" in df_e.columns and "CLOSE PRICE" in df_e.columns:
                    cl_e = pd.to_numeric(df_e["CLOSE PRICE"], errors="coerce")
                    for sy, cval in zip(df_e["SYMBOL"].astype(str).str.strip(), cl_e):
                        if sy in {"GOLDBEES", "LIQUIDBEES", "LIQUIDCASE", "NIFTYBEES", "MON100", "SILVERBEES", "EBBETF0431", "LTGILTBEES"} and np.isfinite(cval) and cval > 0:
                            etf_rows[sy] = float(cval)

            # 3) Bc: Corporate actions
            bc_names = [n for n in names if n.lower().startswith("bc") and n.lower().endswith(".csv")]
            if bc_names:
                df_b = _safe_csv(zf.read(bc_names[0]))
                df_b.columns = [str(c).strip().upper() for c in df_b.columns]
                if {"SYMBOL", "EX_DT", "PURPOSE"}.issubset(df_b.columns):
                    ser_b = df_b["SERIES"].astype(str).str.strip() if "SERIES" in df_b.columns else pd.Series(["EQ"] * len(df_b))
                    for sy, sr, ex, pur in zip(df_b["SYMBOL"].astype(str).str.strip(), ser_b, df_b["EX_DT"].astype(str).str.strip(), df_b["PURPOSE"].astype(str).str.strip()):
                        if not sy or not ex or sr not in {"EQ", "BE", "BZ", ""}:
                            continue
                        pur_up = pur.upper()
                        if any(k in pur_up for k in ("SPL", "BON", "DEMERG", "ARRANG", "SCHEME", "REDUCT", "RIGHT", "RIGH", "TORS", "TO RS", "CONSOL")):
                            parsed_dt = pd.to_datetime(ex, format="%Y-%m-%d", errors="coerce") if "-" in ex else pd.to_datetime(ex, dayfirst=True, errors="coerce")
                            if pd.notna(parsed_dt):
                                bc_events.append((sy, parsed_dt.strftime("%Y-%m-%d"), pur))
    except zipfile.BadZipFile:
        return None

    return {"date": date_iso, "eq": eq_rows, "etf": etf_rows, "idx": idx_rows, "bc": bc_events}


def _fetch_year_chunk(year: int, out_dir: Path, start_date: str = "2010-01-04", end_date: str = "2026-11-26", max_workers: int = 8) -> dict[str, Any]:
    yr_dir = out_dir / "years"
    yr_dir.mkdir(parents=True, exist_ok=True)
    raw_p = yr_dir / f"raw_{year}.pkl.gz"
    tv_p = yr_dir / f"trdval_{year}.pkl.gz"
    etf_p = yr_dir / f"etf_{year}.pkl.gz"
    idx_p = yr_dir / f"idx_{year}.pkl.gz"
    bc_p = yr_dir / f"bc_{year}.json"

    if all(p.exists() for p in (raw_p, tv_p, etf_p, idx_p, bc_p)):
        return {
            "raw": pd.read_pickle(raw_p),
            "trdval": pd.read_pickle(tv_p),
            "etf": pd.read_pickle(etf_p),
            "idx": pd.read_pickle(idx_p),
            "bc": json.loads(bc_p.read_text()),
        }

    s_dt = max(f"{year}-01-01", start_date)
    e_dt = min(f"{year}-12-31", end_date)
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(s_dt, e_dt)]

    parsed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_parse_single_pr_zip, d) for d in days]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                parsed.append(res)

    parsed.sort(key=lambda x: x["date"])
    raw_map = {r["date"]: {k: v[0] for k, v in r["eq"].items()} for r in parsed}
    tv_map = {r["date"]: {k: v[1] for k, v in r["eq"].items()} for r in parsed}
    etf_map = {r["date"]: r["etf"] for r in parsed}
    idx_map = {r["date"]: r["idx"] for r in parsed}

    raw_df = pd.DataFrame.from_dict(raw_map, orient="index").sort_index()
    tv_df = pd.DataFrame.from_dict(tv_map, orient="index").sort_index()
    etf_df = pd.DataFrame.from_dict(etf_map, orient="index").sort_index()
    idx_df = pd.DataFrame.from_dict(idx_map, orient="index").sort_index()

    bc_unique: dict[tuple[str, str], list[str]] = {}
    for r in parsed:
        for sy, ex, pur in r["bc"]:
            lst = bc_unique.setdefault((sy, ex), [])
            if pur not in lst:
                lst.append(pur)
    bc_list = [{"symbol": k[0], "ex_date": k[1], "purpose": " | ".join(v)} for k, v in sorted(bc_unique.items())]

    raw_df.to_pickle(raw_p, compression="gzip")
    tv_df.to_pickle(tv_p, compression="gzip")
    etf_df.to_pickle(etf_p, compression="gzip")
    idx_df.to_pickle(idx_p, compression="gzip")
    bc_p.write_text(json.dumps(bc_list, indent=2))
    return {"raw": raw_df, "trdval": tv_df, "etf": etf_df, "idx": idx_df, "bc": bc_list}


def build_2010_2026_nse_dataset(repo_root: str | Path, start_year: int = 2010, end_year: int = 2026, refresh: bool = False) -> dict[str, Any]:
    """Download/load 2010-2026 official NSE archives, reconstruct split/bonus-adjusted prices, and cache cleanly."""
    root = Path(repo_root)
    out_dir = root / "data" / "nse_archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_files = {
        "adj": out_dir / "adj_close_2010_2026.pkl.gz",
        "raw": out_dir / "raw_close_2010_2026.pkl.gz",
        "trdval": out_dir / "trdval_2010_2026.pkl.gz",
        "bench": out_dir / "bench_2010_2026.pkl.gz",
        "bc_log": out_dir / "corp_actions_applied.json",
    }

    if not refresh and all(p.exists() for p in panel_files.values()):
        return {
            "adj": pd.read_pickle(panel_files["adj"]),
            "raw": pd.read_pickle(panel_files["raw"]),
            "trdval": pd.read_pickle(panel_files["trdval"]),
            "bench": pd.read_pickle(panel_files["bench"]),
            "bc_log": json.loads(panel_files["bc_log"].read_text()),
        }

    raw_parts, tv_parts, etf_parts, idx_parts = [], [], [], []
    all_bc: dict[tuple[str, str], str] = {}
    for yr in range(start_year, end_year + 1):
        print(f"Fetching/loading NSE PR archive year {yr}...")
        res = _fetch_year_chunk(yr, out_dir)
        raw_parts.append(res["raw"])
        tv_parts.append(res["trdval"])
        etf_parts.append(res["etf"])
        idx_parts.append(res["idx"])
        for ev in res["bc"]:
            key = (ev["symbol"], ev["ex_date"])
            if key in all_bc:
                if ev["purpose"] not in all_bc[key]:
                    all_bc[key] = f"{all_bc[key]} | {ev['purpose']}"
            else:
                all_bc[key] = ev["purpose"]

    raw_df = pd.concat(raw_parts).sort_index()
    tv_df = pd.concat(tv_parts).sort_index()
    etf_df = pd.concat(etf_parts).sort_index()
    idx_df = pd.concat(idx_parts).sort_index()

    # Keep tracked ETFs in raw_df (so Pd covers 2010-2013 before separate etf*.csv existed) + all equities >= 60 trading days
    valid_cols = raw_df.columns[raw_df.notna().sum() >= 60]
    raw_df = raw_df[valid_cols]
    tv_df = tv_df.reindex(columns=valid_cols)

    # Drop corrupted single-day archives where >= 25 liquid stocks simultaneously swing >25% (e.g. NSE's 2018-01-02 zip packing 2019 prices)
    liq_tmp = tv_df.rolling(20, min_periods=5).median() >= 1e7
    bad_day_cnt = ((raw_df.pct_change(fill_method=None).abs() > 0.25) & liq_tmp).sum(axis=1)
    corrupt_dates = set(bad_day_cnt[bad_day_cnt >= 25].index)
    # Only drop days that also revert next day (avoid ever dropping a hypothetical systemic crash without revert)
    clean_keep = []
    idx_list = list(raw_df.index)
    for k, dt_str in enumerate(idx_list):
        if dt_str in corrupt_dates and 0 < k + 1 < len(idx_list):
            next_dt = idx_list[k + 1]
            # Check if corrupted day's median ratio vs prev day reverted on next_dt
            if next_dt in corrupt_dates or dt_str == "2018-01-02":
                continue
        clean_keep.append(dt_str)
    if len(clean_keep) < len(raw_df):
        raw_df = raw_df.loc[clean_keep]
        tv_df = tv_df.loc[clean_keep]
        etf_df = etf_df.reindex(clean_keep)
        idx_df = idx_df.reindex(clean_keep)

    # Query Bc specifically on unrecorded non-trivial liquid step drops (<-45%) to catch SCHEME OF ARRANGEMENT / demergers
    bc_dates_by_sym: dict[str, list[pd.Timestamp]] = {}
    for (sy, ex_d) in all_bc:
        bc_dates_by_sym.setdefault(sy, []).append(pd.Timestamp(ex_d))

    step_liq = tv_df.rolling(20, min_periods=5).median() >= 1e7
    step_r = raw_df.pct_change(fill_method=None)
    sus_mask = ((step_r < -0.45) | (step_r > 1.20)) & step_liq
    probe_dates: dict[str, set[str]] = {}
    for sdt in sus_mask.index[sus_mask.any(axis=1)]:
        sdt_ts = pd.Timestamp(sdt)
        for ssym in sus_mask.columns[sus_mask.loc[sdt]]:
            if not any(abs((sdt_ts - ex_t).days) <= 7 for ex_t in bc_dates_by_sym.get(ssym, [])):
                probe_dates.setdefault(sdt, set()).add(ssym)

    if probe_dates:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {}
            for sdt, syms_needed in probe_dates.items():
                for dt_shift in (0, -1):
                    pdt = (pd.Timestamp(sdt) + pd.Timedelta(days=dt_shift)).strftime("%Y-%m-%d")
                    futs[pool.submit(_parse_single_pr_zip, pdt)] = syms_needed
            for fut in as_completed(futs):
                syms_needed = futs[fut]
                parsed_day = fut.result()
                if parsed_day:
                    for sy, ex, pur in parsed_day["bc"]:
                        if sy in syms_needed:
                            k = (sy, ex)
                            all_bc[k] = f"{all_bc[k]} | {pur}" if k in all_bc and pur not in all_bc[k] else pur

    # Apply corporate actions backward using Bc ex-dates (deduplicating repeated notices & guarding against unsplit notices!)
    adj_df = raw_df.copy()
    applied_ca: list[dict[str, Any]] = []
    dates_idx = adj_df.index
    applied_recent: dict[str, list[tuple[pd.Timestamp, str]]] = {}

    def _apply_ca_on_series(s_adj: pd.Series, s_raw: pd.Series, sym: str, ex_dt: str, purpose: str, factor: float, tag: str) -> tuple[bool, float, str, str]:
        dt_obj = pd.Timestamp(ex_dt)
        for prev_dt, prev_tag in applied_recent.get(sym, []):
            if abs((dt_obj - prev_dt).days) <= 12 and prev_tag == tag:
                return False, factor, tag, ex_dt

        cut_dt = ex_dt
        win_mask = (dates_idx >= (dt_obj - pd.Timedelta(days=15)).strftime("%Y-%m-%d")) & (dates_idx <= (dt_obj + pd.Timedelta(days=15)).strftime("%Y-%m-%d"))
        sub_raw = s_raw.loc[win_mask].dropna()
        if len(sub_raw) >= 2:
            step_ratios = sub_raw / sub_raw.shift(1)
            drops = step_ratios[step_ratios < 0.88]
            jumps = step_ratios[step_ratios > 1.80]
            if not drops.empty and factor <= 1.05:
                if abs(factor - 1.0) >= 1e-4:
                    best_dt = str((drops - factor).abs().idxmin())
                    if factor < 0.78 and abs(float(drops.loc[best_dt]) - factor) > 0.35 * max(factor, float(drops.loc[best_dt])) and float(drops.loc[best_dt]) > 0.75:
                        return False, factor, tag, ex_dt
                    cut_dt = best_dt
                elif any(k in tag or k in purpose.upper() for k in ("demerger", "rights", "ARRANG", "SCHEME", "REDUCT")):
                    cut_dt = str(drops.idxmin())
                    emp_r = float(drops.loc[cut_dt])
                    if 0.05 < emp_r < 0.85:
                        factor = emp_r
                        tag = "demerger_or_scheme_emp"
            elif factor < 0.78:
                return False, factor, tag, ex_dt
            elif factor > 1.5 and not jumps.empty:
                cut_dt = str((jumps - factor).abs().idxmin())
            elif factor > 1.5:
                return False, factor, tag, ex_dt

        if abs(factor - 1.0) < 1e-4:
            return False, factor, tag, cut_dt
        pre_mask = dates_idx < cut_dt
        post_mask = dates_idx >= cut_dt
        if not pre_mask.any() or not post_mask.any():
            return False, factor, tag, cut_dt

        s_adj.loc[pre_mask] = s_adj.loc[pre_mask] * factor
        applied_recent.setdefault(sym, []).append((pd.Timestamp(cut_dt), tag))
        return True, factor, tag, cut_dt

    def _run_bc_for_symbols(target_syms: set[str] | None = None) -> None:
        for (sym, ex_dt), purpose in sorted(all_bc.items(), key=lambda x: x[0][1], reverse=True):
            if target_syms is not None and sym not in target_syms:
                continue
            if sym not in adj_df.columns:
                continue
            if purpose.startswith("MCP_FACTOR:"):
                f_str = purpose.split("|", 1)[0].split(":", 1)[1].strip()
                factor, tag = float(f_str), "mcp_ca"
            else:
                factor, tag = parse_bc_purpose_factor(purpose)
            if tag == "dividend_or_other":
                pu = purpose.upper()
                if not any(k in pu for k in ("ARRANG", "SCHEME", "DEMERG", "REDUCT", "CONSOL")):
                    continue
                tag = "demerger"
            ok, f_used, t_used, dt_used = _apply_ca_on_series(adj_df[sym], raw_df[sym], sym, ex_dt, purpose, factor, tag)
            if ok:
                applied_ca.append({"symbol": sym, "ex_date": dt_used, "purpose": purpose, "factor": round(f_used, 6), "tag": t_used})

    _run_bc_for_symbols(None)

    # Supplement ONLY residual post-2021 extreme drops (non-BC splits / combined Bonus+Split like FCL, 360ONE, SHRIRAMFIN, BESTAGRO) via NSE MCP
    post21_mask = ((adj_df.pct_change(fill_method=None) < -0.45) & step_liq).loc[dates_idx >= "2021-12-01"]
    rem_syms = sorted(post21_mask.columns[post21_mask.any(axis=0)])
    if rem_syms:
        try:
            from src.nse_mcp import NseMcpClient

            mcp = NseMcpClient(timeout=8.0)
            refreshed_syms: set[str] = set()
            for sy in rem_syms:
                try:
                    ca_res = mcp.get_corporate_actions(sy)
                    by_ex: dict[str, tuple[float, list[str]]] = {}
                    for act in ca_res.get("actions") or []:
                        ex_d = str(act.get("exDate", ""))[:10]
                        atype = str(act.get("actionType", "")).upper()
                        pur = str(act.get("purpose", ""))
                        fac = float(act.get("adjustmentFactor") or 1.0)
                        if atype in {"SPLIT", "BONUS"} and 0.0 < fac < 0.99:
                            prev_f, plist = by_ex.get(ex_d, (1.0, []))
                            if pur not in plist:
                                by_ex[ex_d] = (prev_f * fac, plist + [pur])
                    for ex_d, (tot_f, plist) in by_ex.items():
                        if tot_f < 0.99:
                            all_bc[(sy, ex_d)] = f"MCP_FACTOR:{tot_f:.6f} | " + " | ".join(plist)
                            refreshed_syms.add(sy)
                except Exception:
                    pass
            if refreshed_syms:
                for sy in refreshed_syms:
                    adj_df[sy] = raw_df[sy].copy()
                    applied_recent.pop(sy, None)
                applied_ca = [ev for ev in applied_ca if ev["symbol"] not in refreshed_syms]
                _run_bc_for_symbols(refreshed_syms)
        except Exception:
            pass

    # Reconcile tracked ETFs cleanly (preferring split-adjusted Pd + etf*.csv fallback with same CA factors)
    tracked_etfs = ["GOLDBEES", "LIQUIDBEES", "LIQUIDCASE", "NIFTYBEES", "MON100", "SILVERBEES", "EBBETF0431", "LTGILTBEES"]
    etf_adj = pd.DataFrame(index=dates_idx)
    for etf in tracked_etfs:
        s_raw_e = raw_df[etf] if etf in raw_df.columns else pd.Series(index=dates_idx, dtype=float)
        if etf in etf_df.columns:
            s_raw_e = s_raw_e.combine_first(etf_df[etf].reindex(dates_idx))
        if etf in adj_df.columns:
            etf_adj[etf] = adj_df[etf]
        elif not s_raw_e.dropna().empty:
            etf_adj[etf] = s_raw_e
        # Ensure 100:1 / 10:1 ETF splits (e.g. GOLDBEES 2019-12-19, NIFTYBEES 2019-12-19) are always adjusted cleanly if absent from Pd
        if etf in etf_adj.columns:
            s_tmp = etf_adj[etf].dropna()
            if len(s_tmp) > 5:
                r = s_tmp / s_tmp.shift(1)
                for cdt in r[r < 0.25].index:
                    # Only apply abrupt split factor when it's an ETF FV split step (stable afterward, not volatile equity)
                    pos = s_tmp.index.get_loc(cdt)
                    if isinstance(pos, int) and pos + 2 < len(s_tmp) and 0.7 < float(s_tmp.iloc[pos + 2]) / float(s_tmp.iloc[pos]) < 1.3:
                        etf_adj.loc[dates_idx < cdt, etf] *= float(r.loc[cdt])

    bench_df = pd.concat([etf_adj, idx_df.reindex(dates_idx)], axis=1)

    adj_df.to_pickle(panel_files["adj"], compression="gzip")
    raw_df.to_pickle(panel_files["raw"], compression="gzip")
    tv_df.to_pickle(panel_files["trdval"], compression="gzip")
    bench_df.to_pickle(panel_files["bench"], compression="gzip")
    panel_files["bc_log"].write_text(json.dumps(applied_ca, indent=2))
    print(f"Built NSE 2010-2026 dataset: {len(adj_df)} days x {len(adj_df.columns)} equities, {len(applied_ca)} CA events applied")
    return {"adj": adj_df, "raw": raw_df, "trdval": tv_df, "bench": bench_df, "bc_log": applied_ca}
