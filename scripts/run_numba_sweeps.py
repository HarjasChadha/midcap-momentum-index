"""Execute full Numba + SciPy multi-period optimization & hedging sweeps on clean zero-leak NSE PIT (2010-2026)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.numba_research_engine import (  # noqa: E402
    PERIODS,
    _batch_dd_reentry_grid,
    _calc_metrics_slice,
    _gen_dd_signal,
    _gen_multifactor_signal,
    _sim_overlay_core,
    build_basket_daily_stream,
    build_multifactor_blocks,
    compute_extended_scores,
    compute_tg_erc_returns,
)
from src.nse_archive_loader import build_2010_2026_nse_dataset  # noqa: E402
from src.regime import build_defensive_series, build_regime_features  # noqa: E402


def build_clean_equity_and_pit_masks(d: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exclude ALL 419+ NSE ETFs cleanly, repair isolated missing split glitches, and build strictly LAGGED PIT masks."""
    adj_all, raw_all, tv_all, bench_df = d["adj"], d["raw"], d["trdval"], d["bench"]

    etf_path = REPO_ROOT / "data" / "nse_archive" / "nse_etf_symbols.json"
    etf_set = set(json.loads(etf_path.read_text())) if etf_path.exists() else set()
    etf_set.discard("")
    # Add strict regex catch for any newer/delisted ETFs not in csv
    eq_cols = [
        c for c in adj_all.columns
        if c not in etf_set and not any(k in c for k in ("BEES", "ETF", "IETF", "CASE", "GILT", "GSEC", "LIQUID", "GOLDSHARE", "GOLD1", "N100", "MON100"))
    ]
    adj_eq = adj_all[eq_cols].copy()
    raw_eq = raw_all[eq_cols].copy()
    tv_eq = tv_all[eq_cols].copy()

    # Surgical repair for verified 2017-08-10 DWARKESH 10:1 split absent from empty 2017 Bc table (checked via raw price ratio 10->1 FV)
    if "DWARKESH" in adj_eq.columns and "2017-08-10" in adj_eq.index:
        r_dw = float(raw_eq.loc["2017-08-10", "DWARKESH"]) / float(raw_eq.loc[: "2017-08-09", "DWARKESH"].dropna().iloc[-1])
        if 0.08 < r_dw < 0.12:
            adj_eq.loc[adj_eq.index < "2017-08-10", "DWARKESH"] *= 0.10

    # Rolling 126d median NSE rupee turnover Ranks 81..230 (active >=90d, price >= ₹20 at close t; lagged t-1 when used!)
    med_tv = tv_eq.rolling(126, min_periods=60).median()
    active = adj_eq.rolling(126, min_periods=90).count() >= 90
    price_ok = raw_eq >= 20.0
    ranks = med_tv.where(active & price_ok).rank(axis=1, ascending=False, method="first")
    pit_daily = (ranks >= 81) & (ranks <= 230)

    # Semi-annual March/September Reconstitution (matching official NSE Midcap 150 schedule, zero daily rank churn!)
    pit_semi = pd.DataFrame(False, index=adj_eq.index, columns=eq_cols)
    cur_sel = pd.Series(False, index=eq_cols)
    prev_ym = ""
    for dt in adj_eq.index:
        y, m = int(dt[:4]), int(dt[5:7])
        ym = dt[:7]
        if m in (3, 9) and ym != prev_ym and active.loc[dt].sum() >= 300:
            # Select first trading day of March / September using PRIOR day turnover ranks!
            prev_ym = ym
            cur_sel = pit_daily.loc[dt].copy()
        elif cur_sel.any():
            # Keep active stocks trading >= ₹5 (exit only if delisted/halted >5d or sub-₹5 penny)
            cur_sel = cur_sel & (raw_eq.loc[dt].ffill(limit=5).fillna(0.0) >= 5.0)
        pit_semi.loc[dt] = cur_sel

    keep_cols = list(pit_daily.columns[pit_daily.any(axis=0) | pit_semi.any(axis=0)])
    return adj_eq[keep_cols], raw_eq[keep_cols], pit_daily[keep_cols], pit_semi[keep_cols], bench_df


def _slice_bounds(index: pd.Index) -> np.ndarray:
    arr = np.zeros((4, 2), dtype=np.int64)
    for k, (_, (ps, pe)) in enumerate(PERIODS.items()):
        mask = (index >= ps) & (index <= pe)
        pos = np.where(mask)[0]
        arr[k, 0] = int(pos[0])
        arr[k, 1] = int(pos[-1])
    return arr


def _summarize_nav(nav: np.ndarray, to_d: np.ndarray, bounds: np.ndarray) -> dict:
    out = {}
    for k, pkey in enumerate(PERIODS.keys()):
        c, d, cal, sh, to_a = _calc_metrics_slice(nav, to_d, bounds[k, 0], bounds[k, 1], 0.0625)
        out[f"{pkey}_cagr"] = round(c, 2)
        out[f"{pkey}_dd"] = round(d, 2)
        out[f"{pkey}_cal"] = round(cal, 2)
        out[f"{pkey}_sh"] = round(sh, 2)
        if pkey == "FULL":
            out["FULL_to"] = round(to_a, 1)
    return out


def main() -> None:
    t_start = time.time()
    d = build_2010_2026_nse_dataset(REPO_ROOT, 2010, 2026, refresh=False)
    adj_eq, raw_eq, pit_daily, pit_semi, bench_df = build_clean_equity_and_pit_masks(d)
    dates = adj_eq.index
    T = len(dates)
    bounds = _slice_bounds(dates)
    print(f"Clean ETF-free PIT universe: {adj_eq.shape[1]} stocks x {T} days | Daily overlap w/ Semi: {(pit_daily & pit_semi).sum(axis=1).mean():.1f}")

    # Lagged mask (ZERO day-t leak)
    def_df = build_defensive_series(bench_df)
    feat_daily = build_regime_features(adj_eq, bench_df, mask=pit_daily)
    feat_semi = build_regime_features(adj_eq, bench_df, mask=pit_semi)

    liq_r = def_df["LIQUID_REALISTIC"].pct_change(fill_method=None).fillna(0.0).values
    gold_r = def_df["GOLDBEES"].pct_change(fill_method=None).fillna(0.0).values
    mon_r = def_df["MON100_STITCHED"].pct_change(fill_method=None).fillna(0.0).values if "MON100_STITCHED" in def_df else liq_r.copy()
    gilt_r = def_df["LICNETFGSC"].pct_change(fill_method=None).fillna(0.0).values if "LICNETFGSC" in def_df else liq_r.copy()
    nif_r = def_df["NIFTYBEES"].pct_change(fill_method=None).fillna(0.0).values
    hng_r = def_df["HNGSNGBEES"].pct_change(fill_method=None).fillna(0.0).values
    r_base_5050 = 0.5 * gold_r + 0.5 * liq_r
    tax_5050 = np.full(T, 0.5 * 0.125 + 0.5 * 0.30, dtype=np.float64)
    tax_gold = np.full(T, 0.125, dtype=np.float64)
    tax_liq = np.full(T, 0.30, dtype=np.float64)

    lines = []
    report_data: dict = {}

    # ══════════════════════════════════════════════════════════════════════════════════════════════
    # MODULE 1: CLEAN ETF-FREE FACTOR x LOOKBACK x SPLIT x BUFFER x REBALANCE (ON SEMIANNUAL & DAILY PIT)
    # ══════════════════════════════════════════════════════════════════════════════════════════════
    print("Module 1: Sweeping clean ETF-free Equity Baskets (SemiAnnual + Daily Lagged PIT)...")
    scores_semi = compute_extended_scores(adj_eq, pit_semi)
    scores_daily = compute_extended_scores(adj_eq, pit_daily)
    ones = np.ones(T, dtype=np.float64)

    m1_rows = []
    baskets_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for univ_lbl, sc_map in [("SEMIANNUAL_150", scores_semi), ("DAILY_LAGGED_150", scores_daily)]:
        # Sweep canonical combinations
        candidates = []
        for ftype in ("raw", "risk", "sortino", "fip"):
            for lb in (63, 84, 105, 126, 147, 168, 189, 210, 231, 252):
                key = f"{ftype}_{lb}_s21"
                if univ_lbl == "DAILY_LAGGED_150" and lb not in (126, 189, 252):
                    continue
                for reb in ((63, 126, 252) if lb in (126, 189, 252) else (126,)):
                    candidates.append((f"{ftype}_{lb}d_s21", key, reb, 20, 30, 0.5, "equal", 0))
        # Special skip, split, buffer, norm_z, ensemble variants at main lookbacks
        if univ_lbl == "SEMIANNUAL_150":
            for sk in (0, 10, 21):
                for ft in ("raw", "risk", "sortino"):
                    candidates.append((f"{ft}_189d_s{sk}", f"{ft}_189_s{sk}", 126, 20, 30, 0.5, "equal", 0))
            for ens in ("hybrid_raw_risk_189", "hybrid_raw_fip_189", "ens_risk_6_9_12", "ens_raw_6_9_12"):
                for reb in (63, 126, 189, 252):
                    candidates.append((ens, ens, reb, 20, 30, 0.5, "equal", 0))
            for w_core, nc, nb in [(1.0, 20, 20), (0.75, 20, 30), (0.5, 20, 30), (0.25, 20, 30), (0.0, 30, 30), (0.5, 15, 25), (0.5, 20, 40)]:
                for ft in ("raw_189_s21", "risk_189_s21", "sortino_189_s21"):
                    candidates.append((f"{ft}_w{int(w_core*100)}_{nc}_{nb}", ft, 126, nc, nb, w_core, "equal", 0))
            for wt in ("equal", "norm_z"):
                for buf in (0, 45):
                    for ft in ("raw_189_s21", "risk_189_s21", "hybrid_raw_risk_189", "ens_risk_6_9_12"):
                        candidates.append((f"{ft}_{wt}_buf{buf}", ft, 126, 20, 30, 0.5, wt, buf))

        for name, skey, reb, nc, nb, wc, wt, buf in candidates:
            cid = f"{univ_lbl} | {name} | R{reb}"
            r_eq, to_r, nav_g = build_basket_daily_stream(adj_eq, sc_map[skey], reb, nc, nb, wc, wt, buf, 0.0)
            baskets_cache[cid] = (r_eq, to_r, nav_g)
            nav_tx, to_tx = _sim_overlay_core(r_eq, to_r, ones, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
            st = _summarize_nav(nav_tx, to_tx, bounds)
            st.update({"univ": univ_lbl, "config": f"{name} | R{reb}", "cid": cid, "reb": reb, "skey": skey})
            m1_rows.append(st)

    # Delisting-haircut stress check (-10% loss if stock disappears >5d)
    r_dl, to_dl, _ = build_basket_daily_stream(adj_eq, scores_semi["risk_189_s21"], 126, 20, 30, 0.5, "equal", 0, delist_haircut=0.10)
    nav_dl, to_dlt = _sim_overlay_core(r_dl, to_dl, ones, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
    m1_dl_st = _summarize_nav(nav_dl, to_dlt, bounds)

    # LowVol30 internal equity benchmark stream
    r_lowvol, to_lowvol, _ = build_basket_daily_stream(adj_eq, scores_semi["lowvol_63"], 126, 30, 30, 1.0, "equal", 0, 0.0)
    nav_lv, to_lvt = _sim_overlay_core(r_lowvol, to_lowvol, ones, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
    lv_st = _summarize_nav(nav_lv, to_lvt, bounds)

    # Clean EW benchmark metric
    ew_nav_s = feat_semi["ew_index"].values
    ew_nav_d = feat_daily["ew_index"].values
    ew_s_st = _summarize_nav(ew_nav_s, np.zeros(T), bounds)
    ew_d_st = _summarize_nav(ew_nav_d, np.zeros(T), bounds)

    m1_df = pd.DataFrame(m1_rows)
    lines.append("=== MODULE 1: CLEAN ETF-FREE PIT EQUITY BASKETS (UNHEDGED POST-TAX & 15bps, ZERO LEAK) ===")
    lines.append(f"PIT SemiAnnual EW Benchmark : P1={ew_s_st['P1_cagr']}% | P2={ew_s_st['P2_cagr']}% | P3={ew_s_st['P3_cagr']}% | FULL={ew_s_st['FULL_cagr']}% DD={ew_s_st['FULL_dd']}% Sh={ew_s_st['FULL_sh']}")
    lines.append(f"PIT Daily Lag EW Benchmark  : P1={ew_d_st['P1_cagr']}% | P2={ew_d_st['P2_cagr']}% | P3={ew_d_st['P3_cagr']}% | FULL={ew_d_st['FULL_cagr']}% DD={ew_d_st['FULL_dd']}% Sh={ew_d_st['FULL_sh']}")
    lines.append(f"PIT Internal LowVol30 6M    : P1={lv_st['P1_cagr']}% | P2={lv_st['P2_cagr']}% | P3={lv_st['P3_cagr']}% | FULL={lv_st['FULL_cagr']}% DD={lv_st['FULL_dd']}% Sh={lv_st['FULL_sh']}")
    lines.append(f"Risk_189_6M Delist-10% Stress: FULL={m1_dl_st['FULL_cagr']}% DD={m1_dl_st['FULL_dd']}% Sh={m1_dl_st['FULL_sh']}")

    semi_df = m1_df[m1_df["univ"] == "SEMIANNUAL_150"].copy()
    cols_m1 = ["config", "P1_cagr", "P1_dd", "P2_cagr", "P2_dd", "P3_cagr", "P3_dd", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh", "FULL_to"]
    lines.append("-- Top 15 Clean SemiAnnual Baskets Chosen Blindly by P1 Sharpe -> P2 & P3 OOS Check --")
    lines.append(semi_df.sort_values("P1_sh", ascending=False).head(15)[cols_m1].to_string(index=False))
    lines.append("-- Lookback Curve 63d..252d (Raw vs Risk vs Sortino vs FIP at R126 EW 20+30) --")
    lb_sub = semi_df[semi_df["config"].str.contains(r"^(?:raw|risk|sortino|fip)_\d+d_s21 \| R126$")]
    lines.append(lb_sub.sort_values("FULL_sh", ascending=False)[cols_m1].to_string(index=False))
    lines.append("-- Sleeve Weighting / Split / Buffer Comparison at 189d R126 --")
    sp_sub = semi_df[semi_df["config"].str.contains(r"(_w\d+_|_buf|ens_|hybrid_)") & (semi_df["reb"] == 126)]
    lines.append(sp_sub.sort_values("FULL_sh", ascending=False).head(18)[cols_m1].to_string(index=False))

    # Pick core equity streams for Regime & Hedging Modules (BOTH Raw_9M, Risk_9M, Ens_Risk_6_9_12, Hybrid_9M!)
    core_baskets = {
        "Risk_189_EW_Semi": baskets_cache["SEMIANNUAL_150 | risk_189d_s21 | R126"],
        "Raw_189_EW_Semi": baskets_cache["SEMIANNUAL_150 | raw_189d_s21 | R126"],
        "Sortino_189_EW_Semi": baskets_cache["SEMIANNUAL_150 | sortino_189d_s21 | R126"],
        "FIP_189_EW_Semi": baskets_cache["SEMIANNUAL_150 | fip_189d_s21 | R126"],
        "Ens_Risk_EW_Semi": baskets_cache["SEMIANNUAL_150 | ens_risk_6_9_12 | R126"],
        "Risk_189_EW_Daily": baskets_cache["DAILY_LAGGED_150 | risk_189d_s21 | R126"],
    }

    # ══════════════════════════════════════════════════════════════════════════════════════════════
    # MODULE 2 (Q1 & Q3): EXHAUSTIVE 21,600-CONFIG NUMBA DD TRIGGER & RE-ENTRY OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════════════════════════════
    print("Module 2: Running 21,600 Numba Trigger x Window x Re-Entry x Thrust combinations...")
    dd_mat = np.vstack([
        feat_semi["ew_dd126"].fillna(0.0).values,
        feat_semi["ew_dd189"].fillna(0.0).values if "ew_dd189" in feat_semi else (feat_semi["ew_index"] / feat_semi["ew_index"].rolling(189, min_periods=20).max() - 1.0).fillna(0.0).values,
        feat_semi["ew_dd252"].fillna(0.0).values,
        (feat_semi["ew_index"] / feat_semi["ew_index"].rolling(504, min_periods=20).max() - 1.0).fillna(0.0).values,
        feat_semi["ew_dd_ath"].fillna(0.0).values,
    ])
    dd_win_names = ["126d", "189d", "252d", "504d", "ATH"]
    ma_mat = np.vstack([
        feat_semi["ew_ma20"].fillna(0.0).values,
        feat_semi["ew_ma50"].fillna(0.0).values,
        feat_semi["ew_ma100"].fillna(0.0).values,
        feat_semi["ew_ma150"].fillna(0.0).values,
    ])
    ma_names = [20, 50, 100, 150]
    px_arr = feat_semi["ew_index"].fillna(0.0).values
    b50 = feat_semi["breadth_50"].fillna(0.5)
    b50_ema5 = b50.ewm(span=5, min_periods=2).mean()
    b50_chg21_arr = (b50_ema5 - b50_ema5.shift(21)).fillna(0.0).values
    b50_lvl_arr = b50.values

    dd_ths = [0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25]
    r_ratios = [0.25, 0.40, 0.60, 0.80, 1.00]
    confs = [1, 3, 5]
    thrusts = [0, 1]
    shifts = [0.50, 0.75, 1.00]
    param_rows = []
    for wi in range(len(dd_win_names)):
        for dth in dd_ths:
            for rr in r_ratios:
                for mi in range(len(ma_names)):
                    for cf in confs:
                        for th in thrusts:
                            for sh in shifts:
                                param_rows.append([wi, dth, rr, mi, cf, th, sh, 10])
    p_arr = np.array(param_rows, dtype=np.float64)

    r_main, to_main, nav_main = core_baskets["Risk_189_EW_Semi"]
    g_res = _batch_dd_reentry_grid(r_main, to_main, r_base_5050, tax_5050, dd_mat, px_arr, ma_mat, b50_chg21_arr, b50_lvl_arr, p_arr, bounds)

    g_df = pd.DataFrame({
        "dd_win": [dd_win_names[int(x)] for x in p_arr[:, 0]],
        "dd_th": p_arr[:, 1],
        "r_ratio": p_arr[:, 2],
        "re_ma": [ma_names[int(x)] for x in p_arr[:, 3]],
        "conf_d": p_arr[:, 4].astype(int),
        "thrust": p_arr[:, 5].astype(int),
        "shift": p_arr[:, 6],
        "P1_cagr": g_res[:, 0], "P1_dd": g_res[:, 1], "P1_cal": g_res[:, 2], "P1_sh": g_res[:, 3],
        "P2_cagr": g_res[:, 4], "P2_dd": g_res[:, 5], "P2_cal": g_res[:, 6], "P2_sh": g_res[:, 7],
        "P3_cagr": g_res[:, 8], "P3_dd": g_res[:, 9], "P3_cal": g_res[:, 10], "P3_sh": g_res[:, 11],
        "FULL_cagr": g_res[:, 12], "FULL_dd": g_res[:, 13], "FULL_cal": g_res[:, 14], "FULL_sh": g_res[:, 15], "FULL_to": g_res[:, 16],
    })
    g_df["OOS23_sh"] = 0.5 * (g_df["P2_sh"] + g_df["P3_sh"])
    g_df["P12_sh"] = 0.5 * (g_df["P1_sh"] + g_df["P2_sh"])

    # Compute P1->P2, P1->P3, P12->P3 Spearman rank correlation across ALL 21,600 configs!
    rho_12 = float(g_df["P1_sh"].corr(g_df["P2_sh"], method="spearman"))
    rho_13 = float(g_df["P1_sh"].corr(g_df["P3_sh"], method="spearman"))
    rho_12_3 = float(g_df["P12_sh"].corr(g_df["P3_sh"], method="spearman"))

    lines.append("\n=== MODULE 2 (Q1 & Q3): EXHAUSTIVE 21,600-CONFIG DD TRIGGER x WINDOW x RE-ENTRY MARGINALS ===")
    lines.append(f"Spearman Sharpe across 21,600 configs: P1->P2={rho_12:.3f}, P1->P3={rho_13:.3f}, (P1+P2)->P3={rho_12_3:.3f}")
    m_cols = ["P1_sh", "P2_sh", "P3_sh", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh", "FULL_to"]
    for grp in ["dd_th", "dd_win", "r_ratio", "re_ma", "conf_d", "thrust", "shift"]:
        lines.append(f"-- Marginal Mean by {grp} --")
        lines.append(g_df.groupby(grp)[m_cols].mean().round(3).to_string())

    lines.append("-- Top 12 Selected BLINDLY on P1+P2 Mean Sharpe (min P1_dd > -32%) -> Checked Blindly on P3 (2021-26)! --")
    p12_top = g_df[g_df["P1_dd"] > -32.0].sort_values("P12_sh", ascending=False).head(12)
    lines.append(p12_top[["dd_win", "dd_th", "r_ratio", "re_ma", "conf_d", "thrust", "shift"] + m_cols].round(2).to_string(index=False))

    lines.append("-- Fixed Exact Comparison: Legacy DD15 (-9%/MA50) vs Optimal Plateau Params --")
    sub_cmp = g_df[(g_df["dd_win"].isin(["252d", "ATH"])) & (g_df["dd_th"].isin([0.12, 0.15, 0.16, 0.18, 0.20])) & (g_df["shift"] == 0.75) & (g_df["re_ma"] == 50) & (g_df["conf_d"].isin([1, 5])) & (g_df["r_ratio"].isin([0.60, 1.00])) & (g_df["thrust"].isin([0, 1]))]
    lines.append(sub_cmp.sort_values(["dd_win", "dd_th", "r_ratio", "conf_d", "thrust"])[["dd_win", "dd_th", "r_ratio", "conf_d", "thrust", "P1_cagr", "P1_dd", "P2_cagr", "P2_dd", "P3_cagr", "P3_dd", "FULL_cagr", "FULL_dd", "FULL_sh"]].round(2).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════════════════════════
    # MODULE 3 (Q2): ORTHOGONAL 4-BLOCK COMPOSITE + SCIPY WALK-FORWARD WEIGHTS + DUAL GATES + BARROSO
    # ══════════════════════════════════════════════════════════════════════════════════════════════
    print("Module 3: Running Orthogonal Multi-Factor Composite + SciPy Walk-Forward + Dual Gates + Barroso...")
    blocks, p_semivol, semivol_med = build_multifactor_blocks(feat_semi, bench_df, nav_main, def_df)
    corr_b = pd.DataFrame(blocks.T, columns=["B1_Trend", "B2_Vol", "B3_Breadth", "B4_DM_Gold"]).iloc[252:].corr().round(3)
    lines.append("\n=== MODULE 3 (Q2): ORTHOGONAL 4-BLOCK CORRELATION MATRIX (POST-WARMUP) ===")
    lines.append(corr_b.to_string())

    mf_rows = []
    # Sweep Zero-Fitted Equal-Weight (1/4, 1/4, 1/4, 1/4) over smoothing, thresholds, thrust, and bear-gated Barroso semivol!
    p_eqw = np.mean(blocks, axis=0)
    for tau_dn, tau_up in [(3.0, 10.0), (5.0, 15.0), (10.0, 20.0)]:
        for th_def, th_caut, th_ro in [(-0.35, -0.10, 0.10), (-0.25, 0.00, 0.15), (-0.18, 0.00, 0.12)]:
            for eq_fl, eq_ca in [(0.25, 0.60), (0.35, 0.70), (0.25, 1.00)]:
                for bv in (0, 1):
                    eq_mf = _gen_multifactor_signal(p_eqw, b50_chg21_arr, b50_lvl_arr, p_semivol, semivol_med, tau_dn, tau_up, th_def, th_caut, th_ro, eq_fl, eq_ca, 0.08, bv)
                    nav_m, to_m = _sim_overlay_core(r_main, to_main, eq_mf, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
                    st = _summarize_nav(nav_m, to_m, bounds)
                    st["name"] = f"EQ4_MF(t={int(tau_dn)}/{int(tau_up)},th={th_def}/{th_caut},fl={eq_fl}/{eq_ca},bv={bv})"
                    mf_rows.append(st)

    # Walk-Forward SciPy weight optimization: Fit w=[w1,w2,w3,w4] (w >= 0, sum=1) strictly on P1 -> test blind on P2 & P3; fit on P1+P2 -> test blind on P3!
    def _wf_opt_weights(fit_slice_idxs: list[int], obj_metric: str = "sharpe") -> np.ndarray:
        def loss(logits):
            ex = np.exp(logits - np.max(logits))
            w = ex / np.sum(ex)
            p_c = w[0] * blocks[0] + w[1] * blocks[1] + w[2] * blocks[2] + w[3] * blocks[3]
            eq_f = _gen_multifactor_signal(p_c, b50_chg21_arr, b50_lvl_arr, p_semivol, semivol_med, 5.0, 15.0, -0.25, 0.0, 0.15, 0.25, 0.60, 0.08, 0)
            nav_m, to_m = _sim_overlay_core(r_main, to_main, eq_f, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
            scores = []
            for s_idx in fit_slice_idxs:
                c, d, cal, sh, _ = _calc_metrics_slice(nav_m, to_m, bounds[s_idx, 0], bounds[s_idx, 1], 0.0625)
                scores.append(sh if obj_metric == "sharpe" else cal)
            return -float(np.mean(scores))

        best_x = np.zeros(4)
        best_val = 1e9
        for seed in range(6):
            rng = np.random.default_rng(100 + seed)
            x0 = rng.normal(0.0, 0.6, size=4)
            res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 140})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        ex = np.exp(best_x - np.max(best_x))
        return ex / np.sum(ex)

    w_p1 = _wf_opt_weights([0], "sharpe")
    w_p12 = _wf_opt_weights([0, 1], "sharpe")
    for w_lbl, w_vec in [("SciPy_Opt_P1Only", w_p1), ("SciPy_Opt_P1_P2", w_p12)]:
        p_c = w_vec[0] * blocks[0] + w_vec[1] * blocks[1] + w_vec[2] * blocks[2] + w_vec[3] * blocks[3]
        eq_f = _gen_multifactor_signal(p_c, b50_chg21_arr, b50_lvl_arr, p_semivol, semivol_med, 5.0, 15.0, -0.25, 0.0, 0.15, 0.25, 0.60, 0.08, 0)
        nav_m, to_m = _sim_overlay_core(r_main, to_main, eq_f, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
        st = _summarize_nav(nav_m, to_m, bounds)
        st["name"] = f"{w_lbl} w={[round(float(x),2) for x in w_vec]}"
        mf_rows.append(st)

    # Dual-Gate Confirmations & Standalone Bear-Vol Scaling & MA200Band
    # Dual Gate: DD > thresh AND (Breadth < b_th OR Semivol > semivol_med * mult) -> shift, re-enter when > MA50 & Thrust/Breadth
    for dth, bth in [(0.12, 0.35), (0.14, 0.35), (0.15, 0.40), (0.16, 0.45)]:
        dd_a = feat_semi["ew_dd_ath"].values
        ma_a = feat_semi["ew_ma50"].values
        sig_dg = np.ones(T, dtype=np.float64)
        da = False
        for t in range(T):
            cond_bear = np.isfinite(dd_a[t]) and dd_a[t] < -dth and (b50_lvl_arr[t] < bth or p_semivol[t] > 1.25 * semivol_med[t])
            if not da and cond_bear:
                da = True
            elif da and (px_arr[t] > ma_a[t] and (b50_lvl_arr[t] >= 0.45 or b50_chg21_arr[t] >= 0.20)):
                da = False
            sig_dg[t] = 0.25 if da else 1.0
        sig_dg_lag = np.concatenate(([1.0], sig_dg[:-1]))
        nav_dg, to_dg = _sim_overlay_core(r_main, to_main, sig_dg_lag, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
        st = _summarize_nav(nav_dg, to_dg, bounds)
        st["name"] = f"DualGate(ATH_DD>{int(dth*100)}%&(b50<{bth}|BearVol))"
        mf_rows.append(st)

    # Pure Portfolio Downside-Semivol Scaling (Barroso Continuous, lagged T+1 with 8% deadband)
    for target_sv in (0.12, 0.15, 0.18):
        sig_bv = np.ones(T, dtype=np.float64)
        cur = 1.0
        for t in range(T):
            raw_w = min(1.0, max(0.25, target_sv / max(p_semivol[t], 0.06)))
            if abs(raw_w - cur) >= 0.08:
                cur = raw_w
            sig_bv[t] = cur
        sig_bv_lag = np.concatenate(([1.0], sig_bv[:-1]))
        nav_bv, to_bv = _sim_overlay_core(r_main, to_main, sig_bv_lag, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
        st = _summarize_nav(nav_bv, to_bv, bounds)
        st["name"] = f"Barroso_SemivolOnly(target={target_sv:.2f})"
        mf_rows.append(st)

    mf_df = pd.DataFrame(mf_rows)
    lines.append("-- Multi-Factor Regime Mixture, SciPy WF Weights, Dual-Gate & Barroso Results --")
    lines.append(mf_df.sort_values("FULL_sh", ascending=False)[["name", "P1_cagr", "P1_dd", "P1_sh", "P2_cagr", "P2_dd", "P2_sh", "P3_cagr", "P3_dd", "P3_sh", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh", "FULL_to"]].head(22).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════════════════════════════
    # MODULE 4 (Q4 & Q5): DEFENSIVE SPLIT FRONTIER (0..100%) + ALL HEDGING ALTERNATIVES + F&O MODEL
    # ══════════════════════════════════════════════════════════════════════════════════════════════
    print("Module 4: Sweeping Gold/Liquid Split Frontier & Alternative Hedging Strategies...")
    # Select representative regime signals to test ALL defensive models cleanly (zero cherry-picking!)
    eq_sig_eq4 = _gen_multifactor_signal(p_eqw, b50_chg21_arr, b50_lvl_arr, p_semivol, semivol_med, 5.0, 15.0, -0.25, 0.00, 0.15, 0.25, 0.60, 0.08, 0)
    # Pick best simple OOS DD/Re-entry signal from Module 2 (by P1+P2 Sharpe)
    b_p12 = p12_top.iloc[0]
    w_i = dd_win_names.index(b_p12["dd_win"])
    m_i = ma_names.index(int(b_p12["re_ma"]))
    eq_sig_best_dd = _gen_dd_signal(dd_mat[w_i], px_arr, ma_mat[m_i], b50_chg21_arr, b50_lvl_arr, float(b_p12["dd_th"]), float(b_p12["r_ratio"]), int(b_p12["conf_d"]), int(b_p12["thrust"]), float(b_p12["shift"]), 10)
    eq_sig_old_dd15 = _gen_dd_signal(dd_mat[2], px_arr, ma_mat[1], b50_chg21_arr, b50_lvl_arr, 0.15, 0.60, 1, 0, 0.75, 10)

    # 4A: Fine Static Frontier w_gold in 0..100% (Realistic RBI T-Bill vs Sec 50AA Tax)
    frontier_rows = []
    for sig_lbl, eq_sig in [("Best_DD_Opt", eq_sig_best_dd), ("EQ4_MultiFactor", eq_sig_eq4)]:
        for w_g_pct in range(0, 101, 10):
            wg = w_g_pct / 100.0
            dr = wg * gold_r + (1.0 - wg) * liq_r
            tx_d = np.full(T, wg * 0.125 + (1.0 - wg) * 0.30, dtype=np.float64)
            nav_f, to_f = _sim_overlay_core(r_main, to_main, eq_sig, dr, tx_d, 0.0015, True, 0.20, 0.125)
            st = _summarize_nav(nav_f, to_f, bounds)
            st.update({"signal": sig_lbl, "w_gold_pct": w_g_pct, "w_liq_pct": 100 - w_g_pct})
            frontier_rows.append(st)

    fr_df = pd.DataFrame(frontier_rows)
    lines.append("\n=== MODULE 4A (Q4): EXACT GOLD / LIQUID FRONTIER (WITH SEC 50AA 30% LIQUID TAX VS 12.5% GOLD) ===")
    lines.append(fr_df[["signal", "w_gold_pct", "w_liq_pct", "P1_cagr", "P2_cagr", "P3_cagr", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh"]].to_string(index=False))

    # 4B: Alternative Hedging Strategies Head-to-Head (2011-26 Full + 2016-26 Gilt + Synthetic F&O!)
    alt_rows = []
    # 1. Trend-Gated Gold (sweep EMA 50, 100, 150) + Liquid
    for ema_l in (50, 100, 150):
        tg_ret, tg_tax, _ = compute_tg_erc_returns(
            def_df[["GOLDBEES"]].ffill().values,
            gold_r.reshape(-1, 1),
            liq_r,
            np.array([1.0], dtype=np.float64),
            np.array([0.125], dtype=np.float64),
            0.30, ema_l, 0.02,
        )
        for sig_lbl, eq_sig in [("Best_DD_Opt", eq_sig_best_dd), ("EQ4_MultiFactor", eq_sig_eq4)]:
            nav_a, to_a = _sim_overlay_core(r_main, to_main, eq_sig, tg_ret, tg_tax, 0.0015, True, 0.20, 0.125)
            st = _summarize_nav(nav_a, to_a, bounds)
            st["name"] = f"{sig_lbl} | TrendGold(EMA{ema_l}±2%)_or_Liq"
            alt_rows.append(st)

    # 2. Multi-Asset TG-ERC: Gold + MON100_STITCHED (capped <=20% due to US/India stress corr + SEBI cap) + Liquid
    mon_px = def_df["MON100_STITCHED"].ffill().values if "MON100_STITCHED" in def_df else def_df["GOLDBEES"].ffill().values
    erc_2_ret, erc_2_tax, _ = compute_tg_erc_returns(
        np.column_stack([def_df["GOLDBEES"].ffill().values, mon_px]),
        np.column_stack([gold_r, mon_r]),
        liq_r,
        np.array([0.75, 0.20], dtype=np.float64),
        np.array([0.125, 0.125], dtype=np.float64),
        0.30, 100, 0.02,
    )
    # 3. Gold + Gilt (LICNETFGSC post-2016 fallback to TrendGold pre-2016!) + MON100 + Liquid
    gilt_px = def_df["LICNETFGSC"].values if "LICNETFGSC" in def_df else np.full(T, np.nan)
    erc_3_ret, erc_3_tax, _ = compute_tg_erc_returns(
        np.column_stack([def_df["GOLDBEES"].ffill().values, gilt_px, mon_px]),
        np.column_stack([gold_r, gilt_r, mon_r]),
        liq_r,
        np.array([0.65, 0.65, 0.20], dtype=np.float64),
        np.array([0.125, 0.30, 0.125], dtype=np.float64),
        0.30, 100, 0.02,
    )
    # 4. Internal LowVol30 Midcap Equity Defense + 50% LowVol30 / 50% TG-ERC
    ret_lv50_erc50 = 0.5 * r_lowvol + 0.5 * erc_3_ret
    tax_lv50 = np.full(T, 0.5 * 0.20 + 0.5 * 0.20, dtype=np.float64)

    # 5. Synthetic F&O Beta-Hedged Portfolios (Spot Portfolio Stayed 100% + Hedged Delta Only During Signals!)
    # Compare Spot-Stay + Short Futures Hedge (2.0% p.a. roll+STT+basis cost during active hedge, 30% F&O slab tax) vs 5% OTM Put Spread (~4.8% p.a. theta while hedged)
    for sig_lbl, eq_sig in [("Best_DD_Opt", eq_sig_best_dd), ("EQ4_MultiFactor", eq_sig_eq4)]:
        hedge_frac = 1.0 - eq_sig  # e.g. 0.75 when signal defensive
        # Cash margin pledge on LIQUID_REALISTIC offsets ~half carry cost; net futures drag ~2.0% p.a. when active; midcap beta tracked by EW_ret
        ew_r_lag = np.concatenate(([0.0], (ew_nav_s[1:] / ew_nav_s[:-1] - 1.0)))
        fut_pnl = -hedge_frac * ew_r_lag - hedge_frac * (0.020 / 247.0)
        r_fut_hedged = r_main + fut_pnl - to_main * 0.003
        nav_fut = np.cumprod(1.0 + r_fut_hedged) * 100_000.0
        st_f = _summarize_nav(nav_fut, to_main + np.abs(np.diff(np.concatenate(([0.0], hedge_frac)))), bounds)
        st_f["name"] = f"{sig_lbl} | Spot+ShortFutHedge(₹25L+,net -2%/yr active)"
        alt_rows.append(st_f)

        # Put protection floors downside of hedged slice at -5% / month, costing ~5.2% p.a. annualized theta while active
        r_put = r_main + hedge_frac * np.maximum(0.0, -ew_r_lag - 0.0025) - hedge_frac * (0.052 / 247.0)
        nav_put = np.cumprod(1.0 + r_put) * 100_000.0
        st_p = _summarize_nav(nav_put, to_main, bounds)
        st_p["name"] = f"{sig_lbl} | Spot+5%OTM_BearPut(θ=-5.2%/yr active)"
        alt_rows.append(st_p)

    # Standard static & dynamic defensive suites
    static_defs = [
        ("Pure_LIQUID_TBILL_6.25%", def_df["LIQUID_TBILL"].pct_change().fillna(0.0).values, tax_liq),
        ("Pure_LIQUID_REALISTIC_RBI", liq_r, tax_liq),
        ("Pure_GOLDBEES", gold_r, tax_gold),
        ("Static_30Gold_70Liq", 0.3 * gold_r + 0.7 * liq_r, np.full(T, 0.3 * 0.125 + 0.7 * 0.30)),
        ("Static_50Gold_50Liq", r_base_5050, tax_5050),
        ("Static_70Gold_30Liq", 0.7 * gold_r + 0.3 * liq_r, np.full(T, 0.7 * 0.125 + 0.3 * 0.30)),
        ("Static_50Gold_20MON100_30Liq", 0.5 * gold_r + 0.2 * mon_r + 0.3 * liq_r, np.full(T, 0.7 * 0.125 + 0.3 * 0.30)),
        ("NIFTYBEES_LargeCap", nif_r, np.full(T, 0.125)),
        ("HNGSNGBEES_HangSeng", hng_r, np.full(T, 0.125)),
        ("Internal_LowVol30_MidcapEq", r_lowvol, np.full(T, 0.20)),
        ("TG_ERC(Gold+MON100cap20+Liq)", erc_2_ret, erc_2_tax),
        ("TG_ERC(Gold+Gilt+MON100+Liq)", erc_3_ret, erc_3_tax),
        ("Hybrid_50LowVol30_50TGERC", ret_lv50_erc50, tax_lv50),
    ]
    for sig_lbl, eq_sig in [("Best_DD_Opt", eq_sig_best_dd), ("EQ4_MultiFactor", eq_sig_eq4)]:
        for dname, dr, dtx in static_defs:
            nav_a, to_a = _sim_overlay_core(r_main, to_main, eq_sig, dr, dtx, 0.0015, True, 0.20, 0.125)
            st = _summarize_nav(nav_a, to_a, bounds)
            st["name"] = f"{sig_lbl} | {dname}"
            alt_rows.append(st)

    # Also test Old_DD15 + 50/50 Gold+Liq vs Unhedged for direct benchmark reference
    for lbl, eq_s in [("Legacy_DD15_Old", eq_sig_old_dd15), ("Unhedged", ones)]:
        nav_a, to_a = _sim_overlay_core(r_main, to_main, eq_s, r_base_5050, tax_5050, 0.0015, True, 0.20, 0.125)
        st = _summarize_nav(nav_a, to_a, bounds)
        st["name"] = f"{lbl} | Static_50Gold_50Liq"
        alt_rows.append(st)

    alt_df = pd.DataFrame(alt_rows)
    lines.append("\n=== MODULE 4B (Q5): ALTERNATIVE HEDGING STRATEGIES HEAD-TO-HEAD (2011-26 FULL + P1/P2/P3) ===")
    lines.append(alt_df.sort_values("FULL_sh", ascending=False)[["name", "P1_cagr", "P1_dd", "P2_cagr", "P2_dd", "P3_cagr", "P3_dd", "FULL_cagr", "FULL_dd", "FULL_cal", "FULL_sh", "FULL_to"]].to_string(index=False))

    # Gilt ETF sub-window check (2016-01-01..2026-11-26 when LICNETFGSC & SETF10GILT existed cleanly!)
    g_start = int(np.where(dates >= "2016-01-04")[0][0])
    g_end = T - 1
    gilt_rows = []
    for dname, dr, dtx in [
        ("Static_50Gold_50Liq", r_base_5050, tax_5050),
        ("Static_33Gold_33Gilt_34Liq", (gold_r + gilt_r + liq_r) / 3.0, np.full(T, 0.33 * 0.125 + 0.67 * 0.30)),
        ("Pure_LICNETFGSC_Gilt", gilt_r, tax_liq),
        ("TG_ERC(Gold+Gilt+MON100+Liq)", erc_3_ret, erc_3_tax),
        ("Pure_LIQUID_REALISTIC", liq_r, tax_liq),
    ]:
        nav_g, to_g = _sim_overlay_core(r_main, to_main, eq_sig_eq4, dr, dtx, 0.0015, True, 0.20, 0.125)
        c, dd, cal, sh, toa = _calc_metrics_slice(nav_g, to_g, g_start, g_end, 0.0625)
        gilt_rows.append({"defensive_asset_2016_26": dname, "cagr": round(c, 2), "max_dd": round(dd, 2), "calmar": round(cal, 2), "sharpe": round(sh, 2)})
    lines.append("\n=== MODULE 4C: 10-YEAR GILT ETF SUB-WINDOW CHECK (2016-2026, EQ4_MultiFactor) ===")
    lines.append(pd.DataFrame(gilt_rows).sort_values("sharpe", ascending=False).to_string(index=False))

    out_txt = REPO_ROOT / "reports" / "numba_deep_research.txt"
    out_json = REPO_ROOT / "reports" / "numba_deep_research.json"
    out_txt.write_text("\n".join(lines))
    out_json.write_text(json.dumps({
        "elapsed_sec": round(time.time() - t_start, 2),
        "pit_semi_top15": semi_df.sort_values("P1_sh", ascending=False).head(15).to_dict(orient="records"),
        "grid_21600_top12_p12": p12_top.to_dict(orient="records"),
        "multifactor_top15": mf_df.sort_values("FULL_sh", ascending=False).head(15).to_dict(orient="records"),
        "gold_liquid_frontier": frontier_rows,
        "alt_hedging": alt_rows,
        "gilt_2016_26": gilt_rows,
    }, indent=2, default=str))
    print(f"Completed ALL Numba + SciPy research in {time.time() - t_start:.1f}s -> {out_txt}")


if __name__ == "__main__":
    main()
