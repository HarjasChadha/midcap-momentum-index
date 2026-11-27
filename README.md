# DIY Nifty Midcap 150 Momentum Index (`V3 · R3 Multi-Factor + TrendGold`)

Autonomous research engine, Numba/SciPy walk-forward optimizer, **16-year official NSE `PR` + `Bc` Archive** & **NSE MCP** live pipeline, and static **GitHub Pages** execution terminal for running an independent **Nifty Midcap 150 Momentum** portfolio starting at `₹1,00,000`.

Live Dashboard: **[harjaschadha.github.io/midcap-momentum-index](https://harjaschadha.github.io/midcap-momentum-index/)** · **Complete Equations, Assumptions & Data Provenance:** **[MATHEMATICAL_SPECIFICATION.md](MATHEMATICAL_SPECIFICATION.md)** · Strategy Report: [reports/FINAL_STRATEGY_REPORT.md](reports/FINAL_STRATEGY_REPORT.md)

---

## Primary Strategy (`R3` — Causal 4-Block Multi-Factor + TrendGold EMA150)

Replacing all single-threshold / static `50-50` heuristics after a **Numba (`21,600+` grid) + SciPy Walk-Forward optimization** across **4,171 NSE trading days × 4,047 equities (`2010–2026`, `419` NSE ETFs purged, lagged PIT mask)** on `772` liquid zero-survivorship midcaps (`2011–16 Train → 2016–21 Val #1 → 2021–26 Val #2`):

| Component | `R3` Specification | Why (`16Y Clean PIT` & `2018–26` Empirical Proof) |
|---|---|---|
| **Basket & Split** | **`50%` Top 20 + `50%` Top 30** (`30` unique stocks: `4.17%` Ranks 1–20, `1.67%` Ranks 21–30) | Ranks 1–20 appear in both sleeves (`2.50% + 1.67%`), capping single-stock exposure at `4.17%` while tilting into Top-20 leaders |
| **Momentum Factor** | **`risk_9m`** (`189d` return / `ann_vol`, 1M skip) · *(Runner-up **`S3` Sortino 9M** also live)* | Separates steady compounders from volatile spikes (`18.16%` 16Y clean PIT vs `12.04%` legacy `F2`) |
| **Regime Composite** | **4 Orthogonal Blocks (`MF4`)**: `[0.44 Trend/Rel, 0.10 Vol/VIX, 0.34 Breadth, 0.12 DM/Gold]` | Single `-15%` DD triggers fail OOS (`Spearman ρ = -0.091`); SciPy weights on `P1` generalize across all 3 periods (`13.76% → 25.69% → 21.26%`) |
| **Hysteresis & Re-Entry** | **Asymmetric EWMA (`8d/22d`) + 3-State Schmitt** (`<-0.25 → 25% Eq`, `<-0.10 → 60% Eq`, `>+0.18 → 100% Eq`) + **Zweig Thrust** | Eliminates `501%` no-hysteresis churn while catching V-bottom inflections cleanly |
| **Dynamic Defensive ETF** | **`TrendGold150` (`GOLDBEES > 1.02×EMA150`, else `LIQUIDCASE`)** | Avoids static Gold's `-31.7%` 2026 bear slump AND `30%` Sec 50AA debt tax drag when Gold is trending up; active gate today: **`LIQUIDCASE`** |
| **Retail `₹1L` Rule** | **`Sequential-SlotCap` + `LIQUIDCASE` Sweep** (`6M` rebal, `330–364d` LTCG shield) | Deploys `30/30` affordable stocks + parks leftover integer-share rupees (`32.02%` CAGR / `-20.68%` DD / `1.47` Sharpe on `2018–26`) |

> **Verified Exact Python FIFO Performance (`15bps` 1-Way Cost + `20%` STCG / `12.5%` LTCG with `₹1.25L` Annual Exemption):**
> - **Clean ETF-Free 16-Year Zero-Survivorship PIT (`2011–2026`, `772` Equities):**
>   - **Passive Equal-Weight Midcap 150 Proxy:** `9.77%` CAGR · `-54.08%` Max DD · `0.18` Sharpe
>   - **Legacy `F2` (`Raw 9M + DD12 + 50/50`):** `12.04%` CAGR · `-27.58%` Max DD · `0.39` Sharpe
>   - **`R2` (`Risk 9M + DD15 + 50/50`):** `15.79%` CAGR · `-28.44%` Max DD · `0.67` Sharpe
>   - **Runner-up `S3` (`Sortino 9M + MF4 + TrendGold150`):** **`17.21%`** CAGR · `-28.69%` Max DD · `0.72` Sharpe
>   - **Recommended `R3` (`Risk 9M + MF4 + TrendGold150`):** **`18.16%`** CAGR · **`-26.93%` Max DD** · **`0.78` Sharpe** · **`0.67` Calmar**
> - **Live Universe Panel (`2018-01-02 → 2026-11-27`):** `R3` delivers **`31.59%`** CAGR (`32.02%` `₹1L` SlotCap), **`-21.10%` Max DD** (`-20.68%` `₹1L`), **`1.43` Sharpe** (`1.47` `₹1L`), **`1.50` Calmar** (`S3`: **`32.06%`**, `-21.02%` DD, **`1.46` Sharpe**).

---

## Repository Architecture

- `src/nse_archive_loader.py`: Official 17-year (`2010–2026`) NSE `PR` + `Bc` parser (`1,328` corporate actions, duplicate/truncated-string guards) + MCP split supplement.
- `src/momentum.py`: `raw`, `risk_adj`, **`sortino`**, PDF `norm_z`, rank buffers, and `select_affordable_blend` (`Sequential-SlotCap`).
- `src/regime.py`: Lagged zero-leak EW index builder (`mask.shift(1)`), causal **`compute_multifactor_state` (`MF4`)**, **`get_trend_gold_status` (`TrendGold150`)**, **`TREND_GOLD150_OR_LIQ`**, **`LIQUID_REALISTIC`**, and `MON100_STITCHED`.
- `src/backtester.py`: Causal `T+1` FIFO engine with Indian STCG/LTCG tax lots, `ffill(limit=5)` delisting guard, weakest-first trimming, integer shares, and cash sweep.
- `scripts/numba_research_engine.py`, `scripts/run_numba_sweeps.py`, `scripts/run_final_verification_suite.py`: Offline Numba (`21,600+` grid) + SciPy walk-forward research scripts *(zero production CI dependency on `numba`/`scipy`)*.
- `notebooks/`: Google Colab `.ipynb` workspace — dedicated subfolder for every standalone research/pipeline notebook (`01`..`09`) **plus** an all-in-one master notebook (`00_master_colab_all_in_one/Midcap_Momentum_Complete_Colab.ipynb`). See [notebooks/README.md](notebooks/README.md) for 1-click Colab launch links.
- `scripts/update_live.py` & `.github/workflows/daily_update.yml`: Automated EOD NSE MCP updater regenerating strict `data/live_state.json`.
- `index.html`: Interactive dashboard featuring live **4-Block Regime Diagnostics**, **`R3` / `S3` / `R2` / `R7` / `F2` / `F4`** preset switcher, `TrendGold` crash simulation, `330–364d` LTCG diff tool, Chart.js comparison charts, 16-year FIFO & Crash/F&O matrices, and Top-60 Scorer.

---

## Google Colab Notebooks (Both Separate Folders & Combined Master)

[![Open Master Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarjasChadha/midcap-momentum-index/blob/main/notebooks/00_master_colab_all_in_one/Midcap_Momentum_Complete_Colab.ipynb)

Full 1-click Colab directory for every individual folder (`01`..`09`): **[notebooks/README.md](notebooks/README.md)**

---

## Local Verification

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 scripts/update_live.py
python3 scripts/verify_dashboard_ui.py
```
