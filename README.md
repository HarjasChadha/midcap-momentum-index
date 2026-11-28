# DIY Nifty Midcap 150 Momentum Index (`V3 · R3 Multi-Factor + TrendGold`)

Autonomous research engine, Numba/SciPy walk-forward optimizer, **16-year official NSE `PR` + `Bc` Archive** & **NSE MCP** live pipeline, and modular static **GitHub Pages** execution terminal for running an independent **Nifty Midcap 150 Momentum** portfolio starting at `₹1,00,000`.

> **Disclaimer:** *Personal Quantitative Research & Paper-Trading Sandbox — Not Investment Advice / Not SEBI Registered.*

Live Dashboard: **[harjaschadha.github.io/midcap-momentum-index](https://harjaschadha.github.io/midcap-momentum-index/)** · **Complete Equations & Provenance:** **[MATHEMATICAL_SPECIFICATION.md](MATHEMATICAL_SPECIFICATION.md)** · **PDF Spec:** **[docs/Momentum_Strategy_Mathematical_Specification.pdf](docs/Momentum_Strategy_Mathematical_Specification.pdf)** · Strategy Report: **[reports/FINAL_STRATEGY_REPORT.md](reports/FINAL_STRATEGY_REPORT.md)**

---

## Primary Strategy (`R3` — Causal 4-Block Multi-Factor + TrendGold EMA150)

Audited across **4,171 NSE trading days × 4,047 equities (`2010–2026`, `419` NSE ETFs purged, lagged PIT mask)** on `772` liquid zero-survivorship midcaps (`P1 2011–16 Train → P2 2016–21 Val #1 → P3 2021–26 Post-Selection / TrendGold150`) with **Sec 50AA `30%` debt/Gold taxes enabled by default (`tax_defensive=True`)** and **regime-conditional slippage (`15 bps` scheduled / `35 bps` risk-off)**:

| Component | `R3` Specification | Verified 16Y PIT & Live Panel Evidence |
|---|---|---|
| **Basket & Split** | **`50%` Top 20 + `50%` Top 30** (`30` unique stocks: `4.17%` Ranks 1–20, `1.67%` Ranks 21–30) | Tilts 2.5× into Top-20 leaders while capping single-stock weight at `4.17%` |
| **Momentum Factor** | **`risk_9m`** (`189d` return / `ann_vol`, 1M skip, `252d` ann, `[-0.40, +0.60]` clip, `0.08` floor, `>=95%` un-padded IPO coverage) | Shared verbatim between research (`scripts/numba_research_engine.py`) and production (`src/momentum.py`) (`0.0` drift); *runner-up `S3` Sortino 9M also live* |
| **Regime Composite** | **4 Orthogonal Blocks (`MF4`)**: `[0.44 Trend/Rel, 0.10 Vol/VIX, 0.34 Breadth, 0.12 DM/Gold]` | Single `-15%` DD triggers fail OOS (`Spearman ρ = -0.092`); `MF4` beats Zero-Fit `EQ4` (`15.61%` vs `13.44%`) and adds `+2.66%` p.a. over Official Midcap 100 |
| **Hysteresis & Re-Entry** | **Asymmetric EWMA (`8d/22d`) + 3-State Schmitt** (`<-0.25 → 25% Eq`, `<-0.10 → 60% Eq`, `>+0.18 → 100% Eq`) + **Zweig Thrust** | Staggered `2`-tranche 6M rebalancing cuts 6-phase CAGR dispersion from `4.60%` down to **`1.96%`** |
| **Dynamic Defensive ETF** | **`TrendGold150` (`GOLDBEES > 1.02×EMA150`, else `LIQUIDCASE`)** | Switches defensive reserves out of gold bear slumps (`2026_GoldSlump`) into `LIQUIDCASE`; active gate (`2026-11-27`): **`LIQUIDCASE`** (`GOLDBEES -12.35%` vs `EMA150`) |
| **Retail `₹1L` Rule** | **`Sequential-SlotCap` + `LIQUIDCASE` Sweep** (`6M` rebal, `330–364d` LTCG shield) | Over 16Y PIT, SlotCap incurs `-1.53%` drag (`14.08%` vs `15.61%`); `2018–26` `+0.09%` (`27.62%` vs `27.53%`) is small-sample luck |

> **Verified Exact Python FIFO Performance (`tax_defensive=True`, `15 bps` scheduled / `35 bps` risk-off + Indian FIFO taxes):**
> - **Clean ETF-Free 16-Year Zero-Survivorship PIT (`2011–2026`, `772` Equities):**
>   - **Bench: Clean PIT Equal-Weight Proxy:** `9.76%` CAGR · `-54.08%` Max DD · `0.18` Sharpe
>   - **Bench: Official NSE Nifty Midcap 100 (`16Y`):** `12.95%` CAGR · `-45.83%` Max DD · `0.31` Sharpe
>   - **Zero-Fit Ablation (`EQ4 [0.25]*4 + 50/50`):** `13.44%` CAGR · `-28.84%` Max DD · `0.52` Sharpe
>   - **`R2` (`Risk 9M + DD15 + 50/50`):** `15.17%` CAGR · `-28.72%` Max DD · `0.61` Sharpe
>   - **`S3` (`Sortino 9M + MF4 + TrendGold150`):** `14.74%` CAGR · `-30.03%` Max DD · `0.56` Sharpe
>   - **Recommended `R3` (`Risk 9M + MF4 + TrendGold150`):** **`15.61%`** CAGR (`17.73%` pre-Sec-50AA) · **`-28.78%` Max DD** · **`0.61` Sharpe** · **`0.54` Calmar** (`30 bps` 2x stress: `13.95%` / `0.50` Sh; `-15%` delist stress: `-34.23%` DD)
> - **Current-Constituent Live Panel (`2018-01-02 → 2026-11-27`, Survivorship-Biased Disclosure):** `R3` = **`27.53%`** CAGR (`30.83%` pre-Sec-50AA; `27.62%` `₹1L` SlotCap), `-26.00%` Max DD, `1.19` Sharpe (`S3`: `27.99%`; `NIFTY_MIDCAP_150` `2019–26`: `17.76%`; Current Universe EW: `21.60%`).

---

## Repository Architecture

- `src/nse_archive_loader.py` & `src/prowess_loader.py`: Official 17-year (`2010–2026`) NSE `PR` + `Bc` parser plus drop-in CMIE Prowessdx market-cap PIT constitution loader (`data/cmie/`).
- `src/momentum.py`: Shared canonical scorer (`compute_score_matrix`, `compute_listing_days`, `select_affordable_blend`) imported identically by research and production.
- `src/regime.py`: Lagged zero-leak EW builder, causal `MF4`, `TrendGold150`, and unified `LIQUID_REALISTIC` defensive series.
- `src/backtester.py`: `T+1` FIFO engine with `tax_defensive=True`, `35 bps` risk-off slippage, `last_valid_px` delisting MTM, and staggered multi-tranche sleeves.
- `src/paper_bot.py`: Autonomous pure-Python paper-trading engine (`R3_1L_SlotCap`, `R3_10L_Standard`) updating `data/paper_ledger.json` & `data/forward_tape.json`.
- `scripts/`: `update_live.py`, `run_16yr_exact_verification.py`, `run_numba_sweeps.py`, `run_final_verification_suite.py`, `build_math_spec_pdf.py`, and `verify_dashboard_ui.py`.
- `index.html`, `assets/styles.css`, `assets/app.js`: Modular static GitHub Pages terminal with `?` metric tooltips, survivorship disclosure banner, `Listing Days` tags, Early-Inception Paper Bot state, and 1:1 `MMI_REPLAY`.

---

## Local Verification

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 scripts/update_live.py
python3 scripts/verify_dashboard_ui.py
```
