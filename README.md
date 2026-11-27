# DIY Nifty Midcap 150 Momentum Index

Autonomous research engine, backtester, official **17-year NSE `PR` + `Bc` Archive** & **NSE MCP** data pipeline, and static **GitHub Pages** dashboard for running an independent **Nifty Midcap 150 Momentum** stock portfolio starting at `₹1,00,000`.

Live Dashboard: **[harjaschadha.github.io/midcap-momentum-index](https://harjaschadha.github.io/midcap-momentum-index/)** · Full Multi-Period & Zero-Survivorship Report: [reports/FINAL_STRATEGY_REPORT.md](reports/FINAL_STRATEGY_REPORT.md)

---

## Upgraded Strategy (`R2` — Validated Across 16 Years & 3 Blind Regimes in `config/strategy.json`)

After auditing **4,171 NSE trading days × 4,047 equities (`2010–2026`, `1,305+` corporate actions)** with a **Zero-Survivorship Point-in-Time** universe (`787` liquid midcaps ranked 81–230) across 3 blind non-overlapping regimes (`2011–16 Train → 2016–21 Val #1 → 2021–26 Val #2`) and an independent `2018–22 Train → 2022–26 OOS` check, `R2` replaces legacy `F2` (`Raw 9M + DD>12%`) with **+2.90% p.a. higher 16-year CAGR** and **-5.36% shallower Max Drawdown**:

| Component | Specification | Why (Verified Across 3 Blind Regimes & 16Y PIT) |
|---|---|---|
| **Basket & Split** | **`50%` Top 20 + `50%` Top 30** (Equal-Weight inside each sleeve → `4.17%` Ranks 1–20, `1.67%` Ranks 21–30) | Preserves conviction in Top-20 leaders while spreading tail exposure across 30 stocks (`4.17%` max single-stock cap) |
| **Momentum Factor** | **`risk_9m` (`189`-day return / `ann_vol`, 1M skip)** *(shares 29/30 stocks with Raw 9M today)* | Avoids high-volatility blowup traps; leads every basket over 16 zero-survivorship years (`19.17%` vs `16.27%` Raw F2) and ranks #1 blindly in `2018–22` Train |
| **Rebalancing** | **Every `6 Months` (`126` trading days)** (`12M` optional above `₹5L`) | Captures factor rotation (`198.9%` TO vs `260.3%` in F2) while letting winners cross 365 days into 12.5% LTCG |
| **Market-Fall Signal** | **Midcap 150 EW Index `DD > 15%`** from 252d peak (Re-enter when `> 50DMA` & `DD > -9.0%`) | Cuts bull-pullback whipsaws vs `-12%` (`2012`, `2015`, `2024–25`) while hedging severe breaks (`2011`, `2018–19`, `2020`, `2022`) |
| **Defensive Shift & Asset** | **Shift `75%` (`weakest-first`) → `50% GOLDBEES + 50% LIQUIDCASE`** | Cuts STCG tax **17%–30%** vs pro-rata, shields against `GOLDBEES`' `-31.7%` 2026 slump and `SILVERBEES`' `-51.8%` crash |
| **₹1L Execution Rule** | **`Sequential-SlotCap` + `LIQUIDCASE` Sweep** | Eliminates zero-share cash drag on high-ticket leaders (`APARINDS`, `OFSS`); deploys **`30/30` stocks (`86.86%`) + `112 LIQUIDCASE` (`99.97%` utilized)** |

> **Verified Performance Summary (`15bps` 1-way cost + `20%` STCG / `12.5%` LTCG with `₹1.25L` exemption):**
> - **16-Year Zero-Survivorship PIT (`2011–2026`, 787 Stocks):**
>   - **Passive PIT Midcap 150 EW Proxy:** `11.61%` CAGR · `-51.84%` Max DD · `0.27` Sharpe
>   - **Legacy `F2` (`Raw 9M + DD12`):** `16.27%` (`15.62%` `₹1L`) · `-33.16%` Max DD · `0.57` Sharpe (`₹9.55L` ending NAV on `₹1L`)
>   - **Upgraded `R2` (`Risk 9M + DD15`):** **`19.17%` (`18.56%` `₹1L`)** · **`-27.80%` Max DD** · **`0.77` Sharpe** (**`₹14.11L` ending NAV on `₹1L`** — `+47.8%` more wealth than `F2`!)
>   - **Concentrated `R7` (`NormZ Risk 9M + DD15`):** **`20.46%` (`20.23%` `₹1L`)** · **`-28.74%` Max DD** · **`0.82` Sharpe** (**`₹17.53L` ending NAV on `₹1L`** — `>3x` passive EW!)
> - **Clean Warm Live Universe (`2018-01-02 → 2026-11-26`):** `R2` delivers **`27.11%`** CAGR (`26.71%` `₹1L`), `-26.25%` Max DD, `1.14` Sharpe, `1.03` Calmar (`R7`: `28.15%`, `1.17` Sharpe; notice `+8.54%` p.a. of the current-constituent headline gap vs PIT is pure constituent-selection hindsight).

---

## Architecture & Repository Layout

- `src/nse_archive_loader.py`: Official 17-year (`2010–2026`) NSE `PR` Bhavcopy & `Bc` corporate action parser (`SPLIT`, `BONUS`, `SCHEME OF ARRANGEMENT`, combined ratios, postponed-event suppression) + MCP split supplement.
- `src/prowess_loader.py`: Standby CMIE Prowessdx pipe-delimited ingestion + point-in-time market-cap midcap mask builder.
- `src/nse_mcp.py`: Client for official NSE India MCP servers (`BHAVCOPY_MCP_URL` & `CMMKT_MCP_URL`).
- `src/data_loader.py`: Fetches Nifty Midcap 150 universe, Yahoo 10y adjusted/raw/volume history, repairs transient glitches, backfills missing REIT quotes (`EMBASSY`).
- `src/momentum.py`: `raw`, `risk_adj`, `blend`, PDF `norm_z`, rank buffers, and `select_affordable_blend` (`Sequential-SlotCap`).
- `src/regime.py`: Zero-survivorship EW Midcap 150 index builder (supporting point-in-time masks), defensive ETF series (`GOLD_50_LIQUID_50`, `LIQUID_TBILL`, etc.), drawdown/MA/breadth/VIX signals.
- `src/backtester.py`: Zero look-ahead `T+1` engine with FIFO STCG/LTCG tax lots, ₹1.25L annual LTCG exemption, weakest-first trimming, integer-share `raw_close` rounding, and `LIQUIDCASE` sweep.
- `scripts/run_17yr_train_val_oos.py` & `scripts/run_train_oos.py`: Strict blind Train vs Validation & Overfitting check harnesses (`2011–16 → 2016–21 → 2021–26` & `2018–22 → 2022–26`).
- `scripts/update_live.py`: Automated EOD updater merging live NSE MCP quotes + auto corporate-action checks into strict `data/live_state.json`.
- `index.html`: Zero-backend dark interactive terminal with live **`R2`/`R7`/`F2`/`F1`/`F4`** switching, capital sizer, crash simulator, `330–364d` LTCG tax-lot diff, Chart.js equity/drawdown charts, 16-year PIT & clean research tables, and Top-60 scorer.
- `.github/workflows/daily_update.yml`: Weekday post-close CI job running `unittest` + `update_live.py` + committing updated `data/live_state.json`.

---

## Local Verification

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 scripts/update_live.py
python3 scripts/verify_dashboard_ui.py
```
