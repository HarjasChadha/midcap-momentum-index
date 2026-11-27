# DIY Nifty Midcap 150 Momentum Index

Autonomous research engine, backtester, official **NSE MCP** data pipeline, and static **GitHub Pages** dashboard for running an independent **Nifty Midcap 150 Momentum** stock portfolio starting at `₹1,00,000`.

Live Dashboard: **[harjaschadha.github.io/midcap-momentum-index](https://harjaschadha.github.io/midcap-momentum-index/)** · Detailed Research Findings: [reports/FINAL_STRATEGY_REPORT.md](reports/FINAL_STRATEGY_REPORT.md)

---

## Winning Strategy (`F2` — Locked in `config/strategy.json`)

| Component | Specification | Why |
|---|---|---|
| **Basket & Split** | **`50%` Raw Top 20 + `50%` Raw Top 30** (Equal-Weight inside each sleeve → `4.17%` Ranks 1–20, `1.67%` Ranks 21–30) | Preserves conviction in Top-20 leaders while spreading tail exposure across 30 stocks (`4.17%` max single-stock cap) |
| **Momentum Factor** | **`raw_9m` (`189`-day return, skipping latest `21` trading days)** | Beats 12M cleanly from warm start dates, reacts faster than 12M without monthly STCG churn |
| **Rebalancing** | **Every `6 Months` (`126` trading days)** (`12M` optional above `₹5L`) | Balances factor freshness (`260%` TO) against India's 365-day LTCG vs 20% STCG spread |
| **Market-Fall Signal** | **Midcap 150 EW Index `DD > 12%`** from 252d peak (Re-enter when `> 50DMA` & `DD > -7.2%`) | Fires only ~9 times in 9 years; cuts Max DD from **`-37.85%` → `-24.57%`** (`Calmar 0.75 → 1.05`) |
| **Defensive Shift & Asset** | **Shift `75%` (`weakest-first`) → `50% GOLDBEES + 50% LIQUIDCASE`** | Cuts STCG tax **17%–30%** vs pro-rata, protects against `GOLDBEES`' `-31.7%` 2026 drawdown, avoids `-51.8%` silver crashes |
| **₹1L Execution Rule** | **`Sequential-SlotCap` + `LIQUIDCASE` Sweep** | Eliminates zero-share cash drag on expensive stocks (`APARINDS ₹14,200`, `OFSS ₹9,291`), lifting `₹1L` equity deployed from `66.6% → 87.6%` (`99.96%` w/ `LIQUIDCASE` sweep) and `+111 bps` clean backtest CAGR |

> **Clean `2018-01-02 → 2026-11-26` Performance (15bps 1-way cost + 20% STCG / 12.5% LTCG):**
> - **`F2` Continuous:** `25.78%` CAGR · `-24.57%` Max DD · `1.05` Calmar · `1.06` Sharpe
> - **`F2` Discrete `₹1L` (`SlotCap + LiqSweep`):** **`25.69%` CAGR** · `-25.82%` Max DD · `1.09` Sharpe · `81.9%` avg equity deployed
> - **Runner-Up `F7` (`NormZ`):** `28.34%` CAGR (`28.08%` ₹1L discrete) · `-24.25%` Max DD · `1.17` Calmar *(higher single-stock concentration)*
> - *Expect `6%–10%` lower live CAGR at larger capital once historical survivorship (`~8%` gap vs `MIDCAPIETF`) and ₹1.25L LTCG exemption exhaustion are accounted for.*

---

## Architecture & Repository Layout

- `src/nse_mcp.py`: Client for official NSE India MCP servers (`BHAVCOPY_MCP_URL` & `CMMKT_MCP_URL`) supporting `get_market_mood`, `get_bulk_quote`, `get_index_history`, `get_corporate_actions`, and intraday `cm_get_stock_quote`.
- `src/data_loader.py`: Fetches Nifty Midcap 150 universe, Yahoo 10y adjusted/raw/volume history, repairs ex-split data glitches, and backfills missing REIT quotes (`EMBASSY`) via NSE Bhavcopy.
- `src/momentum.py`: `raw`, `risk_adj`, `blend`, PDF `norm_z`, rank buffers, and `select_affordable_blend` (`Sequential-SlotCap`).
- `src/regime.py`: EW Midcap 150 index builder, defensive ETF series (`GOLD_50_LIQUID_50`, `GOLD_OR_LIQUID`, `LIQUID_TBILL`, etc.), drawdown/MA/breadth/VIX signals.
- `src/backtester.py`: Zero look-ahead `T+1` engine with FIFO STCG/LTCG tax lots, ₹1.25L annual LTCG exemption, weakest-first trimming, integer-share `raw_close` rounding, and `LIQUIDCASE` idle-cash sweep.
- `scripts/update_live.py`: Automated EOD updater merging live NSE MCP quotes + auto corporate-action checks into strict `data/live_state.json`.
- `index.html`: Zero-backend dark interactive terminal (capital sizer, crash simulator, `330–364d` LTCG tax-lot diff, Chart.js equity/drawdown charts, full research matrix, live Top 60 scorer).
- `.github/workflows/daily_update.yml`: Weekday post-close CI job running `unittest` + `update_live.py` + committing updated `data/live_state.json`.

---

## Local Verification

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 scripts/update_live.py
python3 scripts/verify_dashboard_ui.py
```
