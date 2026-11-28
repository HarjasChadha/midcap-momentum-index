# DIY Nifty Midcap 150 Momentum Index — 16-Year Zero-Survivorship, Multi-Factor & Audit Specification (`V3 / R3`)

**Repository:** `HarjasChadha/midcap-momentum-index` · **Data Sources:** Official **NSE `PR` Archives (`2010-01-04` to `2026-11-26`, `4,171` trading days × `4,047` equities, `1,328` corporate actions, ALL `419` NSE ETFs excluded via `nse_etf_symbols.json`)** + **NSE MCP live quotes** + `2018-01-02 → 2026-11-27` live constituent panel.

---

## Executive Summary & P0/P1 Audit Reconciliation

All core backtesting, research, and production scoring assumptions have been standardized and synchronized under **default full Indian tax and realistic crash-impact accounting**:

1. **Sec 50AA Debt/Gold Switch Taxes Active by Default (`tax_defensive=True`):** All primary benchmarks and autonomous paper-bot defaults now realize **`30%` slab tax** on `LIQUIDCASE`/`LIQUID_REALISTIC` debt gains and **`12.5%` ($\ge 365\text{d}$) / `30%` ($< 365\text{d}$)** on `GOLDBEES` switches directly from cash; pre-defensive-tax numbers are retained strictly as an optional comparison row.
2. **Regime-Conditional Crash Slippage (`15 / 35 bps`):** Scheduled semi-annual rebalances deduct **`15 bps`** one-way; unscheduled risk-off liquidations (`DEFENSIVE` / `CAUTIOUS` trims) deduct **`35 bps`** one-way (`regime_cost_bps_one_way=35.0`), plus an explicit **`30 bps` / `60 bps` (2x Friction) stress test**.
3. **Last-Valid-Price Delisting MTM + Write-Down Stress:** Missing/halted prices beyond 5 days mark to **`last_valid_px`** (never purchase lot cost), accompanied by mandatory **`-15%` and `-25%` forced write-down stress runs** (`11` triggered write-downs across the 16Y archive).
4. **Canonical Unified Scoring Math (`src/momentum.py`):** Research (`scripts/numba_research_engine.py`) and live production import the identical `compute_score_matrix` (`252d` annualization, daily return clipping `[-0.40, +0.60]`, volatility floors `0.08` risk / `0.06` sortino, strict un-padded `>= 95%` IPO coverage -> **`0.0` max abs diff**, **`100%` NaN-mask match**).
5. **Rebalance Timing Dispersion (`Staggered 2-Tranche 6M`) & Window Disclosure:** `P3 (2021-07-01..2026-11-26)` is explicitly disclosed as **`Post-Selection / Partially In-Sample` for `TrendGold150`**. Staggering the 6-month rebalance into `2` equal tranches (`50%` every `3` months) cuts 6-phase start-date CAGR spread from **`4.60%` down to `1.96%`**.

---

### Exact Python FIFO-Lot Verification (`src/backtester.py`, Net Sec 50AA + `15/35 bps`)

| Strategy / Benchmark Configuration | 16Y Clean PIT (`2011–26`) | `P1 (11–16) → P2 (16–21) → P3 (21–26 Post-Sel)` | 16Y MaxDD / Sharpe | Live Panel (`2018–26` Biased) | Notes |
|---|---:|---|---:|---:|---|
| **`R3` Primary (`Risk9M · MF4 · TrendGold150`)** | **15.61%** | `11.43% → 23.31% → 16.78%` | **`-28.78%` / `0.61` Sh (`0.54` Cal)** | **27.53%** (`-26.00%`, `1.19` Sh) | Net of Sec 50AA `₹1.37L` + `35 bps` risk-off (`6-phase mean 15.01%`, `[12.75%–17.35%]`) |
| `R3 Pre-Def-Tax Comparison` (`Sec 50AA Off`) | `17.73%` | `13.15% → 25.14% → 20.48%` | `-28.45%` / `0.75` Sh | `30.83%` (`-22.91%`, `1.39` Sh) | Shows `-2.12%` 16Y / `-3.30%` 2018–26 tax drag of Sec 50AA switches |
| **`R3 Staggered 2-Tranche 6M`** (`50%` q3M) | `13.48%` | `9.82% → 19.63% → 18.19%` | **`-27.46%`** / `0.49` Sh | `26.28%` (**`-21.68%`**, `1.21` Cal) | Cuts 6-phase CAGR dispersion from `4.60%` down to **`1.96%`** (`14.44%` mean) |
| **`R3 2x Cost Stress`** (`30 bps` / `60 bps` cut) | `13.95%` | `9.87% → 21.61% → 14.88%` | `-31.14%` / `0.50` Sh | `25.75%` (`-29.55%`, `1.09` Sh) | Remains profitable above passive indices under doubled friction |
| `R3 Delist Stress (-15% / -25% write-down)` | `15.91% / 15.66%` | `11.03% → 23.08% → 16.70%` (`-15%`) | **`-34.23%`** / `0.57`–`0.56` Sh | `—` | Triggers `11` forced write-downs; widens Max DD by `-5.45%` |
| **`R3 ₹1L Discrete SlotCap`** | **`14.08%`** | `8.83% → 21.64% → 16.08%` | `-30.53%` / `0.51` Sh | `27.62%` (`-25.05%`, `1.22` Sh) | **`-1.53%` 16Y PIT drag** vs unconstrained `R3`; `2018–26` `+0.09%` is small-sample luck |
| **`S3` Sortino 9M · `MF4` · `TrendGold150`** | `14.74%` | `11.07% → 23.04% → 17.67%` | `-30.03%` / `0.56` Sh | **27.99%** (`-25.04%`, `1.22` Sh) | `S3 ₹1L SlotCap`: `12.97%` 16Y PIT / `27.86%` 2018–26 |
| **`R2` Simple Trigger (`DD>15% · 50/50 Gold/Liq`)** | `15.17%` | `8.24% → 28.92% → 18.59%` | `-28.72%` / `0.61` Sh | `25.60%` (`-26.39%`, `1.06` Sh) | Lower `188.9%` turnover vs `R3` `376.9%` cushions Sec 50AA tax; weaker P1 (`8.24%`) |
| **`F2` Legacy V1 (`Raw9M · DD>12% · 50/50`)** | `11.16%` | `7.08% → 25.07% → 12.52%` | `-27.47%` / `0.32` Sh | `24.01%` (`-25.26%`, `0.97` Sh) | Legacy baseline |
| **`ZeroFit_EQ4_GoldLiq50`** (`[0.25]*4` · `50/50`) | `13.44%` | `9.15% → 24.68% → 11.77%` | `-28.84%` / `0.52` Sh | `—` | Un-fitted ablation (`ZeroFit_EQ4_Liquid`: `12.65%`, `0.48` Sh) |
| **`Unhedged Risk-Adj 9M`** | `15.50%` | `11.47% → 24.18% → 20.74%` | `-30.84%` / `0.54` Sh | `27.71%` (`-37.85%`, `1.01` Sh, Raw9M) | Deep bear drawdowns (`-40.74%` P2 COVID, `-37.85%` 2018–26) |
| **Bench: Official NSE Nifty Midcap 100 (`16Y`)** | **12.95%** | `7.03% → 11.23% → 16.65%` | `-45.83%` / `0.31` Sh | `—` | Official capitalization-weighted NSE benchmark (`M100->M150 stitched`: `10.32%`) |
| **Bench: Official NSE Midcap 150 (`2019–26`)** | `—` | `— → — → 15.99%` | `—` | `17.76%` (`-38.67%`, `0.63` Sh) | Index live from `2019-01-14..2026-11-26` |
| **Bench: Clean PIT EW Proxy (`772` Eqs)** | **9.76%** | `0.40% → 15.23% → 14.57%` | `-54.08%` / `0.18` Sh | `21.60%` *(Biased Current 150 EW)* | Reflects real-time survival drag in 2011–16 bear cycle (`P1 = 0.40%`) |

---

## Core Architectural Decisions & Sensitivity Summary

1. **Multi-Factor Composite (`MF4`) vs Single Drawdown Triggers (`21,600+` Numba Grid):**
   - Across `21,600` single-threshold drawdown combinations, cross-period rank correlation was **Spearman ρ = `-0.092` (`P1 → P2`)** and **`+0.123` (`P1+P2 → P3`)**, demonstrating high overfitting risk for single DD rules.
   - Four causally MAD-normalized orthogonal blocks (`B1 Trend 0.44`, `B2 Vol/VIX 0.10`, `B3 Breadth 0.34`, `B4 Macro/Gold 0.12`, post-warmup correlation `0.219`–`0.496`) converged identically on `P1` (`2011–16`) and `P1+P2` (`2011–21`) to `[0.44, 0.10, 0.34, 0.11]`. Against zero-fit equal-weight ablation (`ZeroFit_EQ4_GoldLiq50` `13.44%` / `0.52` Sh), `R3` adds **`+2.17%` CAGR** and **`+0.09` Sharpe** while beating Official NSE Midcap 100 (`12.95%` / `-45.83%` DD / `0.31` Sh) by `+2.66%` p.a. and `17.05%` shallower Max DD.
2. **Hysteresis & Zweig Recovery Override (`tau 8/22`, `[-0.25, -0.10, +0.18]`, `25d` wait):**
   - Removing hysteresis spikes churn above `511%`/yr (`r_ratio = 1.0`). Asymmetric smoothing (`8d` down / `22d` up) + `25d` upward cooldown + Zweig breadth thrust (`21d` breadth surge `>= +25%` while breadth `>= 50%`) balances timely V-bottom re-entry with controlled churn.
3. **`TrendGold150` (`GOLDBEES` vs `EMA150 ± 2%`) & Alternatives (`F&O`, `Gilt`, `LowVol30`):**
   - Under realistic RBI T-bill yields + live `LIQUIDCASE` (`LIQUID_REALISTIC`), static `50/50` lagged `100% Gold` prior to 2026 (`14.32%` vs `16.25%` for `EQ4_MultiFactor`), while static `100% Gold` suffered during the `-31.7%` `2026_GoldSlump`.
   - `TrendGold150` routes defensive cash to `LIQUIDCASE` whenever `GOLDBEES < 0.98 × EMA150` (active today `2026-11-27`: `GOLDBEES ₹103.78` vs `EMA150 ₹118.41`, `-12.35%`). Note: **`P3 (2021–26)`** incorporates this 2026 Gold slump observation and is explicitly tagged **Post-Selection**.
   - Head-to-head alternatives (`numba_part2_verification.txt`): Long Gilt (`0.50` Sh `2016–26`, 30% debt tax), Internal LowVol30 (`0.29` Sh, `-40.96%` DD), VIX-scaled monthly 5% OTM Puts (`0.48`–`0.49` Sh), and Short Index Futures (`0.82` Sh, infeasible below `₹25L+` due to `₹16.5L` NSE minimum contract size).
4. **Live Posture (`as_of = 2026-11-27`):**
   - **Model Posture:** `RISK_ON (100% Target Equity)` · `MF4 Smoothed Composite = -0.032` (`Raw +0.012`; `2` positive/`4`? `B1 -0.457`, `B2 +0.033`, `B3 +0.461`, `B4 +0.440` -> `3 of 4` blocks positive; score exceeds `-0.10` cautious threshold so hysteresis holds `RISK_ON`).
   - **Autonomous Paper Bot (`inception = 2026-11-27`, `tax_defensive=True`):** `R3_1L_SlotCap = ₹99,850.29`, `R3_10L_Standard = ₹998,502.33` (reflecting `0.15%` Day-0 execution friction).

---

*Personal Quantitative Research & Paper-Trading Sandbox — Not Investment Advice / Not SEBI Registered.*
