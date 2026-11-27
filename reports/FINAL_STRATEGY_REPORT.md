# DIY Nifty Midcap 150 Momentum Index — 16-Year Zero-Survivorship & Multi-Period Research Specification

**Repository:** `HarjasChadha/midcap-momentum-index` · **Data Sources:** Official **NSE `PR` Archives (`2010-01-04` to `2026-11-26`, `4,171` trading days × `4,047` equities, `1,305+` corporate actions)** + **NSE MCP live quotes** + `2018–26` current-150 live panel.

---

## Executive Summary: Why Multi-Period + Zero-Survivorship Validation Upgraded `F2` → `R2`

Testing only a single time window or only the current 150 survivors misses two major statistical traps: (1) **survivorship bias (`+8.54%` p.a.)**, and (2) **single-period parameter overfitting**. After constructing a **Zero-Survivorship Point-in-Time Midcap Proxy** (`PIT_NO_SURVIVOR_150`, `787` distinct liquid midcaps ranked 81–230 by trailing 126d median NSE rupee turnover) and evaluating every strategy strictly across **three non-overlapping regimes (`2011–16 Train → 2016–21 Val #1 → 2021–26 Val #2`)** plus an independent **`2018–22 Train → 2022–26 OOS`** grid (150 configs):

| Strategy Configuration (`15bps` Cost + Indian STCG/LTCG Tax) | 16Y PIT Frac (`2011–26`) | 16Y PIT `₹1L` SlotCap (`End NAV`) | 16Y Current (`2011–26`) | Clean Warm (`2018–26`) | Blind P1 → P2 → P3 CAGR (PIT) |
|---|---:|---:|---:|---:|---|
| **Upgraded #1 Recommended (`R2`: Risk-Adj 9M EW · 6M · `DD>15%` Sh75%)** | **19.17%** (`-27.80%` DD, **0.77 Sh**) | **18.56%** (`-27.83%` DD, **`₹14,10,801`**) | **26.01%** (`1.23` Sh) | **27.11%** (`26.71%` ₹1L, `1.14` Sh) | `12.90% → 31.43% → 17.63%` |
| **Upgraded #2 Concentrated (`R7`: NormZ Risk-Adj 9M · 6M · `DD>15%`)** | **20.46%** (`-28.74%` DD, **0.82 Sh**) | **20.23%** (`-28.75%` DD, **`₹17,53,097`**) | — | **28.15%** (`-27.51%` DD, `1.17` Sh) | `12.93% → 42.36% → 21.80%` |
| **Hybrid 50/50 (`Raw + Risk Z · 6M · DD>15% Sh75%`)** | **18.38%** (**-27.68%** DD, `0.69` Sh) | **17.56%** (**-26.87%** DD, `₹12,37,050`) | **25.81%** (`1.17` Sh) | **27.03%** (`26.93%` ₹1L, `1.10` Sh) | `10.41% → 31.14% → 18.04%` |
| **Raw 9M EW + Widened `DD>15%` (`6M`)** | 18.02% (`-30.81%` DD, `0.64` Sh) | 16.76% (`₹11,12,434`) | 25.20% | — | `10.94% → 30.14% → 16.15%` |
| **Risk-Adj 9M EW + `DD>12%` (`6M`)** | 17.38% (`-32.24%` DD, `0.69` Sh) | 17.09% (`₹11,61,294`) | 24.42% (`1.17` Sh) | 26.78% (`26.11%` ₹1L, `-21.44%`) | `10.27% → 31.41% → 14.72%` |
| **Legacy V1 (`F2`: Raw 9M EW · 6M · `DD>12%` Sh75%)** | 16.27% (`-33.16%` DD, `0.57` Sh) | 15.62% (`-32.60%` DD, `₹9,54,524`) | 24.32% (`1.08` Sh) | 25.78% (`25.69%` ₹1L, `1.06` Sh) | `8.63% → 31.24% → 10.95%` |
| **Unhedged Raw 9M EW (`6M`)** | 18.13% (`-39.26%` DD, `0.53` Sh) | — | 27.47% | 28.28% (`-37.85%` DD) | `9.84% → 26.72% → 19.36%` |
| **Passive Equal-Weight Midcap 150 Benchmark** | **11.61%** (**-51.84%** DD, `0.27` Sh) | — | **20.15%** (`-38.96%` DD, `0.82` Sh) | **22.10%** (`-36.97%` DD, `0.89` Sh) | `2.49% → 17.24% → 15.99%` |

---

## 1. Corporate Action Verification & Survivorship Gap Quantification

- **Official NSE `PR` + `Bc` + MCP Pipeline (`src/nse_archive_loader.py`):** Back-adjusted `4,171` daily bhavcopies (`2010–2026`) by parsing `1,305+` Exchange Book-Closure (`Bc`) corporate action notices (`SPLIT`, `BONUS`, combined `BON + SPL`, 12-day duplicate suppression, postponed split suppression like `SUNILHITEC`, `SCHEME OF ARRANGEMENT`/demergers <= 1.05, plus post-2021 `NseMcpClient` split actions). Residual extreme single-day liquid glitches across 16.8M stock-days dropped from `417 → 13` genuine restructurings (`YESBANK`, `GTL`, `MAJESCO`, `INFIBEAM`, etc.), while `GOLDBEES` (2019 100:1 split), `BANKBARODA`, `RELAXO`, and `MOTILALOFS` audited cleanly (`1.15%` to `12.52%` max adj move).
- **Exact Survivorship Gap Measured (`+8.54%` p.a.):** Passive EW Midcap 150 returned **`11.61%`** CAGR (`-51.84%` DD, `0.27` Sharpe) on actual rolling point-in-time midcaps vs **`20.15%`** CAGR (`-38.96%` DD, `0.82` Sharpe) when backtesting 2026 survivors backwards. Live momentum returns after eliminating hindsight should therefore be anchored to the **`18.5%–20.5%`** PIT range (less scale tax drag above `₹1.25L` LTCG exemption), NOT unadjusted `26%–28%` survivor headlines.

---

## 2. Two Blind-Validated Upgrades (`Train → Val #1 → Val #2`)

1. **Factor Upgrade — `Risk-Adj 9M` (`return / ann_vol`) Over Pure `Raw 9M`:**
   - **Spearman Rank Skill (`PIT 2011–16 → 2016–21`):** `Sharpe rho = 0.484` / `Calmar rho = 0.457` (genuine persistence across blind regimes).
   - **Blind Selection:** On `2018–22 Train`, `Risk_9M_EW` averaged **`0.80` IS Calmar** (`#1` basket) and **`1.56` OOS Sharpe** in `2022–26`. On `2011–16 PIT Train`, `R2` finished **`#5` out of 96 configs (`12.90%`, `0.46` Calmar)** while passive PIT Midcap made `2.49%` (`-47.66%` DD), then went on to rank **`#1` hedged in P2 (`31.43%`, `1.22` Cal)** and **`19.17%` / `-27.80%` DD (`0.77` Sh)** over all 16 years.
   - **Practical Continuity (`29/30` Overlap):** On `2026-11-26`, `Risk_9M` holds **29 of the same 30 stocks** as `Raw_9M` (`LAURUSLABS`, `APARINDS`, `LENSKART`, `RADICO`, `AUROPHARMA`, `BHEL`, `LLOYDSME`, `ANTHEM`, `FEDERALBNK`, `IPCALAB`), simply promoting steadier trends inside Core Top-20 (`4.17%` weight) over fragile high-beta spikes and reducing 2018–26 turnover from **`260.3%` → `198.9%`**.
2. **Regime Upgrade — Widening Drawdown Trigger from `DD > 12%` → `DD > 15%`:**
   - Across `2011–16` and `2021–26`, `-12%` triggered premature exits during routine bull corrections (`2012`, `2015`, `2024–25`), missing snapback rallies (`F2` made `10.95%` in P3 vs `17.63%` for `R2`, and `8.63%` in P1 vs `12.90%` for `R2`).
   - Across all 6 baskets on 16y PIT, **`DD15_Sh75_GL50`** ranked **`#1` overall (`18.77%` mean CAGR, `-31.00%` DD, `0.70` Sharpe)** vs `17.34%` / `-32.73%` for `DD12_Sh75_GL50`, and **`#1` regime family on `2018–22` IS Calmar (`0.81`)**. Re-entry remains `> 50-DMA & DD > -9.0%` (`-0.6 × 15%`).

---

## 3. Core Architecture Kept Intact (`EXP2`–`EXP5` & `₹1L SlotCap`)

- **50/50 Top-20 + Top-30 Split:** `₹50k Top-20` (`4.167%` per rank 1–20) + `₹50k Top-30` (`1.667%` per rank 21–30). Optional **`R7` NormZ** (`8.46%` max single-stock weight on `2026-11-26`) raises 16y PIT `₹1L` CAGR to **`20.23%`** (`₹17,53,097` ending NAV).
- **Semi-Annual (`6M`) Rebalancing:** Cuts `398%` monthly STCG churn in half; `12M` optional above `₹5L`.
- **50% `GOLDBEES` + 50% `LIQUIDCASE` (`Sh75%` Weakest-First):** Protects against `GOLDBEES`' `-31.68%` 2026 drawdown and `SILVERBEES`' `-60.67%` crash while cutting STCG tax `17%–30%` by trimming Ranks 16–30 first.
- **`₹1,00,000` Sequential-SlotCap + `LIQUIDCASE` Sweep (`2026-11-26` Live Snapshot):**
  - **Regime Posture:** **`RISK_ON` (`100%` Equity)** — `EW Index = 846.80` (`DD252 = -4.38%` vs `-15.0%` exit trigger · `50DMA = 827.70`), `Nifty Midcap 150 = 22,650.0` (`-3.78%`), `India VIX = 16.84`, `Breadth>50DMA = 59.07%`.
  - **Live `R2` `₹1L` Allocation:** Holds **`30/30` stocks** (`86.86%` equity deployed) + **`112` units `LIQUIDCASE` (`₹13,115.20`)** = **`₹26.33` uninvested cash (`99.97%` deployed)**. Execute `T+1` between **`15:10–15:25 IST`**, skip drifts `< ₹500`, and defer non-emergency sells aged `330–364` days until day 365 for `12.5%` LTCG.
