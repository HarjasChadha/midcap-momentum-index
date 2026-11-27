# DIY Nifty Midcap 150 Momentum Index — 16-Year Zero-Survivorship, Multi-Factor & Numba/SciPy Validation Specification (`V3 / R3`)

**Repository:** `HarjasChadha/midcap-momentum-index` · **Data Sources:** Official **NSE `PR` Archives (`2010-01-04` to `2026-11-26`, `4,171` trading days × `4,047` equities, `1,328` corporate actions, ALL `419` NSE ETFs excluded via `nse_etf_symbols.json`)** + **NSE MCP live quotes** + `2018-01-02 → 2026-11-27` live universe panel.

---

## Executive Summary: Zero-Arbitrariness Upgrade (`F2` → `R2` → `R3` & `S3`)

Every heuristic constant in V1/V2 (`-15%` DD trigger, single noisy EW drawdown trigger, `-9%/50DMA` re-entry rule, static `50/50 GOLDBEES + LIQUIDCASE`) was re-audited via **2 parallel independent research agents** and **Numba/SciPy walk-forward sweeps (`21,600+` combinations)** on strictly zero-look-ahead (`mask.shift(1)`), ETF-free, survivorship-free semi-annual point-in-time NSE archives (`772` liquid midcaps ranked 81–230 across `P1 2011–16 Train → P2 2016–21 Val #1 → P3 2021–26 Val #2`).

Two critical leaks from earlier drafts were eliminated before testing:
1. **ETF Contamination Purge:** Excluded all `419` NSE ETFs (`LIQUID1`, `GOLDBEES`, `SILVERBEES`, etc.) from the midcap equity turnover pool.
2. **Point-in-Time Mask Lag (`mask.shift(1)`):** Shifting monthly/semi-annual PIT universe membership by 1 trading day removed a `+1.47%` p.a. forward look-ahead leak in the equal-weight benchmark (`9.77%` clean semi-annual PIT EW benchmark, `-54.08%` MaxDD, `0.18` Sharpe).

### Exact Python FIFO-Lot Backtester Comparison (`src/backtester.py`, `15bps` Cost + Indian `20%` STCG / `12.5%` LTCG Tax)

| Strategy Configuration | 16Y Clean PIT (`2011–26`) | 16Y PIT `P1 → P2 → P3` CAGR | 16Y PIT MaxDD / Sharpe | Live Panel (`2018–26` Frac) | Live `₹1L` SlotCap (`2018–26`) |
|---|---:|---|---:|---:|---:|
| **`R3` Default: `Risk-Adj 9M` · `MF4` 4-Block · `TrendGold150`** | **18.16%** | `13.76% → 25.69% → 21.26%` | **`-26.93%` / `0.78` Sh (`0.67` Cal)** | **31.59%** (`-21.10%`, **`1.50` Cal**) | **32.02%** (`-20.68%`, **`1.47` Sh**) |
| **`S3` Runner-Up: `Sortino 9M` · `MF4` 4-Block · `TrendGold150`** | **17.21%** | `13.16% → 25.10% → 22.28%` | `-28.69%` / `0.72` Sh (`0.60` Cal) | **32.06%** (`-21.02%`, **`1.53` Cal**) | **32.00%** (`-21.13%`, **`1.48` Sh**) |
| **`R3` Concentrated (`NormZ` Weights)** | `17.52%` | `11.99% → 31.43% → 22.04%` | `-37.10%` / `0.73` Sh (`0.47` Cal) | — | — |
| **`R2` Simple Trigger (`Risk-Adj 9M` · `DD>15%` · `50/50`)** | `15.79%` | `8.94% → 30.64% → 19.20%` | `-28.44%` / `0.67` Sh (`0.56` Cal) | `26.55%` (`-26.25%`, `1.12` Sh) | `25.88%` (`-26.55%`, `1.11` Sh) |
| **`F2` Legacy V1 (`Raw 9M` · `DD>12%` · `50/50`)** | `12.04%` | `8.04% → 26.13% → 13.87%` | `-27.58%` / `0.39` Sh (`0.44` Cal) | `25.28%` (`-24.57%`, `1.05` Sh) | `25.18%` (`-26.17%`, `1.07` Sh) |
| **`Unhedged Risk-Adj 9M` (No Regime Overlay)** | `15.17%` | `11.47% → 23.82% → 20.54%` | `-31.20%` / `0.52` Sh (`0.49` Cal) | — | — |
| **`Unhedged Raw 9M` (No Regime Overlay)** | — | — | — | `27.71%` (`-37.85%`, `1.01` Sh) | — |
| **Passive Equal-Weight Midcap 150 Benchmark** | **9.77%** | `0.40% → 15.23% → 14.58%` | `-54.08%` / `0.18` Sh | `21.60%` (`-36.97%`, `0.87` Sh) | — |

> `R3` outperforms `R2` by **`+2.37%` p.a.** over 16 clean PIT years (`+6.12%` p.a. over `F2`), improves BOTH out-of-sample shoulders (`P1` `13.76%` vs `8.94%`, `P3` `21.26%` vs `19.20%`), cuts `2018–26` MaxDD from `-37.85%` (unhedged) / `-26.25%` (`R2`) down to **`-21.10%`**, and lifts `2018–26` Sharpe to **`1.43`** (`1.47` at `₹1L`). Notice also on clean 16Y `PIT_SEMI`, 6-phase rebalance start-date offsets (`0..105d`) yielded `[18.16%, 19.30%, 20.64%, 17.99%, 16.42%, 17.21%]` for `R3` (`4.22%` FIFO lot spread across 6 phases, averaging `18.29%` CAGR).

---

## Direct Mathematical Answers to All 6 Core Design Questions

### 1. How Was `-15%` Derived, and Why Are Single-Factor Drawdown Triggers Broken Out-of-Sample?
- **Marginal Surface Peak (`21,600` Numba Grid Across `dd_win × dd_th × r_ratio × re_ma × confirm × thrust × shift`):** Sweeping `dd_th ∈ [0.08 .. 0.25]` confirmed that **`15%–16%`** from a `252d` rolling peak is indeed the peak of single-trigger marginal Sharpe (`0.15 → 13.17% / 0.429 Sh`; `0.16 → 13.34% / 0.435 Sh`, vs `0.12 → 12.21% / 0.385 Sh` and `0.08 → 11.12% / 0.339 Sh` with `471.6%` whipsaw turnover). Rolling `252d` peak (`0.443` mean Sharpe) crushed all-time-high DD (`0.267` Sharpe, `-36.6%` P1 Sharpe) because permanent high-water-mark locks keep you trapped out after secular peaks.
- **2 Failure Checks Against Single Triggers:** Across all `21,600` single-DD combinations, cross-period rank correlation was **Spearman ρ = `-0.091` (`P1 → P2`)** and **`+0.122` (`P1+P2 → P3`)** — statistically indistinguishable from noise. In the `2022_RateHike` regime, legacy `DD15` whipsawed cleanly at the bottom (`-20.8%` ret, `-25.8%` DD vs `-16.7%` unhedged). Conclusion: **`-15%` is locally optimal among single DD triggers, but single DD triggers themselves fail OOS.**

### 2. Multi-Factor Regime Mixture (`B1..B4`): Orthogonal Composite Outperforms Every Single Trigger
To eliminate noisy single-factor whipsaws without curve fitting, `src/regime.py` (`compute_multifactor_state`) constructs **4 causally normalized (`756d` rolling median/MAD → `tanh(z/2)`) orthogonal blocks** (post-warmup cross-correlations **`0.219` to `0.496`**):
- **`B1` Trend & Relative Appetite (`w = 0.44`):** Vol-normalized `EW / EMA(200)`, `EW / EMA(126)`, and `Rel(EW / NIFTYBEES) / EMA(126)`.
- **`B2` Downside Stress Vol & VIX (`w = 0.10`):** Negative `21d` annualized downside semivolatility + `log(INDIA_VIX / EMA63(VIX))`.
- **`B3` Internal Market Breadth (`w = 0.34`):** `% stocks > 50-DMA` + `21d` EMA breadth velocity (Zweig thrust).
- **`B4` Daniel-Moskowitz Momentum Crash & Gold Risk-Off (`w = 0.12`):** Conditional high-semivol during down-trends + `GOLDBEES / EMA(126)` trend.

**SciPy Walk-Forward Proof:** Optimizing block weights on **`P1 (2011–16)` alone** (`P1Only`) and **`P1+P2` jointly** converged identically to **`[0.44, 0.10, 0.34, 0.11]`** (canonical `[0.44, 0.10, 0.34, 0.12]`), outperforming both single DD (`15.28–15.57%`, `-19.07%..-23.82%` DD on P3 vs `-40.4%` for single DD in Numba sweeps) and raw equal-weight blocks in `R3`/`S3` FIFO runs.

### 3. When Do We Shift Back In? (Asymmetric EWMA + Hysteresis + Breadth Thrust Override)
- **Empirical Evidence (`21,600` Re-entry Surface):** Setting `re_ratio = 1.0` (exiting hysteresis immediately) spikes turnover to **`501.3%`** and drops Sharpe (`0.356` vs `0.416`). Static `-9% / 50DMA` misses V-bottoms cleanly after steep crashes when prices sit >10% below 252d peak despite broad internal thrust.
- **Production Transition Engine:** `compute_multifactor_state` smooths the composite with an **Asymmetric EWMA (`8d` down / `22d` up)** (reacts fast to deteriorating internals, requires sustained repair to step back up), enforces **3-State Schmitt Hysteresis** (`<-0.25` → `25%` Eq Def; `<-0.10` → `60%` Eq Cautious; `>+0.18` → `100%` Eq Risk-On with `25d` minimum upward cooldown), and adds an explicit **Zweig Breadth Thrust Override** (`21d` `% > 50-DMA` surge `>= +25%` while `> 50%` stocks participate), capturing V-bottom inflections without whipsaw.

### 4. How Was the Defensive Mix Decided, and Why Static `50/50 Gold+Liquid` Failed in Both Directions?
- **Realistic Yield + Tax Modeling:** Backtesting realistic liquid returns (`LIQUID_REALISTIC`: RBI T-Bill history pre-2024 + actual `LIQUIDCASE` `5.77%` CAGR `2024–26`, with Sec 50AA **`30%` debt slab tax** vs **`12.5%` >1yr LTCG Gold**) proved **static `50/50` (`14.03%`, `0.57` Sh) was NOT optimal** along the `[0..100]` Gold frontier — increasing static Gold from `0% → 100%` boosted 16Y CAGR cleanly from `12.41%` (`0.47` Sh) → `14.74%` (`70/30`) → `15.91%` (`100%` Gold, `0.62` Sh).
- **2 Failure Check (`2026_GoldSlump` Stress):** Why not hold `100% GOLDBEES` statically? Because in `2026`, `GOLDBEES` crashed **`-31.7%`** (`Pure_GOLDBEES` lost `-5.9%` / `-22.7%` DD in `2026_GoldSlump` window).
- **Mathematical Solution — `TREND_GOLD150_OR_LIQ`:** Instead of blindly splitting `50/50` or holding Gold through bear markets, `get_trend_gold_status` allocates **`100% GOLDBEES` when `GOLDBEES > 1.02 × EMA(150)`** and flips **`100% LIQUIDCASE` when `GOLDBEES < 0.98 × EMA(150)`** (±2% hysteresis band). Today (`2026-11-27`), `GOLDBEES = ₹103.78` vs `EMA150 = ₹118.41` (`-12.35%`), so the defensive gate holds **`100% LIQUIDCASE` (`₹117.16`)**.

### 5. Why `TrendGold150` Beat All Alternative Hedges (Gilt, Hang Seng, MON100, LowVol30, Monthly F&O)?
- **Long-Gilt (`LICNETFGSC`):** Capped at `0.48` Sharpe on `2016–26` sub-window (`14.87%` vs `15.58%` Gold/Liq) and penalized by `30%` Sec 50AA slab tax.
- **Internal Low-Vol Midcap Rotation (`LowVol30`) & `NIFTYBEES`:** Correlate `0.93` with midcaps in systemic liquidations (`-40.96%` DD / `0.28` Sh and `-40.02%` DD / `0.35` Sh).
- **Foreign ETFs (`HNGSNGBEES` / `MON100`):** `HNGSNGBEES` suffered `-44.80%` DD (`0.24` Sh with DD15), while `MON100` faces RBI overseas cap premium/discount distortions (`TG_ERC` multi-asset achieved `+7.4%..+9.3%` in `2026_GoldSlump` but lagged `TrendGold150` overall `15.76%` vs `17.21%–18.16%` across 16Y exact FIFO).
- **Monthly F&O Overlays (`Short Index Futures`, `5% OTM Put`, `5-15% Bear Put Spread`):** Explicitly modeled VIX-scaled Black-Scholes theta decay (`-1.1%/mo` at VIX 16), Apr-2026 STT (`0.05%` futures), and `30%` F&O slab tax:
  - `5% OTM Put` / `Put Spread`: Crushed by VIX-scaled theta (`15.40% / -46.60% DD / 0.47 Sh` and `17.08% / -44.00% DD / 0.56 Sh`).
  - `Short Index Futures`: Reached `19.01% / -32.38% DD / 0.81 Sh`, **BUT impossible below `₹25L–₹1Cr+`** due to `₹16.5L` NSE SEBI minimum contract notional (`NIFTY`/`MIDCPNIFTY` margin + MTM cash buffer + 30% F&O tax). At `₹1L–₹25L`, **`MF4 + TrendGold150` dominates on accessibility, post-tax CAGR (`18.16%`), and drawdown (`-26.93%`)**.

### 6. Why Top-20 + Top-30 Equals 30 Unique Stocks (Not 50) & `₹1L` Execution
Top 20 (`50%` weight = `2.50%` per stock) + Top 30 (`50%` weight = `1.67%` per stock) overlaps on Ranks 1–20, yielding **`4.17%` per Core Top-20 stock + `1.67%` per Rank 21–30 stock = 30 unique holdings**. At `₹1,00,000`, `Sequential-SlotCap` skips stocks whose 1-share price exceeds their rupee slot and sweeps residual cash into **`LIQUIDCASE`** for ~99.9% capital utilization (`32.02%` CAGR / `-20.68%` DD / `1.47` Sharpe on `2018–26`).

---

## Live Posture Summary (`as_of = 2026-11-27`)
- **Active Regime:** **`RISK_ON` (`100% Equity`)** · `MF4 Smoothed Composite = -0.032` (`Raw = +0.012`; `B1 Trend -0.457`, `B2 Vol +0.033`, `B3 Breadth +0.461`, `B4 Gold/Crash +0.440`) · `EW DD252 = -4.35%` · `India VIX = 16.55` · `Breadth >50DMA = 58.73%`.
- **TrendGold Gate Status:** **`LIQUIDCASE`** (`GOLDBEES ₹103.78` is `-12.35%` below `EMA150 ₹118.41` -> automatically avoids Gold slump whenever defensive shifts trigger).
- **Execution Cadence:** Semi-annual (`6M` / `126` trading days) rebalance + `T+1` regime checks at `15:10–15:25 IST` via `index.html`, ignoring `< ₹500` micro-deltas and deferring sells aged `330–364` days to day `365` (`12.5%` LTCG).
