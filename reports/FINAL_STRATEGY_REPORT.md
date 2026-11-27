# DIY Nifty Midcap 150 Momentum Index — Research & Strategy Specification

**Repository:** `HarjasChadha/midcap-momentum-index` · **Data Window:** `2016-11-28` to `2026-11-26` (150 Nifty Midcap 150 constituents, 2,471 NSE trading days) · **Clean Comparison Window (100% Warm 9M & 12M Lookbacks):** `2018-01-02` to `2026-11-26`.

---

## Executive Recommendation

| Role | Strategy Configuration | Clean `2018–26` Post-Tax (`15bps`) | Defensive Taxed (`TaxDef`) | `₹1L` Discrete (`SlotCap + LiqSweep`) | Max DD / Calmar |
|---|---|---:|---:|---:|---:|
| **#1 Recommended Overall (`F2`)** | **50% Raw20 + 50% Raw30 · 9M LB (1M Skip) · Equal-Wt · 6M Rebal · Index `DD>12%` Shift `75%` Weakest-First → `50/50 Gold+Liquid`** | **25.78% CAGR** (`1.06` Sharpe) | **25.22%** (`1.03` Sharpe) | **25.69% CAGR** (`1.09` Sharpe, **81.9%** avg eq) | **-24.57%** (`1.05`) |
| **Runner-Up Concentrated (`F7`)** | Same as `F2` with **PDF `NormZ` weighting** (`1+z` / `(1-z)^-1`) | **28.34% CAGR** (`1.15` Sharpe) | **27.75%** (`1.12` Sharpe) | **28.08% CAGR** (`1.17` Sharpe) | **-24.25%** (`1.17`) |
| **Low-Turnover Alternative (`F1`)** | Same as `F2` with **`50%` defensive shift** (`214.5%` TO vs `260.3%`) | **26.19% CAGR** (`1.05` Sharpe) | **25.82%** (`1.03` Sharpe) | **25.93% CAGR** (`1.07` Sharpe) | **-27.96%** (`0.94`) |
| **Large-Capital Tax-Opt (`F4`, >₹5L)** | 50/50 Raw20+30 · **12M LB (1M Skip) · 12M Rebal** · `DD>12%` Sh50% → `Gold50/Liq50` | **28.12% CAGR** (`1.14` Sharpe, `168%` TO) | — | — | **-29.06%** (`0.97`) |
| **Unhedged Momentum Baseline (`B0`)** | 50/50 Raw20+30 · 9M LB · 6M Rebal · Always 100% Equity | **28.28% CAGR** (`1.03` Sharpe) | — | — | **-37.85%** (`0.75`) |
| **Equal-Weight Midcap 150 Universe** | Passive 150-stock EW benchmark (`MIDCAP EW`) | **22.10% CAGR** (`0.89` Sharpe) | — | — | **-36.97%** (`0.60`) |

> **Important Realism Haircut (`-6%` to `-10%` p.a. live expectation):** Because historical point-in-time constituent lists back to 2016 are not publicly archived, tests use the current (2026) 150 constituents (creating an observed **+8.0% to +8.8% p.a.** survivorship/EW gap vs `NIFTYMIDCAP150` / `MIDCAPIETF` TRI in `2019–2026`), and early-year returns on `₹1,00,000` benefit from India's `₹1.25L` annual LTCG exemption (`~1.5%–2.0%` extra tax drag at `₹1Cr`). Live net CAGR expectations are lower than headline backtests, while all **relative strategy advantages** hold across clean walk-forward windows.

---

## 1. Rebalancing Intervals, Holding Periods & Indian Tax Drag (`EXP2`)

Indian short-term gains (`< 365 days`, `20%` STCG) severely penalize rapid rotation:

- **Monthly (`1M`) rebalancing (`Raw_6M_Top25`):** `30.34%` pre-tax drops **-4.92% p.a.** to `25.42%` post-tax with `398.0%` annual turnover.
- **Quarterly (`3M`):** `31.39%` pre-tax drops **-3.54% p.a.** to `27.85%` post-tax (`217.3%` turnover).
- **Semi-Annual (`6M`) — Recommended Execution Balance:** `30.15%` → `27.91%` for 6M lookback (`150.0%` TO) and `33.49%` → **`32.18%`** for `Raw_9M_Top25_Reb6M` (`116.3%` TO; only `-1.31%` tax+cost spread). Keeps momentum fresh twice per year while allowing long-term winners kept across multiple cycles to cross 365 days into 12.5% LTCG (`₹1.25L` exempt).
- **Annual (`12M`):** Lowest friction/tax drag (`70.4%` TO unhedged 50/50 12M: **`29.84%`** net CAGR from clean `2018-01-02`, paying only `₹23,616` total tax on ₹1L vs `₹37,294` for 9M/6M). Best once capital exceeds **₹5,00,000** where STCG hurts more. Note: an initial experiment audit uncovered a `2017-12-01` warm-up bug where `12m+1m` lookback (`273` days needed vs `253` available) stayed 0% equity during 2018 — aligning to `2018-01-02` removed that illusion completely.

---

## 2. Portfolio Split (`₹50k Raw Top 20 + ₹50k Raw Top 30`) & Weighting

- **Blend Algebra:** Putting `₹50,000` into Raw Top 20 (`2.50%` weight per stock in sleeve) and `₹50,000` into Raw Top 30 (`1.667%` per stock in sleeve) creates a clean **30-stock tiered index**:
  - **Ranks 1–20:** `4.167%` each (`₹4,167` at ₹1L, holding 83.3% of capital in core momentum leaders)
  - **Ranks 21–30:** `1.667%` each (`₹1,667` at ₹1L, diversifying tail exposure across 10 secondary qualifiers)
- **Weighting Choice (`Equal` vs `NormZ`):**
  - **Equal-Weight (`F2`)** caps any individual stock at `4.17%`, yields tightest start-date stability (`1.60%` clean spread across Jan/Feb/Mar 2018 entry dates), and survives `WF1` (2018–20: `11.08%`, `-23.74%` DD), `WF2` (2021–23: `48.63%`, `-24.18%` DD), and `WF3` (2024–26: `28.95%`, `-17.06%` DD, `1.70` Calmar).
  - **PDF Normalized Z-Score (`F7`)** raises clean CAGR by `+2.56%` (`28.34%` vs `25.78%`), at the cost of `9.9%–12.35%` single-stock spikes and higher `START_OFFSET` sensitivity (`3.76%` spread).
- **Staggered vs Synchronized:** Splitting 6M into two staggered `3M`-offset ₹50k sleeves adds operational complexity (~41 holdings at ₹1L, worsening share-rounding efficiency) without beating single 6M rebalance.

---

## 3. Detecting Market Falls Across 30 Separate Stocks (`EXP4` & `DD_PLATEAU`)

You cannot trade a single portfolio price cleanly without distortions from cash flows and stock churn — instead, the **Equal-Weight Midcap 150 Universe Index (`ew_index`)** serves as the objective aggregate gauge (`50k Raw20+30 9M 6M`, 15bps, post-tax, 2017–26 baseline `31.93%` / `-37.86%` DD / `0.84` Calmar):

| Market-Fall Signal | Post-Tax CAGR | Max DD | Calmar | Ann. Turnover | Verdict |
|---|---:|---:|---:|---:|---|
| **Index `DD > 12%` (Re-enter `>50DMA & DD>-7.2%`, `Sh75%` Gold/Liq50)** | **29.96%** *(25.78% clean 18–26)* | **-24.22%** *(-24.57%)* | **1.24** *(1.05)* | **254.9%** | **Plateau Winner** (only ~9 switches in 9 yrs) |
| **Index `DD > 14%` (`Sh75%` Gold/Liq50)** | **29.63%** | **-25.32%** | **1.17** | **208.0%** | Confirms smooth `12%–15%` plateau |
| **MA200 Band (`-3%/+1%`, `Sh60%` Gold/Liq50)** | **28.65%** | **-25.94%** | **1.10** | **255.4%** | Solid #2 trend-only trigger |
| **Daily `< 200DMA` (unbanded)** | 25.59% | — | — | **674.4%** | Severe boundary whipsaw + STCG drag |
| **India VIX (`>24 / <20`, `Sh50%`)** | 28.20% | -27.80% | 1.01 | 201.0% | Misses grinding low-vol bear markets (2018, 2024–25) |
| **Composite 4-Score Tiered** | 26.05%–26.54% | -25.91% | 1.01 | **625–636%** | Excessive tax churn |
| **Stock Breadth50 (`<35% / >50%`, `Sh100%`)** | 24.91% | -32.19% | 0.77 | 451.9% | Too noisy for regime timing |
| **Per-Stock Trailing Stop 20%** | **23.77%** | **-22.87%** | 1.04 | 291.8% | Lowest DD, but loses `-8.2%` CAGR to premature stop-outs |

---

## 4. Gold vs Alternative Defensive Assets, Shift % & Trim Order (`EXP5`)

1. **Why Pure `GOLDBEES` Alone Is Risky Today (`2026-11-26` Proof):**
   - `GOLDBEES` peaked at `₹146.53` on `2026-01-29` and dropped **`-31.68%`** to `₹100.11` by `2026-11-25` (`₹100.96` latest).
   - `SILVERBEES` collapsed **`-60.67%`** in 2026 (`₹351.91 → ₹138.40`), yielding a catastrophic **`9.80%` CAGR / `-51.77%` MaxDD** inside `2022–26` backtests.
   - US Equities (`MON100`: `-55.08%` in-strategy DD) and Large-Cap Nifty (`NIFTYBEES`: `-35.38%` DD) correlate downward with Midcap crashes and fail as crash buffers.
2. **Winner Defensive Anchor — `50% GOLDBEES + 50% LIQUIDCASE` (`GOLD_50_LIQUID_50`):**
   - Standalone `2018–26` stats: **`11.70%` CAGR, `-14.50%` MaxDD, `0.81` Calmar, `0.65` Sharpe** — cutting pure gold volatility and crash risk in half while staying liquid (`₹100.96` & `₹117.10` per ETF unit — ideal for `₹1L` accounts, unlike `EBBETF0431` at `₹1,441` or `LIQUIDBEES` fractional dividend units). Note `EBBETF0431` (`26.58%` CAGR, `-23.28%` DD in `2022–26`) is a strong fixed-income substitute once capital exceeds `₹5L`.
3. **Shift Size (`75%` vs `50%`):** Shifting **`100%`** performs worst (`24.91%` in `EXP5`, `-25.08%` to `-32.19%` DD, highest STCG tax). Keeping **`25%` equity core (`75%` shift)** or **`50%` (`F1`)** keeps top momentum leaders compounding cleanly.
4. **What to Sell — `Weakest-First` Beats `Pro-Rata`:** Liquidating lower-ranked holdings (`Ranks 16–30`) first rather than cutting all 30 proportionally avoids `1 → 0` odd-lot truncation at `₹1L`, halves DP sell fees (`~₹20/ISIN`), lets Top-15 winners cross 365 days into LTCG, and reduced total backtest tax paid by **17%–30%** in `EXP5`.

---

## 5. Practical `₹1,00,000` Integer-Share Execution Rule (`Sequential-SlotCap`)

With Midcap 150 stock prices rising steeply (`APARINDS` `₹14,200`, `OFSS` `₹9,291`, `FLUOROCHEM` `₹4,781`, `RADICO` `₹4,671`), blind rounding on `2026-11-26` deploys only **`66.59%`** (`22/30` stocks, `₹33,408` idle cash).
Applying **Sequential-SlotCap + LIQUIDCASE Sweep** (implemented in `src/momentum.py` and the live dashboard):
1. Iteratively skip any symbol whose price exceeds its `₹` target slot (`₹4,167` for Top 20; `₹1,667` for Ranks 21–30), shifting affordable `#21+` names (`BHARATFORG ₹1,744`, `LGEINDIA ₹1,758`) into freed `₹4,167` slots and `#31–#41` (`ASTERDM`, `APLAPOLLO`, `GROWW`, `MEDANTA`, `NAM-INDIA`, `PETRONET`) into `#21–30` until **`30/30` slots** are held (`87.55%` equity deployed on `2026-11-26`).
2. Sweep residual cash (`₹12,448`) into **106 units of `LIQUIDCASE`** (`₹117.10`) for **`99.96%` total capital utilization** (`₹35.78` cash leftover).
3. In clean `2018–26` backtests, `Sequential-SlotCap + LiquidSweep` outperforms naive `NoCap` by **+111 bps p.a.** (`25.69%` vs `24.58%` for `F2`). Once your portfolio crosses **`₹3.5L–₹5L`**, disable SlotCap so high-priced leaders like `APARINDS`/`OFSS` fit directly into 1+ shares.

---

## 6. Live State Snapshot (`2026-11-26`, Verified via NSE MCP)

- **Regime Posture:** **`RISK_ON` (`100%` Equity / `0%` Defensive)** — `EW Index = 846.80` (`DD252 = -4.38%` vs `-12.0%` exit trigger; `>50DMA 827.70`), `Nifty Midcap 150 = 22,650.0` (`DD252 = -3.78%`), `India VIX = 16.84`, `Breadth>50DMA = 59.07%`.
- **Execution Discipline:** Generate orders from EOD signal (`T`), place limit/MIT orders `T+1` during **`15:10–15:25 IST`**, skip tiny drifts **`< ₹500`** to save DP/STT charges, and defer non-emergency sells aged **`330–364` days** until day 365 to capture `12.5%` LTCG (`₹1.25L` exemption) instead of `20%` STCG.
