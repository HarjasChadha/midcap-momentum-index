# Complete Mathematical & Systems Specification (`DIY Nifty Midcap 150 Momentum · V3 / R3`)

This document is the single complete source of truth for the entire application — **every formula, constant, data pipeline, approximation, live calculation, offline optimization, tax model, and browser calculation** — mapped directly to its exact source code on disk.

---

## 0. System Map: Where Every Number Comes From

The application has **four distinct computational layers**. Nothing is magic or hidden:

| Layer | Where It Lives | When It Runs | What It Produces | Uses Paid/Secret Keys? |
|---|---|---|---|---|
| **1. Historical Research & Parameter Discovery *(Offline)*** | `scripts/numba_research_engine.py`, `scripts/run_numba_sweeps.py`, `scripts/run_final_verification_suite.py`, `scripts/run_16yr_exact_verification.py` | Manual research runs on Mac/Colab | Tests `risk_9m`/`sortino_9m`, SciPy `MF4` weights `[0.44, 0.10, 0.34, 0.12]`, `TrendGold150` gate, delisting & 2x cost stress; writes `reports/src_exact_16yr_verification.json` & `reports/numba_*.json` | No |
| **2. Daily Live Production Engine *(Automated CI)*** | `scripts/update_live.py` + `src/` (`nse_mcp`, `data_loader`, `momentum`, `regime`, `backtester`, `paper_bot`) | **Mon–Fri `13:20 UTC` (`6:50 PM IST`)** via `.github/workflows/daily_update.yml` | Fetches NSE EOD quotes (`159` symbols + VIX), runs **`16` `2018–current` backtests** (`tax_defensive=True`, `15/35 bps`), updates **Top-150 scores & listing days**, writes `data/live_state.json`, `data/forward_tape.json`, and `data/paper_ledger.json` | No (public NSE MCP + Yahoo) |
| **3. Browser Presentation & Replay *(Client JS)*** | `index.html`, `assets/styles.css`, `assets/app.js` | Every page load & input edit | Fetches JSON data; computes `Sequential-SlotCap`, `HOLD/BUY/SELL` tax diff, and 1:1 `MMI_REPLAY` Tracked Views in `localStorage` | Zero backend |
| **4. Static Configuration & Tests** | `config/strategy.json`, `tests/test_engine.py`, `tests/test_paper_bot.py`, `scripts/verify_dashboard_ui.py` | On every CI run & pre-push | Enforces NaN-free JSON, MTM/Sec-50AA/IPO/parity invariants, and Playwright desktop/mobile E2E | No |

> **Important separation (`ponytail` CI constraint):** Production [`src/`](src/) and [`scripts/update_live.py`](scripts/update_live.py) use **only pure Python, `numpy`, and `pandas`** — `numba` and `scipy` are never imported in production CI ([`requirements.txt`](requirements.txt)).

---

## 1. Data Pipelines, Adjustments & Zero-Leak Masks

### 1.1 Live Production Universe (`2016–Current` Panel -> `data/cache/`)
Handled by [`src/data_loader.py`](src/data_loader.py) and [`scripts/update_live.py`](scripts/update_live.py):
1. **Constituents:** Loaded from [`data/universe/midcap150.csv`](data/universe/midcap150.csv) (official NSE Nifty Midcap 150 CSV, `150` stocks).
2. **Historical OHLC/Adj (`data/cache/`):** `10y` daily adjusted close `adj`, unadjusted `raw`, `volume`, and `BENCHMARK_TICKERS` (`NIFTYMIDCAP150.NS`, `GOLDBEES.NS`, `SILVERBEES.NS`, `LIQUIDBEES.NS`, `EBBETF0431.NS`, `LTGILTBEES.NS`, `LIQUIDCASE.NS`, `NIFTYBEES.NS`, `MON100.NS`, `^INDIAVIX`) cached in compressed CSVs. Days are retained only if $\ge \max(20, \lfloor 0.30 \times N_{\text{cols}} \rfloor)$ equities traded.
3. **Verified Yahoo Feed Fixes (`src/data_loader.py`):**
   - `MOTILALOFS`: Yahoo misapplied the `3:1` bonus (`0.25x`) starting `2024-01-01` instead of backward prior to `2024-01-01`; `adj` and `raw` before `2024-01-01` are multiplied by `0.25`.
   - `PATANJALI`: Pre-`2020-01-27` (pre-NCLT Ruchi Soya relisting `-95%` capital reduction) prices are masked to `NaN`.
   - `_repair_transient_glitches`: Any single-day return $r_t < -0.60$ that snaps back within `5` bars ($|P_{t+k}/P_{t-1} - 1| < 0.20$) is forward-filled with $P_{t-1}$.
4. **Live Daily EOD Merge (`scripts/update_live.py` -> [`src/nse_mcp.py`](src/nse_mcp.py)):**
   - Calls official NSE Streamable HTTP MCP endpoints (`https://mcp.nseindia.in/bhavcopy/cm/mcp` and `/cmmkt/mcp`):
     - `get_market_mood("today")` -> `india_vix`, advance/decline breadth (`stock_breadth`), index moves.
     - `get_bulk_quote(all_syms)` in chunks of `50` across all `150` stocks + `9` ETFs (`159` symbols).
     - `get_index_history("Nifty Midcap 150", months=12)`.
   - **Auto-Corporate Action Protection on Daily Append:** If any symbol exhibits $|C_{\text{new}}/C_{\text{prev}} - 1| > 0.20$ on `nse_date >= last_cached`, Python calls `get_corporate_actions(sym)`. If `actionType in {"SPLIT", "BONUS"}` with $0 < f < 1$ on `exDate == nse_date`, all pre-`ex_dt` prices in `adj_df` and `raw_df` are scaled by $f$.
   - Forward fill at end-of-series is strictly capped (`raw_df.ffill(limit=5)` for trade execution pricing only; `compute_score_matrix` uses strictly un-padded `adj_df` with zero forward-fill padding).

### 1.2 16-Year Official NSE Archive Dataset (`2010–2026`, Used in Research)
Handled by [`src/nse_archive_loader.py`](src/nse_archive_loader.py) (`4,171` trading days from `2010-01-04` to `2026-11-26` across `4,047` equities):
1. **Archive Parsing (`PR{ddmmyy}.zip`):** Extracts daily `Pd*.csv` (`EQ` preferred over `BE`/`BZ`, plus index rows `NIFTY_50`, `NIFTY_MIDCAP_100`, `NIFTY_MIDCAP_150`, `INDIA_VIX`), `etf*.csv` (`GOLDBEES`, `LIQUIDBEES`, `LIQUIDCASE`, `NIFTYBEES`, `MON100`, `SILVERBEES`, `EBBETF0431`, `LTGILTBEES`), and `Bc*.csv` corporate action notices (`1,328` applied actions logged in `corp_actions_applied.json`).
2. **Corrupted Zip Quarantine:** Rejects packed files whose inner `Pd` date mismatches the zip date, and drops days where $\ge 25$ liquid stocks simultaneously move $> 25\%$ and revert (specifically purging NSE's malformed `2018-01-02` zip).
3. **Deterministic Corporate Action Multiplier (`parse_bc_purpose_factor`):**
   - Pure cash/special dividends (`DIV`, `SPLDIV`, `AGM/DIV`) without split/bonus keywords return $f = 1.0$.
   - Bonus `A:B`: $f \leftarrow f \times \frac{B}{A+B}$.
   - Face-value Split / Sub-division / Consolidation `FV_old TO FV_new`: $f \leftarrow f \times \frac{\text{FV}_{\text{new}}}{\text{FV}_{\text{old}}}$.
   - If an unrecorded liquid step (`step_r < -0.45` or `> 1.20` where `20d` median `NET_TRDVAL >= ₹1 Cr`) has no `Bc` notice within $\pm 7$ days, it probes adjacent `PR` archives and NSE MCP `get_corporate_actions`, with empirical ratio fallback (`0.05 < r_drop < 0.85`) for demergers/arrangements.
   - Prevents double-adjustment by suppressing any second corporate action resolving within $\le 12$ calendar days on the same symbol.
   - Known gap patched in research loader [`scripts/run_numba_sweeps.py`](scripts/run_numba_sweeps.py): `DWARKESH` `2017-08-10` `10:1` split ($\times 0.10$ pre-`2017-08-10`).
   - Continuous INR Nasdaq 100 (`MON100_STITCHED`): stitches `0.1 × N100` (`2011–2021`) seamlessly into `MON100` (`2021–2026`).
4. **Strict ETF Purge & Zero-Look-Ahead Point-in-Time Universe (`build_clean_equity_and_pit_masks`):**
   - Drops all `419` NSE ETF symbols (`nse_etf_symbols.json` + substring filter `BEES, ETF, IETF, CASE, GILT, GSEC, LIQUID, GOLDSHARE, GOLD1, N100, MON100`) — eliminating ETF contamination (`LIQUID1`, `GOLD*`, etc.) from equity rankings.
   - **Semi-Annual PIT Proxy (`pit_semi`, `772` distinct liquid midcaps):** Reconstituted on the first trading day of **March and September** (`active >= 90` of trailing `126` trading days, `raw >= ₹20`), taking ranks **`81..230`** by trailing `126d` median NSE rupee turnover (`NET_TRDVAL`). Held stocks stay until the next semi-annual rebalance unless delisted $>5$ days or falling below `₹5`.
   - **Strictly Lagged Mask (`mask.shift(1)`):** Equal-weight benchmark returns (`build_ew_index`) and universe breadth (`build_regime_features`) use `mask.shift(1)` so selection on day $t$'s close only influences returns and breadth starting day $t+1$ (removing a $+1.47\%$ p.a. forward leak).

---

## 2. Stock Momentum Scoring & Weight Construction (`src/momentum.py`)

Given clean adjusted price panel $P_{t, i}$ (`adj_df`), let daily simple return be $r_{t, i} = \frac{P_{t,i}}{P_{t-1,i}} - 1$, lookback window $L \in \{126, 189, 252\}$ trading days, and skip window $s = 21$ days ($\approx 1$ calendar month short-term reversal avoidance).

### 2.1 Scoring Formulas (`compute_score_matrix` — Shared Identically by Research & Production)
For date $t$ and stock $i$, require un-padded valid prices at $P_{t,i}$, $P_{t-s,i}$, and $P_{t-s-L,i}$ AND at least `95%` non-null bar coverage (`count >= ceil(0.95 * (L + s + 1))` for return window, `ceil(0.95 * L)` for volatility windows), using canonical `TRADING_DAYS_PER_YEAR = 252.0` and daily simple return clipping $r^{\text{clip}}_{u,i} = \operatorname{clip}(r_{u,i}, -0.40, +0.60)$:

1. **Raw Momentum (`raw`):**
   $$R_{t,i}(L, s) = \frac{P_{t-s, i}}{P_{t-s-L, i}} - 1$$
2. **Volatility-Adjusted Risk Momentum (`risk_adj` — Default `R3` / `R2`):**
   $$\sigma^{\text{ann}}_{t,i}(L, s) = \max\!\left(0.08,\; \sqrt{252} \cdot \operatorname{std}\!\left(r^{\text{clip}}_{t-s-L+1..t-s, i}\right)\right), \qquad S^{\text{risk}}_{t,i} = \frac{R_{t,i}(L, s)}{\sigma^{\text{ann}}_{t,i}(L, s)}$$
3. **Downside Sortino Momentum (`sortino` — Runner-Up `S3`):**
   $$\sigma^{-}_{t,i}(L, s) = \max\!\left(0.06,\; \sqrt{252 \cdot \frac{1}{|W|}\sum_{u=t-s-L+1}^{t-s} \min\!\left(r^{\text{clip}}_{u,i}, 0\right)^2}\right), \qquad S^{\text{sortino}}_{t,i} = \frac{R_{t,i}(L, s)}{\sigma^{-}_{t,i}(L, s)}$$
*(Both `src/momentum.py` and `scripts/numba_research_engine.py` call the exact same `compute_score_matrix` helper: verified `max_abs_diff = 0.0` and `exact_nan_mask_match = True` across 16 years.)*
4. **PDF Normalized Z-Score Weighting (`norm_z` — Preset `R7`):**
   Over all valid scores on day $t$ with cross-sectional mean $\mu_t$ and std $\sigma_t$:
   $$z_{t,i} = \frac{S_{t,i} - \mu_t}{\sigma_t}, \qquad \phi(z_{t,i}) = \begin{cases} 1 + z_{t,i} & z_{t,i} \ge 0 \\ \frac{1}{1 - z_{t,i}} & z_{t,i} < 0 \end{cases}, \qquad w^{\text{sleeve}}_i = \frac{\phi(z_{t,i})}{\sum_{j \in \text{Top } N} \phi(z_{t,j})}$$

### 2.2 Why 50/50 Top-20 + Top-30 = 30 Stocks (Not 50)
Sleeve 1 holds Top $N_1 = 20$ at weight $\alpha_1 = 0.50$; Sleeve 2 holds Top $N_2 = 30$ at weight $\alpha_2 = 0.50$. Under `equal` intra-sleeve weighting, Ranks $1..20$ sit in **both** sets while Ranks $21..30$ sit only in Sleeve 2:
$$w_k = \begin{cases} \frac{0.50}{20} + \frac{0.50}{30} = 0.025 + 0.016667 = 4.1667\% & k \in \{1,\dots,20\} \\ \frac{0.50}{30} = 1.6667\% & k \in \{21,\dots,30\} \end{cases}$$
Total distinct stocks held $= 30$, with a $2.5\times$ tilt into the top $20$ leaders and $\sum_{k=1}^{30} w_k = 1.00$.

### 2.3 Sequential-SlotCap Affordability Filter (`select_affordable_blend`)
At small capital (e.g., $C = \text{₹}1,00,000$), a Top-20 slot is $\text{₹}4,166.67$ and a Rank 21–30 slot is $\text{₹}1,666.67$. If a stock trades above its rupee slot ($\text{LTP}_i > B \cdot w_i$), buying $\lfloor B \cdot w_i / \text{LTP}_i \rfloor = 0$ shares leaves dead cash.
Python resolves this iteratively (up to $60$ passes):
1. Initialize $B_{\text{ban}} = \emptyset$.
2. Compute blended Top-20 + Top-30 weights $w$ over valid symbols $\setminus B_{\text{ban}}$.
3. Find the **highest-ranked** selected symbol $i^*$ where $P^{\text{raw}}_{i^*} \le 0$ or not finite or $P^{\text{raw}}_{i^*} > B \cdot w_{i^*}$.
4. If none exists, return $w$; otherwise add $i^*$ to $B_{\text{ban}}$ and repeat (next ranks #31, #32... slide up).
- **Backtest affordability budget (`src/backtester.py`):** $B = V_t \cdot \max(0.50, \text{eq\_alloc}_t)$ so deep defensive cuts don't needlessly ban core holdings.
- **Browser default threshold (`index.html`):** Auto-checked when capital $< \text{₹}3,50,000$.

---

## 3. Causal 4-Block Multi-Factor Regime Engine (`src/regime.py`)

Executed by `compute_multifactor_state(feat, bench_df, weights=(0.44, 0.10, 0.34, 0.12))`.

### 3.1 Base Market Indices & Features (`build_ew_index`, `build_regime_features`)
- **Equal-Weight Midcap Index:**
  $$r^{\text{EW}}_t = \frac{1}{|U_{t-1}|} \sum_{i \in U_{t-1}} \left(\frac{P_{t,i}}{P_{t-1,i}} - 1\right), \qquad \text{EW}_t = 100 \prod_{u=1}^t (1 + r^{\text{EW}}_u)$$
- **Breadth above 50-DMA (`breadth_50`):**
  $$b^{50}_t = \frac{1}{|A_{t-1}|} \sum_{i \in A_{t-1}} \mathbf{1}\!\left\{P_{t,i} > \overline{P}^{(50)}_{t,i}\right\}$$
  where $\overline{P}^{(50)}_{t,i}$ uses `rolling(50, min_periods=35).mean()`.

### 3.2 Strictly Causal Rolling MAD Normalization (`_causal_mad_squash`)
For any time series $x_t$, using **only** past observations $S_t = \{x_u : t - 755 \le u \le t,\, x_u \text{ finite}\}$:
$$\text{If } |S_t| < 126:\; \tilde{x}_t = 0; \qquad \text{else } \text{med}_t = \operatorname{median}(S_t),\; \text{MAD}_t = 1.4826 \cdot \operatorname{median}(|S_t - \text{med}_t|)$$
$$\tilde{x}_t = \tanh\!\left(0.5 \cdot \frac{x_t - \text{med}_t}{\max(\text{MAD}_t, 10^{-6})}\right) \in (-1, +1)$$
*(No full-sample mean/std ever enters the calculation.)*

### 3.3 The Four Orthogonal Macro/Internal Blocks
Let `EMA_k(x)` denote pandas `ewm(span=k).mean()`, `vix` denote India VIX, and $\sigma^{(63)}(x) = \sqrt{252 \cdot \text{EMA}_{63}\!\left((r^x)^2\right)}$:

1. **Block 1 — Trend & Relative Strength ($w_1 = 0.44$):**
   $$t^{(200)}_t = \frac{\text{EW}_t / \text{EMA}_{200}(\text{EW})_t - 1}{\max(0.08, \sigma^{(63)}_{\text{EW}, t})}, \quad t^{(126)}_t = \frac{\text{EW}_t / \text{EMA}_{126}(\text{EW})_t - 1}{\max(0.08, \sigma^{(63)}_{\text{EW}, t})}, \quad t^{\text{rel}}_t = \frac{\text{Rel}_t / \text{EMA}_{126}(\text{Rel})_t - 1}{\max(0.04, \sigma^{(63)}_{\text{Rel}, t})}$$
   where $\text{Rel}_t = \text{EW}_t / \max(\text{NIFTYBEES}_t, 10^{-4})$, and:
   $$B_{1}(t) = \frac{\tilde{t}^{(200)}_t + \tilde{t}^{(126)}_t + \tilde{t}^{\text{rel}}_t}{3}$$
   *(or $\frac{1}{2}(\tilde{t}^{(200)}_t + \tilde{t}^{(126)}_t)$ if `NIFTYBEES` is absent).*

2. **Block 2 — Downside Semivolatility & VIX Stress ($w_2 = 0.10$, higher = calmer):**
   $$\sigma^{-}_{21, t} = \sqrt{252 \cdot \frac{1}{21}\sum_{u=t-20}^{t}\min(r^{\text{EW}}_u, 0)^2}, \qquad \nu_t = \ln\!\left(\max\!\left(0.2,\; \frac{\text{VIX}_t}{\text{EMA}_{63}(\text{VIX})_t}\right)\right)$$
   $$B_{2}(t) = -\frac{1}{2}\left(\widetilde{\ln \max(\sigma^{-}_{21,t}, 0.04)} + \tilde{\nu}_t\right)$$

3. **Block 3 — Market Breadth & Zweig Velocity ($w_3 = 0.34$):**
   $$\Delta b_t = \text{EMA}_5(b^{50})_t - \text{EMA}_5(b^{50})_{t-21}, \qquad B_{3}(t) = \frac{1}{2}\left(\widetilde{b^{50}_t} + \widetilde{\Delta b_t}\right)$$

4. **Block 4 — Daniel–Moskowitz Momentum Crash Risk & Gold Risk-Off Trend ($w_4 = 0.12$):**
   $$\text{DM}_t = \mathbf{1}\!\left\{t^{(200)}_t < 0\right\} \cdot \max\!\left(0,\; \widetilde{\ln \max(\sigma^{-}_{21,t}, 0.04)}\right), \qquad t^{\text{gold}}_t = \frac{\text{Gold}_t / \text{EMA}_{126}(\text{Gold})_t - 1}{\max(0.06, \sigma^{(63)}_{\text{Gold}, t})}$$
   $$B_{4}(t) = -\frac{1}{2}\left(\text{DM}_t + \tilde{t}^{\text{gold}}_t\right)$$
   *(If `GOLDBEES` is missing, $B_{4}(t) = -\text{DM}_t$.)*

**Composite Score:**
$$C^{\text{raw}}_t = 0.44 B_{1}(t) + 0.10 B_{2}(t) + 0.34 B_{3}(t) + 0.12 B_{4}(t)$$

### 3.4 Asymmetric EWMA, Breadth Thrust & 3-State Hysteresis FSM
At each daily close $t$:
- **Breadth Thrust Flag:** $\text{Thrust}_t = (\Delta b_t \ge 0.25) \land (b^{50}_t \ge 0.50)$.
- **Asymmetric Smoothing ($\tau_{\downarrow}=8, \tau_{\uparrow}=22$):**
  $$\alpha_t = \begin{cases} \frac{2}{8 + 1} = 0.2222 & \text{if } C^{\text{raw}}_t < C^{\text{sm}}_{t-1} \text{ or } \text{Thrust}_t \\ \frac{2}{22 + 1} = 0.0870 & \text{otherwise} \end{cases}, \qquad C^{\text{sm}}_t = \alpha_t C^{\text{raw}}_t + (1 - \alpha_t) C^{\text{sm}}_{t-1}$$
- **Upward Recovery Gate (`min_hold_up = 25` trading days since last downgrade day $t_{\downarrow}$):**
  $$\text{can\_up}_t = (t - t_{\downarrow} \ge 25) \lor \text{Thrust}_t$$
- **Finite-State Machine $s_t \in \{0=\text{DEFENSIVE},\; 1=\text{CAUTIOUS},\; 2=\text{RISK\_ON}\}$ (starts at $s_0 = 2$):**
  - From $s_{t-1} = 2$: if $C^{\text{sm}}_t < -0.25 \Rightarrow s_t = 0,\, t_{\downarrow} = t$; elif $C^{\text{sm}}_t < -0.10 \Rightarrow s_t = 1,\, t_{\downarrow} = t$; else $s_t = 2$.
  - From $s_{t-1} = 1$: if $C^{\text{sm}}_t < -0.25 \Rightarrow s_t = 0,\, t_{\downarrow} = t$; elif $\text{can\_up}_t \land (C^{\text{sm}}_t > +0.18 \lor (\text{Thrust}_t \land C^{\text{sm}}_t > -0.10)) \Rightarrow s_t = 2$; else $s_t = 1$.
  - From $s_{t-1} = 0$: if $\text{can\_up}_t \land (C^{\text{sm}}_t > -0.10 \lor (\text{Thrust}_t \land C^{\text{sm}}_t > -0.20)) \Rightarrow s_t = 1$; else $s_t = 0$.
- **Target Allocation & Causality Lag:**
  $w^{\text{EOD}}_t \in \{1.00,\, 0.60,\, 0.25\}$ for states $\{2, 1, 0\}$ (with `0.08` deadband).
  - **Live Dashboard Signal (`T+1` tomorrow):** reads $w^{\text{EOD}}_T$ (`target_eq_eod` at today's close).
  - **Backtester (`run_backtest`):** reads `alloc_lagged[t] = raw_eq[t - 1]` so day-$t$ closing regime shifts trade only on day $t+1$'s close (or $t$ closes using $t-1$'s close information).

---

## 4. Defensive Assets, `LIQUID_REALISTIC` & `TrendGold150` Gate (`src/regime.py`)

### 4.1 Unified Liquid Cash Benchmark Series
- **`LIQUID_REALISTIC` *(Used uniformly across ALL defensive presets `R3`, `S3`, `R2`, `F2`, `GOLD_50_LIQUID_50`, and `GOLD_OR_LIQUID`)*:** Uses `INDIA_TBILL_REGIME_YIELD` per calendar year (`2010: 5.50%`, `2011: 7.75%`, `2012–13: 8.15%`, `2014: 8.40%`, `2015: 7.50%`, `2016: 6.60%`, `2017: 6.15%`, `2018: 6.40%`, `2019: 5.65%`, `2020: 3.65%`, `2021: 3.45%`, `2022: 5.10%`, `2023: 6.75%`, `2024: 6.60%`, `2025: 6.15%`, `2026: 5.75%`, daily $r_d = (1+r_y)^{1/252} - 1$), and **automatically replaces synthetic yields with actual `LIQUIDCASE` daily returns** starting `2024-01-24` whenever `LIQUIDCASE` is present.
- **`LIQUID_TBILL`:** Constant $6.25\%$ ($252$-day basis) risk-free reference curve preserved solely for annualized Sharpe/Sortino hurdle comparisons.

### 4.2 Causal `TrendGold150` Hysteresis Gate (`get_trend_gold_status` & `TREND_GOLD150_OR_LIQ`)
Let $G_t$ be `GOLDBEES` (ffill `limit=5`) and $E^{(150)}_t = \text{EMA}_{150}(G)_t$ (`min_periods=30`), starting in `state_g = False` (`LIQUIDCASE`):
$$g_t = \begin{cases} \text{True (GOLDBEES)} & \text{if } \neg g_{t-1} \land G_t > 1.02 \cdot E^{(150)}_t \\ \text{False (LIQUIDCASE)} & \text{if } g_{t-1} \land G_t < 0.98 \cdot E^{(150)}_t \\ g_{t-1} & \text{otherwise} \end{cases}$$
In backtesting (`build_defensive_series`), the composite `TREND_GOLD150_OR_LIQ` index applies **$g_{t-1}$ lagged by 1 day** and deducts **`15 bps` (`0.0015`) switching friction** on every flip $g_{t-1} \neq g_{t-2}$:
$$r^{\text{TG150}}_t = \left(g_{t-1} r^{\text{Gold}}_t + (1 - g_{t-1}) r^{\text{LiqReal}}_t\right) - 0.0015 \cdot \mathbf{1}\{g_{t-1} \neq g_{t-2}\}$$

---

## 5. Event-Driven Backtester, Friction, Tax Lots & Metrics (`src/backtester.py`)

Executed by `run_backtest(...)`:
1. **Delisting / Halt Handling & Forced Write-Down Stress:** Bounded `ffill(limit=5)` on `adj`, `raw`, and `def_series` prevents stale ghost fills beyond 5 days. When an active stock lacks a valid finite quote (`not isfinite(px)` or `px <= 0`), mark-to-market uses `last_valid_px[sym]` (never reverting to purchase lot cost). Setting `delist_haircut_pct > 0` (stressed at `-15%` and `-25%`) forces write-down & permanent liquidation of stalled holdings at `last_valid_px * (1 - haircut)`.
2. **Scheduled Rebalance vs Staggered Tranches vs Regime Switch:**
   - Signal observation strictly uses `score_date = dates[i - 1]` (or `first_prev_date` on `i = 0`).
   - Supports staggered multi-tranche rebalancing (`num_sleeves = 2` splits `126d` 6M rebalance into `2` equal 50% sub-sleeves every `63d` to halve timing luck dispersion) and `phase_offset_days` for phase sensitivity audits.
3. **`weakest_first` Trimming (`R3`, `S3`, `R2`, `F2`, `F4`):** When `target_eq_frac < active_sum - 1e-6`, Python sorts all currently active targets **descending by yesterday's primary momentum score** and fills rupee weights greedily from rank 1 downward until `target_eq_frac` is reached. Ranks 16–30 absorb defensive trims first, preserving winners' 365-day holding clocks.
4. **Regime-Conditional Costs (`15 / 35 bps`) & Integer Shares:**
   - Standard scheduled rebalance friction is `15 bps` (`0.0015`) one-way. Unscheduled risk-off equity cuts (`DEFENSIVE`/`CAUTIOUS` transitions) deduct **`35 bps` (`0.0035`)** one-way impact (`regime_cost_bps_one_way=35.0`), stress-tested at **`30 / 60 bps` (2x friction)**.
   - Discrete target shares convert rupee targets to raw integer units: $n^{\text{raw}}_i = \lfloor (\text{TotalVal}_i \cdot w_i) / P^{\text{raw}}_{i,\text{sym}} \rfloor$. Residual cash sweeps into `LIQUID_REALISTIC` / `LIQUIDCASE` when `sweep_idle_cash=True`.
5. **Dual FIFO Tax Accounting (`apply_tax=True`, `tax_defensive=True` Default):**
   - Indian Fiscal Year: $\text{FY}(d) = \text{year}(d)$ if month $\ge 4$ else $\text{year}(d) - 1$.
   - Equity FIFO sells realize **`20%` STCG** ($< 365\text{d}$) and **`12.5%` LTCG** ($\ge 365\text{d}$ above `₹1,25,000` annual FY exemption) with STCL/LTCL carry-forwards.
   - Defensive & idle-sweep FIFO lots (`tax_defensive=True` default across primary runs and paper bot) charge **`30%` Sec 50AA slab tax** on `LIQUIDCASE`/`LIQUID_REALISTIC` debt redemptions and **`12.5%` ($\ge 365\text{d}$) / `30%` ($< 365\text{d}$)** on `GOLDBEES` switches (including internal `TrendGold150` flips). Pre-defensive-tax (`tax_defensive=False`) is retained only as a comparison column.
6. **Summary Metrics (`compute_metrics`, $r_f = 6.25\%$):**
   - Calendar years $Y = (\text{Date}_{\text{end}} - \text{Date}_{\text{start}}).\text{days} / 365.2425$, effective trading days/yr $D = N / Y$.
   - $\text{CAGR} = (V_N / V_0)^{1/Y} - 1$, $\sigma_{\text{ann}} = \operatorname{std}(r) \sqrt{D}$, $\text{Sharpe} = \frac{\text{CAGR} - 0.0625}{\sigma_{\text{ann}}}$, $\text{Sortino} = \frac{\text{CAGR} - 0.0625}{\operatorname{std}(r_{r<0})\sqrt{D}}$, $\text{Calmar} = \frac{\text{CAGR}}{|\min \text{DD}|}$, $\text{Ulcer} = \sqrt{\frac{1}{N}\sum (100\cdot \text{DD}_t)^2}$, $\text{AnnTurnover} = \frac{\sum (\frac{1}{2}\text{Notional}_t / V_t)}{Y} \times 100\%$.

---

## 6. Offline Numba + SciPy Research Models & F&O Simulator

### 6.1 Fast Numba Overlay Simulator (`_sim_overlay_core` in `scripts/numba_research_engine.py`)
Used to scan `21,600+` parameter sets in ~20 seconds:
- Approximates holding clock via weighted-average `hold_days` (`+= 1.47` calendar days per trading day) and weighted cost basis (`eq_cost_basis`, `def_basis`), charging $2 \cdot |\Delta w| \cdot 0.0015$ friction + `0.20` (`<365d`) / `0.125` (`>=365d`) equity tax scaled by `0.85`/`0.80` exemption calibration factor, and exact per-day defensive tax rate (`0.125` Gold vs `0.30` Liquid/Gilt).
- Metrics in Numba use canonical `252.0` trading days per year (`yrs = N / 252.0`).
- **Critical Validation Step:** Every finalist (`R3`, `S3`, `R2`, `F2`, `R3_Staggered_2x6M`, `ZeroFit_EQ4`) and stress test was evaluated from scratch using the **exact Python FIFO-lot engine** (`src/backtester.py`) in `scripts/run_16yr_exact_verification.py` and `scripts/run_final_verification_suite.py`.

### 6.2 SciPy Walk-Forward Block Weights Optimization (`_wf_opt_weights`)
Optimized `w = softmax(logits)` ($\sum_{k=1}^4 w_k = 1,\; w_k \ge 0$) via **6 random-seed Nelder–Mead optimizations** (`maxiter = 140` each) minimizing `-mean(Sharpe)` across specified slices:
- Optimized strictly on **`P1` (`2011–01-03` to `2016-05-31`) alone**: converged to `w = [0.44, 0.10, 0.34, 0.11]`.
- Optimized on **`P1 + P2` (`2011–2021`) jointly**: converged identically to `w = [0.44, 0.10, 0.34, 0.11]` (canonical `R3` uses `(0.44, 0.10, 0.34, 0.12)` summing strictly to `1.00`).

### 6.3 Realistic Monthly F&O Hedging Model (`sim_monthly_fno_hedges`)
Evaluated on `21`-trading-day monthly expiry cycles (`hedge_frac = 1 - eq_frac` while spot stays invested):
1. **Short Index Futures Overlay:** $r_{\text{fut},t} = r_{\text{eq},t} - h_t r^{\text{EW}}_t + h_t r^{\text{Liq}}_t - h_t \frac{0.020}{252}$ minus switch cost `0.0018 * |dh|` + spot rebalance `0.0030 * to_eq` (models ~`2.0%` p.a. net active carry/roll/Apr-2026 STT drag after earning T-bill on liquid collateral; requires **`₹25L–₹1Cr+`** due to `₹16.5L` NSE minimum futures lot notional + MTM cash buffer + `30%` F&O slab tax).
2. **Monthly `5%` OTM Bear Put & `5%–15%` Bear Put Spread:** At every `21d` expiry when $h_{\text{start}} > 0.05$, charges VIX-scaled Black–Scholes premium + STT/impact using lagged start-of-month VIX (`v_t = clip(VIX_{t-21}, 12, 60)`):
   - Single `5%` OTM Put cost $= h_{\text{start}} \cdot (0.011 \cdot \frac{v_t}{16} + 0.0015)$, payoff $= h_{\text{start}} \cdot \max(0, -0.05 - R_{\text{month}}) \times 0.70$ *(post `30%` slab tax)*.
   - `5%–15%` Bear Put Spread cost $= h_{\text{start}} \cdot (0.0065 \cdot \frac{v_t}{16} + 0.0015)$, payoff $= h_{\text{start}} \cdot \min(0.10, \max(0, -0.05 - R_{\text{month}})) \times 0.70$.

---

## 7. Verified Empirical Benchmarks (`src/backtester.py` Exact FIFO Results)

### 7.1 Clean ETF-Free 16-Year Zero-Survivorship PIT (`2011-01-03` to `2026-11-26`, Net Sec 50AA + `15/35 bps`)
Source: `scripts/run_16yr_exact_verification.py` → `reports/src_exact_16yr_verification.json` *(Note: `P3 (2021–26)` is explicitly labeled **Post-Selection / Partially In-Sample** for `TrendGold150`)*:

| Strategy / Benchmark (`tax_defensive=True`, `15/35 bps`) | P1 `11–16` | P2 `16–21` | P3 `21–26` (Post-Sel) | Full 16Y CAGR | 16Y MaxDD | Calmar | Sharpe | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Bench: Clean PIT EW (`772` Eqs)** | `0.40%` | `15.23%` | `14.57%` | **9.76%** | `-54.08%` | `0.18` | `0.18` | `—` |
| **Bench: Official NSE Nifty Midcap 100 (`16Y`)** | `7.03%` | `11.23%` | `16.65%` | **12.95%** | `-45.83%` | `0.28` | `0.31` | `—` |
| **Bench: Stitched Midcap 100 -> 150 (`16Y`)** | `6.84%` | `8.29%` | `15.99%` | `10.32%` | `-44.74%` | `0.23` | `0.24` | `—` |
| **Bench: Official Nifty Midcap 150 (`2019–26`)** | `—` | `—` | `15.99%` | **17.77%** | `-38.67%` | `0.46` | `0.63` | `—` |
| **`R3_src_Frac` (`Risk9M · MF4 · TrendGold150`)** | **11.43%** | `23.31%` | `16.78%` | **15.61%** | **`-28.78%`** | **`0.54`** | **`0.61`** | `376.9%` |
| **`R3_PreDefTax_Comparison` (`Sec 50AA Off`)** | `13.15%` | `25.14%` | `20.48%` | `17.73%` | `-28.45%` | `0.62` | `0.75` | `377.3%` |
| **`R3_Staggered_2x6M` (`50%` Every `3M`)** | `9.82%` | `19.63%` | `18.19%` | `13.48%` | **`-27.46%`** | `0.49` | `0.49` | `387.0%` |
| **`R3_Stress_30bps_2xCost` (`30 / 60 bps`)** | `9.87%` | `21.61%` | `14.88%` | `13.95%` | `-31.14%` | `0.45` | `0.50` | `374.2%` |
| **`R3_Delist_Haircut_15pct` (`11` write-downs)** | `11.03%` | `23.08%` | `16.70%` | `15.91%` | `-34.23%` | `0.46` | `0.57` | `429.8%` |
| **`R3_Delist_Haircut_25pct` (`11` write-downs)** | `10.80%` | `22.77%` | `16.66%` | `15.66%` | `-34.23%` | `0.46` | `0.56` | `429.9%` |
| **`R3_src_Disc1L` (`₹1L` SlotCap + Sweep)** | `8.83%` | `21.64%` | `16.08%` | **14.08%** | `-30.53%` | `0.46` | `0.51` | `375.2%` |
| **`S3_src_Frac` (`Sortino9M · MF4 · TrendGold150`)** | `11.07%` | `23.04%` | `17.67%` | `14.74%` | `-30.03%` | `0.49` | `0.56` | `368.8%` |
| **`R2_src_Frac` (`Risk9M · DD>15% · 50/50`)** | `8.24%` | `28.92%` | `18.59%` | `15.17%` | `-28.72%` | `0.53` | `0.61` | `188.9%` |
| **`F2_src_Frac` (`Raw9M · DD>12% · 50/50`)** | `7.08%` | `25.07%` | `12.52%` | `11.16%` | `-27.47%` | `0.41` | `0.32` | `206.5%` |
| **`ZeroFit_EQ4_GoldLiq50` (`[0.25]*4` Ablation)** | `9.15%` | `24.68%` | `11.77%` | `13.44%` | `-28.84%` | `0.47` | `0.52` | `353.9%` |
| **`ZeroFit_EQ4_Liquid` (`[0.25]*4` Ablation)** | `9.32%` | `23.79%` | `9.58%` | `12.65%` | `-27.30%` | `0.46` | `0.48` | `353.0%` |
| **`Unhedged_Risk9M` (`No Regime Hedge`)** | `11.47%` | `24.18%` | `20.74%` | `15.50%` | `-30.84%` | `0.50` | `0.54` | `134.8%` |

- **6-Phase Start-Date Rebalance Sensitivity (`0, +21, +42, +63, +84, +105d`):**
  - `R3`: CAGRs = `[15.61%, 15.87%, 17.35%, 14.64%, 12.75%, 13.81%]` (`15.01%` mean, `4.60%` spread).
  - **`R3_Staggered_2x6M`**: CAGRs = `[13.48%, 14.07%, 15.09%, 14.38%, 14.15%, 15.44%]` (`14.44%` mean, **`1.96%` spread** — cuts timing sensitivity by more than half).
  - `S3`: `[14.74%, 16.35%, 17.19%, 14.37%, 12.80%, 13.39%]` (`14.81%` mean, `4.39%` spread); `R2`: `15.77%` mean (`3.77%` spread).

### 7.2 Live Synchronized Current-Constituent Panel (`2018-01-02` to `2026-11-27`, Survivorship-Biased Disclosure)
Source: `scripts/update_live.py` → `data/live_state.json` *(Backtests today's 150 constituents historically with `tax_defensive=True` & `35 bps` risk-off slippage)*:

| Scenario | Ending Value | CAGR% | MaxDD% | Calmar | Sharpe | Sortino | Avg Eq% | Turn% | Total Tax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`R3_Risk9m_6M_MF4_TrendGold150`** | `₹8,69,568` | **27.53%** | `-26.00%` | `1.06` | `1.19` | `1.39` | `68.4%` | `380.7%` | `₹1,98,465` |
| `R3_PreDefTax_Comparison` (`Sec 50AA Off`) | `₹10,91,905` | `30.83%` | `-22.91%` | `1.35` | `1.39` | `—` | `68.4%` | `380.1%` | `₹83,985` |
| **`R3_Staggered_2x6M`** | `₹7,96,689` | `26.28%` | **`-21.68%`** | **`1.21`** | `1.16` | `—` | `68.4%` | `391.8%` | `₹1,98,173` |
| **`R3_Stress_30bps_2xCost`** | `₹7,66,143` | `25.75%` | `-29.55%` | `0.87` | `1.09` | `—` | `68.4%` | `380.9%` | `₹1,79,861` |
| **`R3_Disc1L_SlotCap`** *(2018–26 Luck Caveat)* | `₹8,75,465` | `27.62%` | `-25.05%` | `1.10` | `1.22` | `—` | `64.7%` | `381.8%` | `₹2,03,113` |
| **`S3_Sortino9m_6M_MF4_TrendGold150`** | **`₹8,98,283`** | **27.99%** | `-25.04%` | `1.12` | `1.22` | `—` | `68.4%` | `384.7%` | `₹2,07,903` |
| **`R2_Risk9m_6M_DD15_Sh75_GoldLiq50`** | `₹7,59,110` | `25.60%` | `-26.39%` | `0.97` | `1.06` | `—` | `90.3%` | `195.0%` | `₹58,103` |
| **`F2_9m_6M_DD12_Sh75_GoldLiq50`** | `₹6,77,852` | `24.01%` | `-25.26%` | `0.95` | `0.97` | `—` | `86.0%` | `255.1%` | `₹58,858` |
| **`Unhedged_9m_6M`** | `₹8,81,036` | `27.71%` | `-37.85%` | `0.73` | `1.01` | `—` | `100.0%` | `114.8%` | `₹37,294` |
| **`Midcap150_EW_Universe`** *(Biased)* | `₹5,70,291` | `21.60%` | `-36.97%` | `0.58` | `0.87` | `—` | `100.0%` | `0.0%` | `₹0` |
| **`NIFTY_MIDCAP_150`** *(Official `2019–26`)* | `₹3,62,058` | `17.76%` | `-38.67%` | `0.46` | `0.63` | `—` | `—` | `—` | `—` |

> **Small-Sample ₹1L SlotCap Clarification:** Over `2018–26`, `R3_Disc1L_SlotCap` nudged unconstrained `R3` by `+0.09%` (`27.62%` vs `27.53%`) purely due to avoiding a handful of high-priced laggards. Over the 16-year zero-survivorship PIT archive, slot capping incurs a realistic `-1.53%` annual drag (`14.08%` vs `15.61%`).

---

## 8. Client-Side Website Logic (`index.html`, `assets/app.js`, `assets/styles.css`)

All calculations execute purely in the browser against `data/live_state.json`, `data/forward_tape.json`, and `data/paper_ledger.json`:
1. **Preset Routing (`getLatestComputedOrders`):** Maps `R3`/`S3`/`R2`/`R7`/`F2`/`F4` cleanly with dynamic positive-block diagnostics (`composite_smooth` vs `-0.10` cautious threshold) and SEBI-safe Model Posture / Model Target Qty terminology.
2. **Holdings Diff & Tracked Views (`computeHoldingsDiff`, `MMI_REPLAY`):** Maintains 1:1 parity with `src/paper_bot.py` (`COST_RATE = 0.0015`, `REGIME_CUT_COST_RATE = 0.0035`, `tax_defensive = True`, `last_valid_prices` MTM retention, FIFO LTCG Shield, and `#pbNetDefTaxToggle` checked by default).

---

## 9. Explicit Parameter Provenance & System Boundaries

| Parameter / Choice | Exact Value | Provenance & Empirical Verification | Key Operating Boundary |
|---|---|---|---|
| **Momentum Lookback & Skip** | `189d` (`9M`), `21d` skip, `252d` ann, `>=95%` coverage | Empirical `[63..252d]` curve in `run_numba_sweeps.py`; unified across `src/momentum.py` & Numba (`0.0` drift) | Short-term `1M` reversal assumption in midcap equities |
| **Sleeve Blend** | `50% Top-20 + 50% Top-30` | Yields `30` unique stocks (`4.17%` Ranks `1–20`, `1.67%` Ranks `21–30`) vs `20/20`, `30/30`, `NormZ` | Slightly higher concentration (`4.17%`) in top 20 leaders |
| **Rebalance Cadence** | `126d` (`6M`) & `Staggered 2x6M` option | Balances turnover tax drag vs staleness; staggered `2`-tranche shrinks 6-phase spread from `4.60%` to `1.96%` | Fixed phase-0 rebalance varies across start dates |
| **MF4 Block Weights** | `[0.44, 0.10, 0.34, 0.12]` | Out-of-sample SciPy walk-forward on `P1 (2011–16)` alone (`[0.44, 0.10, 0.34, 0.11]`), confirmed on `P1+P2`; beats Zero-Fit `EQ4` (`15.61%` vs `13.44%`) | Uses causal `756d` rolling median/MAD normalization |
| **EWMA & Schmitt Hysteresis** | `tau 8/22`, `[-0.25, -0.10, +0.18]`, `25d` hold | Targets controlled turnover (`376.9%`) and Zweig thrust recovery (`>= +25%` 21d surge at breadth `>= 50%`) | `P3 (2021–26)` is disclosed as Post-Selection / Partially In-Sample |
| **`TrendGold150` Gate** | `GOLDBEES` vs `EMA150 +/- 2%` | Empirical EMA sweep (`50/100/150`) switching defensive reserves to `LIQUIDCASE` below `EMA150 - 2%` | Post-selection on `2026_GoldSlump`; incurs `2–4%` initial stop band |
| **Transaction Impact** | `15 bps` scheduled / `35 bps` risk-off | Models STT + fees + elevated market-impact slippage during stress trims (`30 / 60 bps` stress verified at `13.95%` 16Y CAGR) | Assumes liquid Midcap 150 execution near closing auction |
| **Indian FIFO & Sec 50AA Taxes** | `20%` STCG, `12.5%` LTCG, `30%` Sec 50AA debt, `12.5%/30%` Gold | Default `tax_defensive=True` in both `src/backtester.py` and `src/paper_bot.py` (`pre-def-tax` retained as comparison column only) | Price-return bhavcopy omits `~0.8–1.2%`/yr cash dividends |
| **Delisted MTM & Write-Downs** | `last_valid_px` (`ffill <= 5d`) | Never reverts halted/delisted stocks to purchase cost; `-15%` and `-25%` forced write-down stress tests reported cleanly (`-34.23%` DD) | Permanent wipeout timing depends on semi-annual purge |
| **PIT Universe Reconstruction** | `NET_TRDVAL` Ranks `81..230` (or CMIE MCap `101..250`) | Semi-annual zero-survivorship reconstitution across `772` liquid midcaps; drop-in CMIE Prowessdx loader wired in `src/prowess_loader.py` | Proxy uses turnover ranks alongside Official Midcap 100/150 benchmarks |

---

## 10. Autonomous Paper Trading Bot, Forward Tape & Tracked Views (`2026-11-27+`)

1. **Official Autonomous Ledger (`src/paper_bot.py` → `data/paper_ledger.json`):**
   - Executed daily at EOD by `scripts/update_live.py` (`Mon–Fri 13:20 UTC` / `6:50 PM IST`), tracking **`R3_1L_SlotCap`** (`₹1,00,000`) and **`R3_10L_Standard`** (`₹10,00,000`) alongside live baselines **`Midcap150_EW`** and **`Unhedged_Risk9M`**.
   - Enforces `T+1` causality, corporate actions, `tax_defensive=True` Sec 50AA / Gold tax deductions from cash (`nav_full_tax` default, `nav_pre_def_tax` comparison), `15 bps` scheduled / `35 bps` risk-off sales friction, `last_valid_prices` MTM retention, and byte-identical idempotency.
2. **Compact Forward Tape (`data/forward_tape.json`) & Tracked Views (`window.mmiForwardReplay`):**
   - Stores daily prices, `corp_actions`, multi-preset regimes, `listing_days` (`ld`) and `new_listing` (`nl`) flags, synced cleanly with the browser JS engine (`assets/app.js`).

---

*Personal Quantitative Research & Paper-Trading Sandbox — Not Investment Advice / Not SEBI Registered.*


