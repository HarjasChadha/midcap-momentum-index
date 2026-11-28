# Complete Mathematical & Systems Specification (`DIY Nifty Midcap 150 Momentum · V3 / R3`)

This document is the single complete source of truth for the entire application — **every formula, constant, data pipeline, approximation, live calculation, offline optimization, tax model, and browser calculation** — mapped directly to its exact source code on disk.

---

## 0. System Map: Where Every Number Comes From

The application has **four distinct computational layers**. Nothing is magic or hidden:

| Layer | Where It Lives | When It Runs | What It Produces | Uses Paid/Secret Keys? |
|---|---|---|---|---|
| **1. Historical Research & Parameter Discovery *(Offline)*** | [`scripts/numba_research_engine.py`](scripts/numba_research_engine.py), [`scripts/run_numba_sweeps.py`](scripts/run_numba_sweeps.py), [`scripts/run_final_verification_suite.py`](scripts/run_final_verification_suite.py) | Manual research runs on Mac/Colab | Proved `risk_9m`/`sortino_9m`, SciPy `MF4` weights `[0.44, 0.10, 0.34, 0.12]`, `TrendGold150` gate; writes [`reports/numba_*.json`](reports/numba_deep_research.json) & [`reports/src_exact_16yr_verification.json`](reports/src_exact_16yr_verification.json) | No |
| **2. Daily Live Production Engine *(Automated CI)*** | [`scripts/update_live.py`](scripts/update_live.py) calling [`src/`](src/) (`nse_mcp`, `data_loader`, `momentum`, `regime`, `backtester`) | **Mon–Fri `13:20 UTC` (`6:50 PM IST`)** via [`.github/workflows/daily_update.yml`](.github/workflows/daily_update.yml) | Fetches NSE EOD quotes (`159` symbols + VIX), runs **13 full `2018–current` backtests**, updates **Top-150 scores**, T+1 `MF4` & `TrendGold150` live signals, `₹1L` orders, and writes [`data/live_state.json`](data/live_state.json) | No (public NSE MCP + Yahoo) |
| **3. Browser Presentation & Custom Sizer *(Client JS)*** | [`index.html`](index.html) | Every page load & input edit | Fetches [`data/live_state.json`](data/live_state.json); recomputes custom capital slot sizing (`Sequential-SlotCap`), `SIM_CAUTIOUS`/`SIM_CRASH`, `HOLD/BUY/SELL` holdings diff (`localStorage`), and renders Chart.js charts | Zero backend |
| **4. Static Configuration & Tests** | [`config/strategy.json`](config/strategy.json), [`tests/test_engine.py`](tests/test_engine.py), [`scripts/verify_dashboard_ui.py`](scripts/verify_dashboard_ui.py) | On every CI run & pre-push | Enforces strict NaN-free JSON schema, causality (`no look-ahead`), Sortino/TrendGold unit tests, and desktop/mobile Playwright E2E | No |

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
   - Forward fill at end-of-series is strictly capped (`raw_df.ffill(limit=5)` for trade execution pricing, `adj_df.ffill(limit=2)` for signal computation).

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

### 2.1 Scoring Formulas (`compute_score_matrix`)
For date $t$ and stock $i$, require $P_{t,i}, P_{t-s,i}, P_{t-s-L,i}$ finite and $\text{count}(P_{t-L-s+1..t,i}) \ge \max(20, \lfloor 0.85 L \rfloor)$:

1. **Raw Momentum (`raw`):**
   $$R_{t,i}(L, s) = \frac{P_{t-s, i}}{P_{t-s-L, i}} - 1$$
2. **Volatility-Adjusted Risk Momentum (`risk_adj` — Default `R3` / `R2`):**
   - In production [`src/momentum.py`](src/momentum.py#L46-L52):
     $$\sigma^{\text{ann}}_{t,i}(L, s) = \sqrt{252} \cdot \operatorname{std}\!\left(r_{t-s-L+1..t-s, i}\right), \qquad S^{\text{risk}}_{t,i} = \frac{R_{t,i}(L, s)}{\sigma^{\text{ann}}_{t,i}(L, s)}$$
     *(requiring at least $\max(20, \lfloor 0.8 L \rfloor)$ observations).*
   - *(Note on research script [`scripts/numba_research_engine.py`](scripts/numba_research_engine.py#L324-L348): uses $r^{\text{clip}} = \operatorname{clip}(r, -0.40, 0.60)$, $\sqrt{247}$, and denominator floor $\max(\sigma^{\text{ann}}, 0.08)$ before FIFO re-verification with exact production `src/momentum.py`.)*
3. **Downside Sortino Momentum (`sortino` — Runner-Up `S3`):**
   $$r^{\text{clip}}_{u,i} = \operatorname{clip}(r_{u,i}, -0.40, +0.60)$$
   $$\sigma^{-}_{t,i}(L, s) = \max\!\left(0.06,\; \sqrt{247 \cdot \frac{1}{|W|}\sum_{u=t-s-L+1}^{t-s} \min\!\left(r^{\text{clip}}_{u,i}, 0\right)^2}\right), \qquad S^{\text{sortino}}_{t,i} = \frac{R_{t,i}(L, s)}{\sigma^{-}_{t,i}(L, s)}$$
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
Let `EMA_k(x)` denote pandas `ewm(span=k).mean()`, `vix` denote India VIX, and $\sigma^{(63)}(x) = \sqrt{247 \cdot \text{EMA}_{63}\!\left((r^x)^2\right)}$:

1. **Block 1 — Trend & Relative Strength ($w_1 = 0.44$):**
   $$t^{(200)}_t = \frac{\text{EW}_t / \text{EMA}_{200}(\text{EW})_t - 1}{\max(0.08, \sigma^{(63)}_{\text{EW}, t})}, \quad t^{(126)}_t = \frac{\text{EW}_t / \text{EMA}_{126}(\text{EW})_t - 1}{\max(0.08, \sigma^{(63)}_{\text{EW}, t})}, \quad t^{\text{rel}}_t = \frac{\text{Rel}_t / \text{EMA}_{126}(\text{Rel})_t - 1}{\max(0.04, \sigma^{(63)}_{\text{Rel}, t})}$$
   where $\text{Rel}_t = \text{EW}_t / \max(\text{NIFTYBEES}_t, 10^{-4})$, and:
   $$B_{1,t} = \frac{\tilde{t}^{(200)}_t + \tilde{t}^{(126)}_t + \tilde{t}^{\text{rel}}_t}{3}$$
   *(or $\frac{1}{2}(\tilde{t}^{(200)}_t + \tilde{t}^{(126)}_t)$ if `NIFTYBEES` is absent).*

2. **Block 2 — Downside Semivolatility & VIX Stress ($w_2 = 0.10$, higher = calmer):**
   $$\sigma^{-}_{21, t} = \sqrt{247 \cdot \frac{1}{21}\sum_{u=t-20}^{t}\min(r^{\text{EW}}_u, 0)^2}, \qquad \nu_t = \ln\!\left(\max\!\left(0.2,\; \frac{\text{VIX}_t}{\text{EMA}_{63}(\text{VIX})_t}\right)\right)$$
   $$B_{2,t} = -\frac{1}{2}\left(\widetilde{\ln \max(\sigma^{-}_{21,t}, 0.04)} + \tilde{\nu}_t\right)$$

3. **Block 3 — Market Breadth & Zweig Velocity ($w_3 = 0.34$):**
   $$\Delta b_t = \text{EMA}_5(b^{50})_t - \text{EMA}_5(b^{50})_{t-21}, \qquad B_{3,t} = \frac{1}{2}\left(\widetilde{b^{50}_t} + \widetilde{\Delta b_t}\right)$$

4. **Block 4 — Daniel–Moskowitz Momentum Crash Risk & Gold Risk-Off Trend ($w_4 = 0.12$):**
   $$\text{DM}_t = \mathbf{1}\!\left\{t^{(200)}_t < 0\right\} \cdot \max\!\left(0,\; \widetilde{\ln \max(\sigma^{-}_{21,t}, 0.04)}\right), \qquad t^{\text{gold}}_t = \frac{\text{Gold}_t / \text{EMA}_{126}(\text{Gold})_t - 1}{\max(0.06, \sigma^{(63)}_{\text{Gold}, t})}$$
   $$B_{4,t} = -\frac{1}{2}\left(\text{DM}_t + \tilde{t}^{\text{gold}}_t\right)$$
   *(If `GOLDBEES` is missing, $B_{4,t} = -\text{DM}_t$.)*

**Composite Score:**
$$C^{\text{raw}}_t = 0.44 B_{1,t} + 0.10 B_{2,t} + 0.34 B_{3,t} + 0.12 B_{4,t}$$

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

### 4.1 Liquid Cash Benchmark Series
- **`LIQUID_TBILL`:** Constant $6.25\%$ annualized ($r_d = 1.0625^{1/252} - 1$). Note: Legacy `GOLD_50_LIQUID_50` and `GOLD_OR_LIQUID` in `build_defensive_series` use `daily_liq` ($6.25\%$) for historical compatibility with `R2`/`F2` presets.
- **`LIQUID_REALISTIC` *(Used by `R3`/`S3` sweep & defensive series)*:** Uses `INDIA_TBILL_REGIME_YIELD` per calendar year (`2010: 5.50%`, `2011: 7.75%`, `2012–13: 8.15%`, `2014: 8.40%`, `2015: 7.50%`, `2016: 6.60%`, `2017: 6.15%`, `2018: 6.40%`, `2019: 5.65%`, `2020: 3.65%`, `2021: 3.45%`, `2022: 5.10%`, `2023: 6.75%`, `2024: 6.60%`, `2025: 6.15%`, `2026: 5.75%`, daily $r_d = (1+r_y)^{1/248} - 1$), and **automatically replaces synthetic yields with actual `LIQUIDCASE` daily returns** starting `2024-01-24` whenever `LIQUIDCASE` is present!

### 4.2 Causal `TrendGold150` Hysteresis Gate (`get_trend_gold_status` & `TREND_GOLD150_OR_LIQ`)
Let $G_t$ be `GOLDBEES` (ffill `limit=5`) and $E^{(150)}_t = \text{EMA}_{150}(G)_t$ (`min_periods=30`), starting in `state_g = False` (`LIQUIDCASE`):
$$g_t = \begin{cases} \text{True (GOLDBEES)} & \text{if } \neg g_{t-1} \land G_t > 1.02 \cdot E^{(150)}_t \\ \text{False (LIQUIDCASE)} & \text{if } g_{t-1} \land G_t < 0.98 \cdot E^{(150)}_t \\ g_{t-1} & \text{otherwise} \end{cases}$$
In backtesting (`build_defensive_series`), the composite `TREND_GOLD150_OR_LIQ` index applies **$g_{t-1}$ lagged by 1 day** and deducts **`15 bps` (`0.0015`) switching friction** on every flip $g_{t-1} \neq g_{t-2}$:
$$r^{\text{TG150}}_t = \left(g_{t-1} r^{\text{Gold}}_t + (1 - g_{t-1}) r^{\text{LiqReal}}_t\right) - 0.0015 \cdot \mathbf{1}\{g_{t-1} \neq g_{t-2}\}$$

---

## 5. Event-Driven Backtester, Friction, Tax Lots & Metrics (`src/backtester.py`)

Executed by `run_backtest(...)`:
1. **Delisting / Halt Handling:** Bounded `ffill(limit=5)` on `adj`, `raw`, and `def_series` avoids carrying ghost valuations on long-delisted tickers past 5 trading days. Note: when an held stock disappears (`not isfinite(px)` or `<= 0`), mark-to-market falls back cleanly to its last lot cost price `sym_lots[-1][1]` until traded out.
2. **Scheduled Rebalance vs Regime Switch:**
   - Signal observation strictly uses `score_date = dates[i - 1]` (or `first_prev_date` on `i = 0`).
   - Trades trigger only on: `i == 0`, scheduled rebalance `(i - off) % reb == 0`, regime allocation change $|\text{target\_eq\_frac}_i - \text{prev\_target\_eq\_frac}| > 0.02$, or active `stopped_out` symbol.
3. **`weakest_first` Trimming (`R3`, `S3`, `R2`, `F2`, `F4`):** When `target_eq_frac < active_sum - 1e-6`, Python sorts all currently active targets **descending by yesterday's primary momentum score** (`score_matrices[sleeves[0]["score_key"]].loc[score_date]`) and fills rupee weights greedily from rank 1 downward until `target_eq_frac` is reached. Ranks 16–30 absorb defensive trims first, preserving winners' 365-day holding clocks.
4. **Costs & Integer Shares (`discrete_shares=True`):**
   - One-way friction rate `c = 15.0 / 10_000 = 0.0015` debited on every buy notional ($+c$) and credited off every sell notional ($-c$), covering STT, stamp/exchange fees, and impact.
   - Discrete target shares convert rupee targets to raw integer units: $n^{\text{raw}}_i = \lfloor (\text{TotalVal}_i \cdot w_i) / P^{\text{raw}}_{i,\text{sym}} \rfloor$, mapped to adjusted lot units via $n^{\text{adj}} = n^{\text{raw}} \cdot P^{\text{raw}} / P^{\text{adj}}$. Residual cash sweeps into `LIQUID_REALISTIC` / `LIQUIDCASE` when `sweep_idle_cash=True`.
5. **FIFO Tax Accounting (`apply_tax=True`):**
   - Indian Fiscal Year: $\text{FY}(d) = \text{year}(d)$ if month $\ge 4$ else $\text{year}(d) - 1$.
   - Every sale pops lots in **FIFO order**:
     - If holding days $\ge 365$: realized gain/loss offsets `fy_ltcg_carry`; positive net LTCG adds to `fy_ltcg_gains[FY]`, taxing only marginal excess above `₹1,25,000` annual exemption at **`12.5%` (`0.125`)**. Negative balance carries forward indefinitely.
     - If holding days $< 365$: realized gain/loss offsets `fy_stcg_carry`; positive net STCG is taxed immediately at **`20.0%` (`0.20`)**. Negative balance carries forward.
   - *(Model boundary: `tax_defensive=False` in `update_live.py` & main `run_backtest` calls unless explicitly enabled via `tax_defensive=True`, which taxes defensive redemptions at `12.5%` if `"GOLD" in defensive_asset` & `>=365d` else `30%`/`stcg`. However, all **Numba research optimization runs explicitly charged `30%` Sec 50AA slab tax on `LIQUID_REALISTIC`/Gilt and `12.5%` on Gold** every single switch!)*
6. **Summary Metrics (`compute_metrics`, $r_f = 6.25\%$):**
   - Calendar years $Y = (\text{Date}_{\text{end}} - \text{Date}_{\text{start}}).\text{days} / 365.2425$, effective trading days/yr $D = N / Y$.
   - $\text{CAGR} = (V_N / V_0)^{1/Y} - 1$, $\sigma_{\text{ann}} = \operatorname{std}(r) \sqrt{D}$, $\text{Sharpe} = \frac{\text{CAGR} - 0.0625}{\sigma_{\text{ann}}}$, $\text{Sortino} = \frac{\text{CAGR} - 0.0625}{\operatorname{std}(r_{r<0})\sqrt{D}}$, $\text{Calmar} = \frac{\text{CAGR}}{|\min \text{DD}|}$, $\text{Ulcer} = \sqrt{\frac{1}{N}\sum (100\cdot \text{DD}_t)^2}$, $\text{AnnTurnover} = \frac{\sum (\frac{1}{2}\text{Notional}_t / V_t)}{Y} \times 100\%$.

---

## 6. Offline Numba + SciPy Research Models & F&O Simulator

### 6.1 Fast Numba Overlay Simulator (`_sim_overlay_core` in `scripts/numba_research_engine.py`)
Used to scan `21,600+` parameter sets in ~20 seconds:
- Approximates holding clock via weighted-average `hold_days` (`+= 1.47` calendar days per trading day) and weighted cost basis (`eq_cost_basis`, `def_basis`), charging $2 \cdot |\Delta w| \cdot 0.0015$ friction + `0.20` (`<365d`) / `0.125` (`>=365d`) equity tax scaled by `0.85`/`0.80` exemption calibration factor, and exact per-day defensive tax rate (`0.125` Gold vs `0.30` Liquid/Gilt).
- Metrics in Numba use exact `247.0` NSE average trading days per year (`yrs = N / 247.0`).
- **Critical Validation Step:** Every finalist found in Numba (`R3`, `S3`, `R2`, `F2`) was then re-evaluated from scratch using the **exact Python FIFO-lot engine** (`src/backtester.py`) in `run_final_verification_suite.py` and `reports/src_exact_16yr_verification.json`.

### 6.2 SciPy Walk-Forward Block Weights Optimization (`_wf_opt_weights`)
Optimized `w = softmax(logits)` ($\sum_{k=1}^4 w_k = 1,\; w_k \ge 0$) via **6 random-seed Nelder–Mead optimizations** (`maxiter = 140` each) minimizing `-mean(Sharpe)` across specified slices:
- Optimized strictly on **`P1` (`2011–01-03` to `2016-05-31`) alone**: converged to `w = [0.44, 0.10, 0.34, 0.11]`.
- Optimized on **`P1 + P2` (`2011–2021`) jointly**: converged identically to `w = [0.44, 0.10, 0.34, 0.11]` (canonical `R3` uses `(0.44, 0.10, 0.34, 0.12)` summing strictly to `1.00`).

### 6.3 Realistic Monthly F&O Hedging Model (`sim_monthly_fno_hedges`)
Evaluated on `21`-trading-day monthly expiry cycles (`hedge_frac = 1 - eq_frac` while spot stays invested):
1. **Short Index Futures Overlay:** $r_{\text{fut},t} = r_{\text{eq},t} - h_t r^{\text{EW}}_t + h_t r^{\text{Liq}}_t - h_t \frac{0.020}{247}$ minus switch cost `0.0018 * |dh|` + spot rebalance `0.0030 * to_eq` (models ~`2.0%` p.a. net active carry/roll/Apr-2026 STT drag after earning T-bill on liquid collateral; requires **`₹25L–₹1Cr+`** due to `₹16.5L` NSE minimum futures lot notional + MTM cash buffer + `30%` F&O slab tax).
2. **Monthly `5%` OTM Bear Put & `5%–15%` Bear Put Spread:** At every `21d` expiry when $h_{\text{start}} > 0.05$, charges VIX-scaled Black–Scholes premium + STT/impact using lagged start-of-month VIX (`v_t = clip(VIX_{t-21}, 12, 60)`):
   - Single `5%` OTM Put cost $= h_{\text{start}} \cdot (0.011 \cdot \frac{v_t}{16} + 0.0015)$, payoff $= h_{\text{start}} \cdot \max(0, -0.05 - R_{\text{month}}) \times 0.70$ *(post `30%` slab tax)*.
   - `5%–15%` Bear Put Spread cost $= h_{\text{start}} \cdot (0.0065 \cdot \frac{v_t}{16} + 0.0015)$, payoff $= h_{\text{start}} \cdot \min(0.10, \max(0, -0.05 - R_{\text{month}})) \times 0.70$.

---

## 7. Verified Empirical Benchmarks (`src/backtester.py` Exact FIFO Results)

### 7.1 Clean ETF-Free 16-Year Zero-Survivorship PIT (`2011-01-03` to `2026-11-26`, `15bps` Cost + Indian Tax)
Source: [`reports/src_exact_16yr_verification.json`](reports/src_exact_16yr_verification.json) & [`reports/numba_deep_research.txt`](reports/numba_deep_research.txt):

| Strategy (`15bps` + STCG/LTCG Tax) | P1 `11–16` | P2 `16–21` | P3 `21–26` | Full 16Y CAGR | 16Y MaxDD | Calmar | Sharpe | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Passive EW Midcap 150 (`pit_semi` Bench)** | `0.40%` | `15.23%` | `14.58%` | **9.77%** | `-54.08%` | `—` | `0.18` | `—` |
| **`R3_src_Frac` (`Risk 9M · MF4 · TrendGold150`)** | **13.76%** (`-25.20%`) | `25.69%` (`-25.79%`) | `21.26%` (`-26.41%`) | **18.16%** | **`-26.93%`** | **`0.67`** | **`0.78`** | `374.1%` |
| **`R3_src_Disc1L` (`₹1L SlotCap + Liq Sweep`)** | `11.00%` | `23.85%` | `20.87%` | **16.83%** | `-29.47%` | `0.57` | `0.69` | `381.2%` |
| **`S3_src_Frac` (`Sortino 9M · MF4 · TrendGold150`)** | `13.16%` (`-25.63%`) | `25.10%` (`-27.63%`) | **22.28%** (`-25.22%`) | **17.21%** | `-28.69%` | `0.60` | `0.72` | `371.9%` |
| **`S3_src_Disc1L` (`₹1L SlotCap + Liq Sweep`)** | `10.99%` | `23.02%` | `20.96%` | `15.78%` | `-31.72%` | `0.50` | `0.63` | `375.0%` |
| **`R3_NormZ_Frac` (`NormZ Risk 9M`)** | `11.99%` | `31.43%` | `22.04%` | `17.52%` | `-37.10%` | `0.47` | `0.73` | `366.5%` |
| **`R2_src_Frac` (`Risk 9M · DD>15% · 50/50`)** | `8.94%` (`-28.44%`) | `30.64%` (`-19.35%`) | `19.20%` (`-31.63%`) | `15.79%` | `-28.44%` | `0.56` | `0.67` | `186.2%` |
| **`F2_src_Frac` (`Raw 9M · DD>12% · 50/50`)** | `8.04%` (`-25.99%`) | `26.13%` (`-18.82%`) | `13.87%` (`-30.46%`) | `12.04%` | `-27.58%` | `0.44` | `0.39` | `205.3%` |
| **`Unhedged_Risk9M` (`No Regime Hedge`)** | `11.47%` | `23.82%` (`-41.64%`) | `20.54%` | `15.17%` | `-31.20%` | `0.49` | `0.52` | `134.2%` |

- **6-Phase Start-Date Rebalance Sensitivity (`0, +21, +42, +63, +84, +105d`):** `R3` CAGRs = `[18.16%, 19.30%, 20.64%, 17.99%, 16.42%, 17.21%]` (`4.22%` spread, `18.29%` mean); `S3` = `[17.21%, 19.80%, 20.40%, 17.29%, 15.99%, 16.86%]` (`4.41%` spread).

### 7.2 Live Synchronized Matrix (`2018-01-02` to `2026-11-27`)
Source: [`scripts/update_live.py`](scripts/update_live.py) -> [`data/live_state.json`](data/live_state.json):

| Scenario | Ending ₹ (`₹1L` Start) | CAGR% | MaxDD% | Calmar | Sharpe | Sortino | Avg Eq% | Turn% | Tax (₹) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`R3_Disc1L_SlotCap`** | **₹11,83,539** | **32.02%** | **`-20.68%`** | **`1.55`** | **`1.47`** | `1.72` | `65.1%` | `380.9%` | `₹93,428` |
| **`R3_Risk9m_6M_MF4_TrendGold150`** | ₹11,49,728 | **31.59%** | `-21.10%` | `1.50` | `1.43` | `1.68` | `68.4%` | `380.0%` | `₹87,582` |
| **`S3_Sortino9m_6M_MF4_TrendGold150`** | ₹11,86,845 | **32.06%** | `-21.02%` | `1.53` | `1.46` | `1.72` | `68.4%` | `384.1%` | `₹96,544` |
| **`S3_Disc1L_SlotCap`** | ₹11,81,794 | **32.00%** | `-21.13%` | `1.51` | `1.48` | `1.73` | `64.8%` | `385.0%` | `₹96,175` |
| **`R7_NormZ_Risk9m_6M_DD15_Sh75_GoldLiq50`** | ₹8,71,111 | `27.55%` | `-27.51%` | `1.00` | `1.15` | `1.41` | `90.3%` | `199.2%` | `₹47,025` |
| **`R2_Risk9m_6M_DD15_Sh75_GoldLiq50`** | — | `26.55%` | `-26.25%` | `1.01` | `1.12` | — | `90.2%` | — | — |
| **`F2_9m_6M_DD12_Sh75_GoldLiq50`** | — | `25.28%` | `-24.57%` | `1.03` | `1.05` | — | `86.0%` | — | — |
| **`Unhedged_9m_6M`** | — | `27.71%` | `-37.85%` | `0.73` | `1.01` | — | `100.0%` | — | — |
| **`Midcap150_EW_Universe`** | ₹5,70,291 | `21.60%` | `-36.97%` | `0.58` | `0.87` | — | `100.0%` | `—` | `—` |

---

## 8. Client-Side Website Logic (`index.html`)

All calculations on the Execution Calculator tab happen purely in your browser using `STATE.universe_scores` and `STATE.regime` from `data/live_state.json`:
1. **Preset Routing (`getLatestComputedOrders`):**
   - `R3`: `rk9` scores, `MF4` live equity %, defensive goes `100%` to `STATE.regime.active_defensive_etf` (`LIQUIDCASE` or `GOLDBEES`).
   - `S3`: `so9` scores, `MF4` live equity %, defensive `100%` to `active_defensive_etf`.
   - `R2` / `R7` / `F2` / `F4`: uses `rk9`/`r9`/`r12`, legacy EW `DD252` trigger (`-15%` or `-12%`), splits defensive `50% GOLDBEES + 50% LIQUIDCASE` on crash simulation.
2. **Posture Override:** `LIVE` reads today's EOD posture; `SIM_CAUTIOUS` forces `60% Eq / 40% Def`; `SIM_CRASH` forces `25% Eq / 75% Def` (`50/50` for `F4`).
3. **Holdings Diff (`computeHoldingsDiff`):** Parses your local `SYMBOL,SHARES,YYYY-MM-DD` CSV in `localStorage`, skips trades whose rupee value `< ₹500` *(unless full exit `tgt === 0`)*, and raises a **LTCG warning** (`⚠ Held {d}d — LTCG in {365-d}d!`) whenever selling a lot held `330–364` days.

---

## 9. Explicit Parameter Provenance & Known Limitations

| Parameter / Choice | Exact Value | How Determined? *(Empirical vs Design Choice)* | Key Limitation / Assumption |
|---|---|---|---|
| **Momentum Lookback & Skip** | `189d` (`9M`) / `21d` skip | Empirical curve `[63..252d]` in `numba_deep_research.txt` (`Sortino_189` / `Risk_189` lead at `6M` cadence) | Assumes 1-month short-term reversal persists in Indian midcaps |
| **Sleeve Structure** | `50% Top-20 + 50% Top-30` | Structural constraint preserving PDF/user split while verifying vs `20/20`, `30/30`, `15/25`, `NormZ` | Slightly higher idiosyncratic weight in Top 20 (`4.17%`) vs pure EW-30 (`3.33%`) |
| **Rebalance Interval** | `126` trading days (`6M`) | Empirical tradeoff (`EXP 2`: cuts `398%` monthly STCG drag while rotating fast enough) | Any fixed rebalance day exhibits `~4.2%` 6-phase CAGR dispersion across start dates |
| **MF4 Block Weights** | `[0.44, 0.10, 0.34, 0.12]` | **Out-of-sample SciPy optimization** strictly on `P1 (2011–16)` (`[0.44, 0.10, 0.34, 0.11]`), confirmed identical on `P1+P2` | Weights `NIFTYBEES` in B1 & `GOLDBEES` in B4; assumes macro/breadth crisis correlations remain informative |
| **EWMA & Hysteresis Thresholds** | `tau 8/22`, `[-0.25, -0.10, +0.18]`, `25d` hold, `[0.25, 0.60, 1.00]` | Sweep across 3-state low-churn grid targeting controlled turnover (`~374%` vs `480%` un-hysteresed) | Turns `31.6%–35%` of time into defense (`2018–26 avg eq 65%–68%`) — can underperform sudden melt-ups when `comp_smooth` lags |
| **`TrendGold150` Band** | `GOLDBEES` vs `EMA150 ±2%` | Empirical shootout (`EMA 50/100/150`) solving static Gold's `-31.7%` 2026 bear market | Can still take initial `-2%` to `-4%` stop-out heat before `EMA150` `-2%` triggers |
| **Transaction Cost** | `15 bps` (`0.15%`) one-way | Indian delivery STT (`0.10%`) + exchange/SEBI/stamp (`~0.035%`) + limit slippage (`~0.015%`) | Assumes liquid Midcap 150 fills at closing auction/15:20 IST match daily close within `1.5 bps` impact (stress-tested at `30 bps` too) |
| **Taxation in Backtester** | `20%` STCG, `12.5%` LTCG (`₹1.25L` exempt) | Union Budget 2024 Indian equity tax rates applied retroactively cleanly on uniform basis | `update_live.py` default `tax_defensive=False` omits 30% debt tax on ETF switches inside the live equity curve *(though Numba research & stress runs explicitly modeled Sec 50AA `30%` debt tax!)*; dividends ignored (Price-Return bhavcopy equity series underestimate total-return by ~`0.8%–1.2%`/yr dividend yield) |
| **Delisted Stock Pricing (`ffill limit=5`)** | Halted `>5d` falls back to last lot cost basis in MTM | Prevents carried frozen prices after delisting | True bankruptcies (e.g., permanent wipeout before semi-annual exit) are stress-tested separately (`-10%` delisting haircut lowered unhedged `Risk_189` FULL CAGR slightly from `14.97%` to `14.62%` in `numba_deep_research.txt`) |
| **PIT Proxy Universe (`2011–26`)** | `NET_TRDVAL` Ranks `81..230` (`>= ₹20`) | Proxy for NSE Nifty Midcap 150 full-market-cap ranks (`NSE PR` archives omit historical share count / float MCap) | Turnover ranks track large-midcap liquidity well (`772` equities) without survivorship bias, though micro-differences vs free-float MCap index reconstitutions remain |

---

## 10. Autonomous Paper Trading Bot, Forward Tape & Tracked Views (`2026-11-27+`)

1. **Official Autonomous Ledger (`src/paper_bot.py` → `data/paper_ledger.json`):**
   - Executed daily at EOD by `scripts/update_live.py` (CI Mon–Fri `13:20 UTC` / `6:50 PM IST`), tracking **`R3_1L_SlotCap`** (`₹1,00,000`, `SlotCap` ON, `LIQUIDCASE` sweep ON) and **`R3_10L_Standard`** (`₹10,00,000`, unconstrained whole-share sizing, `LIQUIDCASE` sweep ON) alongside forward baselines **`Midcap150_EW`** and **`Unhedged_Risk9M`** from `inception_date`.
   - Enforces strict **`T+1` causality** (signals observed at Day `t-1` close execute at Day `t` closing raw prices; Day-0 inception deploys cleanly at `inception_date`), `126`-trading-day 6M rebalances, daily `MF4` regime transitions (`weakest_first` trims on equity cuts), `TrendGold150` ETF flips (`GOLDBEES` vs `LIQUIDCASE`), `0.15%` one-way brokerage/slippage friction, corporate-action FIFO lot adjustments (`SPLIT` / `BONUS` factor `f` updates `shares /= f, buy_price *= f`), and full Indian FY FIFO capital-gains accounting (`20%` STCG `<365d`, `12.5%` LTCG `>=365d` above `₹1,25,000` annual exemption, STCL/LTCL carry-forwards).
   - **Dual Defensive Tax Transparency:** Reports both **`nav`** (`nav_equity_tax_only`, matching `src/backtester.py`) and **`nav_full_tax`** (additionally deducting estimated Sec 50AA `30%` slab tax on `LIQUIDCASE` gains and `12.5%`/`30%` LTCG/STCG on `GOLDBEES` realizations).
   - **Idempotency:** Re-running on an already processed `as_of_date` leaves `data/paper_ledger.json` and `data/forward_tape.json` byte-identical.
2. **Compact Forward Tape (`data/forward_tape.json`):**
   - Append-only daily JSON log containing closing `prices` (`150` midcaps + `9` ETFs), `corp_actions`, multi-preset `regime` states (`target_eq_mf4`, `target_eq_dd15`, `target_eq_dd75`, `target_eq_dd50`, `active_def_etf`, `ew_index_val`), and per-symbol momentum `rankings` (`rk9`, `so9`, `r9`, `r12`, `a50`).
3. **Private Client-Side Tracked Views (`index.html` → `window.mmiForwardReplay` & `localStorage`):**
   - **1:1 Engine Parity:** Implements a deterministic JavaScript mirror of `src/paper_bot.py` (`stepPortfolio`, `computeTargetWeights`, FIFO lot tax, corporate actions, `330–364d` LTCG Shield deferral with automatic liquidation on Day `365+`, and Frozen `"Do-Nothing"` baselines).
   - **Hand Over to Bot Workflow:** `Rebalance & Tax Diff` accepts `SYMBOL,SHARES,YYYY-MM-DD[,AVG_BUY_PRICE]`, `Available Cash (₹)`, per-preset strategy overrides, `Outside Midcap 150` liquidation/exclusion handling, and spawns persistent Tracked Views in `localStorage` (`mmi_tracked_views_v1`) that auto-replay new tape days until explicitly frozen via **End / Freeze View**, with full JSON Export/Import portability.

