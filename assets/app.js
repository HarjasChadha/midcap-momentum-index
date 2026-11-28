const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtInr = (n) => n == null || !isFinite(n) ? '—' : '₹' + Number(n).toLocaleString('en-IN', { maximumFractionDigits: 2 });
const fmtInrInt = (n) => n == null || !isFinite(n) ? '—' : '₹' + Math.round(Number(n)).toLocaleString('en-IN');
const fmtPct = (n, d = 2) => n == null || !isFinite(n) ? '—' : Number(n).toFixed(d) + '%';
const fmtNum = (n, d = 2) => n == null || !isFinite(n) ? '—' : Number(n).toFixed(d);

let STATE = null;
let PAPER_LEDGER = null;
let FORWARD_TAPE = null;
let CHARTS = {};
let chartWindow = 'ALL';
let chartLog = true;
let triggerFilter = 'ALL';
let rankFilter = 'ALL';
const extraSeries = { s3: false, r2: false, f2: false, u0: false };

let PB_SUBTAB = 'official';
let PB_ACCOUNT = 'R3_1L_SlotCap';
let PB_DEC_FILTER = 'ACTIONS';
let PB_TRADE_FILTER = 'ALL';
let PB_VDEC_FILTER = 'ACTIONS';
let PB_SELECTED_VIEW_ID = null;
let PB_COMPARE_IDS = new Set();
let TRACKED_VIEWS = [];
let OUTSIDE_MODE = 'exclude';
let OUTSIDE_PRICES = {};
let LAST_DIFF_CTX = null;
let PENDING_CONFIRM_FN = null;

const TAB_LABELS = {
  portfolio: 'Action & Orders',
  rebalance: 'Rebalance & Tax Diff',
  paperbot: 'Paper Trading Bot',
  charts: 'Performance & Risk',
  rankings: 'Top 60 Stocks',
  research: 'Strategy & Proof',
};

function switchTab(tabId) {
  document.querySelectorAll('.nav-btn[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + tabId));
  document.getElementById('breadcrumbTitle').textContent = TAB_LABELS[tabId] || 'Portfolio';
  closeMobileSidebar();
  if (tabId === 'charts' && STATE) renderCharts();
  if (tabId === 'rebalance' && STATE) computeHoldingsDiff();
  if (tabId === 'paperbot' && STATE) renderPaperBotPage();
}

function closeMobileSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarBackdrop').classList.remove('open');
}

async function init() {
  const ts = Date.now();
  const [liveRes, ledgerRes, tapeRes] = await Promise.all([
    fetch('./data/live_state.json?_=' + ts),
    fetch('./data/paper_ledger.json?_=' + ts).catch(() => null),
    fetch('./data/forward_tape.json?_=' + ts).catch(() => null),
  ]);
  if (!liveRes.ok) throw new Error('Unable to load market state (HTTP ' + liveRes.status + ')');
  STATE = await liveRes.json();
  if (ledgerRes && ledgerRes.ok) PAPER_LEDGER = await ledgerRes.json();
  if (tapeRes && tapeRes.ok) FORWARD_TAPE = await tapeRes.json();

  loadTrackedViewsFromStorage();
  if (FORWARD_TAPE) syncAllActiveTrackedViews(FORWARD_TAPE);

  renderHeaderAndRegime();
  renderPortfolioCalculator();
  renderPerfSummary();
  renderPitResearchTable();
  renderResearchTable();
  renderRankingsTable();
  renderTriggersTable();

  restoreRebalanceState();
  computeHoldingsDiff();
  updatePbViewsBadge();
  renderPaperBotPage();
}

function renderHeaderAndRegime() {
  const r = STATE.regime;
  const mf = r.multifactor_blocks || {};
  const tg = r.trend_gold_status || {};
  const mood = STATE.nse_mcp_status?.market_mood;

  document.getElementById('asOfDate').textContent = STATE.as_of_date;
  document.getElementById('univCount').textContent = STATE.universe_count;
  document.getElementById('liqPxLbl').textContent = fmtNum(r.etfs.LIQUIDCASE.price, 2);

  document.getElementById('nseStatusText').textContent = STATE.nse_mcp_status?.bulk_quotes_fetched
    ? `NSE live (${STATE.nse_mcp_status.bulk_quotes_fetched} quotes)`
    : 'Cached history';

  const isRiskOn = r.state === 'RISK_ON';
  const isCautious = r.state === 'CAUTIOUS';
  const defPct = Math.max(0, 100 - r.target_equity_pct);
  const activeDef = r.active_defensive_etf || 'LIQUIDCASE';
  const posBlocks = [mf.b1_trend, mf.b2_vol, mf.b3_breadth, mf.b4_macro_gold].filter(v => v != null && v > 0).length;

  const stateEl = document.getElementById('regimeStateLabel');
  stateEl.className = isRiskOn ? 't-green' : (isCautious ? 't-amber' : 't-red');
  stateEl.textContent = isRiskOn
    ? 'Model Posture: RISK-ON (100% Target Equity)'
    : (isCautious ? `Model Posture: CAUTIOUS (${r.target_equity_pct}% Target Equity)` : `Model Posture: DEFENSIVE (${r.target_equity_pct}% Target Equity)`);

  document.getElementById('regimeTitle').textContent = isRiskOn
    ? 'Model Target: 100% Momentum Stock Basket'
    : `Model Target: ${r.target_equity_pct}% Stocks / ${defPct}% ${activeDef}`;

  const goldDir = tg.gold_above_gate
    ? `GOLDBEES trades above EMA150 (+${fmtPct(tg.gold_vs_ema150_pct, 1)}), routing defensive gate to Gold.`
    : `GOLDBEES is ${fmtPct(Math.abs(tg.gold_vs_ema150_pct), 1)} below EMA150, keeping defensive gate in LIQUIDCASE.`;

  document.getElementById('regimeSubtitle').textContent = isRiskOn
    ? `Composite score ${fmtNum(mf.composite_smooth, 2)} remains above -0.10 cautious threshold (${posBlocks} of 4 blocks positive); hysteresis holds RISK-ON. ${goldDir}`
    : `Composite score ${fmtNum(mf.composite_smooth, 2)} (${posBlocks} of 4 blocks positive); weakest-first model trim toward ${activeDef}. ${goldDir}`;

  document.getElementById('lblEq').textContent = `Stocks ${fmtPct(r.target_equity_pct, 0)}`;
  document.getElementById('lblGold').textContent = `Gold ${fmtPct(r.target_goldbees_pct, 0)}`;
  document.getElementById('lblLiq').textContent = `Cash ${fmtPct(r.target_liquidcase_pct, 0)}`;
  document.getElementById('barEq').style.width = `${r.target_equity_pct}%`;
  document.getElementById('barGold').style.width = `${r.target_goldbees_pct}%`;
  document.getElementById('barLiq').style.width = `${r.target_liquidcase_pct}%`;

  document.getElementById('gateSummaryText').textContent =
    `Safe-haven gate: ${activeDef} (GOLDBEES ₹${fmtNum(tg.gold_price)} vs 150-day EMA ₹${fmtNum(tg.gold_ema150)})`;

  const sb = mood?.stock_breadth;
  const advTxt = (sb && sb.up != null && sb.down != null) ? ` · NSE Adv/Dec ${sb.up}/${sb.down}` : '';
  document.getElementById('quickMarketStats').textContent =
    `Midcap 1Y Drawdown: ${fmtPct(r.ew_dd252_pct)} · India VIX: ${fmtNum(r.india_vix, 2)} · Stocks > 50-DMA: ${fmtPct(r.breadth_50_sma10_pct, 1)}${advTxt}`;

  document.getElementById('mfGateDetail').textContent =
    `Active Defensive Asset: ${activeDef} (${fmtPct(tg.gold_vs_ema150_pct, 1)} vs EMA150)`;

  const readTone = (v) => v >= 0.10 ? { cls: 't-green', lbl: 'Healthy' } : v <= -0.15 ? { cls: 't-red', lbl: 'Stressed' } : { cls: 't-amber', lbl: 'Mixed' };
  const blocks = [
    { name: 'Trend & Strength', wt: '44%', val: mf.b1_trend, desc: 'Are midcaps above their 6–10 month trend and beating Nifty 50?' },
    { name: 'Market Breadth', wt: '34%', val: mf.b3_breadth, desc: `Are most midcaps rising together? (${fmtPct(r.breadth_50_sma10_pct, 0)} > 50-DMA)` },
    { name: 'Macro & Gold', wt: '12%', val: mf.b4_macro_gold, desc: 'Checks if downtrends + volatility or surging Gold warn of a crash.' },
    { name: 'Volatility & VIX', wt: '10%', val: mf.b2_vol, desc: `Measures downside chop & fear (VIX ${fmtNum(r.india_vix, 1)}); higher = calmer.` },
    { name: 'Combined Score', wt: 'MF4', val: mf.composite_smooth, desc: `Weighted blend (Raw ${fmtNum(mf.composite_raw, 2)}). Caution <-0.10, Def <-0.25.` },
  ];

  document.getElementById('mfBlocksGrid').innerHTML = blocks.map(b => {
    const v = b.val ?? 0;
    const tone = readTone(v);
    const sign = v >= 0 ? '+' : '';
    return `
      <div class="signal-item">
        <div class="signal-name">${esc(b.name)} <span class="signal-weight">(${esc(b.wt)})</span></div>
        <div class="signal-val ${tone.cls}">${sign}${fmtNum(v, 3)} <span style="font-family:var(--font-body); font-size:12px; font-weight:500;">${tone.lbl}</span></div>
        <div class="signal-desc">${esc(b.desc)}</div>
      </div>`;
  }).join('');
}

function computeBlendWeights(presetKey, capital, useSlotCap) {
  const scoreField = presetKey === 'F4' ? 'r12'
    : presetKey === 'S3' ? 'so9'
    : (presetKey === 'R3' || presetKey === 'R2' || presetKey === 'R7') ? 'rk9' : 'r9';
  const useNormZ = presetKey === 'F7' || presetKey === 'R7';
  const sorted = [...STATE.universe_scores]
    .filter(x => x[scoreField] != null && isFinite(x.px) && x.px > 0)
    .sort((a, b) => b[scoreField] - a[scoreField]);

  const mean = sorted.reduce((s, x) => s + x[scoreField], 0) / Math.max(1, sorted.length);
  const sd = Math.sqrt(sorted.reduce((s, x) => s + (x[scoreField] - mean) ** 2, 0) / Math.max(1, sorted.length - 1)) || 1;
  const normZVal = (raw) => { const z = (raw - mean) / sd; return z > 0 ? 1 + z : 1 / (1 - z); };

  const banned = new Set();
  let chosen = [];
  for (let iter = 0; iter < 60; iter++) {
    const elig = sorted.filter(x => !banned.has(x.sym));
    const top20 = elig.slice(0, 20);
    const top30 = elig.slice(0, 30);
    const nz20 = top20.reduce((s, x) => s + normZVal(x[scoreField]), 0) || 1;
    const nz30 = top30.reduce((s, x) => s + normZVal(x[scoreField]), 0) || 1;

    const wMap = new Map();
    top20.forEach(x => wMap.set(x.sym, (wMap.get(x.sym) || 0) + 0.5 * (useNormZ ? normZVal(x[scoreField]) / nz20 : 1 / top20.length)));
    top30.forEach(x => wMap.set(x.sym, (wMap.get(x.sym) || 0) + 0.5 * (useNormZ ? normZVal(x[scoreField]) / nz30 : 1 / top30.length)));

    chosen = top30.map((x, idx) => ({
      ...x,
      rank: idx + 1,
      score: x[scoreField],
      wt: wMap.get(x.sym) || 0,
      bucket: idx < 20 ? 'Top 20 Leader' : 'Rank 21–30',
    }));
    if (!useSlotCap) break;
    const bad = chosen.find(x => x.px > capital * x.wt);
    if (!bad) break;
    banned.add(bad.sym);
  }
  return { chosen, skipped: [...banned] };
}

function buildComputedOrders(capital, preset, posture, useSlotCap, useSweep) {
  const isTrendGold = preset === 'R3' || preset === 'S3';
  const shiftCrash = preset === 'F4' ? 0.50 : 0.75;
  let eqFrac = 1.0;
  if (posture === 'SIM_CRASH') {
    eqFrac = 1.0 - shiftCrash;
  } else if (posture === 'SIM_CAUTIOUS') {
    eqFrac = 0.60;
  } else if (isTrendGold) {
    eqFrac = (STATE.regime.target_equity_pct != null ? STATE.regime.target_equity_pct : 100.0) / 100.0;
  } else {
    const trig = (preset === 'R2' || preset === 'R7') ? -15.0 : -12.0;
    eqFrac = (STATE.regime.ew_dd252_pct < trig) ? (1.0 - shiftCrash) : 1.0;
  }
  const simCrash = eqFrac < 0.99;
  const defFrac = Math.max(0.0, 1.0 - eqFrac);
  const eqBudget = capital * eqFrac;

  const { chosen, skipped } = computeBlendWeights(preset, eqBudget, useSlotCap);
  let eqDeployed = 0;
  let heldCount = 0;
  const rows = [];
  const activeDefEtf = STATE.regime.active_defensive_etf || 'LIQUIDCASE';

  if (defFrac > 0) {
    const gPx = STATE.regime.etfs.GOLDBEES.price;
    const lPx = STATE.regime.etfs.LIQUIDCASE.price;
    if (isTrendGold) {
      const dPx = activeDefEtf === 'GOLDBEES' ? gPx : lPx;
      const dTgt = capital * defFrac;
      const dSh = Math.floor(dTgt / dPx);
      const dNote = activeDefEtf === 'GOLDBEES'
        ? 'Safe haven — Gold above 150-day trend'
        : 'Safe haven — Cash shield (Gold below 150-day trend)';
      rows.push({ rank: 'D1', sym: activeDefEtf, bucket: 'Safe Haven', tone: 't-amber', score: null, wt: defFrac, px: dPx, tgt: dTgt, sh: dSh, inv: dSh * dPx, note: dNote });
    } else {
      const gTgt = capital * defFrac * 0.5;
      const lTgt = capital * defFrac * 0.5;
      const gSh = Math.floor(gTgt / gPx);
      const lSh = Math.floor(lTgt / lPx);
      rows.push({ rank: 'D1', sym: 'GOLDBEES', bucket: 'Safe Haven 50%', tone: 't-amber', score: null, wt: defFrac * 0.5, px: gPx, tgt: gTgt, sh: gSh, inv: gSh * gPx, note: 'Crash hedge ETF' });
      rows.push({ rank: 'D2', sym: 'LIQUIDCASE', bucket: 'Safe Haven 50%', tone: 't-blue', score: null, wt: defFrac * 0.5, px: lPx, tgt: lTgt, sh: lSh, inv: lSh * lPx, note: 'Cash preservation ETF' });
    }
  }

  chosen.forEach(x => {
    const tgt = eqBudget * x.wt;
    const sh = Math.floor(tgt / x.px);
    const inv = sh * x.px;
    if (sh > 0) heldCount += 1;
    eqDeployed += inv;
    rows.push({
      rank: x.rank,
      sym: x.sym,
      bucket: x.bucket,
      tone: x.rank <= 20 ? 't-green' : 't-muted',
      score: x.score,
      wt: x.wt * eqFrac,
      px: x.px,
      tgt,
      sh,
      inv,
      note: sh === 0 ? 'Price exceeds rupee slot (0 shares)' : (eqFrac < 0.99 && x.rank <= 15 ? 'Core winner retained' : ''),
    });
  });

  const defInvested = rows.filter(r => String(r.rank).startsWith('D')).reduce((s, r) => s + r.inv, 0);
  const remCash = Math.max(0, capital - eqDeployed - defInvested);
  const liqPx = STATE.regime.etfs.LIQUIDCASE.price;
  const sweepSh = useSweep ? Math.floor(remCash / liqPx) : 0;
  const sweepInv = sweepSh * liqPx;
  if (sweepSh > 0) {
    rows.push({ rank: 'Cash', sym: 'LIQUIDCASE', bucket: 'Cash Sweep', tone: 't-blue', score: null, wt: sweepInv / capital, px: liqPx, tgt: remCash, sh: sweepSh, inv: sweepInv, note: 'Parks leftover whole-share rupees' });
  }
  const idleCash = Math.max(0, remCash - sweepInv);
  return { capital, preset, posture, eqFrac, simCrash, isTrendGold, activeDefEtf, rows, eqDeployed, defInvested, sweepInv, idleCash, heldCount, totalSlots: chosen.length, skipped };
}

function getLatestComputedOrders() {
  const capital = Math.max(10000, Number(document.getElementById('capitalInput').value) || 100000);
  const preset = document.getElementById('presetSelect').value;
  const posture = document.getElementById('postureOverride').value;
  const useSlotCap = document.getElementById('slotCapToggle').checked;
  const useSweep = document.getElementById('liqSweepToggle').checked;
  return buildComputedOrders(capital, preset, posture, useSlotCap, useSweep);
}

function renderPortfolioCalculator() {
  const res = getLatestComputedOrders();
  const totalDeployed = res.eqDeployed + res.defInvested + res.sweepInv;
  const defDesc = res.isTrendGold ? res.activeDefEtf : '50/50 Gold & Cash';

  // Simulation notice
  const simNotice = document.getElementById('simNotice');
  if (res.posture !== 'LIVE') {
    simNotice.hidden = false;
    document.getElementById('simNoticeText').textContent = res.posture === 'SIM_CAUTIOUS'
      ? `Simulating Caution mode (${fmtPct(res.eqFrac * 100, 0)} Stocks / ${fmtPct((1 - res.eqFrac) * 100, 0)} ${defDesc}). Switch back to Live Signal before placing real orders.`
      : `Simulating Crash mode (${fmtPct(res.eqFrac * 100, 0)} Stocks / ${fmtPct((1 - res.eqFrac) * 100, 0)} ${defDesc}). Switch back to Live Signal before placing real orders.`;
  } else {
    simNotice.hidden = true;
  }

  const skippedNote = res.skipped.length
    ? `${res.skipped.length} pricey skipped (${res.skipped.slice(0, 3).join(', ')}${res.skipped.length > 3 ? '...' : ''})`
    : `All ${res.totalSlots} top stocks affordable`;

  const cells = [
    { lbl: 'Equity Deployed', val: fmtInrInt(res.eqDeployed), note: `${fmtPct(res.eqDeployed / res.capital * 100, 1)} · ${res.heldCount}/${res.totalSlots} stocks`, cls: 't-green' },
    { lbl: 'Safe-Haven ETFs', val: fmtInrInt(res.defInvested), note: res.simCrash ? `${fmtPct(res.defInvested / res.capital * 100, 1)} in ${defDesc}` : '0% (Full equity posture)', cls: res.defInvested > 0 ? 't-amber' : '' },
    { lbl: 'Cash Sweep (LIQUIDCASE)', val: fmtInrInt(res.sweepInv), note: `${fmtPct(res.sweepInv / res.capital * 100, 1)} leftover rupees parked`, cls: 't-blue' },
    { lbl: 'Idle Cash & SlotCap', val: fmtInrInt(res.idleCash), note: `${fmtPct(totalDeployed / res.capital * 100, 2)} utilized · ${skippedNote}`, cls: '' },
  ];

  document.getElementById('calcSummaryGrid').innerHTML = cells.map(c => `
    <div class="metric-cell">
      <div class="metric-label">${esc(c.lbl)}</div>
      <div class="metric-value ${c.cls}">${esc(c.val)}</div>
      <div class="metric-note">${esc(c.note)}</div>
    </div>`).join('');

  document.getElementById('ordersMetaLine').textContent =
    `Preset ${res.preset} · ${res.rows.length} order lines · ${fmtPct(totalDeployed / res.capital * 100, 1)} of ${fmtInrInt(res.capital)} deployed`;

  const isRatioPreset = res.preset === 'R3' || res.preset === 'S3' || res.preset === 'R2' || res.preset === 'R7';
  document.querySelector('#ordersTable tbody').innerHTML = res.rows.map(r => `
    <tr class="${r.sh === 0 ? 'row-muted' : ''}">
      <td class="t-muted" style="font-family:var(--font-mono); font-size:13px;">${esc(r.rank)}</td>
      <td class="asset-name">${esc(r.sym)}</td>
      <td class="role-text ${r.tone}">${esc(r.bucket)}</td>
      <td class="num col-detail">${r.score != null ? (isRatioPreset ? fmtNum(r.score, 2) + 'x' : fmtPct(r.score)) : '—'}</td>
      <td class="num">${fmtPct(r.wt * 100, 2)}</td>
      <td class="num">${fmtNum(r.px)}</td>
      <td class="num col-detail">${fmtInr(r.tgt)}</td>
      <td class="num"><strong>${r.sh}</strong></td>
      <td class="num">${fmtInr(r.inv)}</td>
      <td class="note-text">${esc(r.note)}</td>
    </tr>`).join('');

  updateDiffContextLabel();
}

function buildUniversePriceMap() {
  const pxMap = new Map();
  const lastTapePx = FORWARD_TAPE?.days?.length ? FORWARD_TAPE.days[FORWARD_TAPE.days.length - 1].prices || {} : {};
  Object.entries(lastTapePx).forEach(([k, v]) => { if (isFinite(v) && v > 0) pxMap.set(k, Number(v)); });
  (STATE?.universe_scores || []).forEach(u => { if (isFinite(u.px) && u.px > 0) pxMap.set(u.sym, Number(u.px)); });
  Object.entries(STATE?.regime?.etfs || {}).forEach(([k, v]) => { if (v?.price > 0) pxMap.set(k, Number(v.price)); });
  return pxMap;
}

function saveRebalanceState() {
  try {
    const payload = {
      holdingsCsv: document.getElementById('holdingsInput').value,
      cash: Number(document.getElementById('rebalanceCashInput').value) || 0,
      preset: document.getElementById('rebalancePresetSelect').value,
      slotCap: document.getElementById('rebalanceSlotCapToggle').checked,
      sweep: document.getElementById('rebalanceSweepToggle').checked,
      outsideMode: OUTSIDE_MODE,
      outsidePrices: OUTSIDE_PRICES,
    };
    localStorage.setItem('mmi_rebalance_state_v1', JSON.stringify(payload));
    localStorage.setItem('mmi_holdings_csv', payload.holdingsCsv.trim());
  } catch {}
}

function restoreRebalanceState() {
  try {
    const raw = localStorage.getItem('mmi_rebalance_state_v1');
    if (raw) {
      const s = JSON.parse(raw);
      if (s.holdingsCsv != null) document.getElementById('holdingsInput').value = s.holdingsCsv;
      if (s.cash != null) document.getElementById('rebalanceCashInput').value = s.cash;
      if (s.preset) document.getElementById('rebalancePresetSelect').value = s.preset;
      if (s.slotCap != null) document.getElementById('rebalanceSlotCapToggle').checked = !!s.slotCap;
      if (s.sweep != null) document.getElementById('rebalanceSweepToggle').checked = !!s.sweep;
      if (s.outsideMode === 'liquidate' || s.outsideMode === 'exclude') {
        OUTSIDE_MODE = s.outsideMode;
        document.querySelectorAll('[data-outside-mode]').forEach(b => b.classList.toggle('active', b.dataset.outsideMode === OUTSIDE_MODE));
      }
      if (s.outsidePrices && typeof s.outsidePrices === 'object') OUTSIDE_PRICES = s.outsidePrices;
      return;
    }
  } catch {}
  const saved = localStorage.getItem('mmi_holdings_csv');
  if (saved) document.getElementById('holdingsInput').value = saved;
}

function updateDiffContextLabel() {
  if (!STATE) return;
  const preset = document.getElementById('rebalancePresetSelect').value;
  if (LAST_DIFF_CTX && LAST_DIFF_CTX.targetPortfolioCap > 0) {
    document.getElementById('diffTargetContext').textContent =
      `${preset} target for ${fmtInrInt(LAST_DIFF_CTX.targetPortfolioCap)} (${fmtInrInt(LAST_DIFF_CTX.inUnivVal + LAST_DIFF_CTX.liquidateOutVal)} holdings + ${fmtInrInt(LAST_DIFF_CTX.cashAvail)} cash)`;
  } else {
    document.getElementById('diffTargetContext').textContent = `Comparing against ${preset} strategy rules`;
  }
}

function renderOutsideStocksPanel(outsideMap, outsideMode) {
  const box = document.getElementById('outsideUniverseBox');
  const listEl = document.getElementById('outsideStockList');
  const syms = [...outsideMap.keys()];
  if (!syms.length) {
    box.hidden = true;
    listEl.innerHTML = '';
    return;
  }
  box.hidden = false;
  document.getElementById('outsideCountLbl').textContent = String(syms.length);
  listEl.innerHTML = syms.map(sym => {
    const info = outsideMap.get(sym);
    const estPx = OUTSIDE_PRICES[sym] != null ? OUTSIDE_PRICES[sym] : (info.avgBuy || 0);
    const estVal = info.shares * (Number(estPx) || 0);
    return `
      <div class="outside-row">
        <div>
          <span class="asset-name">${esc(sym)}</span>
          <span class="badge badge-amber" style="margin-left:6px;">Outside Midcap 150</span>
          <div style="font-size:12px; opacity:0.85;">${info.shares} sh${info.oldestDays != null ? ' · ' + info.oldestDays + 'd held' : ''}</div>
        </div>
        ${outsideMode === 'liquidate' ? `
          <div style="display:flex; align-items:center; gap:8px;">
            <label style="font-size:12px; font-weight:600;">Sell Px (&#8377;):</label>
            <input class="input-box mono outside-px-inp" data-outside-sym="${esc(sym)}" type="number" min="0" step="0.05" value="${esc(estPx || '')}" placeholder="Current ₹" style="width:110px; min-height:32px; padding:4px 8px;" />
            <span class="t-muted" style="font-family:var(--font-mono); font-size:12px;">= ${fmtInrInt(estVal)}</span>
          </div>` : `<div style="font-size:12px; font-weight:600;">Kept separate outside sleeve</div>`}
      </div>`;
  }).join('');

  listEl.querySelectorAll('.outside-px-inp').forEach(inp => {
    inp.addEventListener('input', () => {
      OUTSIDE_PRICES[inp.dataset.outsideSym] = Math.max(0, Number(inp.value) || 0);
      computeHoldingsDiff();
    });
  });
}

function computeHoldingsDiff() {
  saveRebalanceState();
  const txt = document.getElementById('holdingsInput').value.trim();
  const cashAvail = Math.max(0, Number(document.getElementById('rebalanceCashInput').value) || 0);
  const rbPreset = document.getElementById('rebalancePresetSelect').value;
  const rbSlotCap = document.getElementById('rebalanceSlotCapToggle').checked;
  const rbSweep = document.getElementById('rebalanceSweepToggle').checked;

  const pxMap = buildUniversePriceMap();
  const etfSet = new Set(['GOLDBEES', 'SILVERBEES', 'EBBETF0431', 'LTGILTBEES', 'LIQUIDCASE', 'LIQUIDBEES', 'NIFTYBEES', 'MON100', 'MIDCAPIETF']);
  const scoreMap = new Map();
  const rankMap = new Map();
  const scField = rbPreset === 'F4' ? 'r12' : rbPreset === 'S3' ? 'so9' : (rbPreset === 'R3' || rbPreset === 'R2' || rbPreset === 'R7') ? 'rk9' : 'r9';
  const sortedU = [...(STATE.universe_scores || [])]
    .filter(u => u[scField] != null)
    .sort((a, b) => (b[scField] - a[scField]) || (a.sym < b.sym ? -1 : 1));
  sortedU.forEach((u, i) => { scoreMap.set(u.sym, u[scField]); rankMap.set(u.sym, i + 1); });

  const asOfMs = Date.parse(STATE.as_of_date + 'T00:00:00Z');
  const inUnivLots = [];
  const heldBySym = new Map();
  const outsideMap = new Map();

  txt.split(/\r?\n/).forEach(line => {
    const clean = line.trim();
    if (!clean || clean.startsWith('#')) return;
    const parts = clean.split(',').map(s => s.trim());
    const sym = (parts[0] || '').toUpperCase();
    const sh = parseFloat(parts[1]);
    if (!sym || !isFinite(sh) || sh <= 0) return;
    const dtStr = (parts[2] || '').slice(0, 10);
    const dtMs = dtStr ? Date.parse(dtStr + 'T00:00:00Z') : NaN;
    const daysHeld = !isNaN(dtMs) ? Math.round((asOfMs - dtMs) / 86400000) : null;
    const costInp = parts[3] != null && parts[3] !== '' ? parseFloat(parts[3]) : NaN;
    const hasCostBasis = isFinite(costInp) && costInp > 0;

    if (pxMap.has(sym)) {
      const curPx = pxMap.get(sym);
      const buyPx = hasCostBasis ? costInp : curPx;
      const buyDate = !isNaN(dtMs) ? dtStr : STATE.as_of_date;
      inUnivLots.push({ symbol: sym, shares: sh, buy_date: buyDate, buy_price: buyPx, daysHeld, hasCostBasis });
      const rec = heldBySym.get(sym) || { shares: 0, lots: [], oldestDays: daysHeld };
      rec.shares += sh;
      rec.lots.push({ shares: sh, buy_date: buyDate, buy_price: buyPx, daysHeld, hasCostBasis });
      if (daysHeld != null && (rec.oldestDays == null || daysHeld > rec.oldestDays)) rec.oldestDays = daysHeld;
      heldBySym.set(sym, rec);
    } else {
      const rec = outsideMap.get(sym) || { sym, shares: 0, oldestDays: daysHeld, avgBuy: hasCostBasis ? costInp : 0, buyDate: !isNaN(dtMs) ? dtStr : STATE.as_of_date };
      rec.shares += sh;
      if (hasCostBasis && !rec.avgBuy) rec.avgBuy = costInp;
      if (daysHeld != null && (rec.oldestDays == null || daysHeld > rec.oldestDays)) rec.oldestDays = daysHeld;
      outsideMap.set(sym, rec);
    }
  });

  renderOutsideStocksPanel(outsideMap, OUTSIDE_MODE);

  let inUnivVal = 0;
  heldBySym.forEach((rec, sym) => { inUnivVal += rec.shares * (pxMap.get(sym) || 0); });
  let outsideTotalVal = 0;
  const outsideHoldings = [];
  outsideMap.forEach((rec, sym) => {
    const estPx = OUTSIDE_PRICES[sym] != null ? Number(OUTSIDE_PRICES[sym]) : (rec.avgBuy || 0);
    const val = rec.shares * Math.max(0, estPx || 0);
    outsideTotalVal += val;
    outsideHoldings.push({ symbol: sym, shares: rec.shares, buy_date: rec.buyDate, est_price: estPx || 0, value: val });
  });
  const liquidateOutVal = OUTSIDE_MODE === 'liquidate' ? outsideTotalVal : 0;
  const totalWithCash = inUnivVal + liquidateOutVal + cashAvail;

  if (!txt && cashAvail <= 0) {
    LAST_DIFF_CTX = null;
    document.getElementById('diffPortTotal').textContent = '₹0';
    document.getElementById('diffPortBreak').textContent = 'Holdings + Cash';
    document.getElementById('diffBuyTotal').textContent = '₹0';
    document.getElementById('diffBuyCount').textContent = '0 trades';
    document.getElementById('diffSellTotal').textContent = '₹0';
    document.getElementById('diffSellCount').textContent = '0 trades';
    document.getElementById('diffTaxFlags').textContent = '0';
    document.getElementById('diffTaxEstInr').textContent = '₹0';
    document.getElementById('diffFeeEstText').textContent = 'Friction (0.15%): ₹0';
    document.getElementById('diffTaxBreakdownText').innerHTML = 'Enter optional 4th column <code>AVG_BUY_PRICE</code> for exact STCG/LTCG rupee tax estimate.';
    document.getElementById('diffOutput').innerHTML = '<div class="empty-state"><p>Paste your current holdings as <code>SYMBOL,SHARES,YYYY-MM-DD[,AVG_BUY_PRICE]</code> or enter cash above.</p></div>';
    updateDiffContextLabel();
    return;
  }

  const targetCapital = Math.max(1000, totalWithCash);
  const res = buildComputedOrders(targetCapital, rbPreset, 'LIVE', rbSlotCap, rbSweep);
  const tgtMap = new Map();
  res.rows.forEach(r => tgtMap.set(r.sym, (tgtMap.get(r.sym) || 0) + r.sh));

  const allSyms = new Set([...heldBySym.keys(), ...tgtMap.keys()]);
  const actions = [];
  let buySum = 0, buyCnt = 0, sellSum = 0, sellCnt = 0, ltcgWarnCnt = 0;
  let stcgGain = 0, ltcgGain = 0, missingBasisCnt = 0;

  allSyms.forEach(sym => {
    const rec = heldBySym.get(sym);
    const cur = rec ? rec.shares : 0;
    const days = rec ? rec.oldestDays : null;
    const tgt = tgtMap.get(sym) || 0;
    const px = pxMap.get(sym) || 0;
    const delta = tgt - cur;
    const deltaInr = Math.abs(delta) * px;
    let act = 'HOLD';
    let tone = 't-muted';
    let note = '';
    let isTaxWarn = false;

    if (delta > 0 && deltaInr >= 500) {
      act = `BUY +${Number(delta.toFixed(2))}`;
      tone = 't-green';
      buySum += deltaInr;
      buyCnt += 1;
      const rk = rankMap.get(sym);
      note = rk ? `${rbPreset} Rank #${rk} target buy` : `Defensive allocation (${sym})`;
    } else if (delta < 0 && (tgt === 0 || deltaInr >= 500)) {
      const sellQty = Math.abs(delta);
      act = `SELL -${Number(sellQty.toFixed(2))}`;
      tone = 't-red';
      sellSum += deltaInr;
      sellCnt += 1;
      // FIFO lot gain / LTCG shield inspection
      let rem = sellQty;
      const lotsSorted = [...(rec?.lots || [])].sort((a, b) => (a.buy_date < b.buy_date ? -1 : 1));
      lotsSorted.forEach(lot => {
        if (rem <= 1e-6) return;
        const take = Math.min(lot.shares, rem);
        rem -= take;
        if (!lot.hasCostBasis) missingBasisCnt += 1;
        const gain = (px - lot.buy_price) * take;
        const dH = lot.daysHeld ?? 0;
        if (dH >= 330 && dH < 365) isTaxWarn = true;
        if (!etfSet.has(sym)) {
          if (dH >= 365) ltcgGain += gain;
          else stcgGain += gain;
        }
      });
      if (isTaxWarn) {
        ltcgWarnCnt += 1;
        note = `Held ${days}d — LTCG in ${365 - days}d (330–364d shield)`;
      } else {
        const rk = rankMap.get(sym);
        const wkNote = etfSet.has(sym) ? 'Defensive rebalance cut' : (rk ? `Rank #${rk} trim` : 'Outside Top 30 (weakest-first exit)');
        note = days != null ? `${wkNote} · ${days}d (${days >= 365 ? 'LTCG 12.5%' : 'STCG 20%'})` : wkNote;
      }
    } else if (delta !== 0) {
      note = `Skipped — below ₹500 min delta (${fmtInrInt(deltaInr)})`;
    } else if (cur > 0) {
      const rk = rankMap.get(sym);
      note = `${rk ? 'Top-30 #' + rk + ' keep' : 'Target match'}${days != null ? ' · ' + days + 'd held' : ''}`;
    }
    actions.push({ sym, cur, tgt, delta, deltaInr, act, tone, note, isTaxWarn, sortPri: act.startsWith('SELL') ? 0 : act.startsWith('BUY') ? 1 : 2 });
  });

  // Add Outside Midcap 150 stocks cleanly into table too
  outsideHoldings.forEach(oh => {
    const val = oh.value;
    if (OUTSIDE_MODE === 'liquidate') {
      sellSum += val;
      sellCnt += 1;
      actions.push({
        sym: oh.symbol,
        cur: oh.shares,
        tgt: 0,
        delta: -oh.shares,
        deltaInr: val,
        act: `SELL -${oh.shares}`,
        tone: 't-red',
        note: `Outside Midcap 150 -> liquidate into strategy cash${val <= 0 ? ' (enter Sell Px)' : ''}`,
        isTaxWarn: false,
        sortPri: 0,
      });
    } else {
      actions.push({
        sym: oh.symbol,
        cur: oh.shares,
        tgt: oh.shares,
        delta: 0,
        deltaInr: val,
        act: 'OUTSIDE M150',
        tone: 't-amber',
        note: 'Outside Midcap 150 — kept untouched in external sleeve',
        isTaxWarn: false,
        sortPri: 3,
      });
    }
  });

  actions.sort((a, b) => a.sortPri - b.sortPri || b.deltaInr - a.deltaInr);

  const stcgTax = Math.max(0, stcgGain) * 0.20;
  const ltcgTax = Math.max(0, ltcgGain - 125000) * 0.125;
  const totalTaxEst = stcgTax + ltcgTax;
  const frictionEst = (buySum + sellSum) * 0.0015;

  document.getElementById('diffPortTotal').textContent = fmtInrInt(totalWithCash);
  document.getElementById('diffPortBreak').textContent = `${heldBySym.size} univ · ${fmtInrInt(cashAvail)} cash`;
  document.getElementById('diffBuyTotal').textContent = fmtInrInt(buySum);
  document.getElementById('diffBuyCount').textContent = `${buyCnt} buy ${buyCnt === 1 ? 'order' : 'orders'}`;
  document.getElementById('diffSellTotal').textContent = fmtInrInt(sellSum);
  document.getElementById('diffSellCount').textContent = `${sellCnt} sell ${sellCnt === 1 ? 'order' : 'orders'}`;
  document.getElementById('diffTaxFlags').textContent = String(ltcgWarnCnt);
  document.getElementById('diffTaxEstInr').textContent = fmtInrInt(totalTaxEst);
  document.getElementById('diffTaxEstInr').className = totalTaxEst > 0 ? 't-red' : 't-green';
  document.getElementById('diffFeeEstText').textContent = `Friction (0.15%): ${fmtInrInt(frictionEst)}`;
  document.getElementById('diffTaxBreakdownText').innerHTML =
    `STCG net ${fmtInrInt(stcgGain)} (tax ${fmtInrInt(stcgTax)}) · LTCG net ${fmtInrInt(ltcgGain)} (tax ${fmtInrInt(ltcgTax)} after ₹1.25L exempt)` +
    (missingBasisCnt > 0 ? ' · <em>Cost omitted on some lots (LTP assumed)</em>' : '');

  LAST_DIFF_CTX = {
    preset: rbPreset,
    slotCap: rbSlotCap,
    sweep: rbSweep,
    inUnivLots,
    inUnivVal,
    cashAvail,
    outsideMode: OUTSIDE_MODE,
    outsideHoldings,
    liquidateOutVal,
    targetPortfolioCap: totalWithCash,
    ltcgWarnCnt,
  };
  updateDiffContextLabel();

  document.getElementById('diffOutput').innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th class="num">Held</th>
          <th class="num">Target</th>
          <th>Action</th>
          <th class="num">Trade (&#8377;)</th>
          <th>Tax &amp; Timing</th>
        </tr>
      </thead>
      <tbody>
        ${actions.map(a => `
          <tr>
            <td class="asset-name">${esc(a.sym)}</td>
            <td class="num">${a.cur}</td>
            <td class="num">${a.tgt}</td>
            <td class="${a.tone}" style="font-weight:600; white-space:nowrap;">${esc(a.act)}</td>
            <td class="num">${fmtInrInt(a.deltaInr)}</td>
            <td class="${a.isTaxWarn ? 't-amber' : 'note-text'}" style="${a.isTaxWarn ? 'font-weight:600;' : ''}">${esc(a.note)}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

function mkTipHtml(label, tipText) {
  return `${esc(label)} <span class="tooltip-q" tabindex="0" data-tip="${esc(tipText)}">?</span>`;
}

function renderPerfSummary() {
  const src16 = STATE.research_v3?.src_exact_16yr || {};
  const row16 = (id) => (src16.rows || []).find(r => r.lbl === id) || {};
  const bench16 = (id) => (src16.benchmarks || []).find(b => b.lbl === id) || {};
  const r3_16 = row16('R3_src_Frac');
  const r3_stag = row16('R3_Staggered_2x6M');
  const r3_2x = row16('R3_Stress_30bps_2xCost');
  const m100_16 = bench16('NIFTY_MIDCAP_100');
  const ew_16 = bench16('PIT_EW_Benchmark');
  const phR3 = src16.phase_spread?.R3 || {};
  const phStag = src16.phase_spread?.R3_Staggered_2x6M || {};

  const pitEl = document.getElementById('pitPerfSummaryGrid');
  if (pitEl) {
    const pitItems = [
      {
        lbl: 'R3 16Y Zero-Surv. CAGR',
        tip: 'Exact Python FIFO annualized return (2011–2026) over 772 clean PIT midcaps, net of 20% STCG, 12.5% LTCG, Sec 50AA 30% debt/Gold tax, and 15/35 bps slippage.',
        val: r3_16.FULL_cagr != null ? fmtPct(r3_16.FULL_cagr) : '—',
        note: `6-phase mean ${phR3.mean != null ? fmtPct(phR3.mean) : '—'} [${phR3.min ?? '—'}%–${phR3.max ?? '—'}%] · Staggered 2x6M: ${r3_stag.FULL_cagr != null ? fmtPct(r3_stag.FULL_cagr) : '—'} (${phStag.spread ?? '—'}% spread)`,
        cls: 't-green',
      },
      {
        lbl: '16Y Max DD & Sharpe',
        tip: 'Worst peak-to-trough portfolio loss and excess return above 6.25% RBI T-Bill divided by annualized volatility.',
        val: `${fmtPct(r3_16.FULL_dd)} / ${fmtNum(r3_16.FULL_sh)}`,
        note: `Calmar ${fmtNum(r3_16.FULL_cal)} · 2x Friction Stress (30 bps): ${fmtPct(r3_2x.FULL_cagr)} / ${fmtNum(r3_2x.FULL_sh)} Sh`,
        cls: 't-blue',
      },
      {
        lbl: 'Official Midcap 100 (16Y)',
        tip: 'Official NSE Nifty Midcap 100 Index over full 2011–2026 history, compared alongside our 772-stock PIT EW proxy.',
        val: m100_16.FULL_cagr != null ? fmtPct(m100_16.FULL_cagr) : '—',
        note: `Max DD ${fmtPct(m100_16.FULL_dd)} · Sharpe ${fmtNum(m100_16.FULL_sh)} (PIT EW: ${fmtPct(ew_16.FULL_cagr)} / ${fmtPct(ew_16.FULL_dd)})`,
        cls: '',
      },
      {
        lbl: '₹1L SlotCap (16Y Realistic)',
        tip: 'Discrete ₹1L SlotCap trails unconstrained R3 over 16 clean PIT years; its slight 2018–26 edge is small-sample luck.',
        val: row16('R3_src_Disc1L').FULL_cagr != null ? fmtPct(row16('R3_src_Disc1L').FULL_cagr) : '—',
        note: `DD ${fmtPct(row16('R3_src_Disc1L').FULL_dd)} · Post-Sec50AA vs pre-def-tax ${fmtPct(row16('R3_PreDefTax_Comparison').FULL_cagr)}`,
        cls: 't-amber',
      },
    ];
    pitEl.innerHTML = pitItems.map(i => `
      <div class="metric-cell">
        <div class="metric-label">${mkTipHtml(i.lbl, i.tip)}</div>
        <div class="metric-value ${i.cls}">${esc(i.val)}</div>
        <div class="metric-note">${esc(i.note)}</div>
      </div>`).join('');
  }

  const psum = STATE.performance_summary || {};
  const r3 = psum.R3_Risk9m_6M_MF4_TrendGold150 || {};
  const r3d = psum.R3_Disc1L_SlotCap || {};
  const ew = psum.Midcap150_EW_Universe || {};
  const m150 = psum.NIFTY_MIDCAP_150 || {};
  const s3 = psum.S3_Sortino9m_6M_MF4_TrendGold150 || {};

  const items = [
    {
      lbl: 'R3 Panel CAGR (2018–26 Biased)',
      tip: 'Compound Annual Growth Rate over 2018–2026 holding today\'s 150 members (survivorship-biased; see 16Y PIT strip above).',
      val: fmtPct(r3.cagr_pct),
      note: `₹1L SlotCap: ${fmtPct(r3d.cagr_pct)} (small-sample luck) · Staggered 2x6M: ${fmtPct((psum.R3_Staggered_2x6M || {}).cagr_pct)}`,
      cls: 't-green',
    },
    {
      lbl: 'Universe EW & Midcap 150',
      tip: 'Passive equal-weight basket of current 150 members (2018–26) alongside Official NSE Nifty Midcap 150 since its Jan-2019 launch.',
      val: fmtPct(ew.cagr_pct),
      note: `₹1L → ${fmtInrInt(ew.final_value)} · Official Midcap 150 (19–26): ${fmtPct(m150.cagr_pct)} (${fmtPct(m150.max_dd_pct)} DD)`,
      cls: '',
    },
    {
      lbl: 'R3 Max Drawdown',
      tip: 'Largest peak-to-bottom decline on 2018–26 panel; net of Sec 50AA debt/Gold taxes and 35 bps risk-off impact.',
      val: fmtPct(r3.max_dd_pct),
      note: `Calmar ${fmtNum(r3.calmar)} · Sharpe ${fmtNum(r3.sharpe)} · 30 bps stress ${fmtPct((psum.R3_Stress_30bps_2xCost || {}).cagr_pct)}`,
      cls: 't-blue',
    },
    {
      lbl: 'S3 Sortino Runner-Up',
      tip: 'Sortino 9M divides 9M return by downside volatility only (6% floor); Calmar = CAGR divided by Max DD magnitude.',
      val: fmtPct(s3.cagr_pct),
      note: `Max DD ${fmtPct(s3.max_dd_pct)} · Sharpe ${fmtNum(s3.sharpe)}`,
      cls: 't-green',
    },
  ];

  document.getElementById('perfSummaryGrid').innerHTML = items.map(i => `
    <div class="metric-cell">
      <div class="metric-label">${mkTipHtml(i.lbl, i.tip)}</div>
      <div class="metric-value ${i.cls}">${esc(i.val)}</div>
      <div class="metric-note">${esc(i.note)}</div>
    </div>`).join('');
}

function renderPitResearchTable() {
  const rows = STATE.pit_validation_16yr || [];
  const tbody = document.querySelector('#pitResearchTable tbody');
  if (tbody) {
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td class="${(r.universe || '').includes('PIT') ? 't-green' : 't-muted'}" style="font-size:13px; white-space:nowrap;">${esc(r.universe)}</td>
        <td class="asset-name">${esc(r.strategy)}</td>
        <td class="num">${r.p1_cagr != null ? fmtPct(r.p1_cagr) : '—'}</td>
        <td class="num">${r.p2_cagr != null ? fmtPct(r.p2_cagr) : '—'}</td>
        <td class="num">${r.p3_cagr != null ? fmtPct(r.p3_cagr) : '—'}</td>
        <td class="num t-green"><strong>${r.full_cagr != null ? fmtPct(r.full_cagr) : '—'}</strong></td>
        <td class="num t-red">${r.full_dd != null ? fmtPct(r.full_dd) : '—'}</td>
        <td class="num">${r.full_sh != null ? fmtNum(r.full_sh) : '—'}</td>
      </tr>`).join('');
  }

  const v3 = STATE.research_v3 || {};
  const src16 = v3.src_exact_16yr || {};
  const r3_16 = (src16.rows || []).find(r => r.lbl === 'R3_src_Frac') || {};
  const f2_16 = (src16.rows || []).find(r => r.lbl === 'F2_src_Frac') || {};
  const m100_16 = (src16.benchmarks || []).find(b => b.lbl === 'NIFTY_MIDCAP_100') || {};
  const ew_16 = (src16.benchmarks || []).find(b => b.lbl === 'PIT_EW_Benchmark') || {};

  if (r3_16.FULL_cagr != null) {
    const elC = document.getElementById('proof16yCagr');
    const elCN = document.getElementById('proof16yCagrNote');
    const elD = document.getElementById('proof16yDd');
    const elDN = document.getElementById('proof16yDdNote');
    const elS = document.getElementById('proof16ySh');
    const elSN = document.getElementById('proof16yShNote');
    const elSub = document.getElementById('proof16ySub');
    const elSubN = document.getElementById('proof16ySubNote');
    if (elC) elC.textContent = fmtPct(r3_16.FULL_cagr);
    if (elCN) elCN.textContent = `Net Sec 50AA + 35 bps vs ${fmtPct(m100_16.FULL_cagr)} Midcap 100 (${fmtPct(ew_16.FULL_cagr)} PIT EW)`;
    if (elD) elD.textContent = fmtPct(r3_16.FULL_dd);
    if (elDN) elDN.textContent = `Peak-to-trough vs ${fmtPct(m100_16.FULL_dd)} Midcap 100 (${fmtPct(ew_16.FULL_dd)} PIT EW)`;
    if (elS) elS.textContent = fmtNum(r3_16.FULL_sh);
    if (elSN) elSN.textContent = `Net return per unit risk vs ${fmtNum(f2_16.FULL_sh)} legacy F2`;
    if (elSub) elSub.textContent = `${fmtNum(r3_16.P1_cagr, 1)}% → ${fmtNum(r3_16.P2_cagr, 1)}% → ${fmtNum(r3_16.P3_cagr, 1)}%`;
    if (elSubN) elSubN.textContent = `P1 Train · P2 OOS · P3 Post-Selection (Sec 50AA ON)`;
  }

  const cTbody = document.querySelector('#crashAnatomyTable tbody');
  if (cTbody && v3.crash_anatomy) {
    cTbody.innerHTML = v3.crash_anatomy.map(c => `
      <tr>
        <td class="asset-name">${esc(c.defensive_strategy)}</td>
        <td class="num">${fmtPct(c['2011_Bear_dd'], 1)}</td>
        <td class="num">${fmtPct(c['2018_19_NBFC_ret'], 1)}</td>
        <td class="num">${fmtPct(c['2020_COVID_dd'], 1)}</td>
        <td class="num">${fmtPct(c['2022_RateHike_ret'], 1)}</td>
        <td class="num">${fmtPct(c['2024_25_MidBear_dd'], 1)}</td>
        <td class="num ${(c['2026_GoldSlump_ret'] ?? 0) >= 0 ? 't-green' : 't-red'}"><strong>${fmtPct(c['2026_GoldSlump_ret'], 1)}</strong></td>
      </tr>`).join('');
  }

  const fTbody = document.querySelector('#fnoCompareTable tbody');
  if (fTbody && v3.fno_hedges) {
    const fRows = [
      ...v3.fno_hedges.map(f => ({
        name: f.name,
        cagr: f.FULL_cagr,
        dd: f.FULL_dd,
        sh: f.FULL_sh,
        note: f.name.includes('ShortFut')
          ? 'Needs ₹25L–₹1Cr+ (₹16.5L lot min, 0.05% STT, 30% tax)'
          : 'VIX-scaled theta bleed + 30% F&O slab tax',
      })),
      {
        name: 'R3 Spot + MF4 + TrendGold150 (Exact FIFO, Net Sec 50AA)',
        cagr: r3_16.FULL_cagr,
        dd: r3_16.FULL_dd,
        sh: r3_16.FULL_sh,
        note: 'Optimal from ₹1L+ (35 bps risk-off impact, Sec 50AA 30% debt & 12.5% Gold tax modeled)',
      },
    ];
    fTbody.innerHTML = fRows.map(f => `
      <tr>
        <td class="asset-name">${esc(f.name)}</td>
        <td class="num t-green"><strong>${fmtPct(f.cagr)}</strong></td>
        <td class="num t-red">${fmtPct(f.dd)}</td>
        <td class="num">${fmtNum(f.sh)}</td>
        <td class="note-text">${esc(f.note)}</td>
      </tr>`).join('');
  }
}

function renderResearchTable() {
  const rows = STATE.clean_verification || [];
  document.querySelector('#researchTable tbody').innerHTML = rows.map(r => `
    <tr>
      <td class="${(r.scenario.includes('R3') || r.scenario.includes('S3')) ? 't-green' : 't-muted'}" style="font-size:13px; font-weight:600; white-space:nowrap;">${esc(r.scenario)}</td>
      <td class="asset-name">${esc(r.name)}</td>
      <td class="num"><strong>${fmtPct(r.cagr_pct)}</strong></td>
      <td class="num t-red">${fmtPct(r.max_dd_pct)}</td>
      <td class="num">${fmtNum(r.calmar)}</td>
      <td class="num">${fmtNum(r.sharpe)}</td>
      <td class="num">${fmtNum(r.sortino)}</td>
      <td class="num">${fmtPct(r.ann_turnover_pct, 1)}</td>
      <td class="num">${r.avg_eq_pct != null ? fmtPct(r.avg_eq_pct, 1) : '—'}</td>
      <td class="num">${r.total_tax_paid != null ? fmtInrInt(r.total_tax_paid) : '—'}</td>
    </tr>`).join('');
}

function renderRankingsTable() {
  const q = (document.getElementById('rankSearch').value || '').trim().toUpperCase();
  const rows = STATE.rankings_top60.filter(r => {
    if (q && !r.symbol.includes(q)) return false;
    const rk = r.rank_risk9m ?? r.rank_9m ?? 999;
    const aff = !!(r.risk_affordable_slotcap_1l ?? r.affordable_slotcap_1l);
    if (rankFilter === 'TOP30' && rk > 30) return false;
    if (rankFilter === 'AFF1L' && !aff) return false;
    if (rankFilter === 'MA50' && !r.above_ma50) return false;
    return true;
  });

  const tbody = document.querySelector('#rankingsTable tbody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="14" style="text-align:center; padding:28px; color:var(--ink-muted);">No stocks match this search or filter.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const aff = !!(r.risk_affordable_slotcap_1l ?? r.affordable_slotcap_1l);
    const ewWt = r.risk_ew_5050_wt_pct ?? r.ew_5050_wt_pct;
    const nzWt = r.risk_nz_5050_wt_pct ?? r.nz_5050_wt_pct;
    const ldBadge = r.new_listing ? ' <span class="badge badge-amber">New Listing</span>' : '';
    return `
      <tr>
        <td style="font-family:var(--font-mono); font-weight:600;">#${r.rank_risk9m ?? r.rank_9m}</td>
        <td class="col-detail t-muted" style="font-family:var(--font-mono);">${r.rank_sortino9m != null ? '#' + r.rank_sortino9m : '—'}</td>
        <td class="col-detail t-muted" style="font-family:var(--font-mono);">${r.rank_9m != null ? '#' + r.rank_9m : '—'}</td>
        <td class="col-detail t-muted" style="font-family:var(--font-mono);">${r.rank_12m != null ? '#' + r.rank_12m : '—'}</td>
        <td class="asset-name">${esc(r.symbol)}</td>
        <td class="num">${fmtNum(r.price)}</td>
        <td class="num ${(r.raw_9m_pct ?? 0) >= 0 ? 't-green' : 't-red'}">${fmtPct(r.raw_9m_pct, 1)}</td>
        <td class="num col-detail">${r.risk_9m != null ? fmtNum(r.risk_9m, 2) + 'x' : '—'}</td>
        <td class="num col-detail">${r.sortino_9m != null ? fmtNum(r.sortino_9m, 2) + 'x' : '—'}</td>
        <td class="num">${r.above_ma50 ? '<span class="t-green">Yes</span>' : '<span class="t-red">No</span>'}</td>
        <td class="num" style="white-space:nowrap;">${r.listing_days != null ? `${r.listing_days}d` : '—'}${ldBadge}</td>
        <td class="num">${ewWt ? fmtPct(ewWt) : '—'}</td>
        <td class="num col-detail">${nzWt ? fmtPct(nzWt) : '—'}</td>
        <td class="${aff ? 't-green' : 't-muted'}" style="font-size:13px; white-space:nowrap;">${aff ? 'Selected ₹1L' : 'Candidate'}</td>
      </tr>`;
  }).join('');
}

function humanTriggerLabel(event) {
  if (event === 'EXIT_TO_DEFENSIVE') return { text: 'Moved to Defensive (25% Stocks)', cls: 't-red' };
  if (event === 'SHIFT_TO_CAUTIOUS') return { text: 'Stepped Down to Caution (60% Stocks)', cls: 't-amber' };
  if (event === 'STEP_UP_CAUTIOUS') return { text: 'Stepped Up to Caution (60% Stocks)', cls: 't-blue' };
  if (event === 'REENTER_EQUITY') return { text: 'Returned to Full Stocks (100%)', cls: 't-green' };
  return { text: event, cls: 't-muted' };
}

function renderTriggersTable() {
  const allRows = [...(STATE.regime.historical_triggers || [])].reverse();
  const rows = allRows.filter(t => {
    if (triggerFilter === 'DEFENSIVE') return t.event.includes('EXIT') || t.event === 'SHIFT_TO_CAUTIOUS';
    if (triggerFilter === 'RISK_ON') return t.event === 'REENTER_EQUITY';
    return true;
  });

  document.querySelector('#triggersTable tbody').innerHTML = rows.map(t => {
    const info = humanTriggerLabel(t.event);
    return `
      <tr>
        <td style="font-family:var(--font-mono); font-size:13px; white-space:nowrap;">${esc(t.date)}</td>
        <td class="${info.cls}" style="font-weight:600;">${esc(info.text)}</td>
        <td class="num">${fmtPct(t.ew_dd252_pct)}</td>
        <td class="num">${fmtPct(t.equity_pct, 0)}</td>
      </tr>`;
  }).join('');
}

function renderCharts() {
  if (typeof Chart === 'undefined' || !STATE) return;
  const ch = STATE.charts;
  const allDates = ch.dates;
  let startIdx = 0;
  const cutMap = { '1Y': -85, '3Y': -252, '5Y': -420, 'ALL': 0 };
  if (chartWindow !== 'ALL') startIdx = Math.max(0, allDates.length + cutMap[chartWindow]);

  const labels = allDates.slice(startIdx);
  const rebased = (arr) => {
    const sub = arr.slice(startIdx);
    const base = sub[0] || 1;
    return sub.map(v => Number(((v / base) * 100000).toFixed(2)));
  };

  const gridColor = '#e2e8f0';
  const tickColor = '#64748b';
  const legendColor = '#334155';

  if (CHARTS.nav) CHARTS.nav.destroy();
  const r3Nav = ch.nav.R3_Risk9m_6M_MF4_TrendGold150 || ch.nav.R2_Risk9m_6M_DD15_Sh75_GoldLiq50;
  const s3Nav = ch.nav.S3_Sortino9m_6M_MF4_TrendGold150 || r3Nav;
  const r2Nav = ch.nav.R2_Risk9m_6M_DD15_Sh75_GoldLiq50 || ch.nav.F2_9m_6M_DD12_Sh75_GoldLiq50;
  const tgNav = ch.nav.TREND_GOLD150_OR_LIQ || ch.nav.GOLD_50_LIQUID_50;

  const navDatasets = [
    { label: 'R3 Optimal Strategy', data: rebased(r3Nav), borderColor: '#059669', borderWidth: 2.5, pointRadius: 0, tension: 0.1 },
    { label: 'Midcap 150 Index (EW)', data: rebased(ch.nav.Midcap150_EW_Universe), borderColor: '#64748b', borderWidth: 1.75, pointRadius: 0, tension: 0.1 },
    { label: 'Safe-Haven TrendGold', data: rebased(tgNav), borderColor: '#d97706', borderWidth: 1.6, borderDash: [4, 4], pointRadius: 0, tension: 0.1 },
  ];
  if (extraSeries.s3) navDatasets.push({ label: 'S3 Sortino', data: rebased(s3Nav), borderColor: '#0284c7', borderWidth: 1.8, pointRadius: 0 });
  if (extraSeries.r2) navDatasets.push({ label: 'R2 Simple (-15%)', data: rebased(r2Nav), borderColor: '#7c3aed', borderWidth: 1.5, pointRadius: 0 });
  if (extraSeries.f2) navDatasets.push({ label: 'F2 Legacy (-12%)', data: rebased(ch.nav.F2_9m_6M_DD12_Sh75_GoldLiq50), borderColor: '#4f46e5', borderWidth: 1.4, borderDash: [3, 3], pointRadius: 0 });
  if (extraSeries.u0) navDatasets.push({ label: 'Unhedged 9M', data: rebased(ch.nav.Unhedged_9m_6M), borderColor: '#dc2626', borderWidth: 1.3, borderDash: [2, 3], pointRadius: 0 });

  CHARTS.nav = new Chart(document.getElementById('navChart'), {
    type: 'line',
    data: { labels, datasets: navDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: {
          type: chartLog ? 'logarithmic' : 'linear',
          ticks: { color: tickColor, callback: (v) => '₹' + Number(v).toLocaleString('en-IN') },
          grid: { color: gridColor },
        },
        x: { ticks: { color: tickColor, maxTicksLimit: 9 }, grid: { display: false } },
      },
      plugins: {
        legend: { position: 'bottom', labels: { color: legendColor, usePointStyle: true, boxWidth: 8, padding: 16 } },
        tooltip: {
          callbacks: { label: (ctx) => `${ctx.dataset.label}: ${fmtInrInt(ctx.parsed.y)}` },
        },
      },
    },
  });

  if (CHARTS.dd) CHARTS.dd.destroy();
  const r3Dd = (ch.drawdown_pct.R3_Risk9m_6M_MF4_TrendGold150 || ch.drawdown_pct.R2_Risk9m_6M_DD15_Sh75_GoldLiq50).slice(startIdx);
  const trigLvl = STATE.regime.trigger_dd_pct || -15;
  CHARTS.dd = new Chart(document.getElementById('ddChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'R3 Optimal Drawdown', data: r3Dd, borderColor: '#059669', backgroundColor: 'rgba(5, 150, 105, 0.10)', fill: true, borderWidth: 2, pointRadius: 0 },
        { label: 'Unhedged Drawdown', data: ch.drawdown_pct.Unhedged_9m_6M.slice(startIdx), borderColor: '#dc2626', borderWidth: 1.4, pointRadius: 0 },
        { label: `${trigLvl}% Reference`, data: labels.map(() => trigLvl), borderColor: '#d97706', borderDash: [5, 5], borderWidth: 1.2, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: { ticks: { color: tickColor, callback: (v) => `${Number(Number(v).toFixed(2))}%` }, grid: { color: gridColor } },
        x: { ticks: { color: tickColor, maxTicksLimit: 7 }, grid: { display: false } },
      },
      plugins: { legend: { position: 'bottom', labels: { color: legendColor, usePointStyle: true, boxWidth: 8 } } },
    },
  });

  const psumY = STATE.performance_summary || {};
  const yR3 = (psumY.R3_Risk9m_6M_MF4_TrendGold150 || {}).yearly_returns_pct || {};
  const yEw = (psumY.Midcap150_EW_Universe || {}).yearly_returns_pct || {};
  const yrs = Object.keys(yR3);

  if (CHARTS.yearly) CHARTS.yearly.destroy();
  CHARTS.yearly = new Chart(document.getElementById('yearlyChart'), {
    type: 'bar',
    data: {
      labels: yrs,
      datasets: [
        { label: 'R3 Optimal', data: yrs.map(y => yR3[y]), backgroundColor: '#059669', borderRadius: 4 },
        { label: 'Midcap 150 Index', data: yrs.map(y => yEw[y]), backgroundColor: '#cbd5e1', borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { ticks: { color: tickColor, callback: (v) => `${Number(Number(v).toFixed(2))}%` }, grid: { color: gridColor } },
        x: { ticks: { color: tickColor }, grid: { display: false } },
      },
      plugins: { legend: { position: 'bottom', labels: { color: legendColor, usePointStyle: true, boxWidth: 8 } } },
    },
  });
}

// Navigation bindings
document.querySelectorAll('.nav-btn[data-tab]').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

const PRESET_INTUITION = {
  R3: '<strong>How R3 works:</strong> Ranks stocks by 9-month return divided by volatility (favoring smooth climbers over wild spikes) and routes safety money to Gold or Cash based on Gold\'s 150-day trend.',
  S3: '<strong>How S3 works:</strong> Same 4-signal safety switch and Gold/Cash gate as R3, but ranks stocks using Sortino 9M (ignores upside jumps and only penalizes downward drops).',
  R2: '<strong>How R2 works:</strong> Uses a simpler single -15% market drop trigger (instead of 4 signals) and splits defensive money 50/50 between Gold and Cash.',
  R7: '<strong>How R7 works:</strong> Uses NormZ weighting to concentrate more money into the highest-scoring momentum outliers, with a -15% drop trigger and 50/50 Gold/Cash split.',
  F2: '<strong>How F2 works:</strong> Legacy V1 baseline ranking by raw 9-month percentage gain (ignoring volatility), switching to 50/50 Gold/Cash on a -12% drop.',
  F4: '<strong>How F4 works:</strong> Designed for larger portfolios (&gt;₹5L) — ranks by 12-month gain and rebalances once a year so more gains qualify for lower 12.5% LTCG tax.',
};

const VAULT_INTUITION = {
  pit: '<div><strong>How to read this table:</strong> Splits 16 years into three 5-year windows (2011–16 Train, 2016–21 Blind Test 1, 2021–26 Blind Test 2). <strong>CAGR</strong> = steady yearly growth; <strong>Max DD</strong> = worst drop; <strong>Sharpe</strong> = return per unit of volatility.</div>',
  crash: '<div><strong>How to read this table:</strong> Shows how each defensive engine survived 6 real market shocks — negative numbers are losses during that crash; notice how TrendGold avoided static Gold\'s -31.7% slump in 2026.</div>',
  fno: '<div><strong>How to read this table:</strong> Compares real futures/options hedges against TrendGold. <strong>OTM Put Options</strong> act like crash insurance but lose money every quiet month (theta bleed), while <strong>Short Futures</strong> need ₹25L+ capital due to ₹16.5L NSE lot minimums.</div>',
  live: '<div><strong>How to read this table:</strong> Full post-tax comparison from 2018–26. <strong>Sortino</strong> = return per unit of downside drops only; <strong>Calmar</strong> = CAGR &divide; Max DD; <strong>Turnover</strong> = % of portfolio replaced yearly; <strong>Avg Eq</strong> = average % held in stocks.</div>',
};

document.getElementById('mobileMenuBtn').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('sidebarBackdrop').classList.toggle('open');
});
document.getElementById('sidebarBackdrop').addEventListener('click', closeMobileSidebar);

// Expandable drawers
document.getElementById('toggleDiagnosticsBtn').addEventListener('click', (e) => {
  const drawer = document.getElementById('diagnosticsDrawer');
  const open = drawer.hidden;
  drawer.hidden = !open;
  e.currentTarget.setAttribute('aria-expanded', String(open));
  e.currentTarget.textContent = open ? 'Hide 4-Signal Breakdown' : 'Inspect 4-Signal Breakdown';
});

document.getElementById('toggleAdvancedBtn').addEventListener('click', (e) => {
  const drawer = document.getElementById('advancedOptionsDrawer');
  const open = drawer.hidden;
  drawer.hidden = !open;
  e.currentTarget.setAttribute('aria-expanded', String(open));
  e.currentTarget.textContent = open ? 'Hide Strategy & Filters' : 'Strategy & Filters';
});

// Quick capital presets
document.querySelectorAll('.quick-caps .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.quick-caps .seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const cap = Number(btn.dataset.cap);
    document.getElementById('capitalInput').value = cap;
    document.getElementById('slotCapToggle').checked = cap < 350000;
    renderPortfolioCalculator();
  });
});

['capitalInput', 'presetSelect', 'postureOverride', 'slotCapToggle', 'liqSweepToggle'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    if (id === 'capitalInput') {
      const cap = Number(document.getElementById('capitalInput').value);
      document.querySelectorAll('.quick-caps .seg-btn').forEach(b => b.classList.toggle('active', Number(b.dataset.cap) === cap));
    }
    if (id === 'presetSelect') {
      const key = document.getElementById('presetSelect').value;
      document.getElementById('presetIntuition').innerHTML = PRESET_INTUITION[key] || '';
    }
    renderPortfolioCalculator();
  });
  if (id === 'capitalInput') {
    document.getElementById(id).addEventListener('input', () => renderPortfolioCalculator());
  }
});

document.getElementById('resetLivePostureBtn').addEventListener('click', () => {
  document.getElementById('postureOverride').value = 'LIVE';
  renderPortfolioCalculator();
});

// Detailed column toggles
document.getElementById('orderDetailsToggle').addEventListener('change', (e) => {
  document.getElementById('ordersTableWrap').classList.toggle('show-details', e.target.checked);
  document.getElementById('orderDetailIntuition').hidden = !e.target.checked;
});

document.getElementById('rankDetailsToggle').addEventListener('change', (e) => {
  document.getElementById('rankingsTableWrap').classList.toggle('show-details', e.target.checked);
  document.getElementById('rankDetailIntuition').hidden = !e.target.checked;
});

// Rebalance & Holdings Diff controls
document.getElementById('computeDiffBtn').addEventListener('click', computeHoldingsDiff);
document.getElementById('clearHoldingsBtn').addEventListener('click', () => {
  document.getElementById('holdingsInput').value = '';
  document.getElementById('rebalanceCashInput').value = '0';
  OUTSIDE_PRICES = {};
  OUTSIDE_MODE = 'exclude';
  document.querySelectorAll('[data-outside-mode]').forEach(b => b.classList.toggle('active', b.dataset.outsideMode === 'exclude'));
  localStorage.removeItem('mmi_holdings_csv');
  localStorage.removeItem('mmi_rebalance_state_v1');
  computeHoldingsDiff();
});
['holdingsInput', 'rebalanceCashInput'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => computeHoldingsDiff());
});
['rebalancePresetSelect', 'rebalanceSlotCapToggle', 'rebalanceSweepToggle'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => computeHoldingsDiff());
});
document.querySelectorAll('[data-outside-mode]').forEach(btn => {
  btn.addEventListener('click', () => {
    OUTSIDE_MODE = btn.dataset.outsideMode;
    document.querySelectorAll('[data-outside-mode]').forEach(b => b.classList.toggle('active', b.dataset.outsideMode === OUTSIDE_MODE));
    computeHoldingsDiff();
  });
});

// Rankings search & filters
document.getElementById('rankSearch').addEventListener('input', renderRankingsTable);
document.querySelectorAll('[data-rank-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-rank-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    rankFilter = btn.dataset.rankFilter;
    renderRankingsTable();
  });
});

// Chart window, log scale, and overlay toggles
document.querySelectorAll('[data-win]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-win]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    chartWindow = btn.dataset.win;
    renderCharts();
  });
});

document.getElementById('logScaleBtn').addEventListener('click', (e) => {
  chartLog = !chartLog;
  e.currentTarget.textContent = 'Log Scale: ' + (chartLog ? 'On' : 'Off');
  renderCharts();
});

document.querySelectorAll('.series-toggle').forEach(chk => {
  chk.addEventListener('change', () => {
    extraSeries[chk.dataset.series] = chk.checked;
    renderCharts();
  });
});

// Trigger log filter
document.querySelectorAll('[data-trig-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-trig-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    triggerFilter = btn.dataset.trigFilter;
    renderTriggersTable();
  });
});

// Research vault table switcher
document.querySelectorAll('[data-vault]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-vault]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const target = btn.dataset.vault;
    document.querySelectorAll('.vault-pane').forEach(p => {
      p.hidden = p.id !== 'vault-' + target;
    });
    document.getElementById('vaultIntuition').innerHTML = VAULT_INTUITION[target] || '';
  });
});

// Export CSV
document.getElementById('exportCsvBtn').addEventListener('click', () => {
  if (!STATE) return;
  const res = getLatestComputedOrders();
  const header = 'Rank,Symbol,Group,WeightPct,LTP,TargetINR,BuyShares,ActualINR\n';
  const body = res.rows.map(r => `${r.rank},${r.sym},"${r.bucket}",${(r.wt * 100).toFixed(2)},${r.px},${r.tgt.toFixed(2)},${r.sh},${r.inv.toFixed(2)}`).join('\n');
  const blob = new Blob([header + body], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `midcap_momentum_orders_${STATE.as_of_date}_${res.capital}.csv`;
  a.click();
});

// ============================================================================
// DETERMINISTIC JS FORWARD REPLAY ENGINE (1:1 Parity With src/paper_bot.py)
// ============================================================================
const MMI_REPLAY = (() => {
  const COST_RATE = 0.0015;
  const REGIME_CUT_COST_RATE = 0.0035;
  const MIN_TRADE_DELTA_INR = 500.0;
  const LTCG_DAYS = 365;
  const LTCG_EXEMPTION_INR = 125000.0;
  const STCG_RATE = 0.20;
  const LTCG_RATE = 0.125;
  const LIQUID_SLAB_RATE = 0.30;
  const GOLD_STCG_RATE = 0.30;
  const GOLD_LTCG_RATE = 0.125;

  const ETF_SYMBOLS = ['GOLDBEES', 'SILVERBEES', 'EBBETF0431', 'LTGILTBEES', 'LIQUIDCASE', 'LIQUIDBEES', 'NIFTYBEES', 'MON100', 'MIDCAPIETF'];
  const GOLD_ETFS = new Set(['GOLDBEES', 'SILVERBEES']);
  const LIQUID_ETFS = new Set(['LIQUIDCASE', 'LIQUIDBEES']);
  const ETF_SET = new Set(ETF_SYMBOLS);

  const PRESET_RULES = {
    R3: { id: 'R3', label: 'R3 · Risk-Adj 9M + MF4 + TrendGold150', score_key: 'rk9', weighting: 'equal', rebalance_days: 126, regime_key: 'target_eq_mf4', def_mode: 'TREND_GOLD150_OR_LIQ', trim_mode: 'weakest_first' },
    S3: { id: 'S3', label: 'S3 · Sortino 9M + MF4 + TrendGold150', score_key: 'so9', weighting: 'equal', rebalance_days: 126, regime_key: 'target_eq_mf4', def_mode: 'TREND_GOLD150_OR_LIQ', trim_mode: 'weakest_first' },
    R2: { id: 'R2', label: 'R2 · Risk-Adj 9M + DD15 + 50/50 Gold/Liq', score_key: 'rk9', weighting: 'equal', rebalance_days: 126, regime_key: 'target_eq_dd15', def_mode: 'GOLD_50_LIQUID_50', trim_mode: 'weakest_first' },
    R7: { id: 'R7', label: 'R7 · NormZ Risk-Adj 9M + DD15 + 50/50', score_key: 'rk9', weighting: 'norm_z', rebalance_days: 126, regime_key: 'target_eq_dd15', def_mode: 'GOLD_50_LIQUID_50', trim_mode: 'weakest_first' },
    F2: { id: 'F2', label: 'F2 · Legacy Raw 9M + DD12 + 50/50', score_key: 'r9', weighting: 'equal', rebalance_days: 126, regime_key: 'target_eq_dd75', def_mode: 'GOLD_50_LIQUID_50', trim_mode: 'weakest_first' },
    F4: { id: 'F4', label: 'F4 · Raw 12M 12M-Reb + DD12 + 50/50', score_key: 'r12', weighting: 'equal', rebalance_days: 252, regime_key: 'target_eq_dd50', def_mode: 'GOLD_50_LIQUID_50', trim_mode: 'weakest_first' },
  };

  function roundN(x, d) {
    const f = 10 ** d;
    return Math.round((Number(x) + Number.EPSILON) * f) / f;
  }

  function finiteOrNull(val) {
    if (val == null || val === '') return null;
    const v = Number(val);
    return Number.isFinite(v) ? v : null;
  }

  function daysBetween(startDate, endDate) {
    const t1 = Date.parse(String(startDate).slice(0, 10) + 'T00:00:00Z');
    const t2 = Date.parse(String(endDate).slice(0, 10) + 'T00:00:00Z');
    if (!Number.isFinite(t1) || !Number.isFinite(t2)) return 0;
    return Math.round((t2 - t1) / 86400000);
  }

  function fyFor(dateStr) {
    const y = Number(String(dateStr).slice(0, 4));
    const m = Number(String(dateStr).slice(5, 7));
    return String(m >= 4 ? y : y - 1);
  }

  function assetClass(symbol) {
    if (GOLD_ETFS.has(symbol)) return 'GOLD_ETF';
    if (LIQUID_ETFS.has(symbol)) return 'LIQUID_ETF';
    if (ETF_SET.has(symbol)) return 'OTHER_ETF';
    return 'EQUITY';
  }

  function postureLabel(eqFrac) {
    if (eqFrac >= 0.99) return 'RISK_ON';
    if (eqFrac >= 0.50) return 'CAUTIOUS';
    return 'DEFENSIVE';
  }

  function computeSleeveWeights(orderedSyms, scoresMap, allValidSyms, topN, weighting) {
    const selected = orderedSyms.slice(0, topN);
    if (!selected.length) return {};
    if (weighting === 'norm_z') {
      const n = allValidSyms.length;
      const mean = n > 0 ? allValidSyms.reduce((s, k) => s + scoresMap[k], 0) / n : 0.0;
      const variance = n > 1 ? allValidSyms.reduce((s, k) => s + (scoresMap[k] - mean) ** 2, 0) / (n - 1) : 0.0;
      const std = variance > 0 ? Math.sqrt(variance) : 0.0;
      const rawW = {};
      let tot = 0.0;
      selected.forEach(sym => {
        const z = std > 1e-12 ? (scoresMap[sym] - mean) / std : 0.0;
        const w = z >= 0.0 ? 1.0 + z : 1.0 / (1.0 - z);
        rawW[sym] = w;
        tot += w;
      });
      const res = {};
      selected.forEach(sym => { res[sym] = tot > 0 ? rawW[sym] / tot : 1.0 / selected.length; });
      return res;
    }
    const wEq = 1.0 / selected.length;
    const res = {};
    selected.forEach(sym => { res[sym] = wEq; });
    return res;
  }

  function computeTargetWeights(rankings, prices, preset, equityBudget, slotCap = false) {
    const rule = PRESET_RULES[preset];
    const scoreKey = rule.score_key;
    const weighting = rule.weighting;
    const baseScores = {};
    const basePrices = {};

    Object.entries(rankings || {}).forEach(([sym, row]) => {
      if (ETF_SET.has(sym) || !row || typeof row !== 'object') return;
      const sc = finiteOrNull(row[scoreKey]);
      const px = finiteOrNull((prices || {})[sym]);
      if (sc !== null && px !== null && px > 0) {
        baseScores[sym] = sc;
        basePrices[sym] = px;
      }
    });

    const syms = Object.keys(baseScores);
    if (!syms.length) return {};
    const orderedAll = syms.sort((a, b) => (baseScores[b] - baseScores[a]) || (a < b ? -1 : a > b ? 1 : 0));
    const excluded = new Set();
    const maxIters = slotCap && equityBudget > 0 ? 60 : 1;
    let finalWeights = {};

    for (let it = 0; it < maxIters; it++) {
      const eligible = orderedAll.filter(s => !excluded.has(s));
      if (!eligible.length) break;
      const w20 = computeSleeveWeights(eligible, baseScores, eligible, 20, weighting);
      const w30 = computeSleeveWeights(eligible, baseScores, eligible, 30, weighting);
      const blend = {};
      let tot = 0.0;
      eligible.slice(0, 30).forEach(sym => {
        const w = 0.5 * (w20[sym] || 0.0) + 0.5 * (w30[sym] || 0.0);
        if (w > 0) {
          blend[sym] = w;
          tot += w;
        }
      });
      if (tot > 0) {
        Object.keys(blend).forEach(s => { blend[s] /= tot; });
      }
      finalWeights = blend;
      if (!slotCap || equityBudget <= 0) break;
      const unaffordable = Object.keys(blend).filter(s => basePrices[s] > equityBudget * blend[s]);
      if (!unaffordable.length || eligible.length <= 30) break;
      unaffordable.sort((a, b) => (basePrices[b] - basePrices[a]) || (b < a ? -1 : b > a ? 1 : 0));
      excluded.add(unaffordable[0]);
    }
    return finalWeights;
  }

  function resolveTargetAllocation(regime, preset, unhedged = false) {
    if (unhedged) return [1.0, 0.0, 0.0];
    const rule = PRESET_RULES[preset];
    let eqFrac = finiteOrNull((regime || {})[rule.regime_key]);
    if (eqFrac === null) eqFrac = finiteOrNull((regime || {}).target_eq_mf4);
    if (eqFrac === null) eqFrac = 1.0;
    eqFrac = Math.max(0.0, Math.min(1.0, eqFrac));
    const defFrac = Math.max(0.0, 1.0 - eqFrac);
    if (defFrac <= 1e-6) return [1.0, 0.0, 0.0];
    if (rule.def_mode === 'GOLD_50_LIQUID_50') return [eqFrac, 0.5 * defFrac, 0.5 * defFrac];
    const activeDef = String((regime || {}).active_def_etf || 'LIQUIDCASE');
    return activeDef === 'GOLDBEES' ? [eqFrac, defFrac, 0.0] : [eqFrac, 0.0, defFrac];
  }

  function computePositionsSummary(lots, prices, lastValidPrices = null) {
    const bySym = {};
    let eqVal = 0.0, goldVal = 0.0, liqVal = 0.0;
    (lots || []).forEach(lot => {
      const sh = Number(lot.shares);
      if (sh <= 1e-6) return;
      const sym = String(lot.symbol);
      let px = finiteOrNull((prices || {})[sym]);
      if (px !== null && px > 0) {
        if (lastValidPrices) lastValidPrices[sym] = roundN(px, 4);
      } else if (lastValidPrices && finiteOrNull(lastValidPrices[sym]) > 0) {
        px = Number(lastValidPrices[sym]);
      } else {
        px = Number(lot.buy_price);
      }
      const mv = sh * px;
      const ac = assetClass(sym);
      if (ac === 'GOLD_ETF') goldVal += mv;
      else if (ac === 'LIQUID_ETF') liqVal += mv;
      else eqVal += mv;

      if (!bySym[sym]) {
        bySym[sym] = {
          symbol: sym, asset_class: ac, shares: 0.0, cost_basis_total: 0.0, market_value: 0.0, mark_price: px,
          oldest_buy_date: lot.buy_date, newest_buy_date: lot.buy_date, shielded_shares: 0.0,
        };
      }
      const e = bySym[sym];
      e.shares += sh;
      e.cost_basis_total += sh * Number(lot.buy_price);
      e.market_value += mv;
      e.mark_price = px;
      if (lot.buy_date < e.oldest_buy_date) e.oldest_buy_date = lot.buy_date;
      if (lot.buy_date > e.newest_buy_date) e.newest_buy_date = lot.buy_date;
      if (lot.shield_defer) e.shielded_shares += sh;
    });
    return { by_sym: bySym, equity_val: eqVal, gold_val: goldVal, liquid_val: liqVal };
  }

  function buildHoldingsRows(state, prices, asOfDate) {
    const pos = computePositionsSummary(state.lots, prices, state.last_valid_prices);
    const basket = state.basket_weights || {};
    const rows = Object.entries(pos.by_sym).map(([sym, info]) => {
      const sh = info.shares;
      const avgBuy = sh > 0 ? info.cost_basis_total / sh : 0.0;
      const px = Number(info.mark_price || avgBuy);
      const mv = sh * px;
      const pnl = mv - info.cost_basis_total;
      const pnlPct = info.cost_basis_total > 0 ? (pnl / info.cost_basis_total) * 100.0 : 0.0;
      const dHeld = daysBetween(info.oldest_buy_date, asOfDate);
      const ac = info.asset_class;
      let bucket = 'Legacy / Shield';
      if (ac === 'GOLD_ETF' || ac === 'LIQUID_ETF') bucket = 'Defensive';
      else {
        const w = basket[sym] || 0.0;
        if (w > 0.025) bucket = 'Top 20';
        else if (w > 0) bucket = 'Ranks 21-30';
      }
      return {
        symbol: sym,
        asset_class: ac,
        bucket,
        shares: roundN(sh, 4),
        avg_buy: roundN(avgBuy, 2),
        live_price: roundN(px, 2),
        value: roundN(mv, 2),
        unrealized_pnl: roundN(pnl, 2),
        unrealized_pnl_pct: roundN(pnlPct, 2),
        oldest_buy_date: info.oldest_buy_date,
        days_held: dHeld,
        ltcg_days_remaining: Math.max(0, LTCG_DAYS - dHeld),
        ltcg_eligible: dHeld >= LTCG_DAYS,
        shield_deferred: info.shielded_shares > 1e-6,
      };
    });
    rows.sort((a, b) => (a.asset_class === 'EQUITY' ? 0 : 1) - (b.asset_class === 'EQUITY' ? 0 : 1) || (b.value - a.value) || (a.symbol < b.symbol ? -1 : 1));
    return rows;
  }

  function applyFifoSell(state, sym, sellQty, execPx, execDate, eventType, reason, allowShielded = true, costRate = COST_RATE) {
    if (sellQty <= 1e-6 || execPx <= 0) return 0.0;
    let remaining = Number(sellQty);
    const fy = fyFor(execDate);
    if (!state.tax_state.fy[fy]) state.tax_state.fy[fy] = { ltcg_exempt_used: 0.0, stcg_tax: 0.0, ltcg_tax: 0.0, stcg_pnl: 0.0, ltcg_pnl: 0.0 };
    const fyMap = state.tax_state.fy[fy];

    let soldTotal = 0.0, totalPnl = 0.0, weightedDays = 0.0;
    const eqTaxBefore = state.tax_state.realized_eq_tax_cum;
    const defTaxBefore = state.tax_state.estimated_def_tax_cum;
    const taxDefensive = state.tax_defensive !== false;
    const newLots = [];
    const ac = assetClass(sym);

    for (const lot of state.lots) {
      if (lot.symbol !== sym || remaining <= 1e-6 || (!allowShielded && lot.shield_defer)) {
        newLots.push(lot);
        continue;
      }
      const lotSh = Number(lot.shares);
      const take = Math.min(lotSh, remaining);
      if (take > 1e-6) {
        const dHeld = daysBetween(lot.buy_date, execDate);
        const lotPnl = (execPx - Number(lot.buy_price)) * take;
        soldTotal += take;
        totalPnl += lotPnl;
        weightedDays += dHeld * take;
        remaining -= take;
        const left = lotSh - take;

        if (ac === 'EQUITY') {
          if (dHeld < LTCG_DAYS) {
            fyMap.stcg_pnl += lotPnl;
            if (lotPnl < 0) state.tax_state.stcl_carry += Math.abs(lotPnl);
            else {
              const offset = Math.min(lotPnl, state.tax_state.stcl_carry);
              state.tax_state.stcl_carry -= offset;
              const tax = (lotPnl - offset) * STCG_RATE;
              fyMap.stcg_tax += tax;
              state.tax_state.realized_eq_tax_cum += tax;
              state.cash -= tax;
            }
          } else {
            fyMap.ltcg_pnl += lotPnl;
            if (lotPnl < 0) state.tax_state.ltcl_carry += Math.abs(lotPnl);
            else {
              const offLt = Math.min(lotPnl, state.tax_state.ltcl_carry);
              state.tax_state.ltcl_carry -= offLt;
              let remGain = lotPnl - offLt;
              const offSt = Math.min(remGain, state.tax_state.stcl_carry);
              state.tax_state.stcl_carry -= offSt;
              remGain -= offSt;
              const exLeft = Math.max(0.0, LTCG_EXEMPTION_INR - fyMap.ltcg_exempt_used);
              const exUsed = Math.min(remGain, exLeft);
              fyMap.ltcg_exempt_used += exUsed;
              const tax = Math.max(0.0, remGain - exUsed) * LTCG_RATE;
              fyMap.ltcg_tax += tax;
              state.tax_state.realized_eq_tax_cum += tax;
              state.cash -= tax;
            }
          }
        } else if (ac === 'LIQUID_ETF') {
          const defTax = Math.max(0.0, lotPnl) * LIQUID_SLAB_RATE;
          state.tax_state.estimated_def_tax_cum += defTax;
          if (taxDefensive) {
            state.tax_state.realized_def_tax_cum = Number(state.tax_state.realized_def_tax_cum || 0.0) + defTax;
            state.cash -= defTax;
          }
        } else {
          const rate = dHeld >= LTCG_DAYS ? GOLD_LTCG_RATE : GOLD_STCG_RATE;
          const defTax = Math.max(0.0, lotPnl) * rate;
          state.tax_state.estimated_def_tax_cum += defTax;
          if (taxDefensive) {
            state.tax_state.realized_def_tax_cum = Number(state.tax_state.realized_def_tax_cum || 0.0) + defTax;
            state.cash -= defTax;
          }
        }

        if (left > 1e-6) newLots.push({ ...lot, shares: roundN(left, 6) });
      } else {
        newLots.push(lot);
      }
    }

    if (soldTotal <= 1e-6) return 0.0;
    state.lots = newLots;
    const gross = soldTotal * execPx;
    const fee = gross * Number(costRate);
    state.cash += gross - fee;
    state.total_fees_cum += fee;

    const avgDays = Math.round(weightedDays / soldTotal);
    const taxType = ac === 'EQUITY'
      ? (avgDays >= LTCG_DAYS ? 'EQ_LTCG' : 'EQ_STCG')
      : (ac === 'LIQUID_ETF' ? 'DEBT_50AA_30%' : (avgDays >= LTCG_DAYS ? 'GOLD_12.5%' : 'GOLD_30%'));

    state.trade_log.push({
      date: execDate,
      event_type: eventType,
      symbol: sym,
      side: 'SELL',
      qty: roundN(soldTotal, 4),
      price: roundN(execPx, 4),
      notional: roundN(gross, 4),
      friction: roundN(fee, 4),
      holding_days: avgDays,
      realized_pnl: roundN(totalPnl, 4),
      tax_type: taxType,
      tax_charged: roundN(state.tax_state.realized_eq_tax_cum - eqTaxBefore, 4),
      def_tax_est: roundN(state.tax_state.estimated_def_tax_cum - defTaxBefore, 4),
      reason,
    });
    return soldTotal;
  }

  function buyLot(state, sym, wantShares, execPx, execDate, eventType, reason) {
    if (wantShares <= 0 || execPx <= 0) return 0;
    const unitCost = execPx * (1.0 + COST_RATE);
    const affordable = Math.floor(Math.max(0.0, state.cash) / unitCost);
    const qty = Math.min(Math.floor(wantShares), affordable);
    if (qty <= 0) return 0;
    const gross = qty * execPx;
    const fee = gross * COST_RATE;
    state.cash -= gross + fee;
    state.total_fees_cum += fee;
    if (!state.last_valid_prices) state.last_valid_prices = {};
    state.last_valid_prices[sym] = roundN(execPx, 4);
    state._lot_seq += 1;
    state.lots.push({
      id: `L${state._lot_seq}`,
      symbol: sym,
      asset_class: assetClass(sym),
      shares: Number(qty),
      buy_date: execDate,
      buy_price: roundN(execPx, 4),
      shield_defer: false,
    });
    state.trade_log.push({
      date: execDate,
      event_type: eventType,
      symbol: sym,
      side: 'BUY',
      qty,
      price: roundN(execPx, 4),
      notional: roundN(gross, 4),
      friction: roundN(fee, 4),
      holding_days: 0,
      realized_pnl: 0.0,
      tax_type: 'OPEN_LOT',
      tax_charged: 0.0,
      reason,
    });
    return qty;
  }

  function syncDefensiveAndSweep(state, prices, execDate, targetGoldInr, targetLiqInr, eventType, reason, liquidSweep) {
    const pos = computePositionsSummary(state.lots, prices, state.last_valid_prices).by_sym;
    const goldPx = finiteOrNull((prices || {}).GOLDBEES) || 0.0;
    const liqPx = finiteOrNull((prices || {}).LIQUIDCASE) || 0.0;

    const curGoldSh = pos.GOLDBEES ? pos.GOLDBEES.shares : 0.0;
    const curGoldVal = curGoldSh * goldPx;
    const tgtGoldSh = (goldPx > 0 && targetGoldInr > MIN_TRADE_DELTA_INR) ? Math.floor(targetGoldInr / goldPx) : 0;
    if (curGoldSh > 0 && (tgtGoldSh === 0 || (curGoldVal - targetGoldInr) > MIN_TRADE_DELTA_INR)) {
      const sellSh = tgtGoldSh === 0 ? curGoldSh : Math.min(curGoldSh, Math.ceil((curGoldVal - targetGoldInr) / goldPx));
      if (sellSh > 0) applyFifoSell(state, 'GOLDBEES', sellSh, goldPx, execDate, eventType, reason);
    }

    const curLiqSh = pos.LIQUIDCASE ? pos.LIQUIDCASE.shares : 0.0;
    const curLiqVal = curLiqSh * liqPx;
    if (!liquidSweep) {
      const tgtLiqSh = (liqPx > 0 && targetLiqInr > MIN_TRADE_DELTA_INR) ? Math.floor(targetLiqInr / liqPx) : 0;
      if (curLiqSh > 0 && (tgtLiqSh === 0 || (curLiqVal - targetLiqInr) > MIN_TRADE_DELTA_INR)) {
        const sellSh = tgtLiqSh === 0 ? curLiqSh : Math.min(curLiqSh, Math.ceil((curLiqVal - targetLiqInr) / liqPx));
        if (sellSh > 0) applyFifoSell(state, 'LIQUIDCASE', sellSh, liqPx, execDate, eventType, reason);
      }
    } else {
      const needCash = Math.max(0.0, (targetGoldInr - curGoldVal) * (1.0 + COST_RATE) - state.cash);
      if (curLiqSh > 0 && liqPx > 0 && needCash > MIN_TRADE_DELTA_INR) {
        const sellSh = Math.min(curLiqSh, Math.ceil(needCash / (liqPx * (1.0 - COST_RATE))));
        if (sellSh > 0) applyFifoSell(state, 'LIQUIDCASE', sellSh, liqPx, execDate, eventType, reason);
      }
    }

    const pos2 = computePositionsSummary(state.lots, prices, state.last_valid_prices).by_sym;
    const curGoldVal2 = (pos2.GOLDBEES ? pos2.GOLDBEES.shares : 0.0) * goldPx;
    if (goldPx > 0 && (targetGoldInr - curGoldVal2) > MIN_TRADE_DELTA_INR) {
      const wantSh = Math.floor((targetGoldInr - curGoldVal2) / (goldPx * (1.0 + COST_RATE)));
      if (wantSh > 0) buyLot(state, 'GOLDBEES', wantSh, goldPx, execDate, eventType, reason);
    }

    if (liqPx > 0) {
      if (liquidSweep) {
        if (state.cash >= liqPx * (1.0 + COST_RATE)) {
          const wantSh = Math.floor(state.cash / (liqPx * (1.0 + COST_RATE)));
          if (wantSh > 0) buyLot(state, 'LIQUIDCASE', wantSh, liqPx, execDate, eventType, `${reason} + LIQUIDCASE sweep`);
        }
      } else {
        const pos3 = computePositionsSummary(state.lots, prices, state.last_valid_prices).by_sym;
        const curLiqVal3 = (pos3.LIQUIDCASE ? pos3.LIQUIDCASE.shares : 0.0) * liqPx;
        if ((targetLiqInr - curLiqVal3) > MIN_TRADE_DELTA_INR) {
          const wantSh = Math.floor((targetLiqInr - curLiqVal3) / (liqPx * (1.0 + COST_RATE)));
          if (wantSh > 0) buyLot(state, 'LIQUIDCASE', wantSh, liqPx, execDate, eventType, reason);
        }
      }
    }
  }

  function settleNegativeCash(state, prices, execDate) {
    if (state.cash >= 0) return;
    for (const defSym of ['LIQUIDCASE', 'LIQUIDBEES', 'GOLDBEES']) {
      if (state.cash >= 0) break;
      const px = finiteOrNull((prices || {})[defSym]) || 0.0;
      const held = state.lots.filter(l => l.symbol === defSym).reduce((s, l) => s + Number(l.shares), 0.0);
      if (held > 0 && px > 0) {
        const need = Math.min(held, Math.ceil(Math.abs(state.cash) / (px * (1.0 - COST_RATE))));
        applyFifoSell(state, defSym, need, px, execDate, 'TAX_SETTLE', 'Cover realized FIFO tax liability');
      }
    }
  }

  function rebalanceToBasket(state, targetWeights, targetEqInr, prices, execDate, eventType, reason, respectLtcgShield = false) {
    const pos = computePositionsSummary(state.lots, prices, state.last_valid_prices).by_sym;
    const heldEqSyms = Object.entries(pos).filter(([, i]) => i.asset_class === 'EQUITY').map(([s]) => s).sort();

    for (const sym of heldEqSyms) {
      const px = finiteOrNull((prices || {})[sym]) || 0.0;
      if (px <= 0) continue;
      const curSh = pos[sym].shares;
      const curVal = curSh * px;
      const tgtW = targetWeights[sym] || 0.0;
      const tgtVal = targetEqInr * tgtW;
      let sellWant = 0.0;
      if (tgtW <= 0) sellWant = curSh;
      else if ((curVal - tgtVal) > MIN_TRADE_DELTA_INR && px > 0) {
        sellWant = Math.max(0.0, curSh - Math.floor(tgtVal / px));
      }
      if (sellWant <= 1e-6) continue;

      if (respectLtcgShield) {
        let allowNow = 0.0;
        for (const lot of state.lots) {
          if (lot.symbol !== sym) continue;
          const dHeld = daysBetween(lot.buy_date, execDate);
          if (dHeld >= 330 && dHeld < LTCG_DAYS) lot.shield_defer = true;
          else allowNow += Number(lot.shares);
        }
        const sellQty = Math.min(sellWant, allowNow);
        if (sellQty > 1e-6) applyFifoSell(state, sym, sellQty, px, execDate, eventType, reason, false);
      } else {
        applyFifoSell(state, sym, sellWant, px, execDate, eventType, reason, true);
      }
    }

    const posAfter = computePositionsSummary(state.lots, prices, state.last_valid_prices).by_sym;
    const buyList = Object.keys(targetWeights).sort((a, b) => (targetWeights[b] - targetWeights[a]) || (a < b ? -1 : a > b ? 1 : 0));
    for (const sym of buyList) {
      const px = finiteOrNull((prices || {})[sym]);
      if (px === null || px <= 0) continue;
      const curVal = (posAfter[sym] ? posAfter[sym].shares : 0.0) * px;
      const tgtVal = targetEqInr * targetWeights[sym];
      const diff = tgtVal - curVal;
      if (diff > MIN_TRADE_DELTA_INR) {
        const wantSh = Math.floor(diff / (px * (1.0 + COST_RATE)));
        if (wantSh > 0) buyLot(state, sym, wantSh, px, execDate, eventType, reason);
      }
    }
  }

  function initializeAccountState(opts) {
    const lotsCopy = [];
    const lastValidPrices = {};
    let seq = 0;
    for (const lot of (opts.initial_lots || [])) {
      const sh = Number(lot.shares);
      if (!Number.isFinite(sh) || sh <= 0) continue;
      seq += 1;
      const sym = String(lot.symbol).toUpperCase().trim();
      const bpx = roundN(Number(lot.buy_price), 4);
      if (bpx > 0) lastValidPrices[sym] = bpx;
      lotsCopy.push({
        id: `L${seq}`,
        symbol: sym,
        asset_class: assetClass(sym),
        shares: roundN(sh, 6),
        buy_date: String(lot.buy_date).slice(0, 10),
        buy_price: bpx,
        shield_defer: Boolean(lot.shield_defer),
      });
    }
    const initCap = Number(opts.initial_capital || 100000);
    return {
      account_id: opts.account_id || 'VIEW',
      label: opts.label || 'Tracked View',
      preset: opts.preset || 'R3',
      slot_cap: Boolean(opts.slot_cap),
      liquid_sweep: opts.liquid_sweep !== undefined ? Boolean(opts.liquid_sweep) : true,
      respect_ltcg_shield: Boolean(opts.respect_ltcg_shield),
      unhedged: Boolean(opts.unhedged),
      do_nothing: Boolean(opts.do_nothing),
      tax_defensive: opts.tax_defensive !== undefined ? Boolean(opts.tax_defensive) : true,
      initial_capital: roundN(initCap, 2),
      inception_date: null,
      last_synced_date: null,
      status: 'ACTIVE',
      ended_date: null,
      cash: Number(opts.starting_cash != null ? opts.starting_cash : initCap),
      total_fees_cum: 0.0,
      peak_nav: 0.0,
      trading_days_count: 0,
      trading_days_since_rebalance: 0,
      last_rebalance_date: null,
      active_eq_frac: 1.0,
      active_def_etf: 'LIQUIDCASE',
      basket_weights: {},
      last_valid_prices: lastValidPrices,
      _lot_seq: seq,
      lots: lotsCopy,
      tax_state: { fy: {}, stcl_carry: 0.0, ltcl_carry: 0.0, realized_eq_tax_cum: 0.0, estimated_def_tax_cum: 0.0, realized_def_tax_cum: 0.0 },
      ew_inception_val: null,
      daily_history: [],
      decision_log: [],
      trade_log: [],
      holdings: [],
    };
  }

  function stepPortfolio(state, tapeEntry, prevTapeEntry = null, unhedgedNavToday = null, benchEwNavToday = null) {
    const execDate = String(tapeEntry.date);
    if (state.status === 'ENDED') return state;
    if (state.last_synced_date && execDate <= state.last_synced_date) return state;

    const prices = tapeEntry.prices || {};
    if (!state.last_valid_prices) state.last_valid_prices = {};
    Object.entries(prices).forEach(([symP, pxP]) => {
      const vP = finiteOrNull(pxP);
      if (vP !== null && vP > 0) state.last_valid_prices[String(symP)] = roundN(vP, 4);
    });
    const regimeToday = tapeEntry.regime || {};
    const preset = state.preset;
    const rule = PRESET_RULES[preset];
    const rebPeriod = rule.rebalance_days;
    const unhedged = Boolean(state.unhedged);
    const doNothing = Boolean(state.do_nothing);

    if (state.inception_date == null) {
      state.inception_date = execDate;
      const ewVal = finiteOrNull(regimeToday.ew_index_val);
      state.ew_inception_val = (ewVal !== null && ewVal > 0) ? ewVal : 100.0;
    }

    const caEventsLogged = [];
    for (const ev of (tapeEntry.corp_actions || [])) {
      const sym = String(ev.symbol || '').toUpperCase();
      const factor = finiteOrNull(ev.factor);
      const atype = String(ev.type || 'SPLIT').toUpperCase();
      if (!sym || factor === null || factor <= 0 || Math.abs(factor - 1.0) < 1e-6) continue;
      let touched = false;
      for (const lot of state.lots) {
        if (lot.symbol === sym && lot.shares > 0) {
          lot.shares = roundN(Number(lot.shares) / factor, 6);
          lot.buy_price = roundN(Number(lot.buy_price) * factor, 4);
          touched = true;
        }
      }
      if (state.last_valid_prices && state.last_valid_prices[sym] != null) {
        state.last_valid_prices[sym] = roundN(Number(state.last_valid_prices[sym]) * factor, 4);
      }
      if (touched) {
        const mult = roundN(1.0 / factor, 4);
        const msg = `${atype} on ${sym} (factor ${factor}): FIFO lot shares multiplied by ${mult}x and cost basis adjusted cleanly.`;
        caEventsLogged.push(`${atype} ${sym} ${mult}x`);
        state.decision_log.push({ date: execDate, event_type: 'CORPORATE_ACTION', summary: msg });
        state.trade_log.push({
          date: execDate, event_type: 'CORPORATE_ACTION', symbol: sym, side: 'ADJUST', qty: mult,
          price: finiteOrNull(prices[sym]), notional: 0.0, friction: 0.0, holding_days: 0,
          realized_pnl: 0.0, tax_type: 'NONE', tax_charged: 0.0, reason: msg,
        });
      }
    }

    const posStart = computePositionsSummary(state.lots, prices, state.last_valid_prices);
    const preTradeNav = state.cash + posStart.equity_val + posStart.gold_val + posStart.liquid_val;
    const isInception = prevTapeEntry === null && state.daily_history.length === 0;
    const actionTags = [...caEventsLogged];

    const sigEntry = isInception ? tapeEntry : prevTapeEntry;
    const sigRegime = sigEntry.regime || {};
    const sigRankings = sigEntry.rankings || {};
    const [eqFrac, goldFrac, liqFrac] = resolveTargetAllocation(sigRegime, preset, unhedged);
    const sigDefEtf = rule.def_mode === 'GOLD_50_LIQUID_50' ? '50_50' : (goldFrac > liqFrac ? 'GOLDBEES' : 'LIQUIDCASE');

    if (doNothing) {
      state.trading_days_count += 1;
      if (isInception) {
        state.last_rebalance_date = execDate;
        state.decision_log.push({
          date: execDate, event_type: 'INCEPTION_HOLD',
          summary: 'Started Frozen Do-Nothing baseline holding Day-0 universe lots untouched (only corporate-action lot splits adjust automatically).',
        });
        actionTags.push('INCEPTION_HOLD');
      } else {
        state.decision_log.push({
          date: execDate, event_type: 'HOLD_CHECK',
          summary: `Frozen Do-Nothing baseline mark-to-market (${execDate}); zero trades executed.`,
        });
      }
    } else if (isInception) {
      state.trading_days_count = 1;
      state.trading_days_since_rebalance = 0;
      state.last_rebalance_date = execDate;
      const eqBudget = preTradeNav * eqFrac;
      const targetW = computeTargetWeights(sigRankings, prices, preset, eqBudget, state.slot_cap);
      state.basket_weights = { ...targetW };
      const evt = state.lots.length ? 'INCEPTION_REBALANCE' : 'INCEPTION_BUY';
      const defTag = goldFrac + liqFrac > 0 ? sigDefEtf : 'Def';
      const reason = `Day-0 deployment under ${preset} (${state.slot_cap ? 'SlotCap ON' : 'Standard'}): ${postureLabel(eqFrac)} posture (${Math.round(eqFrac * 100)}% Eq / ${Math.round((1 - eqFrac) * 100)}% ${defTag}) across ${Object.keys(targetW).length} slots.`;
      rebalanceToBasket(state, targetW, eqBudget, prices, execDate, evt, reason, state.respect_ltcg_shield);
      const posMid = computePositionsSummary(state.lots, prices, state.last_valid_prices);
      const midNav = state.cash + posMid.equity_val + posMid.gold_val + posMid.liquid_val;
      syncDefensiveAndSweep(state, prices, execDate, midNav * goldFrac, midNav * liqFrac, evt, reason, state.liquid_sweep);
      settleNegativeCash(state, prices, execDate);
      state.active_eq_frac = eqFrac;
      state.active_def_etf = sigDefEtf;
      state.decision_log.push({ date: execDate, event_type: evt, summary: reason });
      actionTags.push(evt);
    } else {
      state.trading_days_count += 1;
      state.trading_days_since_rebalance += 1;
      const isRebalanceDue = state.trading_days_since_rebalance >= rebPeriod;

      if (state.respect_ltcg_shield) {
        const targetSet = new Set(Object.keys(state.basket_weights || {}));
        const shieldedSyms = [...new Set(state.lots.filter(l => l.shield_defer).map(l => l.symbol))].sort();
        let anyMatured = false;
        for (const sym of shieldedSyms) {
          if (targetSet.has(sym)) {
            state.lots.forEach(l => { if (l.symbol === sym) l.shield_defer = false; });
            continue;
          }
          const px = finiteOrNull(prices[sym]) || 0.0;
          if (px <= 0) continue;
          const maturedQty = state.lots
            .filter(l => l.symbol === sym && l.shield_defer && daysBetween(l.buy_date, execDate) >= LTCG_DAYS)
            .reduce((s, l) => s + Number(l.shares), 0.0);
          if (maturedQty > 1e-6) {
            state.lots.forEach(l => { if (l.symbol === sym && daysBetween(l.buy_date, execDate) >= LTCG_DAYS) l.shield_defer = false; });
            const msg = `LTCG Shield matured (>=365d) on ${sym}: selling deferred lot cleanly at LTCG rate.`;
            applyFifoSell(state, sym, maturedQty, px, execDate, 'LTCG_SHIELD_MATURED', msg, false);
            state.decision_log.push({ date: execDate, event_type: 'LTCG_SHIELD_MATURED', summary: msg });
            anyMatured = true;
          }
        }
        if (anyMatured) {
          actionTags.push('LTCG_SHIELD_MATURED');
          if (!isRebalanceDue) {
            const targetEqInr = preTradeNav * eqFrac;
            if (eqFrac > 0 && state.basket_weights && Object.keys(state.basket_weights).length) {
              rebalanceToBasket(state, state.basket_weights, targetEqInr, prices, execDate, 'LTCG_SHIELD_MATURED', 'Reallocate matured LTCG proceeds into target equity basket', state.respect_ltcg_shield);
            }
            const posMid = computePositionsSummary(state.lots, prices);
            const midNav = state.cash + posMid.equity_val + posMid.gold_val + posMid.liquid_val;
            syncDefensiveAndSweep(state, prices, execDate, midNav * goldFrac, midNav * liqFrac, 'LTCG_SHIELD_MATURED', 'Sweep LTCG matured residual proceeds', state.liquid_sweep);
            settleNegativeCash(state, prices, execDate);
          }
        }
      }

      const eqShifted = Math.abs(eqFrac - Number(state.active_eq_frac)) > 1e-6;
      const defFlipped = (sigDefEtf !== state.active_def_etf) && (1.0 - eqFrac > 1e-6);

      if (isRebalanceDue) {
        const evt = rebPeriod <= 150 ? 'REBALANCE_6M' : 'REBALANCE_12M';
        state.trading_days_since_rebalance = 0;
        state.last_rebalance_date = execDate;
        const eqBudget = preTradeNav * eqFrac;
        const newW = computeTargetWeights(sigRankings, prices, preset, eqBudget, state.slot_cap);
        state.basket_weights = { ...newW };
        const reason = `Scheduled ${rebPeriod}d rebalance (${evt}): ranked Top 30 via ${preset} signals from ${sigEntry.date}; deployed ${Math.round(eqFrac * 100)}% Eq (${Object.keys(newW).length} slots).`;
        if ((goldFrac + liqFrac) < (1.0 - state.active_eq_frac) - 1e-6) {
          syncDefensiveAndSweep(state, prices, execDate, preTradeNav * goldFrac, preTradeNav * liqFrac, evt, reason, false);
        }
        rebalanceToBasket(state, newW, eqBudget, prices, execDate, evt, reason, state.respect_ltcg_shield);
        const posMid = computePositionsSummary(state.lots, prices);
        const midNav = state.cash + posMid.equity_val + posMid.gold_val + posMid.liquid_val;
        syncDefensiveAndSweep(state, prices, execDate, midNav * goldFrac, midNav * liqFrac, evt, reason, state.liquid_sweep);
        settleNegativeCash(state, prices, execDate);
        state.active_eq_frac = eqFrac;
        state.active_def_etf = sigDefEtf;
        state.decision_log.push({ date: execDate, event_type: evt, summary: reason });
        actionTags.push(evt);
      } else if (eqShifted) {
        const oldPct = Math.round(state.active_eq_frac * 100);
        const newPct = Math.round(eqFrac * 100);
        const evt = 'REGIME_SHIFT';
        const scoreKey = rule.score_key;
        const mf4S = finiteOrNull(sigRegime.mf4_score);
        const mfPrefix = mf4S !== null ? `MF4 ${(mf4S >= 0 ? '+' : '') + mf4S.toFixed(2)} ` : '';
        let msg = '';
        if (eqFrac < state.active_eq_frac) {
          const posCur = computePositionsSummary(state.lots, prices, state.last_valid_prices);
          const curNav = state.cash + posCur.equity_val + posCur.gold_val + posCur.liquid_val;
          const targetEqInr = curNav * eqFrac;
          const needFree = Math.max(0.0, posCur.equity_val - targetEqInr);
          const heldEq = Object.entries(posCur.by_sym).filter(([, i]) => i.asset_class === 'EQUITY').map(([s]) => s);
          const getSc = (s) => finiteOrNull((sigRankings[s] || {})[scoreKey]) ?? -1e9;
          heldEq.sort((a, b) => (getSc(a) - getSc(b)) || (a < b ? -1 : a > b ? 1 : 0));
          const trimmedSyms = [];
          const reasonBase = `${mfPrefix}stepped exposure ${postureLabel(state.active_eq_frac)} (${oldPct}%) -> ${postureLabel(eqFrac)} (${newPct}%); weakest-first trim into ${sigDefEtf} (35 bps risk-off impact).`;
          let freed = 0.0;
          for (const sym of heldEq) {
            if (freed >= needFree - MIN_TRADE_DELTA_INR) break;
            const px = finiteOrNull(prices[sym]) || 0.0;
            const curSh = posCur.by_sym[sym].shares;
            if (px <= 0 || curSh <= 0) continue;
            const remNeed = needFree - freed;
            const wantSh = curSh * px <= remNeed + MIN_TRADE_DELTA_INR ? curSh : Math.max(1.0, Math.floor(remNeed / px));
            const sold = applyFifoSell(state, sym, Math.min(curSh, wantSh), px, execDate, evt, reasonBase, !state.respect_ltcg_shield, REGIME_CUT_COST_RATE);
            if (sold > 0) { freed += sold * px; trimmedSyms.push(sym); }
          }
          const posMid = computePositionsSummary(state.lots, prices, state.last_valid_prices);
          const midNav = state.cash + posMid.equity_val + posMid.gold_val + posMid.liquid_val;
          syncDefensiveAndSweep(state, prices, execDate, midNav * goldFrac, midNav * liqFrac, evt, reasonBase, state.liquid_sweep);
          msg = reasonBase + (trimmedSyms.length ? ` Trimmed: ${trimmedSyms.join(', ')}.` : '');
        } else {
          const reasonBase = `${mfPrefix}stepped exposure UP ${postureLabel(state.active_eq_frac)} (${oldPct}%) -> ${postureLabel(eqFrac)} (${newPct}%); rotates defensive back into target equity basket.`;
          const targetEqInr = preTradeNav * eqFrac;
          syncDefensiveAndSweep(state, prices, execDate, preTradeNav * goldFrac, preTradeNav * liqFrac, evt, reasonBase, false);
          if (!state.basket_weights || !Object.keys(state.basket_weights).length) {
            state.basket_weights = computeTargetWeights(sigRankings, prices, preset, targetEqInr, state.slot_cap);
          }
          rebalanceToBasket(state, state.basket_weights, targetEqInr, prices, execDate, evt, reasonBase, state.respect_ltcg_shield);
          if (state.liquid_sweep) {
            const posMid = computePositionsSummary(state.lots, prices, state.last_valid_prices);
            const midNav = state.cash + posMid.equity_val + posMid.gold_val + posMid.liquid_val;
            syncDefensiveAndSweep(state, prices, execDate, midNav * goldFrac, midNav * liqFrac, evt, reasonBase, true);
          }
          msg = reasonBase;
        }
        settleNegativeCash(state, prices, execDate);
        state.active_eq_frac = eqFrac;
        state.active_def_etf = sigDefEtf;
        state.decision_log.push({ date: execDate, event_type: evt, summary: msg });
        actionTags.push(evt);
      } else if (defFlipped) {
        const evt = 'DEFENSIVE_GATE_FLIP';
        const gPct = finiteOrNull(sigRegime.gold_vs_ema150_pct);
        const gStr = gPct !== null ? `${gPct >= 0 ? '+' : ''}${gPct.toFixed(2)}%` : '';
        const reason = `TrendGold150 gate flipped defensive sleeve from ${state.active_def_etf} -> ${sigDefEtf} (GOLDBEES ${gStr} vs EMA150; 0.15% switching fee).`;
        syncDefensiveAndSweep(state, prices, execDate, preTradeNav * goldFrac, preTradeNav * liqFrac, evt, reason, state.liquid_sweep);
        settleNegativeCash(state, prices, execDate);
        state.active_def_etf = sigDefEtf;
        state.decision_log.push({ date: execDate, event_type: evt, summary: reason });
        actionTags.push(evt);
      } else {
        const daysLeft = Math.max(0, rebPeriod - state.trading_days_since_rebalance);
        const mf4S = finiteOrNull(sigRegime.mf4_score);
        const dayWord = daysLeft === 1 ? 'trading day' : 'trading days';
        const msg = `Hold ${postureLabel(eqFrac)} (${Math.round(eqFrac * 100)}% Eq); ${mf4S !== null ? `MF4 smooth ${(mf4S >= 0 ? '+' : '') + mf4S.toFixed(2)}; ` : ''}next rebalance in ${daysLeft} ${dayWord}; defensive=${sigDefEtf}.`;
        state.decision_log.push({ date: execDate, event_type: 'HOLD_CHECK', summary: msg });
      }
    }

    const posEnd = computePositionsSummary(state.lots, prices, state.last_valid_prices);
    const navEnd = state.cash + posEnd.equity_val + posEnd.gold_val + posEnd.liquid_val;
    state.peak_nav = Math.max(Number(state.peak_nav || 0.0), navEnd);
    const ddPct = state.peak_nav > 0 ? (navEnd / state.peak_nav - 1.0) * 100.0 : 0.0;
    const defTaxCum = Number(state.tax_state.estimated_def_tax_cum || 0.0);
    const realizedDefTaxCum = Number(state.tax_state.realized_def_tax_cum || 0.0);
    const eqTaxCum = Number(state.tax_state.realized_eq_tax_cum || 0.0);
    const navFullTax = state.tax_defensive !== false ? navEnd : Math.max(0.0, navEnd - defTaxCum);
    const navPreDefTax = state.tax_defensive !== false ? (navEnd + realizedDefTaxCum) : navEnd;

    if (benchEwNavToday == null) {
      const ewCur = finiteOrNull(regimeToday.ew_index_val) || state.ew_inception_val || 100.0;
      const ewBase = Number(state.ew_inception_val || ewCur || 100.0);
      benchEwNavToday = ewBase > 0 ? state.initial_capital * (ewCur / ewBase) : state.initial_capital;
    }

    state.holdings = buildHoldingsRows(state, prices, execDate);
    state.last_synced_date = execDate;
    const eqSharePct = navEnd > 0 ? (posEnd.equity_val / navEnd) * 100.0 : 0.0;
    state.daily_history.push({
      date: execDate,
      nav: roundN(navEnd, 4),
      nav_full_tax: roundN(navFullTax, 4),
      nav_pre_def_tax: roundN(navPreDefTax, 4),
      cash: roundN(state.cash, 4),
      equity_val: roundN(posEnd.equity_val, 4),
      gold_val: roundN(posEnd.gold_val, 4),
      liquid_val: roundN(posEnd.liquid_val, 4),
      eq_pct: roundN(eqSharePct, 2),
      posture: String(regimeToday.posture || postureLabel(eqFrac)),
      mf4_score: finiteOrNull(regimeToday.mf4_score),
      active_def_etf: String(regimeToday.active_def_etf || state.active_def_etf),
      benchmark_ew_nav: roundN(benchEwNavToday, 4),
      unhedged_nav: unhedgedNavToday != null ? roundN(unhedgedNavToday, 4) : null,
      drawdown_pct: roundN(ddPct, 4),
      realized_tax_cum: roundN(eqTaxCum, 4),
      estimated_def_tax_cum: roundN(defTaxCum, 4),
      total_fees_cum: roundN(state.total_fees_cum, 4),
      action_summary: actionTags.length ? actionTags.join(', ') : 'HOLD_CHECK',
    });
    return state;
  }

  function runSyntheticSteps(tapeEntries, opts = {}) {
    const state = initializeAccountState(opts);
    (tapeEntries || []).forEach((day, idx) => {
      stepPortfolio(state, day, idx > 0 ? tapeEntries[idx - 1] : null);
    });
    return state;
  }

  function summarizeAccount(acc) {
    const hist = acc.daily_history || [];
    const last = hist.length ? hist[hist.length - 1] : {};
    const cap = Number(acc.initial_capital || 100000);
    const curNav = Number(last.nav != null ? last.nav : cap);
    const curNavFt = Number(last.nav_full_tax != null ? last.nav_full_tax : curNav);
    const curNavPreDef = Number(last.nav_pre_def_tax != null ? last.nav_pre_def_tax : curNav);
    const ewNav = Number(last.benchmark_ew_nav != null ? last.benchmark_ew_nav : cap);
    const unhNav = Number(last.unhedged_nav != null ? last.unhedged_nav : cap);
    const totRet = cap > 0 ? (curNav / cap - 1.0) * 100.0 : 0.0;
    const totRetFt = cap > 0 ? (curNavFt / cap - 1.0) * 100.0 : 0.0;
    const totRetPreDef = cap > 0 ? (curNavPreDef / cap - 1.0) * 100.0 : 0.0;
    const ewRet = cap > 0 ? (ewNav / cap - 1.0) * 100.0 : 0.0;
    const unhRet = cap > 0 ? (unhNav / cap - 1.0) * 100.0 : 0.0;
    const maxDd = hist.length ? Math.min(...hist.map(h => Number(h.drawdown_pct || 0.0))) : 0.0;
    const rebDays = PRESET_RULES[acc.preset]?.rebalance_days || 126;
    const daysSince = Number(acc.trading_days_since_rebalance || 0);
    return {
      inception_date: acc.inception_date,
      as_of_date: acc.last_synced_date,
      trading_days: hist.length,
      days_since_rebalance: daysSince,
      next_rebalance_in_trading_days: Math.max(0, rebDays - daysSince),
      tax_defensive: acc.tax_defensive !== false,
      nav: roundN(curNav, 2),
      nav_full_tax: roundN(curNavFt, 2),
      nav_pre_def_tax: roundN(curNavPreDef, 2),
      total_return_pct: roundN(totRet, 2),
      total_return_full_tax_pct: roundN(totRetFt, 2),
      total_return_pre_def_tax_pct: roundN(totRetPreDef, 2),
      benchmark_ew_nav: roundN(ewNav, 2),
      benchmark_ew_return_pct: roundN(ewRet, 2),
      unhedged_nav: roundN(unhNav, 2),
      unhedged_return_pct: roundN(unhRet, 2),
      alpha_vs_ew_pct: roundN(totRet - ewRet, 2),
      max_drawdown_pct: roundN(maxDd, 2),
      alloc_pct: {
        equity: curNav > 0 ? roundN((Number(last.equity_val || 0) / curNav) * 100.0, 1) : 0.0,
        gold: curNav > 0 ? roundN((Number(last.gold_val || 0) / curNav) * 100.0, 1) : 0.0,
        liquid: curNav > 0 ? roundN((Number(last.liquid_val || 0) / curNav) * 100.0, 1) : 0.0,
        cash: curNav > 0 ? roundN((Number(last.cash || 0) / curNav) * 100.0, 1) : 0.0,
      },
      total_fees_cum: roundN(acc.total_fees_cum || 0, 2),
      realized_eq_tax_cum: roundN(acc.tax_state?.realized_eq_tax_cum || 0, 2),
      estimated_def_tax_cum: roundN(acc.tax_state?.estimated_def_tax_cum || 0, 2),
      held_count: (acc.holdings || []).length,
    };
  }

  return { PRESET_RULES, computeTargetWeights, initializeAccountState, stepPortfolio, runSyntheticSteps, summarizeAccount };
})();
window.mmiForwardReplay = MMI_REPLAY;

// ============================================================================
// TRACKED PORTFOLIO VIEWS (localStorage) + AUTO CATCH-UP + PAPER BOT UI
// ============================================================================
const TRACKED_VIEWS_KEY = 'mmi_tracked_views_v1';

function showToast(msg) {
  const el = document.getElementById('toastBanner');
  if (!el) return;
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(window._mmiToastTimer);
  window._mmiToastTimer = setTimeout(() => { el.hidden = true; }, 4500);
}

function loadTrackedViewsFromStorage() {
  try {
    const raw = localStorage.getItem(TRACKED_VIEWS_KEY);
    TRACKED_VIEWS = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(TRACKED_VIEWS)) TRACKED_VIEWS = [];
  } catch {
    TRACKED_VIEWS = [];
  }
  if (!PB_SELECTED_VIEW_ID && TRACKED_VIEWS.length) {
    PB_SELECTED_VIEW_ID = TRACKED_VIEWS[0].view_id;
  }
  TRACKED_VIEWS.forEach(v => { if (!PB_COMPARE_IDS.size || PB_COMPARE_IDS.size < 4) PB_COMPARE_IDS.add(v.view_id); });
}

function saveTrackedViewsToStorage() {
  try {
    localStorage.setItem(TRACKED_VIEWS_KEY, JSON.stringify(TRACKED_VIEWS));
  } catch {}
  updatePbViewsBadge();
}

function updatePbViewsBadge() {
  const el = document.getElementById('pbViewsCount');
  if (el) el.textContent = String(TRACKED_VIEWS.length);
}

function getFallbackDay0FromState() {
  const prices = {};
  (STATE.universe_scores || []).forEach(u => { prices[u.sym] = u.px; });
  Object.entries(STATE.regime?.etfs || {}).forEach(([k, v]) => { if (v?.price) prices[k] = v.price; });
  const rankings = {};
  (STATE.universe_scores || []).forEach(u => { rankings[u.sym] = { rk9: u.rk9, so9: u.so9, r9: u.r9, r12: u.r12, a50: u.ma50 ? 1 : 0 }; });
  const eqTarget = (STATE.regime?.target_equity_pct ?? 100) / 100.0;
  return {
    date: STATE.as_of_date,
    prices,
    corp_actions: [],
    regime: {
      mf4_score: STATE.regime?.multifactor_blocks?.composite_smooth ?? 0.0,
      posture: STATE.regime?.state || 'RISK_ON',
      target_eq_mf4: eqTarget,
      target_eq_dd15: eqTarget,
      target_eq_dd75: eqTarget,
      target_eq_dd50: eqTarget,
      active_def_etf: STATE.regime?.active_defensive_etf || 'LIQUIDCASE',
      gold_price: STATE.regime?.etfs?.GOLDBEES?.price,
      gold_vs_ema150_pct: STATE.regime?.trend_gold_status?.gold_vs_ema150_pct,
      ew_index_val: STATE.regime?.ew_index || 1000,
    },
    rankings,
  };
}

function syncAllActiveTrackedViews(tapeObj) {
  const days = tapeObj?.days || [];
  if (!days.length) return 0;
  let syncedSteps = 0;
  TRACKED_VIEWS.forEach(view => {
    const st = view.state;
    if (!st || st.status === 'ENDED') return;
    for (let i = 0; i < days.length; i++) {
      const d = days[i];
      if (st.last_synced_date && d.date <= st.last_synced_date) continue;
      if (view.clone_spec?.inception_date && d.date < view.clone_spec.inception_date) continue;
      const prev = i > 0 ? days[i - 1] : null;
      MMI_REPLAY.stepPortfolio(st, d, prev);
      syncedSteps += 1;
    }
    st.summary = MMI_REPLAY.summarizeAccount(st);
  });
  if (syncedSteps > 0) saveTrackedViewsToStorage();
  return syncedSteps;
}

window.mmiSyncPaperTape = function(newTapeObj) {
  FORWARD_TAPE = newTapeObj;
  const n = syncAllActiveTrackedViews(newTapeObj);
  renderPaperBotPage();
  return n;
};

function switchPaperBotSubTab(sub) {
  PB_SUBTAB = sub === 'views' ? 'views' : 'official';
  document.getElementById('pbTabBtnOfficial').classList.toggle('active', PB_SUBTAB === 'official');
  document.getElementById('pbTabBtnViews').classList.toggle('active', PB_SUBTAB === 'views');
  document.getElementById('pbPaneOfficial').hidden = PB_SUBTAB !== 'official';
  document.getElementById('pbPaneViews').hidden = PB_SUBTAB !== 'views';
  renderPaperBotPage();
}

function renderLedgerKpis(containerId, acc, useFullTax, showFullTaxSub = true) {
  const sum = acc.summary || MMI_REPLAY.summarizeAccount(acc);
  const taxDefMode = acc.tax_defensive !== false;
  const navDisp = useFullTax ? sum.nav_full_tax : (taxDefMode ? (sum.nav_pre_def_tax ?? sum.nav) : sum.nav);
  const retDisp = useFullTax ? sum.total_return_full_tax_pct : (taxDefMode ? (sum.total_return_pre_def_tax_pct ?? sum.total_return_pct) : sum.total_return_pct);
  const lastHist = (acc.daily_history || [])[Math.max(0, (acc.daily_history || []).length - 1)] || {};
  const posture = lastHist.posture || 'RISK_ON';
  const alloc = sum.alloc_pct || { equity: 0, gold: 0, liquid: 0, cash: 0 };
  const totalTax = useFullTax ? (sum.realized_eq_tax_cum + sum.estimated_def_tax_cum) : sum.realized_eq_tax_cum;
  const items = [
    {
      lbl: useFullTax ? 'NAV (Net Sec 50AA / Gold Tax)' : 'Pre-Def-Tax Comparison NAV',
      val: fmtInrInt(navDisp),
      cls: retDisp >= 0 ? 't-green' : 't-red',
      note: `Return ${(retDisp >= 0 ? '+' : '') + fmtPct(retDisp)}${showFullTaxSub && sum.estimated_def_tax_cum > 0 ? ` · Pre-def ${fmtInrInt(sum.nav_pre_def_tax ?? sum.nav)}` : ''}`,
    },
    {
      lbl: 'Alpha vs Midcap 150 EW',
      val: (sum.alpha_vs_ew_pct >= 0 ? '+' : '') + fmtPct(sum.alpha_vs_ew_pct),
      cls: sum.alpha_vs_ew_pct >= 0 ? 't-green' : 't-red',
      note: `EW ${fmtInrInt(sum.benchmark_ew_nav)} (${(sum.benchmark_ew_return_pct >= 0 ? '+' : '') + fmtPct(sum.benchmark_ew_return_pct)})`,
    },
    {
      lbl: 'Max Drawdown',
      val: fmtPct(sum.max_drawdown_pct),
      cls: 't-blue',
      note: `Inception ${sum.inception_date || '—'} · ${sum.trading_days}d`,
    },
    {
      lbl: 'Allocation & Posture',
      val: `${alloc.equity}% Eq`,
      cls: posture === 'RISK_ON' ? 't-green' : (posture === 'CAUTIOUS' ? 't-amber' : 't-red'),
      note: `${posture} · ${alloc.gold}% Gold · ${alloc.liquid}% Liq`,
    },
    {
      lbl: 'Fees & Tax / Next Rebal',
      val: fmtInrInt(sum.total_fees_cum + totalTax),
      cls: '',
      note: `Fees ${fmtInrInt(sum.total_fees_cum)} · Rebal in ${sum.next_rebalance_in_trading_days}d`,
    },
  ];
  document.getElementById(containerId).innerHTML = items.map(i => `
    <div class="metric-cell">
      <div class="metric-label">${esc(i.lbl)}</div>
      <div class="metric-value ${i.cls}">${esc(i.val)}</div>
      <div class="metric-note">${esc(i.note)}</div>
    </div>`).join('');
}

function renderLedgerHoldingsTable(wrapId, acc) {
  const rows = acc.holdings || [];
  if (!rows.length) {
    document.getElementById(wrapId).innerHTML = '<div class="empty-state"><p>No open holdings recorded.</p></div>';
    return;
  }
  document.getElementById(wrapId).innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th>Bucket</th>
          <th class="num">Shares</th>
          <th class="num">Avg Buy &#8377;</th>
          <th class="num">Live &#8377;</th>
          <th class="num">Value &#8377;</th>
          <th class="num">Unrealized P&amp;L</th>
          <th class="num">Days Held</th>
          <th>LTCG Status</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => {
          const pnlCls = r.unrealized_pnl >= 0 ? 't-green' : 't-red';
          let ltcgHtml = r.ltcg_eligible
            ? '<span class="badge badge-green">&#10003; LTCG Eligible</span>'
            : `<span class="badge">LTCG in ${r.ltcg_days_remaining}d</span>`;
          if (r.shield_deferred) ltcgHtml += ' <span class="badge badge-amber">Shielded</span>';
          return `
            <tr>
              <td class="asset-name">${esc(r.symbol)}</td>
              <td class="role-text">${esc(r.bucket)}</td>
              <td class="num">${r.shares}</td>
              <td class="num">${fmtNum(r.avg_buy)}</td>
              <td class="num">${fmtNum(r.live_price)}</td>
              <td class="num">${fmtInrInt(r.value)}</td>
              <td class="num ${pnlCls}">${fmtInrInt(r.unrealized_pnl)} (${r.unrealized_pnl_pct >= 0 ? '+' : ''}${fmtPct(r.unrealized_pnl_pct)})</td>
              <td class="num">${r.days_held}d</td>
              <td>${ltcgHtml}</td>
            </tr>`;
        }).join('')}
      </tbody>
    </table>`;
}

function renderLedgerAudit(feedId, tradeWrapId, acc, decFilter, tradeFilter) {
  const decs = [...(acc.decision_log || [])].reverse().filter(d => decFilter === 'ALL' || d.event_type !== 'HOLD_CHECK');
  const feedEl = document.getElementById(feedId);
  if (!decs.length) {
    feedEl.innerHTML = '<div class="empty-state"><p>No decision events match the selected filter.</p></div>';
  } else {
    feedEl.innerHTML = decs.map(d => {
      const badgeCls = d.event_type.includes('REGIME') ? 'badge-amber'
        : d.event_type.includes('GATE') ? 'badge-blue'
        : d.event_type.includes('CORP') ? 'badge-red'
        : d.event_type === 'HOLD_CHECK' ? '' : 'badge-green';
      return `
        <div class="pb-feed-item">
          <div>
            <span class="badge ${badgeCls}">${esc(d.event_type)}</span>
            <span style="margin-left:8px; font-size:13px; color:var(--ink-secondary);">${esc(d.summary)}</span>
          </div>
          <span style="font-family:var(--font-mono); font-size:12px; color:var(--ink-muted); white-space:nowrap;">${esc(d.date)}</span>
        </div>`;
    }).join('');
  }

  const trades = [...(acc.trade_log || [])].reverse().filter(t => {
    if (tradeFilter === 'REBALANCE') return t.event_type.includes('REBALANCE') || t.event_type.includes('INCEPTION');
    if (tradeFilter === 'REGIME') return t.event_type.includes('REGIME');
    if (tradeFilter === 'GATE') return t.event_type.includes('GATE');
    return true;
  });
  const wrapEl = document.getElementById(tradeWrapId);
  if (!trades.length) {
    wrapEl.innerHTML = '<div class="empty-state"><p>No trades executed for this filter yet.</p></div>';
    return;
  }
  wrapEl.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Event</th>
          <th>Side</th>
          <th>Symbol</th>
          <th class="num">Qty</th>
          <th class="num">Price &#8377;</th>
          <th class="num">Value &#8377;</th>
          <th class="num">Fee &#8377;</th>
          <th class="num">Realized P&amp;L</th>
          <th class="num">Tax &#8377;</th>
        </tr>
      </thead>
      <tbody>
        ${trades.map(t => `
          <tr>
            <td style="font-family:var(--font-mono); font-size:12px; white-space:nowrap;">${esc(t.date)}</td>
            <td class="role-text">${esc(t.event_type)}</td>
            <td class="${t.side === 'BUY' ? 't-green' : t.side === 'SELL' ? 't-red' : 't-blue'}" style="font-weight:600;">${esc(t.side)}</td>
            <td class="asset-name">${esc(t.symbol)}</td>
            <td class="num">${t.qty}</td>
            <td class="num">${t.price != null ? fmtNum(t.price) : '—'}</td>
            <td class="num">${fmtInrInt(t.notional)}</td>
            <td class="num">${fmtNum(t.friction)}</td>
            <td class="num ${(t.realized_pnl || 0) > 0 ? 't-green' : (t.realized_pnl || 0) < 0 ? 't-red' : 't-muted'}">${t.side === 'SELL' ? fmtInrInt(t.realized_pnl) : '—'}</td>
            <td class="num">${t.side === 'SELL' ? fmtNum(t.tax_charged) : '—'}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

function renderLedgerLineChart(canvasId, chartSlotKey, acc, extraBaselineAcc = null, useFullTax = false, earlyBoxId = null, chartWrapId = null) {
  const hist = acc.daily_history || [];
  const earlyEl = earlyBoxId ? document.getElementById(earlyBoxId) : null;
  const wrapEl = chartWrapId ? document.getElementById(chartWrapId) : null;
  if (!hist.length) return;

  if (hist.length < 5 && earlyEl && wrapEl) {
    if (CHARTS[chartSlotKey]) { CHARTS[chartSlotKey].destroy(); delete CHARTS[chartSlotKey]; }
    wrapEl.hidden = true;
    earlyEl.hidden = false;
    const sum = acc.summary || MMI_REPLAY.summarizeAccount(acc);
    const lastH = hist[hist.length - 1];
    const taxDefMode = acc.tax_defensive !== false;
    const dispNav = useFullTax ? (lastH.nav_full_tax ?? lastH.nav) : (taxDefMode ? (lastH.nav_pre_def_tax ?? lastH.nav) : lastH.nav);
    const dayLbl = hist.length === 1 ? '1 trading day' : `${hist.length} trading days`;
    earlyEl.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px;">
        <div>
          <span class="badge badge-blue">Early-Inception Summary · ${dayLbl}</span>
          <strong style="margin-left:8px; font-size:14px;">Multi-day curve chart activates automatically after 5 trading days</strong>
        </div>
        <span class="t-muted" style="font-size:12px; font-family:var(--font-mono);">Inception ${esc(acc.inception_date || lastH.date)} &rarr; ${esc(lastH.date)}</span>
      </div>
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; font-size:13px;">
        <div><div class="t-muted" style="font-size:11px;">Starting Capital</div><strong class="mono">${fmtInrInt(acc.initial_capital)}</strong></div>
        <div><div class="t-muted" style="font-size:11px;">Current NAV</div><strong class="mono ${(dispNav >= acc.initial_capital) ? 't-green' : 't-red'}">${fmtInrInt(dispNav)}</strong></div>
        <div><div class="t-muted" style="font-size:11px;">Midcap EW Ref</div><strong class="mono">${fmtInrInt(lastH.benchmark_ew_nav)}</strong></div>
        <div><div class="t-muted" style="font-size:11px;">Friction &amp; Tax Deducted</div><strong class="mono">${fmtInrInt((lastH.total_fees_cum || 0) + (lastH.realized_tax_cum || 0))}</strong></div>
        <div><div class="t-muted" style="font-size:11px;">Latest Action</div><span class="badge badge-green">${esc(lastH.action_summary || 'INCEPTION')}</span></div>
      </div>`;
    return;
  }

  if (earlyEl && wrapEl) {
    earlyEl.hidden = true;
    wrapEl.hidden = false;
  }
  if (typeof Chart === 'undefined') return;
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (CHARTS[chartSlotKey]) CHARTS[chartSlotKey].destroy();

  const taxDefMode = acc.tax_defensive !== false;
  const labels = hist.map(h => h.date);
  const botSeries = hist.map(h => useFullTax ? (h.nav_full_tax ?? h.nav) : (taxDefMode ? (h.nav_pre_def_tax ?? h.nav) : h.nav));
  const ptRadii = hist.map(h => (h.action_summary && !h.action_summary.includes('HOLD_CHECK')) ? 5 : 2);
  const datasets = [
    {
      label: acc.label || 'Strategy Bot NAV',
      data: botSeries,
      borderColor: '#059669',
      backgroundColor: 'rgba(5, 150, 105, 0.10)',
      borderWidth: 2.4,
      pointRadius: ptRadii,
      pointBackgroundColor: hist.map(h => String(h.action_summary || '').includes('REGIME') ? '#b45309' : '#059669'),
      tension: 0.1,
    },
    {
      label: 'Midcap 150 EW Benchmark',
      data: hist.map(h => h.benchmark_ew_nav),
      borderColor: '#64748b',
      borderDash: [4, 4],
      borderWidth: 1.75,
      pointRadius: 2,
      tension: 0.1,
    },
  ];
  if (hist.some(h => h.unhedged_nav != null)) {
    datasets.push({
      label: 'Unhedged 9M Baseline',
      data: hist.map(h => h.unhedged_nav),
      borderColor: '#dc2626',
      borderDash: [2, 3],
      borderWidth: 1.5,
      pointRadius: 1.5,
      tension: 0.1,
    });
  }
  if (extraBaselineAcc && extraBaselineAcc.daily_history?.length) {
    const bMap = new Map(extraBaselineAcc.daily_history.map(h => [h.date, h.nav]));
    datasets.push({
      label: 'Do-Nothing Frozen Baseline',
      data: labels.map(d => bMap.get(d) ?? null),
      borderColor: '#1d4ed8',
      borderWidth: 1.8,
      pointRadius: 2,
      tension: 0.1,
    });
  }

  CHARTS[chartSlotKey] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: { ticks: { color: '#64748b', callback: (v) => '₹' + Number(v).toLocaleString('en-IN') }, grid: { color: '#e2e8f0' } },
        x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { display: false } },
      },
      plugins: {
        legend: { position: 'bottom', labels: { color: '#334155', usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const i = items?.[0]?.dataIndex;
              if (i != null && hist[i]?.action_summary) return `Event: ${hist[i].action_summary}`;
              return '';
            },
          },
        },
      },
    },
  });
}

function downloadLedgerCsv(acc, filePrefix) {
  const trades = acc.trade_log || [];
  const header = 'Date,EventType,Side,Symbol,Qty,PriceINR,NotionalINR,FrictionINR,HoldingDays,RealizedPnL,TaxType,TaxChargedINR,Reason\n';
  const body = trades.map(t => `${t.date},${t.event_type},${t.side},${t.symbol},${t.qty},${t.price ?? ''},${t.notional},${t.friction},${t.holding_days},${t.realized_pnl},"${t.tax_type}",${t.tax_charged},"${String(t.reason || '').replace(/"/g, '""')}"`).join('\n');
  const blob = new Blob([header + body], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${filePrefix}_${acc.last_synced_date || 'ledger'}.csv`;
  a.click();
}

function renderOfficialBotPane() {
  const pill = document.getElementById('pbOfficialStatusPill');
  if (!PAPER_LEDGER || !PAPER_LEDGER.accounts?.[PB_ACCOUNT]) {
    pill.textContent = 'Ledger initializing cleanly after next run';
    document.getElementById('pbOfficialKpis').innerHTML = '';
    document.getElementById('pbOfficialHoldingsWrap').innerHTML = '<div class="empty-state"><p>Run <code>python3 scripts/update_live.py</code> to initialize <code>data/paper_ledger.json</code>.</p></div>';
    return;
  }
  const acc = PAPER_LEDGER.accounts[PB_ACCOUNT];
  const sum = acc.summary || MMI_REPLAY.summarizeAccount(acc);
  const useFullTax = document.getElementById('pbNetDefTaxToggle').checked;
  const tdWord = sum.trading_days === 1 ? 'trading day' : 'trading days';
  pill.textContent = `Active since ${sum.inception_date} · ${sum.trading_days} ${tdWord} · Next 6M Rebalance in ${sum.next_rebalance_in_trading_days}d`;
  document.getElementById('pbOfficialHeldCount').textContent = String((acc.holdings || []).length);

  renderLedgerKpis('pbOfficialKpis', acc, useFullTax, true);
  renderLedgerLineChart('pbOfficialChart', 'pbOfficialChart', acc, null, useFullTax, 'pbOfficialEarlyState', 'pbOfficialChartWrap');
  renderLedgerHoldingsTable('pbOfficialHoldingsWrap', acc);
  renderLedgerAudit('pbOfficialDecisionFeed', 'pbOfficialTradeWrap', acc, PB_DEC_FILTER, PB_TRADE_FILTER);
}

function renderTrackedViewsPane() {
  const wrap = document.getElementById('pbViewsOverviewWrap');
  const cmpPanel = document.getElementById('pbViewsComparePanel');
  const detailPanel = document.getElementById('pbSelectedViewPanel');

  if (!TRACKED_VIEWS.length) {
    wrap.innerHTML = '<div class="empty-state"><p>No Tracked Views saved yet. Go to <strong>Rebalance &amp; Tax Diff</strong>, paste your portfolio, and click <strong>Hand Over to Bot</strong> to start live forward tracking.</p></div>';
    cmpPanel.hidden = true;
    detailPanel.hidden = true;
    return;
  }

  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th style="width:36px;">Cmp</th>
          <th>View Name</th>
          <th>Strategy</th>
          <th>Status</th>
          <th>Inception</th>
          <th>Synced</th>
          <th class="num">Value &#8377;</th>
          <th class="num">Return %</th>
          <th class="num">Alpha vs EW</th>
          <th>Controls</th>
        </tr>
      </thead>
      <tbody>
        ${TRACKED_VIEWS.map(v => {
          const st = v.state;
          const sum = st.summary || MMI_REPLAY.summarizeAccount(st);
          const isEnded = st.status === 'ENDED';
          const isSel = v.view_id === PB_SELECTED_VIEW_ID;
          return `
            <tr style="${isSel ? 'background:#f0fdf4;' : ''}">
              <td><input type="checkbox" class="pb-cmp-chk" data-view-id="${esc(v.view_id)}" ${PB_COMPARE_IDS.has(v.view_id) ? 'checked' : ''} aria-label="Compare ${esc(v.name)}" /></td>
              <td class="asset-name">${esc(v.name)}</td>
              <td class="role-text">${v.do_nothing ? 'Do-Nothing Frozen' : esc(v.preset + (v.respect_ltcg_shield ? ' · LTCG Shield' : ''))}</td>
              <td>${isEnded ? `<span class="badge">ENDED ${esc(st.ended_date || '')}</span>` : '<span class="badge badge-green">ACTIVE</span>'}</td>
              <td style="font-family:var(--font-mono); font-size:12px;">${esc(sum.inception_date || '—')}</td>
              <td style="font-family:var(--font-mono); font-size:12px;">${esc(sum.as_of_date || '—')}</td>
              <td class="num">${fmtInrInt(sum.nav)}</td>
              <td class="num ${sum.total_return_pct >= 0 ? 't-green' : 't-red'}">${(sum.total_return_pct >= 0 ? '+' : '') + fmtPct(sum.total_return_pct)}</td>
              <td class="num ${sum.alpha_vs_ew_pct >= 0 ? 't-green' : 't-red'}">${(sum.alpha_vs_ew_pct >= 0 ? '+' : '') + fmtPct(sum.alpha_vs_ew_pct)}</td>
              <td style="white-space:nowrap;">
                <button type="button" class="btn-secondary btn-sm pb-select-view-btn" data-view-id="${esc(v.view_id)}">${isSel ? 'Viewing' : 'Inspect'}</button>
              </td>
            </tr>`;
        }).join('')}
      </tbody>
    </table>`;

  wrap.querySelectorAll('.pb-select-view-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      PB_SELECTED_VIEW_ID = btn.dataset.viewId;
      renderTrackedViewsPane();
    });
  });
  wrap.querySelectorAll('.pb-cmp-chk').forEach(chk => {
    chk.addEventListener('change', () => {
      if (chk.checked) PB_COMPARE_IDS.add(chk.dataset.viewId);
      else PB_COMPARE_IDS.delete(chk.dataset.viewId);
      renderViewsComparisonChart();
    });
  });

  renderViewsComparisonChart();

  const selView = TRACKED_VIEWS.find(v => v.view_id === PB_SELECTED_VIEW_ID) || TRACKED_VIEWS[0];
  PB_SELECTED_VIEW_ID = selView.view_id;
  detailPanel.hidden = false;

  const st = selView.state;
  const isEnded = st.status === 'ENDED';
  document.getElementById('pbViewDetailTitle').textContent = selView.name;
  const statusBadge = document.getElementById('pbViewStatusBadge');
  statusBadge.className = isEnded ? 'badge' : 'badge badge-green';
  statusBadge.textContent = isEnded ? `ENDED ${st.ended_date || ''}` : 'ACTIVE · Auto-Syncing';
  document.getElementById('pbViewDetailSub').textContent = selView.do_nothing
    ? `Frozen Do-Nothing baseline anchored on ${st.inception_date}.`
    : `${selView.preset} · ${selView.slot_cap ? 'SlotCap ON' : 'Standard'} · ${selView.liquid_sweep ? 'LIQUIDCASE Sweep ON' : 'No Sweep'}${selView.respect_ltcg_shield ? ' · 330-364d LTCG Shield ON' : ''}`;

  const endBtn = document.getElementById('pbViewEndBtn');
  endBtn.disabled = isEnded;
  endBtn.style.opacity = isEnded ? '0.5' : '1';
  document.getElementById('pbViewHeldCount').textContent = String((st.holdings || []).length);

  renderLedgerKpis('pbViewKpis', st, true, true);

  const outEl = document.getElementById('pbViewOutsideNotice');
  if (selView.outside_holdings?.length && selView.outside_mode === 'exclude') {
    outEl.hidden = false;
    outEl.innerHTML = `<div><strong>Kept Outside Strategy Sleeve:</strong> ${selView.outside_holdings.map(h => `${esc(h.symbol)} (${h.shares} sh)`).join(', ')} — tracked separately outside the strategy NAV.</div>`;
  } else {
    outEl.hidden = true;
  }

  const companion = selView.companion_baseline_id
    ? TRACKED_VIEWS.find(v => v.view_id === selView.companion_baseline_id)?.state || null
    : null;
  renderLedgerLineChart('pbViewDetailChart', 'pbViewDetailChart', st, companion, true, 'pbViewEarlyState', 'pbViewDetailChartWrap');
  renderLedgerHoldingsTable('pbViewHoldingsWrap', st);
  renderLedgerAudit('pbViewDecisionFeed', 'pbViewTradeWrap', st, PB_VDEC_FILTER, 'ALL');
}

function renderViewsComparisonChart() {
  const cmpPanel = document.getElementById('pbViewsComparePanel');
  const chosen = TRACKED_VIEWS.filter(v => PB_COMPARE_IDS.has(v.view_id));
  if (!chosen.length || typeof Chart === 'undefined') {
    cmpPanel.hidden = true;
    return;
  }
  cmpPanel.hidden = false;
  const allDates = [...new Set(chosen.flatMap(v => (v.state.daily_history || []).map(h => h.date)))].sort();
  const palette = ['#059669', '#1d4ed8', '#d97706', '#dc2626', '#7c3aed', '#0891b2'];
  const datasets = chosen.map((v, idx) => {
    const cap = Number(v.state.initial_capital || 100000);
    const byDt = new Map((v.state.daily_history || []).map(h => [h.date, cap > 0 ? Number((((h.nav / cap) - 1) * 100).toFixed(2)) : 0]));
    return {
      label: v.name,
      data: allDates.map(d => byDt.has(d) ? byDt.get(d) : null),
      borderColor: palette[idx % palette.length],
      borderWidth: 2.2,
      borderDash: v.do_nothing ? [5, 4] : [],
      pointRadius: 3,
      tension: 0.1,
    };
  });
  if (CHARTS.pbViewsCompare) CHARTS.pbViewsCompare.destroy();
  CHARTS.pbViewsCompare = new Chart(document.getElementById('pbViewsCompareChart'), {
    type: 'line',
    data: { labels: allDates, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: { ticks: { color: '#64748b', callback: (v) => `${Number(Number(v).toFixed(2))}%` }, grid: { color: '#e2e8f0' } },
        x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { display: false } },
      },
      plugins: { legend: { position: 'bottom', labels: { color: '#334155', usePointStyle: true, boxWidth: 8 } } },
    },
  });
}

function renderPaperBotPage() {
  if (!STATE) return;
  updatePbViewsBadge();
  if (PB_SUBTAB === 'official') renderOfficialBotPane();
  else renderTrackedViewsPane();
}

// Paper Bot UI Event Bindings
document.getElementById('pbTabBtnOfficial').addEventListener('click', () => switchPaperBotSubTab('official'));
document.getElementById('pbTabBtnViews').addEventListener('click', () => switchPaperBotSubTab('views'));
document.getElementById('pbNewViewBtn').addEventListener('click', () => switchTab('rebalance'));

document.querySelectorAll('[data-pb-account]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-pb-account]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    PB_ACCOUNT = btn.dataset.pbAccount;
    renderOfficialBotPane();
  });
});
document.getElementById('pbNetDefTaxToggle').addEventListener('change', renderOfficialBotPane);
document.querySelectorAll('[data-pb-dec-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-pb-dec-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    PB_DEC_FILTER = btn.dataset.pbDecFilter;
    renderOfficialBotPane();
  });
});
document.querySelectorAll('[data-pb-trade-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-pb-trade-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    PB_TRADE_FILTER = btn.dataset.pbTradeFilter;
    renderOfficialBotPane();
  });
});
document.querySelectorAll('[data-pb-vdec-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-pb-vdec-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    PB_VDEC_FILTER = btn.dataset.pbVdecFilter;
    renderTrackedViewsPane();
  });
});
document.getElementById('pbExportOfficialCsvBtn').addEventListener('click', () => {
  const acc = PAPER_LEDGER?.accounts?.[PB_ACCOUNT];
  if (acc) downloadLedgerCsv(acc, `official_bot_${PB_ACCOUNT}`);
});
document.getElementById('pbExportViewCsvBtn').addEventListener('click', () => {
  const v = TRACKED_VIEWS.find(x => x.view_id === PB_SELECTED_VIEW_ID);
  if (v) downloadLedgerCsv(v.state, v.name.toLowerCase().replace(/[^a-z0-9]+/g, '_'));
});

// Hand Over to Bot Modal Workflow
function openHandOverModal() {
  computeHoldingsDiff();
  if (!LAST_DIFF_CTX || LAST_DIFF_CTX.targetPortfolioCap <= 0) {
    showToast('Paste holdings or enter cash above before handing over to the bot.');
    return;
  }
  const c = LAST_DIFF_CTX;
  document.getElementById('hoViewNameInput').value = `My Portfolio -> ${c.preset} (${STATE.as_of_date})`;
  document.getElementById('hoPresetSelect').value = c.preset;
  document.getElementById('hoSlotCapCheck').checked = c.slotCap;
  document.getElementById('hoSweepCheck').checked = c.sweep;
  document.getElementById('hoIncludeDoNothing').checked = true;
  const shieldHint = c.ltcgWarnCnt > 0
    ? ` · <strong>${c.ltcgWarnCnt} sell lot(s) are 330–364d old</strong>`
    : '';
  document.getElementById('hoPreviewMeta').innerHTML =
    `Starting Capital: <strong>${fmtInrInt(c.targetPortfolioCap)}</strong> (${c.inUnivLots.length} universe lot(s) + ${fmtInrInt(c.cashAvail + c.liquidateOutVal)} deployable cash)${shieldHint}`;
  document.getElementById('handOverModal').hidden = false;
}

document.getElementById('handOverBotBtn').addEventListener('click', openHandOverModal);
document.getElementById('hoCancelBtn').addEventListener('click', () => { document.getElementById('handOverModal').hidden = true; });
document.getElementById('hoCancelTopBtn').addEventListener('click', () => { document.getElementById('handOverModal').hidden = true; });

document.getElementById('hoSubmitBtn').addEventListener('click', () => {
  const c = LAST_DIFF_CTX;
  if (!c) return;
  const name = (document.getElementById('hoViewNameInput').value || '').trim() || `Portfolio -> ${c.preset} (${STATE.as_of_date})`;
  const preset = document.getElementById('hoPresetSelect').value;
  const slotCap = document.getElementById('hoSlotCapCheck').checked;
  const sweep = document.getElementById('hoSweepCheck').checked;
  const execMode = document.querySelector('input[name="hoExecMode"]:checked')?.value || 'immediate';
  const respectShield = execMode === 'ltcg_shield';
  const includeBaseline = document.getElementById('hoIncludeDoNothing').checked;

  const days = FORWARD_TAPE?.days || [getFallbackDay0FromState()];
  const day0 = days[days.length - 1];
  const uid = 'view_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
  const startCash = c.cashAvail + c.liquidateOutVal;
  const cloneSpec = {
    inception_date: day0.date,
    initial_capital: c.targetPortfolioCap,
    initial_lots: JSON.parse(JSON.stringify(c.inUnivLots)),
    starting_cash: startCash,
    outside_mode: c.outsideMode,
    outside_holdings: JSON.parse(JSON.stringify(c.outsideHoldings)),
  };

  let baselineId = null;
  let baselineView = null;
  if (includeBaseline && c.inUnivLots.length > 0) {
    baselineId = uid + '_base';
    const baseState = MMI_REPLAY.initializeAccountState({
      account_id: baselineId,
      label: `${name} (Do-Nothing)`,
      initial_capital: c.targetPortfolioCap,
      preset,
      slot_cap: false,
      liquid_sweep: false,
      do_nothing: true,
      initial_lots: cloneSpec.initial_lots,
      starting_cash: startCash,
    });
    MMI_REPLAY.stepPortfolio(baseState, day0, null);
    baseState.summary = MMI_REPLAY.summarizeAccount(baseState);
    baselineView = {
      view_id: baselineId,
      created_at: new Date().toISOString(),
      name: `${name} (Do-Nothing Baseline)`,
      preset,
      slot_cap: false,
      liquid_sweep: false,
      respect_ltcg_shield: false,
      do_nothing: true,
      outside_mode: c.outsideMode,
      outside_holdings: cloneSpec.outside_holdings,
      clone_spec: cloneSpec,
      state: baseState,
    };
  }

  const mainState = MMI_REPLAY.initializeAccountState({
    account_id: uid,
    label: name,
    initial_capital: c.targetPortfolioCap,
    preset,
    slot_cap: slotCap,
    liquid_sweep: sweep,
    respect_ltcg_shield: respectShield,
    do_nothing: false,
    initial_lots: cloneSpec.initial_lots,
    starting_cash: startCash,
  });
  MMI_REPLAY.stepPortfolio(mainState, day0, null);
  mainState.summary = MMI_REPLAY.summarizeAccount(mainState);

  const mainView = {
    view_id: uid,
    created_at: new Date().toISOString(),
    name,
    preset,
    slot_cap: slotCap,
    liquid_sweep: sweep,
    respect_ltcg_shield: respectShield,
    do_nothing: false,
    companion_baseline_id: baselineId,
    outside_mode: c.outsideMode,
    outside_holdings: cloneSpec.outside_holdings,
    clone_spec: cloneSpec,
    state: mainState,
  };

  TRACKED_VIEWS.unshift(mainView);
  PB_COMPARE_IDS.add(mainView.view_id);
  if (baselineView) {
    TRACKED_VIEWS.splice(1, 0, baselineView);
    PB_COMPARE_IDS.add(baselineView.view_id);
  }
  PB_SELECTED_VIEW_ID = mainView.view_id;
  saveTrackedViewsToStorage();

  document.getElementById('handOverModal').hidden = true;
  switchTab('paperbot');
  switchPaperBotSubTab('views');
  showToast(`Created Tracked View "${name}" (anchored ${day0.date}).`);
});

// Clone View Workflow
document.getElementById('pbViewCloneBtn').addEventListener('click', () => {
  const src = TRACKED_VIEWS.find(v => v.view_id === PB_SELECTED_VIEW_ID);
  if (!src) return;
  const alt = src.preset === 'R3' ? 'S3' : 'R3';
  document.getElementById('cloneNameInput').value = `${src.name.replace(/\s*\(Do-Nothing.*\)$/, '')} -> Clone ${alt}`;
  document.getElementById('clonePresetSelect').value = alt;
  document.getElementById('cloneSlotCapCheck').checked = src.slot_cap;
  document.getElementById('cloneSweepCheck').checked = src.liquid_sweep;
  document.getElementById('cloneViewModal').hidden = false;
});
document.getElementById('cloneCancelBtn').addEventListener('click', () => { document.getElementById('cloneViewModal').hidden = true; });
document.getElementById('cloneConfirmBtn').addEventListener('click', () => {
  const src = TRACKED_VIEWS.find(v => v.view_id === PB_SELECTED_VIEW_ID);
  if (!src || !src.clone_spec) return;
  const spec = src.clone_spec;
  const preset = document.getElementById('clonePresetSelect').value;
  const slotCap = document.getElementById('cloneSlotCapCheck').checked;
  const sweep = document.getElementById('cloneSweepCheck').checked;
  const name = (document.getElementById('cloneNameInput').value || '').trim() || `${src.name} (${preset})`;
  const uid = 'view_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);

  const clonedState = MMI_REPLAY.initializeAccountState({
    account_id: uid,
    label: name,
    initial_capital: spec.initial_capital,
    preset,
    slot_cap: slotCap,
    liquid_sweep: sweep,
    respect_ltcg_shield: src.respect_ltcg_shield,
    do_nothing: false,
    initial_lots: spec.initial_lots,
    starting_cash: spec.starting_cash,
  });

  const days = FORWARD_TAPE?.days || [getFallbackDay0FromState()];
  let startIdx = days.findIndex(d => d.date >= spec.inception_date);
  if (startIdx < 0) startIdx = days.length - 1;
  MMI_REPLAY.stepPortfolio(clonedState, days[startIdx], null);
  for (let i = startIdx + 1; i < days.length; i++) {
    MMI_REPLAY.stepPortfolio(clonedState, days[i], days[i - 1]);
  }
  clonedState.summary = MMI_REPLAY.summarizeAccount(clonedState);

  const clonedView = {
    view_id: uid,
    created_at: new Date().toISOString(),
    name,
    preset,
    slot_cap: slotCap,
    liquid_sweep: sweep,
    respect_ltcg_shield: src.respect_ltcg_shield,
    do_nothing: false,
    companion_baseline_id: src.companion_baseline_id || null,
    outside_mode: spec.outside_mode,
    outside_holdings: spec.outside_holdings,
    clone_spec: spec,
    state: clonedState,
  };
  TRACKED_VIEWS.unshift(clonedView);
  PB_COMPARE_IDS.add(uid);
  PB_SELECTED_VIEW_ID = uid;
  saveTrackedViewsToStorage();
  document.getElementById('cloneViewModal').hidden = true;
  renderTrackedViewsPane();
  showToast(`Cloned "${name}" and replayed ${clonedState.daily_history.length} day(s).`);
});

// End / Freeze View & Delete View Confirm Modal
function openConfirmModal(title, desc, btnText, onProceed) {
  document.getElementById('confirmModalTitle').textContent = title;
  document.getElementById('confirmModalDesc').textContent = desc;
  document.getElementById('confirmProceedBtn').textContent = btnText;
  PENDING_CONFIRM_FN = onProceed;
  document.getElementById('confirmViewModal').hidden = false;
}
document.getElementById('confirmCancelBtn').addEventListener('click', () => {
  PENDING_CONFIRM_FN = null;
  document.getElementById('confirmViewModal').hidden = true;
});
document.getElementById('confirmProceedBtn').addEventListener('click', () => {
  const fn = PENDING_CONFIRM_FN;
  PENDING_CONFIRM_FN = null;
  document.getElementById('confirmViewModal').hidden = true;
  if (fn) fn();
});

document.getElementById('pbViewEndBtn').addEventListener('click', () => {
  const v = TRACKED_VIEWS.find(x => x.view_id === PB_SELECTED_VIEW_ID);
  if (!v || v.state.status === 'ENDED') return;
  openConfirmModal(
    'End / Freeze Tracked View?',
    `Freezing "${v.name}" stops all future autonomous bot updates permanently and locks its final scorecard as of ${STATE.as_of_date}.`,
    'End / Freeze View',
    () => {
      v.state.status = 'ENDED';
      v.state.ended_date = v.state.last_synced_date || STATE.as_of_date;
      saveTrackedViewsToStorage();
      renderTrackedViewsPane();
      showToast(`Ended & froze "${v.name}".`);
    }
  );
});

document.getElementById('pbViewDeleteBtn').addEventListener('click', () => {
  const v = TRACKED_VIEWS.find(x => x.view_id === PB_SELECTED_VIEW_ID);
  if (!v) return;
  openConfirmModal(
    'Delete Tracked View?',
    `Permanently remove "${v.name}" and its local replay ledger from this browser?`,
    'Delete View',
    () => {
      TRACKED_VIEWS = TRACKED_VIEWS.filter(x => x.view_id !== v.view_id);
      PB_COMPARE_IDS.delete(v.view_id);
      PB_SELECTED_VIEW_ID = TRACKED_VIEWS[0]?.view_id || null;
      saveTrackedViewsToStorage();
      renderTrackedViewsPane();
      showToast(`Deleted "${v.name}".`);
    }
  );
});

// Export / Import All Views JSON
document.getElementById('pbExportViewsBtn').addEventListener('click', () => {
  const payload = { schema_version: 1, exported_at: new Date().toISOString(), views: TRACKED_VIEWS };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `midcap_tracked_views_${STATE?.as_of_date || 'backup'}.json`;
  a.click();
});
document.getElementById('pbImportViewsBtn').addEventListener('click', () => {
  document.getElementById('pbImportViewsFile').click();
});
document.getElementById('pbImportViewsFile').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (!file) return;
  try {
    const parsed = JSON.parse(await file.text());
    const incoming = Array.isArray(parsed) ? parsed : (parsed.views || []);
    if (!Array.isArray(incoming) || !incoming.length) {
      showToast('No valid tracked views found in JSON file.');
      return;
    }
    const existingIds = new Set(TRACKED_VIEWS.map(v => v.view_id));
    let added = 0;
    incoming.forEach(v => {
      if (v && v.view_id && v.state && !existingIds.has(v.view_id)) {
        TRACKED_VIEWS.push(v);
        PB_COMPARE_IDS.add(v.view_id);
        added += 1;
      }
    });
    if (FORWARD_TAPE) syncAllActiveTrackedViews(FORWARD_TAPE);
    if (!PB_SELECTED_VIEW_ID && TRACKED_VIEWS.length) PB_SELECTED_VIEW_ID = TRACKED_VIEWS[0].view_id;
    saveTrackedViewsToStorage();
    renderTrackedViewsPane();
    showToast(`Imported ${added} view(s) cleanly.`);
  } catch (err) {
    showToast('Failed to import JSON: ' + err.message);
  }
});

init().catch(err => {
  document.getElementById('regimeTitle').textContent = 'Unable to load market data';
  document.getElementById('regimeSubtitle').textContent = err.message;
});
