"""Headless Chromium Playwright E2E + Python-to-JS engine parity verification."""

import functools
import http.server
import json
import math
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.paper_bot import run_synthetic_steps  # noqa: E402
from tests.test_paper_bot import build_synthetic_parity_tape  # noqa: E402


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def start_local_server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    handler = functools.partial(QuietHandler, directory=str(REPO_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}/index.html"


def assert_python_js_parity(page):
    """Run identical 10-day synthetic tape through Python paper_bot and JS window.mmiForwardReplay."""
    tape = build_synthetic_parity_tape()
    py_res = run_synthetic_steps(
        tape,
        account_id="PARITY_R3",
        label="Parity 10D",
        initial_capital=250_000.0,
        preset="R3",
        slot_cap=True,
        liquid_sweep=True,
    )
    js_res = page.evaluate(
        """(tape) => window.mmiForwardReplay.runSyntheticSteps(tape, {
            account_id: 'PARITY_R3',
            label: 'Parity 10D',
            initial_capital: 250000.0,
            preset: 'R3',
            slot_cap: true,
            liquid_sweep: true,
        })""",
        tape,
    )

    py_last = py_res["daily_history"][-1]
    js_last = js_res["daily_history"][-1]
    assert abs(py_last["nav"] - js_last["nav"]) < 0.05, f"NAV drift: py={py_last['nav']} vs js={js_last['nav']}"
    assert abs(py_last["cash"] - js_last["cash"]) < 0.05, f"Cash drift: py={py_last['cash']} vs js={js_last['cash']}"
    assert len(py_res["trade_log"]) == len(js_res["trade_log"]), f"Trades count mismatch: py={len(py_res['trade_log'])} vs js={len(js_res['trade_log'])}"

    py_pos = {h["symbol"]: h["shares"] for h in py_res["holdings"]}
    js_pos = {h["symbol"]: h["shares"] for h in js_res["holdings"]}
    assert py_pos == js_pos, f"Holdings mismatch: py={py_pos} vs js={js_pos}"
    print(f"PASSED Python-vs-JS 10-day parity: NAV=₹{py_last['nav']:,.2f}, trades={len(py_res['trade_log'])}, holdings={len(py_pos)}")


def verify_target(url: str, shot_dir: Path | None = None, is_local: bool = True):
    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page = context.new_page()

        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(url, wait_until="networkidle", timeout=35000)
        page.wait_for_function("() => document.getElementById('asOfDate')?.textContent !== '—'", timeout=15000)

        as_of = page.inner_text("#asOfDate").strip()
        assert len(as_of) == 10, f"Unexpected as_of_date: {as_of}"
        assert page.locator("#ordersTable tbody tr").count() >= 10, "Too few orders rendered"

        # 0. P2 Content, SEBI Disclaimer, Dynamic Banner & Header checks on Action & Orders
        sidebar_disc = page.inner_text("#sidebarDisclaimer")
        footer_disc = page.inner_text("#footerDisclaimer")
        assert "Not SEBI Registered" in sidebar_disc and "Not Investment Advice" in footer_disc
        regime_sub = page.inner_text("#regimeSubtitle")
        assert "All 4 market health signals support" not in regime_sub, f"Contradictory static banner still present: {regime_sub}"
        assert "Composite score" in regime_sub and "-0.10 cautious threshold" in regime_sub
        assert "Model Target Qty" in page.inner_text("#ordersTable thead")

        # Test explicit negative composite_smooth / mixed blocks dynamic banner case cleanly
        banner_diag = page.evaluate(
            """() => {
                const orig = JSON.parse(JSON.stringify(STATE.regime));
                STATE.regime.state = 'RISK_ON';
                STATE.regime.multifactor_blocks.b1_trend = -0.45;
                STATE.regime.multifactor_blocks.b2_vol = 0.12;
                STATE.regime.multifactor_blocks.b3_breadth = 0.40;
                STATE.regime.multifactor_blocks.b4_macro_gold = 0.25;
                STATE.regime.multifactor_blocks.composite_smooth = -0.04;
                renderHeaderAndRegime();
                const txt = document.getElementById('regimeSubtitle').textContent;
                STATE.regime = orig;
                renderHeaderAndRegime();
                return txt;
            }"""
        )
        assert "All 4 market health signals support" not in banner_diag
        assert "3 of 4 blocks positive" in banner_diag

        # 1. Exact Python-vs-JS Engine Parity Check
        assert_python_js_parity(page)

        # 1b. Sidebar must contain zero non-clickable nav group headers & all nav buttons work
        assert page.locator(".nav-group-label").count() == 0, "Non-clickable sidebar group labels should be removed"

        # 2. Walk every sidebar tab cleanly + verify P2 Tab requirements
        for tab in ("guide", "charts", "rankings", "research", "paperbot"):
            page.click(f'button.nav-btn[data-tab="{tab}"]')
            page.wait_for_timeout(350)
            assert page.locator(f"#tab-{tab}.active").count() == 1, f"Tab {tab} did not activate"

        # Verify Start Here · Guide interactive simulator, presets, SlotCap toggle & PDF download
        page.click('button.nav-btn[data-tab="guide"]')
        page.wait_for_timeout(250)
        assert as_of in page.inner_text("#guideLiveWeatherBanner")
        page.click('button[data-guide-scen="CRASH"]')
        page.wait_for_timeout(150)
        assert "DEFENSIVE" in page.inner_text("#guideSimKicker")
        assert "Stocks 25%" in page.inner_text("#guideSimLblEq")
        page.click('button[data-guide-scen="THRUST"]')
        page.wait_for_timeout(150)
        assert "RISK_ON" in page.inner_text("#guideSimKicker")
        assert "Stocks 100%" in page.inner_text("#guideSimLblEq")
        page.click('button[data-guide-preset="R2"]')
        page.wait_for_timeout(150)
        assert "15.17%" in page.inner_text("#guidePresetCard")
        page.click('button[data-guide-capmode="10L"]')
        page.wait_for_timeout(150)
        assert "41,667" in page.inner_text("#guideBasketExplainer")
        page.click('button[data-guide-capmode="1L"]')
        with page.expect_download() as dl_info:
            page.click("#guideDownloadPdfBtn")
        dl = dl_info.value
        assert dl.suggested_filename == "Momentum_Strategy_Mathematical_Specification.pdf", f"Unexpected download name: {dl.suggested_filename}"
        dl_path = dl.path()
        assert dl_path and Path(dl_path).stat().st_size > 500_000, "Downloaded PDF looks too small"
        if shot_dir:
            page.screenshot(path=str(shot_dir / "verify_guide_page.png"), full_page=True)
        # Verify Load preset CTA routes cleanly to portfolio tab with R2 selected
        page.click("#guideLoadPresetBtn")
        page.wait_for_timeout(200)
        assert page.locator("#tab-portfolio.active").count() == 1
        assert page.input_value("#presetSelect") == "R2"
        page.click("#toggleAdvancedBtn")
        page.wait_for_timeout(100)
        page.select_option("#presetSelect", "R3")
        page.click("#toggleAdvancedBtn")

        # Verify Performance & Risk tab (16Y Hero first + Survivorship disclosure + ? tooltips + no 4-line key)
        page.click('button.nav-btn[data-tab="charts"]')
        page.wait_for_timeout(250)
        assert page.locator("#pitPerfSummaryGrid .metric-cell").count() >= 4
        assert "Survivorship-Biased" in page.inner_text("#survivorshipBadge")
        assert "Survivorship-Bias Disclosure" in page.inner_text("#survivorshipDisclosureBanner")
        assert page.locator("#tab-charts .intuition-key").count() == 0, "Old bulky glossary block should be removed in favor of ? tooltips"
        assert page.locator("#pitPerfSummaryGrid .tooltip-q").count() >= 4

        # Verify Top 60 Stocks headers & New Listing metadata
        page.click('button.nav-btn[data-tab="rankings"]')
        page.wait_for_timeout(250)
        rk_head = page.inner_text("#rankingsTable thead")
        assert "Above 50-DMA (Trim Signal)" in rk_head
        assert "Listing Days" in rk_head
        assert "New Listing" in page.inner_text("#rankingsTable tbody")

        # Verify Strategy & Proof PDF path, Post-Selection P3 label, and zero stale copy
        page.click('button.nav-btn[data-tab="research"]')
        page.wait_for_timeout(250)
        pdf_link = page.locator('#tab-research a[href$=".pdf"]')
        pdf_href = pdf_link.get_attribute("href")
        assert pdf_href == "./docs/Momentum_Strategy_Mathematical_Specification.pdf", f"Unexpected PDF href: {pdf_href}"
        assert pdf_link.get_attribute("download") == "Momentum_Strategy_Mathematical_Specification.pdf"
        assert "2021–26 (Post-Selection)" in page.inner_text("#pitResearchTable thead")
        assert "2021–26 Post-Selection" in page.inner_text("#vaultIntuition")
        research_txt = page.inner_text("#tab-research")
        for stale in ("Blind Test 2", "15.79%", "18.16%"):
            assert stale not in research_txt, f"Stale copy '{stale}' still present in Strategy & Proof"
        assert "15.17%" in research_txt and "15.61%" in research_txt

        # Return to Paper Trading Bot for Tab A & Tab B checks
        page.click('button.nav-btn[data-tab="paperbot"]')
        page.wait_for_timeout(250)

        # 3. Verify Paper Trading Bot -> Tab A (Official Autonomous Bot)
        assert page.is_checked("#pbNetDefTaxToggle"), "#pbNetDefTaxToggle must default to checked (tax_defensive=True)"
        assert page.locator("#pbOfficialKpis .metric-cell").count() == 5
        pill_txt = page.inner_text("#pbOfficialStatusPill")
        assert "Active since" in pill_txt
        assert "1 trading days" not in pill_txt, f"Plural typo on 1 trading day: {pill_txt}"
        # Check Early-Inception Summary state renders cleanly when < 5 trading days
        early_days = page.evaluate("() => (PAPER_LEDGER?.accounts?.R3_1L_SlotCap?.daily_history || []).length")
        if early_days < 5:
            assert not page.locator("#pbOfficialEarlyState").is_hidden(), "Expected Early-Inception card for <5 days"
            assert page.locator("#pbOfficialChartWrap").is_hidden(), "Chart wrap should be hidden when <5 days"
        assert page.locator("#pbOfficialHoldingsWrap tbody tr").count() >= 25
        page.click('button[data-pb-account="R3_10L_Standard"]')
        page.wait_for_timeout(250)
        assert page.locator("#pbOfficialHoldingsWrap tbody tr").count() >= 25
        page.click('button[data-pb-account="R3_1L_SlotCap"]')
        if shot_dir:
            page.screenshot(path=str(shot_dir / "verify_paperbot_official.png"), full_page=True)

        # 4. Full Rebalance & Tax Diff -> Hand Over to Bot -> My Tracked Views flow
        page.click('button.nav-btn[data-tab="rebalance"]')
        page.wait_for_timeout(250)

        # Pick an in-universe stock outside Top 30 dynamically from window.STATE so it triggers SELL + 345d LTCG shield!
        sample_setup = page.evaluate(
            """() => {
                const asOf = new Date(STATE.as_of_date + 'T00:00:00Z');
                const d345 = new Date(asOf.getTime() - 345 * 86400000).toISOString().slice(0, 10);
                const sorted = [...STATE.universe_scores].sort((a, b) => (b.rk9 || -99) - (a.rk9 || -99));
                const weakSym = sorted[Math.min(45, sorted.length - 1)].sym;
                const weakPx = sorted[Math.min(45, sorted.length - 1)].px;
                const topSym = sorted[0].sym;
                const topPx = sorted[0].px;
                return { d345, weakSym, weakBuy: Math.max(10, Math.round(weakPx * 0.7)), topSym, topBuy: Math.max(10, Math.round(topPx * 0.85)) };
            }"""
        )
        csv_text = (
            f"{sample_setup['weakSym']},25,{sample_setup['d345']},{sample_setup['weakBuy']}\n"
            f"{sample_setup['topSym']},5,2026-06-01,{sample_setup['topBuy']}\n"
            f"RELIANCE,4,2026-02-01,2800\n"
            f"GOLDBEES,150,2026-03-10,98"
        )
        page.fill("#holdingsInput", csv_text)
        page.fill("#rebalanceCashInput", "25000")
        page.click("#computeDiffBtn")
        page.wait_for_timeout(300)

        assert not page.locator("#outsideUniverseBox").is_hidden(), "Outside Midcap 150 notice should be visible"
        assert "1" in page.inner_text("#outsideCountLbl")
        assert int(page.inner_text("#diffTaxFlags")) >= 1, "Expected >=1 330-364d LTCG shield warning"

        # Toggle Outside -> Liquidate into Strategy with Sell Price
        page.click('button[data-outside-mode="liquidate"]')
        page.fill('input.outside-px-inp[data-outside-sym="RELIANCE"]', "3000")
        page.wait_for_timeout(200)
        assert "RELIANCE" in page.inner_text("#diffOutput")

        # Hand Over to Bot -> Respect 330-364d LTCG Shield + Do-Nothing Baseline
        page.click("#handOverBotBtn")
        page.wait_for_timeout(200)
        assert not page.locator("#handOverModal").is_hidden()
        page.fill("#hoViewNameInput", "Zerodha E2E R3 View")
        page.click('input[name="hoExecMode"][value="ltcg_shield"]')
        page.click("#hoSubmitBtn")
        page.wait_for_timeout(400)

        assert page.locator("#tab-paperbot.active").count() == 1
        assert not page.locator("#pbPaneViews").is_hidden()
        assert page.inner_text("#pbViewsCount").strip() == "2"

        # Verify Clone with Different Strategy works cleanly
        page.click("#pbViewCloneBtn")
        page.wait_for_timeout(150)
        page.fill("#cloneNameInput", "Zerodha E2E S3 Clone")
        page.select_option("#clonePresetSelect", "S3")
        page.click("#cloneConfirmBtn")
        page.wait_for_timeout(300)
        assert page.inner_text("#pbViewsCount").strip() == "3"

        # Simulate +1 day tape advance crossing 365d LTCG threshold so shield matures!
        if is_local:
            synced = page.evaluate(
                """(weakSym) => {
                    const lastDay = FORWARD_TAPE.days[FORWARD_TAPE.days.length - 1];
                    const nextDt = new Date(Date.parse(lastDay.date + 'T00:00:00Z') + 25 * 86400000).toISOString().slice(0, 10);
                    const nextDay = JSON.parse(JSON.stringify(lastDay));
                    nextDay.date = nextDt;
                    Object.keys(nextDay.prices).forEach(k => { nextDay.prices[k] = Number((nextDay.prices[k] * 1.015).toFixed(2)); });
                    nextDay.regime.ew_index_val = Number((nextDay.regime.ew_index_val * 1.015).toFixed(2));
                    const extendedTape = { ...FORWARD_TAPE, last_date: nextDt, days: [...FORWARD_TAPE.days, nextDay] };
                    return window.mmiSyncPaperTape(extendedTape);
                }""",
                sample_setup["weakSym"],
            )
            assert synced >= 3, f"Expected all 3 active views to catch up (+1 day each), synced={synced}"
            page.wait_for_timeout(300)
            # Switch back to the R3 Shielded View and verify LTCG_SHIELD_MATURED fired!
            page.locator('#pbViewsOverviewWrap tbody tr:has-text("Zerodha E2E R3 View") button.pb-select-view-btn').first.click()
            page.wait_for_timeout(200)
            feed_txt = page.inner_text("#pbViewDecisionFeed")
            assert "LTCG_SHIELD_MATURED" in feed_txt, f"Expected LTCG_SHIELD_MATURED in feed: {feed_txt}"

        if shot_dir:
            page.screenshot(path=str(shot_dir / "verify_tracked_views.png"), full_page=True)

        # Export All Views JSON & Test Import round-trip
        with page.expect_download() as dl_info:
            page.click("#pbExportViewsBtn")
        dl = dl_info.value
        with tempfile.TemporaryDirectory() as tmpd:
            exp_file = Path(tmpd) / "exported_views.json"
            dl.save_as(str(exp_file))
            exported_json = json.loads(exp_file.read_text())
            assert len(exported_json.get("views", [])) == 3

            # End / Freeze View
            page.click("#pbViewEndBtn")
            page.click("#confirmProceedBtn")
            page.wait_for_timeout(200)
            assert "ENDED" in page.inner_text("#pbViewStatusBadge")

            # Verify advancing tape another day leaves ENDED view frozen while active sibling advances
            if is_local:
                synced2 = page.evaluate(
                    """() => {
                        const lastDay = FORWARD_TAPE.days[FORWARD_TAPE.days.length - 1];
                        const nextDt = new Date(Date.parse(lastDay.date + 'T00:00:00Z') + 2 * 86400000).toISOString().slice(0, 10);
                        const nextDay = JSON.parse(JSON.stringify(lastDay));
                        nextDay.date = nextDt;
                        const extendedTape = { ...FORWARD_TAPE, last_date: nextDt, days: [...FORWARD_TAPE.days, nextDay] };
                        return window.mmiSyncPaperTape(extendedTape);
                    }"""
                )
                assert synced2 == 2, f"Ended view should stay frozen (only 2 active left), synced2={synced2}"

            # Clear views & import JSON back cleanly
            page.evaluate("() => { TRACKED_VIEWS = []; saveTrackedViewsToStorage(); renderTrackedViewsPane(); }")
            assert page.inner_text("#pbViewsCount").strip() == "0"
            page.set_input_files("#pbImportViewsFile", str(exp_file))
            page.wait_for_timeout(300)
            assert page.inner_text("#pbViewsCount").strip() == "3"

        # 5. Desktop horizontal overflow check + Mobile 390x844 check
        overflow_dt = page.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 2")
        assert not overflow_dt, "Desktop horizontal overflow detected"

        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(300)
        overflow_mob = page.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 2")
        assert not overflow_mob, "Mobile horizontal overflow detected on Paper Trading Bot page"
        if shot_dir:
            page.screenshot(path=str(shot_dir / "verify_paperbot_mobile.png"), full_page=True)

        assert not console_errors, f"Console errors: {console_errors}"
        assert not page_errors, f"Page errors: {page_errors}"
        browser.close()


def main():
    shot_dir = Path(os.environ.get("SHOT_DIR", Path(tempfile.gettempdir()) / "mmi_ui_shots"))
    shot_dir.mkdir(parents=True, exist_ok=True)
    target = os.environ.get("DASHBOARD_URL")
    if target:
        verify_target(target, shot_dir=shot_dir, is_local=False)
        print(f"PASSED live dashboard UI E2E for {target}")
        return

    server, local_url = start_local_server()
    try:
        verify_target(local_url, shot_dir=shot_dir, is_local=True)
        print("PASSED local Playwright E2E + Python-to-JS parity + mobile layout checks")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
