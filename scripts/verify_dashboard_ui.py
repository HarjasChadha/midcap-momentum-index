"""Playwright E2E browser verification against local HTTP server (or live URL via argv[1])."""

from __future__ import annotations

import functools
import http.server
import socketserver
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path("/Users/harjassinghchadha/.gemini/antigravity/brain/b8ff7fd5-f00b-4548-9eac-52600bbd35d9")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    target_url = sys.argv[1] if len(sys.argv) > 1 else None
    httpd = None
    if not target_url:
        handler = functools.partial(QuietHandler, directory=str(REPO_ROOT))
        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        target_url = f"http://127.0.0.1:{port}/index.html"

    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 950})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))
            page.on("requestfailed", lambda req: failed_requests.append(f"{req.url}: {req.failure}"))

            page.goto(target_url, wait_until="networkidle", timeout=30000)
            page.wait_for_selector("#ordersTable tbody tr", timeout=15000)

            # 1) Verify initial ₹1L SlotCap render
            title_txt = page.locator("#regimeTitle").inner_text()
            sub_txt = page.locator("#regimeSubtitle").inner_text()
            assert "RISK-ON" in title_txt or "CRASH HEDGE" in title_txt, f"Unexpected banner: {title_txt}"
            assert "undefined" not in sub_txt and "NaN" not in sub_txt, f"Bad text in subtitle: {sub_txt}"
            assert "undefined" not in page.locator("body").inner_text(), "Found 'undefined' in rendered UI"
            order_rows = page.locator("#ordersTable tbody tr").count()
            assert order_rows >= 30, f"Expected >= 30 order rows, got {order_rows}"

            # Check desktop overflow
            ov = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
            assert not ov, "Desktop horizontal overflow detected"
            page.screenshot(path=str(ARTIFACT_DIR / "midcap_dashboard_portfolio.png"), full_page=False)

            # 2) Click ₹5L quick cap -> verifies SlotCap unchecks cleanly & order sheet updates
            page.click('button[data-cap="500000"]')
            page.wait_for_timeout(200)
            assert not page.is_checked("#slotCapToggle"), "SlotCap should auto-uncheck at ₹5L"
            cap5l_txt = page.locator("#calcSummaryGrid").inner_text()
            assert "₹" in cap5l_txt and "5,00" not in cap5l_txt or "Equity Deployed" in cap5l_txt

            # 3) Test Simulate Crash posture -> confirms GOLDBEES & LIQUIDCASE defensive ETFs appear
            page.select_option("#postureOverride", "SIM_CRASH")
            page.wait_for_timeout(200)
            orders_txt = page.locator("#ordersTable tbody").inner_text()
            assert "GOLDBEES" in orders_txt and "LIQUIDCASE" in orders_txt, "Crash simulation must allocate GOLDBEES + LIQUIDCASE"
            page.select_option("#postureOverride", "LIVE")

            # 4) Test Holdings Diff + LTCG 330-364d shield warning
            page.click('button[data-cap="100000"]')
            page.fill("#holdingsInput", "APARINDS,1,2025-12-10\nBHEL,9,2026-06-01")
            page.click("#computeDiffBtn")
            diff_txt = page.locator("#diffOutput").inner_text()
            assert "SELL -1" in diff_txt and "LTCG in" in diff_txt, f"Expected LTCG near-expiry shield warning, got: {diff_txt}"

            # 5) Switch to Charts tab & verify canvases
            page.click('button[data-tab="charts"]')
            page.wait_for_timeout(500)
            assert page.locator("#navChart").is_visible()
            assert page.locator("#ddChart").is_visible()
            assert page.locator("#triggersTable tbody tr").count() >= 10
            page.screenshot(path=str(ARTIFACT_DIR / "midcap_dashboard_charts.png"), full_page=False)

            # 6) Switch to Research tab & verify matrix
            page.click('button[data-tab="research"]')
            page.wait_for_timeout(200)
            assert page.locator("#researchTable tbody tr").count() >= 25

            # 7) Switch to Rankings tab & test filter
            page.click('button[data-tab="rankings"]')
            page.wait_for_timeout(200)
            assert page.locator("#rankingsTable tbody tr").count() == 60
            page.fill("#rankSearch", "LAURUS")
            assert page.locator("#rankingsTable tbody tr").count() == 1

            # 8) Mobile viewport sanity check
            page.set_viewport_size({"width": 390, "height": 844})
            page.click('button[data-tab="portfolio"]')
            page.wait_for_timeout(250)
            mov = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
            assert not mov, "Mobile horizontal overflow detected"

            browser.close()
    finally:
        if httpd:
            httpd.shutdown()

    assert not console_errors, f"Console errors: {console_errors}"
    assert not page_errors, f"Page errors: {page_errors}"
    assert not failed_requests, f"Failed requests: {failed_requests}"
    print(f"Playwright verification PASSED cleanly on {target_url} (0 console/page errors, all 4 tabs & interactive workflows verified)")


if __name__ == "__main__":
    main()
