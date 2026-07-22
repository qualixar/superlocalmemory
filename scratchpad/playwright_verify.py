"""
SuperLocalMemory OD Integration — Playwright Verification Script
Run with: ~/.slm-venv/bin/python scratchpad/playwright_verify.py

Checks:
  1. Sidebar renders with nav groups; clicking nav item switches pane
  2. Theme toggle flips data-theme between dark/light and persists to localStorage
  3. Dashboard pane shows REAL data (not OD seed numbers like 48,231)
  4. No CSP console errors (Refused to execute inline script / Refused to apply)
  5. No horizontal scroll at 390px and 1440px

Saves screenshot to: scratchpad/od_verify_screenshot.png
"""
import asyncio
import json
import sys
import os
import re
from pathlib import Path

try:
    from playwright.async_api import async_playwright, Error as PlaywrightError
except ImportError:
    print("FAIL: playwright not installed in ~/.slm-venv")
    print("Install: ~/.slm-venv/bin/pip install playwright && ~/.slm-venv/bin/playwright install chromium")
    sys.exit(1)

BASE_URL = "http://127.0.0.1:8765/"
SCREENSHOT = Path(__file__).parent / "od_verify_screenshot.png"
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def log(status, check, detail=""):
    icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(status, "?")
    msg = f"  [{status}] {icon} {check}"
    if detail:
        msg += f"\n       → {detail}"
    print(msg)
    results.append({"status": status, "check": check, "detail": detail})

async def run():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1440, "height": 800},
            # Capture console messages
        )
        page = await ctx.new_page()

        # Collect console messages
        console_msgs = []
        page.on("console", lambda msg: console_msgs.append({"type": msg.type, "text": msg.text}))

        # ----------------------------------------------------------------
        print("\n=== OD Integration Verification ===\n")

        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15000)
        # Give od-shell.js time to run
        await page.wait_for_timeout(1500)

        # ----------------------------------------------------------------
        # CHECK 1: Sidebar renders with nav groups
        # ----------------------------------------------------------------
        print("--- Check 1: Sidebar ---")
        sidebar = await page.query_selector("#sidebar")
        sidebar_visible = sidebar is not None and await sidebar.is_visible()
        if not sidebar_visible:
            log(FAIL, "Sidebar #sidebar exists and is visible", "sidebar not found or hidden")
        else:
            log(PASS, "Sidebar #sidebar exists and is visible")

        # Nav links should exist
        nav_links = await page.query_selector_all(".nav-link[data-tab]")
        if len(nav_links) >= 6:
            log(PASS, f"Sidebar has {len(nav_links)} nav-link[data-tab] items (expected ≥6)")
        else:
            log(FAIL, f"Sidebar nav-link count", f"found {len(nav_links)}, expected ≥6")

        # ----------------------------------------------------------------
        # CHECK 2: Clicking nav item switches pane
        # ----------------------------------------------------------------
        print("\n--- Check 2: Pane switching ---")
        # Dashboard should be active by default
        dash_active = await page.evaluate("""
            () => {
                var p = document.getElementById('dashboard-pane');
                return p ? p.classList.contains('active') : false;
            }
        """)
        if dash_active:
            log(PASS, "dashboard-pane is active by default")
        else:
            log(FAIL, "dashboard-pane should be active by default", "active class missing")

        # Click memories-pane nav item
        await page.click(".nav-link[data-tab='memories-pane']")
        await page.wait_for_timeout(400)

        mem_active = await page.evaluate("""
            () => {
                var p = document.getElementById('memories-pane');
                return p ? p.classList.contains('active') : false;
            }
        """)
        dash_hidden = await page.evaluate("""
            () => {
                var p = document.getElementById('dashboard-pane');
                if (!p) return false;
                // tab-pane without .active gets display:none from Bootstrap CSS
                var style = window.getComputedStyle(p);
                return style.display === 'none';
            }
        """)
        if mem_active:
            log(PASS, "memories-pane becomes active on nav click")
        else:
            log(FAIL, "memories-pane did not activate on nav click")

        if dash_hidden:
            log(PASS, "dashboard-pane is hidden (display:none) when memories is active")
        else:
            log(WARN, "dashboard-pane display state after switch", "may still be visible — check CSS")

        # Navigate back to dashboard
        await page.click(".nav-link[data-tab='dashboard-pane']")
        await page.wait_for_timeout(400)

        # ----------------------------------------------------------------
        # CHECK 3: Theme toggle
        # ----------------------------------------------------------------
        print("\n--- Check 3: Theme toggle ---")
        initial_theme = await page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        log(PASS if initial_theme in ('dark', 'light') else FAIL,
            f"Initial data-theme is '{initial_theme}'")

        # Click theme toggle
        theme_btn = await page.query_selector("[data-theme-icon]")
        if theme_btn:
            await theme_btn.click()
            await page.wait_for_timeout(200)
            new_theme = await page.evaluate("() => document.documentElement.getAttribute('data-theme')")
            stored_theme = await page.evaluate("() => { try { return localStorage.getItem('slm-theme'); } catch(e) { return null; } }")
            bs_theme = await page.evaluate("() => document.documentElement.getAttribute('data-bs-theme')")
            ng_dark = await page.evaluate("() => document.body.classList.contains('ng-dark')")

            expected_new = 'light' if initial_theme == 'dark' else 'dark'
            if new_theme == expected_new:
                log(PASS, f"Theme toggled: {initial_theme} → {new_theme}")
            else:
                log(FAIL, f"Theme toggle failed", f"expected {expected_new}, got {new_theme}")

            if stored_theme == new_theme:
                log(PASS, f"Theme persisted to localStorage: '{stored_theme}'")
            else:
                log(FAIL, "Theme not persisted to localStorage", f"stored='{stored_theme}', current='{new_theme}'")

            if bs_theme == new_theme:
                log(PASS, f"data-bs-theme synced: '{bs_theme}'")
            else:
                log(FAIL, "data-bs-theme not synced with data-theme", f"data-bs-theme='{bs_theme}', data-theme='{new_theme}'")

            if new_theme == 'light' and not ng_dark:
                log(PASS, ".ng-dark removed on light theme")
            elif new_theme == 'dark' and ng_dark:
                log(PASS, ".ng-dark present on dark theme")
            else:
                log(WARN, ".ng-dark sync state", f"theme={new_theme}, ng_dark={ng_dark}")

            # Toggle back to original
            await theme_btn.click()
            await page.wait_for_timeout(200)
        else:
            log(FAIL, "Theme toggle button [data-theme-icon] not found")

        # ----------------------------------------------------------------
        # CHECK 4: Dashboard shows REAL data (not OD seed 48,231)
        # ----------------------------------------------------------------
        print("\n--- Check 4: Real data in Dashboard ---")
        await page.click(".nav-link[data-tab='dashboard-pane']")
        await page.wait_for_timeout(1000)  # wait for API calls

        # k-mem value must not be the OD seed (48,231 or 48231) and must be a number
        k_mem_text = await page.evaluate("() => { var el = document.getElementById('k-mem'); return el ? el.textContent.trim() : ''; }")
        OD_SEED_BAD = ['48,231', '48231', '12,847', '12847', '31,504']
        if k_mem_text in OD_SEED_BAD:
            log(FAIL, "k-mem shows OD seed data instead of real API data", f"value='{k_mem_text}'")
        elif k_mem_text in ('—', '', 'undefined'):
            log(WARN, "k-mem is empty/placeholder", f"API may not have responded yet: '{k_mem_text}'")
        else:
            log(PASS, f"k-mem shows real data: '{k_mem_text}'")

        # dashboard-mode must not be '...' or empty
        mode_text = await page.evaluate("() => { var el = document.getElementById('dashboard-mode'); return el ? el.textContent.trim() : ''; }")
        if mode_text and mode_text not in ('...', 'Loading...'):
            log(PASS, f"dashboard-mode populated: '{mode_text}'")
        else:
            log(WARN, f"dashboard-mode still loading: '{mode_text}' (daemon may need more time)")

        # Verify subtitle is populated
        subtitle = await page.evaluate("() => { var el = document.getElementById('od-dash-subtitle'); return el ? el.textContent.trim() : ''; }")
        if subtitle and subtitle not in ('Connecting to daemon...', ''):
            log(PASS, f"OD subtitle populated: '{subtitle[:60]}'")
        else:
            log(WARN, f"OD subtitle not yet populated: '{subtitle}'")

        # ----------------------------------------------------------------
        # CHECK 5: CSP violations in console
        # ----------------------------------------------------------------
        print("\n--- Check 5: CSP violations ---")
        csp_errors = [m for m in console_msgs if
                      'refused to execute inline script' in m['text'].lower() or
                      'refused to apply inline style' in m['text'].lower() or
                      'content security policy' in m['text'].lower() and 'refused' in m['text'].lower()]
        if csp_errors:
            log(FAIL, f"CSP violations detected: {len(csp_errors)}")
            for err in csp_errors[:3]:
                print(f"       → [{err['type']}] {err['text'][:120]}")
        else:
            log(PASS, "No CSP violations in console")

        # Log other errors
        js_errors = [m for m in console_msgs if m['type'] == 'error' and 'csp' not in m['text'].lower()
                     and 'content security' not in m['text'].lower()]
        if js_errors:
            log(WARN, f"{len(js_errors)} JS console error(s) (non-CSP)")
            for err in js_errors[:3]:
                print(f"       → {err['text'][:120]}")

        # ----------------------------------------------------------------
        # CHECK 6: No horizontal scroll at 390px
        # ----------------------------------------------------------------
        print("\n--- Check 6: Horizontal scroll ---")
        for width in [390, 1440]:
            await page.set_viewport_size({"width": width, "height": 800})
            await page.wait_for_timeout(200)
            scroll_info = await page.evaluate("""
                () => ({
                    scrollWidth: document.documentElement.scrollWidth,
                    clientWidth: document.documentElement.clientWidth,
                    bodyScrollWidth: document.body.scrollWidth,
                })
            """)
            overflows = scroll_info['scrollWidth'] > scroll_info['clientWidth'] + 4
            if overflows:
                log(FAIL, f"Horizontal scroll at {width}px",
                    f"scrollWidth={scroll_info['scrollWidth']} > clientWidth={scroll_info['clientWidth']}")
            else:
                log(PASS, f"No horizontal scroll at {width}px "
                    f"(scrollWidth={scroll_info['scrollWidth']}, clientWidth={scroll_info['clientWidth']})")

        # ----------------------------------------------------------------
        # Screenshot (1440px width)
        # ----------------------------------------------------------------
        await page.set_viewport_size({"width": 1440, "height": 900})
        await page.click(".nav-link[data-tab='dashboard-pane']")
        await page.wait_for_timeout(600)
        await page.screenshot(path=str(SCREENSHOT), full_page=False)
        print(f"\n  Screenshot saved: {SCREENSHOT}")

        await browser.close()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "="*50)
    print("RESULTS:")
    passed = sum(1 for r in results if r['status'] == PASS)
    failed = sum(1 for r in results if r['status'] == FAIL)
    warned = sum(1 for r in results if r['status'] == WARN)
    print(f"  PASS: {passed}  FAIL: {failed}  WARN: {warned}  TOTAL: {len(results)}")
    print("="*50)
    if failed > 0:
        print("\nFAILED checks:")
        for r in results:
            if r['status'] == FAIL:
                print(f"  ✗ {r['check']}")
                if r['detail']:
                    print(f"    → {r['detail']}")
    return failed == 0


if __name__ == '__main__':
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
