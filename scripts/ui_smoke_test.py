"""End-to-end UI check against a running ChessLens server (manual verification aid).

    streamlit run app.py
    python scripts/ui_smoke_test.py [--chesscom USERNAME]

Visits every page in demo mode, uploads the sample PGN, runs Stockfish analysis, opens
every dashboard page on the imported data, and optionally imports from Chess.com.
Fails loudly if any page shows a Streamlit exception.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "lichess_beibus_2013-01.pgn"
PAGES = ["Dashboard", "Review mistakes", "Practice", "ML insights", "Methodology"]


def wait_idle(page: Page, timeout: float = 120) -> None:
    page.wait_for_function(
        """() => {const a = document.querySelector('[data-testid=stApp]');
                 return a && a.getAttribute('data-test-script-state') === 'notRunning';}""",
        timeout=timeout * 1000,
    )
    time.sleep(1.0)


def assert_no_exception(page: Page, where: str) -> None:
    n = page.locator("[data-testid=stException]").count()
    if n:
        print(page.locator("[data-testid=stException]").first.inner_text()[:2000])
        raise SystemExit(f"FAIL: exception rendered on {where}")
    print(f"ok   {where}")


def visit_all(page: Page, label: str) -> None:
    for name in PAGES:
        link = page.get_by_role("link", name=name).first
        if name in ("ML insights", "Methodology") and not link.is_visible():
            page.get_by_role("button", name="Technical").click()
            wait_idle(page)
        link.click()
        wait_idle(page)
        assert_no_exception(page, f"{label} / {name}")
        if name == "Dashboard":
            for view in ("Trends", "Error patterns", "Openings", "Overview"):
                page.get_by_text(view, exact=True).first.click()
                wait_idle(page)
                assert_no_exception(page, f"{label} / Dashboard / {view}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--chesscom", default=None, help="Chess.com username for a live import test")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        desktop = browser.new_page(viewport={"width": 1440, "height": 1000})
        page = desktop

        page.goto(args.url)
        wait_idle(page)
        assert_no_exception(page, "start")
        page.get_by_role("button", name="Try demo").click()
        wait_idle(page)
        visit_all(page, "demo")

        page.get_by_role("link", name="Review mistakes").first.click()
        wait_idle(page)
        page.get_by_role("button", name="Next").click()
        wait_idle(page)
        assert "Mistake 2 of" in page.inner_text("body"), "Next button did not advance"
        assert_no_exception(page, "demo / explorer next")

        page.get_by_role("link", name="Practice").first.click()
        wait_idle(page)
        page.get_by_role("button", name="Open this position").first.click()
        wait_idle(page)
        assert "Review mistakes" in page.inner_text("body")
        assert_no_exception(page, "practice -> mistake")

        narrow = browser.new_page(viewport={"width": 390, "height": 844})
        narrow.goto(args.url + "/dashboard?demo=1")
        wait_idle(narrow)
        assert_no_exception(narrow, "narrow / dashboard")
        # The top nav can sit outside a 390px viewport; open the page directly.
        narrow.goto(args.url + "/mistakes?demo=1")
        wait_idle(narrow)
        assert_no_exception(narrow, "narrow / mistakes")
        narrow.close()

        page.get_by_role("link", name="Start").first.click()
        wait_idle(page)
        page.locator("input[type=file]").set_input_files(str(SAMPLE))
        time.sleep(1.5)
        page.get_by_role("button", name="Import games").click()
        wait_idle(page)
        body = page.inner_text("body")
        assert "Imported 12 new games" in body, body[:1500]
        print("ok   PGN upload imported 12 games")
        page.get_by_role("button", name="Analyze 12 games").click()
        started = time.time()
        while time.time() - started < 300:
            time.sleep(2)
            if "Analyzed 12 of 12" in page.inner_text("body"):
                break
        else:
            raise SystemExit("FAIL: analysis did not finish in 5 minutes")
        print(f"ok   Stockfish analysis finished in {time.time() - started:.0f}s")
        wait_idle(page)
        visit_all(page, "uploaded")

        # Re-importing the same file must not create duplicates
        page.get_by_role("link", name="Home").first.click()
        wait_idle(page)
        page.locator("input[type=file]").set_input_files(str(SAMPLE))
        time.sleep(1.5)
        page.get_by_role("button", name="Import games").click()
        wait_idle(page)
        assert "No new games: all 12 valid games were already imported." in page.inner_text("body")
        print("ok   duplicate upload detected")

        if args.chesscom:
            page.get_by_role("tab", name="Import from Chess.com").click()
            page.get_by_label("Chess.com username").fill(args.chesscom)
            page.get_by_label("Recent games").fill("10")
            page.get_by_role("button", name="Download games").click()
            wait_idle(page, timeout=180)
            body = page.inner_text("body")
            match = re.search(r"Imported (\d+) new games", body)
            assert match and int(match.group(1)) > 0, body[:1500]
            print(f"ok   Chess.com import: {match.group(0)}")
            page.get_by_role("button", name=re.compile(r"Analyze \d+ games?")).click()
            started = time.time()
            while time.time() - started < 300:
                time.sleep(2)
                if re.search(r"Analyzed \d+ of \d+", page.inner_text("body")):
                    break
            else:
                raise SystemExit("FAIL: Chess.com analysis did not finish")
            wait_idle(page)
            page.get_by_role("link", name="Dashboard").first.click()
            wait_idle(page)
            page.get_by_text("Error patterns", exact=True).first.click()
            wait_idle(page)
            assert_no_exception(page, "chess.com / error patterns")
            body_lower = page.inner_text("body").lower()
            has_clock = "low-clock error rate" in body_lower
            assert has_clock or "clock data is not in these games" in body_lower
            print(f"ok   Chess.com games analyzed; clock statistics shown: {has_clock}")
            visit_all(page, "chess.com")

        browser.close()
    print("UI smoke test passed")


if __name__ == "__main__":
    sys.exit(main())
