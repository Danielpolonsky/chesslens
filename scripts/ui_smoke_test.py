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
PAGES = ["Overview", "Performance", "Openings", "Mistake Explorer", "Coaching", "ML Insights", "Methodology"]


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
        page.get_by_role("link", name=name).first.click()
        wait_idle(page)
        assert_no_exception(page, f"{label} / {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--chesscom", default=None, help="Chess.com username for a live import test")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})

        page.goto(args.url)
        wait_idle(page)
        assert_no_exception(page, "home")
        page.get_by_role("button", name="Try Demo").click()
        wait_idle(page)
        visit_all(page, "demo")

        # Mistake explorer navigation
        page.get_by_role("link", name="Mistake Explorer").first.click()
        wait_idle(page)
        page.get_by_role("button", name="Next").click()
        wait_idle(page)
        assert "Mistake 2 of" in page.inner_text("body"), "Next button did not advance"
        assert_no_exception(page, "demo / explorer next")

        # Coaching -> view example jumps to the explorer
        page.get_by_role("link", name="Coaching").first.click()
        wait_idle(page)
        page.get_by_role("button", name="View example 1").first.click()
        wait_idle(page)
        assert "Mistake Explorer" in page.inner_text("body")
        assert_no_exception(page, "coaching -> explorer")

        # Upload + analysis flow
        page.get_by_role("link", name="Home").first.click()
        wait_idle(page)
        page.locator("input[type=file]").set_input_files(str(SAMPLE))
        time.sleep(1.5)
        page.get_by_role("button", name="Import games").click()
        wait_idle(page)
        body = page.inner_text("body")
        assert "Imported 12 new games" in body, body[:1500]
        print("ok   PGN upload imported 12 games")
        page.get_by_role("button", name="Analyze").filter(has_text="with Stockfish").click()
        started = time.time()
        while time.time() - started < 300:
            time.sleep(2)
            if "Analyzed 12 games" in page.inner_text("body"):
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
        assert "Imported 0 new games (12 duplicates skipped)" in page.inner_text("body")
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
            page.get_by_role("button", name="Analyze").filter(has_text="with Stockfish").click()
            started = time.time()
            while time.time() - started < 300:
                time.sleep(2)
                if re.search(r"Analyzed \d+ games", page.inner_text("body")):
                    break
            else:
                raise SystemExit("FAIL: Chess.com analysis did not finish")
            wait_idle(page)
            page.get_by_role("link", name="Performance").first.click()
            wait_idle(page)
            assert_no_exception(page, "chess.com / Performance")
            has_clock = "low-clock error rate" in page.inner_text("body").lower()
            print(f"ok   Chess.com games analyzed; clock statistics shown: {has_clock}")
            visit_all(page, "chess.com")

        browser.close()
    print("UI smoke test passed")


if __name__ == "__main__":
    sys.exit(main())
