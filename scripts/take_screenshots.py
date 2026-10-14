"""Capture README screenshots of a running ChessLens instance.

    streamlit run app.py            # in one terminal
    python scripts/take_screenshots.py --url http://localhost:8501

Requires ``pip install playwright``; uses the locally installed Microsoft Edge or Chrome
(``--channel``) so no browser download is needed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
PAGES = [
    ("start", "", 1100),
    ("dashboard", "dashboard?demo=1", 1600),
    ("mistakes", "mistakes?demo=1", 1100),
    ("practice", "practice?demo=1", 1200),
    ("ml_insights", "ml?demo=1", 1700),
    ("methodology", "methodology?demo=1", 1400),
]

WAIT_JS = """() => {
  const app = document.querySelector('[data-testid=stApp]');
  return app && app.getAttribute('data-test-script-state') === 'notRunning'
         && document.body.innerText.length > 400;
}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        for name, path, height in PAGES:
            if args.only and name not in args.only:
                continue
            page = browser.new_page(viewport={"width": 1440, "height": height}, device_scale_factor=1)
            page.goto(f"{args.url.rstrip('/')}/{path}")
            page.wait_for_function(WAIT_JS, timeout=120_000)
            time.sleep(3.0)  # let Plotly finish drawing
            target = OUT / f"{name}.png"
            page.screenshot(path=str(target))
            print("saved", target)
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
