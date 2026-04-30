import argparse
from datetime import datetime
from pathlib import Path
import time

SITE_LIST_PATH = Path(__file__).with_name("sites.txt")
SETTLE_TIME = 3.0
SCROLL_STEP = 500
SCROLL_INTERVAL = 1.0
CLICK_X = 600
CLICK_Y = 400


def load_sites() -> list[str]:
    return [
        line.strip()
        for line in SITE_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def wait_until(target_epoch: float):
    while True:
        remaining = target_epoch - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def normalize_url(site: str) -> str:
    if site.startswith("http://") or site.startswith("https://"):
        return site
    return f"https://{site}"


def deterministic_scroll_with_midclick(page, scroll_seconds: float):
    start = time.time()
    direction = 1
    clicked = False

    while time.time() - start < scroll_seconds:
        elapsed = time.time() - start

        if not clicked and elapsed >= scroll_seconds / 2:
            print("Clicking page center...")
            page.mouse.click(CLICK_X, CLICK_Y)
            clicked = True
            time.sleep(0.5)

        page.mouse.wheel(0, direction * SCROLL_STEP)
        time.sleep(SCROLL_INTERVAL)
        direction *= -1


def visit_site(browser, site: str, scroll_seconds: float):
    context = browser.new_context()
    page = context.new_page()
    url = normalize_url(site)

    try:
        print(f"Navigating to {url}")
        page.goto(url, wait_until="load", timeout=30000)
        time.sleep(SETTLE_TIME)
        deterministic_scroll_with_midclick(page, scroll_seconds)
    finally:
        context.close()


def iter_targets(sites: list[str], site: str | None, launches: int):
    if site is not None:
        yield site, 1, 1
        return

    for current_site in sites:
        for run_idx in range(1, launches + 1):
            yield current_site, run_idx, launches


def main():
    parser = argparse.ArgumentParser(
        description="Visit one scheduled website or the whole shared site list in visible fresh browser contexts."
    )
    parser.add_argument("--launches", type=int, default=1, help="Number of launches per website in batch mode")
    parser.add_argument("--scroll-seconds", type=float, default=12.0, help="How long to scroll on each launch")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between launches in batch mode")
    parser.add_argument("--site", help="Single site to visit instead of iterating over the shared site list")
    parser.add_argument("--run-label", help="Optional label printed in logs for single-site mode")
    parser.add_argument("--headless", action="store_true", help="Run Chromium headless instead of showing a window")
    parser.add_argument(
        "--start-epoch",
        type=float,
        help="Unix epoch for the first scheduled launch. When combined with --slot-seconds, runs use an absolute schedule.",
    )
    parser.add_argument(
        "--slot-seconds",
        type=float,
        help="Fixed spacing between launch start times in batch mode. Requires --start-epoch.",
    )
    parser.add_argument("--print-sites", action="store_true", help="Print the shared site list and exit")
    args = parser.parse_args()

    if args.print_sites:
        for site in load_sites():
            print(site)
        return

    if args.slot_seconds is not None and args.start_epoch is None:
        parser.error("--slot-seconds requires --start-epoch")

    if args.launches < 1:
        parser.error("--launches must be at least 1")

    if args.scroll_seconds < 0 or args.delay < 0:
        parser.error("--scroll-seconds and --delay must be non-negative")

    if args.slot_seconds is not None and args.slot_seconds <= 0:
        parser.error("--slot-seconds must be greater than 0")

    sites = load_sites()
    if not sites and args.site is None:
        parser.error(f"No sites were found in {SITE_LIST_PATH}")

    # Import Playwright only when we are actually going to launch a browser.
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

    scheduled_mode = args.start_epoch is not None
    targets = list(iter_targets(sites, args.site, args.launches))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)

        total = len(targets)

        for index, (site, run_idx, total_runs_for_site) in enumerate(targets, start=1):
            if scheduled_mode:
                target_epoch = args.start_epoch
                if args.slot_seconds is not None:
                    target_epoch += (index - 1) * args.slot_seconds

                wait_seconds = target_epoch - time.time()
                if wait_seconds > 0:
                    target_label = datetime.fromtimestamp(target_epoch).isoformat(timespec="seconds")
                    print(f"Waiting {wait_seconds:.2f}s for scheduled start at {target_label}")
                    wait_until(target_epoch)

            if args.site is not None:
                label = args.run_label or site
                print(f"[1/1] {label}")
            else:
                print(f"[{index}/{total}] {site} run {run_idx}/{total_runs_for_site}")

            try:
                visit_site(browser, site, args.scroll_seconds)
            except PlaywrightTimeoutError:
                print(f"Timeout loading {site}")
            except Exception as exc:
                print(f"Error on {site}: {exc}")

            if index < total and not (scheduled_mode and args.slot_seconds is not None):
                time.sleep(args.delay)

        browser.close()


if __name__ == "__main__":
    main()
