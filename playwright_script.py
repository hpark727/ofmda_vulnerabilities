import argparse
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

SITES = [
    "google.com",
    "facebook.com",
    "microsoft.com",
    "youtube.com",
    "apple.com",
    "instagram.com",
    "twitter.com",
    "linkedin.com",
    "amazon.com",
    "wikipedia.org",
    "github.com",
    "chatgpt.com",
    "bing.com",
    "netflix.com",
    "pinterest.com",
    "whatsapp.com",
    "yahoo.com",
]

SETTLE_TIME = 3.0
SCROLL_STEP = 500
SCROLL_INTERVAL = 1.0
CLICK_X = 600
CLICK_Y = 400


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


def visit_site(browser, url: str, scroll_seconds: float):
    context = browser.new_context()
    page = context.new_page()

    try:
        print(f"Navigating to {url}")
        page.goto(url, wait_until="load", timeout=30000)
        time.sleep(SETTLE_TIME)
        deterministic_scroll_with_midclick(page, scroll_seconds)
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(
        description="Launch a fixed list of websites repeatedly in visible fresh browser contexts, scroll, and click once halfway through."
    )
    parser.add_argument("--launches", type=int, required=True, help="Number of launches per website")
    parser.add_argument("--scroll-seconds", type=float, default=12.0, help="How long to scroll on each launch")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between launches")
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        total = len(SITES) * args.launches
        count = 0

        for site in SITES:
            url = normalize_url(site)

            for run_idx in range(1, args.launches + 1):
                count += 1
                print(f"[{count}/{total}] {site} run {run_idx}/{args.launches}")

                try:
                    visit_site(browser, url, args.scroll_seconds)
                except PlaywrightTimeoutError:
                    print(f"Timeout loading {site}")
                except Exception as e:
                    print(f"Error on {site}: {e}")

                if count < total:
                    time.sleep(args.delay)

        browser.close()


if __name__ == "__main__":
    main()