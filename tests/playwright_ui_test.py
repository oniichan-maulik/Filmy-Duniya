import json
import os
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright  # type: ignore
except ImportError:  # pragma: no cover - handled via caller
    sync_playwright = None

URL = os.getenv("STREAMLIT_BASE_URL", "http://127.0.0.1:8501")
OUT = Path(__file__).resolve().parents[1] / "data" / "ui_test_results.json"


def run_test(save: bool = True):
    """Run a headless Playwright session against the Streamlit UI and return
    the list of recommendation dicts rendered on the page. Also optionally
    save results to data/ui_test_results.json.
    """
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is not installed. Install via `pip install playwright` and run `playwright install`."
        )
    if not os.getenv("STREAMLIT_BASE_URL"):
        # best-effort: allow overriding host when running within containers
        pass
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, timeout=30000)

        # Wait for Streamlit UI to render the input
        locator_source = "aria-label"
        inp = page.locator('input[aria-label=\"Enter a movie title you like (optional):\"]').first
        try:
            inp.wait_for(state="visible", timeout=10000)
        except Exception:
            locator_source = "fallback"
            inp = page.locator('input[type=\"text\"]').first
            inp.wait_for(state="visible", timeout=10000)
        print(f"[ui-test] using {locator_source} locator", flush=True)
        inp.fill("Inception")

        # Click the Recommend button
        try:
            btn = page.get_by_role("button", name="Recommend")
            btn.click()
        except Exception:
            # fallback: click the first button
            page.locator('button').first.click()

        # Wait for movie-title elements to appear
        try:
            page.wait_for_selector('.movie-title', timeout=10000)
        except Exception:
            print("No results rendered in UI within timeout")
            debug_path = OUT.parent / "ui_debug.html"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(page.content(), encoding="utf8")
            browser.close()
            return []

        cards = page.locator('.movie-card')
        results = []
        for i in range(cards.count()):
            card = cards.nth(i)
            title = card.locator('.movie-title').inner_text().strip() if card.locator('.movie-title').count() else ''
            # gather text contents for score/genres/director/description
            text = card.inner_text()
            results.append({
                'title': title,
                'text': text,
            })

        browser.close()

        if save:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with OUT.open('w', encoding='utf8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return results


if __name__ == '__main__':
    res = run_test(save=True)
    print('Found', len(res), 'cards')
