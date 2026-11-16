import json
import os
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright  # type: ignore
except ImportError:  # pragma: no cover - handled via caller
    sync_playwright = None

URL = os.getenv("STREAMLIT_BASE_URL", "http://127.0.0.1:8501")
OUT = Path(__file__).resolve().parents[1] / "data" / "ui_test_results.json"
# Search, unlike the mood grid, doesn't depend on a webcam being present.
QUERY = os.getenv("UI_TEST_QUERY", "spider")
CARDS = '[class*="st-key-card-search"]'


def run_test(save: bool = True):
    """Search the running Streamlit UI and return the poster cards it renders.

    Each result is {"title", "text"}. Optionally saves them to
    data/ui_test_results.json.
    """
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is not installed. Install via `pip install playwright` and run `playwright install`."
        )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(URL, timeout=30000)

            search = page.get_by_label("Search films")
            search.wait_for(state="visible", timeout=60000)
            search.fill(QUERY)
            search.press("Enter")

            try:
                page.wait_for_selector(CARDS, timeout=30000)
            except Exception:
                print("No results rendered in UI within timeout")
                debug_path = OUT.parent / "ui_debug.html"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(page.content(), encoding="utf8")
                return []

            cards = page.locator(CARDS)
            results = []
            for i in range(cards.count()):
                card = cards.nth(i)
                results.append({
                    "title": card.locator("button").first.inner_text().strip(),
                    "text": card.inner_text(),
                })
        finally:
            browser.close()

    if save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    res = run_test(save=True)
    print("Found", len(res), "cards")
