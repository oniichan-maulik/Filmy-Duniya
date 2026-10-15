import os

import pytest

RUN_UI_TESTS = os.getenv("RUN_UI_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_UI_TESTS, reason="Set RUN_UI_TESTS=1 to enable Playwright UI tests"
)

if RUN_UI_TESTS:
    import tests.playwright_ui_test as ui_test


def test_ui_runs_and_produces_results():
    results = ui_test.run_test(save=False)
    assert isinstance(results, list)
    assert len(results) >= 1
