"""The shop loop breaker counts only purchases that changed nothing."""

from __future__ import annotations

from headless_eval import _purchase_repeats


def test_a_purchase_that_went_through_is_not_a_repeat() -> None:
    key = ("shop", "buy_relic", '{"relic_index": 1}')
    count = 1
    # The Courier refills slot 1 after each purchase: the screen changes.
    for before, after in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        count = _purchase_repeats(count, key, key, after, before)
    assert count == 1


def test_a_rejected_purchase_repeats() -> None:
    key = ("shop", "buy_potion", '{"potion_index": 0}')
    count = 1
    for _ in range(3):
        count = _purchase_repeats(count, key, key, 7, 7)
    assert count == 4
