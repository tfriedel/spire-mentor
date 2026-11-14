"""Rest options are tokenized by identity, not position.

After a Miniature Tent first pick the engine drops the taken option, so
Smith sits at idx 0 and the old positional rule called it ``rest_heal``
(found by the pep299 fork). Everything a shipped net was trained on must
featurize exactly as before.
"""

from __future__ import annotations

import pytest
import sts2_rs
from ppo import vocab
from ppo.encoder import _rest_option_token
from ppo.env import observe
from realgame_ppo import _rest_option_labels

HEAL = ("rest_heal", 0)
SMITH = ("rest_smith", 1)


def _tokens(labels, idxs):
    return [_rest_option_token(labels, i) for i in idxs]


def test_identity_survives_the_tent_reindexing():
    full = ["HEAL", "SMITH", "DIG", "HATCH"]
    assert _tokens(full, range(4)) == [HEAL, SMITH, ("rest_other", 2), ("rest_other", 3)]
    assert _tokens(full[1:], range(3)) == [SMITH, ("rest_other", 2), ("rest_other", 3)]
    assert _tokens(["HEAL", "DIG"], range(2)) == [HEAL, ("rest_other", 2)]
    # The bridge: a disabled Smith's slot is occupied but unnamed.
    assert _tokens(["HEAL", "", "DIG"], [0, 2]) == [HEAL, ("rest_other", 2)]


@pytest.mark.parametrize("labels", [[], ["rest", "smith"], ["HEAL", "SMITH", "DIG", "HATCH"]])
def test_untaken_and_legacy_labels_keep_the_positional_encoding(labels):
    cap = vocab.MAX_OPT_IDX - 1
    positional = [HEAL, SMITH, *[("rest_other", min(cap, i)) for i in range(2, 10)]]
    assert _tokens(labels, range(len(labels) or 10)) == positional[: len(labels) or 10]
    if labels == ["rest", "smith"]:
        # A recording's fixed pair says nothing about idx 2 and up.
        assert _tokens(labels, range(10)) == positional


def test_bridge_labels_are_ids_by_slot():
    options = [
        {"index": 0, "option_index": 0, "id": "HEAL"},
        {"index": 1, "option_index": 2, "id": "DIG"},
    ]
    assert _rest_option_labels(options) == ["HEAL", "", "DIG"]
    # An older mod: no ids, so every slot is unnamed and the rule is positional.
    assert _rest_option_labels([{"index": 0}, {"index": 1}]) == ["", ""]


def test_observe_reports_the_engines_current_options():
    rm = sts2_rs.RunManager(seed=1, character="Ironclad", ascension_level=0)
    rm.debug_set_loadout(relics=[*rm.relics(), "MINIATURE_TENT"])
    rm.debug_enter_rest_site([])
    assert observe(rm, "rest_site", "rest")["rest_option_labels"] == ["HEAL", "SMITH"]
    rm.take_action({"action": "rest_option", "idx": 0})
    assert rm.phase() == "rest_site"
    assert observe(rm, "rest_site", "rest")["rest_option_labels"] == ["SMITH"]
