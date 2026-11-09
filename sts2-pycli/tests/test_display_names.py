"""No raw game ids on the HUD (``display_names``, PROTOCOL.md §5 v5.3).

The shop advice read ``SHRUG_IT_OFF``: a shop option's ``label`` is the
merchant entry's id. These pin the fix at its source (the shop / Neow / potion
swap labels come from the payload's display names) and the wire guard behind
it (``sanitize_advice`` / ``sanitize_coach``), plus the one exception the
takeover's cross-check forces: a ``best.label`` that is the mod's own spelling
of an option stays as the mod spelled it.

Tests that read the vendored localization tables (real names such as
``Strike`` for ``STRIKE_IRONCLAD``) are in ``conftest._NEEDS``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import display_names as dn
import pytest
import realgame_advisor as ra
import realgame_ppo as rp


class _StubPolicy:
    def __init__(self, best: int) -> None:
        self.best = best

    def distribution(self, obs: Any, actions: Any, metrics: Any) -> tuple[list[float], float]:
        probs = [0.01] * len(actions)
        probs[self.best] = 1.0 - 0.01 * (len(actions) - 1)
        return probs, 0.1


class _StubPolicies:
    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def get(self, character: str | None) -> tuple[Any, str | None]:
        return self.policy, None


# --------------------------------------------------------------------------
# humanize
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("SHRUG_IT_OFF", "Shrug It Off"),
        ("Strike → SOME_NEW_MONSTER", "Strike → Some New Monster"),
        ("SOME_NEW_CARD+", "Some New Card+"),
        ("remove_card", "remove card"),
        (
            "card_select: the net does not cover this surface",
            "card select: the net does not cover this surface",
        ),
    ],
)
def test_ids_become_words(raw: str, shown: str) -> None:
    assert dn.humanize(raw) == shown


@pytest.mark.parametrize(
    "prose",
    [
        "MCTS 4000 sims · beliefdraw",
        "defaulting to the first option, NOT a recommendation",
        "run-policy net · V=+0.25",
        "Shrug It Off",
        "Strike → Jaw Worm #2",
        r"no Instinct net at C:\data\e2e\ironclad_e2e_d2500_avg5_ema0999.e2ew",
        "data/e2e/ironclad_e2e_d2500_avg5_ema0999.e2ew",
    ],
)
def test_display_text_is_left_alone(prose: str) -> None:
    assert dn.humanize(prose) == prose
    assert dn.humanize(dn.humanize(prose)) == prose


def test_names_come_from_the_game_tables() -> None:
    assert dn.humanize("SHRUG_IT_OFF") == "Shrug It Off"
    assert dn.humanize("remove:STRIKE_IRONCLAD") == "Remove Strike"
    assert dn.humanize("discard GAMBLERS_BREW") == "discard Gambler's Brew"
    assert dn.humanize("BASH+") == "Bash+"
    assert dn.display_name("StrikeIronclad") == "Strike"
    assert dn.display_name("RELIC.BURNING_BLOOD") == "Burning Blood"
    assert dn.display_name("WINGED_BOOTS.title") == "Winged Boots"


# --------------------------------------------------------------------------
# The shop at its source (the reported leak)
# --------------------------------------------------------------------------


def _item(cid: str, name: str) -> dict[str, Any]:
    return {"id": cid, "name": name, "upgrade_level": 0, "type": "Skill", "rarity": "Common"}


def _shop_state() -> dict[str, Any]:
    """The live shop payload's shape: option ``label`` = the entry id."""
    return {
        "type": "shop",
        "purge_cost": 75,
        "options": [
            {"index": 0, "id": "leave_shop", "action": "leave_shop", "label": "Leave shop"},
            {
                "index": 1,
                "id": "buy_card",
                "action": "buy_card",
                "label": "SHRUG_IT_OFF",
                "price": 49,
                "item": _item("SHRUG_IT_OFF", "Shrug It Off"),
            },
            {
                "index": 2,
                "id": "buy_relic",
                "action": "buy_relic",
                "label": "BAG_OF_MARBLES",
                "price": 150,
                "item": {"id": "BAG_OF_MARBLES", "name": "Bag of Marbles"},
            },
            {
                "index": 3,
                "id": "remove_card",
                "action": "remove_card",
                "label": "remove_card",
                "price": 75,
            },
        ],
        "run_state": {
            "character": "Ironclad",
            "gold": 300,
            "master_deck": [
                {"id": "STRIKE_IRONCLAD", "name": "Strike"},
                {"id": "BASH", "name": "Bash"},
            ],
            "potions": [{"slot": 0, "empty": True}],
        },
    }


def _shop_advice(best_label_prefix: str) -> dict[str, Any]:
    state = _shop_state()
    decision = rp._build_shop(state)
    assert decision is not None
    best = next(i for i, lbl in enumerate(decision.labels) if lbl.startswith(best_label_prefix))
    return ra.compute_screen_advice(state, "shop", _StubPolicies(_StubPolicy(best)))


def test_shop_options_are_named_by_their_item() -> None:
    advice = _shop_advice("SHRUG")
    labels = {o["label"] for o in advice["options"]}
    assert {
        "Leave shop",
        "Shrug It Off",
        "Bag of Marbles",
        "Remove Strike",
        "Remove Bash",
    } <= labels
    assert advice["best"]["label"] == "Shrug It Off"
    assert not any("_" in o["label"] for o in advice["options"])


def test_a_shop_removal_names_the_deck_card() -> None:
    advice = _shop_advice("remove:BASH")
    assert advice["best"]["label"] == "Remove Bash"


# --------------------------------------------------------------------------
# The wire guard and the cross-check exception
# --------------------------------------------------------------------------


def test_the_guard_rewrites_every_display_string() -> None:
    advice = {
        "note": "card_select: SOME_NEW_CARD",
        "hand_back": "no head for SOME_SURFACE",
        "pv": ["SOME_NEW_CARD → JAW_WORM", "end turn"],
        "options": [{"label": "SOME_NEW_CARD"}, {"label": "Bash"}],
        "plan": [{"action": "play", "card_index": 0, "label": "SOME_NEW_CARD"}],
        "best": {"action": "choose", "index": 1, "label": "SOME_NEW_CARD"},
    }
    out = dn.sanitize_advice(advice, "shop", {"options": [{"label": "SOME_NEW_CARD"}]})
    assert out["note"] == "card select: Some New Card"
    assert out["hand_back"] == "no head for Some Surface"
    assert out["pv"][0] == "Some New Card → Jaw Worm"
    assert [o["label"] for o in out["options"]] == ["Some New Card", "Bash"]
    assert out["plan"][0]["label"] == "Some New Card"
    # The shop sends no labels to cross-check: best is rewritten too.
    assert out["best"]["label"] == "Some New Card"


def test_a_cross_checked_best_label_keeps_the_mods_spelling() -> None:
    """``SimAdvisor.ReconcileIndex`` refuses the click when ``best.label`` is
    not one of the screen's labels; the potion swap's are ``discard <ID>``."""
    state = {
        "options": [
            {"index": 0, "label": "decline", "discard_slot": None},
            {"index": 1, "label": "discard GAMBLERS_BREW", "discard_slot": 0},
        ]
    }
    advice = {
        "options": [{"label": "discard GAMBLERS_BREW", "is_best": True}],
        "best": {"action": "choose", "index": 1, "label": "discard GAMBLERS_BREW"},
    }
    out = dn.sanitize_advice(advice, "potion_swap", state)
    assert out["best"]["label"] == "discard GAMBLERS_BREW"
    assert "GAMBLERS_BREW" not in out["options"][0]["label"]


def test_a_cross_checked_label_the_mod_did_not_send_is_still_rewritten() -> None:
    """A card screen's mod labels are titles; an id there can only be the
    advisor's fallback, and its title is what the mod compares against."""
    state = {"cards": [{"index": 0, "id": "SOME_NEW_CARD", "name": "Some New Card"}]}
    advice = {"best": {"action": "choose", "index": 0, "label": "SOME_NEW_CARD"}}
    assert dn.sanitize_advice(advice, "card_reward", state)["best"]["label"] == "Some New Card"


def _potion_swap_state() -> dict[str, Any]:
    reward = {"id": "SKILL_POTION", "name": "Skill Potion"}
    return {
        "type": "potion_swap",
        "options": [
            {
                "index": 0,
                "action": "potion_swap",
                "label": "decline",
                "reward": reward,
                "reward_button": 0,
                "discard_slot": None,
            },
            {
                "index": 1,
                "action": "potion_swap",
                "label": "discard GAMBLERS_BREW",
                "reward": reward,
                "reward_button": 0,
                "discard_slot": 0,
            },
        ],
        "run_state": {
            "character": "Ironclad",
            "potions": [{"slot": 0, "id": "GAMBLERS_BREW", "name": "Gambler's Brew"}],
        },
    }


def test_the_potion_swap_reads_in_words_but_keeps_its_cross_check() -> None:
    advice = ra._decline_potion_swap(_potion_swap_state())
    advice = dn.sanitize_advice(advice, "potion_swap", _potion_swap_state())
    labels = [o["label"] for o in advice["options"]]
    assert labels == [
        "Keep your potions (leave Skill Potion)",
        "Discard Gambler's Brew for Skill Potion",
    ]
    assert advice["best"]["label"] == "decline"


def test_the_policys_potion_swap_keeps_its_cross_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "_live_potion_swap_enabled", lambda: True)
    state = _potion_swap_state()
    decision = ra._advisor_builders()["potion_swap"](state)
    assert decision is not None
    best = next(i for i, r in enumerate(decision.responses) if r.get("index") == 1)
    advice = ra.compute_screen_advice(state, "potion_swap", _StubPolicies(_StubPolicy(best)))
    advice = dn.sanitize_advice(advice, "potion_swap", state)
    assert advice["best"]["label"] == "discard GAMBLERS_BREW"
    best_row = next(o for o in advice["options"] if o["is_best"])
    assert best_row["label"] == "Discard Gambler's Brew for Skill Potion"


def test_the_guard_runs_at_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_handle_advice_request`` sanitizes whatever a surface returns, so a
    surface added later cannot leak an id either."""
    leaky = {
        "note": "SOME_SURFACE",
        "options": [{"label": "SOME_NEW_RELIC", "is_best": True, "index": 0}],
        "best": {"action": "choose", "index": 0, "label": "SOME_NEW_RELIC"},
    }
    monkeypatch.setattr(ra, "compute_screen_advice", lambda *a, **k: dict(leaky))
    sent: list[dict[str, Any]] = []
    client = SimpleNamespace(send_action=sent.append)
    log = SimpleNamespace(write=lambda *a, **k: None)
    wlog = SimpleNamespace(log_decision=lambda *a, **k: None, log_advice=lambda *a, **k: None)
    args = SimpleNamespace(sims=10, boss_sims=10, log_search_tree=0)
    msg = {"type": "advice_request", "advice_for": "shop", "request_id": "a1", "options": []}
    ra._handle_advice_request(client, ra.RecommendConfig(), args, log, wlog, msg, None)  # type: ignore[arg-type]
    advice = sent[0]["advice"]
    assert advice["note"] == "Some Surface"
    assert advice["options"][0]["label"] == "Some New Relic"
    assert advice["best"]["label"] == "Some New Relic"


def test_coach_records_are_guarded_recursively() -> None:
    record = {
        "kind": "autopsy",
        "id": "autopsy:RUN_1",
        "items": [
            {
                "kind": "turn",
                "record": {
                    "you": ["SOME_NEW_CARD → JAW_WORM", "end turn"],
                    "ai": ["Bash → Jaw Worm"],
                    "context": {"hand": ["SOME_NEW_CARD"], "enemies": [{"name": "JAW_WORM"}]},
                },
            },
            {"kind": "choice", "record": {"played": {"label": "SOME_NEW_RELIC", "prob": 0.2}}},
        ],
    }
    out = dn.sanitize_coach(record)
    turn = out["items"][0]["record"]
    assert turn["you"][0] == "Some New Card → Jaw Worm"
    assert turn["context"]["hand"] == ["Some New Card"]
    assert turn["context"]["enemies"][0]["name"] == "Jaw Worm"
    assert out["items"][1]["record"]["played"]["label"] == "Some New Relic"
    # Machine fields are not display text.
    assert out["id"] == "autopsy:RUN_1"
