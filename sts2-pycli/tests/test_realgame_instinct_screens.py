"""Instinct on every out-of-combat screen (``realgame_instinct_screens``).

For each screen type, a native campaign run is played to its first such
decision; the mod's payload for that screen is synthesized from the engine at
that position (its envelope as in ``test_realgame_instinct._envelope``, its
options from the engine's own getters), and Instinct answering the payload
must make the SAME decision with the SAME probabilities as the native student
-- the graft and the option injection lose nothing the student reads -- and
answer with an executable ``best``.
"""

from __future__ import annotations

from typing import Any

import pytest

sts2_rs = pytest.importorskip("sts2_rs", reason="Instinct needs the sts2_rs extension")

import realgame_instinct as ri  # noqa: E402
from test_realgame_instinct import WEIGHTS, _campaign_run, _envelope, _wire_id  # noqa: E402

pytestmark = pytest.mark.skipif(not WEIGHTS.is_file(), reason="no Instinct weights (data/e2e)")

#: Decision type -> the ``advice_for`` the mod asks it as.
SCREENS = {
    "NEOW": "event",
    "CARD_REWARD": "card_reward",
    "SHOP": "shop",
    "REST": "rest_site",
    "SMITH": "card_select",
    "EVENT": "event",
    "BOSS_RELIC": "boss_relic",
    "CARD_UPGRADE": "card_select",
    "CARD_GRANT": "card_select",
    "POTION_SWAP": "potion_swap",
}
SEEDS = ["INSTINCTSEED7", "INSTINCTSEED8", "INSTINCTSEED9", "INSTINCTSEED10"]


@pytest.fixture(scope="module")
def positions() -> dict[str, dict[str, Any]]:
    """The first position of each screen type over a few campaign runs."""
    student = sts2_rs.E2eStudent(str(WEIGHTS))
    found: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        rm = _campaign_run(seed)
        driver = sts2_rs.E2eDriver(rm.seed)
        room = None
        for _ in range(4000):
            peek = driver.peek(rm)
            if peek.kind == "terminal":
                break
            if peek.kind == "forced":
                f = peek.forced
                driver.apply(rm, f["action"], f["kind"], strict=False)
                continue
            dt = peek.decision["decision_type"]
            if dt == "COMBAT_TURN":
                room = (rm.current_room() or {}).get("room_type") or room
            if dt in SCREENS and dt not in found:
                found[dt] = {
                    "seed": seed,
                    "rm": rm.clone(),
                    "native": student.decide(driver.copy(), peek, rm),
                    "room": room,
                }
            out = student.decide(driver, peek, rm)
            ca = peek.class_action(out["class"])
            driver.apply(rm, ca["action"], ca["kind"])
        if len(found) == len(SCREENS):
            break
    return found


def _wire_card(i: int, card_id: str, upgraded: Any, ench: Any = None) -> dict[str, Any]:
    c: dict[str, Any] = {
        "index": i,
        "id": _wire_id(card_id),
        "name": card_id,
        "upgrade_level": int(bool(upgraded)),
    }
    if ench:
        pair = ench if isinstance(ench, (list, tuple)) else (ench, 1)
        c["enchantment"] = _wire_id(str(pair[0]))
        c["enchantment_amount"] = int(pair[1]) if len(pair) > 1 else 1
    return c


def _deck_cards(rm: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, up in rm.master_deck_detailed():
        if not up:
            out.append(_wire_card(len(out), name, up))
    return out


def _shop_options(rm: Any, msg: dict[str, Any]) -> list[dict[str, Any]]:
    inv = rm.shop_inventory()
    opts: list[dict[str, Any]] = [
        {"index": 0, "id": "leave_shop", "action": "leave_shop", "enabled": True}
    ]
    for kind, key in (("buy_card", "cards"), ("buy_relic", "relics"), ("buy_potion", "potions")):
        for row in inv[key]:
            if not row.get("available"):
                continue
            item: dict[str, Any] = {"id": _wire_id(row["id"])}
            if kind == "buy_card":
                item["upgrade_level"] = int(bool(row.get("upgraded")))
            opts.append(
                {
                    "index": len(opts),
                    "id": kind,
                    "action": kind,
                    "label": row["id"],
                    "enabled": True,
                    "price": row["price"],
                    "item": item,
                }
            )
    if not inv.get("removal_used"):
        opts.append(
            {
                "index": len(opts),
                "id": "remove_card",
                "action": "remove_card",
                "enabled": True,
                "price": inv["removal_cost"],
            }
        )
        msg["purge_cost"] = inv["removal_cost"]
    return opts


def _event_options(rm: Any) -> list[dict[str, Any]]:
    name = rm.current_event_name()
    out = []
    for i, info in enumerate(rm.event_option_infos()):
        key = info.get("loc_key") or f"{name}.options.{info.get('option_id')}"
        out.append(
            {
                "index": i,
                "event_id": name,
                "text_key": key,
                "enabled": info.get("enabled", True),
                "relic_id": "",
                "label": info.get("option_id"),
            }
        )
    return out


def _payload(dt: str, pos: dict[str, Any]) -> dict[str, Any]:
    """The mod's payload for screen ``dt``, built from the engine at ``pos``."""
    rm = pos["rm"]
    msg = _envelope(rm, pos["seed"], SCREENS[dt])
    if dt == "CARD_REWARD":
        ench = list(rm.card_reward_option_enchantments())
        ids = list(rm.card_reward_option_ids())
        ups = list(rm.card_reward_option_upgraded())
        msg["cards"] = [
            _wire_card(i, cid, ups[i], ench[i] if i < len(ench) else None)
            for i, cid in enumerate(ids)
        ]
        msg["can_skip"] = True
    elif dt == "SHOP":
        msg["options"] = _shop_options(rm, msg)
    elif dt == "REST":
        msg["options"] = [
            {"index": i, "id": oid, "action": "rest_option", "label": oid, "enabled": True}
            for i, (oid, enabled) in enumerate(rm.rest_site_options())
            if enabled
        ]
    elif dt in ("SMITH", "CARD_UPGRADE"):
        msg.update(cards=_deck_cards(rm), prompt_key="TO_UPGRADE", min_select=1, max_select=1)
    elif dt == "CARD_GRANT":
        cards = [_wire_card(i, cid, up) for i, (cid, up) in enumerate(rm.card_grant_options())]
        msg.update(
            cards=cards,
            prompt_key="TO_OBTAIN",
            min_select=rm.pending_card_grant_min,
            max_select=1,
        )
    elif dt == "EVENT":
        msg["options"] = _event_options(rm)
    elif dt == "NEOW":
        msg["options"] = [
            {
                "index": i,
                "event_id": "NEOW",
                "text_key": f"NEOW.pages.INITIAL.options.{relic}",
                "enabled": True,
                "relic_id": relic,
                "label": relic,
            }
            for i, relic in enumerate(rm.neow_offers())
        ]
    elif dt == "BOSS_RELIC":
        msg["options"] = [
            {
                "index": i,
                "action": "pick_relic",
                "enabled": True,
                "relic": {"id": _wire_id(r), "name": r},
            }
            for i, r in enumerate(rm.boss_relic_options())
        ]
    elif dt == "POTION_SWAP":
        potion = _wire_id(rm.pending_potion_swap)
        held = [_wire_id(p) for p in rm.potions()]
        group: list[dict[str, Any]] = [{"discard_slot": None, "label": "decline"}]
        group += [{"discard_slot": k, "label": f"discard {p}"} for k, p in enumerate(held)]
        msg["options"] = [
            {
                "index": i,
                "id": "PotionReward",
                "action": "potion_swap",
                "reward": {"id": potion},
                "reward_button": 0,
                **g,
            }
            for i, g in enumerate(group)
        ]
    return msg


@pytest.mark.parametrize("dt", list(SCREENS))
def test_screen_advice_sees_what_the_native_student_sees(
    dt: str, positions: dict[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    pos = positions.get(dt)
    if pos is None:
        pytest.skip(f"no {dt} position in {len(SEEDS)} campaign runs")
    advisor = ri.InstinctAdvisor(WEIGHTS)
    advisor.last_room = pos["room"]
    advisor.last_screen = "rest_site" if dt == "SMITH" else "event"
    picks: list[Any] = []
    real = advisor.evaluate

    def spy(*a: Any, **k: Any) -> Any:
        picks.append(real(*a, **k))
        return picks[-1]

    monkeypatch.setattr(advisor, "evaluate", spy)
    advice = advisor.advise(_payload(dt, pos), SCREENS[dt], session={})

    assert advice["brain"] == "instinct"
    assert advice.get("best"), advice
    native = pos["native"]
    pick = picks[-1]
    assert pick.decision["decision_type"] == dt
    assert pick.best == native["class"], (pick.probs, native["probs"])
    # Probabilities only to 2e-2: the advisor's driver starts fresh on the rebuilt run, so the
    # history the native driver carries differs (d1250 drifted 1-4e-3 here, d2500 avg5 up to
    # 1.8e-2 on STEADY / 9e-3 on CARD_GRANT). The pick itself must match exactly (above).
    assert pick.probs == pytest.approx(native["probs"], abs=2e-2)
