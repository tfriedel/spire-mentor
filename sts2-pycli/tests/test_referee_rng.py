"""Referee mode of the headless host: keyed RNG, escape audit, locked commands.

The benchmark referee (Spire Bench) runs the real game with randomness a client
cannot predict. These tests pin the properties the design rests on.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
import referee_spike
import sts2_client

pytestmark = pytest.mark.oracle

KEY = "11" * 32
OTHER_KEY = "22" * 32


def _skip_unported_referee() -> None:
    # RefereeRng throws NotSupportedException from v0.111.0 on (its Rng
    # changed shape), so a referee host dies at start there. Port it, then
    # drop this.
    if sts2_client.GAME_VERSION != "v0.107.1":
        pytest.skip(
            f"referee mode is not ported to {sts2_client.GAME_VERSION} (RefereeRng.Install)"
        )


class Host:
    """A raw headless process, for commands the spike driver does not send."""

    def __init__(self, *, referee: bool, diag: bool = True):
        if referee:
            _skip_unported_referee()
        env = os.environ.copy()
        env.pop("STS2_REFEREE", None)
        env.pop("STS2_REFEREE_DIAG", None)
        if referee:
            env["STS2_REFEREE"] = "1"
            if diag:
                env["STS2_REFEREE_DIAG"] = "1"
        self.proc = subprocess.Popen(
            sts2_client.headless_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
            cwd=sts2_client.REPO_ROOT,
        )
        self.ready = self._read()

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            assert line, "EOF from game process"
            if line.startswith("{"):
                return json.loads(line)

    def send(self, **cmd) -> dict:
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        return self._read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.proc.kill()
        self.proc.wait()


def _play(key: str, **kw):
    _skip_unported_referee()
    return referee_spike.play(character="Ironclad", ascension=10, label="T0", key_hex=key, **kw)


def test_the_same_key_and_actions_reproduce_the_game_and_another_key_does_not():
    first = _play(KEY)
    assert "error" not in first and "fault" not in first, first
    assert first["draws"] > 0
    # Every field the host sent in this run was classified as public or hidden.
    assert first["unclassified_fields"] == []

    again = _play(KEY, actions=first["actions"])
    assert again["digest"] == first["digest"]

    # Same public seed label, different secret: the label alone decides nothing.
    other = _play(OTHER_KEY)
    assert "fault" not in other, other
    assert other["digest"] != first["digest"]


def test_the_audit_reports_a_draw_that_bypassed_the_patches(monkeypatch):
    # Negative control: with NextInt left unpatched its draws reach the game's
    # own generator, which is exactly what an inlined call would do.
    monkeypatch.setenv("STS2_REFEREE_SELFTEST_SKIP", "NextInt")
    run = _play(KEY)
    assert "rng escape" in run.get("fault", ""), run


@pytest.mark.parametrize(
    "cmd",
    [
        {"cmd": "get_state"},
        {"cmd": "get_map"},
        {"cmd": "rng_probe", "seed": "X"},
        {"cmd": "set_draw_order", "order": []},
        {"cmd": "dump_rooms"},
        {"cmd": "load_save", "path": "/nonexistent"},
        {"cmd": "legal_actions"},
    ],
    ids=lambda c: c["cmd"],
)
def test_referee_mode_refuses_commands_that_read_or_set_hidden_state(cmd):
    with Host(referee=True) as host:
        assert host.ready["referee"] is True
        reply = host.send(**cmd)
        assert reply == {"type": "error", "code": "not_available"}


def test_referee_mode_refuses_a_run_without_a_key():
    with Host(referee=True) as host:
        reply = host.send(cmd="start_run", character="Ironclad", seed="T0")
        assert reply == {"type": "error", "code": "bad_request"}


def test_a_default_host_is_not_in_referee_mode():
    with Host(referee=False) as host:
        assert host.ready["referee"] is False


def _chi_square(counts: list[int]) -> float:
    expected = sum(counts) / len(counts)
    return sum((c - expected) ** 2 / expected for c in counts)


def test_keyed_draws_are_uniform_through_the_games_own_rng_api():
    samples = 100_000
    with Host(referee=True) as host:
        got = host.send(cmd="referee_selftest", rng_key=KEY, samples=samples)
    assert got["type"] == "referee_selftest"
    # One counter step per draw: int + bool + float + the shuffle's own draws.
    assert got["counter"] >= samples * 7
    # Critical values at p = 0.001. The key is fixed, so this cannot flake.
    for name, critical in [
        ("next_int_10", 27.88),
        ("next_float_decile", 27.88),
        ("shuffle5_first", 18.47),
        ("next_bool", 10.83),
    ]:
        assert sum(got[name]) == samples, name
        assert _chi_square(got[name]) < critical, (name, got[name])


# ── The player-visible view ─────────────────────────────────────────────────


def _combat_payload() -> dict:
    """A real combat decision from a default host: hidden state included."""
    with Host(referee=False) as host:
        state = host.send(cmd="start_run", character="Ironclad", seed="VIEW1", ascension=0)
        rnd = referee_spike.random.Random("VIEW1")
        for _ in range(60):
            if state.get("decision") == "combat_play" and len(state.get("draw_pile", [])) >= 3:
                return state
            state = host.send(**referee_spike.choose(state, rnd))
    raise AssertionError("no combat decision with a draw pile reached")


def _project(payload: dict) -> dict:
    with Host(referee=True) as host:
        reply = host.send(cmd="referee_project", payload=payload)
        stats = host.send(cmd="referee_stats")
    view = json.loads(reply["view"])
    view["_unclassified"] = stats["unclassified_fields"]
    return view


def test_the_view_drops_hidden_state_from_a_real_combat_payload():
    raw = _combat_payload()
    # The default host really does carry these; otherwise the test proves nothing.
    assert "rng" in raw and "ai" in raw["enemies"][0] and "move_state" in raw["enemies"][0]
    assert [c["index"] for c in raw["draw_pile"]] == list(range(len(raw["draw_pile"])))

    view = _project(raw)
    assert view["_unclassified"] == []
    assert "rng" not in view
    for enemy in view["enemies"]:
        assert "ai" not in enemy and "move_state" not in enemy
        assert "hp" in enemy and "intents" in enemy
    assert all("index" not in card for card in view["draw_pile"])
    # Public facts survive: the pile's contents, the hand, the player.
    assert sorted(c["id"] for c in view["draw_pile"]) == sorted(c["id"] for c in raw["draw_pile"])
    assert [c["id"] for c in view["hand"]] == [c["id"] for c in raw["hand"]]
    assert view["player"]["hp"] == raw["player"]["hp"]


def test_the_view_does_not_depend_on_the_draw_order():
    raw = _combat_payload()
    assert len({c["id"] for c in raw["draw_pile"]}) > 1, "need a pile whose order is visible"
    shuffled = json.loads(json.dumps(raw))
    shuffled["draw_pile"].reverse()
    for i, card in enumerate(shuffled["draw_pile"]):
        card["index"] = i
    assert _project(shuffled) == _project(raw)


def test_a_field_nobody_classified_is_dropped_and_reported():
    raw = _combat_payload()
    raw["next_turn_draws"] = ["CARD.BASH"]
    raw["enemies"][0]["secret_plan"] = {"move": "X"}
    view = _project(raw)
    assert "next_turn_draws" not in view
    assert "secret_plan" not in view["enemies"][0]
    assert view["_unclassified"] == ["enemies[].secret_plan", "next_turn_draws"]


def test_a_map_decision_carries_the_whole_act_map():
    with Host(referee=True) as host:
        state = host.send(
            cmd="start_run", character="Ironclad", seed="VIEW2", ascension=0, rng_key=KEY
        )
        rnd = referee_spike.random.Random("VIEW2")
        for _ in range(40):
            if state.get("decision") == "map_select":
                break
            state = host.send(**referee_spike.choose(state, rnd))
    assert state["decision"] == "map_select"
    assert len(state["map"]["rows"]) > 5
    assert {"col", "row", "type", "children"} <= set(state["map"]["rows"][0][0])
    reachable = {(c["col"], c["row"]) for c in state["choices"]}
    on_map = {(n["col"], n["row"]) for row in state["map"]["rows"] for n in row}
    assert reachable <= on_map


# ── Legal actions and error codes ───────────────────────────────────────────


def _referee_run(host: Host, label: str, ascension: int = 0) -> dict:
    return host.send(
        cmd="start_run", character="Ironclad", seed=label, ascension=ascension, rng_key=KEY
    )


def _advance_to(host: Host, state: dict, decision: str, label: str) -> dict:
    rnd = referee_spike.random.Random(label)
    for _ in range(80):
        if state.get("decision") == decision:
            return state
        state = host.send(**referee_spike.choose_legal(state, rnd))
    raise AssertionError(f"never reached {decision}")


def test_every_observation_lists_its_legal_actions_and_they_are_accepted():
    run = _play(KEY)
    # The random-legal policy sent only listed actions; none was refused and no
    # decision came without one.
    assert "error" not in run, run
    assert run.get("errors", 0) == 0, run.get("refused")
    assert run["steps"] > 5


def test_an_action_that_is_not_listed_never_reaches_the_game():
    with Host(referee=True) as host:
        state = _advance_to(host, _referee_run(host, "LEGAL1"), "combat_play", "LEGAL1")
        listed = state["legal_actions"]
        assert {"action": "end_turn"} in listed
        for bad in [
            {"action": "play_card", "args": {"card_index": 99}},
            {"action": "select_map_node", "args": {"col": 0, "row": 0}},
            {"action": "proceed"},
            {"action": "end_turn", "args": {"extra": 1}},
        ]:
            assert bad not in listed
            assert host.send(cmd="action", **bad) == {"type": "error", "code": "illegal_action"}
        # The decision is untouched: a listed action still applies.
        after = host.send(cmd="action", action="end_turn")
        assert after["type"] == "decision"


def test_a_card_selection_is_listed_as_a_choice_and_checked():
    payload = {
        "type": "decision",
        "decision": "card_select",
        "cards": [{"index": 0, "id": "A"}, {"index": 1, "id": "B"}, {"index": 2, "id": "C"}],
        "min_select": 2,
        "max_select": 2,
        "cancelable": False,
    }
    with Host(referee=True) as host:
        view = json.loads(host.send(cmd="referee_project", payload=payload, record=True)["view"])
        assert view["legal_actions"] == [
            {"action": "select_cards", "choose": {"from": [0, 1, 2], "min": 2, "max": 2}}
        ]

        def refused(indices: str) -> bool:
            reply = host.send(cmd="action", action="select_cards", args={"indices": indices})
            return reply == {"type": "error", "code": "illegal_action"}

        assert refused("0") and refused("0,0") and refused("0,5") and refused("0,1,2")
        # A well-formed pick passes the check and reaches the host, which has
        # no run in this test and answers with its own refusal.
        reply = host.send(cmd="action", action="select_cards", args={"indices": "0,2"})
        assert reply == {"type": "error", "code": "rejected"}


@pytest.mark.parametrize(
    "cmd",
    [
        {"cmd": "referee_stats"},
        {"cmd": "referee_selftest", "rng_key": OTHER_KEY, "samples": 10},
        {"cmd": "referee_project", "payload": {"type": "decision", "decision": "game_over"}},
        {"cmd": "host_faults"},
    ],
    ids=lambda c: c["cmd"],
)
def test_a_scoring_worker_refuses_the_operator_diagnostics(cmd):
    with Host(referee=True, diag=False) as host:
        assert host.send(**cmd) == {"type": "error", "code": "not_available"}


def test_projecting_a_payload_does_not_change_what_is_legal():
    forged = {
        "type": "decision",
        "decision": "map_select",
        "choices": [{"col": 0, "row": 0, "type": "Monster"}],
    }
    with Host(referee=True) as host:
        state = _advance_to(host, _referee_run(host, "LEGAL2"), "combat_play", "LEGAL2")
        view = json.loads(host.send(cmd="referee_project", payload=forged)["view"])
        forged_action = {"action": "select_map_node", "args": {"col": 0, "row": 0}}
        assert forged_action in view["legal_actions"]
        assert forged_action not in state["legal_actions"]
        assert host.send(cmd="action", **forged_action) == {
            "type": "error",
            "code": "illegal_action",
        }
        assert host.send(cmd="action", action="end_turn")["type"] == "decision"


def test_errors_carry_a_code_and_no_text():
    with Host(referee=True) as host:
        host.proc.stdin.write("this is not json\n")
        host.proc.stdin.flush()
        assert host._read() == {"type": "error", "code": "bad_request"}


# ── The second boss ─────────────────────────────────────────────────────────


def _map_payload(choices: list[dict]) -> dict:
    node = {"col": 3, "row": 16, "type": "Boss", "children": [], "visited": False, "current": False}
    return {
        "type": "decision",
        "decision": "map_select",
        "choices": choices,
        "map": {
            "rows": [[{"col": 1, "row": 1, "type": "Monster", "children": []}], [node]],
            "boss": {"col": 3, "row": 15, "type": "Boss", "id": "FIRST", "name": "First"},
            "second_boss": {"col": 3, "row": 16, "type": "Boss", "id": "SECOND", "name": "Second"},
        },
    }


def test_the_second_boss_is_not_sent_before_it_is_the_next_node():
    with Host(referee=True) as host:
        early = host.send(
            cmd="referee_project", payload=_map_payload([{"col": 1, "row": 1, "type": "Monster"}])
        )
        late = host.send(
            cmd="referee_project", payload=_map_payload([{"col": 3, "row": 16, "type": "Boss"}])
        )
    assert "SECOND" not in early["view"] and "Second" not in early["view"]
    early_map = json.loads(early["view"])["map"]
    assert "second_boss" not in early_map
    assert all((n["col"], n["row"]) != (3, 16) for row in early_map["rows"] for n in row)
    assert early_map["boss"]["id"] == "FIRST"
    # Once the first boss has fallen it is the next node, as on the map screen.
    assert json.loads(late["view"])["map"]["second_boss"]["id"] == "SECOND"


# ── Where the headless host used to depart from the game ────────────────────


def test_trial_double_down_abandons_the_run(monkeypatch):
    # In the game the option opens the abandon-run confirm; Yes ends the run.
    # The default host can force the event, so the rule is tested there.
    monkeypatch.setenv("STS2_TRIAL_DOUBLE_DOWN_ABANDONS", "1")
    with Host(referee=False) as host:
        host.send(cmd="start_run", character="Ironclad", seed="TRIAL1", ascension=0)
        state = host.send(cmd="enter_room", type="event", event="TRIAL")
        for _ in range(4):
            options = {o["text_key"].rsplit(".", 1)[-1]: o["index"] for o in state["options"]}
            if "DOUBLE_DOWN" in options:
                break
            state = host.send(
                cmd="action", action="choose_option", args={"option_index": options["REJECT"]}
            )
        over = host.send(
            cmd="action", action="choose_option", args={"option_index": options["DOUBLE_DOWN"]}
        )
    assert over["decision"] == "game_over"
    assert over["victory"] is False and over["abandoned"] is True


def _enter_crystal_sphere(host: Host) -> dict:
    host.send(cmd="start_run", character="Ironclad", seed="SPHERE1", ascension=0)
    state = host.send(cmd="enter_room", type="event", event="CRYSTAL_SPHERE")
    return host.send(
        cmd="action", action="choose_option", args={"option_index": state["options"][0]["index"]}
    )


def test_the_crystal_sphere_is_played_by_the_client(monkeypatch):
    # In the game the player chooses where to dig. The default host digs a
    # fixed pattern itself; referee mode must hand over each divination.
    monkeypatch.setenv("STS2_CRYSTAL_SPHERE_INTERACTIVE", "1")
    with Host(referee=False) as host:
        state = _enter_crystal_sphere(host)
        assert state["decision"] == "crystal_sphere"
        left = state["divinations_left"]
        assert left >= 3 and len(state["cells"]) == 121
        gold_before = state["player"]["gold"]

        clear = next(c for c in state["cells"] if not c["hidden"])
        refused = host.send(
            cmd="action", action="crystal_sphere_click", args={"x": clear["x"], "y": clear["y"]}
        )
        assert refused["type"] == "error"

        seen = []
        for x, y in [(1, 3), (4, 3), (7, 3), (1, 6), (4, 6), (7, 6)][:left]:
            assert state["decision"] == "crystal_sphere"
            # A fogged cell tells the client nothing about what lies under it.
            assert all("item" not in c for c in state["cells"] if c["hidden"])
            state = host.send(
                cmd="action", action="crystal_sphere_click", args={"x": x, "y": y, "tool": "big"}
            )
            if state["decision"] == "crystal_sphere":
                assert state["divinations_left"] == left - len(seen) - 1
                seen.append(sum(not c["hidden"] for c in state["cells"]))
        # Each 3x3 dig cleared cells, and the minigame ended with the last one.
        assert seen == sorted(seen) and seen[0] > 24
        assert state["decision"] != "crystal_sphere"
        assert state["player"]["gold"] != gold_before or state["decision"] == "card_reward"


def test_crystal_sphere_clicks_are_listed_for_fogged_cells_only(monkeypatch):
    monkeypatch.setenv("STS2_CRYSTAL_SPHERE_INTERACTIVE", "1")
    with Host(referee=False) as host:
        raw = _enter_crystal_sphere(host)
    view = _project(raw)
    assert view["_unclassified"] == []
    fogged = {(c["x"], c["y"]) for c in view["cells"] if c["hidden"]}
    listed = {(a["args"]["x"], a["args"]["y"], a["args"]["tool"]) for a in view["legal_actions"]}
    assert listed == {(x, y, tool) for x, y in fogged for tool in ("big", "small")}


# ── The reward screen and the potion belt (GH #1282) ────────────────────────

# Potions whose popup offers Use outside combat (PotionUsage.AnyTime).
ANY_TIME_POTIONS = {"FRUIT_JUICE", "BLOOD_POTION", "ENTROPIC_BREW", "FOUL_POTION"}
# A run whose path offers both a potion reward and a chest under KEY.
REWARD_LABEL = "REWARD6"


def _next_screen_action(state: dict, rnd) -> dict:
    """Plays on through everything but the reward screen, never touching a potion."""
    if state["decision"] == "combat_play":
        return referee_spike.choose(state, rnd)
    if state["decision"] == "map_select":
        # A chest is a sure relic reward.
        chest = [c for c in state["choices"] if c["type"] == "Treasure"]
        if chest:
            return {
                "cmd": "action",
                "action": "select_map_node",
                "args": {"col": chest[0]["col"], "row": chest[0]["row"]},
            }
    legal = [a for a in state["legal_actions"] if "potion" not in a["action"]]
    pick = rnd.choice(legal)
    cmd = {"cmd": "action", "action": pick["action"]}
    if "choose" in pick:
        cmd["args"] = {"indices": str(pick["choose"]["from"][0])}
    elif "args" in pick:
        cmd["args"] = pick["args"]
    return cmd


def _held(state: dict) -> tuple[list[str], list[str]]:
    player = state["player"]
    return [r["id"] for r in player["relics"]], [p["entry"] for p in player.get("potions", [])]


def test_relic_and_potion_rewards_can_be_skipped_and_the_run_goes_on():
    skipped: dict[str, str] = {}
    with Host(referee=True) as host:
        state = _referee_run(host, REWARD_LABEL)
        rnd = referee_spike.random.Random(REWARD_LABEL)
        for _ in range(3000):
            if state["decision"] == "game_over" or len(skipped) == 2:
                break
            if state["decision"] != "rewards":
                state = host.send(**_next_screen_action(state, rnd))
                continue
            listed = {a["action"] for a in state["legal_actions"]}
            assert {"take_reward", "skip_reward", "proceed"} <= listed
            target = next(
                (
                    r
                    for r in state["rewards"]
                    if r["type"] in ("relic", "potion") and r["type"] not in skipped
                ),
                None,
            )
            if target is None:
                pick = state["rewards"][0]
                state = host.send(
                    cmd="action", action="take_reward", args={"reward_index": pick["index"]}
                )
                continue
            relics, potions = _held(state)
            after = host.send(
                cmd="action", action="skip_reward", args={"reward_index": target["index"]}
            )
            assert after["type"] == "decision", after
            # Gone from the screen and not given to the player.
            assert all(r["index"] != target["index"] for r in after.get("rewards", []))
            assert _held(after) == (relics, potions)
            skipped[target["type"]] = target["id"]
            # Leave whatever else is on the screen: nothing more is taken.
            gold = after["player"]["gold"]
            if after["decision"] == "rewards":
                after = host.send(cmd="action", action="proceed")
            assert after["type"] == "decision" and after["decision"] != "rewards"
            assert _held(after) == (relics, potions) and after["player"]["gold"] == gold
            state = after
        stats = host.send(cmd="referee_stats")
    assert set(skipped) == {"relic", "potion"}, skipped
    assert stats["unclassified_fields"] == []


def test_the_potion_belt_is_usable_outside_combat_where_the_game_allows_it():
    # A referee run collects potions off the reward screens; on the map every
    # held potion can be discarded, and only an AnyTime potion drunk.
    with Host(referee=True) as host:
        state = _referee_run(host, "BELT1")
        rnd = referee_spike.random.Random("BELT1")
        for _ in range(3000):
            if state["decision"] == "map_select" and state["player"].get("potions"):
                break
            if state["decision"] == "rewards":
                pick = state["rewards"][0]
                state = host.send(
                    cmd="action", action="take_reward", args={"reward_index": pick["index"]}
                )
            else:
                state = host.send(**_next_screen_action(state, rnd))
        assert state["decision"] == "map_select", state["decision"]
        held = state["player"]["potions"]
        listed = state["legal_actions"]
        for potion in held:
            slot = {"potion_index": potion["index"]}
            assert {"action": "discard_potion", "args": slot} in listed
            drinkable = {"action": "use_potion", "args": slot} in listed
            assert drinkable == (potion["entry"] in ANY_TIME_POTIONS), potion
        first = held[0]
        after = host.send(
            cmd="action", action="discard_potion", args={"potion_index": first["index"]}
        )
    assert after["decision"] == "map_select"
    assert len(after["player"]["potions"]) == len(held) - 1


def test_a_potion_is_drunk_and_discarded_on_the_map_screen():
    # The default host can hand out an AnyTime potion; the game rule itself is
    # checked through the referee view of the same screen.
    with Host(referee=False) as host:
        state = host.send(cmd="start_run", character="Ironclad", seed="BELT2", ascension=0)
        rnd = referee_spike.random.Random("BELT2")
        for _ in range(40):
            if state.get("decision") == "map_select":
                break
            state = host.send(**referee_spike.choose(state, rnd))
        assert state["decision"] == "map_select"
        host.send(cmd="set_player", potions=["FRUIT_JUICE", "STRENGTH_POTION"])
        max_hp = state["player"]["max_hp"]

        drunk = host.send(cmd="action", action="use_potion", args={"potion_index": 0})
        assert drunk["decision"] == "map_select"
        assert drunk["player"]["max_hp"] == max_hp + 5  # Fruit Juice
        assert [p["entry"] for p in drunk["player"]["potions"]] == ["STRENGTH_POTION"]

        slot = drunk["player"]["potions"][0]["index"]
        dropped = host.send(cmd="action", action="discard_potion", args={"potion_index": slot})
        assert dropped["decision"] == "map_select"
        assert dropped["player"]["potions"] == []
        assert dropped["player"]["max_hp"] == max_hp + 5


def test_the_reward_screen_lists_each_reward_with_take_and_skip():
    payload = {
        "type": "decision",
        "decision": "rewards",
        "rewards": [
            {"index": 0, "type": "gold", "amount": 15},
            {"index": 2, "type": "relic", "id": "ANCHOR", "name": "Anchor", "description": "x"},
        ],
        "__potion_belt": {
            "type": "ok",
            "slots": [
                {"potion_index": 0, "usable": True, "discardable": True},
                {"potion_index": 1, "usable": False, "discardable": True},
            ],
        },
    }
    view = _project(payload)
    assert view["_unclassified"] == []
    assert "__potion_belt" not in view
    assert view["legal_actions"] == [
        {"action": "take_reward", "args": {"reward_index": 0}},
        {"action": "take_reward", "args": {"reward_index": 2}},
        {"action": "skip_reward", "args": {"reward_index": 0}},
        {"action": "skip_reward", "args": {"reward_index": 2}},
        {"action": "proceed"},
        {"action": "use_potion", "args": {"potion_index": 0}},
        {"action": "discard_potion", "args": {"potion_index": 0}},
        {"action": "discard_potion", "args": {"potion_index": 1}},
    ]


def test_a_default_host_still_takes_the_rewards_itself():
    with Host(referee=False) as host:
        state = host.send(cmd="start_run", character="Ironclad", seed="REWARD2", ascension=0)
        rnd = referee_spike.random.Random("REWARD2")
        seen = set()
        for _ in range(400):
            seen.add(state.get("decision"))
            if state.get("decision") == "game_over" or state.get("floor", 0) >= 4:
                break
            state = host.send(**referee_spike.choose(state, rnd))
    assert "card_reward" in seen and "rewards" not in seen


# ── Fail closed on a swallowed error (GH #1283) ─────────────────────────────

# The empty catch around ProceedFromTerminalRewardsScreen in ForceToMap, which
# every combat on the way to the map passes. STS2_HOST_FAULT_INJECT makes its
# try block throw, as a missing Godot node would.
FORCED_SITE = "ForceToMap"


def test_a_swallowed_error_ends_a_referee_episode(monkeypatch):
    monkeypatch.setenv("STS2_HOST_FAULT_INJECT", FORCED_SITE)
    _skip_unported_referee()
    # A0: the run must win its first fight to reach the map at all.
    run = referee_spike.play(character="Ironclad", ascension=0, label="FAULT0", key_hex=KEY)
    # Void, not lost: the run stops at the first swallow with referee_fault.
    assert "error" not in run, run
    assert run.get("fault", "").startswith("host fault: catch:ForceToMap:"), run
    assert "injected fault" in run["fault"]
    assert "victory" not in run


def test_a_scoring_worker_does_not_say_where_the_host_failed(monkeypatch):
    monkeypatch.setenv("STS2_HOST_FAULT_INJECT", FORCED_SITE)
    with Host(referee=True, diag=False) as host:
        state = _referee_run(host, "FAULT1")
        rnd = referee_spike.random.Random("FAULT1")
        for _ in range(400):
            if state.get("type") != "decision":
                break
            state = host.send(**referee_spike.choose_legal(state, rnd))
    assert state == {"type": "referee_fault", "message": "host fault"}


def test_a_default_host_only_counts_swallowed_errors(monkeypatch):
    monkeypatch.setenv("STS2_HOST_FAULT_INJECT", FORCED_SITE)
    run = referee_spike.play(
        character="Ironclad", ascension=0, label="FAULT2", key_hex=None, census=True
    )
    assert "fault" not in run and "error" not in run, run
    sites = run["host_faults"]
    assert [s for s in sites if s.startswith("catch:ForceToMap:")], sites
    assert run["floor"] > 1
