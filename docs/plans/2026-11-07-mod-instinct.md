# Instinct in the mod: the e2e student as the live game's advisor

**Status:** implemented; two live A10 runs won with Instinct as the brain,
97% of advice requests answered by Instinct (`logs/2026-11-07-instinct-live/FINDINGS.md`).
Branch `feat/mod-instinct`, stacked on #1098 like #1156.
**Goal:** the player picks the advisor's brain in the game, **Search** (today:
MCTS for fights, the PPO run policy for screens) or **Instinct** (the e2e
Ironclad student, #1097: one net, every decision, no search), exactly as the
browser workbench offers it since #1156. Everything the mod does with advice
follows the pick: the HUD, F6 / F7, Autobattle, the coach's choice chip and the
AI co-op partners. Later students for other characters plug into the same
path; the search may go away eventually.

## Architecture

```text
mod (C#)                         advisor (Python)                       sts2_rs (Rust)
advice_request {brain}  ──────▶  realgame_advisor._handle_advice_request
                                   brain == "instinct"?
                                   └─ realgame_instinct.InstinctAdvisor.advise
                                        graft_run(envelope)  ───────────▶ RunManager(seed) + debug_* grafts
                                        + the screen's options injected ─▶ debug_open_card_offer / debug_enter_event / ...
                                        E2eDriver.peek(rm)   ───────────▶ sts2_e2e::env::Driver (merged classes)
                                        E2eStudent.decide    ───────────▶ tokens::encode + E2eNet forward
                                        classes → the screen's option index / stable_idx
                                   InstinctUnavailable → the search answers, says why
advice {brain, options[prob], best} ◀──
```

- **Rust** (`sts2-py/src/e2e.rs`): `sts2_rs.E2eStudent` (the `.e2ew` net),
  `E2eDriver` (the env driver + tokenizer cache: the student's per-run state),
  `E2ePeek`, plus `e2e_new_run` / `e2e_selfplay` for checks. The driver, the
  tokenizer and the forward are the parity-gated port from #1156; nothing is
  re-implemented. `sts2-e2e` joins `sts2_srchash::ENGINE_CRATES` (its
  `data/*.tsv` tables are hashed too) and the uv cache keys.
- **Python** (`realgame_instinct.py`): rebuilds the live run as an engine run at
  the decision, keeps the student's per-run state (one driver per run; every
  combat board observed so the fight's intent log, a policy input, is the
  live fight's), maps the student's merged classes onto the screen.
- **Mod** (C#): the setting, the toolbar switch, `brain` on every request
  (the human's and each AI partner's), the reply's brain shown on the HUD.

### Why inject, not re-roll

The envelope's `rng` block carries every stream's cursor but none of the
history-dependent state (card-rarity odds, unknown-room odds, visited events,
relic grab bags, potion-drop odds). Re-entering a room from the cursors can
roll a different shop / event / reward than the screen shows. So each screen's
options go into the engine as shown (`debug_*` setters that take what the
matching getter emits), and the engine's options are checked against the
screen before the student's answer is trusted.

## Protocol (PROTOCOL.md §5, v5.2, additive)

- `advice_request.brain`: `"search"` | `"instinct"` (absent = search).
- reply `capabilities.instinct`: `[{character, game_version}]` this advisor's
  Instinct plays (empty: none). The mod shows the switch only for a listed run.
- `advice.brain`: who answered. `advice.instinct_unavailable`: why the search
  answered although Instinct was asked (also prefixed to `note`).
- Instinct's options: probability on `visit_share` (the HUD's bar) and `prob`;
  no `p_win` / `q` (a policy, not a value).

## Surfaces

| `advice_for` | engine decision | how the engine gets there | maps back by |
|---|---|---|---|
| `combat_action` | COMBAT_TURN | graft + the search's own board (`combat_state_for`, mirror's carried state when live) + `debug_enter_combat` | the search's `stable_idx` helper |
| `card_select` (in combat) | COMBAT_CHOICE | as combat, board with the pending choice (the fight mirror) | option index |
| `map_select` | MAP | graft (parks on the node) | column → `nodes[].index` |
| `card_reward` / `card_bundle` | CARD_REWARD / CARD_BUNDLE | `debug_open_card_offer` | position |
| `event` (incl. Neow) | EVENT / NEOW | `debug_enter_event` + replay of this event's earlier picks; Neow: a fresh Neow-start run | `text_key` tail ↔ engine option id |
| `rest_site` | REST | `debug_enter_rest_site` | option id |
| `card_select` (smith) | SMITH | rest site + `take_action(SMITH)` | deck card identity |
| `card_select` (remove / upgrade / transform / enchant / grant) | CARD_* | `debug_queue_*` | deck card identity |
| `shop` | SHOP | `debug_open_shop` | (kind, id, price) |
| `boss_relic` (act 2/3 Ancient) | BOSS_RELIC | `debug_open_boss_relic` | relic id |
| `treasure` | forced | — | Instinct has no choice: the search answers |
| `potion_swap` (new lane) | POTION_SWAP | `debug_queue_potion_swap` | `discard_slot` |

Shop removal is the one split that differs: the engine picks the card inside
the shop decision (`shop_remove_card{deck_idx}`), the game asks on a second
screen (`card_select`, `TO_REMOVE`). The advisor answers the shop with
`remove_card` and carries the student's card to the follow-up screen.

The potion swap (full belt) was answered by the mod itself on the reward
screen; it is now an advice lane (`advice_for: "potion_swap"`, asked only of an
advisor advertising `capabilities.potion_swap`), answered by Instinct or, under
Search, by the run policy's swap head (declined when that head is off, as the
engines do).

Where the student has one move (a lone legal action, or one merged class),
Instinct answers it at probability 1 rather than handing it to the search.

Code: `realgame_instinct.py` (graft, fights, the map, in-combat choices via the
fight mirror's fork with an Instinct `picker`), `realgame_instinct_screens.py`
(every other screen). Graft setters: `sts2-wire/src/graft.rs` + `sts2-py`
bindings, round-trip tested in `sts2-wire/tests/graft_setters.rs`.

## Verification

1. **Binding:** `e2e_selfplay` and the Python driver loop reproduce the
   workbench's seed 170000000 run (won, 634 decisions, identical choices).
2. **Grafts:** Rust round-trip tests per setter (`setter(getter())` gives the
   same actions).
2b. **Fidelity, every surface** (`tests/test_realgame_instinct*.py`): a native
   campaign run is played to the first decision of each type; the mod's payload
   for that screen is synthesized from the engine there; Instinct answering it
   must make the same decision with the same probabilities (±5e-3) as the
   native student. Passes for MAP, NEOW, CARD_REWARD, SHOP, REST, SMITH, EVENT,
   BOSS_RELIC, CARD_UPGRADE, CARD_GRANT and POTION_SWAP.
3. **Live corpus:** `realgame_advisor --record-requests DIR` writes every raw
   message; a replay test feeds a recorded v0.111.0 Ironclad run through
   `InstinctAdvisor.advise` and reports, per surface, how many positions it
   answered and why it refused the rest.
4. **Live corpus:** `tests/test_realgame_instinct_live.py` (slow) replays the
   first live run's recorded requests through `InstinctAdvisor`
   (`realgame_instinct_replay.py`), allowing only the designed declines.
5. **Live:** two unattended A10 runs (F7, `--advisor-brain instinct`): both won,
   0 refusals; run 2 had Instinct answer 776 of 798 requests.

## Co-op (AI partners)

The engine simulates single-player runs; the game's co-op acts are one room
shorter (`ActModel.GetNumberOfRooms(isMultiplayer)`), so a co-op map is not
the engine's and the envelope carries no `rng` block. In co-op Instinct stands
on the nearest node of the engine's map in the same act (act, floor and room
context right, the map read approximate), declines `map_select` (the search
answers it), and says "co-op: Instinct trained single-player" in its note. The
student never saw allies; an ally-targeted move that does not map is dropped.

## Open issues

- A card generated into the fight (`card_reward` `surface: "choose_a_card"`)
  goes through the fight mirror's selection path; live, the mirror could not
  stand on it 14 times in run 2 (no source fields to replay).
- Colossal Flower: the engine's "Reach Deeper" goes straight to a reward, the
  game has two more pages; those pages decline.
- Not read, the search answers: enchant screens (the screen does not name the
  enchantment), the chest relic (forced in the student's game).
- Not exercised live yet: the AI partners with Instinct, Autobattle, a
  hand-played run with the HUD.
- An Act 2/3 Ancient's relic offer is shown to the student with the NEW act's
  index (training saw the old one at that moment); the boss-node placement that
  reproduces it needs the previous act's map.
- Instinct trained at A10; other ascensions still get an answer.
