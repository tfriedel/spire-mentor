# Spire Bench referee protocol (draft 0)

Writing an adapter from this view to your own simulator? Read `ADAPTER-GUIDE.md`: how to test it, the common traps, and how to infer hidden state.

Status: draft for review. Covers what one referee worker (the headless host in
referee mode) exchanges with whoever drives it, and the gateway (milestone 2,
protocol 0.2) that puts a pool of workers behind HTTP without changing the
observation or action shapes.

Plan: `docs/plans/2026-11-11-spire-bench.md`.
Code: `sts2-pycli/src/Sts2Headless/RefereeRng.cs`, `RefereeView.cs`; the
gateway, verifier and SDK in `sts2-pycli/python/spire_bench/`.

## Worker lifecycle

One process plays one episode. Newline-delimited JSON on stdin and stdout.

1. Start the host with `STS2_REFEREE=1`. It prints
   `{"type":"ready","referee":true,"game_version":"v0.111.0"}`.
2. Send `start_run`. The reply is the first decision.
3. Send one `action` per decision until the reply is `decision: "game_over"`.
4. Send `quit`.

```json
{"cmd":"start_run","character":"Ironclad","ascension":10,"seed":"EP-000123","rng_key":"<64 hex chars>"}
{"cmd":"action","action":"play_card","args":{"card_index":2,"target_index":0}}
```

- `seed` is a public label. It decides nothing by itself; all randomness is
  keyed by `rng_key`, which never reaches the client.
- Act 1 is rolled like the real lobby does. A potion reward on a full belt is
  a `potion_swap` decision, never a silent drop.
- After a combat and in a treasure room every reward is the player's to take
  or skip, in any order, on a `rewards` decision. A reward index keeps its
  value while other rewards are taken. The chest's gold is paid when the
  chest opens, as in the game; its relic is on the `rewards` screen.
- Every other command (`get_state`, `get_map`, `set_draw_order`, `rng_probe`,
  `load_save`, `dump_*`, ...) is refused in referee mode.

## Replies

| `type` | Meaning |
|---|---|
| `decision` | The observation for the next decision (below). |
| `error` | The action was not applied; `message` says why. Send another action for the same decision. |
| `referee_fault` | A draw bypassed the keyed generator. The process exits; the episode is void, not a loss. |

## Observations

An observation holds only what the game screen shows the player. The host
builds it by projecting its internal payload onto an allow-list
(`RefereeView.cs`); a field that is on neither the allow-list nor the deny-list
is dropped and reported to the operator, so a new field cannot leak by default.

Common to every decision: `decision`, `context` (act, floor, room type, act
boss), `player` (hp, gold, deck, relics, potions), and `map`: the whole act
as the map screen shows it (`rows[][]` with `type`, `children`, `visited`,
`current`; `boss`; `current_coord`), since the player can open the map at
any time. `game_over` carries only the outcome.

| `decision` | Main fields |
|---|---|
| `map_select` | `choices[]` (reachable nodes), `map` |
| `combat_play` | `hand[]`, `draw_pile[]`, `discard_pile[]`, `exhaust_pile[]`, `play_pile[]`, `enemies[]`, `player_powers[]`, `energy`, `round`, `orbs[]`, `osty`, `stars` |
| `card_reward` | `cards[]`, `can_skip`, `gold_earned` |
| `card_select` | `cards[]`, `min_select`, `max_select`, `cancelable`, `source`, `combat` (the combat observation, when mid-combat) |
| `bundle_select` | `bundles[]` |
| `event_choice` | `event_name`, `description`, `options[]` |
| `rest_site` | `options[]`, `can_leave` |
| `shop` | `cards[]`, `relics[]`, `potions[]`, `card_removal_cost` |
| `rewards` | `rewards[]`: `index`, `type` (`gold`, `relic`, `potion`, `card`, `special_card`, `card_removal`, `other`), `amount` (gold), `id`, `name`, `description` |
| `potion_swap` | `potion`, `options[]` |
| `crystal_sphere` | `divinations_left`, `divinations_total`, `grid`, `cells[]` |
| `game_over` | `victory`, `act`, `floor` |

An enemy's `index` is its position among the living enemies (what
`target_index` takes), so it shifts when an enemy before it dies. Its
`combat_id` is the creature's id in the fight, in creation order, and does
not change: track an enemy across turns by `combat_id`.
`slot` is its position in the encounter's slot order, the order the board
is sorted in. A summon takes the first free slot, which can be one an enemy
that died left behind.

The game takes an enemy that died off the board. `removed_enemies[]` lists
those, in creation order, with `id`, `name`, `combat_id`, `max_hp` and the
intent they died with: you watched each of them die, also one killed before
your first decision (a relic's start-of-combat damage). `dead_enemies[]` are
the corpses that stay on the board because they come back.

### What is hidden

| Hidden | What the client gets instead |
|---|---|
| Draw pile order | `draw_pile[]` sorted by content, without positions. Its contents and size are public, as in the game's draw pile view. |
| RNG seed, stream counters, episode key | Nothing. |
| Enemy AI state: move log, forced next move, current move id | `enemies[].intents[]` (type, damage, hits, base_damage, card_count) and visible powers. |
| Future rooms, rewards, shop stock, event outcomes | Each appears only on the screen that shows it. |

### What is shown that the game's UI does not print

The view may carry a value the game's screen does not show as a number,
when a player could work it out from public information: the bestiary, the
card and relic texts, and what the run has shown so far. It never carries
anything a player could not know (the hidden column above).

A value joins the table below only when all three hold:

1. **Exact, not likely.** A player who took perfect notes could compute it
   with certainty. A value that is right in most cases fails: the cases
   where it is uncertain are the ones where sending it leaks. Enemy move ids
   fail this (two moves can show the same intent), so they stay hidden.
2. **A fact, not analysis.** The view carries game state, never a
   conclusion drawn from it: no draw odds, lethal checks or expected
   damage. Working those out is what the benchmark measures.
3. **The game reads it.** Some card, relic or power text refers to it, and a
   client could only rebuild it by tracking earlier observations.

Anything on a screen the player can open (the deck, a relic's counter, the
draw pile's contents, the map) needs none of this: it is in the view as
shown. The table stays a short closed list, because the referee has to
compute each entry correctly for good. A field is added when two
independent adapters would otherwise write the same exact tracker.

| Field | What it is |
|---|---|
| `enemies[].intents[].base_damage` | An attack's own damage per hit, before the attacker's Strength, Weak and the like and before your Vulnerable: the bestiary number for the move at this ascension. `damage` is the number the game shows; it floors at 0, which hides the base when Strength loss outweighs it. |
| `draw_pile_top[]`, `draw_pile_bottom[]` | Draw-pile cards whose position you saw: put on top (Headbutt) or at the bottom, and not drawn or moved since. Top first and bottom first. A shuffle clears both, and so does a card shuffled in at a random spot, since you cannot tell where it landed. `draw_pile[]` itself stays sorted. |
| `player.hp_lost_this_turn`, `player.unblocked_hits_this_combat`, `player.cards_exhausted_this_combat`, `player.cards_exhausted_this_turn`, `player.card_block_gains_this_turn`, `player.powered_hits_this_turn[]`, `player.extra_turns_taken_this_combat`, `player.cards_drawn_this_combat`, `player.cards_generated_this_combat`, `player.discards_this_turn`, `player.bound_card_played_this_turn` | Counts of things that happened in front of you this turn or this combat, which cards and powers read (Spite, Tear Asunder, Midnight, Evil Eye, Unmovable, Gang Up, extra-turn relics, Murder, Supermassive, Memento Mori). |
| `enemies[].powers[]` Painful Stabs `vars.ExtraMultiClawCount` | Test Subject's extra Multi Claw hits: one more for each Multi Claw you watched it use in its second form. |

Anything else a bot has seen stays its own to remember; the referee tracks
only the fields listed above.

## Actions

| `action` | `args` | Legal when |
|---|---|---|
| `select_map_node` | `col`, `row` | the node is in `choices[]` |
| `play_card` | `card_index`, `target_index` (if `target_type` is `AnyEnemy`) | `hand[i].can_play` |
| `end_turn` | | `combat_play` |
| `use_potion` | `potion_index`, `target_index` (if it targets) | in combat: the slot holds a usable potion; outside combat: an `AnyTime` potion (Fruit Juice, Blood Potion, ...) on `map_select`, `event_choice`, `rest_site`, `shop` or `rewards` |
| `discard_potion` | `potion_index` | the slot holds a potion, in combat or on one of the screens above |
| `choose_option` | `option_index` | the option is not locked / is enabled |
| `select_card_reward` | `card_index` | `card_reward` |
| `skip_card_reward` | | `can_skip` |
| `select_bundle` | `bundle_index` | `bundle_select` |
| `select_cards` | `indices` (comma-separated string) | between `min_select` and `max_select` cards |
| `skip_select` | | `cancelable`, or `min_select` is 0 |
| `buy_card` / `buy_relic` / `buy_potion` | `card_index` / `relic_index` / `potion_index` | stocked and affordable |
| `remove_card` | | `shop`, affordable |
| `take_reward` | `reward_index` | the reward is still on the `rewards` screen. A `card` reward opens `card_reward`, a `card_removal` opens `card_select`, a potion on a full belt opens `potion_swap`. |
| `skip_reward` | `reward_index` | the reward is still on the `rewards` screen |
| `potion_swap` | `discard_slot` (omit to decline) | `potion_swap` |
| `crystal_sphere_click` | `x`, `y`, `tool` (`big` or `small`; default `big`) | `crystal_sphere`, and the cell is still fogged |
| `leave_room` | | `shop`, or `rest_site` with `can_leave` |
| `proceed` | | `rewards` (leaves every reward still listed), or a screen with nothing left to choose |

Indices refer to the `index` field of the listed item in the current
observation.

### The Crystal Sphere minigame

The client plays it, one click per divination, as the player does. The
referee offers every click the game takes: either tool on any fogged cell,
including clicks no sensible policy would make.

- `cells[]` is the whole grid: `x`, `y`, `hidden`. A cleared cell that shows
  part of an item also has `item`: `kind` (the texture the screen draws, such
  as `crystal_sphere_relic` or `crystal_sphere_rare_potion`), the item's
  `width` and `height`, and `part_x`, `part_y`, the cell's place inside it. A
  fogged cell says nothing about what is under it.
- `big` clears the clicked cell and its eight neighbours. `small` clears the
  clicked cell only. Each costs one divination.
- An item pays out only when every cell it covers is clear. The rewards are
  handed out after the last divination; a card among them opens `card_reward`.
- The game offers no way to leave with divinations unspent, so neither does
  the referee.

## Gateway (protocol 0.2)

`just bench-serve bench.json` (from `sts2-pycli/`) serves HTTP/JSON. Every
request except `GET /v0/info` and `GET /v0/scores` carries
`Authorization: Bearer <token>`; the config maps tokens to agent ids.

| Request | Body | Reply |
|---|---|---|
| `GET /v0/info` | | protocol, game version, engine hash, limits |
| `POST /v0/episodes` | `character`, `ascension`, `count` | `episodes[]`: one episode state each |
| `POST /v0/step` | `actions[]`: `episode_id`, `seq`, `action`, `args` | `results[]`, in the same order |
| `GET /v0/episodes/{id}` | | the episode state |
| `GET /v0/scores` | | per agent, character and ascension: finished, won, `win_rate`, `wilson_95`, running, void |

An **episode state** is `episode_id`, `seq`, `status` (`running`, `won`,
`lost`, `void`), `fingerprint`, `observation` (the worker's decision, above),
`outcome` (`status`, `reason`) once it is over, and `error` (`code`) when the
action was not applied.

`fingerprint` is the agent's **receipt**: the head of the episode's reply
chain so far. Keep the last one. After the season key is revealed, `verify`
prints the replayed fingerprint of every published transcript; if it differs
from your receipt, the transcript of your episode was cut short or changed.

- **Batching.** One `step` carries actions for many episodes; they run in
  parallel, each on its own worker. An episode may appear once per batch
  (`duplicate_in_batch`).
- **Sequence numbers.** An action names the `seq` of the observation it
  answers. Resending the last applied action returns the same reply, so a
  client can retry a step whose reply was lost. Any other `seq` is refused
  (`stale_seq`). There is no rewind.
- **Refused actions** (`error.code` from the worker) leave the decision and
  `seq` as they were. `max_invalid` refusals in a row lose the episode
  (`invalid_actions`).
- **Deadlines.** No action within `decision_deadline` seconds, or no finish
  within `run_deadline`, loses the episode (`decision_deadline`,
  `run_deadline`). An abandoned episode is a loss.
- **Void.** A worker that fails to start, or a gateway restart, voids the
  episode: the referee failed and the agent could not have caused it. Void
  episodes are reported but not scored.
- **Referee failure during an action.** A `referee_fault` or a worker that
  dies while applying the agent's action loses the episode
  (`referee_fault_during_action`, `worker_error_during_action`), with the
  triggering command in the transcript's `end` row for operator review.
  Voiding it would let an agent that found a crashing action wipe a run it is
  losing.
- **Capacity.** At most `max_workers` episodes run at once; a start beyond
  that is refused with HTTP 429. Another agent's episode is `unknown_episode`
  (HTTP 404 on `GET`).
- **Scores** pool every episode an agent ever started. It cannot drop an
  episode it is losing: abandoning it is a loss.

### Keys and transcripts

The episode key is `HMAC-SHA256(season_key, "<agent_id>:<episode_no>")`, and
the public label is `SB-<agent_id>-<episode_no>`. Keys are per agent, so a
second account cannot scout another's dice.

Each episode writes an append-only transcript (`<state_dir>/transcripts/
<episode_id>.jsonl`):

- a `header`: protocol, agent, episode number, character, ascension, label,
  game version, `engine_hash` (sha256 over `sts2.dll` and the host assembly),
  limits, and `season_id` (below);
- one `command` row per command sent to the worker, refused ones included,
  with the sha256 of its reply;
- an `end` row: status, reason, act, floor, and the `fingerprint`. A loss by a
  host failure during an action also has `trigger` (the action) and `failure`
  (`exited` or `timeout`).

The fingerprint is a sha256 chain (`spire_bench/chain.py`) seeded with the
header's identity (protocol, episode, agent, character, ascension, label,
game version, engine hash, limits, season id), then every reply's sha256, then the
trigger of a host failure. So the agent's receipt covers the header and the
blamed action as well as the replies.

No key is written. After the season key is revealed,
`just bench-verify <season key> [--max-invalid <season limit>] <transcript>...` replays each
transcript on a local host built from the same engine and checks that:

- the header's label and episode id are `SB-<agent_id>-<episode_no>`, and the
  first command, the only `start_run`, plays that header's character,
  ascension and label;
- every reply replays bit for bit under the episode key, and the reply chain
  gives the `end` row's fingerprint;
- the claimed outcome follows from the replay:

  | Outcome | Checked by replay |
  |---|---|
  | `won`, or `lost` by `game_over` | the last reply is the game's victory or defeat |
  | `lost` by `invalid_actions` | the transcript ends with exactly the header's `max_invalid` refused commands |
  | `lost` by `referee_fault_during_action` / `unexpected_reply_during_action` | the `end` row's trigger is the last command, and its reply fails again |
  | `lost` by `worker_error_during_action` | the trigger is a plain action (`cmd`, `action`, `args` only), and resending it after the replay fails the host the same way (`exited` or `timeout`) |
  | `lost` by `decision_deadline` / `run_deadline` | only that the game was still going; timing does not replay, so the receipt is the check |
  | `void` by `worker_error` | no command was played |
  | `void` by `referee_fault` / `unexpected_reply` | only `start_run` was played, and it fails again |
  | `void` by `server_restart` / `server_shutdown` | the game was still going; the receipt is the check |
- each episode number has one transcript (a second would let a publisher show
  the better of two runs).

A void with nothing to replay (the worker never started, or the gateway
restarted) is reported as such, not replayed.

### Seasons (commit-reveal)

Where the season key comes from (`spire_bench/season.py`):

1. **Before the season** the operator runs
   `just bench-season-commit <season> <drand chain hash> <round> season/` and
   publishes `season/commitment.json`: `sha256(K)` of a fresh secret 32-byte
   `K` (kept in `season/K.secret`), a [drand](https://drand.love) round still
   in the future, the game version, engine hash and limits.
2. **When the round is out** the season key is
   `HMAC-SHA256(K, randomness of that round)`. The operator cannot steer it
   (`K` was fixed before the beacon value existed) and nobody else can compute
   it (`K` is secret). The gateway's `bench.json` names the season instead of
   a key: `"season": {"dir": "season/", "drand_randomness": "<hex>"}`. It
   takes its limits from the commitment, and refuses to start when `K` does
   not match it or the engine is not the committed one.
3. `season_id` = sha256 of the commitment. `GET /v0/info` reports it, every
   transcript header carries it, and it is in every receipt.
4. **After the season** the operator runs
   `just bench-season-reveal season/ --state <gateway state_dir>` (it fetches
   the round's randomness from drand's relay, checking it is sha256 of the
   signature) and publishes `reveal.json` (the commitment, `K`, the
   randomness, and `episodes`: how many episodes each agent started, from the
   gateway's index) with every transcript. Each agent checks its count against
   the number of episodes it started.

Readers check the reveal with `python -m spire_bench.season check reveal.json
--fetch`; the BLS signature of the round is drand's own tooling.
`just bench-verify-season reveal.json <transcript>...` checks `K` against the
commitment, derives the key and replays every transcript, holding each to the
`season_id` and to the committed game version, engine hash and limits.

### Leaderboard

`just bench-leaderboard reveal.json site/ <transcript>...` replays every
transcript and writes a static site: `index.html`, `leaderboard.json`, the
commitment, the reveal and a copy of each transcript. An agent is scored
(pooled Wilson interval per character and ascension, every episode started
counts, voids reported but not scored) only when **all** of its transcripts
verify and they are exactly episodes 0..n-1, n being the reveal's count for
that agent: a missing transcript, the last one included, could be a lost run
left out. Agents failing either check, or missing from the reveal's counts,
are listed with the reasons instead. `leaderboard.json` lists every episode's
replayed fingerprint, for the agents to compare with their receipts.

### SDK

`spire_bench.client` uses the standard library only: `SpireBench` (the HTTP
calls, with safe retries of `step`), `VecEnv` (n episodes stepped in one
request) and `random_action` (a random legal action).
`just bench-random-bot <token>` runs the random bot.

## Open questions for review

1. ~~**Enemy move ids.**~~ Settled: hidden. A player can usually infer the
   move from the intent, but two moves can show the same intent, and then
   the id says more than the screen does. A client infers it
   (`ADAPTER-GUIDE.md`, section 4).
2. ~~**Second boss at high ascension.**~~ Settled: the game shows it from act
   entry (top-bar icon and map node), so `map.second_boss` is sent from then
   (GH #1662).
3. ~~**Explicit legal-action lists.**~~ Built: every observation carries
   `legal_actions[]`.
4. ~~**Error text.**~~ Built: errors carry a fixed `code`.
5. **Headless modelling choices** that differ from the real game and need a
   decision each. Both settled: the Crystal Sphere minigame is the client's
   (see above) and `Trial.DoubleDown` abandons the run, as in the game.
6. ~~**Intent-hiding effects.**~~ Settled: on v0.111.0 and v0.113 no
   power, relic, card, affliction, enchantment or modifier hides or alters an
   enemy's intent; the screen always draws the monster's next move
   (`NCreature.UpdateIntent` reads `Monster.NextMove.Intents`). The view sends
   what the screen draws: the type, for an attack the number shown
   (`AttackIntent.GetSingleDamage`) and the hit count, for a status intent
   the card count. `UnknownIntent` (the "?" icon) is never used. The one
   intent the screen hides, `HiddenIntent` (no icon, number or tooltip; only
   the Architect and test monsters), is dropped from the view, so it reads
   like a move with no intents. A later build that adds an intent-hiding
   effect reopens this.
7. **Relic status.** `player.relics[]` carries the counter the game draws on
   the icon, but not the icon's state: pulsing while an effect is pending
   (Venerable Tea Set after a rest site), greyed out once it is used up.
   That is on screen, so it belongs in the view. Not built (GH #1882).
8. **Encounter id.** Not in the view. Every starting roster belongs to one
   encounter (checked on v0.111.0 and v0.113), so the id is exact and could
   be sent as a derived value. Send it, or leave it to the client?
