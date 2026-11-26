# Writing a Spire Bench adapter

You are connecting an agent that thinks in its own simulator to the Spire
Bench referee. On every decision the referee sends a player-visible
observation (`PROTOCOL.md`). Your adapter turns that observation into your
simulator's state; this is the **rebuild**. Your agent decides on the
rebuilt state, and the adapter maps the decision back to a referee action.

The agent plays well through the referee only if the rebuild matches the
state your simulator would hold had it played the same run itself. Every
difference is a **divergence**. This guide covers how to find divergences
(the test loop), what usually causes them (the traps), and how to recover
what the view hides (inference).

It comes from making one agent, Instinct (a net that makes every decision,
no search), play the real game through the referee as it plays natively. On
300 matched seeds it started at 69.0% real vs 75.0% native (Ironclad A10).
After about fifty rebuild fixes it reached 78.7% vs 79.7% with a stronger
net. Every trap below cost that project at least one seed.

## 1. Choose the rebuild shape

There are two shapes, and most adapters end up as a mix of both:

- **Seedless rebuild.** Build the state from the current observation alone,
  every decision. This is simple and self-healing: one bad rebuild does not
  poison the next.
- **Mirror.** Keep your own simulator state across decisions: apply your
  action, step the simulator, then reconcile against the next observation.
  The mirror can hold what the view never shows (enemy AI logs, counters you
  tracked, cards you saw move).

Rebuild from the observation, and carry forward only the fields the view
cannot give you. For each carried field, know which observation field
proves it is still valid, and drop it when that field disagrees. A carried
value that is never checked against the board is where most of our mirror
bugs lived.

## 2. The test loop

Get this loop running before you fix anything. Without it you are guessing,
and a guessed fix moves the win rate by noise.

### 2.1 Make runs comparable

You need the same run played two ways:

1. **Native:** your agent in your own simulator, from the game seed.
2. **Through the view:** your agent on the real game, seeing only what the
   referee would show, and acting through your adapter.

If your simulator can deal the run the real game deals for a seed string
(same map, encounters, rewards), both sides start identical. Use a
**deterministic** agent (greedy, fixed tie-breaks, no sampling, a fixed
search seed). Then every pair should be the same run with the same result,
and **any pair that differs is a divergence by definition.** That property
makes the whole loop work. If your agent samples, fix its seed per decision
before testing.

If your simulator cannot reproduce the game's seeds, use the one-step check
(2.4) as your main tool.

### 2.2 Compare agent inputs, not outcomes

Win rates find nothing: one divergence on floor 12 changes everything after
it. Instead, log the agent's **input** at every decision on both sides: the
feature tensor, token list, or whatever your agent reads. Then find the
**first differing input** per seed:

1. Align the two decision streams by (act, floor, decision kind, index within
   the floor).
2. Walk both streams in order and stop at the first decision whose inputs
   differ.
3. Report the feature (or section) that differs, both values, and enough
   context (floor, screen, previous action) to reproduce it.

Only the *first* difference is trustworthy. After it the runs may play
different cards and the later differences are consequences, not causes.

Track three numbers per **dump** (one full pass over your seed set):

- mean floor of the first differing input (cap the "identical" seeds at the
  run's length);
- seeds identical for the whole run;
- seeds that got *worse* than the previous dump. This should stay zero, and
  a nonzero value is a regression.

Then bucket the first differences by feature and fix the biggest bucket.
After each fix, re-dump and confirm two things: the seeds in that bucket
moved later, and no other seed moved earlier.

A useful size: 40–50 seeds for the dump loop (a few minutes to an hour per
pass), and 300 seeds for the occasional outcome check.

### 2.3 Check outcomes last, paired

When the dump loop slows down, measure win rate on the same seeds both ways.
Report native, through-view, the count of seeds won only natively and the
count won only through the view, and an exact McNemar test on those two
counts. Unpaired win rates on a few hundred seeds have ±5 pp of noise and
will mislead you.

Measure "through the view" against "native", not against a
full-information arm that also goes through your adapter.
Both arms of that comparison share every rebuild bug, so it shows none of
them.

### 2.4 One-step prediction (the mirror check)

The other test needs no seed match. After you act, step your rebuilt state
with that action in your simulator, and compare the result with the next
observation: hand, energy, enemy HP, block and powers, the next intents.
Count **reconciles** (matched) and **desyncs** (did not). A desync is a
divergence found at the exact decision that caused it. This catches most
in-fight bugs. It cannot catch run-level ones (rewards, events, map), which
need 2.2.

### 2.5 Hygiene that bit us

- **Freeze the build while a dump runs.** Rebuilding the game host or your
  simulator mid-dump mixes two versions into one dump. Throw such a dump
  away.
- **Do not read a batch while it is still running.** Runs that finish first
  are biased (short runs are losses). Wait for the whole set.
- **Check that every episode started.** Our gateway refused episodes beyond
  its worker slots with HTTP 429 before any decision. Those rows looked like
  results until we filtered them out.
- **Keep every dump's diff file.** The history of first-difference floors is
  how you prove a fix helped and nothing else regressed.

## 3. Traps

Grouped by where they hit. Each trap describes the general mistake, with
the game case that showed it in brackets.

### Identity and indexing

- **Enemy `index` is a position among the living.** It shifts when an enemy
  before it dies. Key everything you carry (move history, inferred AI state,
  stolen cards) by `combat_id`. If you key by index or by slot, a kill or a
  re-sort hands one enemy's history to another.
- **Corpses.** The game removes an ordinary corpse from the enemy list. A
  reviver's corpse stays and arrives in `dead_enemies`. Your simulator may
  keep every corpse, or none. Decide which, and make sure the slot order of
  the living enemies matches your simulator's spawn order either way. A
  corpse keeps the intent it died with on the game's board.
- **Potion slots.** Use the belt slot the game reports, not the position
  among filled slots. They differ as soon as a slot is empty.
- **Duplicate keys.** Event buttons, event pages and cards can share an id or
  text key. Match by key *and* position (the n-th button with that key),
  never with a dict keyed by id.
- **Option caps.** If your simulator stores choice indices in a small integer
  type, check it against the largest screen (a combat card select can offer
  more than 255 options).

### Numbers the screen shows differently from your simulator

- **Floored intents.** The shown attack `damage` floors at 0. When Strength
  loss outweighs the base, use `base_damage` (the bestiary number, which the
  view carries) to rebuild the move, never the shown 0.
- **Grown cards.** A card whose damage grows during combat or the run
  (Rampage, a Wither grown by FakeUpgrade) shows the grown number. Your
  simulator may keep the base fixed and the growth in a separate counter.
  Split them the same way, or the growth is counted twice.
- **Non-attack damage vars.** A status's "Damage" (Infection's end-of-turn
  hit) is not attack damage. Read card vars by meaning, not by name.
- **Runtime-written amounts.** Some enemies telegraph a buff whose amount the
  game writes at runtime (Axebot, Entomancer). Rebuild the amount from the
  shown intent, not from the move's default.
- **Intent kinds.** Map every intent type the game sends to your simulator's
  vocabulary, and test the rare ones (a card-debuff intent landed as
  "Unknown" for us).
- **Keywords that belong to a power.** A card shown as Ethereal because of
  Hex is not a card with the Ethereal keyword. Rebuild the power and let it
  apply, rather than granting the keyword to each card.
- **Max energy.** Relics change max energy at different moments (Paels Flesh
  counts in the game's max but maybe not in yours; Pyre changes next turn's).
  A rebuild in the middle of a turn must keep this turn's value.

### Timing and snapshots

- **Run HP vs fight HP.** If your simulator records the run's HP at fight
  start, take it from the first observation of the fight. Do not take it
  from the current HP, and do not take it after a turn-1 heal (Blood Vial).
- **Act entrance.** At the start of an act the game has no current map node.
  Do not invent one: your simulator was trained (or tuned) on the act
  entrance, not on standing on the first node.
- **Boss relic timing.** Ask the boss relic where your simulator asks it (the
  end of the act), even when an Ancient's event page comes first.
- **Reward order.** Take a room's rewards in the order your simulator offers
  them (relics before the potion; an event fight's extra potion after the
  card pick). Order changes which belt slot a potion lands in, and with it
  later decisions.
- **Screens that open inside a turn.** A relic or card can open a card
  select at turn start (Toasty Mittens' exhaust) or in the middle of a play.
  Route it to the same decision head your agent uses natively, not to a
  default.
- **A fight that opens on a screen.** Derive the fight's identity and
  starting HP from this fight, not from the last one.

### Multi-pick and select screens

- **Pick-N as single picks.** If your simulator asks N picks one at a time,
  replay the earlier picks into the rebuilt state before asking the next.
  The earlier picks must leave the offer, and the "picked so far" count must
  be right.
- **Minimums and skips.** Copy `min_select` / `max_select` / `cancelable`
  exactly. A skippable choose-a-card is "pick one, with a skip", not "pick
  none".
- **Amounts on the screen.** An enchant screen applies an amount
  (`enchantment_amount`). Read it from the observation, never a default of 1.
- **Card grants vs card rewards.** Some relics offer a choose-a-card that
  plays like a card reward (Lead Paperweight). Use whichever decision your
  agent expects there.

### Events

- **Rolled amounts.** Gold, HP and damage that an event rolled are in the
  option text. Parse them from the screen; never let your simulator roll its
  own (Jungle Maze's gold).
- **Replayed pages.** If you rebuild a multi-page event by replaying earlier
  pages, the replay must hold the screen's deck, HP and gold at each page.
- **Fight-starting events.** An event that starts a fight needs the
  encounter id your simulator would use for that event's fight.
- **Event-specific special cases.** Some events change the run in a way your
  simulator models with a special rule (Wood Carvings only transforms Basic
  cards). Check each event your simulator special-cases.

### Encounters and enemy state

- **Spawn-rolled rosters.** Some encounters roll their roster at spawn, so a
  lookup from the encounter's fixed list misses them. Infer the encounter from
  the set of monsters you see. Prefer the most specific match.
- **Enemies with internal counters.** A sleep counter, ramping damage (the
  Waterfall Giant's gun), or a "woken by damage" flag (the Lagavulin
  Matriarch) is not in the view. Infer it (section 4) and carry it, keyed by
  `combat_id`.
- **Stolen cards.** A thief shows the card it holds (`stolen_cards`). That
  card is out of your deck until the thief dies. Count it out once, not once
  per rebuild.

### Run state

- **Ascension.** Pass the ascension into every rebuild. A rebuild defaulting
  to A0 plays wrong numbers everywhere.
- **Enchantments.** Deck cards carry enchantments. Rebuild them on the card,
  in the run deck as well as in combat piles.
- **Relic counters.** Rebuild counters the way your simulator counts them
  (Sword of Stone counts down in one, up in the other).
- **Second boss.** At A10 the act-3 second boss is shown from act entry
  (`map.second_boss`). Read it there.

## 4. Inferring what the view hides

The view hides only what a player could not know: draw order, RNG, and
enemy AI state. Everything else is either in the view or can be derived
from what you have seen. `PROTOCOL.md` lists the derived values the referee
already computes for you ("What is shown that the game's UI does not
print"). Use those before deriving your own.

Who keeps which state:

| State | Where it lives |
|---|---|
| Exact values the game reads (the `player.*` counts, seen draw-pile positions) | The view. The referee computes them. |
| Guesses about hidden state (enemy moves, enemy counters) | Your adapter. `bench_agents/move_infer.py` is ours; its output is a best guess, not ground truth. |
| What earlier screens showed, and your own plans | Your agent's memory. |

- **Enemy moves.** The view shows intents, not move ids. Keep a per-enemy
  move log keyed by `combat_id`. On each new telegraph, ask your simulator
  which moves can follow the last one, render each candidate's intents, and
  keep the one that matches what the view shows. Match an enemy's first move
  among its opening moves. When two candidates render identically, take the
  most likely and mark it ambiguous; the next telegraph usually tells them
  apart. (`sts2-pycli/python/bench_agents/move_infer.py` does this.)
- **Draw order.** `draw_pile[]` is sorted and tells you nothing about order.
  `draw_pile_top[]` and `draw_pile_bottom[]` give the cards whose position
  you saw (put on top, put at the bottom). Pin those and treat the middle as
  unknown. A search agent should sample the middle (determinize), not read
  the sorted order as the draw order.
- **Combat history.** Cards and powers that read "this turn" / "this combat"
  counts (Spite, Tear Asunder, Unmovable, Gang Up, extra turns) get them from
  `player.*` counters in the view. Anything else you need (cards played this
  turn, discard order), record yourself from the stream of observations.
- **Encounter id.** Infer it from the enemy roster at the start of the
  fight, counting `removed_enemies[]` killed before your first decision.
  Each starting roster belongs to one encounter (checked on v0.111.0 and
  v0.113), so the match is exact once you compare monster counts as well as
  kinds: Bowlbugs, Slimes, Corpse Slugs and the like differ between their
  weak and normal encounters only in how many spawn. Remember spawn-rolled
  rosters and event fights (section 3).
- **Bestiary numbers.** An attack's base damage per hit is in
  `intents[].base_damage`. Strength, Weak and Vulnerable are visible powers,
  so you can check the shown damage against base and modifiers. A mismatch
  means a modifier is missing from your rebuild.
- **Your own memory is legal.** Anything you saw on an earlier screen (a shop's
  stock, an event's outcome, the cards a thief took) is yours to remember.
  The referee does not track it for you.

## 5. Order of work that paid off

1. Run 2.4 (one-step prediction) on a few dozen fights. Fix desyncs until
   they are rare, because fight bugs are the most common.
2. Set up 2.1 and 2.2, dump 40–50 seeds, and bucket the first differences.
3. For each bucket: write a test that rebuilds the failing observation and
   asserts the field. Fix it, re-dump, and check that no seed got worse.
4. Every ten or so fixes, run the paired outcome check (2.3) on 300 seeds.
5. Stop when the remaining buckets are inherent to the view: the seeds that
   still differ do so only in hidden information. That is the price of the
   view, and it shows up as the difference between native and through-view.
