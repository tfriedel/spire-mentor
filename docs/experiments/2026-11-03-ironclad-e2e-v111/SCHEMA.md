# Ironclad e2e v0.111.0: input and interface schema

Work package WP3a of the Ironclad end-to-end no-search PPO campaign (GH #1097,
plan `docs/plans/2026-11-02-ironclad-e2e-ppo-v0111.md`, §4-§5). This document
freezes, for G-schema, every interface the env, tokenizer, network, learner,
server and evaluation work packages share.

| | |
|---|---|
| Status | Draft for G-schema (Mon 21:30); schema revision 2: revision 1 plus the critic's fixes (honest event previews, the bootstrap rule, a stricter validator, the teacher stamp) |
| Source of truth | `sts2-pycli/python/ppo/e2e/contracts.py` (this file is its prose) |
| Content-ID table | `sts2-pycli/python/ppo/e2e/vocab.tsv`, built by `ppo/e2e/vocab.py` |
| Tests | `sts2-pycli/tests/e2e/test_contracts.py` |
| `schema_sha256` | `7afdc933284d80e400138267d46e51e2625a3ff09671d1028b131300c09f7f79` |
| Vocabulary sha256 | `bac088c69d8e149cd9cb7ac81a07d4a3af360d35b7da37f9cc16537705667aaf` (vocabulary revision 2.1) |
| Changelog | Vocabulary revision 2.1 (GH #1097 item A3): ids 1939-1947 append the nine v0.111.0 event-fight encounters (`vocab.EVENT_ENCOUNTERS`), so an event's fight no longer reads `UNK_ENCOUNTER` in X_ID_AUX. Revision 2.0 was `1b98f7630089ad1586c613ec3316f00ab8f45715716a6e1c56d84d1f36887a0b`. `schema_sha256` and `prior_sha256` are unchanged. |
| Game version | v0.111.0 (engine wheel W1, merge commit 879e44015) |

If this document and `contracts.py` disagree, `contracts.py` wins and this
document has a bug. The tables in sections 2, 6 and 10 are generated from
`contracts.py`. "Section 2.1" points into this document; "§3.2" points into
the plan.

## 1. At a glance

- **One row per decision.** A decision is a stored policy step with at least
  two merged options. A step with one legal action, or with one merged class,
  gets no forward pass and no row; the env still runs `honest_snapshot` and
  the intent log on it.
- **One token list per row.** Every row starts with the same head: `RUNCTX`
  (the state token), five `SUMMARY` tokens and one or two `BOSS` tokens. A
  combat row continues with #1081's S_ctx tokens, unchanged and contiguous. An
  out-of-combat (OOC) row continues with the run body, then its option tokens.
- **Two column blocks per token.** Categorical `cat` is int16 `[T, 32]`: the
  15 S_ctx slots, then 17 trailing columns. Continuous `cont` is float16
  `[T, 210]`: the 122 S_ctx columns, then 88 trailing columns. New
  information enters only as new sections and trailing columns.
- **One content-ID table.** Every content reference outside the S_ctx content
  slots is an id into one append-only 4,096-row table (1,948 ids used).
- **Duplicates are removed for the student, kept for the teacher.** The combat
  teacher sees every native action token; the student sees one token per
  merged class. This is frozen.
- **PRIV is value-only.** Privileged tokens travel in separate arrays and
  never enter the trunk.

## 2. Tokens (a)

### 2.1 Token order

| Position | Tokens | Rows |
|---|---|---|
| 0 | `RUNCTX` | every row |
| 1-5 | `SUM_DRAW`, `SUM_DISCARD`, `SUM_EXHAUST`, `SUM_DECK`, `SUM_RELICS` | every row |
| 6, 6-7 | `BOSS`: the current act's boss, then the A10 act-3 second boss | every row |
| `sctx_lo` = 7 or 8 | the S_ctx block in #1081's native order: `STATE`, `HAND`, `DRAW_UNK`, `DISCARD`, `EXHAUST`, `DRAW_TOP`, `DRAW_BOT`, `PLAY`, `RELIC`, `POTION`, `PPOWER`, per enemy `ENEMY` then its `EPOWER` and `EHIST`, and last one `ACT_*` token per legal action, in `obs_actions` order | combat |
| after `BOSS` | `DECK` (one per distinct visible deck card), `RELIC`, `POTION` (one per held potion), `OFFER`, `MAPNODE` (map decisions only), then one `OPT_*` token per merged class, in first-member order | OOC |

The trunk has no positional encoding; the order matters for the teacher
slice (section 3.1), for `mem_tok` and `cls_tok`, and for determinism. In an
OOC row the only S_ctx-range sections allowed are `RELIC` and `POTION`, which
keep their combat meaning. Members keep the env's order, which can follow
hidden state (section 5). `validate_row_batch` enforces all of this.

### 2.2 Sections

Ids 0-21 are #1081's, including its critic-only `SEC_PRIV = 21`. New sections
only append. Reserved ids 39-47, 50-55 and 56-63 carry no tokens until a fill
event (section 2.12). The section-embedding table has 64 rows.

| Id | Section | Stream | Live categorical columns | Live continuous columns |
|---|---|---|---|---|
| 0 | STATE | combat (S_ctx) | choice_kind | 18-39, 66-97, 195 |
| 1 | HAND | combat (S_ctx) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD, hand_slot | 0-17, 168 |
| 2 | DRAW_UNK | combat (S_ctx) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD | 0-14, 168 |
| 3 | DRAW_TOP | combat (S_ctx) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD | 0-14, 168 |
| 4 | DRAW_BOT | combat (S_ctx) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD | 0-14, 168 |
| 5 | DISCARD | combat (S_ctx) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD | 0-14, 168 |
| 6 | EXHAUST | combat (S_ctx) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD | 0-14, 168 |
| 7 | PLAY | combat (S_ctx) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD | 0-14, 168 |
| 8 | RELIC | combat (S_ctx) | relic, X_ID_MAIN | 58-61 |
| 9 | POTION | combat (S_ctx) | potion, X_ID_MAIN, X_POS | - |
| 10 | PPOWER | combat (S_ctx) | power, X_ID_MAIN | 55-57 |
| 11 | ENEMY | combat (S_ctx) | monster, enemy_slot, intent, X_ID_MAIN, X_INTENT_T2, X_INTENT_T3, X_FC1, X_FC2, X_FC3 | 40-49, 176-194 |
| 12 | EPOWER | combat (S_ctx) | power, monster, enemy_slot, card, upgraded, card_type, X_ID_MAIN, X_ID_REL | 0-14, 55-57 |
| 13 | EHIST | combat (S_ctx) | intent, monster, enemy_slot, X_ID_REL | 50-54 |
| 14 | ACT_PLAY | combat (S_ctx) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD, hand_slot, enemy_slot, monster, rl2key, X_KIND, X_ID_REL | 0-17, 62-64, 98-121 |
| 15 | ACT_END | combat (S_ctx) | rl2key, X_KIND | 98-121 |
| 16 | ACT_POTION | combat (S_ctx) | potion, enemy_slot, rl2key, X_KIND, X_ID_MAIN, X_ID_REL, X_POS | 98-121 |
| 17 | ACT_CHOOSE | combat (S_ctx) | card, upgraded, card_type, choice_pile, hand_slot, choice_kind, rl2key, X_KIND, X_ID_MAIN, X_ID_MOD | 65, 98-121, 168 |
| 18 | ACT_SKIP | combat (S_ctx) | choice_kind, rl2key, X_KIND | 98-121 |
| 19 | ACT_OTHER | combat (S_ctx) | rl2key, X_KIND | 98-121 |
| 20 | ACT_RL2 | combat (S_ctx) | rl2key | 66-121 |
| 21 | PRIV_DRAW | PRIV (value) | card, upgraded, card_type, enchantment, affliction, X_ID_MAIN, X_ID_MOD | 0-14, 168 |
| 22 | RUNCTX | both/run | X_DTYPE, X_ID_MAIN, X_ID_AUX, X_ROOM | 122-145 |
| 23 | BOSS | both/run | X_ID_MAIN | 174-175 |
| 24 | SUM_DRAW | both/run | - | 170-173 |
| 25 | SUM_DISCARD | both/run | - | 170-173 |
| 26 | SUM_EXHAUST | both/run | - | 170-173 |
| 27 | SUM_DECK | both/run | - | 170-173 |
| 28 | SUM_RELICS | both/run | - | 170-173 |
| 29 | DECK | both/run | upgraded, card_type, X_ID_MAIN, X_ID_MOD | 0-14 |
| 30 | OFFER | both/run | X_KIND, X_ID_MAIN, X_ID_MOD | 158 |
| 31 | MAPNODE | both/run | X_ROOM, X_POS | 154-157 |
| 32 | OPT_CARD | both/run | upgraded, card_type, X_ID_MAIN, X_ID_MOD, X_KIND, X_POS | 0-14, 158, 168-169 |
| 33 | OPT_RELIC | both/run | X_KIND, X_ID_MAIN, X_POS | 158 |
| 34 | OPT_POTION | both/run | X_KIND, X_ID_MAIN, X_POS | 158 |
| 35 | OPT_MOVE | both/run | X_KIND, X_ROOM, X_POS | 146-153 |
| 36 | OPT_EVENT | both/run | X_KIND, X_ID_MAIN, X_ID_MOD, X_POS | 159-167 |
| 37 | OPT_REST | both/run | X_KIND, X_ID_MAIN, X_POS | - |
| 38 | OPT_FIXED | both/run | X_KIND, X_POS | 158 |
| 48 | PRIV_ENEMY | PRIV (value) | enemy_slot, X_ID_MAIN, X_ID_MOD, X_ID_REL, X_ID_AUX | - |
| 49 | PRIV_RUN | PRIV (value) | X_ID_MAIN, X_ID_MOD, X_ID_REL | 196-201 |

Columns are listed as indices (section 2.12 names them). Reserved sections:
39 `RES_SHOP_SHELF` (unaffordable shelf items), 40 `RES_OPT_REWARD` (#1069's
reward screen), 41 `RES_OPT_SPHERE` (Crystal Sphere), 42 `RES_OPT_POTION_USE`
(OOC potion use and discard), 43 `RES_OPT_MERCHANT` (the Fake Merchant),
44-47 spare, 50-55 spare PRIV, 56-63 spare.

### 2.3 S_ctx, bit for bit

The combat block is exactly what `ppo.combat_ppo.encode.Encoder("S_ctx",
rare, priv_k).encode()` produces for the decision (`ppo/combat_ctx/tokens.py`
`tokens(..., "S_ctx")`, then `RareMap.apply` on `cat`, then `f16` on `cont`):

- **Input:** `honest_snapshot(cs.belief_view(0), room)` passed through the
  fight's `IntentLog` (observed at every combat step, single-option ones
  included), the legal actions re-indexed by `obs_actions`, and
  `rollout_features(bv, version=1)`. #1081 computes the rl2 features on the
  true `cs`; the live walk found `rollout_features(bv) == rollout_features(cs)`
  on all 4,871 decisions it checked, so the e2e tokenizer calls it on the
  belief view and X-C confirms equality on recorded states.
- **Sections:** the 21 S_ctx sections in the order in section 2.1. `ACT_RL2`
  (20) belongs to another #1081 arm and never occurs.
- **Categorical slots** (15), with their table sizes on the W1 wheel. A table
  is `[names..., UNK, PAD]`; an absent slot holds PAD (size - 1). The content
  slots hold the combat teacher's RareMap'd rows, which only the teacher reads.

| Slot | Index | Size | Content slot | Student reads |
|---|---|---|---|---|
| card | 0 | 601 | yes | - |
| upgraded | 1 | 4 | - | yes |
| relic | 2 | 299 | yes | - |
| potion | 3 | 66 | yes | - |
| power | 4 | 272 | yes | - |
| monster | 5 | 123 | yes | - |
| enchantment | 6 | 25 | yes | - |
| affliction | 7 | 11 | - | yes |
| intent | 8 | 16 | - | yes |
| enemy_slot | 9 | 7 | - | yes |
| hand_slot | 10 | 14 | - | yes |
| choice_kind | 11 | 33 | - | yes |
| card_type | 12 | 8 | - | yes |
| choice_pile | 13 | 8 | - | yes |
| rl2key | 14 | 1267 | yes | - |

- **Continuous columns** (122). `x/k` divides the raw value by k; a flag is
  0 or 1; absent values are 0; `f16` saturates at ±65,504.

| Column | Name | Transform |
|---|---|---|
| 0 | COST | cost (cost_override, else base_cost) / 3 |
| 1 | BASE_COST | base_cost / 3 |
| 2 | DMG | base_damage / 20 |
| 3 | BLK | base_block / 20 |
| 4 | CALC | calculated_value / 20 |
| 5 | HAS_CALC | flag: calculated_value present |
| 6 | ETHEREAL | flag |
| 7 | UNPLAYABLE | flag |
| 8 | INNATE | flag |
| 9 | SLY | flag |
| 10 | SLY_TURN | flag |
| 11 | RETAIN_TURN | flag |
| 12 | ENCH_AMT | enchantment_amount / 5 |
| 13 | AFF_STACKS | affliction_stacks / 5 |
| 14 | COUNT | pile multiplicity / 5; DRAW_TOP/DRAW_BOT: pin rank (1 = first) / 5 |
| 15 | PLAYABLE | flag: some legal play uses this hand slot |
| 16 | DMG_PREV | max card_damage_preview over targets / 30 |
| 17 | BLK_PREV | card_block_preview / 20 |
| 18 | HP_FRAC | hp / max(1, max_hp) |
| 19 | HP | hp / 100 |
| 20 | MAXHP | max_hp / 100 |
| 21 | BLOCK | block / 30 |
| 22 | ENERGY | energy / 5 |
| 23 | MAXE | max_energy / 5 |
| 24 | STARS | stars / 5 |
| 25 | ROUND | round / 20 |
| 26 | PLAYED | cards played this turn / 10 |
| 27 | N_HAND | hand size / 30 |
| 28 | N_DRAW | draw pile size / 30 |
| 29 | N_DISC | discard pile size / 30 |
| 30 | N_EXH | exhaust pile size / 30 |
| 31 | POT_SLOTS | potion slots / 5 |
| 32 | IS_BOSS | flag: room == boss |
| 33 | IS_ELITE | flag: room == elite |
| 34 | IS_HALL | flag: room == monster |
| 35 | CH_MIN | pending choice min_select / 5 |
| 36 | CH_MAX | pending choice max_select / 5 |
| 37 | CH_SKIP | flag: pending choice can_skip |
| 38 | INCOMING | sum of live enemies' attack intent totals / 50 |
| 39 | UNBLOCKED | max(0, incoming - block) / 50 |
| 40 | E_HPF | hp / max(1, max_hp) |
| 41 | E_HP | hp / 200 |
| 42 | E_MAXHP | max_hp / 200 |
| 43 | E_BLOCK | block / 30 |
| 44 | E_ALIVE | flag |
| 45 | E_REMOVED | flag: removed_after_death |
| 46 | E_AMT | head intent amount / 30 |
| 47 | E_EFF | head intent effective_amount / 30 |
| 48 | E_TIMES | head intent times / 5 |
| 49 | E_TOT | head intent attack total (effective * times) / 50 |
| 50 | H_LAG | turns ago / 3 |
| 51 | H_AMT | amount / 30 |
| 52 | H_EFF | effective_amount / 30 |
| 53 | H_TIMES | times / 5 |
| 54 | H_ALIVE | flag: the enemy is alive now |
| 55 | P_AMT | power amount / 10 |
| 56 | P_DISP | power display_amount / 10 |
| 57 | P_CTR | power counter / 10 |
| 58 | R_CTR | relic display counter / 10 |
| 59 | R_HAS | flag: relic shows a counter |
| 60 | R_WAX | flag: wax relic |
| 61 | R_MELT | flag: melted wax relic |
| 62 | T_HPF | target hp / max(1, max_hp) |
| 63 | T_EHP | target (hp + block) / 100 |
| 64 | T_DMG | card_damage_preview against the target / 30 |
| 65 | N_SEL | len(selection) / 3 |
| 66-97 | RL2_S00-RL2_S31 | `rollout_features(cs, version=1)` state vector |
| 98-121 | RL2_X00-RL2_X23 | `rollout_features(cs, version=1)` per-action vector |

Test `test_sctx_mirror_matches_1081_tokens` pins every section, slot, intent
kind and column index above against `tokens.py`, and
`test_sctx_table_sizes_match_the_engine` pins the table sizes against the
engine.

### 2.4 RUNCTX

One token at position 0 of every row: the state token whose trunk output is
`h_STATE` (section 4). Categorical: `X_DTYPE` (the decision type, equal to
the row's `dtype`), `X_ID_MAIN` (the event id on `EVENT` and `NEOW` decisions,
NONE on every other type), `X_ID_AUX` (the current fight's encounter id on
combat rows, read as `bv.encounter_id`, and NONE out of combat; reserved until
WP-ENG-1), `X_ROOM` (current room type). The validator checks the three ids.
Continuous: the 24
`R_*` columns (section 2.12): act, floors, rows to the boss, bosses left, HP,
gold, deck size and upgrades, relic and potion counts, ascension, the
selection spec (min, max, picks made, can end) for multi-pick decisions in
and out of combat, the merged option count, and the double-boss flag.

The env fills the selection spec as `EnvDecision.sel` (`Selection`,
section 8) on `COMBAT_CHOICE` and `CARD_GRANT` decisions; every other type
carries zeros. `R_SEL_CAN_SKIP` = 1 when an end option (`CHOOSE_DONE`,
`GRANT_CONFIRM`) is offered, so the selection can end now.

- **Combat choices.** `R_SEL_MIN` and `R_SEL_MAX` are `pending_choice()`'s
  `min_select` and `max_select`. The engine splits a `max_select > 1`
  choice into incremental single picks: after each pick it re-opens the same
  choice, fires no effect, and offers one `CHOOSE` per still-unpicked option
  (sts2-core lib.rs:14385-14436). Its `picked` list is not on the wire, so
  `R_SEL_PICKED` = `len(pending_choice()["cards"])` minus the `CHOOSE`
  options offered, and a picked option's card gets `K_PICKED` = 1 on its
  `HAND` token (an option with a `hand_index`; a pick from another pile
  leaves its pile token unmarked, section 2.9). Once `min_select` picks are
  made the engine also offers
  `CHOOSE_DONE`, so `R_SEL_CAN_SKIP` can be 1 where `pending_choice()`'s
  own `can_skip` is false.
- **Card grants.** A grant that requires k of n picks arrives as k decisions
  whose `rm.pending_card_grant_min` (a property) counts the picks still
  required (live: 8 options with min 2, then 7 with min 1); a picked card
  leaves the options and enters the deck. Sea Glass's optional 0-15 grid has
  min 0 and one extra option, `GRANT_CONFIRM` (`grant_deck_card` with
  `idx == len(options)`), which ends the picks; the live walk reached it with
  15 cards and 16 actions. `R_SEL_MIN` = the picks still required,
  `R_SEL_MAX` = the offered cards, `R_SEL_PICKED` = the grant picks the env
  applied since the grant opened.

### 2.5 BOSS

One token per boss of the current act: `full_map()["boss"]`, then
`full_map()["second_boss"]` when present (A10 act 3). `X_ID_MAIN` is the boss encounter id; `B_ORDER` (0, 1) and
`B_ACT` place it.

### 2.6 SUMMARY

Five pooled tokens in every row, in fixed order: the draw, discard and
exhaust piles (empty out of combat), the master deck and the relics. Each
token owns pool entries (`pool_tok`, `pool_vid`, `pool_w`): one per distinct
content id with its count as weight. The net adds the weighted mean of the
entries' ID embeddings (and, once filled, of their content-prior rows) to the
token. Continuous: `U_N` (pool size), `U_UPG`, `U_ENCH`, `U_DISTINCT`.

`TF_POOLED` marks exactly these five tokens, an empty pile's included (it owns
no entries). Pool entries belong only to them (in PRIV, only to `PRIV_RUN`); a
(token, id) pair occurs at most once, ids lie in 1-4,095, and weights are
finite and positive.

### 2.7 Run body (OOC rows)

- **DECK:** one token per distinct visible deck card (`master_deck_detailed()`
  entry plus its `master_deck_enchantments()` entry), with `COUNT` = copies /
  5 and the S_ctx card block where known; `X_ID_MAIN` = card,
  `X_ID_MOD` = enchantment.
- **RELIC, POTION:** S_ctx sections 8 and 9: `X_ID_MAIN` and, for a potion,
  its slot in `X_POS`. Out of combat a RELIC token is id-only: the
  RunManager has no getter for relic counters or wax state (its relic reads
  are `relics`, `relic_reward_ids`, `boss_relic_options`), so `R_CTR`,
  `R_HAS`, `R_WAX` and `R_MELT` stay 0 there (review finding TOK-5); combat
  rows carry them. The S_ctx content slots stay PAD out of combat.
- **OFFER:** content that is offered but is not an option: the potion a
  potion swap would add (`pending_potion_swap`), the enchantment an enchant
  pick stamps (`pending_enchantment`).
- **Options:** one token per merged class (section 5) in the section of its
  payload: `OPT_CARD` (card rewards, bundles, grants and every deck-card pick,
  including the shop's removal service), `OPT_RELIC`, `OPT_POTION` (buy, or
  the potion a swap discards), `OPT_MOVE`, `OPT_EVENT` (Neow included; the
  honest `event_option_infos()` deltas, below, and on Neow the relic the
  label names in `X_ID_MOD`), `OPT_REST` (the `rest_site_options()` id in
  `X_ID_MAIN`), and
  `OPT_FIXED` for content-free options (skip, leave, decline, confirm,
  sacrifice). Every option token carries `X_KIND` and `X_POS`, and sits in
  its kind's section (`KIND_SECTION`).
- **Order:** the body runs `DECK`, `RELIC`, `POTION`, `OFFER`, `MAPNODE`
  (`OOC_BODY_SECTIONS`), then the options in class order.

Rest options come from `rm.rest_site_options()`, never from `ppo/env.py`'s
`REST_OPTION_LABELS = ("rest", "smith")`, which mislabels any third option a
relic adds (Shovel, Girya, Pumpkin Candle, Meat Cleaver, Byrdonis Egg).

**Event previews are filtered.** `rm.event_option_infos()` previews each
option by resolving it on a clone of the run (sts2-content events/dispatch.rs
`preview_option_effect`). Under the e2e env's `mode="agent"`
(`direct_event_relic_grant` on), an option that grants a relic obtains it on
that clone: the relic is rolled from the hidden run RNG, and its pickup effect
lands in the gold, HP, max-HP, potion and card deltas, while a relic that
grants relics changes the relic count. The raw preview therefore carries
hidden RNG into a policy input. The policy reads previews only through
`honest_event_deltas`:

- An option that gains a relic keeps `relics_gained` = 1 (a flag) and its
  losses (`relics_lost`, `potions_lost`, `cards_lost`); its
  `EVENT_RELIC_MASKED` deltas (`gold_delta`, `hp_delta`, `max_hp_delta`,
  `potions_gained`, `cards_gained`) read 0, even where the option's own cost
  is fixed (for example "pay 50 gold for a relic" reads gold 0).
- Every other option keeps its deltas; an option without a preview reads 0.
- `event_delta_columns(info)` gives the `O_EV_*` values (section 2.12).
  `honest_obs(obs)` drops `seed` (`OBS_HIDDEN_FIELDS`) and filters
  `obs["event_options"]`; the run facade passes `rm.event_option_infos` through
  `honest_event_options` (`POLICY_FILTERS`, section 8).

Evidence (section 11): over all 69 events forced on 30 seeds, two pages deep,
the raw preview differs between seeds within 12 of the 510 option groups that
look the same to the player, and the honest preview within none. Keeping the
raw relic count instead of the flag leaves one (Neow's "gain a cursed relic":
2 or 3 relics).

### 2.8 MAP

Map decisions add up to 40 `MAPNODE` tokens: the cone reachable from the
current node, nearest rows first. Each has `X_ROOM` (the visible node type;
`unknown` for a `?` node), `X_POS` (column + 1), and `N_ROW_OFF`, `N_COL`,
`N_NEXT`, `N_PATHS`. Each `OPT_MOVE` option carries the six route summaries
over the paths from its node to the boss (`M_ELITE_MIN`, `M_ELITE_MAX`,
`M_REST_MIN`, `M_REST_MAX`, `M_SHOP_MAX`, `M_UNKNOWN_MAX`) plus
`M_FIGHTS_TO_REST` and `M_ROWS_LEFT`. Only the current act's map is read.

### 2.9 Trailing combat columns

| Group | Columns | Source | State at launch |
|---|---|---|---|
| Telegraph | `X_INTENT_T2`, `X_INTENT_T3`, `E_N_INTENTS`, `E_T2_*`, `E_T3_*` on `ENEMY` (intent 1 stays in S_ctx) | WP-ENG-1 telegraph getter, `current_move().intents` | zero |
| Forecast | `X_FC1`-`X_FC3`, `E_F1_*`-`E_F3_*` on `ENEMY`: step kind, certainty, amount, times, attack total | WP-ENG-1 `peek_intents(3)`, aligned by `performed_first_move` in Rust | zero |
| Tactical | `E_MAXDMG`, `E_KILLABLE` on `ENEMY`; `S_HP_LOSS_END` on `STATE` | Python: knapsack over `card_damage_preview()`, incoming minus block | live |
| Picks | `K_PICKED` on `HAND` tokens (the picked options' cards; a pick from a pile leaves its pile token unmarked, review finding TOK-6); `R_SEL_PICKED` on `RUNCTX` | the offered actions: an option with no `CHOOSE` action is picked (section 2.4). `ACT_CHOOSE` may carry `K_PICKED` but is always 0 on this engine (picked options are not offered) | live |
| Content prior | 48 virtual columns, section 6.3 | frozen buffer, never stored in rows | zero until the extractor lands |
| Spare | `X_RES0`-`X_RES3`, `X_C_RES0`-`X_C_RES7`, `R_RES0` | - | zero |

Intent kinds in trailing columns use `INTENT_IDS`: 0 = none, else 1 + the
index in S_ctx's intent list; `MultiAttack` folds into `Attack`.

### 2.10 PRIV (value only)

At most 16 tokens per row in the `pcat`, `pcont`, `psec` arrays, read only by
the value module (`PRIV_CALLS`, section 8):

- `PRIV_DRAW` (21, up to 10): the true top-10 draw cards, #1081's priv token
  layout (`COUNT` = rank); WP-ENG-1's `cs.true_draw_top(10)` (it raises on a
  belief view) replaces the second `state_to_dict(cs)`.
- `PRIV_ENEMY` (48, up to 5, one per enemy slot): the true next moves, the
  per-step `move_id` of `cs.peek_intents(3)` (move ids in `X_ID_MAIN`,
  `X_ID_MOD`, `X_ID_REL`; the monster in `X_ID_AUX`). The policy reads
  `bv.peek_intents` and never a step's `move_id`.
- `PRIV_RUN` (49, 1): this act's remaining rooms by type (`P_ROOMS_*`, from
  `generated_rooms()[act_index - 1]`), the next act's boss and the act-3
  bosses (`X_ID_*`), and pooled encounter ids (`ppool_*`). No future maps:
  `rm.act_map_for` is a PRIV call, but no PRIV token reads it at launch
  (section 12).

The validator enforces the per-section caps (10, 5, 1). PRIV permutations
must leave S0 and S1b policy outputs bit-identical (T14, F7; section 8.3).

### 2.11 Token counts

| Part | Tokens | Cap |
|---|---|---|
| Head (`RUNCTX`, `SUMMARY`, `BOSS`) | 7-8 | 8 |
| Combat row (head + S_ctx), live walk | mean 54, max 148 (4,871 rows) | 256 |
| OOC row without `MAPNODE`, live walk | mean 36, max 85 (461 rows) | 256 |
| `MAPNODE` | ≤40 | 40 |
| Merged options, live walk | max 39 OOC (64 raw), 25 combat (68 raw) | 64 |
| PRIV | ≤16 | 16 |

The walk's random policy bloats decks, so its OOC sizes are high; the plan's
estimate for the trained policy is a mean of about 57 tokens [CM].

Buckets: 32, 48, 64, 96, 128, 192, 256 tokens. WP3b measures the real
distribution and may only move bucket edges, not the cap.

### 2.12 Columns

Trailing categorical columns (absent value 0):

| Column | Name | Table size | Meaning | Status |
|---|---|---|---|---|
| 15 | X_DTYPE | 64 | decision-type id (RUNCTX only) | live |
| 16 | X_KIND | 128 | action-kind id of the option (every option token) | live |
| 17 | X_ID_MAIN | 4096 | primary content id (see SECTION_IDS) | live |
| 18 | X_ID_MOD | 4096 | modifier content id, e.g. a card's enchantment | live |
| 19 | X_ID_REL | 4096 | related content id, e.g. a play's target monster | live |
| 20 | X_ID_AUX | 4096 | auxiliary id; RUNCTX: the current fight's encounter on combat rows, NONE out of combat | reserved:WP-ENG-1 |
| 21 | X_ROOM | 16 | room-type id (ROOM_TYPES) | live |
| 22 | X_POS | 64 | 1 + position (slot, column, option index), capped at 63 | live |
| 23 | X_INTENT_T2 | 16 | ENEMY: telegraph intent 2 kind (INTENT_IDS) | reserved:WP-ENG-1 |
| 24 | X_INTENT_T3 | 16 | ENEMY: telegraph intent 3 kind (INTENT_IDS) | reserved:WP-ENG-1 |
| 25 | X_FC1 | 16 | ENEMY: forecast step 1 head-intent kind | reserved:WP-ENG-1 |
| 26 | X_FC2 | 16 | ENEMY: forecast step 2 head-intent kind | reserved:WP-ENG-1 |
| 27 | X_FC3 | 16 | ENEMY: forecast step 3 head-intent kind | reserved:WP-ENG-1 |
| 28 | X_RES0 | 16 | spare | reserved:spare |
| 29 | X_RES1 | 16 | spare | reserved:spare |
| 30 | X_RES2 | 16 | spare | reserved:spare |
| 31 | X_RES3 | 16 | spare | reserved:spare |

Trailing continuous columns (absent value 0):

| Column | Name | Transform | Status |
|---|---|---|---|
| 122 | R_ACT | act_index (1-3) / 3 | live |
| 123 | R_FLOOR | total_floor / 50 | live |
| 124 | R_ACT_FLOOR | floor within the act / 17 | live |
| 125 | R_ROWS_TO_BOSS | (boss row - current row) / 17 | live |
| 126 | R_BOSSES_LEFT | bosses left in the run, this act's included / 4 | live |
| 127 | R_HP | hp / 100 | live |
| 128 | R_MAXHP | max_hp / 100 | live |
| 129 | R_HP_FRAC | hp / max(1, max_hp) | live |
| 130 | R_GOLD | gold / 300 | live |
| 131 | R_DECK | deck size / 40 | live |
| 132 | R_DECK_UPG | upgraded deck cards / 20 | live |
| 133 | R_RELICS | relics held / 20 | live |
| 134 | R_POTIONS | potions held / 5 | live |
| 135 | R_POT_SLOTS | potion slots / 5 | live |
| 136 | R_ASC | ascension / 10 | live |
| 137 | R_IN_COMBAT | flag | live |
| 138 | R_SEL_MIN | selection min picks / 5 (OOC multi-pick, combat choice) | live |
| 139 | R_SEL_MAX | selection max picks / 5 | live |
| 140 | R_SEL_PICKED | picks already made in the open selection / 5 | live |
| 141 | R_SEL_CAN_SKIP | flag: the selection can end now | live |
| 142 | R_N_CLASSES | merged options / 64 | live |
| 143 | R_DOUBLE_BOSS | flag: this act ends in two bosses (A10 act 3) | live |
| 144 | R_HIST_MISSING | flag: intent history unavailable (live gap) | reserved:D9 |
| 145 | R_RES0 | spare | reserved:spare |
| 146 | M_ELITE_MIN | fewest elites on a path from the node to the boss / 5 | live |
| 147 | M_ELITE_MAX | most elites on such a path / 5 | live |
| 148 | M_REST_MIN | fewest rest sites / 5 | live |
| 149 | M_REST_MAX | most rest sites / 5 | live |
| 150 | M_SHOP_MAX | most shops / 5 | live |
| 151 | M_UNKNOWN_MAX | most unknown (?) nodes / 5 | live |
| 152 | M_FIGHTS_TO_REST | fewest monster + elite nodes before the first rest / 5 | live |
| 153 | M_ROWS_LEFT | rows from the node to the boss row / 17 | live |
| 154 | N_ROW_OFF | (node row - current row) / 17 | live |
| 155 | N_COL | column / 6 | live |
| 156 | N_NEXT | flag: a legal move target now | live |
| 157 | N_PATHS | log1p(paths from the current node through this node) / 5 | live |
| 158 | O_PRICE | gold price / 300 (shop items, removal service) | live |
| 159 | O_EV_GOLD | honest event gold delta / 100 (0 if the option gains a relic) | live |
| 160 | O_EV_HP | honest event hp delta / 20 (0 if the option gains a relic) | live |
| 161 | O_EV_MAXHP | honest event max_hp delta / 10 (0 if the option gains a relic) | live |
| 162 | O_EV_RELICS_GAIN | flag: the option gains at least one relic | live |
| 163 | O_EV_POTIONS_GAIN | honest potions gained / 3 (0 if the option gains a relic) | live |
| 164 | O_EV_CARDS_GAIN | honest cards gained / 3 (0 if the option gains a relic) | live |
| 165 | O_EV_CARDS_LOSE | cards lost / 3 | live |
| 166 | O_EV_RELICS_LOSE | relics lost / 3 | live |
| 167 | O_EV_POTIONS_LOSE | potions lost / 3 | live |
| 168 | K_PICKED | flag: already selected in the open multi-pick selection | live |
| 169 | K_DECK_COPIES | copies of this card already in the deck / 5 | live |
| 170 | U_N | pool size / 40 | live |
| 171 | U_UPG | upgraded members / 20 | live |
| 172 | U_ENCH | enchanted members / 10 | live |
| 173 | U_DISTINCT | distinct members / 40 | live |
| 174 | B_ORDER | 0 first boss, 1 second boss | live |
| 175 | B_ACT | act / 3 | live |
| 176 | E_N_INTENTS | intents in the current telegraph / 3 | reserved:WP-ENG-1 |
| 177 | E_T2_EFF | telegraph intent 2 effective_amount / 30 | reserved:WP-ENG-1 |
| 178 | E_T2_TIMES | telegraph intent 2 times / 5 | reserved:WP-ENG-1 |
| 179 | E_T3_EFF | telegraph intent 3 effective_amount / 30 | reserved:WP-ENG-1 |
| 180 | E_T3_TIMES | telegraph intent 3 times / 5 | reserved:WP-ENG-1 |
| 181 | E_F1_CERT | forecast step 1: certainty (highest-weight branch share, product over steps) | reserved:WP-ENG-1 |
| 182 | E_F1_AMT | forecast step 1: head-intent amount / 30 | reserved:WP-ENG-1 |
| 183 | E_F1_TIMES | forecast step 1: head-intent times / 5 | reserved:WP-ENG-1 |
| 184 | E_F1_TOT | forecast step 1: attack total (amount * times) / 50 | reserved:WP-ENG-1 |
| 185 | E_F2_CERT | forecast step 2: certainty (highest-weight branch share, product over steps) | reserved:WP-ENG-1 |
| 186 | E_F2_AMT | forecast step 2: head-intent amount / 30 | reserved:WP-ENG-1 |
| 187 | E_F2_TIMES | forecast step 2: head-intent times / 5 | reserved:WP-ENG-1 |
| 188 | E_F2_TOT | forecast step 2: attack total (amount * times) / 50 | reserved:WP-ENG-1 |
| 189 | E_F3_CERT | forecast step 3: certainty (highest-weight branch share, product over steps) | reserved:WP-ENG-1 |
| 190 | E_F3_AMT | forecast step 3: head-intent amount / 30 | reserved:WP-ENG-1 |
| 191 | E_F3_TIMES | forecast step 3: head-intent times / 5 | reserved:WP-ENG-1 |
| 192 | E_F3_TOT | forecast step 3: attack total (amount * times) / 50 | reserved:WP-ENG-1 |
| 193 | E_MAXDMG | most damage dealable to this enemy this turn (knapsack) / 50 | live |
| 194 | E_KILLABLE | flag: E_MAXDMG >= hp + block | live |
| 195 | S_HP_LOSS_END | hp lost if the turn ended now / 50 | live |
| 196 | P_ROOMS_MONSTER | this act's remaining monster rooms / 10 | live |
| 197 | P_ROOMS_ELITE | this act's remaining elite rooms / 10 | live |
| 198 | P_ROOMS_EVENT | this act's remaining event rooms / 10 | live |
| 199 | P_ROOMS_SHOP | this act's remaining shop rooms / 10 | live |
| 200 | P_ROOMS_REST | this act's remaining rest rooms / 10 | live |
| 201 | P_ROOMS_TREASURE | this act's remaining treasure rooms / 10 | live |
| 202 | X_C_RES0 | spare | reserved:spare |
| 203 | X_C_RES1 | spare | reserved:spare |
| 204 | X_C_RES2 | spare | reserved:spare |
| 205 | X_C_RES3 | spare | reserved:spare |
| 206 | X_C_RES4 | spare | reserved:spare |
| 207 | X_C_RES5 | spare | reserved:spare |
| 208 | X_C_RES6 | spare | reserved:spare |
| 209 | X_C_RES7 | spare | reserved:spare |

**What the WP-ENG-1 intent columns compute.** The table repeats the
contract's column strings, which `schema_sha256` hashes, so they stay as
written; the tokenizer (`tokens._intent_cols`) and its live getter test
agree on this: `E_T2_EFF` and `E_T3_EFF` fall back to `amount` when an
intent has no `effective_amount` (non-attack intents), and `E_F*_TOT` is
`effective_amount x times` for Attack and DeathBlow intents, so it includes
Strength, Vulnerable and Weak (the wire maps MultiAttack to Attack, so no
attack is missed). Review finding TOK-8, integrator A's note 6.

**Live columns per section.** `SECTION_LIVE` lists, for every section, the
columns its tokens may fill (table in section 2.2). A dense row is lossless
under the compact layout if every other column holds the absent value (S_ctx
slots: PAD; trailing categorical: 0; continuous: 0); `check_compact_lossless`
asserts it.

**Zero-filled until later.** Columns with status `reserved:<event>` hold their
absent value until that fill event; the stamp's `filled` list records which
events happened (section 11). The network zero-initializes the first-layer
weights of every reserved continuous column and the ID-table rows of every
registry block no live column references at launch (encounters, moves), so a
fill at a restart preserves the function (the identity rule). Fill events:
`WP-ENG-1` (telegraph, `peek_intents(3)`, encounter id, true top-k draw),
`content-prior`, `D9`, `#1069`, and `chars` (the other characters' mechanics,
section 13; it takes spare slots, so the tables above still list them as spare).

A launch whose rows already carry a fill event declares it (`--filled`; GH
#1097 A4: `["WP-ENG-1"]` on the W2 wheel). Its fresh net inits that event's
inputs like live ones (`E2ENet(filled=...)`): normal init for the stem weights,
the embedding rows and the encounter and move ID rows, and the honest
`X_ID_AUX` gain at 1. The identity rule covers only events that arrive after a
checkpoint exists. `ppo.e2e.fillcheck` checks a declaration against rows, and
the pre-flight runs it: with `WP-ENG-1` declared, RUNCTX's `X_ID_AUX` is
nonzero on at least 95% of combat rows, and the telegraph columns, the
forecast columns and `PRIV_ENEMY` on at least 50% of live enemies; without
it, every `WP-ENG-1` column is zero and no row has `PRIV_ENEMY`. Either way,
dead enemies carry no `WP-ENG-1` value and every combat row with a draw pile
has `PRIV_DRAW`. Another undeclared event leaves no trace: zero reserved
columns, none of its reserved decision types or action kinds.

## 3. Teacher views (b)

### 3.1 Combat teacher (X-C)

The combat teacher reads a combat row through a section mask and a column
slice: the tokens with `sec < 21` (exactly the contiguous block
`[sctx_lo, sctx_lo + n_sctx)`, flagged `TF_TEACHER`), `cat[:, :15]`,
`cont[:, :122]`, `sec`, and `n_act = n_mem`. Its policy logits are the
`n_mem` action tokens at the end of the block, in member order (`mem_tok`).
The S_ctx content slots are stored RareMap'd exactly as
`Encoder(ck_teacher).encode` stores them, so the slice is byte-equal to
#1081's own `Decision` arrays.

**X-C:** on recorded states, the slice must be byte-equal to
`Encoder.encode()`'s `cat`, `cont` and `sec`, and the teacher's logits through
the slice must be `torch.equal` to its logits on #1081's own pipeline, in the
same batch layout. Class probabilities then follow by log-sum-exp over each
class's members (`merge_probs`).

### 3.2 Run teacher (T8)

The run teacher (the deployed run PPO) scores `ppo.encoder.featurize_row(obs,
actions, metrics, features)`, one candidate per env action, in the order of
`Decision.actions` after `mask_rejectable`. The map from its candidates to
the env actions is the identity (a bijection; WP3b asserts equal length and
kinds). Each env action is one member; members map onto merged classes, and a
class's teacher probability is the sum of its members' probabilities. T8
asserts that these sum to 1 for every row with `t_src = 2`.

The run teacher reads the raw `obs`, raw event previews included
(`featurize_row` reads `gold_delta`, `hp_delta`, `max_hp_delta` and
`relics_gained`), because that is the input it was trained and measured on.
Only the student is held to the honest rule (section 12).

## 4. Heads: the state token and option tokens (c)

- `h_STATE` is the trunk output at position 0 (`RUNCTX`) in both streams. In
  a combat row, the S_ctx `STATE` token is an ordinary token for the student.
- Option tokens are the class representatives (`cls_tok`, flagged `TF_REP`).
  Each head scores `[h_action; h_STATE; e_kind]`, where `h_action` is the
  trunk output at the representative and `e_kind` is the embedding of the
  option's action kind (`X_KIND`, 128 rows). The softmax runs over the
  row's `n_classes` options.
- The decision type (`dtype`) selects the head: `COMBAT_TURN` and
  `COMBAT_CHOICE` use the combat head, every other type the OOC head.
- **Dedupe, frozen:** duplicates are removed from the student's view. In a
  combat row, non-representative members stay in the S_ctx block for the
  teacher but carry `TF_HIDDEN`; the student key-masks them in attention and
  never scores them. An OOC row materializes one token per class.

## 5. Dedupe rule (e)

Members of a class share a key; classes are numbered by first occurrence;
the representative is the first member. Keys come from the belief view and
the run's visible state only.

| Option | Key |
|---|---|
| Play | `("play", card_key(card), target_index)`, where `card_key = _canon(_card(card))`: id, upgraded, cost, base cost, base damage and block, calculated value, enchantment and amount, affliction and stacks, flags |
| Potion | `("potion", potion id, target_index)` |
| Choose | `("choose", choose_outcome_key(bv after the action), sorted selected-option content)`; content is `card_key` of the selected card in its pile, else its (id, upgraded, type, pile) |
| End turn, other combat kinds | never merge |
| Deck-card picks (remove, upgrade, transform, enchant, smith, shop removal service) | `(kind, deck entry, enchantment)` |
| Offered cards (card reward, grant) | `(kind, id, upgraded, enchantment)` |
| Shop items | `(kind, inventory item without its index)`, so equal items at different prices stay apart |
| Potion-swap discard | `(kind, potion id in that slot)` |
| Everything else (moves, events, Neow, rest, boss relics, bundles, skip, leave, decline, confirm) | position: never merges |

Rules:

- **T6:** an enchanted Strike and a plain Strike stay apart; a played
  Rampage (higher `base_damage`) and a fresh one stay apart; plain Strikes
  merge. `play_edge_key` would merge both pairs (snapshot.py:487-503), so the
  dedupe never uses it.
- **Multi-pick choices:** an incremental pick of a `max_select > 1` choice
  (for example `discard_from_hand_and_draw`, min 0, max 3-6) re-opens the
  same choice, fires no effect, and the engine's `picked` list is not on the
  wire (section 2.4). The honest snapshot is therefore unchanged whichever
  card the pick takes, so `choose_outcome_key` alone merges different cards.
  The live walk hit this on its first such choice; the selected-option
  content in the key fixes it. #1081's own `action_classes` has the same
  merge.
- **One class, no row.** A decision whose members all merge into one class
  is auto-advanced like a single-action step: the env applies the
  representative, and no forward pass and no row happen (the walk: 2 combat
  and 1 OOC decisions).
- **Members are token-identical.** Members of one combat class have
  byte-identical S_ctx tokens except the `hand_slot` column (asserted; the
  live walk checked 2,532 merged members in 4,871 decisions). The converse
  need not hold: S_ctx's `ACT_CHOOSE` token has no enchantment, so two
  classes may share S_ctx tokens; the student tells them apart through
  `X_ID_MOD`.
- **Teacher probabilities merge by log-sum-exp** over member logits (combat)
  or by summing member probabilities (run).
- **Member order is the env's.** Members, and so classes and `ACT_*` tokens,
  keep the env's action order. In a combat choice from the draw pile,
  `cs.legal_actions()` lists the options in the true draw order, and
  `obs_actions` re-indexes each selection into the belief view's option list
  without reordering the actions. The order can therefore follow hidden state.
  Token contents come from the belief view and the trunk has no positional
  encoding, so no class's logit depends on that order; only the class
  numbering does. F7 accounts for this (section 8.3).

## 6. Ids (d, f)

### 6.1 Action kinds

uint8 on the wire, `X_KIND` in rows, 128-row table. Ids never change. Spare
ids (not listed): 9-15 combat, 30 and 42-47 OOC, 58-63 further gaps, 64-95
the D9 block; 96-127 are never assigned.

| Id | Kind | Status | Meaning |
|---|---|---|---|
| 0 | NONE | live | not an option token |
| 1 | PLAY | live | play {card_index}, no target |
| 2 | PLAY_TARGET | live | play {card_index, target_index} |
| 3 | END_TURN | live | end_turn |
| 4 | POTION | live | drink_potion / potion {slot}, no target |
| 5 | POTION_TARGET | live | drink_potion / potion {slot, target_index} |
| 6 | CHOOSE | live | choose {index, selection: [i]} |
| 7 | CHOOSE_DONE | live | choose {selection: []}: finish or skip the choice |
| 8 | COMBAT_OTHER | reserved:non-Ironclad | channel / evoke / any other combat kind |
| 16 | MOVE | live | move {col} |
| 17 | PICK_CARD_REWARD | live | pick_card_reward {idx} |
| 18 | SKIP_CARD_REWARD | live | pick_card_reward {idx: None} |
| 19 | SACRIFICE_CARD_REWARD | live | sacrifice_card_reward (Pael's Wing) |
| 20 | PICK_CARD_BUNDLE | live | pick_card_reward {idx} with card_reward_bundles() |
| 21 | PICK_RELIC_REWARD | live | pick_relic_reward: the head of the relic queue |
| 22 | SKIP_RELIC_REWARD | live | skip_relic |
| 23 | PICK_BOSS_RELIC | live | pick_boss_relic {idx} |
| 24 | SHOP_BUY_CARD | live | shop_buy_card {idx} |
| 25 | SHOP_BUY_RELIC | live | shop_buy_relic {idx} |
| 26 | SHOP_BUY_POTION | live | shop_buy_potion {idx} |
| 27 | SHOP_REMOVE_CARD | live | shop_remove_card {deck_idx}: the removal service |
| 28 | LEAVE | live | leave_shop (also leaves a rest site after Miniature Tent) |
| 29 | REST_OPTION | live | rest_option {idx}; the option id rides in X_ID_MAIN |
| 31 | SMITH_CARD | live | smith_card {idx} |
| 32 | EVENT_CHOICE | live | event_choice {idx} (Neow included) |
| 33 | COLLECT_TREASURE | live | collect_treasure |
| 34 | POTION_SWAP_DISCARD | live | potion_swap {discard_slot: i} |
| 35 | POTION_SWAP_DECLINE | live | potion_swap {discard_slot: None} |
| 36 | REMOVE_DECK_CARD | live | remove_deck_card {idx} |
| 37 | UPGRADE_DECK_CARD | live | upgrade_deck_card {idx} |
| 38 | TRANSFORM_DECK_CARD | live | transform_deck_card {idx} |
| 39 | ENCHANT_DECK_CARD | live | enchant_deck_card {idx} |
| 40 | GRANT_DECK_CARD | live | grant_deck_card {idx < len(card_grant_options())} |
| 41 | GRANT_CONFIRM | live | grant_deck_card {idx == len(options)}: end Sea Glass |
| 48 | TAKE_REWARD_POTION | reserved:#1069 | reward screen: take a potion |
| 49 | SKIP_REWARD_POTION | reserved:#1069 | reward screen: leave a potion |
| 50 | TAKE_REWARD_GOLD | reserved:#1069 | reward screen: take gold |
| 51 | SKIP_TREASURE | reserved:treasure | treasure room: leave the chest |
| 52 | SPHERE_TOOL | reserved:crystal-sphere | Crystal Sphere: pick a tool |
| 53 | SPHERE_CELL | reserved:crystal-sphere | Crystal Sphere: reveal a cell |
| 54 | USE_POTION_OOC | reserved:ooc-potion | drink a potion out of combat |
| 55 | DISCARD_POTION | reserved:ooc-potion | discard a potion out of combat |
| 56 | MERCHANT_BUY | reserved:fake-merchant | Fake Merchant: buy one item |
| 57 | MERCHANT_LEAVE | reserved:fake-merchant | Fake Merchant: leave |

### 6.2 Decision types

uint8 `dtype` in rows and `X_DTYPE` on `RUNCTX`, 64-row table. Spare ids:
3-7 combat, 25-39 OOC, 45-47 gaps, 48-63 D9. A decision's type comes from its
action set (`classify_ooc`, `classify_combat`), never from `phase()`; Neow is
an event decision with `current_event_name() == "Neow"`; a card reward is a
bundle when `card_reward_bundles()` is non-empty. A kind outside its type's
set raises `SchemaError`, so a new surface fails loudly instead of training
on noise (the #317 failure).

| Id | Decision type | Status | Head | Action kinds |
|---|---|---|---|---|
| 1 | COMBAT_TURN | live | combat | PLAY, PLAY_TARGET, END_TURN, POTION, POTION_TARGET |
| 2 | COMBAT_CHOICE | live | combat | CHOOSE, CHOOSE_DONE |
| 8 | NEOW | live | OOC | EVENT_CHOICE |
| 9 | MAP | live | OOC | MOVE |
| 10 | CARD_REWARD | live | OOC | PICK_CARD_REWARD, SKIP_CARD_REWARD, SACRIFICE_CARD_REWARD |
| 11 | CARD_BUNDLE | live | OOC | PICK_CARD_BUNDLE |
| 12 | RELIC_REWARD | live | OOC | PICK_RELIC_REWARD, SKIP_RELIC_REWARD |
| 13 | BOSS_RELIC | live | OOC | PICK_BOSS_RELIC |
| 14 | SHOP | live | OOC | SHOP_BUY_CARD, SHOP_BUY_RELIC, SHOP_BUY_POTION, SHOP_REMOVE_CARD, LEAVE |
| 15 | REST | live | OOC | LEAVE, REST_OPTION |
| 16 | SMITH | live | OOC | SMITH_CARD |
| 17 | EVENT | live | OOC | EVENT_CHOICE |
| 18 | POTION_SWAP | live | OOC | POTION_SWAP_DISCARD, POTION_SWAP_DECLINE |
| 19 | CARD_REMOVE | live | OOC | REMOVE_DECK_CARD |
| 20 | CARD_UPGRADE | live | OOC | UPGRADE_DECK_CARD |
| 21 | CARD_TRANSFORM | live | OOC | TRANSFORM_DECK_CARD |
| 22 | CARD_ENCHANT | live | OOC | ENCHANT_DECK_CARD |
| 23 | CARD_GRANT | live | OOC | GRANT_DECK_CARD, GRANT_CONFIRM |
| 24 | TREASURE | live | OOC | COLLECT_TREASURE |
| 40 | REWARD_SCREEN | reserved:#1069 | OOC | TAKE_REWARD_POTION, SKIP_REWARD_POTION, TAKE_REWARD_GOLD |
| 41 | TREASURE_CHOICE | reserved:treasure | OOC | COLLECT_TREASURE, SKIP_TREASURE |
| 42 | CRYSTAL_SPHERE | reserved:crystal-sphere | OOC | SPHERE_TOOL, SPHERE_CELL |
| 43 | POTION_MENU | reserved:ooc-potion | OOC | USE_POTION_OOC, DISCARD_POTION |
| 44 | FAKE_MERCHANT | reserved:fake-merchant | OOC | MERCHANT_BUY, MERCHANT_LEAVE |

Two live types never reached a stored row in the walk (21 runs so far):

- `RELIC_REWARD`: `PpoRunEnv` builds its `RunManager` with `mode="agent"`,
  and `RunState::set_agent_decisions` (runstate.rs:2255-2260) turns on
  `direct_event_relic_grant`, which mirrors the C# TestMode oracle's
  auto-collect (`RewardsSet.Offer`). The engine then obtains every queued
  relic reward at once: elite relics always (run_manager.rs:2609-2628), and
  event, shop, Neow and chained relics wherever the flag is checked. No path
  we found leaves a `pick_relic_reward`/`skip_relic` pair for the player. The
  real game's skip is #1069's reward screen.
- `TREASURE`: a single `collect_treasure` auto-advances.

The plan's pre-flight (§3.2) requires a relic reward to occur; under these
flags it cannot (open question 10).

### 6.3 Content-ID table

- **Layout:** one table of 4,096 rows shared by the trunk, the value module
  and the content prior. Id 0 is NONE (zero embedding, never trained); ids
  1-11 are `UNK_<registry>`, the missing row of each registry; 12-31 are
  spare; the registry blocks start at 32.
- **Registries**, in block order, from the W1 engine: card (599,
  `list_registry_names("card")`), relic (299, `relic_id`), potion (64), power
  (280, `power_all`), monster (121, `monster_species()`), encounter (81,
  `list_encounters()`; bosses included), event (69, Neow included),
  enchantment (23), orb (6), rest option (9, `RestSiteOption::option_id`,
  runstate.rs:1030), move (356 species-move pairs, `monster_moves()`, for
  PRIV). 1,919 rows, ids 0-1938. Revision 2.1 appends 9 more encounters at
  ids 1939-1947, after the move block: the event fights' encounter classes
  of the v0.111.0 decomp (`vocab.EVENT_ENCOUNTERS`), which
  `list_encounters()` omits but `CombatState.encounter_id` reports for an
  event's fight. The engine reports `BATTLEWORN_DUMMY_EVENT_ENCOUNTER`, the
  abstract base, for all three settings, and has no PunchOff or TheArchitect
  fight; the game's V1-V3 settings and those two have rows too. So the
  encounter block is 90 ids and not contiguous. 1,928 rows, ids 0-1947.
- **Assignment:** append-only. `python -m ppo.e2e.vocab --append` adds a new
  wheel's names after the last id, whatever their registry; no row is ever
  renumbered or removed. `--check` exits 1 when the engine has names the
  table lacks; `test_vocab_is_append_only_complete_for_the_engine` runs it.
- **Lookup:** per-registry canonicalization (`CANON`), so `StrikeIronclad`,
  `CARD.STRIKE_IRONCLAD` and `STRIKE_IRONCLAD` share an id, and
  `Tezcatara's Ember` finds `TEZCATARAS_EMBER`. An unknown name gets its
  registry's UNK id; canonical names are unique per registry (asserted).
- **sha256** over the canonical serialization (`id\tregistry\tname\n` per
  row, UTF-8, LF), never the file bytes.
- **Content prior:** 48 columns per id (`PRIOR_COLUMNS`): card type (6),
  rarity (6), cost, X-cost, cost upgrade delta, damage, hits, block and their
  upgrade deltas, magic number and delta, 7 keywords, 4 target classes, 3
  pools, draw, energy, debuff, strength, exhaust-other, monster max HP, elite,
  boss, act, power-is-debuff, 1 spare, and `MISSING` (column 47: the
  extractor has no row for this id). The net gathers `PRIOR[X_ID_MAIN]` (the
  weighted pool mean for summary tokens) from a frozen float32 `[4096, 48]`
  buffer with its own sha256 (`prior_sha256`); the launch buffer is all zero
  and inert until the "content-prior" fill event (the learner's `--filled`),
  which loads `content_prior.table()` and stamps its sha256. EMA shadows copy
  the buffer, never average it, and a load checks it against `prior_sha256`.

## 7. Caps (g)

| Cap | Value | On breach |
|---|---|---|
| Tokens per row | 256 (buckets 32-256) | assert |
| Merged options per decision | 64 | assert |
| Raw legal actions per decision | 255 | assert |
| PRIV tokens per row | 16 | assert |
| `MAPNODE` tokens | 40 | nearest rows kept |
| Fight | 100 turns or 1,000 decisions | capped = loss |
| Run | 6,000 decisions | capped = loss |
| Capped rate | 0.1% | alarm |

The engine itself has no turn cap: a uniform random policy in the walk ran a
fight without end, so the e2e env must enforce the fight cap. `PpoRunEnv`'s
own `max_steps` (default 3,000) counts every env step, combat actions and
auto-advanced steps included, and would end a long run before the 6,000
decision cap; the e2e env sets it explicitly (for example 20,000) so the
schema's caps are the binding ones.

## 8. Env to tokenizer interface (h)

The policy tokenizer reads only what the player sees. In combat it reads the
belief view `bv = cs.belief_view(0)`, and the true state `cs` only for the
actions the player is offered. Out of combat it reads `honest_obs(d.obs)` and
the run facade (`PolicySource.run()`), which serves exactly the `rm.*` names
below. The WP1 AST guard checks call sites against both lists.

| List | Calls |
|---|---|
| `POLICY_CALLS`, combat: offered actions | `cs.belief_view`, `cs.legal_actions`, `cs.pending_choice` |
| `POLICY_CALLS`, combat: belief view | `bv.apply`, `bv.card_block_preview`, `bv.card_damage_preview`, `bv.clone` (`choose_outcome_key` applies a choose on a clone), `bv.pending_choice`, `honest_snapshot(bv)`, `rollout_features(bv)`, `state_to_dict(bv)`; WP-ENG-1: `bv.telegraph`, `bv.peek_intents` (never a step's `move_id`), `bv.encounter_id` (combat rows only) |
| `POLICY_CALLS`, out of combat | `honest_obs(obs)`, `rm.act_floor`, `rm.act_index`, `rm.ascension_level`, `rm.available_actions`, `rm.card_grant_options`, `rm.card_reward_bundles`, `rm.current_event_name`, `rm.current_hp`, `rm.current_room`, `rm.event_option_infos` (filtered), `rm.full_map` (the current act only), `rm.gold`, `rm.map_coord`, `rm.master_deck_detailed`, `rm.master_deck_enchantments`, `rm.materialize_event`, `rm.max_hp`, `rm.max_potion_slots`, `rm.pending_card_grant_min`, `rm.pending_enchantment`, `rm.pending_potion_swap`, `rm.potions`, `rm.relics`, `rm.rest_site_options`, `rm.shop_inventory`, `rm.total_floor` |
| `POLICY_FILTERS` | `rm.event_option_infos` returns `honest_event_options(...)` (section 2.7) |
| `PRIV_CALLS` (value path only) | `state_to_dict(cs)`, `rollout_features(cs)`, `cs.get_state`, `cs.sample_world`, `cs.true_draw_top`, `cs.peek_intents:move_id` (the steps' move ids), `rm.generated_rooms`, `rm.act_map_for` (not read at launch) |

Revision 2 removed policy reads of the true state that the tokenizer does
not need or must not make: `cs.card_damage_preview` and `cs.card_block_preview`
(the tokenizer reads them on `bv`), `cs.observation_hash`, `cs.telegraph`,
`cs.peek_intents` (all on `bv`) and `rm.encounter_id` (`bv.encounter_id`
carries the fight's encounter).

### 8.1 The env decision

For each stored decision the env yields one `EnvDecision` (contracts.py
section 11b). It lives in one process and is never stored, so it is not in
`schema_sha256`; its fields only append. `make_decision(...)` fills
`member_class` and `rep` from `merge_classes(keys)`, and
`validate_decision(d)` checks it (below).

| Field | Type | Meaning |
|---|---|---|
| `run_uid` | int | the row's `run_uid` |
| `t` | int | the row's `t`: order of this stored decision within its run or segment |
| `dtype` | int | decision type (`classify_combat`, `classify_ooc`) |
| `actions` | list of dict | the members: legal actions in env order. Out of combat: `PpoRunEnv`'s `Decision.actions`, after `mask_rejectable`. In combat: `cs.legal_actions()` re-indexed to the belief view by `snapshot.obs_actions` |
| `kinds` | uint8 `[M]` | action kind per member |
| `keys` | list of str `[M]` | dedupe key per member (`combat_class_keys`, `ooc_class_keys`) |
| `member_class` | uint16 `[M]` | class of each member, numbered by first occurrence |
| `rep` | uint16 `[C]` | representative (first) member of each class |
| `policy` | `PolicySource` | everything the policy tokenizer may read |
| `priv` | `PrivSource` or None | value path only; None when the value path is off |
| `sel` | `Selection` | the open selection on `COMBAT_CHOICE` and `CARD_GRANT` (section 2.4); zeros elsewhere |
| `event_name` | str or None | `rm.current_event_name()` on `EVENT` and `NEOW` (`"Neow"`), else None |
| `obs`, `metrics` | dict or None | `PpoRunEnv.observe()` and `metrics_snapshot()`: required out of combat (the run teacher's `featurize_row` input and `ooc_payloads`' input), None in combat |
| `cell` | int | practice cell, 255 outside practice |
| `fight_turn` | int | combat: the fight's round (`snapshot["round"]`, 1-based); 0 out of combat |
| `fight_decision` | int | combat: stored decisions before this one in the fight; 0 out of combat |

Derived: `is_combat`, `head`, `n_members`, `n_classes`, `class_kinds()` (the
row's `cls_kind`) and `member_action(c)`, the env action that takes class
`c`.

`validate_decision(d)` rejects: a type that is not live (reserved types pass
with `allow_reserved=True`; spare types never do), a kind count that is not
one uint8 per member, a kind the type cannot offer, a member whose raw action
name does not fit its kind, a class map other than `merge_classes(keys)`, a
class that mixes kinds, fewer than 2 or more than 64 classes, more than 255
members, a selection on a type without one, `min > max`, `can_skip` that
disagrees with the offered end option, a grant whose `max` is not its card
count, a missing or misplaced event name, `obs` or `metrics` missing out of
combat or present in combat, a bad cell, `t` at or past the run cap, a fight
turn outside 1-100 or a fight decision at or past 1,000, fight counters out
of combat, and a policy source that also exposes PRIV accessors.

### 8.2 Sources and the tokenizer

The env hands the tokenizer two objects. `PolicySource` exposes
`belief()`, `legal()` and a read-only run facade; `PrivSource` exposes
`true_state()`, `act_rooms()` and `future()`, and is passed only to the value
path. `act_rooms()` is this act's `generated_rooms()` record itself (`boss`,
`second_boss`, `normal`, `elite`, `events_shuffled`) and `future()` the later
acts' records; `encode_priv` raises `SchemaError` on anything that is not a
record. Both protocols are `runtime_checkable`; an object that is both is
rejected.

The `Tokenizer` protocol is how the env and the actor drive WP3b's code:

- `reset_fight()` at every fight start: a fresh `IntentLog`.
- `observe(policy)` at every combat step, single-option and auto-advanced
  ones included, before the action (#1081's `Encoder.observe`), so the
  intent history is complete.
- `encode(d)` for each stored decision: one row's policy arrays under the
  `ROW_FIELDS` names (`cat`, `cont`, `sec`, `tok_flags`, `mem_tok`,
  `mem_class`, `cls_tok`, `cls_kind`, `pool_*`, `sctx_lo`, `n_sctx`).
- `encode_priv(d)` on the value path only: `pcat`, `pcont`, `psec`,
  `ppool_*`. It reads `d.priv`; `encode` never does.

### 8.3 Honesty checks (F7, T14)

F7 and T14 permute what `PRIV_CALLS` read (the draw order through
`cs.sample_world`, the combat RNG, the rooms and bosses) and require
bit-identical policy outputs. Because members keep the env's order
(section 5), a draw-order permutation can reorder the classes of a draw-pile
choice without changing any class's logit. So F7 either compares outputs as
a map from class key to logit, or permutes only after the stored decision's
tokens are built. Out of combat, the same rule holds for the run RNG:
reseeding it on a clone (`rm.clone()`, `reseed_run`) must not change any
input the policy reads where the player sees no change. The live walk checks
this at every event decision (section 11).

## 9. Actor to learner row batch (i)

A batch is a struct of arrays: fixed-width per-row fields, and ragged arrays
whose lengths are the per-row counts (`n_tok`, `n_mem`, `n_classes`,
`n_priv`, `n_pool`, `n_ppool`), concatenated in row order, as in #1081's
`Packed`. Indices inside a row (`mem_tok`, `cls_tok`, `pool_tok`) are
row-local. The 14 required fields come first and keep their names; the rest
may grow (append only).

| Field | dtype | Axis | Width | Meaning |
|---|---|---|---|---|
| `run_uid` | int64 | row | - | unique per run or practice segment |
| `t` | int32 | row | - | order of the stored row within its run or segment |
| `is_combat` | bool | row | - | combat decision |
| `dtype` | uint8 | row | - | decision type (DECISION_TYPES) |
| `reward` | float32 | row | - | all reward from this row to the next stored row of the run, or to the end; 0 on a bootstrap row |
| `done` | bool | row | - | the run or segment ended: after this row, or at it when it is a bootstrap row |
| `bootstrap` | bool | row | - | value-only row, its run's last: an iteration cut (stored again next iteration, same t) or, with done, a practice segment's end; no loss or statistic |
| `actor` | uint8 | row | - | 0 student, 1 combat teacher, 2 run teacher |
| `logp_behav` | float32 | row | - | log prob of the taken class under the actor |
| `n_classes` | uint16 | row | - | merged options (>= 2) |
| `action` | uint16 | row | - | index of the taken merged class |
| `iter` | int32 | row | - | collection iteration |
| `v_behav` | float32 | row | - | acting-time value (logging only) |
| `cell` | uint8 | row | - | practice cell (CELLS), 255 = not practice |
| `n_tok` | int32 | row | - | tokens in the row |
| `n_mem` | uint16 | row | - | members (raw legal actions) |
| `n_priv` | int32 | row | - | PRIV tokens |
| `n_pool` | int32 | row | - | pooled entries |
| `n_ppool` | int32 | row | - | PRIV pooled entries |
| `sctx_lo` | uint16 | row | - | first S_ctx token (combat rows) |
| `n_sctx` | uint16 | row | - | S_ctx tokens (0 out of combat) |
| `t_src` | uint8 | row | - | teacher of `tprob`: 0 none, 1 combat, 2 run |
| `cat` | int16 | tok | 32 | categorical columns |
| `cont` | float16 | tok | 210 | continuous columns |
| `sec` | int8 | tok | - | section id |
| `tok_flags` | uint8 | tok | - | TF_* bits |
| `mem_tok` | uint16 | mem | - | token of each member, env action order |
| `mem_class` | uint16 | mem | - | class of each member |
| `cls_tok` | uint16 | cls | - | representative token of each class |
| `cls_kind` | uint8 | cls | - | action kind of each class |
| `tprob` | float32 | cls | - | teacher class probabilities, NaN when none |
| `pcat` | int16 | priv | 32 | PRIV categorical columns |
| `pcont` | float16 | priv | 210 | PRIV continuous columns |
| `psec` | int8 | priv | - | PRIV section id |
| `pool_tok` | uint16 | pool | - | token owning the entry |
| `pool_vid` | uint16 | pool | - | vocabulary id |
| `pool_w` | float16 | pool | - | weight (count) |
| `ppool_tok` | uint16 | ppool | - | PRIV token owning the entry |
| `ppool_vid` | uint16 | ppool | - | vocabulary id |
| `ppool_w` | float16 | ppool | - | weight (count) |

Semantics:

- `action` indexes the merged classes; the env applies the class's
  representative member. `logp_behav` and `v_behav` are the acting policy's
  (student or teacher) log probability and value. `n_classes >= 2` always.
- `tprob` holds the teacher's class probabilities: all NaN when
  `t_src = 0`, a distribution otherwise. Combat-teacher probabilities occur
  only on combat rows and run-teacher probabilities only on OOC rows.
- `cell` is the practice cell (`ELITE_ACT1`-`ELITE_ACT3`, `BOSS_ACT1`,
  `BOSS_ACT2`, `BOSS_ACT3_DOUBLE`), 255 outside practice. `bootstrap` and
  `done` follow section 9.1.
- `tok_flags`: `TF_OPTION` (1, a member token), `TF_REP` (2, a class
  representative), `TF_HIDDEN` (4, hidden from the student), `TF_TEACHER` (8,
  inside the S_ctx block), `TF_POOLED` (16, owns pool entries).
- `validate_row_batch(batch)` checks, per batch: fields, dtypes, widths,
  ragged lengths, the caps (`n_pool`, `n_ppool` at most 1,024 each, `t` below
  6,000, `iter` not negative), id ranges, finite `reward`, `v_behav` and
  `logp_behav`, the actor against the row kind (1 only in combat, 2 only out
  of combat), and the bootstrap rule (section 9.1). Per row: the head (section
  2.1) and the RUNCTX ids (section 2.4); the member-to-class map (onto,
  numbered by first occurrence); class tokens equal to their first member's
  token; `TF_OPTION`, `TF_REP`, `TF_HIDDEN`, `TF_TEACHER` and `TF_POOLED` on
  exactly their tokens; each member's `X_KIND` and section against its class
  kind; a combat row's S_ctx block from `sctx_lo` to the row's end, starting
  with `STATE` and ending with its `n_mem` action tokens in member order; an
  OOC row's body order and its option tokens last, in class order; the pool
  rules (section 2.6); the PRIV caps; and `tprob`. Reserved ids and sections
  fail unless `allow_reserved=True`. `test_contracts.py` runs one mutation per
  rule and checks the reason.
- `take_rows(batch, rows)` slices a batch to rows (a minibatch, a stratum),
  every ragged axis with it; `concat_row_batches(batches)` concatenates.

### 9.1 Runs, cuts and practice ends

A run is stored as consecutive rows `t`, `t + 1`, ... of one `run_uid`; the
learner chains returns by row adjacency (§6). Each iteration's rows of a run
end in exactly one of three ways:

| End | Last row | `done` | `bootstrap` | Next iteration |
|---|---|---|---|---|
| Terminal | the run's last decision; its `reward` includes the outcome | true | false | - |
| Iteration cut | the run's next decision, value only: it supplies V for the previous row's return | false | true | stored again with the same `t`, then acted on and trained once |
| Practice end | the segment's first out-of-combat decision (cell not 255), value only | true | true | never stored again |

This is WP4a's convention (`ppo/e2e/algo.py`), adopted for revision 2. A
bootstrap row carries reward 0: its reward belongs to its next storing. Its
acting fields (`action`, `logp_behav`, `tprob`) are not forced to dummy
values, so a cut row may carry the action it is about to take, but no loss
or statistic reads them. A double-boss segment plays to the true outcome and
ends as a terminal.

`validate_row_batch(batch, runs=True)` adds the per-run checks for one
iteration's fresh table: each run's `t` distinct and contiguous, one cell per
run, the last row `done` or `bootstrap`, and no `done` or `bootstrap` row
before it. Leave `runs=False` for derived batches (minibatches, strata, a
learner table with held rows).

**Size.** Dense rows cost 486 bytes per token (32 × 2 + 210 × 2 + 2), about
28 KB for a 57-token row. The compact layout (S1b) stores only each
section's live columns (`compact_token_bytes`: `STATE` 114 bytes, `HAND` 56,
pile cards 48, `ENEMY` 78, `ACT_PLAY` 118, `RUNCTX` 58, `SUMMARY` 10, ...),
2.4 KB of tokens per combat row on average in the live walk (4,871 rows),
plus about 0.3 KB of per-row, member and pool data: within the ~3.5 KB goal
(§8). `to_compact` and `from_compact` convert losslessly (tested); the
learner should keep rows compact from day one, and dense rows are a working
layout.

## 10. Stamps (j)

`stamps()` builds the dict every artifact carries. `check_stamps(have, want)`
raises unless an artifact's stamps fit the process's: equal schema,
vocabulary and game version; equal teacher stamps where both name a teacher;
and the same fill state (`filled` and `prior_sha256`). With `restart=True`
(a restart that loads `have` under section 2.12's identity rule) the process
may add fill events and the prior: `have`'s events must be a subset of
`want`'s, and `have`'s prior must be `want`'s or null. A checkpoint that read
filled columns is never served by a process that leaves them at zero.

| Key | Value |
|---|---|
| `schema_sha256` | sha256 of the canonical JSON of `schema_spec()` (every constant in `contracts.py`) |
| `vocab_sha256` | sha256 of the content-ID table's canonical serialization |
| `prior_sha256` | sha256 of the content-prior buffer, or null |
| `filled` | fill events that happened (`WP-ENG-1`, `content-prior`, `D9`, `#1069`) |
| `teacher` | `teacher_stamp(...)` where the combat teacher or its RareMap'd rows are involved, else null: the teacher checkpoint's sha256, and sha256s of its RareMap (`ck["rare"]`) and S_ctx table sizes (`ck["vocab"]`) |
| `game_version` | `v0.111.0` |
| `wheel` | engine wheel identity (F1) |
| `source_digest` | source digest of the run |

Where they go: the run cfg (`cfg["stamps"]`), every checkpoint's metadata,
the header of every stats file, every eval row, and the metadata of every
row batch (the learner rejects a batch whose stamps differ from its cfg).
`champions.toml`'s `[ironclad_e2e]` lane records `schema_sha` and
`game_version` (§13).

## 11. Evidence

`test_live_walk_maps_every_decision` walks seven v0.111.0 A10 Ironclad runs
(seeds 162,000,200-162,000,205, and 162,000,356, which reaches Sea Glass)
through `PpoRunEnv(combat="yield")`, with HP held at 999 so a random policy
reaches act 3. It runs in about 5 seconds. Results:

- 5 runs won, 2 lost; every decision mapped to a live type and every option
  to a live kind.
- Multi-option decisions: `COMBAT_TURN` 4,598, `COMBAT_CHOICE` 273,
  `CARD_REWARD` 168, `MAP` 89, `EVENT` 53, `SHOP` 42, `REST` 41, `SMITH` 25,
  `BOSS_RELIC` 12, `CARD_REMOVE` 8, `NEOW` 7, `CARD_GRANT` 5 (3 of them
  Sea Glass's optional grid), `CARD_ENCHANT` 4, `CARD_BUNDLE` 2,
  `CARD_UPGRADE` 2, `POTION_SWAP` 2, `CARD_TRANSFORM` 1.
- 5,329 of them have at least two merged classes and would be stored; each
  one, built as an `EnvDecision`, passes `validate_decision`. The other 3
  (2 combat, 1 OOC) merge into one class. One combat choice re-opened after
  a pick (`R_SEL_PICKED` = 1).
- Largest decision: 64 raw (39 merged) OOC options, 68 raw (25 merged)
  combat options; largest combat row 148 tokens.
- 2,532 merged combat members, all token-identical up to `hand_slot`;
  `rollout_features(bv) == rollout_features(cs)` on 4,871 of 4,871 checks;
  every visible card, relic, potion, power, monster, event, rest option and
  boss name is in the content-ID table; the A10 act-3 second boss is on the
  map on 105 decisions.
- Event previews: at each of the 60 event and Neow decisions the walk reseeds
  the run RNG on six clones. Of 142 options, 52 gain a relic and are masked.
  717 option pairs show the player the same fields on the clone; their raw
  deltas differ in 8 and their honest deltas in none.

`test_event_previews_are_honest` forces each of the 69 events on seeds 1-30
(the e2e env's config) and walks two pages deep: 510 option groups that look
the same to the player, 12 of them with raw previews that differ between
seeds, none with honest previews that do (about 3 seconds).

## 12. Open questions

Each has a recommended answer; G-schema confirms or overrides it.

1. **`e_kind`: action kind or decision type?** §5 says "decision-kind
   embedding". Recommended: the option's action kind (128 rows). It is finer,
   and the decision type already reaches the head through `h_STATE`
   (`X_DTYPE` on `RUNCTX`) and head selection.
2. **rl2 features on the belief view.** Recommended: compute
   `rollout_features(bv)`; it matched `rollout_features(cs)` on all 4,871
   live checks, it is honest by construction, and X-C keeps verifying equality on
   recorded states. If X-C ever finds a difference, the teacher view uses the
   recorded `cs` value and the student's copy is zeroed.
3. **Multi-pick choose key.** Recommended: keep the selected-content
   extension (section 5). Report #1081's `action_classes` merge to its
   owners; it changes their class probabilities on these choices.
4. **PRIV cap 16.** Recommended: `PRIV_DRAW` 10 + `PRIV_ENEMY` 5 +
   `PRIV_RUN` 1 = 16 exactly, with rooms and future bosses pooled into one
   token. Raise the cap only if WP-ENG-1's true-next-move getter needs more
   than one token per enemy.
5. **Move registry in the vocabulary.** Recommended: keep the 356 moves
   (ids 1,583-1,938) for `PRIV_ENEMY`; policy tokens never reference them.
   Their rows start zero (identity rule).
6. **Out-of-combat dedupe.** Recommended: merge only content payloads (the
   table in section 5). Deck cards carry no per-copy counters in
   `master_deck_detailed()`, so two copies whose hidden growth differs (for
   example Ritual Dagger) merge out of combat; the player cannot tell them
   apart either.
7. **`COMBAT_CHOICE` and potions.** Recommended: no potion kinds during a
   pending choice; none occurred in the walk, and an occurrence raises.
8. **Content prior as virtual columns.** Recommended: gather from a frozen
   buffer net-side instead of storing 48 columns per token: rows stay small,
   and a later extractor fill changes only `prior_sha256`. A regenerated
   table is not only a fill: the tactical columns read its HITS column
   (`tactical.hits_of`), so `content_prior.TABLE_SHA256` pins the table in
   a source file that `py_src` covers.
9. **Dense versus compact rows.** Recommended: ship the compact codec with
   WP3b instead of waiting for S1b; dense v1 rows are about 1.8 times
   #1081's 15.7 KB.
10. **Relic rewards and treasure.** Under `PpoRunEnv`'s `mode="agent"`
    (`direct_event_relic_grant` on) the engine auto-collects relic rewards
    (section 6.2), so the plan's pre-flight item "relic reward occurs"
    (§3.2) cannot pass. Recommended: keep `RELIC_REWARD` and `TREASURE` live
    (cheap; they fire if a flag or a wheel changes), drop "relic reward" from
    the pre-flight's must-occur list and report its count instead, and leave
    the real-game relic skip to #1069.
11. **Rest-site labels.** Recommended: WP2 takes rest options from
    `rest_site_options()`; `ppo/env.py`'s two-label table is wrong for three
    or more options.
12. **`max_steps`.** Recommended: WP2 passes `max_steps=20_000` (or more) to
    `PpoRunEnv`, so a capped run is always the schema's cap, never the env's
    3,000-step default.
13. **Picks in a multi-pick combat choice.** The engine keeps `picked` off
    the wire. Recommended: derive `R_SEL_PICKED` and `K_PICKED` from the
    offered actions (an option with no `CHOOSE` action is picked), which is
    exact under the engine's rule (lib.rs:14423-14436) and needs no env
    memory. If a later wheel puts `picked` on the wire, switch to it and
    assert that both agree.
14. **Event previews in the engine.** Section 2.7's filter costs the policy
    the fixed side of a relic option (a gold, HP or max-HP cost reads 0).
    Recommended: fix the preview in Rust (resolve the option without the
    rolled relic's pickup, or on a clone whose relic roll is scrubbed), then
    lift the mask in a schema revision. Until then a refinement may read the
    fixed cost from the option's `vars`, if a probe shows them exact. The
    Rust no-peek test (sts2-content/tests/event_preview_no_peek.rs) runs only
    under `V0_107_1`; nobody ran it under `V0_111_0` (this work package may
    not run cargo).
15. **The run teacher reads raw previews.** Recommended: keep it so (section
    3.2). The teacher's sealed baselines were measured with the same input,
    and T8's distillation target on relic-option event rows is then partly
    out of the student's reach. Tell the run-teacher owners; if KL on those
    rows stays high, mask the teacher's event deltas the same way and re-measure.
16. **Bootstrap rows (section 9.1).** Integration needs three changes outside
    this work package: WP2's `segment_rows` sets `done` on the practice-end
    bootstrap row (today it never does); WP4c's `table_to_rows` can drop its
    flag translation; and a practice end's row never trains, so the learner
    must not hold it as it holds a cut.
17. **Deviations from the plan.** (a) §4 says merged members have
    byte-identical tokens; S_ctx's `hand_slot` column differs between them
    (section 5), and the assertion skips it. (b) §4 lists future maps among
    the privileged inputs; `PRIV_RUN` carries no map (section 2.10): a map is
    tens of nodes against a 16-token PRIV cap, and the reserved PRIV sections
    50-55 can add it at a restart.
18. **WP2 and WP3b alignment.** WP2's `RunFacade` (on `wp/2-env`) forwards
    every allowed `rm.*` read raw; it must pass `rm.event_option_infos`
    through `POLICY_FILTERS`, and the policy tokenizer must read
    `honest_obs(d.obs)` while the run teacher keeps `d.obs`. The tokenizer
    on `wp/3b-tok` must take the `O_EV_*` values from
    `event_delta_columns`, take `X_ID_AUX` from `bv.encounter_id` on combat
    rows only (the facade no longer serves `encounter_id`), and keep
    `TF_POOLED` on all five SUM_* tokens (it does). Its reads of
    `rm.master_deck_enchantments`, `rm.ascension_level` and
    `rm.pending_enchantment` are now in `POLICY_CALLS`.

## 13. Other characters: the "chars" fill event (GH #1373)

The schema was frozen for Ironclad. This section says what a row of another
character's run carries, and why an Ironclad row, fixture and checkpoint does
not change.

### 13.1 The rule

- **A row carries the event or it does not; `X_CHAR` says which.** Every row
  of a Silent, Defect, Regent or Necrobinder run carries it. An Ironclad row
  carries it only in a launch that declares the event. Without the
  declaration an Ironclad row has `X_CHAR` 0, no `ALLY` or `ORB` token and
  `E_DOOMED` 0: it is bit for bit what it was (section 13.4), which is what
  the trained Ironclad net and its fixtures need.
- **A declared Ironclad launch shows orbs and Doom.** An Ironclad can hold
  orbs (Potion of Capacity, Essence of Darkness, Infused Core, Runic
  Capacitor) and face Doom (Potion of Doom, Undying Sigil). With "chars" in
  `--filled` its rows carry `X_CHAR` 1, the `ORB` tokens and `E_DOOMED` like
  any other character's.
- **It takes reserved slots under alias names.** No table grows: `cat` stays
  `[T, 32]`, `cont` `[T, 210]`, the section table 64 rows. The hashed tables
  keep the reserved names (`RES_44`, `RES_45`, `X_RES0`, `X_C_RES0`-`X_C_RES2`),
  so `schema_sha256` does not move. `contracts.SEC`, `XC` and `XF` resolve the
  aliases; `contracts.SECTION_LIVE` is the frozen table plus the event's
  columns, and the hash reads the frozen one.
- **The identity rule holds.** A net that does not declare the event starts
  the two sections' embedding rows, `X_CHAR`'s rows and the three columns'
  stem weights at zero, as for every reserved input. An Ironclad checkpoint
  loads unchanged. `E2ENet(filled=[..., "chars"])` inits them live.
- **Declaring it.** A learner for another character sets
  `STS2_E2E_CHARACTER=<name>` (`factory.CHARACTER`; the workers inherit it) and
  passes `--character <name> --filled '["WP-ENG-1", "chars"]'`. An Ironclad
  learner passes "chars" in `--filled` to opt in. The learner refuses a
  character that disagrees with the process and another character without
  "chars"; it sets `STS2_E2E_CHARS` for its workers from `--filled`
  (`factory.set_fill_chars`), and the Rust training env takes the same flag.
  The fill check (`fillcheck.check`) holds the rows to the declaration. A
  resume may add the event to an Ironclad checkpoint (the identity rule).
- **Serving.** The Rust hosts that load a net (`sts2_rs` student runs, the
  workbench brain's playout) read the event from the weight file's `filled`
  (`E2eNet::fills("chars")`). The per-position drivers (`E2eDriver`,
  `E2eState`) do not hold a net: a host that serves a declared Ironclad net
  through them calls `TokCache::set_chars` itself. Python evaluation of such
  a net needs `STS2_E2E_CHARS=1` in the evaluating process.

### 13.2 What the event adds

| Alias | Reserved name | Where | Meaning |
|---|---|---|---|
| `X_CHAR` | `X_RES0` (cat 28) | `RUNCTX` | the run's character: 1 Ironclad, 2 Silent, 3 Defect, 4 Regent, 5 Necrobinder, 6 Fool (v0.113.0 on); 0 = the row carries no "chars" data |
| `ALLY` | `RES_44` (section 44) | combat head | the player's pet, one token while the run has one: Necrobinder's Osty, alive or dead. `X_ID_MAIN` = the monster id; `E_HPF`, `E_HP`, `E_MAXHP`, `E_ALIVE` as on `ENEMY` |
| `ORB` | `RES_45` (section 45) | combat head | one token per orb slot in belt order, at most 10. `X_ID_MAIN` = the orb id (`EMPTY` for an empty slot), `X_POS` = 1 + slot |
| `C_ORB_PASSIVE` | `X_C_RES0` (cont 202) | `ORB` | passive value with Focus / 10 |
| `C_ORB_EVOKE` | `X_C_RES1` (cont 203) | `ORB` | evoke value with Focus / 20 |
| `E_DOOMED` | `X_C_RES2` (cont 204) | `ENEMY` | flag: Doom >= hp, so the enemy dies at the end of its turn (`DoomPower.IsOwnerDoomed`) |

The `ALLY` and `ORB` tokens sit between the `BOSS` tokens and the S_ctx block,
in that order, so `sctx_lo` = 6 + bosses + their count. The S_ctx block and
the teacher slice do not change. They never occur out of combat. Their source
is `state_to_dict(bv)["player"]` (`osty`, `orbs`, `orb_slots`), a policy read;
the character is `rm.character`, which the event adds to the run facade
(`contracts.FILL_POLICY_CALLS`).

`S_HP_LOSS_END` keeps its meaning (the HP the player loses if the turn ended
now) and gains two terms that are zero for Ironclad:

- the block the Frost orbs' passives add at the end of the player's turn
  (`OrbQueue.BeforeTurnEnd`, `FrostOrb.Passive`), taken off the incoming damage
  with the player's block;
- the HP of a living Osty, taken off what is left: Osty takes the unblocked
  damage of an attack on its owner, and the overkill passes on to the owner
  (`DieForYouPower.ModifyUnblockedDamageTarget`, `CreatureCmd.Damage`).

### 13.3 Coverage per character

Checked against the v0.111.0 decompilation. "Covered" means the policy row
holds it.

| Character | Mechanic | Row | Covered |
|---|---|---|---|
| all | the character | `X_CHAR` | yes (0 on an undeclared Ironclad row) |
| Necrobinder | Osty: alive, HP, max HP | `ALLY` | yes |
| Necrobinder | Osty soaks unblocked attack damage | `S_HP_LOSS_END` | yes |
| Necrobinder | Osty's attacks (Poke, Unleash, ...) | hand `DMG_PREV`, `E_MAXDMG`, `E_KILLABLE` | yes: `card_damage_preview` prices an Osty attack from Osty as the dealer |
| Necrobinder | Osty's block | - | not needed: a hit on Osty spends its owner's block (`CreatureCmd.Damage`), which `STATE` has |
| Necrobinder | powers on Osty | - | not needed: the only one is Die For You, which every Osty has (GH #760's finding); no card, potion or monster move puts another on it |
| Necrobinder | Doom on an enemy | `EPOWER` (amount), `E_DOOMED` | yes |
| Necrobinder | Souls and other generated cards | card tokens | yes |
| Defect | orb belt: kinds, order, passive, evoke, empty slots | `ORB` | yes |
| Defect | Focus | `PPOWER` | yes (and inside the orb numbers) |
| Defect | Frost block at the end of the turn | `S_HP_LOSS_END` | yes |
| Defect | Lightning and Glass passive damage at the end of the turn | `ORB` only | not in `E_MAXDMG` or `E_KILLABLE`: the targets are random |
| Defect | orb-slot choices (`orb_slot` choice kind) | `ACT_CHOOSE`, `STATE` | yes (S_ctx) |
| Regent | stars | `STATE` `STARS` | yes (S_ctx) |
| Regent | forge / Sovereign Blade | the Sovereign Blade card's own tokens | as that card's numbers only; no forge total on `STATE` (not audited further here) |
| Silent | Shivs, Sly, poison | card and power tokens | yes |
| Ironclad | orbs and Doom from shared potions and relics | `ORB`, `E_DOOMED` | yes in a launch that declares "chars"; else unchanged |

Left for later, on purpose:

- **The content prior's `POOL_IRONCLAD` column** still means the Ironclad
  pool; another character's own cards read `POOL_OTHER`. A per-character prior
  has another `prior_sha256`, which `check_stamps` refuses for an Ironclad
  checkpoint.
- **The Rust serving helpers** (`sts2_e2e::env::new_run_at`, the workbench
  brain) build Ironclad runs only. The tokenizer and the training env take any
  character's run.
- **No combat teacher** exists for another character, so such a launch runs
  without one (GH #1197).

### 13.4 Evidence

- `tests/e2e/test_contracts.py::test_schema_sha256_golden`: the hash is the
  one this document quotes.
- Ironclad rows before and after the event, same wheel: 2,330 rows of six
  random A10 runs, every policy and PRIV array hashed, equal sha256 on the
  tokenizer of befdab54 and on this one.
- `tests/e2e/test_e2e_chars.py`: the slots and the frozen tables; an
  undeclared Ironclad row carries none of the event, and a declared one shows
  an orb belt and Doom; each other character's rows validate, are
  compact-lossless and hold the tokens their `state_to_dict(bv)` implies; the
  tactical terms against hand-computed values of the decomp's rules; a net
  that does not declare the event keeps its inputs at zero and its logits do
  not move with `X_CHAR` or `E_DOOMED`; the fill check; and
  `rsenv_lockstep` on each other character (the Rust training env's rows
  equal the Python env's, PRIV included).
- Not measured: learning. No lane has trained on these rows yet.
