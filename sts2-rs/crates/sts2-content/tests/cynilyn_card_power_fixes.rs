//! Card / power / relic engine fixes first found by the cynilyn fork against
//! the real game, ported for BOTH builds (v0.107.1 and v0.111.0) unless the
//! two decomps differ. Each test cites the C# it pins.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V107: GameVersion = GameVersion::V0_107_1;
const V111: GameVersion = GameVersion::V0_111_0;
const BOTH: [GameVersion; 2] = [V107, V111];

fn add_enemy(state: &mut CombatState, raw: u32, hp: i32) -> CreatureUid {
    let id = NonZeroU32::new(raw).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: hp,
        max_hp: hp,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    CreatureUid(id)
}

/// Player with 5 energy in the play phase, no enemies yet.
fn fresh(version: GameVersion, character: CharacterId) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(character);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn new_card(state: &mut CombatState, id: CardId, upgraded: bool) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, upgraded, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn push_hand(state: &mut CombatState, id: CardId, upgraded: bool) -> u8 {
    let uid = new_card(state, id, upgraded);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

fn hp_of(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.current_hp)
}

fn add_player_power_with_applier(
    state: &mut CombatState,
    raw: u32,
    kind: PowerKind,
    amount: i32,
    applier: CreatureUid,
) {
    let pid = NonZeroU32::new(raw).expect("nz");
    let mut p = Power::new(pid, kind, amount);
    p.applier = Some(applier);
    state.powers.push(p);
    state.player.powers.push(PowerUid(pid));
}

// ── 1. Thunderclap + Shrink whose applier dies in the same attack ──

/// Enemy B (1 HP, the Shrink applier) is hit first and dies; enemy A (100 HP)
/// is the second target. `CreatureCmd.Damage` resolves every target's
/// `ModifyDamage` before any death, and `ShrinkPower.cs:88` `AfterDeath`
/// removes the power only afterwards: A still takes floor(4 * 0.7) = 2.
#[test]
fn thunderclap_keeps_shrink_for_the_whole_attack() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let applier = add_enemy(&mut state, 2, 1);
        let other = add_enemy(&mut state, 3, 100);
        add_player_power_with_applier(&mut state, 900, PowerKind::Shrink, -1, applier);
        let i = push_hand(&mut state, CardId::Thunderclap, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert_eq!(hp_of(&state, applier), 0, "{v:?}");
        assert_eq!(hp_of(&state, other), 98, "{v:?}");
    }
}

// ── 2. LoseMaxHp below current HP deals damage first ──

fn add_relic(state: &mut CombatState, kind: sts2_core::RelicKind) {
    let raw = u32::try_from(state.relics.len() + 1).expect("u32");
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(raw).expect("nz"),
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

fn fill_draw(state: &mut CombatState, n: usize) {
    for _ in 0..n {
        let uid = new_card(state, CardId::StrikeIronclad, false);
        state.player.draw.push(uid);
    }
}

/// Full-HP player with Centennial Puzzle plays Brightest Flame (draw 2, lose
/// max HP). `CreatureCmd.LoseMaxHp` (CreatureCmd.cs:824, v0.111 :879): when
/// `newMaxHp < CurrentHp` it first deals `CurrentHp - newMaxHp`
/// Unblockable|Unpowered(|Move) damage, so Centennial Puzzle's after-damage
/// draw 3 fires: 2 + 3 cards in hand.
#[test]
fn lose_max_hp_below_current_hp_fires_after_damage_listeners() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut state, 2, 100);
        add_relic(&mut state, sts2_core::RelicKind::CentennialPuzzle);
        fill_draw(&mut state, 10);
        let i = push_hand(&mut state, CardId::BrightestFlame, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        let loss = if v >= V111 { 2 } else { 1 };
        assert_eq!(state.player.hand.len(), 5, "{v:?}");
        assert_eq!(state.player.max_hp, 80 - loss, "{v:?}");
        assert_eq!(state.player.current_hp, 80 - loss, "{v:?}");
    }
}

/// With current HP already under the new cap no damage is dealt: only the
/// cap shrinks, Centennial Puzzle stays quiet.
#[test]
fn lose_max_hp_above_current_hp_deals_no_damage() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut state, 2, 100);
        state.player.current_hp = 50;
        state.sync_player_creature();
        add_relic(&mut state, sts2_core::RelicKind::CentennialPuzzle);
        fill_draw(&mut state, 10);
        let i = push_hand(&mut state, CardId::BrightestFlame, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert_eq!(state.player.hand.len(), 2, "{v:?}");
        assert_eq!(state.player.current_hp, 50, "{v:?}");
    }
}

/// Enemies take the same path (nothing in `LoseMaxHp` is player-only). A
/// loss of the whole cap deals `CurrentHp - (MaxHp - amount)` >= CurrentHp
/// damage, which KILLS; the old clamp-only shrink left the creature at 1 HP
/// (`SetMaxHp` floors the cap at 1, CreatureCmd.cs:827).
#[test]
fn enemy_losing_its_whole_max_hp_dies() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let e = add_enemy(&mut state, 2, 10);
        let _keep_alive = add_enemy(&mut state, 3, 10);
        let _ = state.creature_lose_max_hp(e, 10);
        assert_eq!(hp_of(&state, e), 0, "{v:?}");
    }
}

// ── 3. Philosopher's Stone strengthens mid-combat summons ──

fn power_amount(state: &CombatState, uid: CreatureUid, kind: PowerKind) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| {
            c.powers
                .iter()
                .filter_map(|pu| state.powers.iter().find(|p| p.id == pu.0))
                .filter(|p| p.kind == kind)
                .map(|p| p.amount)
                .sum()
        })
}

/// `PhilosophersStone.cs:41` `AfterCreatureAddedToCombat`: an enemy added
/// mid-combat (`CreatureCmd.Add` -> `Hook.AfterCreatureAddedToCombat`) gains
/// the Stone's +1 Strength; the room-entry leg only covers the starters.
#[test]
fn philosophers_stone_strengthens_mid_combat_summons() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut state, 2, 50);
        add_relic(&mut state, sts2_core::RelicKind::PhilosophersStone);
        let summon = state.spawn_enemy(MonsterKind::Nibbit);
        assert_eq!(
            power_amount(&state, summon, PowerKind::Strength),
            1,
            "{v:?}"
        );

        // Without the Stone nothing is added.
        let mut plain = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut plain, 2, 50);
        let summon = plain.spawn_enemy(MonsterKind::Nibbit);
        assert_eq!(
            power_amount(&plain, summon, PowerKind::Strength),
            0,
            "{v:?}"
        );
    }
}

// ── 4. Stomp / Pinpoint clones are not back-filled again ──

/// `Stomp.cs:53` / `Pinpoint.cs:43` `AfterCardEnteredCombat` return early on
/// `IsClone`: a copy of an already-discounted card keeps the cost it was
/// cloned with and is not shaved again for the plays finished this turn. A
/// freshly generated card (no bank) still gets the back-fill.
#[test]
fn stomp_and_pinpoint_clones_keep_their_cloned_cost() {
    use sts2_core::cards::{CardType, VarKey};
    let cases = [
        (
            CardId::Stomp,
            VarKey::StompCostReductionThisTurn,
            CardType::Attack,
        ),
        (
            CardId::Pinpoint,
            VarKey::PinpointCostReductionThisTurn,
            CardType::Skill,
        ),
    ];
    for v in BOTH {
        for (id, bank, ty) in cases {
            let mut state = fresh(v, CharacterId::Ironclad);
            let _ = add_enemy(&mut state, 2, 50);
            let orig = new_card(&mut state, id, false);
            let base = state.cards[orig.0.get() as usize - 1].cost;
            state.player.hand.push(orig);
            state.player.cards_played_this_turn[ty.index()] = 2;
            // The in-hand original was shaved by 2 by those plays.
            let c = &mut state.cards[orig.0.get() as usize - 1];
            c.cost = base - 2;
            c.set_combat_var(bank, 2);

            let clone = state.cards[orig.0.get() as usize - 1];
            let uid = sts2_core::card_effects::spawn_card_into_hand(&mut state, clone);
            let cost = state.cards[uid.0.get() as usize - 1].cost;
            assert_eq!(cost, base - 2, "{v:?} {id:?} clone");

            let fresh_card = card_pod_for(id, false, v).expect("spec");
            let uid = sts2_core::card_effects::spawn_card_into_hand(&mut state, fresh_card);
            let cost = state.cards[uid.0.get() as usize - 1].cost;
            assert_eq!(cost, base - 2, "{v:?} {id:?} fresh");
        }
    }
}

// ── 5. A Juggling copy of Stomp keeps its cloned cost (v0.111) ──

/// Stomp played as the turn's 3rd Attack under Juggling(1); returns the cost
/// of the Juggling copy in hand.
fn juggled_stomp_copy_cost(v: GameVersion) -> i8 {
    use sts2_core::cards::{CardType, VarKey};
    let mut state = fresh(v, CharacterId::Ironclad);
    let _ = add_enemy(&mut state, 2, 100);
    let p = state.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Juggling, 1);
    for pid in state.player_power_uids() {
        if let Some(pw) = state.powers.get_mut_by_id(pid.0) {
            if pw.kind == PowerKind::Juggling {
                pw.counter = 2;
            }
        }
    }
    state.player.cards_played_this_turn[CardType::Attack.index()] = 2;
    let i = push_hand(&mut state, CardId::Stomp, false);
    // Two Attacks already shaved it 3 -> 1.
    let c = &mut state.cards[0];
    c.cost = 1;
    c.set_combat_var(VarKey::StompCostReductionThisTurn, 2);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.hand.len(), 1, "{v:?}: the Juggling copy");
    state.cards[state.player.hand[0].0.get() as usize - 1].cost
}

/// v0.111 `JugglingPower.cs:36` clones in `BeforeCardPlayed`; the hook's
/// listener list is snapshotted before it runs, so `Stomp.cs:62`
/// `BeforeCardPlayed` never visits the fresh copy: it keeps the cost it was
/// cloned with (1). v0.107.1 clones after the play, from the already-shaved
/// card (0).
#[test]
fn juggled_stomp_copy_keeps_its_cloned_cost() {
    assert_eq!(juggled_stomp_copy_cost(V111), 1);
    assert_eq!(juggled_stomp_copy_cost(V107), 0);
}

// ── 6. Dampen restores only the cards it downgraded ──

/// `DampenPower.cs:29` records the exact card instances it downgraded and
/// `AfterRemoved` re-upgrades only those. Anger's copy (made while the
/// original is downgraded) is not in that list and stays unupgraded.
#[test]
fn dampen_restores_the_original_but_not_its_clone() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let e = add_enemy(&mut state, 2, 100);
        let original = new_card(&mut state, CardId::Anger, true);
        state.player.hand.push(original);
        let oi = original.0.get() as usize - 1;
        assert!(state.cards[oi].upgraded);
        sts2_core::monster_power_effects::install_dampen_power_handlers();
        let player = state.player.creature_uid.expect("player");
        let _ = sts2_core::monster_power_effects::dispatch_after_applied_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
            Some(e),
        );
        assert!(!state.cards[oi].upgraded, "{v:?}");

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
        let clone = state
            .cards
            .iter()
            .find(|c| c.card_id == CardId::Anger && c.uid != original)
            .map(|c| c.uid)
            .expect("Anger clone");
        let _ = sts2_core::monster_power_effects::dispatch_after_removed_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
        );
        assert!(state.cards[oi].upgraded, "{v:?}");
        assert!(!state.cards[clone.0.get() as usize - 1].upgraded, "{v:?}");
    }
}

/// `EchoingSlash.cs:31` passes `base.DynamicVars.Damage` to every wave, so a
/// wave that kills the Dampen caster re-upgrades the card (`DampenPower.cs:41`
/// `AfterDeath`) and the next wave hits for the upgraded damage. Rust read the
/// damage once before the loop: act23 Ironclad-3003 (act-2 checkpoint,
/// branch 3) left the survivor at 39 where C# had 33.
#[test]
fn echoing_slash_rewave_after_killing_the_dampen_caster_is_upgraded() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let caster = add_enemy(&mut state, 2, 5);
        let survivor = add_enemy(&mut state, 3, 100);
        let slash = new_card(&mut state, CardId::EchoingSlash, true);
        state.player.hand.push(slash);
        let si = slash.0.get() as usize - 1;
        let upgraded_damage = state.cards[si].base_damage;
        sts2_core::monster_power_effects::install_dampen_power_handlers();
        let player = state.player.creature_uid.expect("player");
        // As MagiKnight's DAMPEN move does: the power instance records its
        // caster, so the caster's death removes it and re-upgrades the cards.
        sts2_core::card_effects::apply_power_to_creature(&mut state, player, PowerKind::Dampen, 1);
        let pid = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Dampen)
            .map(|p| p.id)
            .expect("Dampen applied");
        state.powers.get_mut_by_id(pid).expect("Dampen").applier = Some(caster);
        let _ = sts2_core::monster_power_effects::dispatch_after_applied_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
            Some(caster),
        );
        let dampened_damage = state.cards[si].base_damage;
        assert!(
            dampened_damage < upgraded_damage,
            "{v:?}: Dampen downgraded"
        );

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
        let hp = state
            .creatures
            .iter()
            .find(|c| c.id == survivor.0)
            .expect("survivor")
            .current_hp;
        assert_eq!(
            100 - hp,
            dampened_damage + upgraded_damage,
            "{v:?}: wave 1 dampened, wave 2 re-upgraded"
        );
    }
}

/// `BoneShards.cs:35-40` reads `DynamicVars.Block` for its `GainBlock` only
/// AFTER the Osty attack, so an attack that kills the Dampen caster
/// re-upgrades the card (`DampenPower.AfterDeath` -> `AfterRemoved`) and the
/// block is the upgraded 12. Rust read both values before the attack: loadout
/// Necrobinder 1159-v0.107.1 had C# 6 -> 18 block, Rust 15.
#[test]
fn bone_shards_block_after_killing_the_dampen_caster_is_upgraded() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Necrobinder);
        let caster = add_enemy(&mut state, 2, 5);
        let _survivor = add_enemy(&mut state, 3, 100);
        let _ = state.summon_osty(5).expect("osty summoned");
        let shards = new_card(&mut state, CardId::BoneShards, true);
        state.player.hand.push(shards);
        sts2_core::monster_power_effects::install_dampen_power_handlers();
        let player = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, player, PowerKind::Dampen, 1);
        let pid = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Dampen)
            .map(|p| p.id)
            .expect("Dampen applied");
        state.powers.get_mut_by_id(pid).expect("Dampen").applier = Some(caster);
        let _ = sts2_core::monster_power_effects::dispatch_after_applied_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
            Some(caster),
        );
        assert!(!state.cards[shards.0.get() as usize - 1].upgraded, "{v:?}");
        state.player.block = 0;

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
        assert_eq!(hp_of(&state, caster), 0, "{v:?}: the caster died");
        assert_eq!(state.player.block, 12, "{v:?}: re-upgraded block");
    }
}

/// `HiddenCache.OnPlay` gains its star FIRST and reads `StarNextTurnPower`
/// after: a Black Hole hit off that star that kills the Dampen caster
/// re-upgrades the card, so the power is the upgraded 4. Rust applied the
/// power before the stars (loadout Regent 1258 v0.111.0: C# 4, Rust 3).
#[test]
fn hidden_cache_star_next_turn_after_killing_the_dampen_caster_is_upgraded() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Regent);
        let caster = add_enemy(&mut state, 2, 1);
        let _survivor = add_enemy(&mut state, 3, 100);
        let cache = new_card(&mut state, CardId::HiddenCache, true);
        state.player.hand.push(cache);
        sts2_core::monster_power_effects::install_dampen_power_handlers();
        let player = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player,
            PowerKind::BlackHole,
            3,
        );
        sts2_core::card_effects::apply_power_to_creature(&mut state, player, PowerKind::Dampen, 1);
        let pid = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Dampen)
            .map(|p| p.id)
            .expect("Dampen applied");
        state.powers.get_mut_by_id(pid).expect("Dampen").applier = Some(caster);
        let _ = sts2_core::monster_power_effects::dispatch_after_applied_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
            Some(caster),
        );
        assert!(!state.cards[cache.0.get() as usize - 1].upgraded, "{v:?}");

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
        assert_eq!(
            hp_of(&state, caster),
            0,
            "{v:?}: Black Hole killed the caster"
        );
        assert_eq!(
            power_amount(&state, player, PowerKind::StarNextTurn),
            4,
            "{v:?}: re-upgraded amount"
        );
    }
}

/// C# fires the player's `AfterCardPlayed` powers in application order, and
/// Black Hole's after-play AoE is one of them: with Calamity applied first,
/// Calamity's card lands before the Black Hole kill's Gremlin Horn draw
/// (loadout Regent 1299 v0.107.1: Rust fired Black Hole ahead of every power).
#[test]
fn black_hole_after_play_hit_follows_an_earlier_calamity() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Regent);
        let _target = add_enemy(&mut state, 2, 100);
        let victim = add_enemy(&mut state, 3, 2);
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        fill_draw(&mut state, 1);
        state.player.stars = 5;
        let player = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player,
            PowerKind::Calamity,
            1,
        );
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player,
            PowerKind::BlackHole,
            3,
        );
        let blast = push_hand(&mut state, CardId::GammaBlast, false);

        let _ = sts2_core::combat_loop::play_card(&mut state, blast, Some(0));

        assert_eq!(
            hp_of(&state, victim),
            0,
            "{v:?}: Black Hole killed the victim"
        );
        let hand: Vec<_> = state
            .player
            .hand
            .iter()
            .map(|u| state.cards[u.0.get() as usize - 1])
            .collect();
        assert_eq!(hand.len(), 2, "{v:?}: {hand:?}");
        assert_eq!(
            hand[0].combat_var(sts2_core::cards::VarKey::GeneratedInCombat),
            1,
            "{v:?}: Calamity's card first"
        );
        assert_eq!(
            hand[1].card_id,
            CardId::StrikeIronclad,
            "{v:?}: the Gremlin Horn draw after"
        );
    }
}

/// v0.111.0 Juggling clones the turn's 3rd Attack in `BeforeCardPlayed`,
/// ahead of the card listeners, so an auto-played Stomp's copy is not shaved
/// for that play: it keeps the cost it was cloned with (loadout Ironclad 3165
/// v0.111.0: a Catastrophe-played Stomp's copy cost 0 in Rust, 1 in C#).
#[test]
fn juggling_copy_of_an_auto_played_stomp_keeps_its_cloned_cost() {
    let mut state = fresh(V111, CharacterId::Ironclad);
    let _enemy = add_enemy(&mut state, 2, 200);
    let player = state.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut state, player, PowerKind::Juggling, 1);
    for p in &mut state.powers {
        if p.kind == PowerKind::Juggling {
            p.counter = 2;
        }
    }
    // Two Attacks this turn already shaved the Stomp from 3 to 1.
    state.player.cards_played_this_turn[sts2_core::cards::CardType::Attack.index()] = 2;
    let stomp = new_card(&mut state, CardId::Stomp, false);
    {
        let card = &mut state.cards[stomp.0.get() as usize - 1];
        card.cost = 1;
        card.set_combat_var(sts2_core::cards::VarKey::StompCostReductionThisTurn, 2);
    }
    state.player.draw.push(stomp);

    let _ = sts2_core::combat_loop::auto_play_card_from_draw_top(&mut state, false);

    let copies: Vec<_> = state
        .player
        .hand
        .iter()
        .map(|u| state.cards[u.0.get() as usize - 1])
        .filter(|c| c.card_id == CardId::Stomp)
        .collect();
    assert_eq!(copies.len(), 1, "Juggling copied the 3rd Attack");
    assert_eq!(copies[0].cost, 1, "the copy keeps its cloned cost");
}

/// Scrape filters the cards `CardPileCmd.Draw` returned. Hellraiser
/// auto-plays a drawn Pommel Strike mid-draw, and the card Pommel Strike draws
/// is not part of Scrape's draw, so Scrape keeps it (loadout Ironclad 3138
/// v0.107.1: Rust discarded that Burning Pact).
#[test]
fn scrape_keeps_the_card_a_hellraiser_pommel_strike_drew() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _enemy = add_enemy(&mut state, 2, 200);
        let player = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player,
            PowerKind::Hellraiser,
            1,
        );
        let pommel = new_card(&mut state, CardId::PommelStrike, false);
        let pact = new_card(&mut state, CardId::BurningPact, false);
        state.player.draw.push(pommel);
        state.player.draw.push(pact);
        for _ in 0..4 {
            let defend = new_card(&mut state, CardId::DefendIronclad, false);
            state.player.draw.push(defend);
        }
        let scrape = push_hand(&mut state, CardId::Scrape, false);

        let _ = sts2_core::combat_loop::play_card(&mut state, scrape, Some(0));

        assert!(
            state.player.hand.contains(&pact),
            "{v:?}: Burning Pact stays"
        );
        assert_eq!(
            state.player.hand.len(),
            1,
            "{v:?}: the drawn Defends are discarded"
        );
        assert!(
            state.player.discard.contains(&pommel),
            "{v:?}: Pommel Strike was auto-played"
        );
    }
}

/// Thrash folds an exhausted `CalculatedDamage` attack's `Calculate(null)`: a
/// Supermassive adds 5 + 3 per card generated this combat, not its bare 5
/// (loadout Ironclad 3111 v0.107.1: C# folded 21, Rust 5).
#[test]
fn thrash_folds_a_supermassive_with_its_generated_card_bonus() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _enemy = add_enemy(&mut state, 2, 500);
        state.player.generated_cards_this_combat = 2;
        let thrash = push_hand(&mut state, CardId::Thrash, false);
        let supermassive = new_card(&mut state, CardId::Supermassive, false);
        state.player.hand.push(supermassive);
        let thrash_uid = state.player.hand[usize::from(thrash)];

        let _ = sts2_core::combat_loop::play_card(&mut state, thrash, Some(0));

        assert!(state.player.exhaust.contains(&supermassive), "{v:?}");
        assert_eq!(
            state.cards[thrash_uid.0.get() as usize - 1].base_damage,
            4 + 5 + 3 * 2,
            "{v:?}: Thrash grew by Supermassive's calculated damage"
        );
    }
}

/// A free-this-turn Banshee's Cry shaved by a later Ethereal play folds
/// `Absolute(0) - 2` locally, so Borrowed Time's `+1` global surcharge skips
/// the negative cost and it stays free (loadout Necrobinder 3109 v0.111.0:
/// C# 0, Rust 1).
#[test]
fn borrowed_time_leaves_a_free_banshees_cry_shaved_after_it_free() {
    let mut state = fresh(V111, CharacterId::Necrobinder);
    let cry = new_card(&mut state, CardId::BansheesCry, false);
    state.player.hand.push(cry);
    state.cards[cry.0.get() as usize - 1].set_combat_var(sts2_core::cards::VarKey::FreeThisTurn, 1);
    let player = state.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(
        &mut state,
        player,
        PowerKind::BorrowedTime,
        1,
    );
    assert_eq!(
        sts2_core::hooks::effective_card_cost(&state, cry),
        1,
        "0 + Borrowed Time"
    );

    state.discount_banshees_cry_on_ethereal_play();

    assert_eq!(sts2_core::hooks::effective_card_cost(&state, cry), 0);
}

/// v0.107.1 Scrape keeps a drawn card whose LOCAL cost is 0, and
/// `GetWithModifiers` clamps at 0: a Stomp shaved past 0 this turn stays
/// (loadout Ironclad 3138 v0.107.1: Rust discarded it on its debt).
#[test]
fn scrape_keeps_a_stomp_shaved_past_zero() {
    let mut state = fresh(V107, CharacterId::Ironclad);
    let _enemy = add_enemy(&mut state, 2, 200);
    let stomp = new_card(&mut state, CardId::Stomp, false);
    {
        let card = &mut state.cards[stomp.0.get() as usize - 1];
        card.cost = 0;
        card.set_combat_var(sts2_core::cards::VarKey::StompCostReductionThisTurn, 3);
        card.set_combat_var(sts2_core::cards::VarKey::CostDebtThisTurn, 1);
    }
    state.player.draw.push(stomp);
    fill_draw(&mut state, 3);
    let scrape = push_hand(&mut state, CardId::Scrape, false);

    let _ = sts2_core::combat_loop::play_card(&mut state, scrape, Some(0));

    assert!(state.player.hand.contains(&stomp), "the 0-cost Stomp stays");
}

/// `HighFive.cs:31-35` reads `DynamicVars.Vulnerable` for its
/// `PowerCmd.Apply<VulnerablePower>` only AFTER the Osty attack, so a hit that
/// kills the Dampen caster re-upgrades the card (`DampenPower.AfterDeath` ->
/// `AfterRemoved`) and the survivors get the upgraded 3 Vulnerable. Rust read
/// both values before the attack: loadout Necrobinder 1284-v0.111.0 had C#
/// Vulnerable 3, Rust 2.
#[test]
fn high_five_vulnerable_after_killing_the_dampen_caster_is_upgraded() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Necrobinder);
        let caster = add_enemy(&mut state, 2, 5);
        let survivor = add_enemy(&mut state, 3, 100);
        let _ = state.summon_osty(5).expect("osty summoned");
        let five = new_card(&mut state, CardId::HighFive, true);
        state.player.hand.push(five);
        sts2_core::monster_power_effects::install_dampen_power_handlers();
        let player = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, player, PowerKind::Dampen, 1);
        let pid = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Dampen)
            .map(|p| p.id)
            .expect("Dampen applied");
        state.powers.get_mut_by_id(pid).expect("Dampen").applier = Some(caster);
        let _ = sts2_core::monster_power_effects::dispatch_after_applied_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
            Some(caster),
        );
        assert!(!state.cards[five.0.get() as usize - 1].upgraded, "{v:?}");
        state.player.energy = 3;

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
        assert_eq!(hp_of(&state, caster), 0, "{v:?}: the caster died");
        let vuln: i32 = state
            .creatures
            .iter()
            .find(|c| c.id == survivor.0)
            .expect("survivor")
            .powers
            .iter()
            .filter_map(|u| state.powers.get_by_id(u.0))
            .filter(|p| p.kind == PowerKind::Vulnerable)
            .map(|p| p.amount)
            .sum();
        assert_eq!(vuln, 3, "{v:?}: re-upgraded Vulnerable");
    }
}

/// `CardCmd.Upgrade` returns early while `CombatManager.IsEnding`, so a kill
/// of the LAST enemy (the Dampen caster) removes Dampen without re-upgrading
/// anything. Hand of Greed reads its Gold var after the attack, still at the
/// downgraded 20 (loadout Necrobinder 1085: C# +20 gold, Rust +25).
#[test]
fn hand_of_greed_killing_the_last_dampen_caster_pays_base_gold() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Necrobinder);
        let caster = add_enemy(&mut state, 2, 5);
        let greed = new_card(&mut state, CardId::HandOfGreed, true);
        state.player.hand.push(greed);
        sts2_core::monster_power_effects::install_dampen_power_handlers();
        let player = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, player, PowerKind::Dampen, 1);
        let pid = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Dampen)
            .map(|p| p.id)
            .expect("Dampen applied");
        state.powers.get_mut_by_id(pid).expect("Dampen").applier = Some(caster);
        let _ = sts2_core::monster_power_effects::dispatch_after_applied_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
            Some(caster),
        );
        state.player.energy = 3;
        let gold = state.gold;
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
        assert_eq!(hp_of(&state, caster), 0, "{v:?}: the caster died");
        assert_eq!(state.gold - gold, 20, "{v:?}: Dampen's restore is a no-op");
    }
}

/// Drain Power's `TakeRandom` shuffles the discard's upgradable cards
/// BEFORE `CardCmd.Upgrade`, which returns early while
/// `CombatManager.IsEnding` (CardCmd.cs:267-270). A killing blow on the last
/// enemy therefore spends `n - 1` CombatCardSelection reads but upgrades
/// nothing, and (v0.107.1, whose play loop does not stop on an ending combat;
/// v0.111.0 breaks, CardModel.cs:1898) its replay shuffles the same `n` cards
/// again. Loadout Necrobinder v0.107.1 1340 (a replayed Drain Power, C# 6 reads, Rust 4)
/// and 1316 (Glam, C# 20, Rust 17).
#[test]
fn drain_power_killing_blow_shuffles_but_upgrades_nothing() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Necrobinder);
        let enemy = add_enemy(&mut state, 2, 8);
        let i = push_hand(&mut state, CardId::DrainPower, false);
        let uid = state.player.hand[usize::from(i)];
        state.cards[uid.0.get() as usize - 1].base_replay_count = 1;
        let mut discard = Vec::new();
        for id in [
            CardId::Poke,
            CardId::Entropy,
            CardId::Omnislice,
            CardId::Volley,
        ] {
            let d = new_card(&mut state, id, false);
            state.player.discard.push(d);
            discard.push(d);
        }
        let before = state.run_rng.combat_card_selection.counter();
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(hp_of(&state, enemy), 0, "{v:?}: the last enemy died");
        assert_eq!(
            state.run_rng.combat_card_selection.counter() - before,
            if v == V107 { 6 } else { 3 },
            "{v:?}: each play shuffles all 4 upgradable cards"
        );
        for d in discard {
            assert!(
                !state.cards[d.0.get() as usize - 1].upgraded,
                "{v:?}: no upgrade while combat is ending"
            );
        }
    }
}

// ── 8. Unceasing Top after a top-level auto-play empties the hand ──

/// Hand = one Defend; draw = [Setup Strike, Defend]; Hellraiser + Unceasing
/// Top. Playing the Defend empties the hand: Top draws Setup Strike, which
/// Hellraiser auto-plays. That auto-play's `OnPlayWrapper` ends with
/// `CheckForEmptyHand` (CardModel.cs:1993, v0.111 :2006) outside any card
/// effect (the outer play already left its effect body), so the empty hand
/// fires Top again and the second Defend is drawn.
#[test]
fn unceasing_top_redraws_after_a_hellraiser_auto_play_empties_the_hand() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut state, 2, 100);
        add_relic(&mut state, sts2_core::RelicKind::UnceasingTop);
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Hellraiser, 1);
        let setup = new_card(&mut state, CardId::SetupStrikeCard, false);
        let defend = new_card(&mut state, CardId::DefendIronclad, false);
        state.player.draw.push(setup);
        state.player.draw.push(defend);
        let i = push_hand(&mut state, CardId::DefendIronclad, false);

        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert!(!state.player.hand.contains(&setup), "{v:?}: auto-played");
        assert_eq!(state.player.hand.as_slice(), &[defend], "{v:?}");
        assert!(state.player.draw.is_empty(), "{v:?}");
    }
}

// ── 6. A Calamity-generated Stomp counts the Attack that generated it ──

/// C# `CardModel.OnPlayWrapper` records `History.CardPlayFinished` BEFORE
/// `Hook.AfterCardPlayed` (CardModel.cs:1959-1966, v0.107.1 :1957-1960), so
/// a Stomp that `CalamityPower.AfterCardPlayed` conjures sees the Attack being
/// played in `Stomp.AfterCardEnteredCombat`'s count: the first Attack of the
/// turn yields a Stomp at 3 - 1 = 2. Rust tallied the per-type bucket after the
/// dispatch, so the Stomp cost 3 (sts2-sim-2a, Ironclad v0.111 round).
#[test]
fn calamity_stomp_counts_the_attack_that_generated_it() {
    for v in BOTH {
        let mut found = false;
        for seed in 0..4000_i32 {
            let mut state = fresh(v, CharacterId::Ironclad);
            let _ = add_enemy(&mut state, 2, 500);
            state.run_rng.combat_card_generation = sts2_core::dotnet_rng::Rng::new(seed);
            let p = state.player.creature_uid.expect("player");
            sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Calamity, 1);
            let i = push_hand(&mut state, CardId::StrikeIronclad, false);
            let before: Vec<_> = state.player.hand.to_vec();
            let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
            let Some(stomp) = state.player.hand.iter().copied().find(|u| {
                !before.contains(u) && state.cards[u.0.get() as usize - 1].card_id == CardId::Stomp
            }) else {
                continue;
            };
            let cost = state.cards[stomp.0.get() as usize - 1].cost;
            assert_eq!(cost, 2, "{v:?} seed {seed}: Stomp 3 - 1 Attack played");
            found = true;
            break;
        }
        assert!(found, "{v:?}: no seed generated a Stomp");
    }
}
