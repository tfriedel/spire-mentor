//! Auto-play / replay audit (cluster B): C#-first scenario tests for the
//! auto-play paths (`CardCmd.AutoPlay`, `CardPileCmd.AutoPlayFromDrawPile`,
//! `CardModel.OnPlayWrapper`'s replay loop) and the cards and relics that drive
//! them. Source of truth: the v0.107.1 and v0.111.0 C# decompilation.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::{CardId, EnchantmentId};
use sts2_core::potions::CharacterId;
use sts2_core::{
    CardUid, CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory,
    MonsterId, MonsterKind,
};

const V107: GameVersion = GameVersion::V0_107_1;
const V111: GameVersion = GameVersion::V0_111_0;

fn add_enemy(state: &mut CombatState, raw: u32, hp: i32) -> CreatureUid {
    let id = NonZeroU32::new(raw).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: hp,
        max_hp: hp.max(1),
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

/// One Nibbit at `enemy_hp`; player on 3 energy in the play phase.
fn fresh(version: GameVersion, character: CharacterId, enemy_hp: i32) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let _ = add_enemy(&mut state, 1, enemy_hp);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    state.player.character = Some(character);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn new_card(state: &mut CombatState, id: CardId, upgraded: bool) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, upgraded, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn card_mut(state: &mut CombatState, uid: CardUid) -> &mut sts2_core::cards::Card {
    &mut state.cards[uid.0.get() as usize - 1]
}

// ── Imbued ────────────────────────────────────────────────────────────────

/// `Imbued.AfterAutoPrePlayPhaseEntered` calls `CardCmd.AutoPlay` for each
/// Imbued card (decomp Enchantments/Imbued.cs:15-21, both builds), and
/// `CardCmd.AutoPlay` returns at once when `CombatManager.IsOverOrEnding`
/// (Commands/CardCmd.cs:54-57). So once the first Imbued card's play kills the
/// last enemy, the second Imbued card stays where it is, unplayed.
fn imbued_second_card_after_lethal(version: GameVersion) {
    let mut state = fresh(version, CharacterId::Necrobinder, 1);
    // Capture Spirit (Skill, AnyEnemy, unblockable damage) kills the 1-HP Nibbit.
    let capture = new_card(&mut state, CardId::CaptureSpirit, false);
    card_mut(&mut state, capture).set_enchantment(EnchantmentId::Imbued, 1);
    let defend = new_card(&mut state, CardId::DefendNecrobinder, false);
    card_mut(&mut state, defend).set_enchantment(EnchantmentId::Imbued, 1);
    let filler = new_card(&mut state, CardId::StrikeNecrobinder, false);
    // Draw pile (top first): filler, Capture Spirit, Defend.
    state.player.draw = [filler, capture, defend].into_iter().collect();

    sts2_core::enchantment_hooks::dispatch_after_player_turn_start(&mut state);

    assert_eq!(
        state.creatures[0].current_hp, 0,
        "Capture Spirit killed the Nibbit"
    );
    assert_eq!(
        state.player.block, 0,
        "the second Imbued Defend never played"
    );
    assert!(
        state.player.draw.contains(&defend),
        "the unplayed Imbued Defend stays in the draw pile"
    );
    assert_eq!(state.combat_log.player_card_plays_this_combat, 1);
}

#[test]
fn imbued_stops_once_the_combat_is_ending_v0107() {
    imbued_second_card_after_lethal(V107);
}

#[test]
fn imbued_stops_once_the_combat_is_ending_v0111() {
    imbued_second_card_after_lethal(V111);
}

fn add_player_power(state: &mut CombatState, id: u32, kind: sts2_core::PowerKind, amount: i32) {
    let pid = NonZeroU32::new(id).expect("nz");
    state.powers.push(sts2_core::Power::new(pid, kind, amount));
    state.player.powers.push(sts2_core::PowerUid(pid));
}

fn push_hand(state: &mut CombatState, id: CardId, upgraded: bool) -> u8 {
    let uid = new_card(state, id, upgraded);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

// ── Havoc ─────────────────────────────────────────────────────────────────

/// Havoc auto-plays a Strike that Hidden Gem gave Replay 2 into a 1-HP enemy.
/// v0.107.1's `OnPlayWrapper` loop has no combat-ending stop (CardModel.cs:
/// 1905-1960), so all three iterations record `CardPlayFinished`; v0.111.0
/// breaks at the top of the second (CardModel.cs:1896-1901).
fn havoc_replayed_strike_kills(version: GameVersion) -> CombatState {
    let mut state = fresh(version, CharacterId::Ironclad, 1);
    let strike = new_card(&mut state, CardId::StrikeIronclad, false);
    card_mut(&mut state, strike).base_replay_count = 2;
    state.player.draw = [strike].into_iter().collect();
    let havoc = push_hand(&mut state, CardId::Havoc, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, havoc, None);
    state
}

#[test]
fn havoc_replays_after_lethal_v0107() {
    let state = havoc_replayed_strike_kills(V107);
    assert_eq!(state.creatures[0].current_hp, 0);
    assert_eq!(state.combat_log.player_card_plays_this_combat, 4);
}

#[test]
fn havoc_replay_stops_after_lethal_v0111() {
    let state = havoc_replayed_strike_kills(V111);
    assert_eq!(state.creatures[0].current_hp, 0);
    assert_eq!(state.combat_log.player_card_plays_this_combat, 2);
}

/// `GetResultLocationForCardPlay` returns `PileType.None` for a Power before it
/// looks at `ExhaustOnNextPlay` (CardModel.cs:2084-2096 v0.111.0, same order in
/// v0.107.1), so a Havoc'd Inflame leaves combat instead of exhausting: Feel No
/// Pain gains no block.
#[test]
fn havoc_power_card_leaves_combat_without_exhausting() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Ironclad, 50);
        add_player_power(&mut state, 9001, sts2_core::PowerKind::FeelNoPain, 3);
        let inflame = new_card(&mut state, CardId::Inflame, false);
        state.player.draw = [inflame].into_iter().collect();
        let havoc = push_hand(&mut state, CardId::Havoc, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, havoc, None);
        assert_eq!(
            state.player_power_amount(sts2_core::PowerKind::Strength),
            2,
            "{version:?}: Inflame resolved"
        );
        assert!(!state.player.exhaust.contains(&inflame), "{version:?}");
        assert!(!state.player.discard.contains(&inflame), "{version:?}");
        assert!(!state.player.play.contains(&inflame), "{version:?}");
        assert_eq!(
            state.player.block, 0,
            "{version:?}: no exhaust, no Feel No Pain"
        );
    }
}

// ── AfterCardPlayed after the kill ───────────────────────────────────────

/// `Hook.AfterCardPlayed` walks `combatState.IterateHookListeners()` directly,
/// NOT the `IterateCombatHookListeners` that goes quiet once the combat
/// `IsOverOrEnding` (Hook.cs:422-438, both builds), and `OnPlayWrapper` calls
/// it while `IsInProgress` (still true while ending). So Rage gains block from
/// the play that killed the last enemy, and in v0.107.1 (no loop stop) from
/// both Hidden Gem replays after it: 9. v0.111.0 breaks before the first
/// replay (CardModel.cs:1896-1901): 3.
#[test]
fn rage_counts_a_lethal_replayed_strike_per_play() {
    for (version, block) in [(V107, 9), (V111, 3)] {
        let mut state = fresh(version, CharacterId::Ironclad, 1);
        add_player_power(&mut state, 9002, sts2_core::PowerKind::Rage, 3);
        let strike = push_hand(&mut state, CardId::StrikeIronclad, false);
        let uid = state.player.hand[usize::from(strike)];
        card_mut(&mut state, uid).base_replay_count = 2;
        let _ = sts2_core::combat_loop::play_card(&mut state, strike, Some(0));
        assert_eq!(state.creatures[0].current_hp, 0, "{version:?}");
        assert_eq!(state.player.block, block, "{version:?}");
    }
}

// ── Lingering ExhaustOnNextPlay ──────────────────────────────────────────

fn smogged_defy(state: &mut CombatState) -> CardUid {
    let defy = new_card(state, CardId::Defy, false);
    assert!(card_mut(state, defy).set_affliction(
        sts2_core::card_afflictions::CardAffliction::new(
            sts2_core::card_afflictions::CardAfflictionKind::Smog,
            1,
        )
    ));
    defy
}

/// `CardPileCmd.AutoPlayFromDrawPile` stamps `ExhaustOnNextPlay = forceExhaust`
/// (CardPileCmd.cs:1177 v0.111.0); Havoc passes `true`. Smoggy vetoes the
/// Smog-afflicted Defy (SmoggyPower.cs:64-71), so `CardCmd.AutoPlay` sends it
/// through `MoveToResultPileWithoutPlaying`, which exhausts it WITHOUT clearing
/// the flag (CardModel.cs:2104-2120; only `GetResultLocationForCardPlay`
/// :2090-2092 and `EndOfTurnCleanup` :1613 clear it). Eidolon then auto-plays
/// the Ethereal Defy out of the exhaust pile (Eidolon.cs:27-31), Smoggy vetoes
/// it again, and the still-set flag exhausts it once more. The control (Defy
/// exhausted without the Havoc stamp) lands in discard instead.
///
/// The flag can only linger on an exhausted card: every real play clears it
/// before resolving (CardModel.cs:1881), and no v0.107.1/v0.111.0 effect moves
/// a card out of the exhaust pile except by auto-play (Eidolon, Knife Trap,
/// Howl from Beyond). So a MANUAL play of a lingering-flag card, and a Cascade
/// (`forceExhaust: false`, draw pile only) overwriting one, cannot happen.
#[test]
fn eidolon_reblocked_card_keeps_havocs_lingering_exhaust() {
    // Havoc → blocked → exhaust (flag set) → Eidolon → blocked → exhaust.
    let mut state = fresh(V111, CharacterId::Necrobinder, 50);
    add_player_power(&mut state, 9003, sts2_core::PowerKind::Smoggy, 1);
    let defy = smogged_defy(&mut state);
    state.player.draw = [defy].into_iter().collect();
    let havoc = push_hand(&mut state, CardId::Havoc, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, havoc, None);
    assert!(
        state.player.exhaust.contains(&defy),
        "Havoc's blocked Defy exhausts"
    );
    let eidolon = push_hand(&mut state, CardId::Eidolon, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, eidolon, None);
    assert!(
        state.player.exhaust.contains(&defy),
        "the lingering ExhaustOnNextPlay exhausts the re-blocked Defy again"
    );
    assert!(!state.player.discard.contains(&defy));

    // Control: the same Defy exhausted without the Havoc stamp.
    let mut state = fresh(V111, CharacterId::Necrobinder, 50);
    add_player_power(&mut state, 9003, sts2_core::PowerKind::Smoggy, 1);
    let defy = smogged_defy(&mut state);
    state.player.exhaust = [defy].into_iter().collect();
    let eidolon = push_hand(&mut state, CardId::Eidolon, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, eidolon, None);
    assert!(
        state.player.discard.contains(&defy),
        "no flag: the blocked Defy goes to discard"
    );
}

// ── Nostalgia ─────────────────────────────────────────────────────────────

/// `NostalgiaPower.ModifyCardPlayResultLocation` (NostalgiaPower.cs:16-50
/// v0.111.0) counts this turn's `CardPlaysStarted` Attack/Skill entries, and
/// `OnPlayWrapper` resolves the result location ONCE, before the play loop
/// records this card's own `CardPlayStarted` (CardModel.cs:1881 vs :1925). So
/// the first Attack/Skill of the turn goes to draw-top even when Hidden Gem
/// replays it and its `OnPlay` opens a choice each time (Armaments' upgrade
/// pick).
#[test]
fn nostalgia_redirects_a_replayed_choice_card_played_first() {
    for version in [V107, V111] {
        for replays in [0, 1] {
            let mut state = fresh(version, CharacterId::Ironclad, 50);
            add_player_power(&mut state, 9004, sts2_core::PowerKind::Nostalgia, 1);
            let arm = push_hand(&mut state, CardId::Armaments, false);
            let arm_uid = state.player.hand[usize::from(arm)];
            card_mut(&mut state, arm_uid).base_replay_count = replays;
            let _ = push_hand(&mut state, CardId::StrikeIronclad, false);
            let _ = push_hand(&mut state, CardId::DefendIronclad, false);
            let _ = push_hand(&mut state, CardId::Bash, false);
            let _ = sts2_core::combat_loop::play_card(&mut state, arm, None);
            let mut guard = 0;
            while state.pending_choice.is_some() {
                let _ = sts2_core::resolve_choice_atomic_apply(&mut state, &[0]);
                guard += 1;
                assert!(guard < 8, "choice loop");
            }
            assert_eq!(guard, replays + 1, "{version:?}: one upgrade pick per play");
            assert_eq!(
                state.player.draw.first().copied(),
                Some(arm_uid),
                "{version:?} replays={replays}: Armaments goes to draw-top (discard={:?})",
                state.player.discard
            );
        }
    }
}

// ── Reaper Form on an auto-played Osty attack ────────────────────────────

fn enemy_power(state: &CombatState, enemy: usize, kind: sts2_core::PowerKind) -> i32 {
    state.creatures[enemy]
        .powers
        .iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.amount)
        .sum()
}

/// `ReaperFormPower.AfterDamageGiven` (ReaperFormPower.cs:57-63 v0.111.0)
/// counts the owner's PET as a dealer and gates on `result.TotalDamage` (blocked
/// plus unblocked), so a Havoc'd Right Hand Hand's 4-damage Osty hit into 3
/// block dooms the Nibbit by 4. A Hidden Gem replay hits again: 8.
#[test]
fn reaper_form_dooms_from_a_havocd_osty_attack() {
    for version in [V107, V111] {
        for (replays, doom) in [(0, 4), (1, 8)] {
            let mut state = fresh(version, CharacterId::Necrobinder, 50);
            let _ = state.summon_osty(10);
            state.creatures[0].block = 3;
            add_player_power(&mut state, 9005, sts2_core::PowerKind::ReaperForm, 1);
            let rhh = new_card(&mut state, CardId::RightHandHand, false);
            card_mut(&mut state, rhh).base_replay_count = replays;
            state.player.draw = [rhh].into_iter().collect();
            let havoc = push_hand(&mut state, CardId::Havoc, false);
            let _ = sts2_core::combat_loop::play_card(&mut state, havoc, None);
            assert_eq!(
                enemy_power(&state, 0, sts2_core::PowerKind::Doom),
                doom,
                "{version:?} replays={replays}"
            );
            assert!(
                state.player.exhaust.contains(&rhh),
                "{version:?}: Havoc exhausts it"
            );
        }
    }
}

// ── Envenom → Outbreak on a replayed auto-play (v0.107.1) ────────────────

/// Each Hidden Gem replay of a Havoc'd Strike is its own powered hit, so
/// `EnvenomPower.AfterDamageGiven` (EnvenomPower.cs:20-26) applies Poison three
/// times with the player as applier, and `OutbreakPower.AfterPowerAmountChanged`
/// (OutbreakPower.cs:30-50 v0.107.1) fires its 11 all-enemy damage on the third.
#[test]
fn envenom_poisons_feed_outbreak_across_havoc_replays_v0107() {
    let mut state = fresh(V107, CharacterId::Silent, 100);
    add_player_power(&mut state, 9006, sts2_core::PowerKind::Envenom, 1);
    add_player_power(&mut state, 9007, sts2_core::PowerKind::Outbreak, 11);
    let strike = new_card(&mut state, CardId::StrikeSilent, false);
    card_mut(&mut state, strike).base_replay_count = 2;
    state.player.draw = [strike].into_iter().collect();
    let havoc = push_hand(&mut state, CardId::Havoc, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, havoc, None);
    assert_eq!(enemy_power(&state, 0, sts2_core::PowerKind::Poison), 3);
    assert_eq!(state.creatures[0].current_hp, 100 - 3 * 6 - 11);
}

// ── Inferno on an auto-played self-damage card ───────────────────────────

/// A Havoc'd Bloodletting damages its owner through `CreatureCmd.Damage`
/// (Bloodletting.cs:31) during the player's side, so
/// `InfernoPower.AfterDamageReceived` (InfernoPower.cs:37-49) fans its Amount
/// to every hittable enemy; Hidden Gem's replay fans it again.
#[test]
fn inferno_fans_on_a_havocd_bloodletting_per_replay() {
    for version in [V107, V111] {
        for (replays, fans) in [(0, 1), (1, 2)] {
            let mut state = fresh(version, CharacterId::Ironclad, 100);
            add_player_power(&mut state, 9008, sts2_core::PowerKind::Inferno, 6);
            let bl = new_card(&mut state, CardId::Bloodletting, false);
            card_mut(&mut state, bl).base_replay_count = replays;
            state.player.draw = [bl].into_iter().collect();
            let havoc = push_hand(&mut state, CardId::Havoc, false);
            let _ = sts2_core::combat_loop::play_card(&mut state, havoc, None);
            assert_eq!(state.player.current_hp, 80 - 3 * fans, "{version:?}");
            assert_eq!(state.creatures[0].current_hp, 100 - 6 * fans, "{version:?}");
        }
    }
}

// ── Necro Mastery on an auto-played Osty kill ────────────────────────────

/// A Havoc'd Sacrifice kills the Osty through `CreatureCmd.Kill`
/// (Sacrifice.cs OnPlay), whose `AfterCurrentHpChanged(-currentHp)` makes
/// `NecroMasteryPower` (NecroMasteryPower.cs:17-23) deal the Osty's 10 HP times
/// Amount to every hittable enemy.
#[test]
fn necro_mastery_reflects_a_havocd_sacrifice() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Necrobinder, 100);
        let _ = state.summon_osty(10);
        add_player_power(&mut state, 9009, sts2_core::PowerKind::NecroMastery, 2);
        let sac = new_card(&mut state, CardId::Sacrifice, false);
        state.player.draw = [sac].into_iter().collect();
        let havoc = push_hand(&mut state, CardId::Havoc, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, havoc, None);
        assert_eq!(state.creatures[0].current_hp, 100 - 20, "{version:?}");
    }
}

// ── Void Form counts manual series only ──────────────────────────────────

/// `VoidFormPower.AfterCardPlayed` (VoidFormPower.cs:100-112 v0.111.0) counts a
/// play only when `!cardPlay.IsAutoPlay && cardPlay.IsLastInSeries`: Havoc's
/// auto-played Strike and a Hidden Gem replay of a Defend spend no free play.
/// With Void Form 2: Havoc (free), the replayed Defend (free), then a Defend
/// that pays 1.
#[test]
fn void_form_skips_auto_plays_and_replays() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Ironclad, 100);
        add_player_power(&mut state, 9010, sts2_core::PowerKind::VoidForm, 2);
        let strike = new_card(&mut state, CardId::StrikeIronclad, false);
        state.player.draw = [strike].into_iter().collect();
        let havoc = push_hand(&mut state, CardId::Havoc, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, havoc, None);
        assert_eq!(state.player.energy, 3, "{version:?}: Havoc free");
        let d1 = push_hand(&mut state, CardId::DefendIronclad, false);
        let d1_uid = state.player.hand[usize::from(d1)];
        card_mut(&mut state, d1_uid).base_replay_count = 1;
        let _ = sts2_core::combat_loop::play_card(&mut state, d1, None);
        assert_eq!(state.player.energy, 3, "{version:?}: replayed Defend free");
        let d2 = push_hand(&mut state, CardId::DefendIronclad, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, d2, None);
        assert_eq!(state.player.energy, 2, "{version:?}: third series pays");
    }
}

// ── Void Form applied mid-batch ──────────────────────────────────────────

fn add_relic(state: &mut CombatState, kind: sts2_core::RelicKind) {
    let id = NonZeroU32::new(u32::try_from(state.relics.len() + 1).expect("u32")).expect("nz");
    state.relics.push(sts2_core::Relic {
        id,
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

/// `VoidFormPower.BeforeApplied` maxes `cardsPlayedThisTurn`
/// (`HideTemporaryZeroCostVisual`, VoidFormPower.cs:60-75 / :126-133 v0.111.0,
/// same in v0.107.1), so a Void Form that lands in the middle of a Cascade
/// batch frees nothing for the rest of the turn: the Bludgeon Cascade plays
/// after it keeps `EnergyValue` 3 (`CardCmd.AutoPlay`, CardCmd.cs:124-126) and
/// Intimidating Helmet (EnergyValue >= 2, IntimidatingHelmet.cs) grants for
/// Cascade (X = 2), Void Form (3) and Bludgeon (3): 12 block.
#[test]
fn void_form_applied_mid_cascade_frees_nothing() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Defect, 500);
        state.player.energy = 2;
        add_relic(&mut state, sts2_core::RelicKind::IntimidatingHelmet);
        let vf = new_card(&mut state, CardId::VoidForm, false);
        let bludgeon = new_card(&mut state, CardId::Bludgeon, false);
        state.player.draw = [vf, bludgeon].into_iter().collect();
        let cascade = push_hand(&mut state, CardId::Cascade, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, cascade, None);
        assert_eq!(
            state.player_power_amount(sts2_core::PowerKind::VoidForm),
            2,
            "{version:?}: Void Form applied"
        );
        assert!(
            state.creatures[0].current_hp < 500,
            "{version:?}: Bludgeon played"
        );
        assert_eq!(state.player.block, 12, "{version:?}");
    }
}

// ── Dealer-side AfterDamageGiven order ───────────────────────────────────

fn enemy_power_kinds(state: &CombatState, enemy: usize) -> Vec<sts2_core::PowerKind> {
    state.creatures[enemy]
        .powers
        .iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .map(|p| p.kind)
        .collect()
}

/// `Hook.AfterDamageGiven` walks `IterateHookListeners` (Hook.cs), the owner's
/// powers in the order they were gained, so Envenom's Poison
/// (EnvenomPower.cs:20-26) and Reaper Form's Doom (ReaperFormPower.cs:57-63)
/// land on the target in that order.
#[test]
fn envenom_and_reaper_form_fire_in_the_order_they_were_gained() {
    use sts2_core::PowerKind::{Doom, Envenom, Poison, ReaperForm};
    for version in [V107, V111] {
        for (first, second, expected) in [
            (Envenom, ReaperForm, [Poison, Doom]),
            (ReaperForm, Envenom, [Doom, Poison]),
        ] {
            let mut state = fresh(version, CharacterId::Silent, 100);
            let player = state.player.creature_uid.expect("player");
            for kind in [first, second] {
                sts2_core::card_effects::apply_power_to_creature(&mut state, player, kind, 1);
            }
            let strike = push_hand(&mut state, CardId::StrikeSilent, false);
            let _ = sts2_core::combat_loop::play_card(&mut state, strike, Some(0));
            assert_eq!(
                enemy_power_kinds(&state, 0),
                expected.to_vec(),
                "{version:?}: {first:?} then {second:?}"
            );
        }
    }
}

// ── Poison ticks have no dealer ──────────────────────────────────────────

/// `PoisonPower.Trigger` damages its owner with `dealer: null`
/// (PoisonPower.cs:66-79 v0.111.0), so v0.111.0 Outbreak's trigger is recorded
/// as a source-less `DamageReceivedEntry` ("took"), not one the enemy dealt to
/// itself.
#[test]
fn outbreak_poison_trigger_has_no_dealer_v0111() {
    let mut state = fresh(V111, CharacterId::Silent, 100);
    state.player.energy = 3;
    state.combat_log.event_log.set_recording(true);
    let outbreak = push_hand(&mut state, CardId::OutbreakCard, false);
    let _ = sts2_core::combat_loop::play_card(&mut state, outbreak, None);
    assert_eq!(
        state.creatures[0].current_hp,
        100 - 9,
        "Outbreak's Poison ticked"
    );
    let ticks: Vec<_> = state
        .combat_log
        .event_log
        .rows()
        .iter()
        .filter_map(|r| match r.entry {
            sts2_core::event_log::EventLogEntry::DamageReceived { dealer, .. } => Some(dealer),
            _ => None,
        })
        .collect();
    assert_eq!(ticks, vec![None], "one source-less Poison tick");
}

/// The turn-start Poison tick (`PoisonPower.AfterSideTurnStart` → `Trigger`,
/// PoisonPower.cs:55-79 v0.111.0; inline in v0.107.1) is source-less too.
#[test]
fn turn_start_poison_tick_has_no_dealer() {
    for version in [V107, V111] {
        let mut state = fresh(version, CharacterId::Silent, 100);
        let enemy = state.enemy_uids[0];
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            enemy,
            sts2_core::PowerKind::Poison,
            4,
        );
        state.combat_log.event_log.set_recording(true);
        sts2_core::hooks::dispatch_after_side_turn_start(
            &mut state,
            &sts2_core::hooks::TurnSideCtx {
                side: sts2_core::Side::Enemy,
            },
        );
        assert_eq!(state.creatures[0].current_hp, 96, "{version:?}");
        let dealers: Vec<_> = state
            .combat_log
            .event_log
            .rows()
            .iter()
            .filter_map(|r| match r.entry {
                sts2_core::event_log::EventLogEntry::DamageReceived { dealer, .. } => Some(dealer),
                _ => None,
            })
            .collect();
        assert_eq!(dealers, vec![None], "{version:?}");
    }
}

// ── A setup pause cancelled by an early side-turn-start kill ─────────────

/// Tools of the Trade's discard select pauses `SetupPlayerTurn`, so C#'s
/// `StartTurn` runs `Hook.AfterSideTurnStart` early (CombatManager.cs:514-524
/// v0.107.1). Noxious Fumes' poison is Outbreak's third and its damage kills
/// the last enemy: `ActionQueueSet.CombatEnded` (ActionQueueSet.cs:354-371)
/// cancels the queued `CombatPlayPhaseOnly` hook action the select waits on
/// (HookPlayerChoiceContext.cs:150-167), so the screen never opens and the rest
/// of setup (Pendulum's tick) never runs (loadout Silent 6808 v0.107.1).
#[test]
fn setup_pause_is_cancelled_when_the_early_side_turn_start_ends_combat_v0107() {
    let mut state = fresh(V107, CharacterId::Silent, 1);
    add_player_power(&mut state, 9101, sts2_core::PowerKind::ToolsOfTheTrade, 1);
    add_player_power(&mut state, 9102, sts2_core::PowerKind::NoxiousFumes, 2);
    add_player_power(&mut state, 9103, sts2_core::PowerKind::Outbreak, 11);
    state
        .powers
        .iter_mut()
        .find(|p| p.kind == sts2_core::PowerKind::Outbreak)
        .expect("outbreak")
        .counter = 2;
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: sts2_core::RelicKind::Pendulum,
        counter: 1,
        is_wax: false,
        is_melted: false,
    });
    let draw: Vec<CardUid> = (0..8)
        .map(|_| new_card(&mut state, CardId::StrikeSilent, false))
        .collect();
    state.player.draw = draw.into_iter().collect();
    state.phase = CombatPhase::PlayerTurnStart;
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);
    assert!(state.pending_choice.is_none(), "the select never opens");
    assert!(state.is_over && state.player_won);
    let pendulum = state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::Pendulum)
        .expect("pendulum");
    assert_eq!(pendulum.counter, 1, "Pendulum never ticks");
}
