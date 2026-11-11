//! Full-path (`combat_loop::play_card` + the real content registry)
//! regression test for the OnPlay-choice ordering fix (Silent fuzz seed 442).
//!
//! C# `CardModel.OnPlayWrapper` `await`s a card's `OnPlay` selector INSIDE
//! `OnPlay` (CardModel.cs:1498), so every consequence of the selection lands
//! BEFORE `Hook.AfterCardPlayed`. DAGGER_THROW opens a `FromHandForDiscard`
//! grid; the discard fires Tingsha's `AfterCardDiscarded` 3-damage tick. An
//! enemy's CurlUp grants its block in `AfterCardPlayed`, so in C# the Tingsha
//! damage hits the (still-blockless) enemy HP and the CurlUp block lands at its
//! FULL amount afterward. RS used to leave the discard choice pending past
//! `after_card_played`, so CurlUp's block was already up when the drained
//! discard's Tingsha damage landed — it hit block instead of HP (seed 442:
//! monster block cs=14 rs=11, hp cs=114 rs=117). `play_card` now SURFACES the
//! OnPlay choice (bead `sts2-rs-mcts-surface-onplay-select`) and defers the
//! play tail until it is answered, so the discard + its Tingsha tick still land
//! BEFORE `after_card_played` — matching C#'s await-the-selector-inside-OnPlay
//! ordering. This test drives the surfaced choice first-legal to reproduce it.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::{Card, CardId, CardType, TargetType};
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, IntentHistory, MonsterId, MonsterKind, Power,
    PowerKind, PowerUid, Relic, RelicKind,
};

fn push_card(
    state: &mut CombatState,
    id: CardId,
    card_type: CardType,
    target_type: TargetType,
    dmg: i32,
) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    state.cards.push(Card {
        uid,
        card_id: id,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type,
        target_type,
        upgraded: false,
        base_damage: dmg,
        base_block: 0,
        ..Card::pod_defaults()
    });
    uid
}

/// DAGGER_THROW (deal damage, draw 1, discard 1) played into a CurlUp enemy
/// while the player holds Tingsha: the discard's Tingsha tick must land BEFORE
/// CurlUp's `after_card_played` block grant, so CurlUp's block ends at its full
/// 14 (not 11) and the Tingsha 3 comes off the enemy's HP.
#[test]
fn dagger_throw_discard_tingsha_lands_before_curl_up_block() {
    content_dispatch::ensure_installed();
    // The relic listener registry (Tingsha's `after_card_discarded`) installs
    // separately from the card registry.
    sts2_core::relic_effects::install_default_registry();
    // Defensive: the CurlUp AfterDamageReceived latch hook is installed when a
    // LouseProgenitor AI is built; this test mounts CurlUp directly, so install
    // it explicitly.
    sts2_content::monsters::act2_hive::install_curl_up_handlers();

    let mut state = CombatState::default();

    // Enemy with CurlUp(14), finite HP, no block.
    let eid = NonZeroU32::new(1).expect("nz");
    let curl_pid = NonZeroU32::new(50).expect("nz");
    state.powers.push(Power {
        id: curl_pid,
        kind: PowerKind::CurlUp,
        amount: 14,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    let mut epowers: SmallVec<[PowerUid; 8]> = SmallVec::new();
    epowers.push(PowerUid(curl_pid));
    state.creatures.push(Creature {
        id: eid,
        current_hp: 100,
        max_hp: 100,
        block: 0,
        powers: epowers,
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(eid));

    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 99;
    state.player.max_energy = 99;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = sts2_core::Side::Player;

    // Player holds Tingsha (3 damage on discard).
    state.relics.push(Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: RelicKind::Tingsha,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });

    // Hand: DAGGER_THROW + a junk card to discard. Draw pile: one junk card so
    // DAGGER_THROW's "draw 1" succeeds and the discard grid has options.
    let dagger = push_card(
        &mut state,
        CardId::DaggerThrow,
        CardType::Attack,
        TargetType::Enemy,
        9,
    );
    let junk_hand = push_card(
        &mut state,
        CardId::DefendSilent,
        CardType::Skill,
        TargetType::Self_,
        0,
    );
    let junk_draw = push_card(
        &mut state,
        CardId::DefendSilent,
        CardType::Skill,
        TargetType::Self_,
        0,
    );
    state.player.hand.push(dagger);
    state.player.hand.push(junk_hand);
    state.player.draw.push(junk_draw);

    let before_hp = state.creatures[0].current_hp;

    // Play DAGGER_THROW (hand index 0) targeting the only enemy. Its OnPlay
    // opens an interactive discard select which is now SURFACED (bead
    // sts2-rs-mcts-surface-onplay-select) rather than drained inline — the play
    // tail (incl. CurlUp's AfterCardPlayed block grant) is deferred until the
    // pick lands.
    let step = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert_eq!(
        step,
        sts2_core::StepResult::PendingChoice,
        "DAGGER_THROW's discard select must surface for the agent to pick"
    );
    assert!(
        state.pending_choice.is_some() && state.pending_play.is_some(),
        "the play tail is deferred behind the open choice"
    );

    // Answer the discard select first-legal (discard option 0). This re-enters
    // the deferred finish tail: the discard's Tingsha tick fires, THEN CurlUp's
    // AfterCardPlayed block grant — the exact order the former inline drain
    // produced.
    let selection = sts2_core::legal_actions(&state)
        .into_iter()
        .find_map(|a| match a {
            sts2_core::Action::ResolveChoice { selection } if !selection.is_empty() => {
                Some(selection)
            }
            _ => None,
        })
        .expect("a discard selection must be offered");
    let _ = sts2_core::apply(&mut state, sts2_core::Action::ResolveChoice { selection });

    let enemy = &state.creatures[0];
    // CurlUp granted its FULL 14 block — Tingsha's 3 did NOT eat into it.
    assert_eq!(
        enemy.block, 14,
        "CurlUp must grant its full block; the discard's Tingsha tick lands on HP first"
    );
    // CurlUp consumed itself.
    assert!(
        !enemy.powers.iter().any(|u| u.0 == curl_pid),
        "CurlUp removes itself after granting block"
    );
    // The enemy lost HP to BOTH the DAGGER_THROW hit AND the Tingsha 3 (the 3
    // came off HP, not the just-granted block). With base 9 + Tingsha 3 = 12.
    assert_eq!(
        before_hp - enemy.current_hp,
        12,
        "DAGGER_THROW (9) + Tingsha (3) both hit HP before CurlUp blocked"
    );
}
