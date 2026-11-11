//! Enchantment-granted keywords reach combat (GH #1021).
//!
//! C# v0.107.1 `Models.Enchantments`: `Steady.OnEnchant` adds Retain,
//! `RoyallyApproved.OnEnchant` adds Innate and Retain, `Goopy.OnEnchant`
//! adds Exhaust, and `SoulsPower.OnEnchant` removes Exhaust. The keyword
//! lives on the card, so it holds in every combat the card enters.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::card_effects::{card_exhausts_inst, card_innate_inst, card_retain_inst};
use sts2_core::cards::{Card, CardId, CardTag, CardTags, CardType, EnchantmentId, TargetType};
use sts2_core::{CombatPhase, CombatState, Creature, IntentHistory, MonsterId, MonsterKind};

fn defend() -> Card {
    Card {
        uid: sts2_core::CardUid::new(1).expect("nz"),
        card_id: CardId::DefendIronclad,
        cost: sts2_core::CardEnergyCost::new(0),
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        base_block: 5,
        tags: CardTags::NONE.with(CardTag::Defend),
        ..Card::pod_defaults()
    }
}

fn enchanted(id: EnchantmentId) -> Card {
    let mut card = defend();
    card.set_enchantment(id, 1);
    card
}

#[test]
fn steady_retains() {
    assert!(!card_retain_inst(&defend()));
    assert!(card_retain_inst(&enchanted(EnchantmentId::Steady)));
    assert!(!card_innate_inst(&enchanted(EnchantmentId::Steady)));
}

#[test]
fn royally_approved_is_innate_and_retains() {
    let card = enchanted(EnchantmentId::RoyallyApproved);
    assert!(card_retain_inst(&card));
    assert!(card_innate_inst(&card));
    assert!(!card_innate_inst(&defend()));
}

#[test]
fn goopy_exhausts() {
    assert!(!card_exhausts_inst(&defend()));
    assert!(card_exhausts_inst(&enchanted(EnchantmentId::Goopy)));
}

#[test]
fn other_enchantments_leave_the_keywords_alone() {
    for id in [
        EnchantmentId::Sharp,
        EnchantmentId::Nimble,
        EnchantmentId::Swift,
    ] {
        let card = enchanted(id);
        assert!(!card_retain_inst(&card), "{id:?}");
        assert!(!card_innate_inst(&card), "{id:?}");
        assert!(!card_exhausts_inst(&card), "{id:?}");
    }
}

/// Through the real `play_card` path: a played Goopy Defend goes to the
/// exhaust pile, not the discard pile.
#[test]
fn a_played_goopy_card_is_exhausted() {
    content_dispatch::ensure_installed();
    let mut state = CombatState::default();
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 9999,
        max_hp: 9999,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(sts2_core::CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    let card = enchanted(EnchantmentId::Goopy);
    let uid = card.uid;
    state.cards.push(card);
    state.player.hand.push(uid);

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    assert_eq!(state.player.block, 5);
    assert!(
        state.player.exhaust.contains(&uid),
        "Goopy card not exhausted"
    );
    assert!(!state.player.discard.contains(&uid));
}
