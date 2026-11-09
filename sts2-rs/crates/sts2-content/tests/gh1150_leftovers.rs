//! GH #1150 leftovers: the parity gaps #1163 did not cover. Each test cites the
//! C# it pins, for both builds unless the decomps differ.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CardUid, CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory,
    MonsterId, MonsterKind, Power, PowerKind, PowerUid,
};

const BOTH: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

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

fn new_card(state: &mut CombatState, id: CardId, upgraded: bool) -> CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = CardUid::new(raw).expect("nz");
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

fn add_player_power(state: &mut CombatState, raw: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state.player.powers.push(PowerUid(pid));
}

fn hp_of(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.current_hp)
}

fn player_power(state: &CombatState, kind: PowerKind) -> i32 {
    state.player_power_amount(kind)
}

fn card(state: &CombatState, uid: CardUid) -> &sts2_core::cards::Card {
    &state.cards[uid.0.get() as usize - 1]
}

// ── Multi-command attacks: Echoing Slash is ONE attack command ──

/// `EchoingSlash.cs:25-43` opens one `AttackContext` around every wave
/// (`CreatureCmd.Damage` per wave, not `DamageCmd.Attack`), so `BeforeAttack`
/// and `AfterAttack` run once for the whole card: Vigor (`VigorPower.cs`) and
/// Gigantification (`GigantificationPower.cs`) boost every wave and are spent
/// once, after the last. Only v0.107.1 Mad Science issues several commands in
/// one play; no other card does in either build.
#[test]
fn echoing_slash_vigor_and_gigantification_boost_every_wave() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Silent);
        // (10 + 5 Vigor) * 3 Gigantification = 45: wave 1 kills the first
        // enemy, so a second wave runs.
        let dying = add_enemy(&mut state, 2, 40);
        let survivor = add_enemy(&mut state, 3, 200);
        add_player_power(&mut state, 900, PowerKind::Vigor, 5);
        add_player_power(&mut state, 901, PowerKind::Gigantification, 1);
        let i = push_hand(&mut state, CardId::EchoingSlash, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert_eq!(hp_of(&state, dying), 0, "{v:?}");
        assert_eq!(
            hp_of(&state, survivor),
            200 - 45 - 45,
            "{v:?}: both waves boosted"
        );
        assert_eq!(
            player_power(&state, PowerKind::Vigor),
            0,
            "{v:?}: Vigor spent"
        );
        assert_eq!(
            player_power(&state, PowerKind::Gigantification),
            0,
            "{v:?}: Gigantification spent"
        );
    }
}

// ── In-combat upgrades are no-ops once combat is ending ──

/// `KnifeTrap.cs` upgrades each exhausted Shiv with `CardCmd.Upgrade` right
/// before auto-playing it, and `CardCmd.Upgrade` returns at once when
/// `CombatManager.IsEnding` (CardCmd.cs:267-270). Once the first Shiv kills
/// the last enemy, the second Shiv stays unupgraded.
#[test]
fn knife_trap_plus_stops_upgrading_shivs_once_combat_is_ending() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Silent);
        let enemy = add_enemy(&mut state, 2, 1);
        let first = new_card(&mut state, CardId::Shiv, false);
        let second = new_card(&mut state, CardId::Shiv, false);
        state.player.exhaust.push(first);
        state.player.exhaust.push(second);
        let i = push_hand(&mut state, CardId::KnifeTrap, true);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(hp_of(&state, enemy), 0, "{v:?}");
        assert!(card(&state, first).upgraded, "{v:?}: first Shiv upgraded");
        assert!(
            !card(&state, second).upgraded,
            "{v:?}: second Shiv upgraded after the killing blow"
        );
    }
}

/// `Jackpot.cs` attacks first, then `CardCmd.Upgrade`s each generated card:
/// after a killing blow on the last enemy that is a no-op
/// (CardCmd.cs:267-270), so Jackpot+ generates unupgraded cards.
#[test]
fn jackpot_plus_kill_generates_unupgraded_cards() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let enemy = add_enemy(&mut state, 2, 1);
        let i = push_hand(&mut state, CardId::Jackpot, true);
        let before = state.cards.len();
        let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
        assert_eq!(hp_of(&state, enemy), 0, "{v:?}");
        let generated: Vec<_> = state.cards[before..].iter().collect();
        assert!(!generated.is_empty(), "{v:?}: Jackpot generated cards");
        for c in generated {
            assert!(
                !c.upgraded,
                "{v:?}: {:?} upgraded after the kill",
                c.card_id
            );
        }
    }
}
