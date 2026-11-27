//! `preview_option_effect` counts effects that land only after a later screen.
//!
//! Some options finish the event but leave their payoff queued on the run
//! state: a choose-a-card screen (Brain Leech's Share Knowledge), a pick-N grid
//! (Room Full of Cheese's Gorge), an interactive removal with a curse or a gold
//! cost queued behind it (Wellspring, Zen Weaver, Field of Man-Sized Holes,
//! Luminous Choir). The preview diffed only the deck, gold and HP, so all of
//! these featurized as an option that does nothing, and the e2e Ironclad net
//! never took Share Knowledge (0 of 232 in the 600M all-character DEV runs)
//! while Rip previewed as "+1 card, -5 HP".
use sts2_content::events::dispatch::{
    present, preview_option_effect, previews_visible_part_only, resolve_by_index, OptionEffectDelta,
};
use sts2_content::run_callbacks::new_run_manager_for_character;
use sts2_core::events::EventId;
use sts2_core::map::MapPointType;
use sts2_core::potions::{CharacterId, PotionKind};
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::runstate::RunState;
use sts2_core::RelicId;

fn run_state(seed: i32, act: u8) -> RunManager {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
    let rs = &mut rm.run_state;
    rs.act_index = act;
    rs.gold = 300;
    rs.current_hp = 60;
    rs.potions = vec![PotionKind::BlockPotion, PotionKind::FirePotion];
    rs.potion_slot_indices = vec![0, 1];
    rs.relics.extend([RelicId::Anchor, RelicId::Lantern]);
    rs.set_agent_decisions();
    rs.interactive_potion_swap = true;
    rm
}

/// The preview of `option` on the event's first page, from the first act that
/// offers it enabled.
fn preview(ev: EventId, option: &str) -> OptionEffectDelta {
    for act in 1..=3 {
        let mut rm = run_state(7, act);
        let rs = &mut rm.run_state;
        rs.enter_event_room(ev);
        if rs.current_event.is_none() {
            continue;
        }
        let opts = present(rs).expect("options");
        if let Some(i) = opts.iter().position(|o| o.option_id == option && o.enabled) {
            return preview_option_effect(rs, i).expect("previewable");
        }
    }
    panic!("{ev:?}/{option} not offered enabled in acts 1-3");
}

#[test]
fn share_knowledge_previews_the_card_it_grants() {
    let d = preview(EventId::BrainLeech, "share_knowledge");
    assert_eq!((d.cards_gained, d.cards_lost, d.hp_delta), (1, 0, 0));
    let rip = preview(EventId::BrainLeech, "rip");
    assert_eq!((rip.cards_gained, rip.hp_delta), (1, -5), "Rip unchanged");
}

#[test]
fn gorge_previews_both_picks() {
    let d = preview(EventId::RoomFullOfCheese, "gorge");
    assert_eq!((d.cards_gained, d.cards_lost), (2, 0));
}

#[test]
fn removal_then_curse_previews_both() {
    let bathe = preview(EventId::Wellspring, "bathe");
    assert_eq!((bathe.cards_gained, bathe.cards_lost), (1, 1));
    let resist = preview(EventId::FieldOfManSizedHoles, "resist");
    assert_eq!((resist.cards_gained, resist.cards_lost), (1, 2));
    let reach = preview(EventId::LuminousChoir, "reach");
    assert_eq!((reach.cards_gained, reach.cards_lost), (1, 2));
}

#[test]
fn zen_weaver_previews_the_price_and_the_removals() {
    let one = preview(EventId::ZenWeaver, "emotional");
    assert_eq!((one.gold_delta, one.cards_lost), (-125, 1));
    let two = preview(EventId::ZenWeaver, "acupuncture");
    assert_eq!((two.gold_delta, two.cards_lost), (-250, 2));
}

#[test]
fn spirit_grafter_rejection_damage_goes_through_lose_hp() {
    let plain = preview(EventId::SpiritGrafter, "rejection");
    assert_eq!(plain.hp_delta, -10);
    // Tungsten Rod takes 1 off each HP loss, the deferred one included.
    let mut rm = run_state(7, 1);
    let rs = &mut rm.run_state;
    rs.relics.push(RelicId::TungstenRod);
    rs.enter_event_room(EventId::SpiritGrafter);
    let i = option_index(rs, "rejection");
    assert_eq!(preview_option_effect(rs, i).unwrap().hp_delta, -9);
    // HP stops at 0. Parity mode: agent mode withholds a lethal Rejection
    // (`ThatDoesDamage(RejectionHpLoss)`, GH #1907).
    let mut rm = run_state(7, 1);
    let rs = &mut rm.run_state;
    rs.withhold_abandon_run_options = false;
    rs.current_hp = 4;
    rs.enter_event_room(EventId::SpiritGrafter);
    let i = option_index(rs, "rejection");
    assert_eq!(preview_option_effect(rs, i).unwrap().hp_delta, -4);
}

#[test]
fn interactive_transforms_preview_one_out_one_in() {
    let hug = preview(EventId::WhisperingHollow, "hug");
    assert_eq!((hug.cards_gained, hug.cards_lost), (1, 1));
}

/// A Neow option whose relic queues its own pickup pick previews no card
/// change for it: Precise Scissors' removal and Neow's Bones' curses are the
/// relic's effect (GH #1540). (Neow's Bones' pickup also grants relics at
/// once; those land in `relics` and read as relic gains, as before.)
#[test]
fn a_granted_relics_pickup_pick_is_not_the_options() {
    let mut rm = run_state(7, 0);
    let rs = &mut rm.run_state;
    rs.neow_offers = Some(sts2_core::NeowOffers {
        positive_1: sts2_core::NeowRelic::PreciseScissors,
        positive_2: sts2_core::NeowRelic::NeowsBones,
        cursed: sts2_core::NeowRelic::LeafyPoultice,
    });
    rs.enter_event_room(EventId::Neow);
    let queued = |i: usize| {
        let mut c = rs.clone();
        resolve_by_index(&mut c, i).expect("resolves");
        c
    };
    assert_eq!(queued(0).pending_deck_removals, 1, "Precise Scissors");
    let scissors = preview_option_effect(rs, 0).expect("previewable");
    assert_eq!(
        (
            scissors.relics_gained,
            scissors.cards_gained,
            scissors.cards_lost
        ),
        (1, 0, 0)
    );
    assert!(queued(1).pending_neows_bones_curses > 0, "Neow's Bones");
    let bones = preview_option_effect(rs, 1).expect("previewable");
    assert_eq!((bones.cards_gained, bones.cards_lost), (0, 0));
}

/// Neow's Bones previews what its text says, not the two relics it rolls on
/// pickup (`NeowsBones.cs:40-44`, `PlayerRng.Rewards`): the preview resolved
/// them, so a Leafy Poultice among them read as two cards lost and a gold
/// relic as gold. Spire Bench seeds 180001810, 180001893, 180001205,
/// 180000607 and 180001857 showed the student two cards lost natively and
/// none in the advisor's seedless rebuild of the same screen.
#[test]
fn neows_bones_previews_the_same_whatever_relics_it_rolls() {
    let mut previews = std::collections::BTreeSet::new();
    let mut pickups = std::collections::BTreeSet::new();
    let mut last = None;
    for seed in 1..=40 {
        let mut rm = run_state(seed, 0);
        let rs = &mut rm.run_state;
        rs.neow_offers = Some(sts2_core::NeowOffers {
            positive_1: sts2_core::NeowRelic::WingedBoots,
            positive_2: sts2_core::NeowRelic::LeadPaperweight,
            cursed: sts2_core::NeowRelic::NeowsBones,
        });
        rs.enter_event_room(EventId::Neow);
        let bones = option_index(rs, "cursed");
        assert!(previews_visible_part_only(rs, bones));
        let d = preview_option_effect(rs, bones).expect("previewable");
        previews.insert(format!("{d:?}"));
        last = Some(d);
        // Non-vacuity: what the pickup really does moves with the seed.
        let mut live = rs.clone();
        resolve_by_index(&mut live, bones).expect("resolves");
        let mut deck: Vec<i32> = live.master_deck.iter().map(|c| c.card_id as i32).collect();
        deck.sort_unstable();
        pickups.insert(format!(
            "{} {} {deck:?} {:?} {:?}",
            live.gold,
            live.max_hp,
            live.relics,
            live.pending_effect_counts()
        ));
    }
    assert!(
        pickups.len() > 1,
        "every seed's Bones pickup was alike: test is vacuous"
    );
    assert_eq!(previews.len(), 1, "{previews:?}");
    let d = last.expect("40 seeds");
    assert_eq!(
        (
            d.relics_gained,
            d.cards_gained,
            d.cards_lost,
            d.gold_delta,
            d.max_hp_delta
        ),
        (3, 0, 0, 0, 0)
    );
}

fn option_index(rs: &RunState, option: &str) -> usize {
    present(rs)
        .expect("options")
        .iter()
        .position(|o| o.option_id == option && o.enabled)
        .unwrap_or_else(|| panic!("{option} not offered enabled"))
}

/// What an option is judged on: the deck, relics and belt as multisets of
/// ids (an upgrade or enchant keeps the id, a transform changes it), and
/// gold, HP and max HP.
struct Snapshot {
    deck: Vec<i32>,
    relics: Vec<i32>,
    potions: Vec<i32>,
    gold: i32,
    hp: i32,
    max_hp: i32,
}

fn snapshot(rs: &RunState) -> Snapshot {
    Snapshot {
        deck: rs.master_deck.iter().map(|c| c.card_id as i32).collect(),
        relics: rs.relics.iter().map(|r| *r as i32).collect(),
        potions: rs.potions.iter().map(|p| *p as i32).collect(),
        gold: rs.gold,
        hp: rs.current_hp,
        max_hp: rs.max_hp,
    }
}

/// How many items of `after` are not matched by one in `before`.
fn added(before: &[i32], after: &[i32]) -> i32 {
    let mut left = before.to_vec();
    let mut n = 0;
    for x in after {
        match left.iter().position(|y| y == x) {
            Some(i) => {
                left.swap_remove(i);
            }
            None => n += 1,
        }
    }
    n
}

/// The played change in the preview's own slots: cards, relics and potions
/// gained and lost separately (a transform previewed as nothing fails), then
/// gold, HP and max HP.
fn played(b: &Snapshot, a: &Snapshot) -> [i32; 9] {
    [
        added(&b.deck, &a.deck),
        added(&a.deck, &b.deck),
        added(&b.relics, &a.relics),
        added(&a.relics, &b.relics),
        added(&b.potions, &a.potions),
        added(&a.potions, &b.potions),
        a.gold - b.gold,
        a.hp - b.hp,
        a.max_hp - b.max_hp,
    ]
}

fn promised(d: &OptionEffectDelta) -> [i32; 9] {
    [
        d.cards_gained,
        d.cards_lost,
        d.relics_gained,
        d.relics_lost,
        d.potions_gained,
        d.potions_lost,
        d.gold_delta,
        d.hp_delta,
        d.max_hp_delta,
    ]
}

/// Whether resolving the option left anything queued behind a later screen.
fn queues_something(before: &RunState, after: &RunState) -> bool {
    after.pending_effect_counts() != before.pending_effect_counts()
        || after.pending_post_removal_hp_loss != before.pending_post_removal_hp_loss
        || after.pending_post_removal_max_hp_gain != before.pending_post_removal_max_hp_gain
        || after.pending_post_removal_gold_loss != before.pending_post_removal_gold_loss
        || after.pending_post_upgrade_hp_loss != before.pending_post_upgrade_hp_loss
}

/// One action that takes what a follow-up screen offers: the first card,
/// the relic, the swap into slot 0, the first deck card to remove,
/// transform, upgrade or enchant. `None` once nothing is left to take (the
/// map, a fight, the event's next page).
fn take_step(rm: &RunManager) -> Option<RunAction> {
    let actions = rm.get_available_actions();
    let rank = |a: &RunAction| match a {
        RunAction::PickCardReward { idx: Some(_) }
        | RunAction::GrantDeckCard { .. }
        | RunAction::RemoveDeckCard { .. }
        | RunAction::TransformDeckCard { .. }
        | RunAction::UpgradeDeckCard { .. }
        | RunAction::EnchantDeckCard { .. }
        | RunAction::PickRelicReward
        | RunAction::PotionSwap {
            discard_slot: Some(_),
        } => Some(0),
        RunAction::ConfirmDeckSelection => Some(1),
        _ => None,
    };
    actions
        .iter()
        .filter_map(|a| rank(a).map(|r| (r, a)))
        .min_by_key(|(r, _)| *r)
        .map(|(_, a)| *a)
}

/// A run standing in an unbound event room on `seed`, or `None` when no
/// first move of the map lands in one.
fn event_room(seed: i32) -> Option<RunManager> {
    let mut rm = new_run_manager_for_character(seed, CharacterId::Ironclad);
    let map = rm.run_state.current_act_map.as_mut()?;
    for p in &mut map.points {
        if p.coord.row == 1 {
            p.point_type = MapPointType::Unknown;
        }
    }
    rm.get_available_actions()
        .into_iter()
        .filter(|a| matches!(a, RunAction::Move { .. }))
        .find_map(|a| {
            let mut probe = rm.clone();
            probe.take_action(a);
            (probe.phase() == Phase::Event && probe.run_state.current_event.is_none())
                .then_some(probe)
        })
}

/// Every event, a few seeds and all three acts: every option previews what
/// taking it and playing its follow-up screens through actually does, card,
/// relic and potion gains and losses counted by id, and gold, HP and max HP,
/// including the part it leaves queued behind a later screen (GH #1540). The
/// played run is the oracle, so a new `pending_*` queue the preview does not
/// read fails here.
///
/// Left out: options that grant a relic (the preview leaves the relic's own
/// pickup picks to the relic), options whose payoff is rolled on pick (the
/// preview shows only their visible part, GH #1035), and any option whose
/// follow-up starts a fight.
#[test]
fn deferred_preview_matches_playing_the_screens_through() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut bad = Vec::new();
    let mut checked = 0;
    let mut deferred = 0;
    for seed in [3, 11, 29] {
        let Some(base) = event_room(seed) else {
            continue;
        };
        for act in 0..3u8 {
            for ev in EventId::all() {
                let mut rm = base.clone();
                {
                    let rs = &mut rm.run_state;
                    rs.act_index = act;
                    rs.gold = 300;
                    rs.current_hp = 60;
                    // An empty belt: a potion reward past a full belt goes to
                    // the swap screen, which the preview does not model.
                    rs.potions.clear();
                    rs.potion_slot_indices.clear();
                    rs.set_agent_decisions();
                    rs.interactive_potion_swap = true;
                    rs.enter_event_room(*ev);
                }
                if rm.run_state.current_event.is_none() {
                    continue;
                }
                let Some(opts) = present(&rm.run_state) else {
                    continue;
                };
                for (i, o) in opts.iter().enumerate().filter(|(_, o)| o.enabled) {
                    let key = format!("{ev:?}/{}@act{act}", o.option_id);
                    let mut probe = rm.run_state.clone();
                    if previews_visible_part_only(&rm.run_state, i)
                        || resolve_by_index(&mut probe, i).is_none()
                    {
                        continue;
                    }
                    let delta = preview_option_effect(&rm.run_state, i).expect("previewable");
                    if delta.relics_gained > 0 || probe.relics.len() > rm.run_state.relics.len() {
                        continue;
                    }
                    deferred += usize::from(queues_something(&rm.run_state, &probe));
                    let mut play = rm.clone();
                    let before = snapshot(&play.run_state);
                    play.take_action(RunAction::EventChoice { idx: i });
                    let mut steps = 0;
                    while let Some(a) = take_step(&play) {
                        play.take_action(a);
                        steps += 1;
                        assert!(steps < 50, "{key}: follow-up screens never settle");
                    }
                    if play.phase() == Phase::Combat {
                        continue;
                    }
                    let actual = played(&before, &snapshot(&play.run_state));
                    checked += 1;
                    if actual != promised(&delta) {
                        bad.push(format!(
                            "{key}: preview {:?} vs played {actual:?}",
                            promised(&delta)
                        ));
                    }
                }
            }
        }
    }
    assert!(
        deferred >= 10,
        "only {deferred} of {checked} options checked queue something"
    );
    bad.sort();
    bad.dedup();
    assert!(
        bad.is_empty(),
        "preview differs from the played screens:\n{}",
        bad.join("\n")
    );
}
