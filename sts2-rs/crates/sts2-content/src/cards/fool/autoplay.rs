//! Tarot tokens that auto-play cards: TheFool (random hand cards) and
//! TheDevil (exhaust the hand, play that many from the draw pile). Values
//! from the v0.113.0 decomp `Models.Cards/<Name>.cs`.
use smallvec::SmallVec;
use sts2_core::card_effects::{self, EffectResult};
use sts2_core::{CardUid, CombatState, CreatureUid};

/// `TheFool.cs`: 2 (+1) times, `StableShuffle(Rng.CombatCardSelection)`
/// the playable hand cards (falling back to the whole hand) and
/// `CardCmd.AutoPlay` the first. Both shuffles read the stream even when
/// empty; an empty hand plays nothing but the loop goes on. A surfaced select
/// parks the iterations still owed (Catastrophe's shape).
pub(crate) fn play_the_fool(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let Some(me) = card_effects::resolving_card_uid(state) else {
        return EffectResult::Resolved;
    };
    the_fool_run(state, me, 0);
    EffectResult::Resolved
}

fn the_fool_iterations(state: &CombatState, me: CardUid) -> u32 {
    if state
        .cards
        .get(me.0.get() as usize - 1)
        .is_some_and(|c| c.upgraded)
    {
        3
    } else {
        2
    }
}

/// `StableShuffle(Rng.CombatCardSelection).FirstOrDefault()` over `pool`.
fn stable_first(state: &mut CombatState, mut pool: SmallVec<[CardUid; 16]>) -> Option<CardUid> {
    let version = state.game_version;
    let cards = &state.cards;
    let cmp = |a: CardUid, b: CardUid| {
        sts2_core::cards::card_model_compare_for(
            &cards[a.0.get() as usize - 1],
            &cards[b.0.get() as usize - 1],
            version,
        )
    };
    state
        .run_rng
        .combat_card_selection
        .stable_shuffle_by_for(version, pool.as_mut_slice(), &cmp);
    pool.first().copied()
}

fn the_fool_run(state: &mut CombatState, me: CardUid, done: u32) {
    let mut n = done;
    while n < the_fool_iterations(state, me) {
        let playable: SmallVec<[CardUid; 16]> = state
            .player
            .hand
            .iter()
            .copied()
            .filter(|u| !card_effects::card_unplayable(state.cards[u.0.get() as usize - 1].card_id))
            .collect();
        let mut pick = stable_first(state, playable);
        if pick.is_none() {
            let hand: SmallVec<[CardUid; 16]> = state.player.hand.iter().copied().collect();
            pick = stable_first(state, hand);
        }
        n += 1;
        let Some(uid) = pick else {
            continue;
        };
        sts2_core::combat_loop::auto_play_specific_card_from_discard(state, uid, false);
        if let Some(resume) = state.pending_autoplay_resume.as_mut() {
            resume.continuation = Some(sts2_core::AutoPlayLoopCont {
                resume_fn: the_fool_resume,
                iters: n,
                uids: smallvec::smallvec![me],
                force_exhaust: false,
                then: None,
            });
            return;
        }
    }
}

fn the_fool_resume(state: &mut CombatState, cont: &sts2_core::AutoPlayLoopCont) {
    if let Some(&me) = cont.uids.first() {
        the_fool_run(state, me, cont.iters);
    }
}

/// `TheDevil.cs`: `CardCmd.Exhaust` every hand card, then
/// `AutoPlayFromDrawPile(that many, Top)`, upgrading each card as it is
/// taken when TheDevil is upgraded.
pub(crate) fn play_the_devil(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    let hand: SmallVec<[CardUid; 16]> = state.player.hand.iter().copied().collect();
    for &uid in &hand {
        let _ = card_effects::exhaust_card_uid(state, uid);
    }
    let up = super::helpers::upgraded(state);
    let n = i32::try_from(hand.len()).unwrap_or(i32::MAX);
    sts2_core::combat_loop::auto_play_n_from_draw_top_with(state, n, false, up);
    EffectResult::Resolved
}
