//! `PlayerCombatState::fool` (v0.113.0 Fool mechanics) on the wire, as the
//! flat integer list `$.fool_state`.
//!
//! A rebuild re-allocates every `CardUid` from the pile walk and every enemy
//! `CreatureUid` from the roster, so card references travel as a
//! `(pile, index)` locator and creatures as their index in `enemy_uids`. A
//! reference that no longer resolves (a card in no pile, a creature off the
//! roster) is dropped: card refs become `None` and impaled entries vanish.
//!
//! Layout (`FORMAT` first), every field of [`FoolState`] in declaration
//! order — the encoder destructures the struct exhaustively, so a new field
//! does not compile until it is carried here.

use smallvec::SmallVec;
use sts2_core::fool::{ArrowSlot, ArrowStatus, FoolState, MelodyTurn};
use sts2_core::{CardUid, CombatState};

const FORMAT: i64 = 1;
const NONE: i64 = -1;
/// Pile names in locator order; the card ref is `pile * PILE_STRIDE + index`.
const PILES: [&str; 5] = ["hand", "draw", "discard", "exhaust", "play"];
const PILE_STRIDE: i64 = 100_000;

fn card_ref(state: &CombatState, uid: CardUid) -> i64 {
    crate::emit::locate_card(state, uid)
        .and_then(|(pile, i)| {
            let p = PILES.iter().position(|n| *n == pile)?;
            Some(i64::try_from(p).ok()? * PILE_STRIDE + i64::try_from(i).ok()?)
        })
        .unwrap_or(NONE)
}

fn card_of(state: &CombatState, r: i64) -> Option<CardUid> {
    if r < 0 {
        return None;
    }
    let p = &state.player;
    let pile = match r / PILE_STRIDE {
        0 => p.hand.as_slice(),
        1 => p.draw.as_slice(),
        2 => p.discard.as_slice(),
        3 => p.exhaust.as_slice(),
        4 => p.play.as_slice(),
        _ => return None,
    };
    pile.get(usize::try_from(r % PILE_STRIDE).ok()?).copied()
}

fn opt(v: Option<i64>) -> [i64; 2] {
    v.map_or([0, 0], |x| [1, x])
}

/// Encode `f` against `state`'s piles and roster.
#[must_use]
pub fn encode(state: &CombatState, f: &FoolState) -> Vec<i64> {
    let FoolState {
        arrows,
        impaled,
        melody,
        play_seq,
        open_plays,
        aim_latch,
        aim_source,
        quietus_latched,
        arrow_attack_depth,
        stretch_latch,
        encore_plays,
        last_finished_seq,
        lullaby_round,
        attacked_card,
    } = f;
    let MelodyTurn {
        stamp,
        finished,
        melody_cards_finished,
        started,
        first_started_seq,
    } = melody;
    let mut w = vec![FORMAT, arrows.len() as i64];
    for a in arrows {
        let status = match a.status {
            ArrowStatus::Retrieved => 0,
            ArrowStatus::Away => 1,
            ArrowStatus::Destroyed => 2,
        };
        w.extend([status, i64::from(a.destroy_after_impaled)]);
    }
    let roster: Vec<(i64, &SmallVec<[u8; 4]>)> = impaled
        .iter()
        .filter_map(|(c, slots)| {
            let i = state.enemy_uids.iter().position(|u| u == c)?;
            Some((i as i64, slots))
        })
        .collect();
    w.push(roster.len() as i64);
    for (i, slots) in roster {
        w.extend([i, slots.len() as i64]);
        w.extend(slots.iter().map(|s| i64::from(*s)));
    }
    match stamp {
        Some((round, player)) => w.extend([1, i64::from(*round), i64::from(*player)]),
        None => w.extend([0, 0, 0]),
    }
    w.extend(finished.iter().map(|x| i64::from(*x)));
    w.push(i64::from(*melody_cards_finished));
    w.extend(started.iter().map(|x| i64::from(*x)));
    w.extend(first_started_seq.iter().map(|x| i64::from(*x)));
    w.push(i64::from(*play_seq));
    w.push(open_plays.len() as i64);
    for (seq, card) in open_plays {
        w.extend([i64::from(*seq), card_ref(state, *card)]);
    }
    w.extend(opt(aim_latch.map(i64::from)));
    w.push(aim_source.map_or(NONE, |c| card_ref(state, c)));
    w.push(i64::from(*quietus_latched));
    w.push(i64::from(*arrow_attack_depth));
    match stretch_latch {
        Some((card, amount, gave)) => w.extend([
            1,
            card_ref(state, *card),
            i64::from(*amount),
            i64::from(*gave),
        ]),
        None => w.extend([0, 0, 0, 0]),
    }
    w.push(i64::from(*encore_plays));
    w.push(i64::from(*last_finished_seq));
    w.extend(opt(lullaby_round.map(i64::from)));
    w.push(attacked_card.map_or(NONE, |c| card_ref(state, c)));
    w
}

struct Reader<'a> {
    words: &'a [i64],
    at: usize,
}

impl Reader<'_> {
    fn next(&mut self) -> Result<i64, String> {
        let v = *self
            .words
            .get(self.at)
            .ok_or_else(|| format!("truncated at word {}", self.at))?;
        self.at += 1;
        Ok(v)
    }

    fn int<T: TryFrom<i64>>(&mut self) -> Result<T, String> {
        let at = self.at;
        T::try_from(self.next()?).map_err(|_| format!("word {at} out of range"))
    }

    fn flag(&mut self) -> Result<bool, String> {
        match self.next()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(format!("word {} must be 0/1, got {v}", self.at - 1)),
        }
    }

    fn opt<T: TryFrom<i64>>(&mut self) -> Result<Option<T>, String> {
        let present = self.flag()?;
        let v = self.int::<T>();
        if present {
            v.map(Some)
        } else {
            Ok(None)
        }
    }
}

/// Decode `words` against the rebuilt `state`'s piles and roster.
///
/// # Errors
///
/// A wrong format tag, a truncated or over-long list, or an out-of-range word.
pub fn decode(state: &CombatState, words: &[i64]) -> Result<FoolState, String> {
    let mut r = Reader { words, at: 0 };
    if r.next()? != FORMAT {
        return Err(format!("unknown format (expected {FORMAT})"));
    }
    let mut f = FoolState::default();
    for _ in 0..r.int::<usize>()? {
        let status = match r.next()? {
            0 => ArrowStatus::Retrieved,
            1 => ArrowStatus::Away,
            2 => ArrowStatus::Destroyed,
            v => return Err(format!("bad arrow status {v}")),
        };
        f.arrows.push(ArrowSlot {
            status,
            destroy_after_impaled: r.flag()?,
        });
    }
    for _ in 0..r.int::<usize>()? {
        let i = r.int::<usize>()?;
        let mut slots = SmallVec::new();
        for _ in 0..r.int::<usize>()? {
            slots.push(r.int::<u8>()?);
        }
        if let Some(c) = state.enemy_uids.get(i).copied() {
            f.impaled.push((c, slots));
        }
    }
    let stamped = r.flag()?;
    let round = r.int::<u16>()?;
    let player = r.flag()?;
    f.melody.stamp = stamped.then_some((round, player));
    for x in &mut f.melody.finished {
        *x = r.int()?;
    }
    f.melody.melody_cards_finished = r.int()?;
    for x in &mut f.melody.started {
        *x = r.int()?;
    }
    for x in &mut f.melody.first_started_seq {
        *x = r.int()?;
    }
    f.play_seq = r.int()?;
    for _ in 0..r.int::<usize>()? {
        let seq = r.int::<u32>()?;
        if let Some(c) = card_of(state, r.next()?) {
            f.open_plays.push((seq, c));
        }
    }
    f.aim_latch = r.opt()?;
    f.aim_source = card_of(state, r.next()?);
    f.quietus_latched = r.flag()?;
    f.arrow_attack_depth = r.int()?;
    let latched = r.flag()?;
    let card = card_of(state, r.next()?);
    let amount = r.int::<i32>()?;
    let gave = r.flag()?;
    f.stretch_latch = if latched {
        card.map(|c| (c, amount, gave))
    } else {
        None
    };
    f.encore_plays = r.int()?;
    f.last_finished_seq = r.int()?;
    f.lullaby_round = r.opt()?;
    f.attacked_card = card_of(state, r.next()?);
    if r.at != words.len() {
        return Err(format!("{} trailing words", words.len() - r.at));
    }
    Ok(f)
}
