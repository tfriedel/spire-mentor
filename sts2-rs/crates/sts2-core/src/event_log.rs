//! Opt-in, STATE-RESIDENT combat event log — the engine's port of C#
//! `CombatHistory` (`MegaCrit.Sts2.Core.Combat.History/CombatHistory.cs`).
//!
//! # What it is
//!
//! An ordered, combat-scoped list of everything the game's own combat log
//! shows: card plays started/finished, every hit (dealer, receiver, amount,
//! blocked, HP either side), block gains, powers applied, monster moves, card
//! movement, orbs, potions, energy. One [`EventLogRow`] per C#
//! `CombatHistoryEntry`, carrying the same fields its `Description` renders
//! from, plus the SOURCE (the card or monster move that was mid-flight) which
//! C# leaves implicit in the entry ordering.
//!
//! # Why it is on the state, not in a thread-local
//!
//! [`crate::damage_trace`] — the forensic tool this complements — arms and
//! buffers in a thread-local. That is correct for its job (a mirror step on a
//! known thread) and fatal for this one: a caller that arms on thread A and
//! runs the step on a rayon worker gets an EMPTY trace back. A log that
//! silently returns nothing when the work moved threads cannot back a UI.
//!
//! Living on [`crate::CombatState`] removes the failure mode instead of
//! documenting it: there is nothing to arm, the rows travel with the state
//! through every clone and every thread, and a state handed to a worker keeps
//! recording exactly as it did on the thread that built it.
//!
//! The flip side of state residency is that a SEARCH would inherit recording
//! from the live state it clones. It does not: `MctsTree::new` calls
//! [`EventLog::disarm_for_search`] on its root, so rollouts neither record nor
//! carry the live log's rows. Both halves are covered by
//! `tests/event_log.rs`.
//!
//! # Cost when disabled
//!
//! One `bool` read per would-be event. Recording is OFF by default; every
//! call site guards on [`EventLog::recording`] before it builds a row, so a
//! headless / parity / self-play run allocates nothing. The log is excluded
//! from equality (see the [`PartialEq`] impl below) and is never emitted by
//! `state_to_dict` — it is a display side-channel, not a parity surface.
//!
//! # Text rendering
//!
//! [`EventLogRow::human_readable`] reproduces C#
//! `CombatHistoryEntry.HumanReadableString` — `Rd {n} ({Side} turn):
//! {Description}.` — using the same ids C# prints (`STRIKE_IRONCLAD`,
//! `IRONCLAD`), so a Rust log and the world-side `[HISTTRACE]` stream
//! (`Sts2Headless/RunSimulator.cs`, `STS2_COMBAT_HISTORY_TRACE=1`) diff
//! line-for-line. Where C#'s own string is buggy this port keeps the bug —
//! see [`EventLogEntry::description`]. Pretty, localized rendering ("The
//! Ironclad dealt 6 damage to Nibbit using Strike") is the workbench's job and
//! happens in Python, off the structured fields.

use crate::cards::CardId;
use crate::monsters::MonsterId;
use crate::orbs::OrbKind;
use crate::potions::{CharacterId, PotionKind};
use crate::powers::PowerKind;
use crate::Side;

/// Hard cap on rows per combat. A long fight records a few thousand; the cap
/// only bites a pathological loop, and the overflow is COUNTED
/// ([`EventLog::dropped`]) rather than silently dropped, because a truncated
/// log that looks complete is worse than a log that says it lost rows.
pub const MAX_ROWS: usize = 20_000;

/// Environment variable that echoes every recorded row to stderr as
/// `[COMBATLOG] {EntryKind}: {HumanReadableString}` — the shape
/// `RunSimulator.cs` prints under `STS2_COMBAT_HISTORY_TRACE=1`, so the two
/// streams diff after stripping the tag. Only echoes rows that are recorded,
/// so it needs the log armed as well.
pub const ECHO_ENV: &str = "STS2_COMBAT_LOG";

/// Which creature an entry is about.
///
/// Carries the id STRING rather than a uid alone: a row outlives the creature
/// (a corpse is removed from `state.creatures`), and a reader should not have
/// to re-derive "who was uid 4" from a board that no longer contains it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogCreature {
    /// The player. C# prints `Player.Character.Id.Entry`; `None` covers the
    /// fixtures that never set [`crate::PlayerCombatState::character`].
    Player(Option<CharacterId>),
    /// A monster — enemy or summoned ally. C# prints `Monster.Id.Entry`.
    Monster {
        /// Creature uid, for joining a row to a board slot.
        uid: u32,
        /// Species id (`state.creatures[..].monster_id`).
        id: MonsterId,
    },
}

impl LogCreature {
    /// The id C# would print for this creature.
    #[must_use]
    pub fn id_str(&self) -> &'static str {
        match self {
            Self::Player(c) => character_entry(*c),
            Self::Monster { id, .. } => id.0,
        }
    }

    /// Build the row-side identity of `uid` on `state`. Returns the player
    /// variant when `uid` is the player's mirror creature.
    #[must_use]
    pub fn of_uid(state: &crate::CombatState, uid: crate::CreatureUid) -> Self {
        if state.player.creature_uid == Some(uid) {
            return Self::Player(state.player.character);
        }
        match state.creatures.iter().find(|c| c.id == uid.0) {
            Some(c) => Self::Monster {
                uid: uid.0.get(),
                id: c.monster_id,
            },
            // A uid with no creature is a row about something already torn
            // down. Name it honestly rather than mislabelling it the player.
            None => Self::Monster {
                uid: uid.0.get(),
                id: MonsterId("?"),
            },
        }
    }

    /// Build the row-side identity of a [`crate::hooks::CreatureRef`] dealer.
    #[must_use]
    pub fn of_dealer(state: &crate::CombatState, dealer: crate::hooks::CreatureRef) -> Self {
        match dealer {
            crate::hooks::CreatureRef::Player => Self::Player(state.player.character),
            crate::hooks::CreatureRef::Enemy(uid) => Self::of_uid(state, uid),
            crate::hooks::CreatureRef::Ally(idx) => match state.creatures.get(idx as usize) {
                Some(c) => Self::Monster {
                    uid: c.id.get(),
                    id: c.monster_id,
                },
                None => Self::Monster {
                    uid: 0,
                    id: MonsterId("?"),
                },
            },
            // The null-target sentinel is never a dealer — it exists so a
            // null-TARGET fold fails every `owner == ctx.target` gate
            // (`CreatureRef::Nobody`). Name it honestly rather than
            // mislabelling it the player, same as the torn-down-uid arm above.
            crate::hooks::CreatureRef::Nobody => Self::Monster {
                uid: 0,
                id: MonsterId("?"),
            },
        }
    }
}

/// C# `Character.Id.Entry` for a character, or `PLAYER` when unset.
#[must_use]
pub const fn character_entry(character: Option<CharacterId>) -> &'static str {
    match character {
        Some(CharacterId::Ironclad) => "IRONCLAD",
        Some(CharacterId::Silent) => "SILENT",
        Some(CharacterId::Defect) => "DEFECT",
        Some(CharacterId::Regent) => "REGENT",
        Some(CharacterId::Necrobinder) => "NECROBINDER",
        Some(CharacterId::Fool) => "FOOL",
        None => "PLAYER",
    }
}

/// C# `Orb.Id.Entry` for an orb kind.
#[must_use]
pub const fn orb_entry(orb: OrbKind) -> &'static str {
    match orb {
        OrbKind::Empty => "EMPTY",
        OrbKind::Lightning => "LIGHTNING",
        OrbKind::Frost => "FROST",
        OrbKind::Dark => "DARK",
        OrbKind::Plasma => "PLASMA",
        OrbKind::Glass => "GLASS",
    }
}

/// What was mid-flight when an event happened — the "using Strike" half of a
/// damage line, which C# leaves implicit in entry ORDER.
///
/// Maintained by [`EventLog::push`] itself from the rows going past (a
/// `CardPlayStarted` opens a source, its `CardPlayFinished` closes it, a
/// `MonsterMove` replaces the stack), so no call site has to push/pop and no
/// early return can leak a stale frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogSource {
    /// A card resolving. Nests: an auto-play inside `Havoc` sits above it.
    Card {
        /// The resolving card.
        card: CardId,
        /// Upgrade level of the resolving instance.
        upgraded: bool,
    },
    /// A monster move performing.
    Move {
        /// The acting monster.
        monster: MonsterId,
        /// C# `Move.Id` — the AI state id.
        move_id: &'static str,
    },
}

/// One combat event. Variants mirror the C# `CombatHistoryEntry` subclasses in
/// `MegaCrit.Sts2.Core.Combat.History.Entries/`, field for field where the
/// engine models the same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventLogEntry {
    /// C# `CardPlayStartedEntry`. Recorded at the play chokepoints, BEFORE the
    /// card body dispatches, so a card that auto-plays others appears ahead of
    /// its children.
    CardPlayStarted {
        /// The card being played.
        card: CardId,
        /// Upgrade level of the played instance.
        upgraded: bool,
        /// The X value for an X-cost card; 0 for any other card, manual or
        /// auto-played (see `cards::observational_energy_spent`).
        energy_spent: i32,
        /// `true` for engine auto-plays (Cascade / Havoc / Stampede / Mayhem /
        /// DistilledChaos); `false` for a manual play.
        auto_played: bool,
        /// Player-chosen enemy target, if any.
        target: Option<u8>,
    },
    /// C# `CardPlayFinishedEntry`. Recorded in `combat_loop::finish_card_play`
    /// — the single funnel both the inline path and the surfaced-choice resume
    /// path pass through.
    CardPlayFinished {
        /// The card that finished resolving.
        card: CardId,
        /// Upgrade level of the played instance.
        upgraded: bool,
    },
    /// C# `MonsterPerformedMoveEntry`.
    MonsterMove {
        /// The acting monster.
        monster: LogCreature,
        /// C# `Move.Id` — the AI state id the monster performed.
        move_id: &'static str,
    },
    /// C# `DamageReceivedEntry`, plus the HP either side of the write so a row
    /// is self-checking (`hp_before - hp_after == hp_loss`, except on overkill
    /// where `hp_after` clamps at 0).
    DamageReceived {
        /// Who dealt it. `None` for source-less HP loss (C# prints "took").
        dealer: Option<LogCreature>,
        /// Who lost the HP.
        receiver: LogCreature,
        /// C# `DamageResult.UnblockedDamage` — what got through block.
        unblocked: i32,
        /// What block absorbed.
        blocked: i32,
        /// HP before the write.
        hp_before: i32,
        /// HP after the write (clamped at 0).
        hp_after: i32,
    },
    /// C# `BlockGainedEntry`.
    BlockGained {
        /// Who gained it.
        receiver: LogCreature,
        /// How much, after the modify chain.
        amount: i32,
    },
    /// C# `PowerReceivedEntry`.
    PowerReceived {
        /// Who applied it, if the apply carried an applier.
        applier: Option<LogCreature>,
        /// Who received it.
        receiver: LogCreature,
        /// The power.
        power: PowerKind,
        /// Stacks applied (post-modify-chain).
        amount: i32,
    },
    /// A power TRIGGERING — C# `PowerModel.Flash()` (`PowerModel.cs:452`),
    /// which raises `Flashed` and reaches `NCreature.OnPowerFlashed`
    /// (`NCreature.cs:925`), spawning `power_flash_vfx` on the owner.
    ///
    /// Deliberately NOT a port of a C# `CombatHistoryEntry`: the game's combat
    /// history has no such row, because a flash is a VIEW signal — nothing
    /// changed, the power just did its job. It lives in this log anyway
    /// because it wants precisely what the log already provides: the same
    /// arming, ordering, source stack, row cap, `disarm_for_search`, and PyO3
    /// pipeline. A second parallel channel would duplicate all of it.
    ///
    /// Two consequences follow, and both are enforced below rather than left
    /// to convention: [`Self::kind`] returns a name WITHOUT the `Entry`
    /// suffix, and [`EventLog::push`] never echoes it — so a `[HISTTRACE]`
    /// diff against the world side still matches line for line.
    PowerFlashed {
        /// Whose power fired.
        creature: LogCreature,
        /// Which power.
        power: PowerKind,
    },
    /// C# `CardDrawnEntry`.
    CardDrawn {
        /// The drawn card.
        card: CardId,
        /// Upgrade level.
        upgraded: bool,
        /// `true` for the turn-start hand draw, `false` for effect draws —
        /// the C# `FromHandDraw` discriminator several powers count on.
        from_hand_draw: bool,
    },
    /// C# `CardGeneratedEntry` — a card created DURING combat (a monster's
    /// Slimed / Wound, a Regent's generated Attack), as opposed to one drawn
    /// from the deck.
    CardGenerated {
        /// The generated card.
        card: CardId,
        /// Upgrade level.
        upgraded: bool,
        /// Where it landed: `"draw"` / `"discard"` / `"hand"` / `"exhaust"`.
        pile: &'static str,
        /// C#'s `Player? creator` — `false` when a monster inflicted it.
        by_player: bool,
    },
    /// C# `CardDiscardedEntry`.
    CardDiscarded {
        /// The discarded card.
        card: CardId,
        /// Upgrade level.
        upgraded: bool,
    },
    /// C# `CardExhaustedEntry`.
    CardExhausted {
        /// The exhausted card.
        card: CardId,
        /// Upgrade level.
        upgraded: bool,
    },
    /// C# `OrbChanneledEntry`.
    OrbChanneled {
        /// The channeled orb.
        orb: OrbKind,
    },
    /// Orb evoke. C# has no history entry for it (the log shows the evoke's
    /// damage instead); recorded here because an evoke with no damage —
    /// Frost, Plasma — is otherwise invisible in the log.
    OrbEvoked {
        /// The evoked orb.
        orb: OrbKind,
    },
    /// C# `PotionUsedEntry`.
    PotionUsed {
        /// The potion drunk.
        potion: PotionKind,
        /// Chosen enemy target, if any.
        target: Option<u8>,
    },
    /// C# `EnergySpentEntry`.
    EnergySpent {
        /// Energy spent by the play.
        amount: i32,
    },
}

impl EventLogEntry {
    /// C# entry class name, for the `[COMBATLOG] {kind}:` echo tag.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::CardPlayStarted { .. } => "CardPlayStartedEntry",
            Self::CardPlayFinished { .. } => "CardPlayFinishedEntry",
            Self::MonsterMove { .. } => "MonsterPerformedMoveEntry",
            Self::DamageReceived { .. } => "DamageReceivedEntry",
            Self::BlockGained { .. } => "BlockGainedEntry",
            Self::PowerReceived { .. } => "PowerReceivedEntry",
            // No `Entry` suffix, and deliberately so — C# has no
            // `PowerFlashedEntry`. See the variant's doc.
            Self::PowerFlashed { .. } => "PowerFlashed",
            Self::CardDrawn { .. } => "CardDrawnEntry",
            Self::CardGenerated { .. } => "CardGeneratedEntry",
            Self::CardDiscarded { .. } => "CardDiscardedEntry",
            Self::CardExhausted { .. } => "CardExhaustedEntry",
            Self::OrbChanneled { .. } => "OrbChanneledEntry",
            Self::OrbEvoked { .. } => "OrbEvokedEntry",
            Self::PotionUsed { .. } => "PotionUsedEntry",
            Self::EnergySpent { .. } => "EnergySpentEntry",
        }
    }

    /// C# `CombatHistoryEntry.Description`, reproduced verbatim — INCLUDING
    /// two upstream copy-paste bugs, because the point of this string is to
    /// diff against the world-side `[HISTTRACE]` stream and a "corrected"
    /// line is a false difference on every row:
    ///
    /// * `CardDrawnEntry.Description` says "discarded", not "drew"
    ///   (`CardDrawnEntry.cs:20`).
    /// * `EnergySpentEntry.Description` interpolates nothing — it prints the
    ///   literal word `Amount` (`EnergySpentEntry.cs:14`).
    ///
    /// The structured fields carry the truth; only this rendering is bug-
    /// compatible. The workbench never calls it.
    #[must_use]
    pub fn description(&self, player: Option<CharacterId>) -> String {
        let me = character_entry(player);
        match self {
            Self::CardPlayStarted { card, target, .. } => {
                let mut s = format!("{me} started playing {}", card.to_python_name());
                if let Some(t) = target {
                    s.push_str(&format!(" targeting #{t}"));
                }
                s
            }
            Self::CardPlayFinished { card, .. } => {
                format!("{me} played {}", card.to_python_name())
            }
            Self::MonsterMove { monster, move_id } => {
                format!("{} performed {move_id}", monster.id_str())
            }
            Self::DamageReceived {
                dealer,
                receiver,
                unblocked,
                ..
            } => match dealer {
                Some(d) => format!(
                    "{} dealt {unblocked} damage to {}",
                    d.id_str(),
                    receiver.id_str()
                ),
                None => format!("{} took {unblocked} damage", receiver.id_str()),
            },
            Self::BlockGained { receiver, amount } => {
                format!("{} gained {amount} block", receiver.id_str())
            }
            Self::PowerReceived {
                applier,
                receiver,
                power,
                amount,
            } => match applier {
                Some(a) => format!(
                    "{} applied {amount} {} to {}",
                    a.id_str(),
                    power.python_name(),
                    receiver.id_str()
                ),
                None => format!(
                    "{} received {amount} {}",
                    receiver.id_str(),
                    power.python_name()
                ),
            },
            // C# bug preserved: "discarded" (CardDrawnEntry.cs:20).
            Self::CardDrawn { card, .. } => {
                format!("{me} discarded {}", card.to_python_name())
            }
            Self::CardGenerated { card, .. } => {
                format!("{me} generated {} during combat", card.to_python_name())
            }
            Self::CardDiscarded { card, .. } => {
                format!("{me} discarded {}", card.to_python_name())
            }
            Self::CardExhausted { card, .. } => {
                format!("{me} exhausted {}", card.to_python_name())
            }
            Self::OrbChanneled { orb } => format!("{me} channeled {}", orb_entry(*orb)),
            Self::OrbEvoked { orb } => format!("{me} evoked {}", orb_entry(*orb)),
            Self::PotionUsed { potion, target } => {
                let mut s = format!("{me} drank {}", potion.python_name());
                if let Some(t) = target {
                    s.push_str(&format!(" targeting #{t}"));
                }
                s
            }
            // C# bug preserved: the literal word "Amount" (EnergySpentEntry.cs:14).
            Self::EnergySpent { .. } => format!("{me} spent Amount energy"),
            // No C# counterpart to reproduce — see the variant's doc. Never
            // rendered: the workbench reads the structured fields, and the
            // `[HISTTRACE]` echo skips this variant entirely.
            Self::PowerFlashed { .. } => String::new(),
        }
    }
}

/// One recorded event plus the C# `CombatHistoryEntry` header fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventLogRow {
    /// C# `RoundNumber` at the time of the event.
    pub round: u16,
    /// C# `CurrentSide` — whose turn it was. NOT the actor's side: a Thorns
    /// hit during the player's turn has an enemy dealer and `Side::Player`
    /// here, exactly as C# documents on `CombatHistoryEntry.CurrentSide`.
    pub side: Side,
    /// The event.
    pub entry: EventLogEntry,
    /// The card or monster move that was mid-flight, innermost first — the
    /// "using Strike" attribution. `None` outside any play (turn-start
    /// effects, relic triggers, power ticks).
    pub source: Option<LogSource>,
}

impl EventLogRow {
    /// C# `CombatHistoryEntry.HumanReadableString`:
    /// `Rd {round} ({Side} turn): {Description}.`
    #[must_use]
    pub fn human_readable(&self, player: Option<CharacterId>) -> String {
        let side = match self.side {
            Side::Player => "Player",
            Side::Enemy => "Monster",
        };
        format!(
            "Rd {} ({side} turn): {}.",
            self.round,
            self.entry.description(player)
        )
    }
}

/// The combat-scoped log itself. Lives on [`crate::TurnEvents`], so it is
/// cloned with the state and travels across threads.
#[derive(Clone, Debug, Default)]
pub struct EventLog {
    /// Recorded rows, in event order.
    rows: Vec<EventLogRow>,
    /// Open sources, outermost first; `last()` is what a row attributes to.
    ///
    /// A `Vec`, deliberately, not a `SmallVec`: this struct is part of
    /// `CombatState`, which the engine clones on the hottest path there is. An
    /// inline buffer would copy its full capacity on every clone of every
    /// non-recording state; an empty `Vec` clones without touching the heap.
    open: Vec<LogSource>,
    /// Opt-in flag. Default OFF; only a UI / debugging caller sets it.
    recording: bool,
    /// Rows refused after [`MAX_ROWS`].
    dropped: u32,
}

/// Two states that differ only in their event log are the SAME state.
///
/// [`crate::TurnEvents`] derives `PartialEq`, and the log is a display
/// side-channel: letting it into equality would make a recording state
/// compare unequal to an identical non-recording one and quietly change
/// dedup / transposition behaviour in search. Hand-writing it here (rather
/// than hand-writing all of `TurnEvents`) keeps the derive — and its
/// compile-time coverage of every real field — intact.
impl PartialEq for EventLog {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for EventLog {}

impl EventLog {
    /// Is recording on?
    #[must_use]
    #[inline]
    pub const fn recording(&self) -> bool {
        self.recording
    }

    /// Turn recording on/off. Turning it OFF keeps the rows already recorded;
    /// [`Self::clear`] drops them.
    pub fn set_recording(&mut self, on: bool) {
        self.recording = on;
    }

    /// The recorded rows, oldest first.
    #[must_use]
    pub fn rows(&self) -> &[EventLogRow] {
        &self.rows
    }

    /// Rows refused because the log hit [`MAX_ROWS`]. Non-zero means
    /// [`Self::rows`] is a PREFIX of what happened, not the whole fight.
    #[must_use]
    pub const fn dropped(&self) -> u32 {
        self.dropped
    }

    /// Field-wise equality: the rows, the drop count, the open-source stack
    /// and the recording flag.
    ///
    /// `==` is always `true` (see the `PartialEq` impl), so a check that must
    /// see a log write uses this beside the state comparison: the debug
    /// shadows of a skipped dispatch, which replay it on a clone and require
    /// the clone to equal the state.
    #[must_use]
    pub fn same_contents(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.dropped == other.dropped
            && self.open == other.open
            && self.recording == other.recording
    }

    /// Drop every row and reset the open-source stack, keeping the flag.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.open.clear();
        self.dropped = 0;
    }

    /// Silence and empty this log for a search rollout.
    ///
    /// Called on the MCTS root clone. Without it a search cloned from a
    /// recording live state would record every rollout event into a log
    /// nobody reads — pure allocation on the hottest path in the engine.
    pub fn disarm_for_search(&mut self) {
        self.recording = false;
        self.clear();
    }

    /// Append one row, maintaining the open-source stack from the entry
    /// itself. No-op when recording is off.
    pub fn push(
        &mut self,
        round: u16,
        side: Side,
        entry: EventLogEntry,
        player: Option<CharacterId>,
    ) {
        if !self.recording {
            return;
        }
        if self.rows.len() >= MAX_ROWS {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        // A monster move is caused by nothing that precedes it, so it must not
        // inherit the PREVIOUS move's frame: clear before stamping. (A card
        // play is different — an auto-played card really is caused by the
        // parent that played it, so it keeps the open frame.)
        if matches!(entry, EventLogEntry::MonsterMove { .. }) {
            self.open.clear();
        }
        let row = EventLogRow {
            round,
            side,
            entry,
            source: self.open.last().copied(),
        };
        // The echo mirrors C#'s `[HISTTRACE]` stream so the two diff
        // line-for-line; a row C# never prints must not appear in it.
        if echo_enabled() && !matches!(entry, EventLogEntry::PowerFlashed { .. }) {
            eprintln!(
                "[COMBATLOG] {}: {}",
                entry.kind(),
                row.human_readable(player)
            );
        }
        self.rows.push(row);
        self.track_source(&entry);
    }

    /// Update the open-source stack for a row that was just recorded.
    ///
    /// A card play opens a frame that its finish closes; a monster move
    /// replaces the whole stack (a move is the outermost thing on the enemy
    /// side, and clearing here means a missed close on the player side cannot
    /// mis-attribute the enemy turn). A `CardPlayFinished` with no matching
    /// open frame pops nothing rather than underflowing.
    fn track_source(&mut self, entry: &EventLogEntry) {
        match entry {
            EventLogEntry::CardPlayStarted { card, upgraded, .. } => {
                self.open.push(LogSource::Card {
                    card: *card,
                    upgraded: *upgraded,
                });
            }
            EventLogEntry::CardPlayFinished { card, .. } => {
                // Pop the innermost frame for THIS card. A mismatch means a
                // play resolved without its finish row (an early return on a
                // path that has no funnel); dropping the innermost card frame
                // anyway keeps the stack from growing without bound.
                let pos = self.open.iter().rposition(|s| match s {
                    LogSource::Card { card: c, .. } => c == card,
                    LogSource::Move { .. } => false,
                });
                match pos {
                    Some(i) => {
                        self.open.truncate(i);
                    }
                    None => {
                        if matches!(self.open.last(), Some(LogSource::Card { .. })) {
                            self.open.pop();
                        }
                    }
                }
            }
            EventLogEntry::MonsterMove { monster, move_id } => {
                self.open.clear();
                if let LogCreature::Monster { id, .. } = monster {
                    self.open.push(LogSource::Move {
                        monster: *id,
                        move_id,
                    });
                }
            }
            _ => {}
        }
    }
}

/// Is `STS2_COMBAT_LOG` set? When it is, `combat_loop::start_combat` arms
/// every combat it starts, so the switch works on any binary in the workspace
/// (CLI, arena, parity harness) with no code change — and the rows echo to
/// stderr as they are recorded.
#[must_use]
pub fn forced_by_env() -> bool {
    echo_enabled()
}

/// Is the stderr echo armed? Read once — the env var is process-scoped and
/// re-reading it per event would put a `getenv` on the damage chokepoint.
fn echo_enabled() -> bool {
    use std::sync::OnceLock;
    static ECHO: OnceLock<bool> = OnceLock::new();
    *ECHO.get_or_init(|| std::env::var(ECHO_ENV).is_ok_and(|v| v != "0"))
}

/// Is `state`'s log recording? Call sites guard on this BEFORE building a row,
/// so a disabled log costs one `bool` read and no creature lookups.
#[must_use]
#[inline]
pub fn armed(state: &crate::CombatState) -> bool {
    state.combat_log.event_log.recording()
}

/// Record `entry` against `state`'s current round/side. No-op when the log is
/// not recording.
pub fn record(state: &mut crate::CombatState, entry: EventLogEntry) {
    let (round, side, character) = (state.round, state.side, state.player.character);
    state
        .combat_log
        .event_log
        .push(round, side, entry, character);
}

/// Record that `power`, owned by the creature at `uid`, just triggered.
///
/// The ONE call shape every flash site uses. It is a statement, not an
/// expression: it returns nothing, reads no RNG, and touches nothing but the
/// log, so inserting it into live power logic cannot change what the engine
/// does. With the log disarmed it costs one `bool` read.
pub fn flash_power(
    state: &mut crate::CombatState,
    uid: crate::CreatureUid,
    power: crate::PowerKind,
) {
    if !armed(state) {
        return;
    }
    let creature = LogCreature::of_uid(state, uid);
    record(state, EventLogEntry::PowerFlashed { creature, power });
}

/// [`flash_power`] for a power the PLAYER owns, who has no creature uid.
pub fn flash_player_power(state: &mut crate::CombatState, power: crate::PowerKind) {
    if !armed(state) {
        return;
    }
    let creature = LogCreature::Player(state.player.character);
    record(state, EventLogEntry::PowerFlashed { creature, power });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log() -> EventLog {
        let mut l = EventLog::default();
        l.set_recording(true);
        l
    }

    fn push(l: &mut EventLog, entry: EventLogEntry) {
        l.push(1, Side::Player, entry, Some(CharacterId::Ironclad));
    }

    fn strike() -> EventLogEntry {
        EventLogEntry::CardPlayStarted {
            card: CardId::StrikeIronclad,
            upgraded: false,
            energy_spent: 1,
            auto_played: false,
            target: Some(0),
        }
    }

    fn hit_on(uid: u32) -> EventLogEntry {
        EventLogEntry::DamageReceived {
            dealer: Some(LogCreature::Player(Some(CharacterId::Ironclad))),
            receiver: LogCreature::Monster {
                uid,
                id: MonsterId("Nibbit"),
            },
            unblocked: 6,
            blocked: 0,
            hp_before: 20,
            hp_after: 14,
        }
    }

    #[test]
    fn off_by_default_and_records_nothing() {
        let mut l = EventLog::default();
        assert!(!l.recording());
        push(&mut l, strike());
        assert!(l.rows().is_empty(), "a disarmed log must record nothing");
    }

    #[test]
    fn damage_attributes_to_the_open_card_play() {
        let mut l = log();
        push(&mut l, strike());
        push(&mut l, hit_on(4));
        let src = l.rows()[1]
            .source
            .expect("damage inside a play has a source");
        assert_eq!(
            src,
            LogSource::Card {
                card: CardId::StrikeIronclad,
                upgraded: false
            }
        );
    }

    #[test]
    fn finish_closes_the_source_so_later_rows_are_unattributed() {
        let mut l = log();
        push(&mut l, strike());
        push(
            &mut l,
            EventLogEntry::CardPlayFinished {
                card: CardId::StrikeIronclad,
                upgraded: false,
            },
        );
        push(&mut l, hit_on(4));
        assert_eq!(
            l.rows()[2].source,
            None,
            "a hit after the play finished must not borrow the finished card"
        );
    }

    #[test]
    fn auto_plays_nest_innermost_first() {
        let mut l = log();
        push(
            &mut l,
            EventLogEntry::CardPlayStarted {
                card: CardId::Havoc,
                upgraded: false,
                energy_spent: 1,
                auto_played: false,
                target: None,
            },
        );
        push(
            &mut l,
            EventLogEntry::CardPlayStarted {
                card: CardId::Anger,
                upgraded: false,
                energy_spent: 0,
                auto_played: true,
                target: None,
            },
        );
        push(&mut l, hit_on(4));
        assert_eq!(
            l.rows()[2].source,
            Some(LogSource::Card {
                card: CardId::Anger,
                upgraded: false
            }),
            "the auto-played child, not its Havoc parent, dealt the hit"
        );
        push(
            &mut l,
            EventLogEntry::CardPlayFinished {
                card: CardId::Anger,
                upgraded: false,
            },
        );
        push(&mut l, hit_on(4));
        assert_eq!(
            l.rows()[4].source,
            Some(LogSource::Card {
                card: CardId::Havoc,
                upgraded: false
            }),
            "closing the child re-exposes the parent play"
        );
    }

    #[test]
    fn a_move_row_does_not_inherit_the_previous_move() {
        // The row that OPENS a frame must not carry the frame it replaced —
        // a display consumer reading `source` off a move row would otherwise
        // attribute "Sticky Shot" to the "Butt" that came before it.
        let mut l = log();
        push(
            &mut l,
            EventLogEntry::MonsterMove {
                monster: LogCreature::Monster {
                    uid: 4,
                    id: MonsterId("Nibbit"),
                },
                move_id: "BUTT",
            },
        );
        push(
            &mut l,
            EventLogEntry::MonsterMove {
                monster: LogCreature::Monster {
                    uid: 5,
                    id: MonsterId("TwigSlimeM"),
                },
                move_id: "STICKY_SHOT",
            },
        );
        assert_eq!(l.rows()[1].source, None);
    }

    #[test]
    fn a_monster_move_replaces_the_stack() {
        let mut l = log();
        push(&mut l, strike());
        push(
            &mut l,
            EventLogEntry::MonsterMove {
                monster: LogCreature::Monster {
                    uid: 4,
                    id: MonsterId("Nibbit"),
                },
                move_id: "BUTT_MOVE",
            },
        );
        push(&mut l, hit_on(4));
        assert_eq!(
            l.rows()[2].source,
            Some(LogSource::Move {
                monster: MonsterId("Nibbit"),
                move_id: "BUTT_MOVE"
            }),
            "an enemy-turn hit attributes to the move, never to a leaked card"
        );
    }

    #[test]
    fn human_readable_matches_the_csharp_string() {
        let mut l = log();
        push(&mut l, hit_on(4));
        assert_eq!(
            l.rows()[0].human_readable(Some(CharacterId::Ironclad)),
            "Rd 1 (Player turn): IRONCLAD dealt 6 damage to Nibbit."
        );
    }

    #[test]
    fn source_less_damage_reads_took() {
        let mut l = log();
        push(
            &mut l,
            EventLogEntry::DamageReceived {
                dealer: None,
                receiver: LogCreature::Player(Some(CharacterId::Ironclad)),
                unblocked: 3,
                blocked: 0,
                hp_before: 80,
                hp_after: 77,
            },
        );
        assert_eq!(
            l.rows()[0].human_readable(Some(CharacterId::Ironclad)),
            "Rd 1 (Player turn): IRONCLAD took 3 damage."
        );
    }

    #[test]
    fn overflow_is_counted_not_silent() {
        let mut l = log();
        for _ in 0..(MAX_ROWS + 5) {
            push(&mut l, hit_on(4));
        }
        assert_eq!(l.rows().len(), MAX_ROWS);
        assert_eq!(l.dropped(), 5, "refused rows must be counted");
    }

    #[test]
    fn equality_ignores_the_log_contents() {
        let mut a = log();
        push(&mut a, strike());
        let b = EventLog::default();
        assert_eq!(a, b, "the log is a side-channel, never a state difference");
    }

    #[test]
    fn disarm_for_search_silences_and_empties() {
        let mut l = log();
        push(&mut l, strike());
        l.disarm_for_search();
        assert!(!l.recording());
        assert!(l.rows().is_empty());
        push(&mut l, strike());
        assert!(l.rows().is_empty(), "disarmed stays disarmed");
    }
}
