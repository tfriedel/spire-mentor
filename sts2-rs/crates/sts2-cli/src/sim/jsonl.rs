//! Per-step JSONL writer for ML training-data export.
//!
//! Each line is one self-contained JSON object — no header, no trailing
//! comma. Schema is documented in `docs/sts2-cli-jsonl-schema.md`.
//!
//! Reward (v1, deterministic from state delta):
//!   * +1 per HP of damage dealt to enemies that step
//!   * -1 per HP of damage taken by the player that step
//!   * +10 on transition into a `WON` step
//!   * -10 on transition into a `LOST` step

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use sts2_core::{Action, CombatState, StepResult};

pub struct JsonlWriter {
    out: BufWriter<File>,
    step: u32,
}

impl JsonlWriter {
    pub fn open(path: &Path) -> io::Result<Self> {
        let f = File::create(path)?;
        Ok(Self {
            out: BufWriter::new(f),
            step: 0,
        })
    }

    pub fn write_step(
        &mut self,
        before: &CombatState,
        action: &Action,
        after: &CombatState,
        result: StepResult,
    ) -> io::Result<()> {
        let dmg_dealt = enemy_hp_total(before).saturating_sub(enemy_hp_total(after));
        let dmg_taken = before
            .player
            .current_hp
            .saturating_sub(after.player.current_hp);
        let outcome_reward = match result {
            StepResult::CombatWon => 10,
            StepResult::CombatLost => -10,
            _ => 0,
        };
        let reward = dmg_dealt - dmg_taken + outcome_reward;

        let line = format!(
            "{{\"step\":{step},\"phase\":\"{phase:?}\",\"round\":{round},\"state_summary\":{state},\"action\":{action_json},\"reward\":{reward},\"result\":\"{result:?}\"}}",
            step = self.step,
            phase = before.phase,
            round = before.round,
            state = serialize_state_summary(before),
            action_json = serialize_action(action),
        );
        writeln!(self.out, "{line}")?;
        self.step += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

fn enemy_hp_total(state: &CombatState) -> i32 {
    state
        .enemy_uids
        .iter()
        .filter_map(|uid| state.creatures.iter().find(|c| c.id == uid.0))
        .map(|c| c.current_hp.max(0))
        .sum()
}

fn serialize_state_summary(state: &CombatState) -> String {
    let monsters: Vec<String> = state
        .enemy_uids
        .iter()
        .filter_map(|uid| state.creatures.iter().find(|c| c.id == uid.0))
        .map(|c| {
            format!(
                "{{\"hp\":{},\"max_hp\":{},\"block\":{},\"intent\":\"{:?}\"}}",
                c.current_hp, c.max_hp, c.block, c.intent
            )
        })
        .collect();
    let hand: Vec<String> = state
        .player
        .hand
        .iter()
        .filter_map(|uid| state.cards.iter().find(|c| c.uid == *uid))
        .map(|card| {
            format!(
                "{{\"card_id\":\"{:?}\",\"cost\":{}}}",
                card.card_id,
                card.local_cost()
            )
        })
        .collect();
    format!(
        "{{\"player_hp\":{hp},\"player_block\":{block},\"player_energy\":{energy},\"draw\":{draw},\"discard\":{discard},\"monsters\":[{m}],\"hand\":[{h}]}}",
        hp = state.player.current_hp,
        block = state.player.block,
        energy = state.player.energy,
        draw = state.player.draw.len(),
        discard = state.player.discard.len(),
        m = monsters.join(","),
        h = hand.join(","),
    )
}

fn serialize_action(a: &Action) -> String {
    match a {
        Action::PlayCard { hand_index, target } => target.as_ref().map_or_else(
            || format!("{{\"type\":\"play_card\",\"hand_index\":{hand_index}}}"),
            |t| format!("{{\"type\":\"play_card\",\"hand_index\":{hand_index},\"target\":{t}}}"),
        ),
        Action::PlayCardAtAlly { hand_index, ally } => {
            format!("{{\"type\":\"play_card\",\"hand_index\":{hand_index},\"ally\":{ally}}}")
        }
        Action::DrinkPotionAtAlly { potion_slot, ally } => {
            format!("{{\"type\":\"drink_potion\",\"slot\":{potion_slot},\"ally\":{ally}}}")
        }
        Action::EndTurn => "{\"type\":\"end_turn\"}".into(),
        Action::UsePotion { slot, target } => target.as_ref().map_or_else(
            || format!("{{\"type\":\"use_potion\",\"slot\":{slot}}}"),
            |t| format!("{{\"type\":\"use_potion\",\"slot\":{slot},\"target\":{t}}}"),
        ),
        Action::DrinkPotion {
            potion_slot,
            target,
        } => target.as_ref().map_or_else(
            || format!("{{\"type\":\"drink_potion\",\"slot\":{potion_slot}}}"),
            |t| format!("{{\"type\":\"drink_potion\",\"slot\":{potion_slot},\"target\":{t}}}"),
        ),
        Action::Channel { orb, target } => target.as_ref().map_or_else(
            || format!("{{\"type\":\"channel\",\"orb\":\"{orb:?}\"}}"),
            |t| format!("{{\"type\":\"channel\",\"orb\":\"{orb:?}\",\"target\":{t}}}"),
        ),
        Action::Evoke { times } => format!("{{\"type\":\"evoke\",\"times\":{times}}}"),
        Action::ResolveChoice { selection } => {
            format!("{{\"type\":\"resolve_choice\",\"selection\":{selection:?}}}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::scenario::{MonsterSpec, Scenario};
    use crate::sim::PlayerClass;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use sts2_core::orbs::Orb;
    use sts2_core::MonsterKind;

    static CTR: AtomicUsize = AtomicUsize::new(0);

    fn tmp_path() -> std::path::PathBuf {
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("sts2jsonl_{}_{n}.jsonl", std::process::id()))
    }

    /// A one-Nibbit combat state with the monster pinned to 50 HP, ready
    /// to clone-and-mutate for before/after step deltas.
    fn base_state() -> CombatState {
        let scenario = Scenario::from_monsters(vec![MonsterSpec {
            slot: None,
            kind: MonsterKind::Nibbit,
            hp_override: Some(50),
            ..MonsterSpec::new(MonsterKind::Nibbit)
        }]);
        crate::sim::scenario::starter::build_state(
            &scenario,
            1,
            PlayerClass::Ironclad,
            Some(40),
            Some(80),
        )
    }

    #[test]
    fn serialize_action_covers_every_variant() {
        assert_eq!(
            serialize_action(&Action::PlayCard {
                hand_index: 2,
                target: None
            }),
            "{\"type\":\"play_card\",\"hand_index\":2}"
        );
        assert_eq!(
            serialize_action(&Action::PlayCard {
                hand_index: 2,
                target: Some(1)
            }),
            "{\"type\":\"play_card\",\"hand_index\":2,\"target\":1}"
        );
        assert_eq!(
            serialize_action(&Action::EndTurn),
            "{\"type\":\"end_turn\"}"
        );
        assert_eq!(
            serialize_action(&Action::DrinkPotion {
                potion_slot: 0,
                target: Some(3)
            }),
            "{\"type\":\"drink_potion\",\"slot\":0,\"target\":3}"
        );
        assert_eq!(
            serialize_action(&Action::Evoke { times: 2 }),
            "{\"type\":\"evoke\",\"times\":2}"
        );
        // Channel + UsePotion + ResolveChoice exercise the remaining arms.
        assert!(serialize_action(&Action::Channel {
            orb: Orb::Lightning,
            target: None
        })
        .contains("\"type\":\"channel\""));
        assert!(serialize_action(&Action::UsePotion {
            slot: 1,
            target: None
        })
        .contains("\"type\":\"use_potion\""));
        assert!(serialize_action(&Action::ResolveChoice {
            selection: smallvec::smallvec![0, 1]
        })
        .contains("resolve_choice"));
    }

    #[test]
    fn write_step_reward_is_damage_delta_plus_win_bonus() {
        let before = base_state();
        // Put a card in hand so the hand-summary branch is exercised.
        let mut before = before;
        if let Some(&uid) = before.player.draw.first() {
            before.player.hand.push(uid);
        }
        let mut after = before.clone();
        // 12 damage dealt to the (single) enemy, 5 taken by the player.
        if let Some(c) = after.creatures.first_mut() {
            c.current_hp -= 12;
        }
        after.player.current_hp -= 5;

        let path = tmp_path();
        {
            let mut w = JsonlWriter::open(&path).expect("open");
            w.write_step(&before, &Action::EndTurn, &after, StepResult::CombatWon)
                .expect("write");
            w.flush().expect("flush");
        }
        let body = std::fs::read_to_string(&path).expect("read");
        let v: serde_json::Value = serde_json::from_str(body.trim()).expect("valid json line");
        // reward = dealt(12) - taken(5) + win(10) = 17.
        assert_eq!(v["reward"], 17);
        assert_eq!(v["result"], "CombatWon");
        assert_eq!(v["step"], 0);
        // The hand summary must reflect the single card we seeded.
        assert_eq!(v["state_summary"]["hand"].as_array().unwrap().len(), 1);
        assert_eq!(v["state_summary"]["monsters"].as_array().unwrap().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_step_increments_step_and_loss_bonus_is_negative() {
        let before = base_state();
        let after = before.clone();
        let path = tmp_path();
        {
            let mut w = JsonlWriter::open(&path).expect("open");
            w.write_step(&before, &Action::EndTurn, &after, StepResult::Advanced)
                .expect("write 0");
            w.write_step(&before, &Action::EndTurn, &after, StepResult::CombatLost)
                .expect("write 1");
            w.flush().expect("flush");
        }
        let body = std::fs::read_to_string(&path).expect("read");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let l0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let l1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(l0["step"], 0);
        assert_eq!(l1["step"], 1);
        // No HP delta, loss transition => reward = -10.
        assert_eq!(l1["reward"], -10);
        assert_eq!(l1["result"], "CombatLost");
        let _ = std::fs::remove_file(&path);
    }
}
