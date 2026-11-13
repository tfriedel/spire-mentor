/**
 * Dev-only visual gallery: renders every decision screen with a representative
 * fixture so the full set can be eyeballed (and screenshot) at once. Served by
 * the Vite dev server at /gallery.html. Not part of the shipped app.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./theme.css";
// The scene primitives (.gs-scene and its backdrop/vignette pseudo-elements)
// live here, and every non-combat screen below is built on them. Without this
// import .gs-scene has no rules at all — not even `position: relative` — so
// every absolutely-positioned scenery layer escapes its screen and anchors to
// the document instead. The gallery rendered that way for its whole life and
// it read as the SCREENS being broken.
import "./game-skin.css";
// Last, as in main.tsx: the phone floors win by source order.
import "./phone.css";
import type { Step } from "./api";
import Board from "./components/Board";
import Shop from "./components/Shop";
import EventScreen from "./components/EventScreen";
import TreasureReveal from "./components/TreasureReveal";
import RestSite from "./components/RestSite";
import CardSelect from "./components/CardSelect";
import BundleSelect from "./components/BundleSelect";
import CardReward from "./components/CardReward";
import MapChoices from "./components/MapChoices";
import MapView from "./components/MapView";
import type { FullMapNode } from "./api";
import GameOver from "./components/GameOver";
import combatFixture from "./components/__tests__/fixture-step.json";

function mkStep(decision: string, state: Record<string, unknown>): Step {
  return {
    idx: 0, decision, state: { decision, ...state } as Step["state"],
    scores: [], pv: null, chosen_action: null, source: "live",
    meta: { turn: null, step_no: 0, floor: 42, act: 3, act_name: "Act 3" },
  };
}

const card = (index: number, name: string, type: string, cost = 1) =>
  ({ index, id: `CARD.${name.toUpperCase()}`, name, type, cost, image: null, description: `${name} does a thing.` });

/** The combat fixture with a different cast on the enemy side. The Board reads
 *  the whole step, so a variant screen is a deep-ish clone with `enemies`
 *  swapped — which is the only way to eyeball the states that need a live run
 *  to reach (a revived Test Subject, a Doomed creature, a rig group). */
function combatWith(enemies: unknown[]): Step {
  const base = combatFixture as unknown as Step;
  return { ...base, state: { ...base.state, enemies } as Step["state"] };
}

const RIG = (slug: string) => `/assets/monster_rigs/${slug}/${slug}`;

/** The combat fixture recast as a Defect with a channelled orb queue — the one
 *  board state that needs a character the fixture isn't, and the only way to
 *  eyeball the orb belt's arc without playing a Defect run to a Zap. */
function defectCombat(orbs: unknown[], slots: number, rig = true): Step {
  const base = combatFixture as unknown as Step;
  return {
    ...base,
    state: {
      ...base.state,
      player: {
        ...base.state.player, name: "Defect", hp: 65, max_hp: 75,
        image: "/assets/character_sprites/defect.png",
        // `rig: false` renders the flat character sprite instead of the Spine
        // rig — the belt is scaled to whatever the character is DRAWN at, so
        // both paths are worth a look (and the flat one is the one that shows
        // up in a headless screenshot, where WebGL has no textures).
        ...(rig
          ? { rig: "/assets/character_rigs/defect/defect", rig_scale: 0.125, rig_bounds_h: 260 }
          : { rig: null, rig_scale: null, rig_bounds_h: null }),
      },
      orbs, orb_slots: slots,
    } as Step["state"],
  };
}

const ORB = (index: number, type: string, passive: number, evoke: number) =>
  ({ index, type, name: type, passive, evoke, image: `/assets/orb_atlas/${type.toLowerCase()}.png` });

/** A full 7-column act map with the first row selectable: the live-run map
 *  screen (App.tsx renders MapView as the picker). Four neighbouring first
 *  steps is the tightest the game offers, so it is the case where tap targets
 *  could collide. */
function actMap(): { rows: FullMapNode[][] } {
  const types = ["Monster", "Event", "Shop", "Rest", "Elite", "Treasure", "Monster"];
  const rows: FullMapNode[][] = [];
  for (let r = 0; r < 7; r++) {
    const row: FullMapNode[] = [];
    for (let c = 0; c < 7; c++) {
      if (r === 0 && (c < 2 || c > 5)) continue;
      row.push({
        col: c, row: r, type: types[(c + r * 3) % types.length],
        children: r < 6 ? [{ col: c, row: r + 1 }, ...(c < 6 ? [{ col: c + 1, row: r + 1 }] : [])] : [],
      });
    }
    rows.push(row);
  }
  return { rows };
}

const screens: { label: string; node: React.ReactNode; height?: number; wide?: boolean }[] = [
  // A Board is a full-height scene (parallax stack, arena, hand strip); without
  // an explicit height it collapses into the section box's 120px minimum and
  // renders as a sliver — the same trap the .gs-scene screens fell into.
  { label: "Combat (dark theme)", height: 820, node: <Board step={combatFixture as unknown as Step} /> },
  {
    // The Defect's orb belt: a part-filled 3-slot queue beside a full 5-slot
    // one. Both halves matter — the arc has to hang off the rig's CROWN (it
    // used to hang off the sprite box's top edge, which on a short character
    // parked the orbs in the empty sky) and the empty slots have to draw the
    // game's dashed ring rather than a broken <img>'s alt text.
    label: "Defect orb belt (3 slots, 2 channelled)",
    height: 820,
    node: <Board step={defectCombat([ORB(0, "Lightning", 3, 8), ORB(1, "Frost", 2, 5)], 3)} />,
  },
  {
    // All five orb types, on the widest belt the queue can reach (10 slots —
    // the capacity the game's radius lerp tops out at). The ring grows with the
    // queue, so this is the case that would run off the stage or stack the
    // slots on top of each other if the arc were wrong.
    label: "Defect orb belt (10 slots, all five orb types channelled)",
    height: 820,
    node: <Board step={defectCombat([
      ORB(0, "Lightning", 3, 8), ORB(1, "Frost", 2, 5), ORB(2, "Dark", 6, 6),
      ORB(3, "Plasma", 1, 2), ORB(4, "Glass", 4, 9),
    ], 10, false)} />,
  },
  {
    // Three things the plain fixture cannot show, in one frame: the power row
    // under an enemy (which used to be cropped by the foreground occluder), the
    // Doom band on a health bar, and the Test Subject's revive forms — whose
    // idle clip is chosen by `rig_form`, and which rendered one-headed for its
    // whole fight while that was derived from a display name.
    label: "Combat (powers, Doom, Test Subject form 3)",
    height: 820,
    node: <Board step={combatWith([
      {
        index: 0, name: "Mawler", hp: 44, max_hp: 60, block: 0,
        image: null, rig: RIG("mawler"), rig_scale: 0.5, rig_bounds_h: 327,
        intents: [{ type: "Attack", damage: 10, hits: 3 }],
        powers: [
          { name: "Strength", id: "STRENGTH", amount: 3, image: "/assets/power_atlas/strength_power.png" },
          { name: "Enrage", id: "ENRAGE", amount: 2, image: "/assets/power_atlas/enrage_power.png" },
          { name: "Vulnerable", id: "VULNERABLE", amount: 2, image: "/assets/power_atlas/vulnerable_power.png" },
          { name: "Thorns", id: "THORNS", amount: 4, image: "/assets/power_atlas/thorns_power.png" },
          { name: "Plating", id: "PLATING", amount: 6, image: "/assets/power_atlas/plating_power.png" },
          { name: "Weak", id: "WEAK", amount: 1, image: "/assets/power_atlas/weak_power.png" },
        ],
      },
      {
        index: 1, name: "Scroll of Biting", hp: 9, max_hp: 40, block: 6,
        image: "/assets/monsters/scroll_of_biting.png",
        intents: [{ type: "Debuff" }],
        powers: [
          { name: "Doom", id: "DOOM", amount: 12, image: "/assets/power_atlas/doom_power.png" },
          { name: "Weak", id: "WEAK", amount: 2, image: "/assets/power_atlas/weak_power.png" },
        ],
      },
    ])} />,
  },
  {
    // The Test Subject's three revive forms, side by side — one head, two, then
    // three. They are the same skeleton: the form lives entirely in which idle
    // clip plays (idle_loop1/2/3), chosen from `rig_form`, which the backend
    // derives from the revive-only powers (TestSubject.cs RespawnMove). Deriving
    // it from a display NAME silently pinned every Test Subject to form 1, and a
    // one-headed rig is a perfectly healthy-looking render — nothing but a
    // side-by-side catches that, hence this screen.
    //
    // Drawn at a fraction of the authored 0.3 scale: at full size one of these
    // fills the whole stage, so three of them could not be compared at all.
    label: "Test Subject — the three revive forms (fixture-scaled)",
    height: 820,
    // Three enemy columns do not fit the gallery's 1100px reading width: the
    // arena's side insets plus three HP bars overrun it and the last form falls
    // outside the frame. This screen alone opts into the app's real width.
    wide: true,
    node: <Board step={combatWith([1, 2, 3].map((form) => ({
      index: form - 1,
      name: `Test Subject (form ${form})`,
      hp: [100, 200, 300][form - 1], max_hp: [100, 200, 300][form - 1], block: 0,
      rig_form: form,
      image: null, rig: RIG("test_subject"), rig_scale: 0.075, rig_bounds_h: 292,
      intents: [{ type: "Attack", damage: 20, hits: 1 }],
      powers: form === 3
        ? [{ name: "Nemesis", id: "NEMESIS", amount: 1, image: "/assets/power_atlas/nemesis_power.png" }]
        : form === 2
          ? [{ name: "Painful Stabs", id: "PAINFUL_STABS", amount: 1, image: "/assets/power_atlas/painful_stabs_power.png" }]
          : [{ name: "Adaptable", id: "ADAPTABLE", amount: 1, image: "/assets/power_atlas/adaptable_power.png" }],
    })))} />,
  },
  {
    // The TALL rigs, which is the case the arena's framing has to survive and
    // the one nothing else here covers: every other combat fixture is at or
    // under the Ironclad's 470px box, so "the enemy's head is cut off above the
    // stage" could not show up in the gallery at all. These three are the
    // biggest boxes in creature_scales.json (Architect 1057px, Bygone Effigy
    // 1008px, Ceremonial Beast 941px, against an arena that offers ~509px), so
    // if the fit ever regresses it is visible here first.
    label: "Combat — the tallest rigs (Architect / Bygone Effigy / Ceremonial Beast)",
    height: 820,
    wide: true,
    node: <Board step={combatWith([
      { index: 0, name: "Architect", hp: 300, max_hp: 300, block: 0,
        image: null, rig: RIG("architect"), rig_scale: 0.35, rig_bounds_h: 629,
        intents: [{ type: "Attack", damage: 18, hits: 1 }] },
      { index: 1, name: "Bygone Effigy", hp: 127, max_hp: 160, block: 0,
        image: null, rig: RIG("bygone_effigy"), rig_scale: 0.32, rig_bounds_h: 600,
        intents: [{ type: "Attack", damage: 13, hits: 1 }] },
      { index: 2, name: "Ceremonial Beast", hp: 210, max_hp: 210, block: 0,
        image: null, rig: RIG("ceremonial_beast"), rig_scale: 0.434394, rig_bounds_h: 560,
        intents: [{ type: "Debuff" }] },
    ])} />,
  },
  {
    // The Kaiser Crab boss: two targetable claws (Crusher = left arm, Rocket =
    // right) on ONE shared skeleton. In the game this rig is not a combatant
    // standing on the enemy side at all — it is the ROOM
    // (`NKaiserCrabBossBackground`, mounted at %KaiserCrab on the combat
    // background), with a claw reaching in from each side of the arena and the
    // player standing between them.
    label: "Kaiser Crab boss (Crusher + Rocket, shared rig)",
    height: 820,
    wide: true,
    node: <Board step={combatWith([
      {
        index: 0, name: "Crusher", hp: 180, max_hp: 209, block: 0,
        image: null, rig: RIG("kaiser_crab"), rig_scale: 0.5,
        rig_group: "kaiser_crab", rig_part: "left",
        intents: [{ type: "Attack", damage: 12, hits: 1 }],
        powers: [{ name: "Crab Rage", id: "CRAB_RAGE", amount: 1, image: null }],
      },
      {
        index: 1, name: "Rocket", hp: 209, max_hp: 209, block: 0,
        image: null, rig: RIG("kaiser_crab"), rig_scale: 0.5,
        rig_group: "kaiser_crab", rig_part: "right",
        intents: [{ type: "Buff" }],
      },
    ])} />,
  },
  {
    label: "Shop",
    height: 700,
    node: <Shop onChoose={() => {}} step={mkStep("shop", {
      player: { name: "Ironclad", gold: 142 },
      cards: [
        // The merchant's ONE discounted card: "%" tag over the top-right
        // corner, chartreuse price. Exactly one slot per shop can carry it.
        { ...card(0, "Anger", "Attack"), cost: 142 - 100, card_cost: 0, is_stocked: true, on_sale: true },
        { ...card(1, "Inflame", "Power", 1), cost: 110, card_cost: 1, is_stocked: true },
      ],
      relics: [
        { index: 0, name: "Akabeko", cost: 75, is_stocked: true, image: "/assets/relic_atlas/akabeko.png" },
        { index: 1, name: "Bag of Marbles", cost: 130, is_stocked: true, image: null },
      ],
      potions: [{ index: 0, name: "Fire Potion", cost: 48, is_stocked: true, image: "/assets/potion_atlas/attack_potion.png" }],
      card_removal_cost: 75,
    })} />,
  },
  {
    // Disabled wares: too expensive (gold 10) and sold out. A disabled price
    // tag still has to hold its size so the shelf does not shift.
    label: "Shop (unaffordable and sold out)",
    height: 700,
    node: <Shop onChoose={() => {}} step={mkStep("shop", {
      player: { name: "Ironclad", gold: 10 },
      cards: [
        { ...card(0, "Anger", "Attack"), cost: 42, card_cost: 0, is_stocked: true },
        { ...card(1, "Inflame", "Power", 1), cost: 110, card_cost: 1, is_stocked: false },
      ],
      relics: [{ index: 0, name: "Akabeko", cost: 75, is_stocked: true, image: "/assets/relic_atlas/akabeko.png" }],
      potions: [{ index: 0, name: "Fire Potion", cost: 48, is_stocked: false, image: "/assets/potion_atlas/attack_potion.png" }],
      card_removal_cost: 75,
    })} />,
  },
  {
    label: "Event (real illustration)",
    node: <EventScreen onChoose={() => {}} step={mkStep("event_choice", {
      event_name: "Abyssal Baths",
      description: "You find a steaming pool. Strange bubbles rise from the depths.",
      art: { illustration: "/assets/events/abyssal_baths.png" },
      event_options: [
        { index: 0, title: "Bathe", description: "Heal [green]10[/green] HP. Add a [red]Curse[/red].", is_locked: false },
        { index: 1, title: "Drink", description: "Gain [gold]50[/gold] Gold.", is_locked: false },
        { index: 2, title: "Leave", description: "Nothing happens.", is_locked: false },
      ],
    })} />,
  },
  {
    // Neow's blessing is the tallest event the game serves: four relic options,
    // each with a description line, and in live play a win chip under every one.
    // Framed at a real viewport height because that is the only way the
    // overflow shows up — see `.gs-event-col`'s scroll note in event.css.
    label: "Neow (four relic options, viewport-height frame)",
    height: 640,
    node: <EventScreen onChoose={() => {}} step={mkStep("event_choice", {
      event_name: "Neow",
      description: "",
      event_options: [
        { index: 0, title: "Small Capsule", description: "Upon pickup, obtain a random Relic.", is_locked: false },
        { index: 1, title: "Kaleidoscope", description: "At the start of each combat, Channel 1 random Orb.", is_locked: false },
        { index: 2, title: "Precarious Shears", description: "At the start of each combat, gain 3 Block. Whenever you lose HP, lose 1 Block.", is_locked: false },
        { index: 3, title: "Venerable Tea Set", description: "At the start of your first turn each combat, gain 2 Energy.", is_locked: false },
      ],
    })} />,
  },
  {
    // A treasure room: the game's own chest rig, and the chest is the button.
    label: "Treasure (chest room)",
    height: 520,
    node: <EventScreen onChoose={() => {}} step={mkStep("event_choice", {
      event_name: "Treasure",
      event_options: [{ index: 0, title: "Open", is_locked: false }],
    })} />,
  },
  {
    // What a chest actually looks like in a run: the engine opens it on entry
    // (no decision), so this panel is the only thing that makes the room
    // visible. The chest-room entry above is the legacy decision surface.
    label: "Treasure (what's inside)",
    height: 640,
    node: (
      <div style={{ position: "relative", height: 640 }}>
        <TreasureReveal
          onDismiss={() => {}}
          reveal={{
            kind: "treasure", act: 1, gold: 43,
            relic: { name: "Venerable Tea Set", description: "At the start of your first turn each combat, gain 2 Energy." },
          }}
        />
      </div>
    ),
  },
  {
    // Framed at a real viewport height, and with a PLAYER: the campfire scene
    // is `position: absolute` scenery sized off the measured scene height, so
    // in an auto-height section the hero comes out too small to review — and
    // without `player` it is not drawn at all (the hero/log overlap, the one
    // thing this screen has to get right, was invisible here for its whole
    // life). Regent is the demanding case: its rig carries the throne, so it
    // is the widest, tallest silhouette any character brings to the log.
    label: "Rest site",
    height: 820,
    node: <RestSite onChoose={() => {}} step={mkStep("rest_site", {
      player: { name: "Regent", hp: 52, max_hp: 70 },
      rest_options: [
        { index: 0, option_id: "Rest", name: "Rest (heal 30%)", is_enabled: true },
        { index: 1, option_id: "Smith", name: "Smith (upgrade a card)", is_enabled: true },
        { index: 2, option_id: "Dig", name: "Dig", is_enabled: false },
      ],
    })} />,
  },
  {
    label: "Card reward",
    // Framed at a real viewport height on purpose. The reward scene is
    // `overflow: hidden` and its column is pinned to the box it is given (see
    // .gs-cardreward-frame), so in an auto-height section it would size itself
    // to the 400px scene minimum and the trio would come down the card-size
    // ladder — a harness artefact, not what the app does.
    height: 820,
    node: <CardReward onChoose={() => {}} step={mkStep("card_reward", {
      can_skip: true, gold_earned: 22,
      cards: [card(0, "Cleave", "Attack", 1), card(1, "Shrug It Off", "Skill", 1), card(2, "Demon Form", "Power", 3)],
    })} />,
  },
  {
    label: "Card select",
    node: <CardSelect onChoose={() => {}} step={mkStep("card_select", {
      min_select: 1, max_select: 2,
      cards: [card(0, "Strike", "Attack"), card(1, "Defend", "Skill"), card(2, "Bash", "Attack", 2)],
    })} />,
  },
  {
    // Upgrade picks (rest-site Smith, Armaments) carry each option's upgraded
    // twin, shown as the game's before → after comparison.
    label: "Card select (upgrade)",
    node: <CardSelect onChoose={() => {}} step={mkStep("card_select", {
      context: "upgrade", min_select: 1, max_select: 1,
      cards: [
        {
          ...card(0, "Bash", "Attack", 2), description: "Deal 8 damage.\nApply 2 Vulnerable.",
          upgrade_preview: {
            ...card(0, "Bash", "Attack", 2), upgraded: true,
            description: "Deal 10 damage.\nApply 3 Vulnerable.",
          },
        },
        {
          ...card(1, "Afterimage", "Power", 1),
          description: "Whenever you play a card, gain 1 Block.",
          upgrade_preview: {
            ...card(1, "Afterimage", "Power", 1), upgraded: true,
            description: "Whenever you play a card, gain 1 Block.", keyword_delta: "+Innate",
          },
        },
      ],
    })} />,
  },
  {
    label: "Bundle select",
    node: <BundleSelect onChoose={() => {}} step={mkStep("bundle_select", {
      bundles: [
        { index: 0, name: "Attacks", cards: [card(0, "Anger", "Attack"), card(1, "Cleave", "Attack")] },
        { index: 1, name: "Skills", cards: [card(0, "Flex", "Skill"), card(1, "Warcry", "Skill")] },
      ],
    })} />,
  },
  {
    label: "Map choices (play)",
    node: <MapChoices onChoose={() => {}} step={mkStep("map_select", {
      choices: [
        { col: 2, row: 5, type: "Monster" }, { col: 3, row: 5, type: "Shop" },
        { col: 4, row: 5, type: "Event" }, { col: 5, row: 5, type: "Rest" },
      ],
    })} />,
  },
  {
    label: "Map (live run, selectable nodes)",
    height: 820,
    node: <MapView rows={[]} currentIdx={0} live
      graphs={[{ idx: 0, act: 1, act_name: "Act 1", floor: 1, full_map: {
        ...actMap(), boss: { col: 3, row: 7, type: "Boss", name: "The Guardian" }, current_coord: null,
      } }]}
      selectable={new Set(["2,0", "3,0", "4,0", "5,0"])} onSelect={() => {}} />,
  },
  { label: "Game over (victory)", node: <GameOver step={mkStep("game_over", { victory: true, ascension: 15, player: { name: "Ironclad", hp: 34, max_hp: 80 } })} /> },
];

function Gallery() {
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
      <h1 style={{ fontWeight: 700 }}>Workbench Screen Gallery</h1>
      {screens.map((s) => (
        <section key={s.label} style={{
          margin: "32px 0",
          // `wide` breaks a screen out of the 1100px reading column to the
          // viewport width the app actually runs at, for the few screens whose
          // defect only appears when there is room for the full cast.
          ...(s.wide ? { width: "min(1700px, calc(100vw - 48px))", marginLeft: "calc(50% - min(850px, 50vw - 24px))" } : null),
        }}>
          <h2 style={{ fontSize: 14, color: "var(--muted)", borderBottom: "1px solid var(--border)", paddingBottom: 6 }}>{s.label}</h2>
          {/* `height` frames a screen at a real viewport height. Screens whose
              layout only misbehaves when the frame is SHORTER than their
              content (the event column) cannot show that in an auto-height
              box, which is what let Neow's options run off the bottom
              unnoticed. */}
          <div className="surface"
            style={{ marginTop: 12, overflow: "hidden", minHeight: 120, height: s.height }}>
            {s.node}
          </div>
        </section>
      ))}
    </div>
  );
}

createRoot(document.getElementById("gallery")!).render(<StrictMode><Gallery /></StrictMode>);
