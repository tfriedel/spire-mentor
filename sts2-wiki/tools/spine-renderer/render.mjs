/**
 * Headless Spine skeleton renderer.
 * Reads <rig>/<rig>.skel + .atlas (+ every texture page the atlas names) for
 * each rig folder and renders the idle pose to a square PNG.
 *
 * Usage: node render.mjs [--rigs DIR] [--out DIR] [--size N] [--force] [rig ...]
 *   --rigs   folder of rig folders (default ../../site/public/images/monster_rigs,
 *            the layout `scripts/extract_rig.py --all` writes)
 *   --out    where <rig>.png goes (default ../../site/public/images/monsters)
 *   --size   output edge in px (default 512; the character sprites use 640)
 *   --force  re-render rigs whose PNG already exists (default: skip them)
 *   rig ...  only these rig folders (default: all of them)
 */
import { createCanvas, loadImage, Image } from "canvas";
import {
  TextureAtlas,
  AtlasAttachmentLoader,
  SkeletonBinary,
  Skeleton,
  AnimationState,
  AnimationStateData,
  SkeletonRenderer,
  Texture,
  TextureFilter,
  TextureWrap,
  Physics,
  RegionAttachment,
  Skin,
} from "@esotericsoftware/spine-canvas";
import fs from "node:fs";
import path from "node:path";
import { renderSkeleton, imageDataToPng } from "./render_utils.mjs";

function parseArgs(argv) {
  const opts = {
    rigs: path.resolve(import.meta.dirname, "../../site/public/images/monster_rigs"),
    out: path.resolve(import.meta.dirname, "../../site/public/images/monsters"),
    size: 512,
    force: false,
    names: [],
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--rigs") opts.rigs = path.resolve(argv[++i]);
    else if (a === "--out") opts.out = path.resolve(argv[++i]);
    else if (a === "--size") opts.size = Number(argv[++i]);
    else if (a === "--force") opts.force = true;
    else if (a.startsWith("--")) throw new Error(`unknown option ${a}`);
    else opts.names.push(a);
  }
  if (!Number.isInteger(opts.size) || opts.size <= 0) throw new Error("--size wants a positive integer");
  return opts;
}

const OPTS = parseArgs(process.argv.slice(2));
const MONSTERS_DIR = OPTS.rigs;
const OUTPUT_DIR = OPTS.out;

const OUTPUT_SIZE = OPTS.size; // final output image size
const SUPERSAMPLE = 2;  // render at Nx and downscale to hide triangle seams
const RENDER_SIZE = OUTPUT_SIZE * SUPERSAMPLE;
const PADDING = Math.round(OUTPUT_SIZE / 512 * 20) * SUPERSAMPLE;

/** Minimal Texture wrapper for node-canvas Image */
class NodeTexture extends Texture {
  constructor(image) {
    super(image);
  }
  setFilters(_min, _mag) {}
  setWraps(_u, _v) {}
  dispose() {}
}

// Rigs whose look is a NAMED skin the game sets in code. Mirrors
// web/src/monsterSkins.ts (keyed there by monster slug, here by rig folder) and
// MonsterModel's phobia toggle, whose off state is "normal".
const RIG_SKINS = {
  bowlbug_egg: "cocoon",
  bowlbug_nectar: "goop",
  bowlbug_rock: "rock",
  bowlbug_silk: "web",
  battle_friend_v1: "v1",
  battle_friend_v2: "v2",
  battle_friend_v3: "v3",
  calcified_cultist: "coral",
  damp_cultist: "slug",
  byrdpip: "version1",
  paels_legion: "eyes",
  phantasmal_gardener: "tall",
};

/**
 * Wear the rig's skin. A mapped variant skin wins (merged over the default
 * one); otherwise "normal" when the rig has the phobia pair; otherwise, like
 * SpineSprite's composeSetupSkin, the default skin plus whichever named skins
 * supply the attachments the setup pose names but the default lacks (the
 * random cosmetic skins: one deterministic pick, not an empty slot).
 */
function applySkin(skeleton, data, rig) {
  const def = data.defaultSkin;
  const mapped = RIG_SKINS[rig];
  const variant = (mapped && data.findSkin(mapped)) || data.findSkin("normal");
  const composed = new Skin("__composed");
  if (def) composed.addSkin(def);
  if (variant) {
    composed.addSkin(variant);
  } else {
    const named = data.skins.filter(s => s !== def);
    for (let i = 0; i < data.slots.length; i++) {
      const att = data.slots[i].attachmentName;
      if (!att || composed.getAttachment(i, att)) continue;
      const src = named.find(s => s.getAttachment(i, att));
      if (src) composed.addSkin(src);
    }
  }
  skeleton.setSkin(composed);
  skeleton.setSlotsToSetupPose();
}

function findFile(dir, name, ext) {
  const exact = path.join(dir, `${name}${ext}`);
  if (fs.existsSync(exact)) return exact;
  // Fallback: find any file with matching extension (handles filename mismatches/typos)
  const files = fs.readdirSync(dir).filter(f => f.endsWith(ext) && !f.endsWith(".import"));
  if (files.length > 0) {
    console.log(`  Using fallback ${ext}: ${files[0]}`);
    return path.join(dir, files[0]);
  }
  return null;
}

async function renderMonster(monsterDir, monsterName) {
  const skelPath = findFile(monsterDir, monsterName, ".skel");
  const atlasPath = findFile(monsterDir, monsterName, ".atlas");

  if (!skelPath || !atlasPath) {
    console.warn(`  Skipping ${monsterName}: missing files`);
    return false;
  }

  // Load atlas
  const atlasText = fs.readFileSync(atlasPath, "utf-8");
  const atlas = new TextureAtlas(atlasText);

  // Each atlas page names its own texture (ironclad.png, ironclad_2.png, ...);
  // giving every page the first texture scrambles any multi-page rig.
  for (const page of atlas.pages) {
    const pagePath = path.join(path.dirname(atlasPath), page.name);
    if (!fs.existsSync(pagePath)) {
      console.warn(`  Skipping ${monsterName}: atlas page ${page.name} is missing`);
      return false;
    }
    page.setTexture(new NodeTexture(await loadImage(pagePath)));
  }

  // Load skeleton binary
  const attachmentLoader = new AtlasAttachmentLoader(atlas);
  const skelBinary = new SkeletonBinary(attachmentLoader);
  const skelBytes = fs.readFileSync(skelPath);
  const skelData = skelBinary.readSkeletonData(new Uint8Array(skelBytes));

  // Create skeleton, pick the skin the game would wear, and set to setup pose
  const skeleton = new Skeleton(skelData);
  applySkin(skeleton, skelData, monsterName);
  skeleton.setToSetupPose();

  // Try to apply idle animation: exact names first, then a prefix match (some
  // rigs only have per-phase idles, "idle_loop1/2/3" on the Test Subject).
  const stateData = new AnimationStateData(skelData);
  const state = new AnimationState(stateData);
  const names = skelData.animations.map(a => a.name);
  const idleNames = ["idle_loop", "idle", "Idle_loop", "Idle"];
  const idle = idleNames.find(n => names.includes(n))
    ?? idleNames.map(c => names.find(n => n.startsWith(c))).find(Boolean)
    ?? names[0];
  if (idle) {
    state.setAnimation(0, idle, false);
    state.apply(skeleton);
  }

  skeleton.updateWorldTransform(Physics.reset);

  // Compute bounds (exclude shadow/ground/VFX slots for tighter framing)
  const EXCLUDE_EXACT = new Set(["shadow", "shadow2", "ground", "ground_shadow", "main shadow", "sword shadow"]);
  const EXCLUDE_PATTERNS = [/path/i, /whoosh/i, /windpath/i, /vulnerable/i, /projectile/i, /megablade/i, /megatail/i, /whip_path/i, /smoke_?tex/i, /smoke mesh/i];
  function shouldExcludeSlot(slotName, attName) {
    if (EXCLUDE_EXACT.has(slotName) || EXCLUDE_EXACT.has(attName)) return true;
    for (const pat of EXCLUDE_PATTERNS) {
      if (pat.test(slotName) || pat.test(attName)) return true;
    }
    return false;
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const slots = skeleton.slots;
  for (const slot of slots) {
    const attachment = slot.getAttachment();
    if (!attachment) continue;
    const slotName = slot.data.name.toLowerCase();
    const attName = (attachment.name || "").toLowerCase();
    if (shouldExcludeSlot(slotName, attName)) continue;
    if (attachment.computeWorldVertices) {
      const verts = new Float32Array(1000);
      try {
        let numFloats;
        if (attachment instanceof RegionAttachment) {
          // RegionAttachment: computeWorldVertices(slot, worldVertices, offset, stride)
          numFloats = 8;
          attachment.computeWorldVertices(slot, verts, 0, 2);
        } else {
          // MeshAttachment: computeWorldVertices(slot, start, count, worldVertices, offset, stride)
          numFloats = attachment.worldVerticesLength || 8;
          attachment.computeWorldVertices(slot, 0, numFloats, verts, 0, 2);
        }
        for (let i = 0; i < numFloats; i += 2) {
          const x = verts[i], y = verts[i + 1];
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      } catch {
        // skip problematic attachments
      }
    }
  }

  if (!isFinite(minX)) {
    console.warn(`  Skipping ${monsterName}: no renderable attachments`);
    return false;
  }

  // Hide VFX slots so they aren't rendered (but keep shadows — they only affect bounds, not visuals)
  // `smokeTex/smoke_tex` (Gas Bomb, Living Smog/Fog) and `smoke1/smoke mesh` (The
  // Forgotten, The Lost) are labelled "smoke placeholder"
  // textures the game only ever draws through a smoke shader; raw, they are
  // blobs with text on them.
  const HIDE_PATTERNS = [/path/i, /whoosh/i, /windpath/i, /vulnerable/i, /projectile/i, /megablade/i, /megatail/i, /smoke_?tex/i, /smoke mesh/i];
  for (const slot of slots) {
    const attachment = slot.getAttachment();
    if (!attachment) continue;
    const slotName = slot.data.name.toLowerCase();
    const attName = (attachment.name || "").toLowerCase();
    for (const pat of HIDE_PATTERNS) {
      if (pat.test(slotName) || pat.test(attName)) {
        slot.setAttachment(null);
        break;
      }
    }
  }

  const skelWidth = maxX - minX;
  const skelHeight = maxY - minY;

  // Calculate canvas size to fit with padding, maintaining aspect ratio
  const availableSize = RENDER_SIZE - PADDING * 2;
  const scale = Math.min(availableSize / skelWidth, availableSize / skelHeight);

  // Render skeleton (with automatic slot-by-slot fallback for complex meshes)
  const imgData = renderSkeleton(skeleton, RENDER_SIZE, scale, minX, minY, maxX, maxY);
  const buffer = imageDataToPng(imgData, RENDER_SIZE, OUTPUT_SIZE);

  // Save to PNG
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const outPath = path.join(OUTPUT_DIR, `${monsterName}.png`);
  fs.writeFileSync(outPath, buffer);
  console.log(`  Rendered ${monsterName} (${skelWidth.toFixed(0)}x${skelHeight.toFixed(0)}) -> ${outPath}`);
  return true;
}

async function main() {
  if (!fs.existsSync(MONSTERS_DIR)) {
    console.error(`Rig directory not found: ${MONSTERS_DIR} (run scripts/extract_rig.py --all first)`);
    process.exit(1);
  }
  const all = fs.readdirSync(MONSTERS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name)
    .sort();
  for (const name of OPTS.names) {
    if (!all.includes(name)) {
      console.error(`Rig directory not found: ${path.join(MONSTERS_DIR, name)}`);
      process.exit(1);
    }
  }
  const dirs = OPTS.names.length ? OPTS.names : all;
  console.log(`Rendering idle poses: ${MONSTERS_DIR} -> ${OUTPUT_DIR} (${OUTPUT_SIZE}px)\n`);

  let rendered = 0, present = 0, skipped = 0;
  for (const name of dirs) {
    if (!OPTS.force && fs.existsSync(path.join(OUTPUT_DIR, `${name}.png`))) {
      present++;
      continue;
    }
    const ok = await renderMonster(path.join(MONSTERS_DIR, name), name);
    if (ok) rendered++;
    else skipped++;
  }
  console.log(`\nDone! Rendered: ${rendered}, Already present: ${present}, Skipped: ${skipped}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
