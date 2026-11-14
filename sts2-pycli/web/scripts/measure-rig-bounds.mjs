/**
 * Regenerates `src/rigBounds.ts` — every shipped Spine rig's setup/idle bounding
 * box, in RIG UNITS.
 *
 * Why a generated table rather than a runtime measure: the combat scene lays the
 * creature row out by each creature's footprint (see `src/scene.ts`, a port of
 * `NCombatRoom.PositionEnemies`), and a layout that has to wait for four atlas
 * pages to decode would place every fighter twice — once at a guess, once for
 * real — on every combat. The rigs are content, so their boxes are constants.
 *
 * The measurement mirrors what `SpineSprite` renders: the rig's own idle clip
 * applied to the setup pose, then `Skeleton.getBounds`.
 *
 * Usage (from `web/`, with the extracted art tree present):
 *   node scripts/measure-rig-bounds.mjs [path/to/images]     # default: ../../sts2-wiki/site/public/images
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as core from "@esotericsoftware/spine-core";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const IMAGES = process.argv[2] ?? path.join(HERE, "..", "..", "..", "sts2-wiki", "site", "public", "images");
const OUT = path.join(HERE, "..", "src", "rigBounds.ts");
const DIRS = ["monster_rigs", "character_rigs", "room_rigs"];

// spine-core insists on a texture per atlas page; nothing here rasterizes, so a
// stub with the page's declared size is enough to build the attachments whose
// vertices the bounds come from.
class StubTexture extends core.Texture {
  setFilters() {} setWraps() {} dispose() {}
}

// MUST stay in step with SpineSprite: it hides these slots before measuring, so
// a table built without them would describe a different rig than the one the
// board draws (the Vantom's parked `mega*` overlays alone are 3x its body).
const HIDE = [/slash/i, /zap/i, /whoosh/i, /windpath/i, /path\b/i, /projectile/i, /^shadow/i,
  /chadow/i, /ground/i, /^mega/i, /^straight /i, /^soulface/i, /^light squig/i, /^glowie2/i];

function hideFx(skeleton) {
  for (const slot of skeleton.slots) {
    const a = slot.getAttachment();
    if (!a) continue;
    // Rendered visuals only — nulling a path/clipping guide breaks the
    // constraint that follows it (SpineSprite documents the Vantom tail case).
    if (!(a instanceof core.RegionAttachment || a instanceof core.MeshAttachment)) continue;
    if (HIDE.some((p) => p.test(slot.data.name) || p.test(a.name ?? ""))) slot.setAttachment(null);
  }
}

/** SpineSprite's composeSetupSkin: rigs that keep their body in NAMED skins
 *  (Scroll of Biting, Cubex Construct) measure as an empty shell without it. */
function composeSetupSkin(skeleton, data) {
  const def = data.defaultSkin;
  const missing = [];
  for (let i = 0; i < data.slots.length; i++) {
    const attName = data.slots[i].attachmentName;
    if (attName && !(def && def.getAttachment(i, attName))) missing.push({ index: i, name: attName });
  }
  if (!missing.length) return;
  const named = data.skins.filter((sk) => sk !== def);
  if (!named.length) return;
  const composed = new core.Skin("__composed");
  if (def) composed.addSkin(def);
  for (const { index, name } of missing) {
    if (composed.getAttachment(index, name)) continue;
    const src = named.find((sk) => sk.getAttachment(index, name));
    if (src) composed.addSkin(src);
  }
  skeleton.setSkin(composed);
  skeleton.setSlotsToSetupPose();
}

function measure(base) {
  const atlas = new core.TextureAtlas(fs.readFileSync(base + ".atlas", "utf8"));
  for (const page of atlas.pages) {
    page.setTexture(new StubTexture({ width: page.width || 2048, height: page.height || 2048 }));
  }
  const data = new core.SkeletonBinary(new core.AtlasAttachmentLoader(atlas))
    .readSkeletonData(new Uint8Array(fs.readFileSync(base + ".skel")));
  const skeleton = new core.Skeleton(data);
  composeSetupSkin(skeleton, data);
  const names = data.animations.map((a) => a.name);
  // SpineSprite's idle resolution, exact-then-prefix (a rig with only numbered
  // idles — the Test Subject's idle_loop1/2/3 — must not fall through to
  // animations[0], which on that rig is a knocked-out sprawl).
  const candidates = ["idle_loop", "idle", "Idle_loop", "Idle"];
  const idle = candidates.find((n) => names.includes(n))
    ?? candidates.map((c) => names.find((n) => n.startsWith(c))).find(Boolean)
    // …plus the multi-track bosses, whose body idle is namespaced.
    ?? names.find((n) => n === "body/idle_loop")
    ?? names[0];
  const state = new core.AnimationState(new core.AnimationStateData(data));
  if (idle) state.setAnimation(0, idle, true);
  // The Kaiser Crab's arms idle on their own tracks; the base track alone is a
  // bodiless crab.
  for (const [track, side] of [[1, "left"], [2, "right"]]) {
    if (names.includes(`${side}/idle_loop`)) state.setAnimation(track, `${side}/idle_loop`, true);
  }
  skeleton.setToSetupPose();
  state.apply(skeleton);
  hideFx(skeleton);
  skeleton.updateWorldTransform(core.Physics.update);
  const off = new core.Vector2(), size = new core.Vector2();
  skeleton.getBounds(off, size, []);
  return { x: off.x, y: off.y, w: size.x, h: size.y, idle };
}

const rows = new Map();
// The shared tree, then each later build's art overlay (`v0.113.0/…`, the rigs
// that build added), whose rows follow; a slug the shared tree has keeps its
// row, as the backends' resolvers keep the shared rig.
const overlays = fs.existsSync(IMAGES)
  ? fs.readdirSync(IMAGES).filter((d) => /^v\d/.test(d)).sort()
  : [];
const roots = [...DIRS.map((d) => d), ...overlays.flatMap((o) => DIRS.map((d) => path.join(o, d)))];
for (const dir of roots) {
  const root = path.join(IMAGES, dir);
  if (!fs.existsSync(root)) continue;
  for (const slug of fs.readdirSync(root).sort()) {
    if (rows.has(slug)) continue;
    const sub = path.join(root, slug);
    if (!fs.statSync(sub).isDirectory()) continue;
    const skel = fs.readdirSync(sub).find((f) => f.endsWith(".skel"));
    if (!skel) continue;
    try {
      rows.set(slug, measure(path.join(sub, skel.replace(/\.skel$/, ""))));
    } catch (e) {
      console.error(`skip ${slug}: ${e}`);
    }
  }
}

const r = (v) => Math.round(v);
const body = [...rows.entries()]
  .map(([slug, b]) => `  ${/^[a-z_][a-z0-9_]*$/.test(slug) ? slug : JSON.stringify(slug)}: ` +
    `{ x: ${r(b.x)}, y: ${r(b.y)}, w: ${r(b.w)}, h: ${r(b.h)} },`)
  .join("\n");

fs.writeFileSync(OUT, `// GENERATED by scripts/measure-rig-bounds.mjs — do not edit by hand.
//
// Every shipped Spine rig's idle bounding box, in RIG UNITS, keyed by rig slug
// (the basename of the rig URL the backend sends: \`monster_rigs/<slug>/<slug>\`).
// Multiply by the creature's authored scene scale (\`rig_scale\`) to get its size
// in the game's 1920x1080 scene px — see src/scene.ts.
export interface RigBox { x: number; y: number; w: number; h: number }

export const RIG_BOUNDS: Record<string, RigBox> = {
${body}
};

/** The rig slug for a rig base URL (\`.../monster_rigs/nibbit/nibbit\` -> \`nibbit\`). */
export function rigSlug(rigBase: string | null | undefined): string {
  if (!rigBase) return "";
  const tail = rigBase.split("?")[0].split("/").filter(Boolean).pop() ?? "";
  return tail;
}

/** The rig's idle box in rig units, or null when the rig is unknown here. */
export function rigBox(rigBase: string | null | undefined): RigBox | null {
  return RIG_BOUNDS[rigSlug(rigBase)] ?? null;
}
`);
console.error(`wrote ${OUT} (${rows.size} rigs)`);
