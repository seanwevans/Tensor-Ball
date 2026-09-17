// The app's own numbers, read out of script.js rather than restated here.
//
// Everything this solver computes is a claim about *the app's* physics, so a
// constant copied into this folder and left behind when script.js moved would
// not produce a wrong-looking answer — it would produce a confident, exact,
// precisely wrong one. The same argument the app makes for COURT ("a zone test
// that disagreed with the arc the ball actually flew over would be reporting on
// a court that isn't there") applies with more force to an oracle.
//
// So there are no physics constants in this file, only the places to find them.
// Two mechanisms:
//
//   1. CONFIG, COURT, SHOT_ZONES and shotZone are one contiguous, dependency-
//      free slice of script.js — from the CONFIG literal to the `const UP`
//      line tools/hpsearch/patch.mjs already anchors on — so they are taken
//      whole and evaluated, not transcribed.
//   2. The handful of constants that live inside class bodies (gravity, the
//      timestep, the ball's mass and damping, the rim and backboard geometry)
//      are pulled out by anchored regex. Each one throws by name if its anchor
//      moves, which turns drift into a loud failure instead of a silent
//      disagreement.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
export const DEFAULT_SCRIPT = join(HERE, "..", "..", "script.js");

// The same span patch.mjs treats as "the config region": everything from the
// CONFIG literal up to the first line that touches three.js.
const SLICE_START = "const CONFIG = {";
const SLICE_END = "const UP = new THREE.Vector3(0, 1, 0);";

function slice(src, start, end, label) {
  const a = src.indexOf(start);
  const b = src.indexOf(end, a + 1);
  if (a < 0) throw new Error(`appconfig: ${label} start anchor not found`);
  if (b < 0) throw new Error(`appconfig: ${label} end anchor not found`);
  return src.slice(a, b);
}

function num(src, re, label) {
  const m = src.match(re);
  if (!m) throw new Error(`appconfig: could not read ${label} from script.js`);
  const v = Number(m[1]);
  if (!Number.isFinite(v))
    throw new Error(`appconfig: ${label} is not a number: "${m[1]}"`);
  return v;
}

// "reward.rim=3" -> { reward: { rim: 3 } }. Same spelling as tools/hpsearch, so
// a config a search trial ran under can be handed to the solver verbatim.
export function applySet(overlay, assignment) {
  const eq = assignment.indexOf("=");
  if (eq < 0) throw new Error(`--set expects key=value, got "${assignment}"`);
  const path = assignment.slice(0, eq).split(".");
  const raw = assignment.slice(eq + 1);
  const value =
    raw === "true" ? true : raw === "false" ? false : Number(raw);
  if (typeof value === "number" && Number.isNaN(value))
    throw new Error(`--set value for ${path.join(".")} is not a number: "${raw}"`);
  let node = overlay;
  for (const key of path.slice(0, -1)) node = node[key] ??= {};
  node[path.at(-1)] = value;
  return overlay;
}

function merge(dst, src) {
  for (const k of Object.keys(src)) {
    const v = src[k];
    if (v && typeof v === "object" && !Array.isArray(v) && dst[k] && typeof dst[k] === "object")
      merge(dst[k], v);
    else dst[k] = v;
  }
  return dst;
}

export async function loadApp({ scriptPath = DEFAULT_SCRIPT, overlay = {} } = {}) {
  const src = await readFile(scriptPath, "utf8");

  // 1. The config region, evaluated as written.
  const region = slice(src, SLICE_START, SLICE_END, "config region");
  for (const bad of ["THREE", "CANNON"])
    if (new RegExp(`\\b${bad}\\b`).test(region))
      throw new Error(
        `appconfig: the config region now references ${bad}; it can no longer be evaluated standalone`
      );
  const { CONFIG, COURT, SHOT_ZONES, shotZone } = new Function(
    `${region}\nreturn { CONFIG, COURT, SHOT_ZONES, shotZone };`
  )();
  merge(CONFIG, overlay);

  // 2. The constants that live inside classes.
  const ball = slice(src, "class Ball {", "  _resetState()", "Ball class");
  const hoop = slice(src, "  _buildHoop(isLeft) {", "  _boardMarkings(", "_buildHoop");
  const ring = slice(src, "const RIM_SEGMENTS = ", "this.physics.add(rimBody);", "rim body");
  const sensor = slice(src, "const sensorBody = new CANNON.Body({", "if (!isLeft) this.scoringSensor", "scoring sensor");
  const step = slice(src, "class PhysicsWorld {", "  onBeginContact(", "PhysicsWorld");
  const update = slice(src, "  update() {", "  _reward(b)", "TrainingArena.update");
  const spawn = slice(src, "  spawn(maxRadius = ", "  syncMesh()", "Ball.spawn");

  const PHYS = {
    // PhysicsWorld: the world's gravity and the fixed step it takes.
    gravity: num(step, /this\.world\.gravity\.set\(0,\s*(-?[\d.]+),\s*0\)/, "gravity"),
    dt: 1 / num(step, /this\.world\.step\(1 \/ ([\d.]+)\)/, "timestep"),
    // Ball: the body the shot is taken with. linearDamping is 0 and the drag
    // in CONFIG.air stands in for it; angularDamping is what bleeds the spin
    // off over a flight, and it is the only reason a reverse pass has to do
    // anything to the angular velocity at all.
    mass: num(ball, /mass:\s*([\d.]+),/, "ball mass"),
    linearDamping: num(ball, /linearDamping:\s*([\d.]+),/, "linearDamping"),
    angularDamping: num(ball, /angularDamping:\s*([\d.]+)\s*\n/, "angularDamping"),
    // Only reached once a shot touches something, which no solution here does —
    // carried so that cannoncheck.mjs's world is the app's world even when it
    // is asked to re-fly a shot that rattles.
    friction: num(step, /friction:\s*([\d.]+),/, "contact friction"),
    restitution: num(step, /restitution:\s*([\d.]+)/, "contact restitution"),
    solverIterations: num(step, /solver\.iterations = (\d+)/, "solver iterations")
  };

  const sign = 1; // The right-hand hoop, which is the one CONFIG.rim points at.
  const baseX = sign * num(hoop, /const baseX = sign \* ([\d.]+);/, "baseX");
  const boardX = baseX - sign * num(hoop, /const boardX = baseX - sign \* ([\d.]+);/, "board setback");
  const rimX = boardX - sign * num(hoop, /const rimX = boardX - sign \* ([\d.]+);/, "rim setback");
  const boardWidth = num(hoop, /const BOARD_WIDTH = ([\d.]+);/, "BOARD_WIDTH");
  const boardHeight = num(hoop, /const BOARD_HEIGHT = ([\d.]+);/, "BOARD_HEIGHT");
  const boardBottom = num(hoop, /const BOARD_BOTTOM = ([\d.]+);/, "BOARD_BOTTOM");
  const boardHalfDepth = num(hoop, /const boardHalfDepth = ([\d.]+);/, "boardHalfDepth");
  const poleHalf = num(hoop, /new CANNON\.Box\(new CANNON\.Vec3\(([\d.]+), [\d.]+, [\d.]+\)\)/, "pole half-extent");
  const poleHeight = num(hoop, /new CANNON\.Box\(new CANNON\.Vec3\([\d.]+, ([\d.]+), [\d.]+\)\)/, "pole height");
  const poleOffset = num(hoop, /pole\.position\.set\(baseX \+ sign \* ([\d.]+),/, "pole offset");

  if (Math.abs(rimX - CONFIG.rim.x) > 1e-9)
    throw new Error(
      `appconfig: the hoop is built at x=${rimX} but CONFIG.rim.x is ${CONFIG.rim.x}`
    );

  const GEOM = {
    // The ring the ball has to fall through, as cannon sees it: a circle of
    // `segments` spheres, not a torus. The mesh is a torus of the same radii,
    // but it is the bodies that deflect a shot, so it is the bodies that decide
    // whether a trajectory is clean.
    ring: {
      x: rimX,
      y: CONFIG.rim.y,
      z: CONFIG.rim.z,
      radius: num(hoop, /const RIM_RING_RADIUS = ([\d.]+);/, "rim ring radius"),
      tube: num(hoop, /const RIM_BAR_RADIUS = ([\d.]+);/, "rim bar radius"),
      segments: num(ring, /const RIM_SEGMENTS = (\d+);/, "rim segments")
    },
    // The backboard slab. The face is the regulation plane; the depth is spent
    // backwards, away from the court.
    board: {
      faceX: boardX,
      centerX: boardX + sign * boardHalfDepth,
      centerY: boardBottom + boardHeight / 2,
      centerZ: 0,
      halfDepth: boardHalfDepth,
      halfHeight: boardHeight / 2,
      halfWidth: boardWidth / 2
    },
    pole: {
      centerX: baseX + sign * poleOffset,
      centerY: poleHeight,
      centerZ: 0,
      halfDepth: poleHalf,
      halfHeight: poleHeight,
      halfWidth: poleHalf
    },
    // Court._buildHoop's scoring trigger. A cylinder in cannon, which it
    // approximates with an eight-sided prism; nothing here deflects off it.
    sensor: {
      x: rimX,
      y: num(sensor, /sensorBody\.position\.set\(rimX, ([\d.]+), 0\)/, "sensor height"),
      z: 0,
      radius: num(sensor, /new CANNON\.Cylinder\(([\d.]+),/, "sensor radius"),
      halfHeight:
        num(sensor, /new CANNON\.Cylinder\([\d.]+, [\d.]+, ([\d.]+),/, "sensor height extent") / 2
    },
    floorY: 0,
    // Ball.spawn's own acceptance test for a launch spot. A reverse pass that
    // lands the shooter outside these is a shot the app would never have taken.
    spawnBounds: {
      maxX: num(spawn, /x >= 0 && x <= ([\d.]+)/, "spawn max x"),
      maxAbsZ: num(spawn, /Math\.abs\(z\) <= ([\d.]+)/, "spawn max |z|")
    }
  };

  // TrainingArena.update's own stopping rules, so a trajectory this solver
  // calls a make is one the app would also have kept alive long enough to
  // credit.
  const STOP = {
    floorY: CONFIG.ballRadius + num(update, /position\.y < CONFIG\.ballRadius \+ ([\d.]+)/, "floor cutoff"),
    oobX: num(update, /Math\.abs\(b\.body\.position\.x\) > ([\d.]+)/, "OOB x"),
    oobZ: num(update, /Math\.abs\(b\.body\.position\.z\) > ([\d.]+)/, "OOB z"),
    stoppedSpeed: num(update, /b\.body\.velocity\.length\(\) < ([\d.]+)/, "stopped speed"),
    stoppedAfter: num(update, /isStopped && b\.path\.length > (\d+)/, "stopped grace frames"),
    timeout: num(update, /b\.path\.length > (\d+)/, "flight timeout")
  };

  return { CONFIG, COURT, SHOT_ZONES, shotZone, PHYS, GEOM, STOP, scriptPath };
}
