// Lazy chunk: the 3D Mataf twin (react-three-fiber). One top view of the Mataf floor: every camera's latest density
// map projected onto the floor at its real position and merged (backend/floormap.py, /api/mataf/floor.png), on the
// twin's people-per-m² scale: warm where it is crowded, faded as a camera's reading ages, grey where no camera saw the
// floor in the last 10 minutes. Around it a low-detail model of the real edge (./matafArch.ts: the golden Ottoman
// arcade ring with small domes, the grey Mataf expansion stepping up behind it), the Kaaba, Hijr Ismail and Maqam
// Ibrahim (./kaaba.ts). Camera markers stand where the calibrated cameras are (config/twin.json).
// GPU budget (the crowd collector shares this GPU): no shadow map, pixel ratio <= 1.5, 24 fps (20 on phones) while it
// sways, nothing off screen or in a hidden tab, few draw calls. A lost WebGL context is restored and redrawn.
// ?view=kaaba frames the Kaaba close up (for review and screenshots); ?perf counts frames (window.__twinFrames).
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type { FloorMeta, TwinCam } from "../api";
import { useOnScreen } from "../hooks";
import { buildMatafArch, FLOOR_R, RING_MID, S } from "./matafArch";
import { buildKaaba, KAABA_ROT } from "./kaaba";

/** Camera state in the twin: live / stale reading, no reading yet (idle), no reference frame (waiting). */
export interface TwinState extends TwinCam {
  id: string;
  quality: string | null;
  state: "live" | "stale" | "idle" | "waiting";
  k: number; // heat intensity 0..1 (level rank once calibrated)
  selected: boolean;
}

interface Props {
  cams: TwinState[];
  /** the merged floor map (/api/mataf/floor), null before the first one */
  floor: FloorMeta | null;
  reduce: boolean;
  mobile: boolean;
  avoid?: string; // selector of an overlay (the crowd number) that camera labels must not cover
  /** main dashboard since 2026-11-01: the floor in flat level colours, camera labels and the "estimated" tag off unless
   * asked, and where on the canvas the model is framed (px from each edge) */
  floorStyle?: "flat";
  labels?: boolean;
  pad?: { t: number; r: number; b: number; l: number };
}

const QUERY = new URLSearchParams(location.search);
const CLOSE = QUERY.get("view") === "kaaba";
const PERF = QUERY.has("perf");
// Frame-rate cap: the motion is a slow sway, so 24 fps (20 on phones) looks the same as 60 and leaves the GPU to the
// live video and the collector's crowd model on the same machine.
const FPS = { desktop: 24, mobile: 20 };
const SWAY = 0.14; // rad, the slow turn of the model

declare global {
  interface Window { __twinFrames?: number; __twinLost?: number }
}

export default function Twin({ cams, floor, reduce, mobile, avoid, floorStyle, labels: showLabels = true, pad }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const onScreen = useOnScreen(box);
  const labels = useRef<(HTMLElement | null)[]>([]);
  const est = useRef<HTMLSpanElement>(null);
  const [lost, setLost] = useState(false);
  return (
    <div className="twin-canvas" ref={box}>
      {/* off screen or hidden tab: no rendering at all; on screen: frames only when the capped clock or new data asks */}
      <Canvas dpr={[1, 1.5]} frameloop={onScreen ? "demand" : "never"} camera={{ fov: 26, near: 0.5, far: 400 }}
        gl={{ antialias: true, alpha: true, toneMappingExposure: 1.05, powerPreference: "default" }} aria-hidden="true"
        onCreated={({ gl, invalidate }) => {
          // a lost context (the GPU short of memory, a driver reset) is restored by the browser when we ask; redraw then
          const c = gl.domElement;
          c.addEventListener("webglcontextlost", (e) => {
            e.preventDefault();
            window.__twinLost = (window.__twinLost ?? 0) + 1;
            setLost(true);
          });
          c.addEventListener("webglcontextrestored", () => {
            setLost(false);
            invalidate();
          });
        }}>
        <Clock fps={mobile ? FPS.mobile : FPS.desktop} active={onScreen && !lost} animate={!reduce} />
        <Environment />
        <hemisphereLight color="#dfe9e8" groundColor="#3b342a" intensity={0.55} />
        <Scene cams={cams} floor={floor} reduce={reduce} mobile={mobile} labels={labels} est={est} avoid={avoid} box={box}
          floorStyle={floorStyle} pad={pad} />
      </Canvas>
      {/* the floor behind the Kaaba: estimated from the floor beside it at the same distance (backend/floormap.py) */}
      <span className="est3d" ref={est} style={floor?.hidden_at && !floorStyle ? undefined : { display: "none" }}>estimated</span>
      {lost && <div className="twin-paused">3D view paused: the graphics card is short of memory. It resumes by itself.</div>}
      {cams.map((c, i) => (
        <button key={c.id} type="button" className={`label3d${c.state === "live" ? " live" : ""}`} ref={(el) => void (labels.current[i] = el)}
          style={{ ...((c.state !== "live" && c.state !== "stale") || !showLabels ? { display: "none" } : {}), ["--c" as string]: c.color }}
          aria-label={`${c.label}${c.state === "stale" ? ", earlier reading" : c.quality === "low" ? ", lower quality" : ""}`}>
          <i /><b>{c.label.replace(/ camera\b/i, "")}</b>
          <span className="more">
            {c.label}{c.state === "stale" ? " · earlier reading" : c.state === "live" ? " · reading now" : ""}
            {c.quality === "low" ? " · lower quality" : ""}
          </span>
        </button>
      ))}
    </div>
  );
}

/**
 * The capped render clock: while the twin is on screen and moving, ask for a frame at most `fps` times a second
 * (requestAnimationFrame also stops in hidden tabs). One frame whenever it comes back on screen; none off screen.
 */
function Clock({ fps, active, animate }: { fps: number; active: boolean; animate: boolean }) {
  const { invalidate } = useThree();
  useEffect(() => {
    if (!active) return;
    invalidate();
    if (!animate) return;
    const step = 1000 / fps;
    let id = 0, last = -Infinity;
    const tick = (t: number) => {
      id = requestAnimationFrame(tick);
      if (t - last < step - 1.5) return; // 1.5 ms slack for the vsync jitter
      last = t;
      invalidate();
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [fps, active, animate, invalidate]);
  return null;
}

/** Soft image-based light from three's procedural RoomEnvironment (no download, small); dim, it is night in Makkah. */
function Environment() {
  const { gl, scene, invalidate } = useThree();
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    const env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = env;
    scene.environmentIntensity = 0.3;
    invalidate();
    return () => {
      scene.environment = null;
      env.dispose();
      pmrem.dispose();
    };
  }, [gl, scene, invalidate]);
  return null;
}

/** Kaaba frame (where the architecture and the floor map are drawn) <-> scene coordinates of the camera config. */
const COS = Math.cos(KAABA_ROT), SIN = Math.sin(KAABA_ROT);
const toLocal = (x: number, z: number) => new THREE.Vector2(x * COS - z * SIN, x * SIN + z * COS);

interface Placed {
  cam: TwinState;
  B: THREE.Vector2; // camera foot on the floor (scene x, z)
  aim: THREE.Vector2; // a floor point it looks at (scene x, z)
}

/** Cameras the config does not place yet get the free angle farthest from every other camera (no pile-ups). */
function spread(cams: TwinState[]): TwinState[] {
  const taken = cams.filter((c) => c.placed !== false).map((c) => c.ang);
  const gap = (a: number) => Math.min(...taken.map((b) => Math.abs(((a - b + 3 * Math.PI) % (2 * Math.PI)) - Math.PI)), Math.PI);
  return cams.map((c) => {
    if (c.placed !== false) return c;
    let best = c.ang;
    for (let i = 0; i < 24; i++) if (gap((i / 24) * 2 * Math.PI) > gap(best)) best = (i / 24) * 2 * Math.PI;
    taken.push(best);
    return { ...c, ang: best };
  });
}

/** Where a camera stands (config/twin.json; cameras not placed yet on the first-floor gallery) and where it looks. */
function place(c: TwinState): Placed {
  // drawn on the outer ring at the camera's bearing (its real distance and height are in config/twin.json): the marker
  // and its label stay outside the floor, the floor keeps the space
  const r = RING_MID * S, h = 12 * S;
  const B = new THREE.Vector2(Math.cos(c.ang) * r, -Math.sin(c.ang) * r);
  const D = B.clone().multiplyScalar(-1).normalize().rotateAround(new THREE.Vector2(), c.aim);
  return { cam: { ...c, r, h }, B, aim: B.clone().addScaledVector(D, Math.max(1, (c.near + c.far) / 2 - (c.r - r))) };
}

interface SceneProps {
  cams: TwinState[];
  floor: FloorMeta | null;
  reduce: boolean;
  mobile: boolean;
  labels: RefObject<(HTMLElement | null)[]>;
  est: RefObject<HTMLSpanElement | null>;
  avoid?: string;
  box: RefObject<HTMLDivElement | null>;
  floorStyle?: "flat";
  pad?: { t: number; r: number; b: number; l: number };
}

const Y_AXIS = new THREE.Vector3(0, 1, 0);
// label anchors tried in order: above the marker, above to the right, above to the left
const ANCHORS = [-0.5, -0.1, -0.9];

function Scene({ cams, floor, reduce, mobile, labels, est, avoid, box, floorStyle, pad }: SceneProps) {
  const twin = useRef<THREE.Group>(null);
  const { camera, size, invalidate } = useThree();
  const arch = useMemo(buildMatafArch, []);
  const kaaba = useMemo(buildKaaba, []);
  const line25 = useMemo(buildLine25, []);
  const geomKey = cams.map((c) => [c.id, c.ang, c.r, c.h, c.near, c.far, c.aim, c.state, c.placed].join()).join("|");
  const placedCams = useMemo(() => spread(cams).map(place), [geomKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const markers = useMemo(() => buildMarkers(placedCams, arch.topAt, arch.floors), [placedCams, arch]);
  const heat = useFloorMap(floor?.png ? (floorStyle ? `${floor.png}&style=${floorStyle}` : floor.png) : null, invalidate);
  useEffect(() => () => dispose(arch.group), [arch]);
  useEffect(() => () => dispose(kaaba), [kaaba]);
  useEffect(() => () => dispose(markers.group), [markers]);

  // frame the whole model (and the camera markers with their labels) inside the canvas, with a margin, at every
  // angle of the sway: right of the number on desktop, the whole strip on phones
  const fitPts = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    const add = (x: number, y: number, z: number) => {
      for (const a of [-SWAY, 0, SWAY]) pts.push(new THREE.Vector3(x, y, z).applyAxisAngle(new THREE.Vector3(0, 1, 0), a));
    };
    arch.extent.forEach((p) => add(p.x, p.y, p.z));
    markers.labelAt.forEach((p) => { add(p.x, p.y, p.z); add(p.x, p.y + 0.7, p.z); });
    return pts;
  }, [arch, markers]);
  useLayoutEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    if (CLOSE) { // review view: the Kaaba from the door side, Maqam Ibrahim in front, Hijr Ismail to the right
      cam.position.set(2.0, 2.6, 6.2);
      cam.lookAt(-0.5, 0.5, 0.4);
    } else {
      const W = size.width, Hh = size.height;
      // desktop: the free area right of the number block and above the camera chips; phones: the whole strip
      // desktop: the canvas covers only the area right of the number block (styles.css); leave the chips below free
      const region = pad ? [pad.l, pad.t, W - pad.r, Hh - pad.b] : mobile ? [6, 8, W - 6, Hh - 8] : [4, 46, W - 14, Hh - 96];
      // fairly top-down, so the pattern on the floor reads; still enough angle to show the building's height
      fit(cam, fitPts, W, Hh, region, Math.PI / 2, ((mobile ? 62 : 58) * Math.PI) / 180);
    }
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
    invalidate();
  }, [camera, size, mobile, fitPts, invalidate, pad?.t, pad?.r, pad?.b, pad?.l]); // eslint-disable-line react-hooks/exhaustive-deps

  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ clock }) => {
    const g = twin.current;
    if (!g) return;
    if (PERF) window.__twinFrames = (window.__twinFrames ?? 0) + 1;
    if (!reduce) g.rotation.y = Math.sin(clock.getElapsedTime() * 0.1) * SWAY; // slow sway
    g.updateMatrixWorld();
    camera.updateMatrixWorld(); // gl.render updates it only after this callback; demand mode has no next frame
    const own = box.current?.getBoundingClientRect();
    const av = avoid ? box.current?.parentElement?.querySelector(avoid)?.getBoundingClientRect() : undefined;
    const block = own && av && !mobile ? { l: av.left - own.left, t: av.top - own.top, r: av.right - own.left, b: av.bottom - own.top } : null;
    const used: { l: number; t: number; r: number; b: number }[] = [];
    v.set(0, 0, 0).applyMatrix4(g.matrixWorld).project(camera);
    const cx = ((v.x + 1) / 2) * size.width, cy = ((1 - v.y) / 2) * size.height; // the floor's centre on screen
    const ha = floor?.hidden_at, el = est.current;
    if (ha && el && el.offsetWidth) { // centred on the hidden floor (twin-local x, z in the Kaaba frame)
      v.set(ha[0] * S, 0.1, ha[1] * S).applyAxisAngle(Y_AXIS, KAABA_ROT).applyMatrix4(g.matrixWorld).project(camera);
      const l = ((v.x + 1) / 2) * size.width - el.offsetWidth / 2, t = ((1 - v.y) / 2) * size.height - el.offsetHeight / 2;
      el.style.transform = `translate(${l.toFixed(1)}px, ${t.toFixed(1)}px)`;
      used.push({ l, t, r: l + el.offsetWidth, b: t + el.offsetHeight });
    }
    markers.labelAt.forEach((p, i) => {
      const el = labels.current?.[i];
      if (!el || !el.offsetWidth) return; // hidden: only cameras with a reading are labelled (the chips list the rest)
      v.copy(p).applyMatrix4(g.matrixWorld).project(camera);
      const mx = ((v.x + 1) / 2) * size.width, my = ((1 - v.y) / 2) * size.height;
      const w = el.offsetWidth, h = el.offsetHeight;
      // pushed outwards from the floor's centre, so the chip sits on the ring, not on the crowd
      const dl = Math.hypot(mx - cx, my - cy) || 1, push = 6 + Math.max(w, h) / 2;
      const x = mx + ((mx - cx) / dl) * push, y = my + ((my - cy) / dl) * push + h / 2;
      const clash = (a: { l: number; t: number; r: number; b: number }) =>
        used.some((u) => a.l < u.r + 4 && a.r > u.l - 4 && a.t < u.b + 3 && a.b > u.t - 3)
        || (block != null && a.l < block.r + 6 && a.r > block.l && a.t < block.b && a.b > block.t);
      // above the marker (centred, right, left), then one row higher, then below it; past the number block
      const xs = [x - w / 2, ...ANCHORS.map((k) => x + k * w), ...(block ? [block.r + 8] : [])];
      let best: { l: number; t: number; r: number; b: number } | null = null;
      for (const top of [y - h, y - 2 * h - 4, y + 4]) {
        for (const l of xs) {
          const a = { l, t: top, r: l + w, b: top + h };
          if (l >= 4 && a.r <= size.width - 4 && top >= 2 && a.b <= size.height - 2 && !clash(a)) {
            best = a;
            break;
          }
        }
        if (best) break;
      }
      best ??= { l: Math.min(Math.max(4, x - w / 2), size.width - w - 4), t: Math.max(2, y - h), r: 0, b: 0 };
      used.push({ ...best, r: best.l + w, b: best.t + h });
      el.style.transform = `translate(${best.l.toFixed(1)}px, ${best.t.toFixed(1)}px)`;
    });
  });

  return (
    <group ref={twin}>
      <directionalLight color="#fffaf3" intensity={1.9} position={[13, 22, 9]} />
      <group rotation-y={KAABA_ROT}>
        <primitive object={arch.group} />
        <primitive object={line25} />
        <mesh rotation-x={-Math.PI / 2} position-y={0.03} renderOrder={1}>
          <circleGeometry args={[FLOOR_R * S, 96]} />
          <meshBasicMaterial map={heat} transparent depthWrite={false} toneMapped={false} visible={heat != null}
            polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
        </mesh>
      </group>
      <primitive object={markers.group} />
      <primitive object={kaaba} />
    </group>
  );
}

/** Thin line on the floor 25 m from the Kaaba's walls (between the outer ring and the far Mataf; twin-local x = -X,
 * z = Y, to scale up to it: backend/floormap.py). */
function buildLine25(): THREE.LineLoop {
  const L = 11.86, W = 10.04, D = 25;
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < 240; i++) {
    const a = (2 * Math.PI * i) / 240;
    let lo = 0, hi = 60; // the distance from the centre where the wall distance reaches D, by bisection
    for (let k = 0; k < 30; k++) {
      const r = (lo + hi) / 2;
      const dx = Math.max(Math.abs(r * Math.cos(a)) - L / 2, 0), dy = Math.max(Math.abs(r * Math.sin(a)) - W / 2, 0);
      if (Math.hypot(dx, dy) < D) lo = r; else hi = r;
    }
    pts.push(new THREE.Vector3(-lo * Math.cos(a) * S, 0.06, lo * Math.sin(a) * S));
  }
  const mat = new THREE.LineBasicMaterial({ color: 0xece7dc, transparent: true, opacity: 0.55, depthWrite: false });
  return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), mat);
}

/**
 * The floor map texture. A new one replaces the old only once it has loaded (no blank frame in between); the PNG's
 * row 0 is z_local = -FLOOR_R and three.js puts it at v = 1, which the floor disc maps to z_local = -FLOOR_R.
 */
function useFloorMap(url: string | null, invalidate: () => void): THREE.Texture | null {
  const [tex, setTex] = useState<THREE.Texture | null>(null);
  useEffect(() => {
    if (!url) return;
    let gone = false;
    new THREE.TextureLoader().load(url, (t) => {
      if (gone) return t.dispose();
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 4;
      t.generateMipmaps = false; // a small texture seen from above: linear filtering is enough, no mip chain
      t.minFilter = THREE.LinearFilter;
      setTex((old) => {
        old?.dispose();
        return t;
      });
      invalidate();
    });
    return () => void (gone = true);
  }, [url, invalidate]);
  useEffect(() => () => tex?.dispose(), []); // eslint-disable-line react-hooks/exhaustive-deps
  return tex;
}

/**
 * Put the camera at elevation `elev` and azimuth `az` (scene xz angle of the direction towards the camera) so that
 * the points' screen box fills `region` [x0, y0, x1, y1] (px) as far as it allows, centred in it.
 */
function fit(cam: THREE.PerspectiveCamera, pts: THREE.Vector3[], W: number, Hh: number, region: number[], az: number, elev: number) {
  const dir = new THREE.Vector3(Math.cos(az) * Math.cos(elev), Math.sin(elev), Math.sin(az) * Math.cos(elev));
  const target = new THREE.Vector3(), v = new THREE.Vector3(), right = new THREE.Vector3(), up = new THREE.Vector3();
  const [rx0, ry0, rx1, ry1] = region;
  let d = 80;
  for (let it = 0; it < 14; it++) {
    cam.position.copy(target).addScaledVector(dir, d);
    cam.lookAt(target);
    cam.updateMatrixWorld();
    cam.updateProjectionMatrix();
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of pts) {
      v.copy(p).project(cam);
      const x = ((v.x + 1) / 2) * W, y = ((1 - v.y) / 2) * Hh;
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
    const k = Math.max((x1 - x0) / (rx1 - rx0), (y1 - y0) / (ry1 - ry0));
    const wpp = (2 * d * Math.tan((cam.fov / 2) * (Math.PI / 180))) / Hh; // world units per px at the target
    right.setFromMatrixColumn(cam.matrixWorld, 0);
    up.setFromMatrixColumn(cam.matrixWorld, 1);
    target.addScaledVector(right, -((rx0 + rx1) / 2 - (x0 + x1) / 2) * wpp).addScaledVector(up, ((ry0 + ry1) / 2 - (y0 + y1) / 2) * wpp);
    d *= k;
  }
  cam.position.copy(target).addScaledVector(dir, d);
  cam.lookAt(target);
}

function dispose(o: THREE.Object3D) {
  o.traverse((x) => {
    const m = x as THREE.Mesh;
    m.geometry?.dispose();
    (Array.isArray(m.material) ? m.material : m.material ? [m.material] : []).forEach((mat) => {
      Object.values(mat).forEach((val) => (val as THREE.Texture)?.isTexture && (val as THREE.Texture).dispose());
      mat.dispose();
    });
  });
}

/** Camera markers: a small cone on a thin mast from the building (or the gallery floor) below it, pointing where it looks. */
function buildMarkers(cams: Placed[], topAt: (x: number, z: number) => number, floors: number[]): { group: THREE.Group; labelAt: THREE.Vector3[] } {
  const group = new THREE.Group();
  const labelAt: THREE.Vector3[] = [];
  const mast = new THREE.LineBasicMaterial({ color: 0xece7dc, transparent: true, opacity: 0.45 });
  for (const f of cams) {
    const c = f.cam, on = c.state === "live" || c.state === "stale";
    const p = new THREE.Vector3(f.B.x, c.h, f.B.y);
    const q = toLocal(f.B.x, f.B.y), top = topAt(q.x, q.y);
    const foot = top < c.h - 0.05 ? top : Math.max(...floors.filter((h) => h < c.h - 0.05)); // in a gallery: its floor
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.5, 4),
      new THREE.MeshStandardMaterial({ color: on ? 0xece7dc : 0x6f7c80, emissive: on ? 0x3a3526 : 0, roughness: 0.4 }));
    cone.position.copy(p);
    cone.lookAt(f.aim.x, 0, f.aim.y);
    cone.rotateX(Math.PI / 2);
    group.add(cone);
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p.x, foot, p.z), p]), mast));
    labelAt.push(p.clone().add(new THREE.Vector3(0, 0.3, 0)));
  }
  return { group, labelAt };
}
