// Procedural textures for the 3D twin (drawn on canvases at load, no image downloads).
// Each "maps" pair is a colour map plus an RM map (green = roughness, blue = metalness, the channels three.js
// reads), so gold thread is metallic and shiny while the black silk stays matte.
import * as THREE from "three";

const GOLD = ["#8f6d2c", "#d9b86c", "#f0d79a", "#b8913f"];
const CLOTH = "#0c0c0e";
const RM_CLOTH = "rgb(0, 225, 0)"; // rough, not metal
const RM_GOLD = "rgb(0, 105, 255)"; // smoother, metal

function canvas(w: number, h: number) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return [c, c.getContext("2d")!] as const;
}

function tex(c: HTMLCanvasElement, colour: boolean, repeat?: [number, number]) {
  const t = new THREE.CanvasTexture(c);
  if (colour) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (repeat) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(...repeat);
  }
  return t;
}

const goldFill = (ctx: CanvasRenderingContext2D, y0: number, y1: number) => {
  const g = ctx.createLinearGradient(0, y0, 0, y1);
  GOLD.forEach((c, i) => g.addColorStop(i / (GOLD.length - 1), c));
  return g;
};

/** Small deterministic random generator (the same ornament on every load). */
const rng = (seed: number) => () => (seed = (seed * 16807) % 2147483647) / 2147483647;

export interface Maps {
  map: THREE.CanvasTexture;
  rm: THREE.CanvasTexture;
}

/** Draws `paint` twice: colour pass (gold gradient) and RM pass (metal mask). `paint` must be deterministic. */
function paired(w: number, h: number, base: string, paint: (ctx: CanvasRenderingContext2D, ink: string | CanvasGradient) => void,
  repeat?: [number, number], keep = 1): Maps {
  const [c1, x1] = canvas(w, h);
  const [c2, x2] = canvas(w, h);
  x1.fillStyle = base;
  x1.fillRect(0, 0, w, h);
  paint(x1, goldFill(x1, 0, h));
  x2.fillStyle = RM_CLOTH;
  x2.fillRect(0, 0, w, h);
  paint(x2, RM_GOLD);
  // drawn at full size, kept smaller (GPU memory: the texture is only a few pixels tall on screen)
  const small = (c: HTMLCanvasElement) => {
    if (keep >= 1) return c;
    const [s, sx] = canvas(Math.round(w * keep), Math.round(h * keep));
    sx.drawImage(c, 0, 0, s.width, s.height);
    return s;
  };
  return { map: tex(small(c1), true, repeat), rm: tex(small(c2), false, repeat) };
}

/**
 * Abstract gold ornament that reads as calligraphy from a distance, with NO letters or words (user decision
 * 2026-10-24: no text on the model): a flowing baseline, leaning upright strokes with hooked tops, bowls
 * dipping below the line, a long sweeping stroke above, and small lozenges.
 */
function pseudoScript(ctx: CanvasRenderingContext2D, x0: number, x1: number, top: number, bottom: number, seed: number) {
  const r = rng(seed);
  const base = bottom - (bottom - top) * 0.3;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.lineWidth = 4;
  ctx.beginPath(); // baseline with a slow wave
  for (let x = x0; x <= x1; x += 6) ctx.lineTo(x, base + Math.sin((x - x0) / 23 + seed) * 2.5);
  ctx.stroke();
  for (let x = x0 + 10 + r() * 10; x < x1 - 10; x += 13 + r() * 22) {
    const tall = r() < 0.58, h = tall ? (bottom - top) * (0.6 + r() * 0.22) : (bottom - top) * (0.16 + r() * 0.12);
    ctx.lineWidth = tall ? 4.6 : 4;
    ctx.beginPath(); // upright stroke, leaning slightly, hooked at the top when tall
    ctx.moveTo(x, base);
    ctx.lineTo(x + h * 0.12, base - h);
    if (tall) ctx.quadraticCurveTo(x + h * 0.12 - 3, base - h - 6, x + h * 0.12 - 9, base - h + 2);
    ctx.stroke();
    if (r() < 0.28 && x - 40 > x0) { // tail dipping under the line and sweeping left, as the pen moves right to left
      const w = 24 + r() * 20;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(x, base);
      ctx.quadraticCurveTo(x - w * 0.35, base + 24 + r() * 6, x - w, base + 6);
      ctx.stroke();
    }
    if (r() < 0.4) { // lozenge above or below
      const ly = r() < 0.6 ? base - (bottom - top) * (0.42 + r() * 0.12) : base + 11;
      ctx.save();
      ctx.translate(x + 9, ly);
      ctx.rotate(Math.PI / 4);
      ctx.fillRect(-2.6, -2.6, 5.2, 5.2);
      ctx.restore();
    }
  }
  const s0 = x0 + (x1 - x0) * (0.1 + r() * 0.3), s1 = s0 + (x1 - x0) * (0.35 + r() * 0.2);
  ctx.lineWidth = 3; // long sweeping stroke over part of the panel
  ctx.beginPath();
  ctx.moveTo(s0, top + 10);
  ctx.quadraticCurveTo((s0 + s1) / 2, top + 2, Math.min(s1, x1 - 6), top + 12);
  ctx.stroke();
}

/**
 * The Hizam (belt two thirds up the Kiswa): gold-embroidered panels on black, framed by gold borders, with
 * roundels between the panels and at the corners; four panels per wall, 16 in all, like the real belt.
 * The panels hold abstract ornament (pseudoScript), never text.
 */
export function hizamMaps(seed: number): Maps {
  const W = 2048, H = 152, n = 4;
  return paired(W, H, CLOTH, (ctx, ink) => {
    ctx.fillStyle = ink;
    ctx.strokeStyle = ink;
    ctx.fillRect(0, 0, W, 7); // outer gold borders
    ctx.fillRect(0, H - 7, W, 7);
    ctx.lineWidth = 2;
    ctx.strokeRect(-2, 13, W + 4, H - 26);
    const gap = 92, pw = (W - gap * (n + 1)) / n;
    const roundel = (cx: number) => {
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, H / 2, 34, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 2;
      for (let k = 0; k < 8; k++) { // eight-petal rosette
        const a = (k / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.ellipse(cx + Math.cos(a) * 14, H / 2 + Math.sin(a) * 14, 12, 5, a, 0, Math.PI * 2);
        ctx.stroke();
      }
    };
    for (let i = 0; i <= n; i++) roundel(gap / 2 + i * (pw + gap));
    for (let i = 0; i < n; i++) {
      const x = gap + i * (pw + gap);
      ctx.lineWidth = 2;
      ctx.strokeRect(x, 22, pw, H - 44);
      ctx.strokeRect(x + 6, 28, pw - 12, H - 56);
      pseudoScript(ctx, x + 16, x + pw - 16, 38, H - 34, seed * 31 + i * 7 + 1);
    }
  }, undefined, 0.5);
}

/** The Kiswa silk: black, with an abstract woven black-on-black pattern (only visible as a change in sheen). */
export function kiswaMaps(): Maps {
  const S = 512;
  return paired(S, S, CLOTH, (ctx, ink) => {
    const colourPass = typeof ink !== "string";
    ctx.fillStyle = colourPass ? "#101012" : "rgb(0, 160, 0)"; // pattern: barely lighter, smoother (shows as sheen)
    ctx.strokeStyle = ctx.fillStyle;
    const row = S / 4;
    for (let r = 0; r < 4; r++) {
      const y = r * row;
      ctx.lineWidth = 3;
      ctx.beginPath(); // zigzag band between rows
      for (let x = 0; x <= S; x += 32) ctx.lineTo(x, y + (x / 32) % 2 * 14);
      ctx.stroke();
      ctx.lineWidth = 2.5;
      for (let x = (r % 2) * 32 + 32; x < S; x += 64) { // chain of lozenges, each with a dot
        const cy = y + row / 2 + 8;
        ctx.beginPath();
        ctx.moveTo(x, cy - 22);
        ctx.lineTo(x + 22, cy);
        ctx.lineTo(x, cy + 22);
        ctx.lineTo(x - 22, cy);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, cy, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [3, 3]);
}

/**
 * The Sitara (door curtain), drawn fully gold: dense gold embroidery read from a distance, with the relief of
 * nested frames, an arch over the door, rosettes and lozenge bands in darker gold (ornament only, no text).
 * The door itself takes the lower middle: 1.9 x 3.1 m of the 3.5 x 5.75 m curtain.
 */
export function doorMaps(): Maps {
  const W = 256, H = 420;
  const [c1, x1] = canvas(W, H);
  const [c2, x2] = canvas(W, H);
  const bright = x1.createLinearGradient(0, 0, W, H); // even, bright gold (the belt's gradient is for a thin band)
  bright.addColorStop(0, "#e6c77f");
  bright.addColorStop(0.5, "#f2dc9f");
  bright.addColorStop(1, "#d4b062");
  x1.fillStyle = bright;
  x1.fillRect(0, 0, W, H);
  x2.fillStyle = RM_GOLD;
  x2.fillRect(0, 0, W, H);
  const dw = W * (1.9 / 3.5), dh = H * (3.1 / 5.75), dx = (W - dw) / 2, dy = H - dh - 6;
  const paint = (ctx: CanvasRenderingContext2D, ink: string) => {
    ctx.strokeStyle = ink;
    ctx.fillStyle = ink;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.lineWidth = 5;
    ctx.strokeRect(5, 5, W - 10, H - 10);
    ctx.lineWidth = 2;
    ctx.strokeRect(14, 14, W - 28, H - 28);
    for (const y of [30, 92]) { // lozenge bands across the top
      for (let x = 26; x < W - 20; x += 22) {
        ctx.beginPath();
        ctx.moveTo(x, y - 8);
        ctx.lineTo(x + 8, y);
        ctx.lineTo(x, y + 8);
        ctx.lineTo(x - 8, y);
        ctx.closePath();
        ctx.stroke();
      }
    }
    for (const cx of [W * 0.3, W * 0.7]) { // rosettes between the bands and the arch
      ctx.beginPath();
      ctx.arc(cx, 140, 16, 0, Math.PI * 2);
      ctx.stroke();
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.ellipse(cx + Math.cos(a) * 8, 140 + Math.sin(a) * 8, 6, 2.5, a, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.lineWidth = 4; // arch over the door
    ctx.beginPath();
    ctx.moveTo(dx - 8, dy + 20);
    ctx.quadraticCurveTo(W / 2, dy - 46, dx + dw + 8, dy + 20);
    ctx.stroke();
    ctx.lineWidth = 4; // the door: two leaves with panels
    ctx.strokeRect(dx, dy, dw, dh);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(W / 2, dy);
    ctx.lineTo(W / 2, dy + dh);
    ctx.stroke();
    for (const lx of [dx + 8, W / 2 + 8]) {
      ctx.strokeRect(lx, dy + 10, dw / 2 - 16, dh * 0.42);
      ctx.strokeRect(lx, dy + 20 + dh * 0.42, dw / 2 - 16, dh * 0.5 - 28);
    }
  };
  paint(x1, "#8a6a2a");
  paint(x2, "rgb(0, 165, 255)");
  return { map: tex(c1, true), rm: tex(c2, false) };
}

/** White marble with soft grey veins (Mataf floor, Hijr wall, Shadharwan). */
export function marbleMap(size = 1024, repeat?: [number, number]): THREE.CanvasTexture {
  const [c, ctx] = canvas(size, size);
  ctx.fillStyle = "#ece9e2";
  ctx.fillRect(0, 0, size, size);
  let seed = 11;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  ctx.filter = "blur(1.2px)";
  for (let i = 0; i < 70; i++) {
    let x = rnd() * size, y = rnd() * size, a = rnd() * Math.PI * 2;
    ctx.strokeStyle = `rgba(150, 146, 138, ${0.05 + rnd() * 0.12})`;
    ctx.lineWidth = 0.6 + rnd() * 1.6;
    ctx.beginPath();
    ctx.moveTo(x, y);
    for (let s = 0; s < 40; s++) { // wandering vein
      a += (rnd() - 0.5) * 0.7;
      x += Math.cos(a) * 9;
      y += Math.sin(a) * 9;
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.filter = "none";
  return tex(c, true, repeat);
}
